#pragma once

#include <functional>
#include <string_view>
#include <vector>

#include <Eigen/Core>

namespace nav_executor {

// block-scaled L-BFGS。调用方提供完整的变量 block 与物理尺度；算法在无量纲缩放
// 坐标内维护历史，并以自适应 max-step cap 初始化 Armijo 回溯。
class LbfgsMinimizer {
public:
    struct VariableBlock {
        int offset = 0;
        int size = 0;
        double scale = 1.0;
    };

    struct StepControlOptions {
        double initial_step_cap = 0.5;
        double min_step_cap = 1e-10;
        double max_step_cap = 2.0;
        double expansion_min_model_ratio = 0.75;
        double backtrack_factor = 0.25;
        double expansion_factor = 2.0;
        int max_rejections_per_iteration = 20;
        int recovery_after_rejections = 3;
    };

    struct Options {
        int max_iterations = 200;
        int max_function_evaluations = 400;
        int history_size = 8;
        // 一阶最优性阈值：max_block ‖D_b g_b‖₂。D_b 是调用方提供的变量物理
        // 尺度。使用绝对梯度，避免不可控的大 cost 基线放宽驻点判据。
        double gradient_tolerance = 1e-5;
        // accepted incumbent 窗口首尾的 raw 目标相对改善阈值。
        double cost_window_relative_tolerance = 1e-5;
        int cost_window_size = 10;
        // cost plateau 只有在一阶最优性也低于该宽松阈值时才是收敛，否则进入恢复。
        double cost_plateau_gradient_tolerance = 1e-4;
        double scaled_step_tolerance = 1e-12;
        StepControlOptions step_control;
        double curvature_cosine_threshold = 1e-8;
        double history_update_min_model_ratio = 0.25;
    };

    enum class Status {
        CONVERGED_FIRST_ORDER,
        CONVERGED_COST,
        ITERATION_LIMIT,
        EVALUATION_LIMIT,
        STALLED_SMALL_STEP,
        STALLED_LINE_SEARCH,
        STALLED_COST_PLATEAU,
        INVALID_INITIAL_EVALUATION,
        NUMERICAL_FAILURE,
    };

    struct Result {
        Status status = Status::ITERATION_LIMIT;
        bool has_finite_incumbent = false;
        // 相对初值取得至少 cost_window_relative_tolerance 的真实目标下降。
        bool made_progress = false;
        double cost = 0.0; // 当前有限 incumbent 的 raw cost
        double initial_grad_inf_norm = 0.0; // raw gradient
        double grad_inf_norm = 0.0;         // raw gradient
        double scaled_gradient_max_block_norm = 0.0;
        double objective_scale = 1.0;

        int accepted_iterations = 0;
        int function_evaluations = 0; // 含初始评估
        int trial_evaluations = 0;
        int rejected_trials = 0;
        int nonfinite_trials = 0;

        double initial_step_cap = 0.0;
        double final_step_cap = 0.0;
        double min_step_cap = 0.0;
        double max_step_cap = 0.0;
        int step_cap_shrinks = 0;
        int step_cap_expansions = 0;
        int step_cap_hits = 0;

        int history_updates = 0;
        int history_skips = 0;
        int history_resets = 0;

        double last_actual_reduction = 0.0;    // 归一化目标
        double last_predicted_reduction = 0.0; // 归一化目标
        double last_model_ratio = 0.0;
        double last_relative_cost_reduction = 0.0; // raw 目标，以上一个 accepted cost 归一
        double window_relative_cost_reduction = 0.0;
        int cost_plateau_recoveries = 0;

        [[nodiscard]] bool converged() const noexcept {
            return status == Status::CONVERGED_FIRST_ORDER
                || status == Status::CONVERGED_COST;
        }
    };

    using CostFunction = std::function<double(const Eigen::VectorXd& x, Eigen::VectorXd& grad)>;

    explicit LbfgsMinimizer(Options options) : opt_(options) {}

    [[nodiscard]] static std::string_view status_string(Status status) noexcept;

    Result minimize(
        const CostFunction& cost_fn,
        Eigen::VectorXd& x,
        const std::vector<VariableBlock>& blocks
    ) const;

private:
    Options opt_;
};

} // namespace nav_executor
