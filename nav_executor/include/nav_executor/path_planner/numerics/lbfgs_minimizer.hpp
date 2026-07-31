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

    struct TrustRegionOptions {
        double initial_radius = 0.5;
        double min_radius = 1e-10;
        double max_radius = 2.0;
        double acceptance_ratio = 0.1;
        double shrink_ratio = 0.25;
        double expansion_ratio = 0.75;
        double shrink_factor = 0.25;
        double expansion_factor = 2.0;
        int max_consecutive_rejections = 20;
        int history_reset_after_rejections = 3;
    };

    struct Options {
        int max_iterations = 200;
        int max_function_evaluations = 400;
        int history_size = 8;
        // 一阶最优性阈值：max_block ‖D_b g_b‖₂ / max(1, |f|)。
        // D_b 是调用方提供的变量物理尺度，与算法内部的目标归一化无关。
        double first_order_tolerance = 1e-5;
        // 连续 accepted step 的 raw 目标相对改善均不超过该值时，按实际收益收敛。
        double relative_cost_tolerance = 1e-5;
        int cost_convergence_window = 10;
        double scaled_step_tolerance = 1e-12;
        TrustRegionOptions trust_region;
        double curvature_relative_threshold = 1e-8;
        double history_acceptance_ratio = 0.25;
    };

    enum class Status {
        FIRST_ORDER_CONVERGED,
        COST_CONVERGED,
        MAX_ITERATIONS,
        MAX_EVALUATIONS,
        TRUST_REGION_TOO_SMALL,
        STAGNATED,
        INITIAL_EVALUATION_NONFINITE,
        NUMERICAL_FAILURE,
    };

    struct Result {
        Status status = Status::MAX_ITERATIONS;
        double cost = 0.0; // 当前有限 incumbent 的 raw cost
        double initial_grad_inf_norm = 0.0; // raw gradient
        double grad_inf_norm = 0.0;         // raw gradient
        double scaled_grad_max_block_norm = 0.0;
        double first_order_optimality = 0.0;
        double objective_scale = 1.0;

        int accepted_iterations = 0;
        int function_evaluations = 0; // 含初始评估
        int trial_evaluations = 0;
        int rejected_trials = 0;
        int nonfinite_trials = 0;

        double initial_radius = 0.0;
        double final_radius = 0.0;
        double min_radius = 0.0;
        double max_radius = 0.0;
        int radius_shrinks = 0;
        int radius_expansions = 0;
        int boundary_steps = 0;

        int history_updates = 0;
        int history_skips = 0;
        int history_resets = 0;

        double last_actual_reduction = 0.0;    // 归一化目标
        double last_predicted_reduction = 0.0; // 归一化目标
        double last_reduction_ratio = 0.0;
        double last_relative_cost_reduction = 0.0; // raw 目标，以上一个 accepted cost 归一
        int consecutive_small_cost_reductions = 0;
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
