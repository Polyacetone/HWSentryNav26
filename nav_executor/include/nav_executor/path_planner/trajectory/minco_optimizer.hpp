#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/trajectory/minco_trajectory.hpp>
#include <nav_executor/path_planner/numerics/lbfgs_minimizer.hpp>
#include <nav_executor/path_planner/trajectory/minco_minjerk.hpp>
#include <nav_executor/common/environment/nav_map.hpp>

namespace nav_executor {

// ── MINCO 几何塑形优化器 ──
//
// 决策变量 = 内部路点 Q（DIM×(N-1)）+ MINCO 参数时长 T（经正性重参数化）。T 只
// 作为内部速度见证参与 min-jerk 正则和台阶速度窗塑形，不成为执行速度。固定几何上
// 的唯一执行时标仍由 PathSpeedProfile 给出。
//
// 关键不变量：
//   1. 边界由单位切向与曲率描述，不接收伪造的物理速度；
//   2. 除台阶速度窗外，几何罚项只使用曲率、弧长曲率变化率和方向等时标不变量；
//   3. MINCO 参数时长绝不写入最终 PathSpeedProfile。
//
// 目标：
//   J = w_energy·∫‖p'''‖² + w_time·ΣT
//     + ∫_弧长 [ 障碍 + 曲率/曲率变化率包络 + 有向正则性
//               + runup-to-exit 速度窗/方向/曲率正则 + 禁止方向 ] ds
class MincoOptimizer {
public:
    struct Weights {
        double energy = 1.0;                   // 参数域 min-jerk 正则
        double time = 16.0;                    // 参数区间总长度
        double obstacle = 1000.0;
        double curvature = 100.0;
        double curvature_rate = 100.0;
        double directed_regularity = 1000.0;
        double traversal_velocity_window = 1600.0;
        double traversal_alignment = 200.0;
        double prohibited_traversal = 1000.0;
        double runup_curvature = 100.0;       // 台阶场及其助跑区内的 κ² 正则
    };

    // 方向地形罚的平滑门控（连续 smoothstep，替代离散 label/阈值硬开关）。
    // 以方向场插值模长 ‖dir‖ 为自变量：< norm_lo 罚项关闭，> norm_hi 全强度，
    // 区间内 C1 平滑过渡。
    struct TerrainGate {
        double norm_lo = 0.1;
        double norm_hi = 0.9;
    };

    struct OptimizerParams {
        double position_scale = 0.5;       // 每个 2D waypoint block 的物理尺度 (m)
        double physical_time_scale = 0.1;  // 每个 time block 的目标物理尺度 (s)
        double max_virtual_time_scale = 20.0;
        int max_function_evaluations = 4000;
        int history_size = 8;
        double gradient_tolerance = 1e-5;
        double cost_window_relative_tolerance = 1e-5;
        int cost_window_size = 10;
        double cost_plateau_gradient_tolerance = 1e-4;
        double scaled_step_tolerance = 1e-12;
        LbfgsMinimizer::StepControlOptions step_control;
        double curvature_cosine_threshold = 1e-8;
        double history_update_min_model_ratio = 0.25;
    };

    struct Params {
        Weights weights;
        GeometryLimits geometry;
        double directed_cosine_min = 0.1;
        TerrainGate terrain_gate;
        int samples_per_segment = 16;   // 每段约束采样点数
        int max_iterations = 200;
        OptimizerParams optimizer;
        double min_segment_time = 0.05;
        double runup_body_norm_lo = 0.9; // 助跑源仅由物理本体附近激活
        double runup_body_norm_hi = 0.95;
        double runup_saturation_length = 0.05; // 前向地形暴露积分达到该弧长后基本饱和
        double runup_transition_distance = 0.1; // run_up 外侧的平滑弧长过渡带
        bool debug_diagnostics = false;       // 若开：记录 seed/final 分项代价、路点位移、收敛状态
    };

    // 分项代价拆解（诊断用）：与 J 各项一一对应，便于定位优化被哪一项主导 / 卡住。
    struct CostTerms {
        double energy = 0.0;
        double time = 0.0;
        double obstacle = 0.0;
        double curvature = 0.0;
        double curvature_rate = 0.0;
        double directed_regularity = 0.0;
        double traversal_velocity_window = 0.0;
        double traversal_alignment = 0.0;
        double prohibited_traversal = 0.0;
        double runup_curvature = 0.0;
        [[nodiscard]] double total() const {
            return energy + time + obstacle + curvature + curvature_rate
                + directed_regularity + traversal_velocity_window
                + traversal_alignment + prohibited_traversal + runup_curvature;
        }
    };

    struct Result {
        MincoTrajectory trajectory;
        bool success = false;
        double cost = 0.0;
        LbfgsMinimizer::Status optimizer_status = LbfgsMinimizer::Status::ITERATION_LIMIT;
        int accepted_iterations = 0;
        int function_evaluations = 0;
        int trial_evaluations = 0;
        int rejected_trials = 0;
        int nonfinite_trials = 0;
        double final_grad_inf_norm = 0.0; // raw mixed-coordinate gradient
        double final_scaled_gradient_max_block_norm = 0.0;

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
        double last_relative_cost_reduction = 0.0;
        double window_relative_cost_reduction = 0.0;
        int cost_plateau_recoveries = 0;
        double last_actual_reduction = 0.0;
        double last_predicted_reduction = 0.0;
        double last_model_ratio = 0.0;

        [[nodiscard]] std::string_view optimizer_status_string() const noexcept {
            return LbfgsMinimizer::status_string(optimizer_status);
        }

        // 失败诊断（success=false 时填充，供调用方日志）。
        std::string error;

        // 分项与 block 梯度诊断（debug_diagnostics=true 时填充）。
        bool diagnostics_valid = false;
        CostTerms seed_costs;              // 优化前（种子）分项代价
        CostTerms final_costs;             // 优化后（最终解）分项代价
        double initial_grad_inf_norm = 0.0; // 种子处 raw ‖g‖_inf
        // 最终解处梯度按变量类别拆分（区分几何 vs 时序的剩余下降方向）。
        double final_grad_pos_inf_norm = 0.0; // 位置变量分量 ‖g_q‖_inf
        double final_grad_time_inf_norm = 0.0; // virtual-time 变量分量 ‖g_tau‖_inf
        double waypoint_total_displacement = 0.0; // Σ‖q_final − q_seed‖（仅自由路点）
        double waypoint_max_displacement = 0.0;    // max‖q_final − q_seed‖（仅自由路点）
        int free_waypoint_count = 0;       // 参与优化的自由路点数
    };

    explicit MincoOptimizer(Params params);

    // 由几何 seed 初始化并优化。边界中的 tangent 必须为非零有向切向。
    Result optimize(
        const std::vector<GeometricBoundary>& seed_boundaries,
        const std::vector<double>& seed_durations,
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const TerrainTraversalConstraints& terrain_constraints
    ) const;

private:
    struct Workspace;

    // L-BFGS 目标 + 梯度回调。
    double evaluate(Workspace& ws, const Eigen::VectorXd& vars, Eigen::VectorXd& grad) const;

    // 沿轨迹采样点累积罚代价与对 (c, T) 的梯度。
    // terms 非空时额外按分项拆解代价（诊断用，不影响梯度）。
    double accumulate_penalties(
        Workspace& ws,
        Eigen::MatrixXd& grad_c,
        Eigen::VectorXd& grad_t_explicit,
        CostTerms* terms = nullptr
    ) const;

    Params params_;
};

} // namespace nav_executor
