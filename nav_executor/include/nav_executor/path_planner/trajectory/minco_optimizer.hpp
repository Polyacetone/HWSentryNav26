#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/trajectory/minco_trajectory.hpp>
#include <nav_executor/common/trajectory/trajectory_limits.hpp>
#include <nav_executor/path_planner/numerics/lbfgs_minimizer.hpp>
#include <nav_executor/path_planner/trajectory/minco_minjerk.hpp>
#include <nav_executor/common/environment/nav_map.hpp>

namespace nav_executor {

// ── MINCO 几何优化器 ──
//
// 只优化 2D 平坦输出 (x,y) 的**空间形状**：决策变量 = 内部路点 Q（DIM×(N-1)）+ 段时长 T
// （经虚拟时间正性重参数化）。物理时标不在这里决定，而由 PathSpeedProfile 按弧长统一给出；
// 因此本优化器的 τ 只是一个归一化参数，速度被约束在 [directed_speed_min, velocity_max]
// 之内，使 w_time·ΣT 成为弧长的代理罚。
//
// 关键不变量：
//   1. 有向正则性 p_τ·t̂_seed ≥ directed_speed_min > 0，排除内部零速点、尖点与逆向；
//   2. 曲率与曲率变化率使用与跟随层同一份真实几何定义（TrajSample::kappa/kappa_rate）。
//
// 目标：
//   J = w_energy·∫‖jerk‖² + w_time·ΣT
//     + Σ_采样点 [ 障碍 + 参数化速度上界 + 有向正则性 + |κ|界 + |dκ/ds|界
//                 + 膨胀方向场对齐 + 禁止方向 + 助跑区 κ² ]
class MincoOptimizer {
public:
    struct Weights {
        double energy = 1.0;                  // min-jerk 能量
        double time = 16.0;                   // 归一化参数总长（弧长代理）
        double obstacle = 1000.0;             // 障碍罚
        double parameterization_velocity = 100.0; // ‖p_τ‖ ≤ velocity_max
        double directed_regularity = 1000.0;  // p_τ·t̂_seed ≥ directed_speed_min
        double curvature = 400.0;             // |κ| ≤ κ_max
        double curvature_rate = 200.0;        // |dκ/ds| ≤ κ'_max
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
        double scaled_step_tolerance = 1e-8;
        LbfgsMinimizer::TrustRegionOptions trust_region;
        double curvature_relative_threshold = 1e-8;
        double history_acceptance_ratio = 0.25;
    };

    struct Params {
        Weights weights;
        TrajectoryLimits limits;
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
        double parameterization_velocity = 0.0;
        double directed_regularity = 0.0;
        double curvature = 0.0;
        double curvature_rate = 0.0;
        double traversal_alignment = 0.0;
        double prohibited_traversal = 0.0;
        double runup_curvature = 0.0;
        [[nodiscard]] double total() const {
            return energy + time + obstacle + parameterization_velocity
                + directed_regularity + curvature + curvature_rate
                + traversal_alignment + prohibited_traversal + runup_curvature;
        }
    };

    struct Result {
        MincoTrajectory trajectory;
        bool success = false;
        double cost = 0.0;
        LbfgsMinimizer::Status optimizer_status = LbfgsMinimizer::Status::MAX_ITERATIONS;
        int accepted_iterations = 0;
        int function_evaluations = 0;
        int trial_evaluations = 0;
        int rejected_trials = 0;
        int nonfinite_trials = 0;
        double final_grad_inf_norm = 0.0; // raw mixed-coordinate gradient，仅作兼容诊断
        double final_normalized_scaled_grad_max_block_norm = 0.0;

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

    // 由平坦 seed 初始化并优化。
    //   seed_states：N+1 个 2D 边界全状态（含 head/tail 的 pos/vel/acc，仅 x,y）；
    //   seed_durations：N 段初始参数长度；
    //   seed_tangents：N+1 个边界处的 A* 有向单位切向，定义有向正则性参考方向。
    Result optimize(
        const std::vector<MincoMinJerk::BoundaryPVA>& seed_states,
        const std::vector<double>& seed_durations,
        const std::vector<Eigen::Vector2d>& seed_tangents,
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
