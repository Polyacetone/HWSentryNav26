#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/minco_trajectory.hpp>
#include <nav_executor/path_planner/minco_minjerk.hpp>
#include <nav_executor/path_planner/nav_map.hpp>

namespace nav_executor {

// ── MINCO 时空优化器（替换 bspline_optimizer）──
//
// 全状态独立 (x,y,θ) MINCO min-jerk 消元：决策变量 = 内部路点 Q + 段时长 T（经虚拟时间
// 正性重参数化）。L-BFGS（GCOPTER 依赖，已 vendored）无约束优化，约束以采样罚 + 增广
// 拉格朗日（非完整约束）实现。梯度经 MincoMinJerk::propagate_gradient 从系数回传到 Q/T。
//
// 目标（落地版 §[3]）：
//   J = w_energy·∫‖jerk‖² + w_time·ΣT
//     + Σ_采样点 [ 障碍 + 速度窗 + |v·ω|≤a_lat + |ω|界 + |a|界 + θ̇正则 ]
//     + AL(非完整 ẋsinθ − ẏcosθ = 0)
class MincoOptimizer {
public:
    struct Weights {
        double energy = 1.0;         // min-jerk 能量
        double time = 16.0;          // 总时长
        double obstacle = 1000.0;    // 障碍罚
        double velocity = 100.0;     // |v| ≤ v_max（带符号，倒车用 v_min）
        double lateral_acc = 100.0;  // |v·ω| ≤ a_lat_max
        double omega = 100.0;        // |ω| ≤ omega_max
        double accel = 100.0;        // |dv/dt| ≤ acc_max
        double theta_rate = 1.0;     // θ̇ 正则（抑制不该转处乱转）
        double heading_follow = 1.0; // |v| 大时 θ 跟随运动方向的软偏好
        double step_alignment = 200.0; // 方向地形内车身轴与穿越方向对齐
        double step_velocity = 200.0;  // 方向地形内所选模式的速度窗
    };

    struct Limits {
        double vel_max = 2.0;
        double vel_min = -1.6;
        double omega_max = 6.0;
        double acc_max = 1.8;
        double a_lat_max = 2.0;
    };

    struct Params {
        Weights weights;
        Limits limits;
        int samples_per_segment = 16;   // 每段约束采样点数
        int max_iterations = 200;
        double nonholonomic_rho_init = 1.0;   // AL 初始罚因子
        double nonholonomic_rho_max = 1e4;
        double nonholonomic_rho_scale = 4.0;  // 每轮 AL 外层放大
        int nonholonomic_al_rounds = 4;       // AL 外层轮数
        double nonholonomic_tolerance = 1e-2; // AL 提前收敛目标：max |横向速度| (m/s)
        double nonholonomic_acceptance_tolerance = 3e-2; // 最终轨迹接纳上限 (m/s)
        double min_segment_time = 0.05;
        double step_entry_window_fraction = 0.25;
        bool debug_check_gradient = false;    // 若开：optimize 起始处对比解析梯度 vs 有限差分并记录
    };

    // 台阶入口只硬约束位置和朝向；内部速度仍是 MINCO 自由变量。
    struct HardWaypoint {
        int waypoint_index = -1;
        Eigen::Vector2d position = Eigen::Vector2d::Zero();
        double theta = 0.0;
    };

    struct StepEntrySpeedWindow {
        int waypoint_index = -1;
        double speed_min = 0.0;
        double speed_max = 0.0;
    };

    struct Result {
        MincoTrajectory trajectory;
        bool success = false;
        double cost = 0.0;
        int iterations = 0;
        int al_rounds = 0;
        double final_rho = 0.0;
        double final_grad_inf_norm = 0.0;
        int line_search_iterations = 0;
        double max_nonholonomic_violation = 0.0;

        // 失败诊断（success=false 时填充，供调用方日志）。
        std::string error;

        // 梯度自检结果（debug_check_gradient=true 时填充；否则为负表示未做）。
        double grad_check_max_abs_err = -1.0;
        double grad_check_max_rel_err = -1.0;
        int grad_check_worst_index = -1;
        int grad_check_num_time_vars = 0;
    };

    explicit MincoOptimizer(Params params);

    // 由几何 seed（含朝向 / 速度猜测）初始化并优化。
    //   seed_states：N+1 个边界全状态（含 head/tail）；durations：N 段初始时长。
    //   cost_map / 其梯度用于障碍罚。
    Result optimize(
        const std::vector<MincoMinJerk::BoundaryPVA>& seed_states,
        const std::vector<double>& seed_durations,
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const TerrainTraversalConstraints& terrain_constraints,
        const std::vector<HardWaypoint>& hard_waypoints,
        const std::vector<StepEntrySpeedWindow>& step_entry_speed_windows
    ) const;

private:
    struct Workspace;

    // L-BFGS 目标 + 梯度回调。
    double evaluate(Workspace& ws, const Eigen::VectorXd& vars, Eigen::VectorXd& grad) const;

    // 沿轨迹采样点累积罚代价与对 (c, T) 的梯度；同时累积非完整违反的 AL 项。
    double accumulate_penalties(
        Workspace& ws,
        Eigen::MatrixXd& grad_c,
        Eigen::VectorXd& grad_t_explicit
    ) const;

    // AL 外层：用当前 ws.minco 逐采样点算非完整违反 h，更新乘子 lambda += rho·h。
    // 返回 max|h|。
    double update_nonholonomic_multipliers(Workspace& ws) const;

    Params params_;
};

} // namespace nav_executor
