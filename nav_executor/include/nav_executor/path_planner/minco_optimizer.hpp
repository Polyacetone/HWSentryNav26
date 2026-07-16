#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/minco_trajectory.hpp>
#include <nav_executor/path_planner/minco_minjerk.hpp>
#include <nav_executor/path_planner/nav_map.hpp>

namespace nav_executor {

// ── MINCO 时空优化器（微分平坦，替换 bspline_optimizer）──
//
// 只优化 2D 平坦输出 (x,y)：决策变量 = 内部路点 Q（DIM×(N-1)）+ 段时长 T（经虚拟时间
// 正性重参数化）。朝向 θ = atan2(gear·运动方向) 由平坦输出解析导出，非完整约束因此**恒等
// 满足**，不再需要增广拉格朗日或 θ 自由度。L-BFGS 无约束优化，约束以采样罚实现，梯度经
// MincoMinJerk::propagate_gradient 从系数回传到 Q/T。换向尖点由 seed 冻结（两侧 v=0）。
//
// 目标：
//   J = w_energy·∫‖jerk‖² + w_time·ΣT
//     + Σ_采样点 [ 障碍 + 带符号速度窗 + |v·ω|≤a_lat + |ω|界 + |a|界
//                 + 方向地形对齐 + 台阶速度窗 ]
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
        double min_segment_time = 0.05;
        double step_entry_window_fraction = 0.25;
        bool debug_check_gradient = false;    // 若开：optimize 起始处对比解析梯度 vs 有限差分并记录
    };

    // 台阶入口硬约束位置（朝向由平坦输出的运动方向导出，无需单独固定）。
    struct HardWaypoint {
        int waypoint_index = -1;
        Eigen::Vector2d position = Eigen::Vector2d::Zero();
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
        double final_grad_inf_norm = 0.0;
        int line_search_iterations = 0;

        // 失败诊断（success=false 时填充，供调用方日志）。
        std::string error;

        // 梯度自检结果（debug_check_gradient=true 时填充；否则为负表示未做）。
        double grad_check_max_abs_err = -1.0;
        double grad_check_max_rel_err = -1.0;
        int grad_check_worst_index = -1;
        int grad_check_num_time_vars = 0;
    };

    explicit MincoOptimizer(Params params);

    // 由平坦 seed 初始化并优化。
    //   seed_states：N+1 个 2D 边界全状态（含 head/tail 的 pos/vel/acc，仅 x,y）；
    //   seed_durations：N 段初始时长；
    //   seed_gears：N 段换向符号 ±1（前端冻结的离散拓扑）；
    //   cusp_waypoints：N-1 内部节点掩码，true 表示换向尖点（两侧 v=0）。
    Result optimize(
        const std::vector<MincoMinJerk::BoundaryPVA>& seed_states,
        const std::vector<double>& seed_durations,
        const std::vector<double>& seed_gears,
        const std::vector<char>& cusp_waypoints,
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

    // 沿轨迹采样点累积罚代价与对 (c, T) 的梯度。
    double accumulate_penalties(
        Workspace& ws,
        Eigen::MatrixXd& grad_c,
        Eigen::VectorXd& grad_t_explicit
    ) const;

    Params params_;
};

} // namespace nav_executor
