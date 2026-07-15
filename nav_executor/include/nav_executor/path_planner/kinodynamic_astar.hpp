#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/path_planner/dijkstra_cost_to_goal.hpp>
#include <nav_executor/path_planner/nav_map.hpp>

namespace nav_executor {

// ── Kinodynamic A*（最简差速车模型）──
//
// 状态 (x, y, θ, v)：倒车折进 v 符号，原地旋转由 v=0,ω≠0 自然表达。
// 原语：采样 (a, ω)，前向积分 ẋ=v cosθ, ẏ=v sinθ, θ̇=ω, v̇=a，逐细步碰撞检查 + |v·ω|≤a_lat 剪枝。
// 只做**拓扑发现**（倒车/绕行/原地转的离散结构）——保真度交给 MINCO/FDDP。
// h = Dijkstra cost-to-goal（质点最短路下界，O(1) 查）。
//
// 产出：从 start 到 goal 邻域的全状态机动序列（供 MINCO 播种）。
class KinodynamicAstar {
public:
    struct State {
        double x = 0.0;
        double y = 0.0;
        double theta = 0.0;
        double v = 0.0;
    };

    struct Params {
        // 运动学界
        double vel_max = 2.0;
        double vel_min = -1.6;
        double omega_max = 6.0;
        double accel_max = 1.8;
        double a_lat_max = 2.0;

        // 原语采样
        int accel_samples = 5;      // [-accel_max, accel_max] 均匀采样
        int omega_samples = 7;      // [-omega_max, omega_max] 均匀采样
        double primitive_duration = 0.3; // 每个原语积分时长 (s)
        int collision_substeps = 4;      // 原语内碰撞检查细分步数

        // 去重分辨率
        double dedup_xy = 0.2;      // 位置 (m)
        double dedup_theta = 0.314; // 朝向 (rad)，≈18°
        double dedup_v = 0.4;       // 速度 (m/s)

        // 代价权重
        double time_weight = 1.0;       // 每原语时长代价
        double reverse_weight = 0.3;    // 倒车额外代价（偏好前进，但不禁止）
        double heuristic_weight = 1.0;  // Dijkstra h 权重（>1 加速但失去最优性）

        // 终止
        double goal_tolerance = 0.3;    // 到 goal 位置容差 (m)
        int max_expansions = 200000;    // 展开节点上限（内存/时间护栏）
    };

    // 原语逐子步可行性回调。除碰撞外，调用方在这里施加方向地形的通行方向、
    // 入口朝向和速度窗约束。
    using TransitionFeasibleFn = std::function<bool(const State& from, const State& to)>;
    using GoalReachedFn = std::function<bool(const State& state)>;

    struct Result {
        std::vector<State> states; // 含起点与终点的全状态序列
        bool success = false;
        int expansions = 0;
        std::string error;
    };

    explicit KinodynamicAstar(Params params) : params_(std::move(params)) {}

    // dijkstra 提供 h；transition_feasible 逐子步判定可行性；goal_reached 定义任务终止语义。
    Result search(
        const State& start,
        const DijkstraCostToGoal& dijkstra,
        const TransitionFeasibleFn& transition_feasible,
        const GoalReachedFn& goal_reached
    ) const;

private:
    Params params_;
};

} // namespace nav_executor
