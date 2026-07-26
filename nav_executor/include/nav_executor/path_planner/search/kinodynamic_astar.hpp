#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/path_planner/search/dijkstra_cost_to_goal.hpp>

namespace nav_executor {

// 空间域前向时空搜索。几何状态为 (p, theta)，每个 label 携带该姿态处的可达速度平方区间。
// 固定弧长、恒曲率原语保证每次展开都有空间进展；纵向控制通过速度区间传播压缩，避免枚举
// 切向/法向加速度笛卡尔积。搜索只证明拓扑及速度可达性，最终连续动力学由 MINCO 负责。
class KinodynamicAstar {
public:
    struct State {
        Eigen::Vector2d position = Eigen::Vector2d::Zero();
        Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
    };

    struct Params {
        struct StateLimits {
            double speed_max = 3.2;
            double angular_velocity_max = 6.0;
            double acceleration_max = 1.8;
            double lateral_acceleration_max = 2.0;
        } state_limits;

        int curvature_samples = 7;
        double curvature_max = 4.0;
        double primitive_length = 0.3;
        double collision_check_resolution = 0.075;
        double goal_connection_max_length = 1.0;

        double dedup_xy = 0.2;
        double dedup_theta = 0.314;

        double heuristic_weight = 3.0;
        double goal_tolerance = 0.3;
        int max_expansions = 200000;
    };

    struct Pose {
        Eigen::Vector2d position = Eigen::Vector2d::Zero();
        double theta = 0.0;
    };

    struct SpeedRange {
        double min = 0.0;
        double max = 0.0;
    };

    // 多源搜索根。initial_cost 与空间路径 g 使用相同的等效米单位，用于表达
    // 起始运动状态松弛的偏好代价，不参与最终轨迹时长重建。
    struct SearchRoot {
        State state;
        double initial_cost = 0.0;
        bool relaxed = false;
    };

    // 返回几何子段终点允许的速度范围；nullopt 表示碰撞、越界或地形方向不可行。
    using TransitionConstraintFn =
        std::function<std::optional<SpeedRange>(const Pose& from, const Pose& to)>;

    struct Result {
        std::vector<State> states;
        std::vector<double> durations;
        bool success = false;
        int expansions = 0;
        int generated_labels = 0;
        int dominated_labels = 0;
        int transition_checks = 0;
        int goal_connection_attempts = 0;
        size_t open_peak = 0;
        bool selected_relaxed_root = false;
        double selected_root_cost = 0.0;
        std::string error;
    };

    explicit KinodynamicAstar(Params params) : params_(std::move(params)) {}

    Result search(
        const std::vector<SearchRoot>& roots,
        const Eigen::Vector2d& goal_position,
        const DijkstraCostToGoal& dijkstra,
        const TransitionConstraintFn& transition_constraint
    ) const;

private:
    Params params_;
};

} // namespace nav_executor
