#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/path_planner/dijkstra_cost_to_goal.hpp>

namespace nav_executor {

// 前向二维平坦状态搜索。状态为 (p, p_dot)，控制为世界系恒定平坦加速度。
// 运动原语在当前速度切向/法向基中采样，因此高速时仍保留满足侧向加速度约束的缓弯分支。
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

        int tangential_accel_samples = 5;
        int normal_accel_samples = 7;
        double primitive_duration = 0.3;
        int collision_substeps = 4;

        double dedup_xy = 0.2;
        double dedup_theta = 0.314;
        double dedup_speed = 0.4;

        double heuristic_weight = 1.0;
        double goal_tolerance = 0.3;
        int max_expansions = 200000;
    };

    using TransitionFeasibleFn = std::function<bool(const State& from, const State& to)>;

    struct Result {
        std::vector<State> states;
        std::vector<double> durations;
        bool success = false;
        int expansions = 0;
        std::string error;
    };

    explicit KinodynamicAstar(Params params) : params_(std::move(params)) {}

    Result search(
        const State& start,
        const Eigen::Vector2d& goal_position,
        const DijkstraCostToGoal& dijkstra,
        const TransitionFeasibleFn& transition_feasible
    ) const;

private:
    Params params_;
};

} // namespace nav_executor
