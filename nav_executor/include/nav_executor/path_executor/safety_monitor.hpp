#pragma once

#include <chrono>
#include <optional>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/path_planner/nav_map.hpp>
#include <nav_executor/path_executor/state_machine.hpp>

namespace nav_executor {

class SafetyMonitor {
public:
    SafetyMonitor(const FsmParams& fsm_params, rclcpp::Logger logger);

    bool check_stuck(
        const Eigen::Vector2d& chassis_pos,
        double last_cmd_vel,
        std::chrono::steady_clock::time_point stamp
    );
    void reset_stuck();

    [[nodiscard]] bool compute_is_hazard(
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const Eigen::Vector2d& pos
    ) const;

    bool check_recovery_safe(
        const CostMap& final_cost_map,
        const DirectionMap& direction_map,
        const Eigen::Vector2d& pos,
        std::chrono::steady_clock::time_point stamp
    );

    void update_recovery_goal_if_needed(
        const Eigen::Vector3d& chassis_pose,
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const DirectionMap* base_dir_map,
        std::chrono::steady_clock::time_point stamp
    );

    void reset_recovery();

    [[nodiscard]] const std::optional<Eigen::Vector2d>& recovery_goal() const {
        return recovery_goal_map_;
    }

private:
    FsmParams fsm_params_;
    rclcpp::Logger logger_;

    bool stuck_active_ = false;
    std::chrono::steady_clock::time_point stuck_start_time_;
    Eigen::Vector2d stuck_start_pos_ = Eigen::Vector2d::Zero();

    std::optional<Eigen::Vector2d> recovery_goal_map_;
    std::optional<std::chrono::steady_clock::time_point> recovery_goal_set_time_;
    std::optional<std::chrono::steady_clock::time_point> recovery_safe_since_;
};

} // namespace nav_executor
