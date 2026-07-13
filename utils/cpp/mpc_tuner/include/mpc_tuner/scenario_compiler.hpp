#pragma once

#include <chrono>
#include <memory>

#include <rclcpp/logger.hpp>

#include <mpc_tuner/types.hpp>

namespace mpc_tuner {

class ScenarioPlanner {
public:
    ScenarioPlanner(
        const RuntimeConfig& config,
        nav_executor::CostMap::ConstPtr global_cost_map,
        nav_executor::DirectionMap::ConstPtr direction_map,
        rclcpp::Logger logger
    );
    ~ScenarioPlanner();

    ScenarioPlanner(const ScenarioPlanner&) = delete;
    ScenarioPlanner& operator=(const ScenarioPlanner&) = delete;

    [[nodiscard]] CompiledScenario plan(
        const ScenarioSpec& spec,
        std::chrono::seconds timeout = std::chrono::seconds(15)
    );

private:
    nav_executor::CostMap::ConstPtr global_cost_map_;
    nav_executor::DirectionMap::ConstPtr direction_map_;
    nav_executor::TerrainTraversalConstraints terrain_constraints_;
    nav_executor::PerformanceState performance_ {.high_performance = true};
    std::shared_ptr<nav_executor::StepRoutingMask> step_mask_;
    std::unique_ptr<nav_executor::PathPlanner> planner_;
    rclcpp::Logger logger_;
    uint64_t next_goal_id_ = 1;
};

} // namespace mpc_tuner
