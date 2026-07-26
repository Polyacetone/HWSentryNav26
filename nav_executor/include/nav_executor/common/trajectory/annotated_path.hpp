#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/trajectory/minco_trajectory.hpp>
#include <nav_executor/common/trajectory/path_speed_profile.hpp>
#include <nav_executor/common/trajectory/step_plan.hpp>
#include <nav_executor/common/environment/nav_map.hpp>

namespace nav_executor {

// 一次规划产出的不可变路径包。
struct AnnotatedPath {
    using ConstPtr = std::shared_ptr<const AnnotatedPath>;

    explicit AnnotatedPath(MincoTrajectory trajectory_in) : trajectory(std::move(trajectory_in)) {}

    MincoTrajectory trajectory;
    PathSpeedProfile speed_profile;

    Eigen::Vector2d goal_pos = Eigen::Vector2d::Zero();
    bool goal_fixed = false;

    // 用于拒绝过期规划结果。
    uint64_t goal_id = 0;
    PerformanceState planning_performance;

    std::vector<StepPlanSegment> step_segments;
    std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule;

    CostMap::ConstPtr step_cost_layer;
    DirectionMap::ConstPtr masked_direction_map;
};

} // namespace nav_executor
