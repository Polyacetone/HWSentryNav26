#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/chassis_defs.hpp>
#include <nav_executor/common/spline_path.hpp>
#include <nav_executor/path_planner/nav_map.hpp>
#include <nav_executor/path_executor/solver/mpc_types.hpp>

namespace nav_executor {

// 规划期产出的台阶几何标注，运行时只读。
struct StepPlanSegment {
    double prepare_u = 0.0;
    double active_u = 0.0;
    double step_enter_u = 0.0;
    double step_exit_u = 0.0;
    double release_u = 1.0;
    Eigen::Vector2d step_enter_pos_map = Eigen::Vector2d::Zero();
    Eigen::Vector2d step_exit_pos_map = Eigen::Vector2d::Zero();
    Eigen::Vector2d dir_map = Eigen::Vector2d::Zero();
    StepDirection direction = StepDirection::UP;
    ActiveStepMode command;
    uint8_t terrain_label = 0;
};

// 一次规划产出的不可变路径包。
struct AnnotatedPath {
    using ConstPtr = std::shared_ptr<const AnnotatedPath>;

    explicit AnnotatedPath(SplinePath spline_in) : spline(std::move(spline_in)) {}

    SplinePath spline;

    Eigen::Vector2d goal_pos = Eigen::Vector2d::Zero();
    bool goal_fixed = false;

    // 对应 PlanRequest 所携带的 goal id，用于接纳验证和 goal_reached 守卫。
    uint64_t goal_id = 0;

    std::vector<StepPlanSegment> step_segments;

    // 针对本条样条产出的台阶掩码层。
    CostMap::ConstPtr step_cost_layer;
    DirectionMap::ConstPtr masked_direction_map;
};

} // namespace nav_executor
