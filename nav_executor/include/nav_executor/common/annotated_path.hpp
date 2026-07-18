#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/chassis_defs.hpp>
#include <nav_executor/common/minco_trajectory.hpp>
#include <nav_executor/path_planner/nav_map.hpp>
#include <nav_executor/path_executor/solver/mpc_types.hpp>

namespace nav_executor {

// 规划期生成、运行期只读的台阶区段。
struct StepPlanSegment {
    double prepare_u = 0.0; // 开始渐变 capability 的位置。
    double active_u = 0.0; // 开始给底盘发送台阶模式的位置。同时是状态机进入台阶模式的位置。
    double commit_u = 0.0; // 底盘执行器实际开始做地形跨越动作的预期位置。因此是 runup 约束开始处。
    double step_enter_u = 0.0; // 物理台阶边缘入口。
    double step_exit_u = 0.0; // 物理台阶边缘出口。
    double release_u = 1.0; // 状态机退出台阶模式的位置。
    Eigen::Vector2d step_enter_pos_map = Eigen::Vector2d::Zero();
    Eigen::Vector2d step_exit_pos_map = Eigen::Vector2d::Zero();
    Eigen::Vector2d dir_map = Eigen::Vector2d::Zero();
    StepDirection direction = StepDirection::UP;
    StepChassisCommand chassis_command;
    StepTraversalConstraint traversal_constraint;
    uint8_t terrain_label = 0;
    bool requires_high_performance = false;
};

// 一次规划产出的不可变路径包。
struct AnnotatedPath {
    using ConstPtr = std::shared_ptr<const AnnotatedPath>;

    explicit AnnotatedPath(MincoTrajectory trajectory_in) : trajectory(std::move(trajectory_in)) {}

    MincoTrajectory trajectory;

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
