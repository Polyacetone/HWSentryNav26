#pragma once

#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/common/annotated_path.hpp>
#include <nav_executor/path_planner/nav_map.hpp>

namespace nav_executor {

struct FollowProjectionGuardParams {
    double dist_max;
    double cost_max;
    int cost_samples;
};

struct StepBlockReplanParams {
    bool enable;
    double lookahead_distance;
    double sample_resolution;
    double step_norm_threshold;
    double obstacle_cost_threshold;
    double predicted_obstacle_ratio_threshold;
};

struct PerformanceReplanParams { double lookahead_distance; };

// RouteMonitor 输出的重规划原因。前三种由本模块产出，EXECUTOR_REPLAN_EVENT 由 TaskManager 消费，此处仅用于诊断记录。
enum class ReplanReason : uint8_t {
    NONE = 0,
    PROJECTION_GUARD = 1,
    STEP_BLOCKED = 2,
    EXECUTOR_REPLAN_EVENT = 3,
    PERFORMANCE_DEGRADED = 4,
    PERFORMANCE_RECOVERED = 5,
};

const char* replan_reason_str(ReplanReason reason);

// RouteMonitor 输入，由 nav_executor_node 每周期组装后传入。
struct RouteMonitorInput {
    AnnotatedPath::ConstPtr active_path;
    double current_u = 0.0;

    Eigen::Vector2d chassis_pos_map = Eigen::Vector2d::Zero();

    const CostMap* masked_global_cost_map = nullptr;
    const CostMap* current_dynamic_cost_map = nullptr;
    std::vector<const CostMap*> per_step_dynamic_cost_maps;

    const DirectionMap* masked_direction_map = nullptr;

    FollowProjectionGuardParams proj_guard{};
    StepBlockReplanParams step_block{};
    PerformanceReplanParams performance{};
    PerformanceState current_performance{};
};

struct RouteMonitorReport {
    double current_u = 0.0;
    bool needs_replan = false;
    ReplanReason reason = ReplanReason::NONE;
};

// 顶层 FOLLOW 阶段的 path monitoring：无状态纯函数，仅当 motion_state==FOLLOW 且 active_path 非空时调用。
// 输出 needs_replan 的合法原因见 ReplanReason。
RouteMonitorReport run_route_monitor(const RouteMonitorInput& input, rclcpp::Logger logger);

} // namespace nav_executor
