#pragma once

#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/common/trajectory/annotated_path.hpp>
#include <nav_executor/common/tracking/route_tracker.hpp>
#include <nav_executor/common/environment/nav_map.hpp>
#include <nav_executor/common/environment/obstacle_semantics.hpp>

namespace nav_executor {

struct FollowProjectionGuardParams {
    double cost_max;
    int cost_samples;
};

struct StepBlockReplanParams {
    bool enable;
    double lookahead_distance;
    double sample_resolution;
    double predicted_obstacle_ratio_threshold;
};

struct PerformanceReplanParams { double lookahead_distance; };

// RouteMonitor 输出的重规划原因。EXECUTOR_REPLAN_EVENT 由 TaskManager 消费，
// 其余原因由 RouteMonitor 直接产出。
enum class ReplanReason : uint8_t {
    NONE = 0,
    ROUTE_TRACKING_LOST = 1,
    PROJECTION_OUT_OF_MAP = 2,
    PROJECTION_COST_EXCEEDED = 3,
    STEP_BLOCKED_CURRENT = 4,
    STEP_BLOCKED_PREDICTED = 5,
    MPC_LETHAL = 6,
    EXECUTOR_REPLAN_EVENT = 7,
    PERFORMANCE_DEGRADED = 8,
    PERFORMANCE_RECOVERED = 9,
    HIGH_PRIORITY_SPIN_PREEMPTION = 10,
};

const char* replan_reason_str(ReplanReason reason);

// RouteMonitor 输入，由 nav_executor_node 每周期组装后传入。
struct RouteMonitorInput {
    AnnotatedPath::ConstPtr active_path;
    RouteEstimate route;

    Eigen::Vector2d chassis_pos_map = Eigen::Vector2d::Zero();

    const FollowerObstacleView* obstacles = nullptr;

    FollowProjectionGuardParams proj_guard{};
    StepBlockReplanParams step_block{};
    PerformanceReplanParams performance{};
    PerformanceState current_performance{};

    // 上一周期 PathExecutor 输出的 one-shot MPC_LETHAL 事实。
    bool mpc_lethal = false;
};

struct RouteMonitorReport {
    bool needs_replan = false;
    ReplanReason reason = ReplanReason::NONE;
};

// 顶层可抢占阶段（含 STEPPING/PREPARING、ARMED）的 path monitoring：
// 无状态纯函数，仅在 active_path 非空时调用。
// 输出 needs_replan 的合法原因见 ReplanReason。
RouteMonitorReport run_route_monitor(const RouteMonitorInput& input, rclcpp::Logger logger);

} // namespace nav_executor
