#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/common/trajectory/annotated_path.hpp>
#include <nav_executor/task_manager/route_monitor.hpp>
#include <nav_executor/common/control_arbitration.hpp>
#include <nav_executor/path_executor/state/state_machine.hpp>
#include <nav_executor/path_planner/path_planner.hpp>

namespace nav_executor {

struct TaskManagerParams {
    double goal_equivalence_distance; // goal-to-goal 去重阈值 (m)
    double plan_cooldown;             // 同一 goal 失败重试冷却 (s)
};

enum class PlannerState : uint8_t { IDLE, PLANNING, COOLDOWN };

enum class PlannerResultState : uint8_t {
    NONE = 0,
    PATH_ACCEPTED = 1,
    COMPLETE = 2,
    FIXED_GOAL = 3,
    FAILED = 4,
};

struct TaskDiagnostics {
    uint64_t goal_id = 0;
    Eigen::Vector2d goal_position = Eigen::Vector2d::Zero();
    bool goal_fixed = false;
    uint64_t active_path_goal_id = 0;
    bool has_hold_goal = false;
    Eigen::Vector2d hold_goal_position = Eigen::Vector2d::Zero();
    uint64_t plan_generation = 0;
    bool needs_plan = false;
    PlannerState planner_state = PlannerState::IDLE;
    double planner_cooldown_remaining = 0.0;
    PlannerResultState planner_last_result = PlannerResultState::NONE;
    std::string planner_last_failure_reason;
    ReplanReason last_replan_reason = ReplanReason::NONE;
    uint64_t replan_count = 0;
    std::vector<Eigen::Vector2d> debug_spatial_path;
    std::vector<Eigen::Vector2d> debug_smoothed_spatial_path;
    std::vector<Eigen::Vector2d> debug_kino_path;
};

struct PlanRequestSnapshot {
    Eigen::Vector2d current_pos_map = Eigen::Vector2d::Zero();
    double current_yaw = 0.0;
    double current_velocity = 0.0;
    PlannerObstacleView obstacles;
    DirectionMap::ConstPtr direction_map;
    TerrainTraversalConstraints terrain_constraints;
    PerformanceState performance;
};

struct MotionFeedback {
    bool goal_reached = false;
    // goal_reached 对应的不可变路径包；用于拒绝晚到的旧路径完成事件。
    AnnotatedPath::ConstPtr goal_reached_path;
    bool executor_replan_event = false;
    bool mpc_lethal = false;
    AnnotatedPath::ConstPtr lethal_path;
};

struct TaskUpdateInput {
    std::optional<Goal> incoming_goal;
    MotionFeedback feedback;
    NavigationAccess navigation_access = NavigationAccess::AVAILABLE;
    PlanRequestSnapshot plan_snapshot;
    std::optional<RouteMonitorInput> route_monitor;
    std::chrono::steady_clock::time_point stamp;
};

struct TaskCommandView {
    AnnotatedPath::ConstPtr active_path;
    std::optional<Eigen::Vector2d> hold_goal;
};

struct TaskUpdateOutput {
    TaskCommandView command;
    TaskDiagnostics diagnostics;
};

// 管理已提交任务、活动路径、保持目标和规划派发。
class TaskManager {
public:
    TaskManager(
        const TaskManagerParams& params,
        PathPlanner* planner,
        rclcpp::Logger logger
    );

    TaskUpdateOutput update(const TaskUpdateInput& input);

    [[nodiscard]] AnnotatedPath::ConstPtr active_path() const { return active_path_; }
    [[nodiscard]] const std::optional<Eigen::Vector2d>& hold_goal() const { return hold_goal_; }
    [[nodiscard]] std::optional<uint64_t> current_goal_id() const {
        return current_goal_ ? std::optional<uint64_t>(current_goal_->id) : std::nullopt;
    }

    [[nodiscard]] TaskDiagnostics diagnostics(std::chrono::steady_clock::time_point stamp) const;

private:
    [[nodiscard]] bool goals_equivalent(const Goal& a, const Goal& b) const;
    [[nodiscard]] TaskCommandView command_view() const;

    bool ingest_goal(const std::optional<Goal>& incoming, bool execution_replaceable);
    void apply_navigation_access(NavigationAccess access);
    void apply_deferred_goal_preemption(bool execution_replaceable);
    void ingest_executor_replan_event(bool event);
    void ingest_goal_reached(bool goal_reached, const AnnotatedPath::ConstPtr& reached_path);
    void poll_planner_result(bool result_acceptable);
    bool maybe_submit_plan(bool planning_allowed, const PlanRequestSnapshot& snapshot, std::chrono::steady_clock::time_point stamp);
    void monitor_route(const std::optional<RouteMonitorInput>& input);
    void on_route_invalid(ReplanReason reason);
    void begin_new_plan_generation();

    TaskManagerParams params_;
    PathPlanner* planner_;
    rclcpp::Logger logger_;

    std::optional<Goal> current_goal_;
    AnnotatedPath::ConstPtr active_path_;
    std::optional<Eigen::Vector2d> hold_goal_;

    bool needs_plan_ = false;
    uint64_t plan_generation_ = 0;
    NavigationAccess navigation_access_ = NavigationAccess::AVAILABLE;
    bool executor_replan_pending_ = false;

    // 失败冷却元数据
    bool in_cooldown_ = false;
    std::chrono::steady_clock::time_point cooldown_start_;
    uint64_t last_failure_generation_ = 0;
    std::optional<std::string> last_failure_reason_;

    uint64_t next_goal_id_ = 1;
    ReplanReason last_replan_reason_ = ReplanReason::NONE;
    PlannerResultState planner_last_result_ = PlannerResultState::NONE;
    uint64_t replan_count_ = 0;

    // 缓存最新一次有效规划结果的前端调试路径（失败结果同样保留）。
    std::vector<Eigen::Vector2d> last_debug_spatial_path_;
    std::vector<Eigen::Vector2d> last_debug_smoothed_spatial_path_;
    std::vector<Eigen::Vector2d> last_debug_kino_path_;
};

} // namespace nav_executor
