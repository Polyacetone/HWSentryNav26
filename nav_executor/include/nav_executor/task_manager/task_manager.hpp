#pragma once

#include <cstdint>
#include <optional>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/common/annotated_path.hpp>
#include <nav_executor/task_manager/route_monitor.hpp>
#include <nav_executor/path_executor/state_machine.hpp>
#include <nav_executor/path_planner/path_planner.hpp>

namespace nav_executor {

struct TaskManagerParams {
    double goal_equivalence_distance; // goal-to-goal 去重阈值 (m)
    double plan_cooldown;             // 同一 goal 失败重试冷却 (s)
};

enum class PlannerState : uint8_t { IDLE, PLANNING, COOLDOWN };

struct TaskDiagnostics {
    bool has_goal = false;
    bool has_path = false;
    bool has_hold_goal = false;
    PlannerState planner_state = PlannerState::IDLE;
    ReplanReason last_replan_reason = ReplanReason::NONE;
    std::vector<Eigen::Vector2d> debug_rough_path;
    std::vector<Eigen::Vector2d> debug_warmup_path;
    std::vector<Eigen::Vector2d> global_path;
};

struct PlanRequestSnapshot {
    Eigen::Vector2d current_pos_map = Eigen::Vector2d::Zero();
    double current_yaw = 0.0;
    double current_velocity = 0.0;
    CostMap::ConstPtr global_cost_map;
    CostMap::ConstPtr merged_cost_map;
    DirectionMap::ConstPtr direction_map;
};

struct MotionFeedback {
    bool goal_reached = false;
    bool executor_replan_event = false;
    bool mpc_lethal = false;
    const AnnotatedPath* lethal_path = nullptr;
    double route_u = 0.0;
    MotionState motion_state = MotionState::IDLE;
    bool preemptible = true;
};

struct TaskUpdateInput {
    std::optional<Goal> incoming_goal;
    MotionFeedback feedback;
    PlanRequestSnapshot plan_snapshot;
    std::optional<RouteMonitorInput> route_monitor;
    std::chrono::steady_clock::time_point stamp;
};

struct TaskCommandView {
    const AnnotatedPath* active_path = nullptr;
    std::optional<Eigen::Vector2d> hold_goal;
};

struct TaskUpdateOutput {
    TaskCommandView command;
    TaskDiagnostics diagnostics;
};

// Owns task commitment: current goal, active route, hold target, and planner dispatch.
class TaskManager {
public:
    TaskManager(
        const TaskManagerParams& params,
        PathPlanner* planner,
        rclcpp::Logger logger
    );

    TaskUpdateOutput update(const TaskUpdateInput& input);

    [[nodiscard]] const AnnotatedPath* active_path() const {
        return active_path_ ? active_path_.get() : nullptr;
    }
    [[nodiscard]] const std::optional<Eigen::Vector2d>& hold_goal() const { return hold_goal_; }
    [[nodiscard]] std::optional<uint64_t> current_goal_id() const {
        return current_goal_ ? std::optional<uint64_t>(current_goal_->id) : std::nullopt;
    }

    [[nodiscard]] TaskDiagnostics diagnostics() const;

private:
    [[nodiscard]] bool goals_equivalent(const Goal& a, const Goal& b) const;
    [[nodiscard]] TaskCommandView command_view() const;

    void ingest_goal(const std::optional<Goal>& incoming, bool preemptible);
    void ingest_executor_replan_event(bool event);
    void ingest_goal_reached(bool goal_reached);
    void poll_planner_result(bool preemptible);
    bool maybe_submit_plan(bool preemptible, const PlanRequestSnapshot& snapshot, std::chrono::steady_clock::time_point stamp);
    void monitor_route(const std::optional<RouteMonitorInput>& input);
    void on_route_invalid(ReplanReason reason);

    TaskManagerParams params_;
    PathPlanner* planner_;
    rclcpp::Logger logger_;

    std::optional<Goal> current_goal_;
    AnnotatedPath::ConstPtr active_path_;
    std::optional<Eigen::Vector2d> hold_goal_;

    bool needs_plan_ = false;

    // 失败冷却元数据
    bool in_cooldown_ = false;
    std::chrono::steady_clock::time_point cooldown_start_;

    uint64_t next_goal_id_ = 1;
    ReplanReason last_replan_reason_ = ReplanReason::NONE;

    // 缓存最新一次成功规划的调试路径（供 node 发布用）
    std::vector<Eigen::Vector2d> last_debug_rough_path_;
    std::vector<Eigen::Vector2d> last_debug_warmup_path_;
    std::vector<Eigen::Vector2d> last_global_path_;
};

} // namespace nav_executor
