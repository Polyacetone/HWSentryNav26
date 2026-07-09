#pragma once

#include <cstdint>
#include <optional>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/path/annotated_path.hpp>
#include <nav_executor/path/route_monitor.hpp>
#include <nav_executor/planner/path_planner.hpp>

namespace nav_executor {

struct TaskExecutorParams {
    double goal_equivalence_distance; // goal-to-goal 去重阈值 (m)
    double plan_cooldown;             // 同一 goal 失败重试冷却 (s)
};

// 顶层每周期向底层暴露的输入。
struct ExecutorInterface {
    const AnnotatedPath* active_path = nullptr;
    std::optional<Eigen::Vector2d> hold_goal;
};
enum class PlannerState : uint8_t { IDLE, PLANNING, COOLDOWN };

struct TaskDiagnostics {
    bool has_goal = false;
    bool has_path = false;
    bool has_hold_goal = false;
    PlannerState planner_state = PlannerState::IDLE;
    ReplanReason last_replan_reason = ReplanReason::NONE;

    // 调试路径（planner 各阶段），仅在 enable_debug 时非空
    std::vector<Eigen::Vector2d> debug_rough_path;
    std::vector<Eigen::Vector2d> debug_warmup_path;
    std::vector<Eigen::Vector2d> debug_optimized_path;
};

// 任务层：拥有 goal 生命周期、planner 调度、active_path/hold_goal 输出、
// FOLLOW 阶段 path 失效监视。
class TaskExecutor {
public:
    TaskExecutor(
        const TaskExecutorParams& params,
        PathPlanner* planner,
        rclcpp::Logger logger
    );

    // ── 主循环步骤，由 node 顺序调用 ──

    // step 2：处理新 goal 输入（含等价判定），更新 current_goal / 清 hold_goal / 设 needs_plan。
    void ingest_goal(const std::optional<Goal>& incoming, bool preemptible);

    // step 3：消费 executor_replan_event（若无 hold_goal 则设 needs_plan，否则吞掉）。
    void ingest_executor_replan_event(bool event);

    // step 4：轮询 planner 结果并按接纳规则应用。
    void poll_planner_result(bool preemptible);

    // step 5：统一规划调度——满足条件则提交 PlanRequest。
    //         请求所需的位姿/地图快照由 node 通过回调填充。
    struct PlanRequestSnapshot {
        Eigen::Vector2d current_pos_map;
        double current_yaw;
        double current_velocity;
        CostMap::ConstPtr global_cost_map;
        CostMap::ConstPtr merged_cost_map;
        DirectionMap::ConstPtr direction_map;
    };
    // 返回 true 表示本周期提交了规划请求（node 需已填 snapshot）。
    bool maybe_submit_plan(bool preemptible, const PlanRequestSnapshot& snapshot, std::chrono::steady_clock::time_point stamp);

    // step 7：RouteMonitor 判 path invalid → 清 active_path、设 needs_plan。
    void on_route_invalid(ReplanReason reason);

    // step 9：消费 goal_reached。
    void ingest_goal_reached(bool goal_reached);

    // ── 顶层暴露给底层的接口 ──
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

    TaskExecutorParams params_;
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
    std::vector<Eigen::Vector2d> last_debug_optimized_path_;
};

} // namespace nav_executor
