#include <nav_executor/task_manager/task_manager.hpp>

#include <rclcpp/logging.hpp>

namespace nav_executor {

TaskManager::TaskManager(
    const TaskManagerParams& params,
    PathPlanner* planner,
    rclcpp::Logger logger
) : params_(params), planner_(planner), logger_(logger) {}

bool TaskManager::goals_equivalent(const Goal& a, const Goal& b) const {
    return a.fixed == b.fixed && (a.position_map - b.position_map).norm() < params_.goal_equivalence_distance;
}

TaskUpdateOutput TaskManager::update(const TaskUpdateInput& input) {
    ingest_goal(input.incoming_goal, input.feedback.preemptible);
    ingest_executor_replan_event(input.feedback.executor_replan_event);
    poll_planner_result(input.feedback.preemptible);
    maybe_submit_plan(input.feedback.preemptible, input.plan_snapshot, input.stamp);
    monitor_route(input.route_monitor);
    ingest_goal_reached(input.feedback.goal_reached);

    return {
        .command = command_view(),
        .diagnostics = diagnostics(),
    };
}

TaskCommandView TaskManager::command_view() const {
    return {
        .active_path = active_path(),
        .hold_goal = hold_goal_,
    };
}

// New semantic goals replace the committed task, but the old path may continue until a new route is ready.
void TaskManager::ingest_goal(const std::optional<Goal>& incoming, const bool preemptible) {
    if (!incoming) return;

    if (current_goal_ && goals_equivalent(*current_goal_, *incoming)) {
        return;
    }

    // 语义不同的 new goal（latest-wins）。
    Goal goal = *incoming;
    goal.id = next_goal_id_++;
    current_goal_ = goal;

    hold_goal_.reset();       // 新 goal 清空 hold_goal
    needs_plan_ = true;       // 标记需要重新规划
    in_cooldown_ = false;     // 新 goal 立即打断旧冷却

    RCLCPP_INFO(
        logger_, "New goal #%lu (%.2f, %.2f) fixed=%d [%s]",
        static_cast<unsigned long>(goal.id), goal.position_map.x(), goal.position_map.y(),
        goal.fixed, preemptible ? "preemptible" : "non-preemptible"
    );
}

// Recovery replan events keep the semantic goal but may change the execution form.
void TaskManager::ingest_executor_replan_event(const bool event) {
    if (!event) return;

    if (!current_goal_) {
        RCLCPP_DEBUG(logger_, "executor_replan_event swallowed (no goal)");
        return;
    }

    // 当前执行形式已不可信 → 清掉 path/hold，回到 planner 重新决定 FOLLOW 还是 FIXED。
    active_path_.reset();
    hold_goal_.reset();
    needs_plan_ = true;
    last_replan_reason_ = ReplanReason::EXECUTOR_REPLAN_EVENT;
    RCLCPP_INFO(logger_, "executor_replan_event → drop path/hold, replan current goal");
}

// Planner results are accepted only when they still match the committed goal and current motion phase.
void TaskManager::poll_planner_result(const bool preemptible) {
    auto result = planner_->try_take_result();
    if (!result) return;

    if (!current_goal_ || result->goal_id != current_goal_->id) {
        RCLCPP_INFO(logger_, "Discard plan result: goal changed (result goal_id mismatch)");
        return;
    }

    if (!preemptible) {
        RCLCPP_INFO(logger_, "Discard plan result: system became non-preemptible");
        return;
    }

    switch (result->kind) {
        case PlanResult::Kind::PATH:
            active_path_ = result->path;
            hold_goal_.reset();
            needs_plan_ = false;
            in_cooldown_ = false;
            // 缓存调试路径（即便 enable_debug=false 也是空 vector，无开销）
            last_debug_rough_path_ = std::move(result->debug_rough_path);
            last_debug_warmup_path_ = std::move(result->debug_warmup_path);
            last_global_path_ = std::move(result->global_path);
            RCLCPP_INFO(logger_, "Accepted new path for goal #%lu", static_cast<unsigned long>(result->goal_id));
            break;

        case PlanResult::Kind::COMPLETE_NO_PLAN_NEEDED:
            active_path_.reset();
            hold_goal_.reset();
            needs_plan_ = false;
            in_cooldown_ = false;
            current_goal_.reset();
            RCLCPP_INFO(logger_, "Goal complete (no plan needed)");
            break;

        case PlanResult::Kind::USE_AS_FIXED_GOAL:
            active_path_.reset();
            hold_goal_ = result->goal_pos;
            needs_plan_ = false;
            in_cooldown_ = false;
            RCLCPP_INFO(logger_, "Goal near → enter FIXED hold directly");
            break;

        case PlanResult::Kind::FAILED:
            active_path_.reset();
            hold_goal_.reset();
            needs_plan_ = false;
            in_cooldown_ = true;
            cooldown_start_ = std::chrono::steady_clock::now();
            RCLCPP_WARN(logger_, "Plan failed → cooldown retry");
            break;
    }
}

// Planning is dispatched from one gate so event handlers only mark intent.
bool TaskManager::maybe_submit_plan(
    const bool preemptible,
    const PlanRequestSnapshot& snapshot,
    const std::chrono::steady_clock::time_point stamp
) {
    // 冷却到期检查（仅对同一 goal 的失败重试生效）。
    if (in_cooldown_ && std::chrono::duration<double>(stamp - cooldown_start_).count() >= params_.plan_cooldown) {
        in_cooldown_ = false;
    }

    // 提交条件（全部满足）：
    if (!current_goal_) return false;
    if (!preemptible) return false;
    const bool trigger = needs_plan_ || (!active_path_ && !hold_goal_);
    if (!trigger) return false;
    if (planner_->busy()) return false;
    if (in_cooldown_) return false;

    PlanRequest request;
    request.goal = *current_goal_;
    request.current_pos_map = snapshot.current_pos_map;
    request.current_yaw = snapshot.current_yaw;
    request.current_velocity = snapshot.current_velocity;
    request.global_cost_map = snapshot.global_cost_map;
    request.merged_cost_map = snapshot.merged_cost_map;
    request.direction_map = snapshot.direction_map;
    request.terrain_constraints = snapshot.terrain_constraints;
    request.performance = snapshot.performance;

    planner_->submit(request);
    RCLCPP_DEBUG(logger_, "Submitted plan request for goal #%lu", static_cast<unsigned long>(current_goal_->id));
    return true;
}

void TaskManager::monitor_route(const std::optional<RouteMonitorInput>& input) {
    if (!input || !input->active_path || input->active_path != active_path_) return;

    const RouteMonitorReport report = run_route_monitor(*input, logger_);
    if (report.needs_replan) {
        on_route_invalid(report.reason);
    }
}

void TaskManager::on_route_invalid(const ReplanReason reason) {
    active_path_.reset();
    needs_plan_ = true;
    last_replan_reason_ = reason;
    RCLCPP_INFO(logger_, "Path invalid (%s) → drop path, replan", replan_reason_str(reason));
}

// A stale path may finish after a newer goal has already been committed; only the path is cleared.
void TaskManager::ingest_goal_reached(const bool goal_reached) {
    if (!goal_reached) return;

    // 仅当 active_path.goal_id == current_goal.id 时才视为有效。
    const bool valid = active_path_ && current_goal_ && active_path_->goal_id == current_goal_->id;

    if (!valid) {
        // 旧路径按旧任务语义自然完结：只清 path，不动 current_goal_。
        active_path_.reset();
        RCLCPP_INFO(logger_, "goal_reached on stale path → drop path only, keep current goal");
        return;
    }

    if (current_goal_->fixed) {
        // fixed goal：清 path，设置 hold_goal，进入持续保持。
        hold_goal_ = active_path_->goal_pos;
        active_path_.reset();
        needs_plan_ = false;
        RCLCPP_INFO(logger_, "fixed goal reached → enter FIXED hold");
    } else {
        // 普通 goal：任务完成，清 goal / path / needs_plan。
        active_path_.reset();
        current_goal_.reset();
        needs_plan_ = false;
        RCLCPP_INFO(logger_, "goal reached → task complete");
    }
}

// ═══════════════════ 诊断 ═══════════════════════════════════════

TaskDiagnostics TaskManager::diagnostics() const {
    TaskDiagnostics d;
    d.has_goal = current_goal_.has_value();
    d.has_path = static_cast<bool>(active_path_);
    d.has_hold_goal = hold_goal_.has_value();
    d.planner_state = planner_->busy() ? PlannerState::PLANNING : (in_cooldown_ ? PlannerState::COOLDOWN : PlannerState::IDLE);
    d.last_replan_reason = last_replan_reason_;
    d.debug_rough_path = last_debug_rough_path_;
    d.debug_warmup_path = last_debug_warmup_path_;
    d.global_path = last_global_path_;
    return d;
}

} // namespace nav_executor
