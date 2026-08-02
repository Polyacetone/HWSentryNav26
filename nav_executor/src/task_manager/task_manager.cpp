#include <nav_executor/task_manager/task_manager.hpp>

#include <algorithm>

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
    apply_navigation_access(input.navigation_access);
    const bool navigation_available = input.navigation_access == NavigationAccess::AVAILABLE;

    const bool goal_changed = ingest_goal(input.incoming_goal, navigation_available);
    apply_deferred_goal_preemption(navigation_available);
    ingest_executor_replan_event(input.feedback.executor_replan_event && !goal_changed);
    poll_planner_result(navigation_available);
    monitor_route(input.route_monitor);
    ingest_goal_reached(input.feedback.goal_reached, input.feedback.goal_reached_path);
    maybe_submit_plan(navigation_available, input.plan_snapshot, input.stamp);

    return {
        .command = command_view(),
        .diagnostics = diagnostics(input.stamp),
    };
}

void TaskManager::apply_navigation_access(const NavigationAccess access) {
    if (access == NavigationAccess::REVOKED) {
        // REVOKED 是持续权限。只在进入该生命周期时推进 generation，随后每帧
        // 重申“无执行产物”不变量，避免高优先级小陀螺期间重复失效。
        if (navigation_access_ != NavigationAccess::REVOKED && current_goal_) {
            begin_new_plan_generation();
            last_replan_reason_ = ReplanReason::HIGH_PRIORITY_SPIN_PREEMPTION;
            ++replan_count_;
            RCLCPP_INFO(
                logger_,
                "High-priority spin revoked navigation execution; preserving goal #%lu for replanning",
                static_cast<unsigned long>(current_goal_->id)
            );
        }
        active_path_.reset();
        hold_goal_.reset();
        executor_replan_pending_ = false;
    } else if (navigation_access_ == NavigationAccess::REVOKED
        && access == NavigationAccess::AVAILABLE && current_goal_) {
        RCLCPP_INFO(
            logger_,
            "Navigation access restored; current goal #%lu is eligible for replanning",
            static_cast<unsigned long>(current_goal_->id)
        );
    }

    navigation_access_ = access;
}

void TaskManager::apply_deferred_goal_preemption(const bool execution_replaceable) {
    if (!execution_replaceable || !current_goal_ || !active_path_) return;
    if (active_path_->goal_id == current_goal_->id) return;

    active_path_.reset();
    RCLCPP_INFO(
        logger_,
        "Deferred goal preemption released; invalidated stale active path before replanning goal #%lu",
        static_cast<unsigned long>(current_goal_->id)
    );
}

TaskCommandView TaskManager::command_view() const {
    return {
        .active_path = active_path(),
        .hold_goal = hold_goal_,
    };
}

void TaskManager::begin_new_plan_generation() {
    ++plan_generation_;
    needs_plan_ = true;
}

// 新语义目标立即替换已提交任务。当前阶段可抢占时，旧路径同时失去执行权，
// Executor 在没有候选路径的间隔内减速；新路径就绪后可直接接管，无需等待停稳。
// COMMITTED 等不可抢占阶段只更新 latest-wins 目标，保留当前路径至阶段释放。
bool TaskManager::ingest_goal(const std::optional<Goal>& incoming, const bool execution_replaceable) {
    if (!incoming) return false;

    if (current_goal_ && goals_equivalent(*current_goal_, *incoming)) {
        return false;
    }

    // 语义不同的 new goal（latest-wins）。
    Goal goal = *incoming;
    goal.id = next_goal_id_++;
    current_goal_ = goal;

    hold_goal_.reset();       // 新 goal 清空 hold_goal
    if (execution_replaceable && active_path_) {
        active_path_.reset();
        RCLCPP_INFO(logger_, "New goal invalidated active path; braking until replacement is ready");
    }
    begin_new_plan_generation(); // 同时淘汰所有旧目标/旧路径周期的在途结果
    in_cooldown_ = false;         // 新 goal 立即打断旧冷却
    last_failure_reason_.reset();
    last_failure_generation_ = 0;
    last_replan_reason_ = ReplanReason::NONE;
    planner_last_result_ = PlannerResultState::NONE;
    replan_count_ = 0;
    executor_replan_pending_ = false;

    RCLCPP_INFO(
        logger_, "New goal #%lu (%.2f, %.2f) fixed=%d [%s]",
        static_cast<unsigned long>(goal.id), goal.position_map.x(), goal.position_map.y(),
        goal.fixed, execution_replaceable ? "replaceable" : "execution-locked"
    );
    return true;
}

// 恢复重规划保留任务语义，但可能改变执行形式。
void TaskManager::ingest_executor_replan_event(const bool event) {
    if (event) executor_replan_pending_ = true;
    if (!executor_replan_pending_) return;

    if (!current_goal_) {
        RCLCPP_DEBUG(logger_, "executor_replan_event swallowed (no goal)");
        executor_replan_pending_ = false;
        return;
    }

    if (navigation_access_ == NavigationAccess::LOCK_CURRENT) {
        if (event) {
            RCLCPP_DEBUG(logger_, "executor_replan_event deferred by locked navigation execution");
        }
        return;
    }

    // REVOKED 已经淘汰了进入抢占前的 generation，且冻结期间不会提交新规划。
    if (navigation_access_ == NavigationAccess::REVOKED) {
        active_path_.reset();
        hold_goal_.reset();
        needs_plan_ = true;
        executor_replan_pending_ = false;
        RCLCPP_DEBUG(logger_, "executor_replan_event covered by revoked navigation access");
        return;
    }

    // 当前执行形式已不可信 → 清掉 path/hold，回到 planner 重新决定 FOLLOW 还是 FIXED。
    active_path_.reset();
    hold_goal_.reset();
    begin_new_plan_generation();
    last_replan_reason_ = ReplanReason::EXECUTOR_REPLAN_EVENT;
    ++replan_count_;
    executor_replan_pending_ = false;
    RCLCPP_INFO(logger_, "executor_replan_event → drop path/hold, replan current goal");
}

// 只接纳仍匹配已提交目标和当前运动阶段的规划结果。
void TaskManager::poll_planner_result(const bool result_acceptable) {
    auto result = planner_->try_take_result();
    if (!result) return;

    if (!current_goal_ || result->goal_id != current_goal_->id) {
        RCLCPP_DEBUG(logger_, "Discard plan result: goal changed (result goal_id mismatch)");
        return;
    }

    if (result->plan_generation != plan_generation_) {
        RCLCPP_DEBUG(
            logger_,
            "Discard plan result: stale generation (result=%lu current=%lu)",
            static_cast<unsigned long>(result->plan_generation),
            static_cast<unsigned long>(plan_generation_)
        );
        return;
    }

    if (!result_acceptable) {
        RCLCPP_DEBUG(logger_, "Discard plan result: navigation access is not available");
        return;
    }

    last_debug_spatial_path_ = std::move(result->debug_spatial_path);
    last_debug_smoothed_spatial_path_ = std::move(
        result->debug_smoothed_spatial_path
    );
    last_debug_kino_path_ = std::move(result->debug_kino_path);

    switch (result->kind) {
        case PlanResult::Kind::PATH:
            planner_last_result_ = PlannerResultState::PATH_ACCEPTED;
            active_path_ = result->path;
            hold_goal_.reset();
            needs_plan_ = false;
            in_cooldown_ = false;
            last_failure_reason_.reset();
            RCLCPP_INFO(logger_, "Accepted new path for goal #%lu", static_cast<unsigned long>(result->goal_id));
            for (const std::string& warning : result->warnings) {
                RCLCPP_WARN(
                    logger_, "Accepted path for goal #%lu with warning: %s",
                    static_cast<unsigned long>(result->goal_id), warning.c_str()
                );
            }
            break;

        case PlanResult::Kind::COMPLETE_NO_PLAN_NEEDED:
            planner_last_result_ = PlannerResultState::COMPLETE;
            active_path_.reset();
            hold_goal_.reset();
            needs_plan_ = false;
            in_cooldown_ = false;
            last_failure_reason_.reset();
            current_goal_.reset();
            RCLCPP_INFO(logger_, "Goal complete (no plan needed)");
            break;

        case PlanResult::Kind::USE_AS_FIXED_GOAL:
            planner_last_result_ = PlannerResultState::FIXED_GOAL;
            active_path_.reset();
            hold_goal_ = result->goal_pos;
            needs_plan_ = false;
            in_cooldown_ = false;
            last_failure_reason_.reset();
            RCLCPP_INFO(logger_, "Goal near → enter FIXED hold directly");
            break;

        case PlanResult::Kind::FAILED:
            planner_last_result_ = PlannerResultState::FAILED;
            active_path_.reset();
            hold_goal_.reset();
            needs_plan_ = false;
            in_cooldown_ = true;
            cooldown_start_ = std::chrono::steady_clock::now();
            if (last_failure_generation_ != result->plan_generation
                || !last_failure_reason_
                || *last_failure_reason_ != result->failure_reason) {
                last_failure_generation_ = result->plan_generation;
                last_failure_reason_ = result->failure_reason;
                RCLCPP_WARN(
                    logger_, "Plan failed for goal #%lu: %s; cooldown retry",
                    static_cast<unsigned long>(result->goal_id),
                    result->failure_reason.empty()
                        ? "unspecified failure" : result->failure_reason.c_str()
                );
            } else {
                RCLCPP_DEBUG(
                    logger_, "Repeated plan failure for goal #%lu: %s",
                    static_cast<unsigned long>(result->goal_id),
                    result->failure_reason.empty()
                        ? "unspecified failure" : result->failure_reason.c_str()
                );
            }
            break;
    }
}

// 统一从此处派发规划，事件处理函数只记录意图。
bool TaskManager::maybe_submit_plan(
    const bool planning_allowed,
    const PlanRequestSnapshot& snapshot,
    const std::chrono::steady_clock::time_point stamp
) {
    // 冷却到期检查（仅对同一 goal 的失败重试生效）。
    if (in_cooldown_ && std::chrono::duration<double>(stamp - cooldown_start_).count() >= params_.plan_cooldown) {
        in_cooldown_ = false;
    }

    // 提交条件（全部满足）：
    if (!current_goal_) return false;
    if (!planning_allowed) return false;
    const bool trigger = needs_plan_ || (!active_path_ && !hold_goal_);
    if (!trigger) return false;
    if (planner_->busy()) return false;
    if (in_cooldown_) return false;

    PlanRequest request;
    request.goal = *current_goal_;
    request.plan_generation = plan_generation_;
    request.current_pos_map = snapshot.current_pos_map;
    request.current_yaw = snapshot.current_yaw;
    request.current_velocity = snapshot.current_velocity;
    request.obstacles = snapshot.obstacles;
    request.direction_map = snapshot.direction_map;
    request.terrain_constraints = snapshot.terrain_constraints;
    request.performance = snapshot.performance;

    planner_->submit(request);
    RCLCPP_DEBUG(
        logger_, "Plan #%lu queued (generation=%lu)",
        static_cast<unsigned long>(current_goal_->id),
        static_cast<unsigned long>(plan_generation_)
    );
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
    executor_replan_pending_ = false;
    begin_new_plan_generation();
    last_replan_reason_ = reason;
    ++replan_count_;
    RCLCPP_INFO(logger_, "Path invalid (%s) → drop path, replan", replan_reason_str(reason));
}

// 仅当原始不可变路径包仍在执行时，完成事件才有效。
void TaskManager::ingest_goal_reached(const bool goal_reached, const AnnotatedPath::ConstPtr& reached_path) {
    if (!goal_reached) return;

    if (!reached_path || reached_path != active_path_) {
        RCLCPP_DEBUG(logger_, "goal_reached on superseded path → ignore event");
        return;
    }

    // 当前 active path 已被新语义目标取代，但尚未获得新 path：只清掉这条旧 path。
    const bool valid = current_goal_ && reached_path->goal_id == current_goal_->id;

    if (!valid) {
        // 旧路径按旧任务语义自然完结：只清 path，不动 current_goal_。
        active_path_.reset();
        RCLCPP_DEBUG(logger_, "goal_reached on stale path → drop path only, keep current goal");
        return;
    }

    if (current_goal_->fixed) {
        // fixed goal：清 path，设置 hold_goal，进入持续保持。
        hold_goal_ = reached_path->goal_pos;
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

TaskDiagnostics TaskManager::diagnostics(const std::chrono::steady_clock::time_point stamp) const {
    TaskDiagnostics d;
    if (current_goal_) {
        d.goal_id = current_goal_->id;
        d.goal_position = current_goal_->position_map;
        d.goal_fixed = current_goal_->fixed;
    }
    d.active_path_goal_id = active_path_ ? active_path_->goal_id : 0;
    d.has_hold_goal = hold_goal_.has_value();
    if (hold_goal_) d.hold_goal_position = *hold_goal_;
    d.plan_generation = plan_generation_;
    d.needs_plan = needs_plan_;
    d.planner_state = planner_->busy() ? PlannerState::PLANNING : (in_cooldown_ ? PlannerState::COOLDOWN : PlannerState::IDLE);
    if (in_cooldown_) {
        d.planner_cooldown_remaining = std::max(
            0.0,
            params_.plan_cooldown - std::chrono::duration<double>(stamp - cooldown_start_).count()
        );
    }
    d.planner_last_result = planner_last_result_;
    d.planner_last_failure_reason = last_failure_reason_.value_or("");
    d.last_replan_reason = last_replan_reason_;
    d.replan_count = replan_count_;
    d.debug_spatial_path = last_debug_spatial_path_;
    d.debug_smoothed_spatial_path = last_debug_smoothed_spatial_path_;
    d.debug_kino_path = last_debug_kino_path_;
    return d;
}

} // namespace nav_executor
