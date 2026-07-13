#include <nav_executor/path_executor/state_machine.hpp>

#include <Eigen/Core>
#include <rclcpp/logging.hpp>

namespace nav_executor {

StateMachine::StateMachine(const FsmParams& params, rclcpp::Logger logger)
    : params_(params), logger_(logger) {}

FsmOutput StateMachine::update(const FsmInput& input) {
    switch (active_state_) {
        case MotionState::IDLE: return on_idle(input);
        case MotionState::FIXED: return on_fixed(input);
        case MotionState::FOLLOW: return on_follow(input);
        case MotionState::STEPPING: return on_stepping(input);
        case MotionState::SPIN: return on_spin(input);
        case MotionState::STOPPING: return on_stopping(input);
        case MotionState::STUCK_REVERSE: return on_stuck_reverse(input);
        case MotionState::HAZARD_RECOVERY: return on_hazard_recovery(input);
        default: return { .state = active_state_ };
    }
}

// ═══════════════════════════ 辅助 ═══════════════════════════

FsmOutput StateMachine::transition_to(const MotionState next) {
    if (active_state_ == MotionState::STUCK_REVERSE && next != MotionState::STUCK_REVERSE) {
        reverse_entry_initialized_ = false;
    }
    active_state_ = next;
    return { .state = next };
}

StateMachine::Terminal StateMachine::terminal_target(const FsmInput& in) const {
    const bool should_spin = in.spin_requested
        && (in.spin_high_priority || (!in.has_path && !in.has_hold_goal));
    if (should_spin) return Terminal::SPIN;
    if (in.has_path) return Terminal::FOLLOW;
    if (in.has_hold_goal) return Terminal::FIXED;
    return Terminal::IDLE;
}

bool StateMachine::stopping_ready(const FsmInput& in, const Terminal target) const {
    const auto& t = params_.transition;
    switch (target) {
        case Terminal::IDLE:
        case Terminal::FIXED:
            return std::abs(in.velocity) < t.to_idle_vel_max && std::abs(in.omega) < t.to_idle_omega_max;
        case Terminal::SPIN:
            return std::abs(in.velocity) < t.follow_to_spin_vel_max && std::abs(in.omega) < t.to_idle_omega_max;
        case Terminal::FOLLOW:
            return true;
    }
    return false;
}

FsmOutput StateMachine::route_to_terminal(const FsmInput& in) {
    switch (terminal_target(in)) {
        case Terminal::SPIN:   return transition_to(MotionState::SPIN);
        case Terminal::FOLLOW: return transition_to(MotionState::FOLLOW);
        case Terminal::FIXED:  return transition_to(MotionState::FIXED);
        case Terminal::IDLE:   return transition_to(MotionState::IDLE);
    }
    return transition_to(MotionState::IDLE);
}

FsmOutput StateMachine::finish_recovery_chain(const FsmInput& in) {
    FsmOutput out = route_to_terminal(in);
    if (replan_after_recovery_) {
        replan_after_recovery_ = false;
        out.executor_replan_event = true;
    }
    return out;
}

bool StateMachine::should_start_resume_hazard_recovery(const FsmInput& in) const {
    if (!in.resumed_from_stopped || !in.is_hazard_now) {
        return false;
    }

    switch (active_state_) {
        case MotionState::FOLLOW:
        case MotionState::STEPPING:
        case MotionState::FIXED:
            return true;
        default:
            return false;
    }
}

FsmOutput StateMachine::exit_reverse(const FsmInput& in, const double displacement, const double mature_elapsed) {
    if (in.is_hazard_now) {
        RCLCPP_WARN(
            logger_,
            "STUCK_REVERSE: displaced %.2f m (mature=%.1f s), current position hazard, entering HAZARD_RECOVERY",
            displacement, mature_elapsed
        );
        return transition_to(MotionState::HAZARD_RECOVERY);
    }

    RCLCPP_INFO(
        logger_,
        "STUCK_REVERSE: displaced %.2f m (mature=%.1f s), current position safe, skipping HAZARD_RECOVERY",
        displacement, mature_elapsed
    );
    return finish_recovery_chain(in);
}

// ═══════════════════════════ IDLE ═══════════════════════════

FsmOutput StateMachine::on_idle(const FsmInput& in) {
    if (in.is_hazard_now) {
        replan_after_recovery_ = false;
        RCLCPP_WARN(logger_, "FSM -> HAZARD_RECOVERY (is hazard)");
        return transition_to(MotionState::HAZARD_RECOVERY);
    }
    if (in.has_path) {
        RCLCPP_INFO(logger_, "FSM -> FOLLOW (has path)");
        return transition_to(MotionState::FOLLOW);
    }
    if (in.has_hold_goal) {
        RCLCPP_INFO(logger_, "FSM -> FIXED (has hold goal)");
        return transition_to(MotionState::FIXED);
    }
    if (in.spin_requested) {
        RCLCPP_INFO(logger_, "FSM -> SPIN (spin requested)");
        return transition_to(MotionState::SPIN);
    }
    return { .state = MotionState::IDLE };
}

// ═══════════════════════════ FIXED ══════════════════════════

FsmOutput StateMachine::on_fixed(const FsmInput& in) {
    if (should_start_resume_hazard_recovery(in)) {
        replan_after_recovery_ = true;
        RCLCPP_WARN(logger_, "FSM -> HAZARD_RECOVERY (resumed from stopped into hazard during FIXED)");
        return transition_to(MotionState::HAZARD_RECOVERY);
    }

    if (in.is_stuck) {
        replan_after_recovery_ = true;
        pending_reverse_start_time_ = in.stamp;
        pending_reverse_start_pos_ = in.chassis_pos_map;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE (is stuck)");
        return transition_to(MotionState::STUCK_REVERSE);
    }
    if (in.has_path) {
        RCLCPP_INFO(logger_, "FSM -> FOLLOW (has path)");
        return transition_to(MotionState::FOLLOW);
    }
    if (in.spin_requested && in.spin_high_priority) {
        stopping_start_time_ = in.stamp;
        RCLCPP_INFO(logger_, "FSM -> STOPPING (spin requested)");
        return transition_to(MotionState::STOPPING);
    }
    // hold_goal 消失（任务被清空）→ 平滑退出
    if (!in.has_hold_goal) {
        stopping_start_time_ = in.stamp;
        RCLCPP_INFO(logger_, "FSM -> STOPPING (no hold goal)");
        return transition_to(MotionState::STOPPING);
    }
    return { .state = MotionState::FIXED };
}

// ═══════════════════════════ FOLLOW ═════════════════════════

FsmOutput StateMachine::on_follow(const FsmInput& in) {
    if (should_start_resume_hazard_recovery(in)) {
        replan_after_recovery_ = true;
        RCLCPP_WARN(logger_, "FSM -> HAZARD_RECOVERY (resumed from stopped into hazard during FOLLOW)");
        return transition_to(MotionState::HAZARD_RECOVERY);
    }

    // 台阶不可抢占区最高优先，且不被 stuck/no_progress 打断前置处理。
    if (in.step_nonpreemptible) {
        RCLCPP_INFO(logger_, "FSM -> STEPPING (step nonpreemptible)");
        return transition_to(MotionState::STEPPING);
    }

    // stuck-like（follow no-progress 或卡住）统一走脱困链，恢复后 one-shot replan。
    if (in.no_progress_detected || in.is_stuck) {
        replan_after_recovery_ = true;
        pending_reverse_start_time_ = in.stamp;
        pending_reverse_start_pos_ = in.chassis_pos_map;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE (stuck-like)");
        return transition_to(MotionState::STUCK_REVERSE);
    }

    // path 被顶层清空（RouteMonitor 判 invalid 或消费）→ 平滑停止后回落。
    if (!in.has_path) {
        stopping_start_time_ = in.stamp;
        RCLCPP_INFO(logger_, "FSM -> STOPPING (no path)");
        return transition_to(MotionState::STOPPING);
    }

    if (in.spin_requested && in.spin_high_priority) {
        stopping_start_time_ = in.stamp;
        RCLCPP_INFO(logger_, "FSM -> STOPPING (spin requested)");
        return transition_to(MotionState::STOPPING);
    }

    if (in.reach_goal) {
        stopping_start_time_ = in.stamp;
        RCLCPP_INFO(logger_, "FSM -> STOPPING (goal reached)");
        FsmOutput out = transition_to(MotionState::STOPPING);
        out.goal_reached = true;
        return out;
    }

    return { .state = MotionState::FOLLOW };
}

// ═══════════════════════════ STEPPING ═══════════════════════

FsmOutput StateMachine::on_stepping(const FsmInput& in) {
    if (should_start_resume_hazard_recovery(in)) {
        replan_after_recovery_ = true;
        RCLCPP_WARN(logger_, "FSM -> HAZARD_RECOVERY (resumed from stopped into hazard during STEPPING)");
        return transition_to(MotionState::HAZARD_RECOVERY);
    }

    if (in.no_progress_detected || in.is_stuck) {
        replan_after_recovery_ = true;
        pending_reverse_start_time_ = in.stamp;
        pending_reverse_start_pos_ = in.chassis_pos_map;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE (stuck-like)");
        return transition_to(MotionState::STUCK_REVERSE);
    }

    if (in.step_nonpreemptible) {
        return { .state = MotionState::STEPPING };
    }

    RCLCPP_INFO(logger_, "FSM -> FOLLOW (step completed)");
    return transition_to(MotionState::FOLLOW);
}

// ═══════════════════════════ SPIN ═══════════════════════════

FsmOutput StateMachine::on_spin(const FsmInput& in) {
    if (in.is_hazard_now) {
        replan_after_recovery_ = false;
        RCLCPP_WARN(logger_, "FSM -> HAZARD_RECOVERY (is hazard)");
        return transition_to(MotionState::HAZARD_RECOVERY);
    }

    const bool keep_spinning = in.spin_requested && (in.spin_high_priority || (!in.has_path && !in.has_hold_goal));

    if (!keep_spinning) {
        stopping_start_time_ = in.stamp;
        RCLCPP_INFO(logger_, "FSM -> STOPPING (spin completed)");
        return transition_to(MotionState::STOPPING);
    }

    return { .state = MotionState::SPIN };
}

// ═══════════════════════════ STOPPING ═══════════════════════

FsmOutput StateMachine::on_stopping(const FsmInput& in) {
    const Terminal target = terminal_target(in);

    // 新任务（有 path）→ 立即跟随，无需等待减速。
    if (target == Terminal::FOLLOW) {
        RCLCPP_INFO(logger_, "FSM -> FOLLOW (new task)");
        return transition_to(MotionState::FOLLOW);
    }

    const bool timeout = std::chrono::duration<double>(in.stamp - stopping_start_time_).count()
        > params_.transition.stopping_timeout;

    if (stopping_ready(in, target) || timeout) {
        switch (target) {
            case Terminal::FIXED: RCLCPP_INFO(logger_, "FSM -> FIXED"); return transition_to(MotionState::FIXED);
            case Terminal::SPIN: RCLCPP_INFO(logger_, "FSM -> SPIN");  return transition_to(MotionState::SPIN);
            case Terminal::IDLE: RCLCPP_INFO(logger_, "FSM -> IDLE");  return transition_to(MotionState::IDLE);
            case Terminal::FOLLOW: return transition_to(MotionState::FOLLOW);
        }
    }

    return { .state = MotionState::STOPPING };
}

// ═══════════════════════════ STUCK_REVERSE ══════════════════

FsmOutput StateMachine::on_stuck_reverse(const FsmInput& in) {
    if (!reverse_entry_initialized_) {
        reverse_entry_initialized_ = true;
        reverse_mature_accumulated_ = 0.0;
        reverse_last_mature_stamp_ = pending_reverse_start_time_;
        reverse_entry_pos_ = pending_reverse_start_pos_;
    }

    if (in.command_blocked) {
        reverse_mature_accumulated_ += std::chrono::duration<double>(in.stamp - reverse_last_mature_stamp_).count();
        reverse_last_mature_stamp_ = in.stamp;
        return { .state = MotionState::STUCK_REVERSE };
    }

    const double mature_elapsed = reverse_mature_accumulated_ + std::chrono::duration<double>(in.stamp - reverse_last_mature_stamp_).count();
    const double displacement = (in.chassis_pos_map - reverse_entry_pos_).norm();

    if (displacement >= params_.stuck.reverse_displacement) {
        reverse_mature_accumulated_ = 0.0;
        return exit_reverse(in, displacement, mature_elapsed);
    }

    if (mature_elapsed >= params_.stuck.reverse_timeout) {
        RCLCPP_ERROR(
            logger_,
            "STUCK_REVERSE TIMEOUT: mature elapsed %.1f s >= %.1f s but displacement only %.2f m < %.2f m — robot may be physically stuck",
            mature_elapsed, params_.stuck.reverse_timeout,
            displacement, params_.stuck.reverse_displacement
        );
        reverse_mature_accumulated_ = 0.0;
        return exit_reverse(in, displacement, mature_elapsed);
    }

    return { .state = MotionState::STUCK_REVERSE };
}

// ═══════════════════════════ HAZARD_RECOVERY ════════════════

FsmOutput StateMachine::on_hazard_recovery(const FsmInput& in) {
    if (in.is_stuck) {
        // 保留 replan_after_recovery_（不重置），恢复链结束后仍需 replan。
        pending_reverse_start_time_ = in.stamp;
        pending_reverse_start_pos_ = in.chassis_pos_map;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE (is stuck)");
        return transition_to(MotionState::STUCK_REVERSE);
    }

    if (in.is_recovery_safe) {
        return finish_recovery_chain(in);
    }

    return { .state = MotionState::HAZARD_RECOVERY };
}

} // namespace nav_executor
