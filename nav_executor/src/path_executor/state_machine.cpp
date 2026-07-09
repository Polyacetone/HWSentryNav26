#include <nav_executor/path_executor/state_machine.hpp>

#include <Eigen/Core>
#include <rclcpp/logging.hpp>

namespace nav_executor {

StateMachine::StateMachine(const FsmParams& params, rclcpp::Logger logger)
    : params_(params), logger_(logger) {}

FsmOutput StateMachine::update(const FsmInput& input) {
    switch (active_state_) {
        case MotionState::IDLE:            return on_idle(input);
        case MotionState::FIXED:           return on_fixed(input);
        case MotionState::FOLLOW:          return on_follow(input);
        case MotionState::STEPPING:        return on_stepping(input);
        case MotionState::SPIN:            return on_spin(input);
        case MotionState::STOPPING:        return on_stopping(input);
        case MotionState::STUCK_REVERSE:   return on_stuck_reverse(input);
        case MotionState::HAZARD_RECOVERY: return on_hazard_recovery(input);
        default:                           return { .state = active_state_ };
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

FsmOutput StateMachine::exit_reverse(const FsmInput& in, const double displacement, const double mature_elapsed) {
    if (in.is_hazard) {
        RCLCPP_WARN(
            logger_,
            "STUCK_REVERSE: displaced %.2f m (mature=%.1f s), current position IS hazard, entering HAZARD_RECOVERY",
            displacement, mature_elapsed
        );
        return transition_to(MotionState::HAZARD_RECOVERY);
    }

    RCLCPP_WARN(
        logger_,
        "STUCK_REVERSE: displaced %.2f m (mature=%.1f s), current position safe, skipping HAZARD_RECOVERY",
        displacement, mature_elapsed
    );
    return finish_recovery_chain(in);
}

// ═══════════════════════════ IDLE ═══════════════════════════

FsmOutput StateMachine::on_idle(const FsmInput& in) {
    if (in.is_hazard) {
        replan_after_recovery_ = false;
        RCLCPP_WARN(logger_, "FSM -> HAZARD_RECOVERY");
        return transition_to(MotionState::HAZARD_RECOVERY);
    }
    if (in.has_path) {
        RCLCPP_INFO(logger_, "FSM -> FOLLOW");
        return transition_to(MotionState::FOLLOW);
    }
    if (in.has_hold_goal) {
        RCLCPP_INFO(logger_, "FSM -> FIXED");
        return transition_to(MotionState::FIXED);
    }
    if (in.spin_requested) {
        RCLCPP_INFO(logger_, "FSM -> SPIN");
        return transition_to(MotionState::SPIN);
    }
    return { .state = MotionState::IDLE };
}

// ═══════════════════════════ FIXED ══════════════════════════
// FIXED 只能通过顶层设 hold_goal 进入，不靠 current_goal.fixed 直接推导。
// FIXED 是持续保持模式，不再上报"再次完成"。
// 当 hold_goal 消失（任务被清空时）→ 平滑退出到 STOPPING。
// 注意：executor_replan_event 在此状态下会被顶层吞掉（见 ingest_executor_replan_event），
// 所以 FIXED 下的 stuck 恢复后不会导致路径重规划——恢复后因 hold_goal 仍在，自然回 FIXED。

FsmOutput StateMachine::on_fixed(const FsmInput& in) {
    if (in.is_stuck) {
        replan_after_recovery_ = true;
        pending_reverse_start_time_ = in.stamp;
        pending_reverse_start_pos_ = in.chassis_pos_map;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE");
        return transition_to(MotionState::STUCK_REVERSE);
    }
    if (in.has_path) {
        RCLCPP_INFO(logger_, "FSM -> FOLLOW");
        return transition_to(MotionState::FOLLOW);
    }
    if (in.spin_requested && in.spin_high_priority) {
        stopping_start_time_ = in.stamp;
        RCLCPP_INFO(logger_, "FSM -> STOPPING");
        return transition_to(MotionState::STOPPING);
    }
    // hold_goal 消失（任务被清空）→ 平滑退出
    if (!in.has_hold_goal) {
        stopping_start_time_ = in.stamp;
        RCLCPP_INFO(logger_, "FSM -> STOPPING");
        return transition_to(MotionState::STOPPING);
    }
    return { .state = MotionState::FIXED };
}

// ═══════════════════════════ FOLLOW ═════════════════════════
// 优先级：step_active（不可抢占 STEPPING）> stuck-like > 无 path（顶层已判 invalid）> spin > reach_goal
// stuck-like（no_progress_detected / is_stuck）统一走 STUCK_REVERSE 脱困链，
// 不同于旧版直接进入 WAIT_REPLAN。恢复链结束后发出 one-shot executor_replan_event。
// path 被顶层清空（RouteMonitor 判 invalid 或消费后）→ 平滑停止后回落到对应终态。
// step_active 最高优先，不被 stuck/no_progress 打断前置处理。

FsmOutput StateMachine::on_follow(const FsmInput& in) {
    // step_active 最高优先：进入不可抢占的 STEPPING，且不被 stuck/no_progress 打断前置处理。
    if (in.step_active) {
        RCLCPP_INFO(logger_, "FSM -> STEPPING");
        return transition_to(MotionState::STEPPING);
    }

    // stuck-like（follow no-progress 或卡住）统一走脱困链，恢复后 one-shot replan。
    if (in.no_progress_detected || in.is_stuck) {
        replan_after_recovery_ = true;
        pending_reverse_start_time_ = in.stamp;
        pending_reverse_start_pos_ = in.chassis_pos_map;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE (stuck-like in FOLLOW)");
        return transition_to(MotionState::STUCK_REVERSE);
    }

    // path 被顶层清空（RouteMonitor 判 invalid 或消费）→ 平滑停止后回落。
    if (!in.has_path) {
        stopping_start_time_ = in.stamp;
        RCLCPP_INFO(logger_, "FSM -> STOPPING");
        return transition_to(MotionState::STOPPING);
    }

    if (in.spin_requested && in.spin_high_priority) {
        stopping_start_time_ = in.stamp;
        RCLCPP_INFO(logger_, "FSM -> STOPPING");
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
// 约束：no_progress 和 stuck 视为 stuck-like，不走直接 replan，
// 统一走 STUCK_REVERSE→HAZARD_RECOVERY 恢复链。恢复后 one-shot replan event。

FsmOutput StateMachine::on_stepping(const FsmInput& in) {
    if (in.no_progress_detected || in.is_stuck) {
        replan_after_recovery_ = true;
        pending_reverse_start_time_ = in.stamp;
        pending_reverse_start_pos_ = in.chassis_pos_map;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE (stuck-like in STEPPING)");
        return transition_to(MotionState::STUCK_REVERSE);
    }

    if (in.step_active) {
        return { .state = MotionState::STEPPING };
    }

    RCLCPP_INFO(logger_, "FSM -> FOLLOW");
    return transition_to(MotionState::FOLLOW);
}

// ═══════════════════════════ SPIN ═══════════════════════════

FsmOutput StateMachine::on_spin(const FsmInput& in) {
    if (in.is_hazard) {
        replan_after_recovery_ = false;
        RCLCPP_WARN(logger_, "FSM -> HAZARD_RECOVERY");
        return transition_to(MotionState::HAZARD_RECOVERY);
    }

    const bool keep_spinning = in.spin_requested
        && (in.spin_high_priority || (!in.has_path && !in.has_hold_goal));

    if (!keep_spinning) {
        stopping_start_time_ = in.stamp;
        RCLCPP_INFO(logger_, "FSM -> STOPPING");
        return transition_to(MotionState::STOPPING);
    }

    return { .state = MotionState::SPIN };
}

// ═══════════════════════════ STOPPING ═══════════════════════

FsmOutput StateMachine::on_stopping(const FsmInput& in) {
    const Terminal target = terminal_target(in);

    // 新任务（有 path）→ 立即跟随，无需等待减速。
    if (target == Terminal::FOLLOW) {
        RCLCPP_INFO(logger_, "FSM -> FOLLOW");
        return transition_to(MotionState::FOLLOW);
    }

    const bool timeout = std::chrono::duration<double>(in.stamp - stopping_start_time_).count()
        > params_.transition.stopping_timeout;

    if (stopping_ready(in, target) || timeout) {
        switch (target) {
            case Terminal::FIXED: RCLCPP_INFO(logger_, "FSM -> FIXED"); return transition_to(MotionState::FIXED);
            case Terminal::SPIN:  RCLCPP_INFO(logger_, "FSM -> SPIN");  return transition_to(MotionState::SPIN);
            case Terminal::IDLE:  RCLCPP_INFO(logger_, "FSM -> IDLE");  return transition_to(MotionState::IDLE);
            case Terminal::FOLLOW: return transition_to(MotionState::FOLLOW);
        }
    }

    return { .state = MotionState::STOPPING };
}

// ═══════════════════════════ STUCK_REVERSE ══════════════════
// stuck-like 统一走此恢复链。退出后若当前位姿不安全进入 HAZARD_RECOVERY，
// 若安全则直接 finish_recovery_chain（后者发出 one-shot executor_replan_event）。
// 不允许在恢复链中途提前发出 executor_replan_event。

FsmOutput StateMachine::on_stuck_reverse(const FsmInput& in) {
    if (!reverse_entry_initialized_) {
        reverse_entry_initialized_ = true;
        reverse_mature_accumulated_ = 0.0;
        reverse_last_mature_stamp_ = pending_reverse_start_time_;
        reverse_entry_pos_ = pending_reverse_start_pos_;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE");
    }

    if (in.command_blocked) {
        reverse_mature_accumulated_ +=
            std::chrono::duration<double>(in.stamp - reverse_last_mature_stamp_).count();
        reverse_last_mature_stamp_ = in.stamp;
        return { .state = MotionState::STUCK_REVERSE };
    }

    const double mature_elapsed = reverse_mature_accumulated_
        + std::chrono::duration<double>(in.stamp - reverse_last_mature_stamp_).count();
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
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE");
        return transition_to(MotionState::STUCK_REVERSE);
    }

    if (in.is_recovery_safe) {
        return finish_recovery_chain(in);
    }

    return { .state = MotionState::HAZARD_RECOVERY };
}

} // namespace nav_executor
