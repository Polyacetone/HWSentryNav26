#include <path_follower/control/state_machine.hpp>

#include <Eigen/Core>
#include <rclcpp/logging.hpp>

namespace path_follower {

StateMachine::StateMachine(const FsmParams& params, rclcpp::Logger logger)
    : params_(params), logger_(logger) {}

FsmOutput StateMachine::update(const FsmInput& input) {
    switch (active_state_) {
        case FsmState::IDLE:            return on_idle(input);
        case FsmState::FIXED:           return on_fixed(input);
        case FsmState::FOLLOW:          return on_follow(input);
        case FsmState::STEPPING:        return on_stepping(input);
        case FsmState::SPIN:            return on_spin(input);
        case FsmState::STOPPING:        return on_stopping(input);
        case FsmState::STUCK_REVERSE:   return on_stuck_reverse(input);
        case FsmState::HAZARD_RECOVERY: return on_hazard_recovery(input);
        case FsmState::WAIT_REPLAN:     return on_wait_replan(input);
        default:                        return { .state = active_state_ };
    }
}

// ═══════════════════════════ 辅助 ═══════════════════════════

bool StateMachine::stopping_ready(const FsmInput& in) const {
    const auto& t = params_.transition;
    switch (stopping_dest_) {
        case DestState::IDLE:
        case DestState::FIXED:
            return std::abs(in.velocity) < t.to_idle_vel_max &&
                std::abs(in.omega) < t.to_idle_omega_max;
        case DestState::SPIN:
            return std::abs(in.velocity) < t.follow_to_spin_vel_max &&
                std::abs(in.omega) < t.to_idle_omega_max;
        case DestState::FOLLOW:
            return std::abs(in.velocity) < t.to_idle_vel_max &&
                std::abs(in.omega) < t.spin_to_follow_omega_max;
    }
    return false;
}

FsmOutput StateMachine::route_to_terminal(const FsmInput& in) {
    const bool should_spin = in.spin_requested
        && (in.spin_high_priority || (!in.has_path && !in.fixed_goal_flag));
    if (should_spin) return { .state = FsmState::SPIN };
    if (in.has_path) return { .state = FsmState::FOLLOW };
    if (in.fixed_goal_flag) return { .state = FsmState::FIXED };

    FsmOutput out;
    out.state = FsmState::IDLE;
    out.consume_global_path = true;
    return out;
}

FsmOutput StateMachine::exit_reverse(const FsmInput& in, double displacement, double mature_elapsed) {
    if (in.is_hazard) {
        RCLCPP_WARN(
            logger_,
            "STUCK_REVERSE: displaced %.2f m (mature=%.1f s), current position IS hazard, entering HAZARD_RECOVERY",
            displacement, mature_elapsed
        );
        return { .state = FsmState::HAZARD_RECOVERY };
    }

    RCLCPP_WARN(
        logger_,
        "STUCK_REVERSE: displaced %.2f m (mature=%.1f s), current position safe, skipping HAZARD_RECOVERY",
        displacement, mature_elapsed
    );

    if (replan_after_recovery_) {
        replan_after_recovery_ = false;
        pending_wait_replan_start_time_ = in.stamp;
        FsmOutput out;
        out.state = FsmState::WAIT_REPLAN;
        out.consume_global_path = true;
        out.request_replan = true;
        return out;
    }

    return route_to_terminal(in);
}

// ═══════════════════════════ IDLE ═══════════════════════════

FsmOutput StateMachine::on_idle(const FsmInput& in) {
    if (in.is_hazard) {
        return { .state = FsmState::HAZARD_RECOVERY };
    }
    if (in.has_new_path) {
        active_state_ = FsmState::FOLLOW;
        RCLCPP_INFO(logger_, "FSM -> FOLLOW");
        return { .state = FsmState::FOLLOW };
    }
    if (in.spin_requested) {
        active_state_ = FsmState::SPIN;
        RCLCPP_INFO(logger_, "FSM -> SPIN");
        return { .state = FsmState::SPIN };
    }
    return { .state = FsmState::IDLE };
}

// ═══════════════════════════ FIXED ══════════════════════════

FsmOutput StateMachine::on_fixed(const FsmInput& in) {
    if (in.is_stuck) {
        replan_after_recovery_ = true;
        pending_reverse_start_time_ = in.stamp;
        pending_reverse_start_pos_ = in.chassis_pos_map;
        active_state_ = FsmState::STUCK_REVERSE;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE");
        return { .state = FsmState::STUCK_REVERSE };
    }
    if (in.spin_requested && in.spin_high_priority) {
        stopping_dest_ = DestState::SPIN;
        stopping_start_time_ = in.stamp;
        active_state_ = FsmState::STOPPING;
        RCLCPP_INFO(logger_, "FSM -> STOPPING");
        return { .state = FsmState::STOPPING };
    }
    if (in.has_new_path) {
        active_state_ = FsmState::FOLLOW;
        RCLCPP_INFO(logger_, "FSM -> FOLLOW");
        return { .state = FsmState::FOLLOW };
    }
    return { .state = FsmState::FIXED };
}

// ═══════════════════════════ FOLLOW ═════════════════════════

FsmOutput StateMachine::on_follow(const FsmInput& in) {
    // step_active has highest priority: stepping locks the path and
    // must not be interrupted by replan/stuck/no_progress signals.
    if (in.step_active) {
        active_state_ = FsmState::STEPPING;
        RCLCPP_INFO(logger_, "FSM -> STEPPING");
        return { .state = FsmState::STEPPING };
    }

    if (in.replan_requested) {
        pending_wait_replan_start_time_ = in.stamp;
        active_state_ = FsmState::WAIT_REPLAN;
        RCLCPP_INFO(logger_, "FSM -> WAIT_REPLAN");
        FsmOutput out;
        out.state = FsmState::WAIT_REPLAN;
        out.consume_global_path = true;
        out.request_replan = true;
        return out;
    }

    if (in.no_progress_detected) {
        if (in.is_hazard) {
            replan_after_recovery_ = true;
            active_state_ = FsmState::HAZARD_RECOVERY;
            RCLCPP_WARN(logger_, "FSM -> HAZARD_RECOVERY");
            return { .state = FsmState::HAZARD_RECOVERY };
        }
        pending_wait_replan_start_time_ = in.stamp;
        active_state_ = FsmState::WAIT_REPLAN;
        RCLCPP_INFO(logger_, "FSM -> WAIT_REPLAN");
        FsmOutput out;
        out.state = FsmState::WAIT_REPLAN;
        out.consume_global_path = true;
        out.request_replan = true;
        return out;
    }

    if (!in.has_path) {
        const bool should_spin = in.spin_requested && (in.spin_high_priority || (!in.fixed_goal_flag));
        if (should_spin) {
            stopping_dest_ = DestState::SPIN;
        } else if (in.fixed_goal_flag) {
            stopping_dest_ = DestState::FIXED;
        } else {
            stopping_dest_ = DestState::IDLE;
        }
        stopping_start_time_ = in.stamp;
        active_state_ = FsmState::STOPPING;
        RCLCPP_INFO(logger_, "FSM -> STOPPING");
        return { .state = FsmState::STOPPING };
    }

    if (in.is_stuck) {
        replan_after_recovery_ = true;
        pending_reverse_start_time_ = in.stamp;
        pending_reverse_start_pos_ = in.chassis_pos_map;
        active_state_ = FsmState::STUCK_REVERSE;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE");
        return { .state = FsmState::STUCK_REVERSE };
    }

    if (in.spin_requested && in.spin_high_priority) {
        stopping_dest_ = DestState::SPIN;
        stopping_start_time_ = in.stamp;
        active_state_ = FsmState::STOPPING;
        RCLCPP_INFO(logger_, "FSM -> STOPPING");
        return { .state = FsmState::STOPPING };
    }

    if (in.reach_goal) {
        stopping_dest_ = in.fixed_goal_flag ? DestState::FIXED : DestState::IDLE;
        stopping_start_time_ = in.stamp;
        active_state_ = FsmState::STOPPING;
        RCLCPP_INFO(logger_, "FSM -> STOPPING");
        return { .state = FsmState::STOPPING };
    }

    return { .state = FsmState::FOLLOW };
}

// ═══════════════════════════ STEPPING ═══════════════════════

FsmOutput StateMachine::on_stepping(const FsmInput& in) {
    if (in.no_progress_detected) {
        if (in.is_hazard) {
            replan_after_recovery_ = true;
            active_state_ = FsmState::HAZARD_RECOVERY;
            RCLCPP_WARN(logger_, "FSM -> HAZARD_RECOVERY");
            return { .state = FsmState::HAZARD_RECOVERY };
        }
        pending_wait_replan_start_time_ = in.stamp;
        active_state_ = FsmState::WAIT_REPLAN;
        RCLCPP_INFO(logger_, "FSM -> WAIT_REPLAN");
        FsmOutput out;
        out.state = FsmState::WAIT_REPLAN;
        out.consume_global_path = true;
        out.request_replan = true;
        return out;
    }

    if (in.is_stuck) {
        replan_after_recovery_ = false;
        pending_reverse_start_time_ = in.stamp;
        pending_reverse_start_pos_ = in.chassis_pos_map;
        active_state_ = FsmState::STUCK_REVERSE;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE");
        return { .state = FsmState::STUCK_REVERSE };
    }

    if (in.step_active) {
        return { .state = FsmState::STEPPING };
    }

    if (in.replan_requested) {
        pending_wait_replan_start_time_ = in.stamp;
        active_state_ = FsmState::WAIT_REPLAN;
        RCLCPP_INFO(logger_, "FSM -> WAIT_REPLAN");
        FsmOutput out;
        out.state = FsmState::WAIT_REPLAN;
        out.consume_global_path = true;
        out.request_replan = true;
        return out;
    }

    active_state_ = FsmState::FOLLOW;
    RCLCPP_INFO(logger_, "FSM -> FOLLOW");
    return { .state = FsmState::FOLLOW };
}

// ═══════════════════════════ SPIN ═══════════════════════════

FsmOutput StateMachine::on_spin(const FsmInput& in) {
    if (in.is_hazard) {
        RCLCPP_WARN(logger_, "FSM -> HAZARD_RECOVERY");
        return { .state = FsmState::HAZARD_RECOVERY };
    }

    const bool keep_spinning = in.spin_requested
        && (in.spin_high_priority || (!in.has_path && !in.fixed_goal_flag));

    if (!keep_spinning) {
        if (in.has_path) {
            stopping_dest_ = DestState::FOLLOW;
        } else if (in.fixed_goal_flag) {
            stopping_dest_ = DestState::FIXED;
        } else {
            stopping_dest_ = DestState::IDLE;
        }
        stopping_start_time_ = in.stamp;
        active_state_ = FsmState::STOPPING;
        RCLCPP_INFO(logger_, "FSM -> STOPPING");
        return { .state = FsmState::STOPPING };
    }

    return { .state = FsmState::SPIN };
}

// ═══════════════════════════ STOPPING ═══════════════════════

FsmOutput StateMachine::on_stopping(const FsmInput& in) {
    if (stopping_dest_ != DestState::FOLLOW && in.has_new_path) {
        RCLCPP_INFO(logger_, "FSM -> FOLLOW");
        return { .state = FsmState::FOLLOW };
    }

    const bool timeout = std::chrono::duration<double>(in.stamp - stopping_start_time_).count()
        > params_.transition.stopping_timeout;

    if (stopping_ready(in) || timeout) {
        switch (stopping_dest_) {
            case DestState::IDLE: {
                active_state_ = FsmState::IDLE;
                RCLCPP_INFO(logger_, "FSM -> IDLE");
                FsmOutput out;
                out.state = FsmState::IDLE;
                out.consume_global_path = true;
                return out;
            }
            case DestState::FIXED: {
                active_state_ = FsmState::FIXED;
                RCLCPP_INFO(logger_, "FSM -> FIXED");
                FsmOutput out;
                out.state = FsmState::FIXED;
                out.consume_global_path = true;
                return out;
            }
            case DestState::SPIN: {
                active_state_ = FsmState::SPIN;
                RCLCPP_INFO(logger_, "FSM -> SPIN");
                return { .state = FsmState::SPIN };
            }
            case DestState::FOLLOW: {
                active_state_ = FsmState::FOLLOW;
                RCLCPP_INFO(logger_, "FSM -> FOLLOW");
                return { .state = FsmState::FOLLOW };
            }
        }
    }

    return { .state = FsmState::STOPPING };
}

// ═══════════════════════════ STUCK_REVERSE ══════════════════

FsmOutput StateMachine::on_stuck_reverse(const FsmInput& in) {
    if (reverse_mature_accumulated_ == 0.0) {
        // 首次进入，初始化
        active_state_ = FsmState::STUCK_REVERSE;
        reverse_mature_accumulated_ = 0.0;
        reverse_last_mature_stamp_ = pending_reverse_start_time_;
        reverse_entry_pos_ = pending_reverse_start_pos_;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE");
    }

    if (in.command_blocked) {
        reverse_mature_accumulated_ +=
            std::chrono::duration<double>(in.stamp - reverse_last_mature_stamp_).count();
        reverse_last_mature_stamp_ = in.stamp;
        return { .state = FsmState::STUCK_REVERSE };
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

    return { .state = FsmState::STUCK_REVERSE };
}

// ═══════════════════════════ HAZARD_RECOVERY ════════════════

FsmOutput StateMachine::on_hazard_recovery(const FsmInput& in) {
    if (in.is_stuck) {
        replan_after_recovery_ = false;
        pending_reverse_start_time_ = in.stamp;
        pending_reverse_start_pos_ = in.chassis_pos_map;
        active_state_ = FsmState::STUCK_REVERSE;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE");
        return { .state = FsmState::STUCK_REVERSE };
    }

    if (in.is_recovery_safe) {
        if (replan_after_recovery_) {
            replan_after_recovery_ = false;
            pending_wait_replan_start_time_ = in.stamp;
            active_state_ = FsmState::WAIT_REPLAN;
            RCLCPP_INFO(logger_, "FSM -> WAIT_REPLAN");
            FsmOutput out;
            out.state = FsmState::WAIT_REPLAN;
            out.consume_global_path = true;
            out.request_replan = true;
            return out;
        }
        return route_to_terminal(in);
    }

    return { .state = FsmState::HAZARD_RECOVERY };
}

// ═══════════════════════════ WAIT_REPLAN ════════════════════

FsmOutput StateMachine::on_wait_replan(const FsmInput& in) {
    if (in.has_new_path) {
        active_state_ = FsmState::FOLLOW;
        RCLCPP_INFO(logger_, "FSM -> FOLLOW");
        return { .state = FsmState::FOLLOW };
    }

    if (in.replan_failed) {
        RCLCPP_WARN(logger_, "WAIT_REPLAN: replan failed (empty path)");
        const bool should_spin = in.spin_requested && (in.spin_high_priority || (!in.fixed_goal_flag));
        if (should_spin) {
            active_state_ = FsmState::SPIN;
            RCLCPP_INFO(logger_, "FSM -> SPIN");
            return { .state = FsmState::SPIN };
        }
        active_state_ = FsmState::IDLE;
        RCLCPP_INFO(logger_, "FSM -> IDLE");
        FsmOutput out;
        out.state = FsmState::IDLE;
        out.consume_global_path = true;
        return out;
    }

    const bool timeout = std::chrono::duration<double>(in.stamp - pending_wait_replan_start_time_).count()
        > params_.transition.wait_replan_timeout;
    if (timeout) {
        RCLCPP_WARN(logger_, "WAIT_REPLAN timed out after %.2f s without a valid path",
            params_.transition.wait_replan_timeout);
        const bool should_spin = in.spin_requested && (in.spin_high_priority || (!in.fixed_goal_flag));
        if (should_spin) {
            active_state_ = FsmState::SPIN;
            RCLCPP_INFO(logger_, "FSM -> SPIN");
            return { .state = FsmState::SPIN };
        }
        active_state_ = FsmState::IDLE;
        RCLCPP_INFO(logger_, "FSM -> IDLE");
        FsmOutput out;
        out.state = FsmState::IDLE;
        out.consume_global_path = true;
        return out;
    }

    FsmOutput out;
    out.state = FsmState::WAIT_REPLAN;
    out.consume_global_path = true;
    out.request_replan = true;
    return out;
}

} // namespace path_follower
