#include <local_planner/state_machine.hpp>

#include <cmath>
#include <rclcpp/logging.hpp>

namespace local_planner {

LocalPlannerStateMachine::LocalPlannerStateMachine(const LocalPlannerFsmParams& params, rclcpp::Logger logger)
    : params_(params), logger_(logger) {}

void LocalPlannerStateMachine::enter_state(PlannerState new_state) {
    if (state_ == new_state) return;
    state_ = new_state;
    switch (new_state) {
        case PlannerState::IDLE: RCLCPP_INFO(logger_, "FSM -> IDLE"); break;
        case PlannerState::TRACK: RCLCPP_INFO(logger_, "FSM -> TRACK"); break;
        case PlannerState::SPIN: RCLCPP_INFO(logger_, "FSM -> SPIN"); break;
        case PlannerState::STOP_TRANSITION: RCLCPP_INFO(logger_, "FSM -> STOP_TRANSITION"); break;
        case PlannerState::HOLD_FIXED: RCLCPP_INFO(logger_, "FSM -> HOLD_FIXED"); break;
        case PlannerState::REVERSE: RCLCPP_WARN(logger_, "FSM -> REVERSE"); break;
        case PlannerState::HAZARD_RECOVERY: RCLCPP_WARN(logger_, "FSM -> HAZARD_RECOVERY"); break;
    }
}

bool LocalPlannerStateMachine::stopping_ready(const LocalPlannerFsmInput& in) const {
    // 根据 stop_dest_ 使用不同的速度门槛（实现 §4.1 中 follow_to_spin_vel_max / spin_to_follow_omega_max 的语义）
    switch (stop_dest_) {
        case StopDest::SPIN:
            return std::abs(in.velocity) < params_.transition.follow_to_spin_vel_max;
        case StopDest::TRACK:
            return std::abs(in.omega) < params_.transition.spin_to_follow_omega_max;
        default: // IDLE, FIXED
            return std::abs(in.velocity) < params_.transition.to_idle_vel_max &&
                std::abs(in.omega) < params_.transition.to_idle_omega_max;
    }
}

LocalPlannerFsmOutput LocalPlannerStateMachine::update(const LocalPlannerFsmInput& input) {
    LocalPlannerFsmOutput output;
    output.consume_global_path = false;

    switch (state_) {

    // ═══════════════════ IDLE ═══════════════════════════════
    case PlannerState::IDLE: {
        // 障碍物中直接进入 hazard recovery
        if (input.is_in_hazard && params_.stuck.enable) {
            output.consume_global_path = true;
            enter_state(PlannerState::HAZARD_RECOVERY);
            break;
        }
        if (input.has_new_path) {
            enter_state(PlannerState::TRACK);
            break;
        }
        if (input.spin_requested) {
            enter_state(PlannerState::SPIN);
            break;
        }
        break;
    }

    // ═══════════════════ TRACK ══════════════════════════════
    case PlannerState::TRACK: {
        // 路径没了（被上层消费/丢失）
        if (!input.has_path) {
            const bool should_spin = input.spin_requested &&
                (input.spin_high_priority || !input.fixed_goal_flag);
            if (should_spin) {
                stop_dest_ = StopDest::SPIN;
            } else if (input.fixed_goal_flag) {
                stop_dest_ = StopDest::FIXED;
            } else {
                stop_dest_ = StopDest::IDLE;
            }
            stopping_start_time_ = input.stamp;
            enter_state(PlannerState::STOP_TRANSITION);
            break;
        }

        // stuck → REVERSE (V1: 简单倒车后回 IDLE)
        if (input.is_stuck && params_.stuck.enable) {
            reverse_start_time_ = input.stamp;
            enter_state(PlannerState::REVERSE);
            break;
        }

        // high-priority spin 打断跟随
        if (input.spin_requested && input.spin_high_priority) {
            stop_dest_ = StopDest::SPIN;
            stopping_start_time_ = input.stamp;
            enter_state(PlannerState::STOP_TRANSITION);
            break;
        }

        // 到达终点
        if (input.reach_goal) {
            if (input.fixed_goal_flag) {
                stop_dest_ = StopDest::FIXED;
            } else {
                stop_dest_ = StopDest::IDLE;
            }
            stopping_start_time_ = input.stamp;
            enter_state(PlannerState::STOP_TRANSITION);
            break;
        }
        break;
    }

    // ═══════════════════ SPIN ═══════════════════════════════
    case PlannerState::SPIN: {
        const bool keep_spinning = input.spin_requested &&
            (input.spin_high_priority || (!input.has_path && !input.fixed_goal_flag));

        // 障碍物中直接进入 hazard recovery
        if (input.is_in_hazard && params_.stuck.enable) {
            output.consume_global_path = true;
            enter_state(PlannerState::HAZARD_RECOVERY);
            break;
        }

        if (input.is_stuck && params_.stuck.enable) {
            reverse_start_time_ = input.stamp;
            enter_state(PlannerState::REVERSE);
            break;
        }

        if (!keep_spinning) {
            if (input.has_path) {
                stop_dest_ = StopDest::TRACK;
            } else if (input.fixed_goal_flag) {
                stop_dest_ = StopDest::FIXED;
            } else {
                stop_dest_ = StopDest::IDLE;
            }
            stopping_start_time_ = input.stamp;
            enter_state(PlannerState::STOP_TRANSITION);
            break;
        }
        break;
    }

    // ═══════════════════ STOP_TRANSITION ════════════════════
    case PlannerState::STOP_TRANSITION: {
        if (input.is_stuck && params_.stuck.enable) {
            reverse_start_time_ = input.stamp;
            enter_state(PlannerState::REVERSE);
            break;
        }

        // 如果收到新路径且当前目的不是 TRACK，立即切 TRACK
        if (stop_dest_ != StopDest::TRACK && input.has_new_path) {
            enter_state(PlannerState::TRACK);
            break;
        }

        const bool timeout = std::chrono::duration<double>(input.stamp - stopping_start_time_).count() > params_.transition.stopping_timeout;
        if (stopping_ready(input) || timeout) {
            switch (stop_dest_) {
                case StopDest::IDLE:
                    output.consume_global_path = true;
                    enter_state(PlannerState::IDLE);
                    break;
                case StopDest::FIXED:
                    output.consume_global_path = true;
                    enter_state(PlannerState::HOLD_FIXED);
                    break;
                case StopDest::SPIN:
                    enter_state(PlannerState::SPIN);
                    break;
                case StopDest::TRACK:
                    enter_state(PlannerState::TRACK);
                    break;
            }
            break;
        }
        break;
    }

    // ═══════════════════ HOLD_FIXED ════════════════════════
    case PlannerState::HOLD_FIXED: {
        if (input.is_stuck && params_.stuck.enable) {
            reverse_start_time_ = input.stamp;
            enter_state(PlannerState::REVERSE);
            break;
        }

        // high-priority spin 打断 fixed
        if (input.spin_requested && input.spin_high_priority) {
            stop_dest_ = StopDest::SPIN;
            stopping_start_time_ = input.stamp;
            enter_state(PlannerState::STOP_TRANSITION);
            break;
        }

        // 收到新路径，切 TRACK
        if (input.has_new_path) {
            enter_state(PlannerState::TRACK);
            break;
        }
        break;
    }

    // ═══════════════════ REVERSE ════════════════════════════
    case PlannerState::REVERSE: {
        const double elapsed = std::chrono::duration<double>(input.stamp - reverse_start_time_).count();
        if (elapsed > params_.stuck.reverse_duration) {
            output.consume_global_path = true;
            enter_state(PlannerState::HAZARD_RECOVERY);
        }
        break;
    }

    // ═══════════════════ HAZARD_RECOVERY ════════════════════
    case PlannerState::HAZARD_RECOVERY: {
        // 收到新路径任务 → 退出 recovery
        if (input.has_new_path) {
            enter_state(PlannerState::TRACK);
            break;
        }
        // 已经连续回到安全区域 → 退出 recovery
        if (input.is_recovery_safe) {
            enter_state(PlannerState::IDLE);
            break;
        }
        break;
    }

    } // switch

    output.state = state_;
    return output;
}

} // namespace local_planner