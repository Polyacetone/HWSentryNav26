#include <nav_executor/path_executor/state/state_machine.hpp>

#include <Eigen/Core>
#include <rclcpp/logging.hpp>

namespace nav_executor {

const char* step_phase_str(const StepPhase phase) {
    switch (phase) {
        case StepPhase::NONE: return "NONE";
        case StepPhase::PREPARING: return "PREPARING";
        case StepPhase::ARMED: return "ARMED";
        case StepPhase::COMMITTED: return "COMMITTED";
    }
    return "NONE";
}

StateMachine::StateMachine(const FsmParams& params, rclcpp::Logger logger)
    : params_(params), logger_(logger) {}

FsmOutput StateMachine::update(const FsmInput& input) {
    switch (active_state_) {
        case MotionState::IDLE: return on_idle(input);
        case MotionState::FIXED: return on_fixed(input);
        case MotionState::FOLLOW: return on_follow(input);
        case MotionState::STEPPING: return on_stepping(input);
        case MotionState::SPIN: return on_spin(input);
        case MotionState::PREPARE_SPIN: return on_prepare_spin(input);
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
    if (next != MotionState::STEPPING) {
        step_phase_ = StepPhase::NONE;
        step_path_epoch_ = 0;
        step_segment_index_.reset();
    }
    return { .state = next };
}

FsmOutput StateMachine::enter_stepping(
    const StepPhase phase,
    const uint64_t path_epoch,
    const std::optional<size_t> segment_index
) {
    step_phase_ = phase;
    step_path_epoch_ = path_epoch;
    step_segment_index_ = segment_index;
    return transition_to(MotionState::STEPPING);
}

void StateMachine::synchronize_step_phase(
    const StepPhase observed_phase,
    const uint64_t path_epoch,
    const std::optional<size_t> segment_index
) {
    if (!is_step_phase_active(observed_phase)) return;

    // segment 下标只在一条 AnnotatedPath 内有意义。路径身份变化时，即使下标相同，
    // 也必须从新路径观测到的阶段开启新的生命周期，不能继承旧路径的 ARMED/COMMITTED。
    if (path_epoch != step_path_epoch_) {
        if (segment_index) {
            RCLCPP_INFO(
                logger_, "STEPPING path epoch changed to %lu, segment=#%zu phase=%s",
                static_cast<unsigned long>(path_epoch), *segment_index,
                step_phase_str(observed_phase)
            );
        } else {
            RCLCPP_INFO(
                logger_, "STEPPING path epoch changed to %lu, segment=none phase=%s",
                static_cast<unsigned long>(path_epoch), step_phase_str(observed_phase)
            );
        }
        step_path_epoch_ = path_epoch;
        step_segment_index_ = segment_index;
        step_phase_ = observed_phase;
        return;
    }

    // 相邻台阶段可以在上一段 release 的同一位置进入下一段 prepare。段身份变化时，
    // PREPARING 是新生命周期的合法起点，不属于同一段内的阶段回退。
    if (segment_index && segment_index != step_segment_index_) {
        RCLCPP_INFO(
            logger_, "STEPPING segment switched to #%zu, phase=%s",
            *segment_index, step_phase_str(observed_phase)
        );
        step_segment_index_ = segment_index;
        step_phase_ = observed_phase;
        return;
    }

    // 同一台阶段一旦越过边界便只向前推进，投影抖动不能解除 ARMED 或 COMMITTED。
    if (static_cast<uint8_t>(observed_phase) <= static_cast<uint8_t>(step_phase_)) return;

    RCLCPP_INFO(
        logger_, "STEPPING phase %s -> %s",
        step_phase_str(step_phase_), step_phase_str(observed_phase)
    );
    step_phase_ = observed_phase;
}

bool StateMachine::spin_control_requested(const FsmInput& in) const {
    return is_spin_owner(in.requested_owner);
}

bool StateMachine::prepare_spin_ready(const FsmInput& in) const {
    const PrepareSpinParams& limits = params_.prepare_spin;
    return !in.command_blocked && in.command_state_tracked
        && std::abs(in.command_velocity) < limits.command_velocity_max
        && std::abs(in.measured_velocity) < limits.measured_velocity_max;
}

FsmOutput StateMachine::route_to_normal_state(const FsmInput& in) {
    if (in.route_tracked) return transition_to(MotionState::FOLLOW);
    if (in.has_hold_goal) return transition_to(MotionState::FIXED);
    return transition_to(MotionState::IDLE);
}

FsmOutput StateMachine::route_to_requested_state(const FsmInput& in) {
    return spin_control_requested(in)
        ? transition_to(MotionState::PREPARE_SPIN)
        : route_to_normal_state(in);
}

FsmOutput StateMachine::finish_recovery_chain(const FsmInput& in) {
    if (replan_after_recovery_) {
        replan_after_recovery_ = false;
        // Recovery invalidates the execution that led to it. Do not resume the
        // old path for one cycle while TaskManager consumes the one-shot event.
        FsmOutput out = transition_to(MotionState::IDLE);
        out.executor_replan_event = true;
        RCLCPP_INFO(logger_, "Recovery completed; holding IDLE until current path is replanned");
        return out;
    }
    return route_to_requested_state(in);
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
    if (spin_control_requested(in)) {
        RCLCPP_INFO(logger_, "FSM -> PREPARE_SPIN (spin owns normal control)");
        return transition_to(MotionState::PREPARE_SPIN);
    }
    if (in.route_tracked) {
        RCLCPP_INFO(logger_, "FSM -> FOLLOW (has path)");
        return transition_to(MotionState::FOLLOW);
    }
    if (in.has_hold_goal) {
        RCLCPP_INFO(logger_, "FSM -> FIXED (has hold goal)");
        return transition_to(MotionState::FIXED);
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
    if (spin_control_requested(in)) {
        RCLCPP_INFO(logger_, "FSM -> PREPARE_SPIN (spin owns normal control)");
        return transition_to(MotionState::PREPARE_SPIN);
    }
    if (in.route_tracked) {
        RCLCPP_INFO(logger_, "FSM -> FOLLOW (has path)");
        return transition_to(MotionState::FOLLOW);
    }
    if (!in.has_hold_goal) {
        RCLCPP_INFO(logger_, "FSM -> IDLE (no hold goal)");
        return transition_to(MotionState::IDLE);
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

    // COMMITTED 是本周期路径替换锁的最高优先级；即使同时出现普通中断信号，
    // 也必须先进入不可抢占子状态。
    if (in.step_phase == StepPhase::COMMITTED) {
        RCLCPP_INFO(
            logger_, "FSM -> STEPPING/%s (step lifecycle entered)",
            step_phase_str(in.step_phase)
        );
        return enter_stepping(in.step_phase, in.step_path_epoch, in.step_segment_index);
    }

    // stuck-like（follow no-progress 或卡住）统一走脱困链，恢复后 one-shot replan。
    if (in.no_progress_detected || in.is_stuck) {
        replan_after_recovery_ = true;
        pending_reverse_start_time_ = in.stamp;
        pending_reverse_start_pos_ = in.chassis_pos_map;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE (stuck-like)");
        return transition_to(MotionState::STUCK_REVERSE);
    }

    if (spin_control_requested(in)) {
        RCLCPP_INFO(logger_, "FSM -> PREPARE_SPIN (spin owns normal control)");
        return transition_to(MotionState::PREPARE_SPIN);
    }

    if (!in.route_tracked) {
        if (!in.has_active_path && in.is_hazard_now) {
            replan_after_recovery_ = false;
            RCLCPP_WARN(logger_, "FOLLOW path invalidated in hazard; entering HAZARD_RECOVERY");
            return transition_to(MotionState::HAZARD_RECOVERY);
        }
        RCLCPP_INFO(logger_, "FSM routing after path loss");
        return route_to_requested_state(in);
    }

    if (in.reach_goal) {
        // TaskManager consumes this event on the next control cycle. Keep FOLLOW
        // for this cycle so the current MPC command remains continuous until the
        // updated path/hold intent can route directly to the next control state.
        RCLCPP_INFO(logger_, "FSM -> IDLE (goal reached, deferring state transition to next cycle)");
        FsmOutput out{ .state = MotionState::FOLLOW };
        out.goal_reached = true;
        return out;
    }

    // PREPARING/ARMED 与 FOLLOW 共享上述中断策略；仅在本周期没有中断时进入父状态。
    if (is_step_phase_precommit(in.step_phase)) {
        RCLCPP_INFO(
            logger_, "FSM -> STEPPING/%s (step lifecycle entered)",
            step_phase_str(in.step_phase)
        );
        return enter_stepping(in.step_phase, in.step_path_epoch, in.step_segment_index);
    }

    return { .state = MotionState::FOLLOW };
}

// ═══════════════════════════ STEPPING ═══════════════════════

FsmOutput StateMachine::on_stepping(const FsmInput& in) {
    synchronize_step_phase(in.step_phase, in.step_path_epoch, in.step_segment_index);

    if (should_start_resume_hazard_recovery(in)) {
        replan_after_recovery_ = true;
        RCLCPP_WARN(logger_, "FSM -> HAZARD_RECOVERY (resumed from stopped into hazard during STEPPING)");
        const bool cancelled = is_step_phase_precommit(step_phase_);
        FsmOutput out = transition_to(MotionState::HAZARD_RECOVERY);
        out.step_cancelled = cancelled;
        return out;
    }

    // 路径硬失效高于台阶生命周期锁。COMMITTED 只阻止普通抢占，不能继续保护
    // 已无法跟踪的路径；当前位置危险时先完成恢复，再由任务层重规划。
    if (!in.has_active_path) {
        FsmOutput out;
        if (in.is_hazard_now) {
            replan_after_recovery_ = false;
            RCLCPP_WARN(
                logger_, "STEPPING/%s path invalidated in hazard; entering HAZARD_RECOVERY",
                step_phase_str(step_phase_)
            );
            out = transition_to(MotionState::HAZARD_RECOVERY);
        } else {
            RCLCPP_INFO(
                logger_, "FSM routing after STEPPING/%s path cancellation",
                step_phase_str(step_phase_)
            );
            out = route_to_requested_state(in);
        }
        out.step_cancelled = true;
        return out;
    }

    if (in.no_progress_detected || in.is_stuck) {
        replan_after_recovery_ = true;
        pending_reverse_start_time_ = in.stamp;
        pending_reverse_start_pos_ = in.chassis_pos_map;
        RCLCPP_WARN(logger_, "FSM -> STUCK_REVERSE (stuck-like)");
        const bool cancelled = is_step_phase_precommit(step_phase_);
        FsmOutput out = transition_to(MotionState::STUCK_REVERSE);
        out.step_cancelled = cancelled;
        return out;
    }

    // commit 前保持与 FOLLOW 一致的外部打断能力。
    if (is_step_phase_precommit(step_phase_)) {
        if (spin_control_requested(in)) {
            RCLCPP_INFO(
                logger_, "FSM -> PREPARE_SPIN (spin owns control during STEPPING/%s)",
                step_phase_str(step_phase_)
            );
            FsmOutput out = transition_to(MotionState::PREPARE_SPIN);
            out.step_cancelled = true;
            return out;
        }

        if (in.reach_goal) {
            RCLCPP_INFO(
                logger_, "FSM -> IDLE (goal reached during STEPPING/%s)",
                step_phase_str(step_phase_)
            );
            FsmOutput out = transition_to(MotionState::IDLE);
            out.goal_reached = true;
            out.step_cancelled = true;
            return out;
        }
    }

    // RouteMonitor 会把 LOST 作为硬失效并在同周期清除 active_path，因此这里只
    // 处理尚未被任务层确认失效的瞬时无投影输入。
    if (!in.route_tracked) {
        return { .state = MotionState::STEPPING };
    }

    if (is_step_phase_active(in.step_phase)) {
        return { .state = MotionState::STEPPING };
    }

    RCLCPP_INFO(logger_, "STEPPING completed; routing to current control owner");
    return route_to_requested_state(in);
}

// ═══════════════════════════ SPIN ═══════════════════════════

FsmOutput StateMachine::on_spin(const FsmInput& in) {
    if (in.is_hazard_now) {
        replan_after_recovery_ = false;
        RCLCPP_WARN(logger_, "FSM -> HAZARD_RECOVERY (is hazard)");
        return transition_to(MotionState::HAZARD_RECOVERY);
    }

    if (!spin_control_requested(in)) {
        RCLCPP_INFO(logger_, "FSM leaving SPIN for NORMAL control");
        return route_to_normal_state(in);
    }

    return { .state = MotionState::SPIN };
}

// ═══════════════════════════ PREPARE_SPIN ═══════════════════

FsmOutput StateMachine::on_prepare_spin(const FsmInput& in) {
    if (in.is_hazard_now) {
        replan_after_recovery_ = false;
        RCLCPP_WARN(logger_, "PREPARE_SPIN cancelled by hazard; entering HAZARD_RECOVERY");
        return transition_to(MotionState::HAZARD_RECOVERY);
    }
    if (!spin_control_requested(in)) {
        RCLCPP_INFO(logger_, "PREPARE_SPIN cancelled; returning to NORMAL target");
        return route_to_normal_state(in);
    }
    if (prepare_spin_ready(in)) {
        RCLCPP_INFO(logger_, "PREPARE_SPIN ready; transferring control to SPIN");
        // SPIN 的唯一入口：任何请求来源都必须先经过本状态的物理互锁。
        return transition_to(MotionState::SPIN);
    }
    return { .state = MotionState::PREPARE_SPIN };
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
