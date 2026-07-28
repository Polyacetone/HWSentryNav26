#include <nav_executor/path_executor/path_executor.hpp>
#include <nav_executor/path_executor/monitoring/recovery_helpers.hpp>
#include <rclcpp/logging.hpp>

namespace nav_executor {

// ═══════════════════════ 构造函数 ════════════════════════════

PathExecutor::PathExecutor(
    const PathExecutorParams& params,
    const FsmParams& fsm_params,
    std::shared_ptr<MPCSolver> mpc_controller,
    const CapabilityProfile& normal_profile,
    const std::array<CapabilityProfile, 3>& capability_profiles,
    const ProfileBlendParams& blend_params,
    rclcpp::Logger logger
) : control_fsm_(std::make_unique<StateMachine>(fsm_params, logger)),
    mpc_controller_(std::move(mpc_controller)),
    step_controller_(params.step_dist_offset, normal_profile, capability_profiles, blend_params, logger),
    progress_monitor_(logger),
    safety_monitor_(fsm_params, logger),
    logger_(logger),
    params_(params),
    fsm_params_(fsm_params) {
    last_motion_state_ = control_fsm_->state();
}

StepExecutionPreview PathExecutor::preview_step_execution(
    const AnnotatedPath::ConstPtr& path,
    const double path_progress,
    const bool route_tracked
) const {
    const MotionState state = control_fsm_->state();
    const StepPhase latched_phase = control_fsm_->step_phase();

    // 投影无效时不能把“未观测到阶段”解释成 release。保留 FSM 已锁存策略，
    // 由 RouteMonitor 在可抢占阶段决定是否清除 active_path。
    if (!path || !route_tracked) {
        return {
            .phase = latched_phase,
            .preemptible = preemptible(),
        };
    }

    const bool same_path = path == bound_path_;
    const StepPhaseObservation observed = same_path
        ? step_controller_.observe_step_phase(path_progress)
        : classify_step_phase(*path, path_progress);

    StepPhase effective_phase = observed.phase;
    const bool same_lifecycle = state == MotionState::STEPPING
        && same_path
        && bound_path_epoch_ == control_fsm_->step_path_epoch()
        && observed.segment_index == control_fsm_->step_segment_index();
    if (same_lifecycle
        && static_cast<uint8_t>(latched_phase) > static_cast<uint8_t>(effective_phase)) {
        effective_phase = latched_phase;
    }

    const bool can_preempt = state == MotionState::STEPPING
        ? effective_phase != StepPhase::COMMITTED
        : preemptible() && effective_phase != StepPhase::COMMITTED;
    return {
        .phase = effective_phase,
        .preemptible = can_preempt,
    };
}

// ═══════════════════════ 辅助 ════════════════════════════════

void PathExecutor::sync_mpc_context(const ExecutorInput& input, const bool allow_observer_update) {
    mpc_controller_->set_command_state(mpc_command_state_, mpc_command_rate_);
    if (!allow_observer_update) {
        return;
    }

    const uint64_t sequence = input.observation.chassis_state_sequence;
    if (sequence == 0) {
        mpc_controller_->update_observer(input.observation.chassis_state, sequence);
        return;
    }
    if (last_observer_state_sequence_ && sequence == *last_observer_state_sequence_) return;
    if (last_observer_state_sequence_ && sequence != *last_observer_state_sequence_ + 1) {
        reset_mpc_observer(ObserverResetReason::STATE_SEQUENCE_GAP);
    }
    mpc_controller_->update_observer(input.observation.chassis_state, sequence);
    last_observer_state_sequence_ = sequence;
}

void PathExecutor::reset_mpc_observer(const ObserverResetReason reason) {
    mpc_controller_->reset_observer(reason);
    last_observer_state_sequence_.reset();
}

void PathExecutor::reanchor_mpc_command_state(const ChassisMotionState& chassis_state) {
    mpc_command_state_ = {chassis_state.velocity, chassis_state.omega};
    mpc_command_rate_.setZero();
    mpc_controller_->set_command_state(mpc_command_state_, mpc_command_rate_);
    mpc_controller_->reset_warm_start();
    reset_mpc_observer(ObserverResetReason::COMMAND_RESYNCHRONIZED);
}

void PathExecutor::invalidate_mpc_command_history(const ObserverResetReason reason) {
    if (mpc_command_history_ == MpcCommandHistory::NEEDS_REANCHOR) return;
    mpc_command_history_ = MpcCommandHistory::NEEDS_REANCHOR;
    mpc_controller_->reset_warm_start();
    reset_mpc_observer(reason);
}

bool PathExecutor::state_uses_mpc(const MotionState state) {
    switch (state) {
        case MotionState::FOLLOW:
        case MotionState::STEPPING:
        case MotionState::PREPARE_SPIN:
        case MotionState::HAZARD_RECOVERY:
        case MotionState::FIXED:
            return true;
        default:
            return false;
    }
}

void PathExecutor::apply_held_command(ExecutorOutput& output) const {
    if (!has_last_command_output_) {
        output.velocity = 0.0;
        output.omega = 0.0;
        output.mode = chassis_mode::NORMAL;
        output.step_dist_cm = 0;
        output.valid = true;
        return;
    }
    output.velocity = last_command_output_.velocity;
    output.omega = last_command_output_.omega;
    output.mode = last_command_output_.mode;
    output.step_dist_cm = last_command_output_.step_dist_cm;
    output.valid = true;
}

void PathExecutor::remember_command_output(
    const ExecutorOutput& output,
    const std::chrono::steady_clock::time_point stamp
) {
    last_command_output_.velocity = output.velocity;
    last_command_output_.omega = output.omega;
    last_command_output_.mode = output.mode;
    last_command_output_.step_dist_cm = output.step_dist_cm;
    has_last_command_output_ = true;
    last_command_output_stamp_ = stamp;
}

// ═══════════════════════ 主更新接口 ══════════════════════════

ExecutorOutput PathExecutor::update(const ExecutorInput& input) {
    if (!last_command_output_stamp_) {
        mpc_command_history_ = MpcCommandHistory::NEEDS_REANCHOR;
    } else {
        const double command_output_interval = std::chrono::duration<double>(
            input.observation.stamp - *last_command_output_stamp_
        ).count();
        if (command_output_interval >= params_.command_history_timeout) {
            RCLCPP_WARN(
                logger_,
                "Command output gap %.3f s reached the %.3f s history timeout; resynchronizing from chassis feedback",
                command_output_interval,
                params_.command_history_timeout
            );
            invalidate_mpc_command_history(ObserverResetReason::COMMAND_RESYNCHRONIZED);
        }
    }
    if (last_update_stamp_) {
        const double interval = std::chrono::duration<double>(
            input.observation.stamp - *last_update_stamp_
        ).count();
        if (interval > 1.5 * MPC_DT) {
            mpc_command_rate_.setZero();
            invalidate_mpc_command_history(ObserverResetReason::CONTROL_UPDATE_GAP);
        }
    }
    last_update_stamp_ = input.observation.stamp;

    uint8_t leg_mode = input.observation.chassis_leg_mode;
    if (leg_mode > static_cast<uint8_t>(LegMode::ABNORMAL)) {
        RCLCPP_ERROR(logger_, "Invalid chassis_leg_mode=%hhu (expected 0~6), treating as DEAD", leg_mode);
        leg_mode = static_cast<uint8_t>(LegMode::DEAD);
    }
    const ChassisControlState chassis_control_state = classify_chassis_control_state(leg_mode, input.observation.comp_stage);
    const bool chassis_dead = chassis_control_state == ChassisControlState::STOPPED;
    const bool command_blocked = chassis_control_state == ChassisControlState::BLOCKED;
    const bool chassis_controllable = chassis_control_state == ChassisControlState::NORMAL;
    const bool resumed_from_stopped = chassis_controllable && last_cycle_chassis_control_state_ == ChassisControlState::STOPPED;
    const bool entered_controllable = chassis_controllable && !last_cycle_chassis_controllable_;
    last_cycle_chassis_control_state_ = chassis_control_state;
    last_cycle_chassis_controllable_ = chassis_controllable;

    // 全局中断优先：底盘 Dead 直接外部拦截，不进入 FSM。
    if (chassis_dead) {
        ExecutorOutput out;
        out.velocity = 0.0;
        out.omega = 0.0;
        out.mode = chassis_mode::NORMAL;
        out.step_dist_cm = 0;
        out.motion_state = MotionState::DEAD;
        out.valid = true;

        invalidate_mpc_command_history(ObserverResetReason::CONTROL_UNAVAILABLE);
        safety_monitor_.reset_stuck();
        safety_monitor_.reset_recovery();
        out.observer_diagnostics = mpc_controller_->observer_diagnostics();
        remember_command_output(out, input.observation.stamp);
        return out;
    }

    if (entered_controllable) {
        RCLCPP_DEBUG(logger_, "Chassis entered mature control state: resetting Luenberger observer");
        reset_mpc_observer(ObserverResetReason::EXPLICIT_REQUEST);
    }
    if (command_blocked) {
        invalidate_mpc_command_history(ObserverResetReason::CONTROL_UNAVAILABLE);
    }

    const MotionState prev_state = last_motion_state_;
    const bool observer_update_allowed = chassis_controllable
        && mpc_command_history_ == MpcCommandHistory::TRACKED
        && has_last_command_output_
        && last_command_output_.mode == chassis_mode::NORMAL;
    sync_mpc_context(input, observer_update_allowed);

    const bool has_active_path = static_cast<bool>(input.intent.active_path);
    const bool has_path = has_active_path && input.route
        && input.route->path == input.intent.active_path
        && input.route->status == RouteTrackingStatus::TRACKED;

    // ── path 绑定切换（新的不可变包 → 重置台阶/进度状态）──
    if (input.intent.active_path != bound_path_) {
        bound_path_ = input.intent.active_path;
        ++bound_path_epoch_;
        step_controller_.set_path(bound_path_);
        progress_monitor_.reset();
        mpc_controller_->reset_warm_start();
    }

    if (prev_state == MotionState::HAZARD_RECOVERY
        && input.environment.masked_global_cost_map && input.environment.masked_direction_map) {
        safety_monitor_.update_recovery_goal_if_needed(
            input.observation.chassis_pose_map,
            *input.environment.masked_global_cost_map,
            *input.environment.masked_direction_map,
            input.environment.base_direction_map,
            input.observation.stamp
        );
    }

    double current_progress = 0.0;
    if (has_path) {
        current_progress = input.route->arc_length;
        if (prev_state == MotionState::FOLLOW || prev_state == MotionState::STEPPING) {
            step_controller_.update_active_segment(current_progress);
        }
    }

    const StepPhaseObservation step_observation = has_path
        ? step_controller_.observe_step_phase(current_progress)
        : StepPhaseObservation {};
    const StepExecutionPreview current_step_preview = preview_step_execution(
        input.intent.active_path,
        current_progress,
        has_path
    );

    // ── stuck-like 检测：commit 前沿用 follow，commit 后使用 stepping 策略 ──
    bool no_progress_detected = false;
    if (!command_blocked && has_path
        && (prev_state == MotionState::FOLLOW || prev_state == MotionState::STEPPING)) {
        if (current_step_preview.phase != StepPhase::COMMITTED) {
            no_progress_detected = progress_monitor_.update_and_check_no_progress(
                input.route->arc_length, params_.follow_no_progress_guard,
                MotionState::FOLLOW, prev_state, input.observation.stamp
            );
        } else {
            no_progress_detected = progress_monitor_.update_and_check_no_progress(
                input.route->arc_length, params_.stepping_no_progress_guard,
                MotionState::STEPPING, prev_state, input.observation.stamp
            );
        }
    }

    const bool endpoint_reached = has_path
        && ((input.observation.chassis_pose_map.head<2>() - bound_path_->trajectory.position(1.0)).norm() < params_.stop_threshold_dist);
    const bool progress_reached = has_path
        && input.route->remaining_length < params_.stop_threshold_remaining_distance;

    // ── 组装 FSM 输入 ──
    FsmInput fsm_input;
    fsm_input.has_active_path = has_active_path;
    fsm_input.route_tracked = has_path;
    fsm_input.has_hold_goal = input.intent.hold_goal.has_value();
    fsm_input.reach_goal = endpoint_reached && progress_reached;
    fsm_input.step_phase = step_observation.phase;
    fsm_input.step_path_epoch = bound_path_epoch_;
    fsm_input.step_segment_index = step_observation.segment_index;
    fsm_input.resumed_from_stopped = resumed_from_stopped;
    fsm_input.command_blocked = command_blocked;
    fsm_input.command_state_tracked =
        mpc_command_history_ == MpcCommandHistory::TRACKED;
    fsm_input.spin_requested = input.intent.spin_requested;
    fsm_input.spin_high_priority = input.intent.spin_high_priority;
    last_spin_high_priority_ = input.intent.spin_high_priority;
    fsm_input.no_progress_detected = no_progress_detected;

    const bool is_hazard_now = !command_blocked
        && input.environment.masked_global_cost_map && input.environment.masked_direction_map
        && safety_monitor_.compute_is_hazard(
            *input.environment.masked_global_cost_map, *input.environment.masked_direction_map, input.observation.chassis_pose_map.head<2>()
        );
    fsm_input.is_hazard_now = is_hazard_now;
    const double published_command_velocity = has_last_command_output_
        ? last_command_output_.velocity
        : 0.0;
    fsm_input.is_stuck = !command_blocked && safety_monitor_.check_stuck(
        input.observation.chassis_pose_map.head<2>(), published_command_velocity,
        input.observation.stamp
    );
    fsm_input.is_recovery_safe = !command_blocked
        && input.environment.final_cost_map && input.environment.masked_direction_map
        && safety_monitor_.check_recovery_safe(
            *input.environment.final_cost_map, *input.environment.masked_direction_map, input.observation.chassis_pose_map.head<2>(), input.observation.stamp
        );
    fsm_input.chassis_pos_map = input.observation.chassis_pose_map.head<2>();
    fsm_input.command_velocity = mpc_command_state_.x();
    fsm_input.command_omega = mpc_command_state_.y();
    fsm_input.measured_velocity = input.observation.chassis_state.velocity;
    fsm_input.measured_omega = input.observation.chassis_state.omega;
    fsm_input.stamp = input.observation.stamp;

    const FsmOutput fsm_output = control_fsm_->update(fsm_input);
    const MotionState state = fsm_output.state;
    on_state_transition(prev_state, state, !command_blocked);
    last_motion_state_ = state;

    if (!command_blocked && state_uses_mpc(state)
        && mpc_command_history_ == MpcCommandHistory::NEEDS_REANCHOR) {
        reanchor_mpc_command_state(input.observation.chassis_state);
    }

    ExecutorOutput output;
    if (command_blocked) {
        apply_held_command(output);
    } else {
        switch (state) {
            case MotionState::IDLE: output = execute_idle(); break;
            case MotionState::FOLLOW: output = execute_follow(input, true); break;
            case MotionState::SPIN: output = execute_spin(input); break;
            case MotionState::PREPARE_SPIN: output = execute_prepare_spin(input); break;
            case MotionState::HAZARD_RECOVERY: output = execute_recovery(input); break;
            case MotionState::STUCK_REVERSE: output = execute_stuck_reverse(); break;
            case MotionState::FIXED: output = execute_fixed(input); break;
            case MotionState::STEPPING:
                if (control_fsm_->step_phase() == StepPhase::COMMITTED
                    && has_active_path && !has_path) {
                    // COMMITTED 中的短暂投影丢失不能停止刷新台阶提示。沿用上一帧
                    // 已验证命令，等待 RouteTracker 恢复，而不是产生 invalid 输出。
                    apply_held_command(output);
                } else {
                    output = execute_follow(
                        input,
                        is_step_phase_precommit(control_fsm_->step_phase())
                    );
                }
                break;
            case MotionState::DEAD: output = execute_idle(); break;
        }
    }

    // 台阶计划在 commit 前被打断时，取消 mode 的优先级高于 MPC 输出是否成功。
    // 台阶取消必须确定地撤销下位机台阶模式；必要时使用零速 NORMAL fallback。
    if (fsm_output.step_cancelled) {
        output.mode = chassis_mode::NORMAL;
        output.step_dist_cm = 0;
        if (command_blocked) {
            // BLOCKED 分支默认复用上一帧完整命令；取消事件不能继续携带旧台阶速度意图。
            output.velocity = 0.0;
            output.omega = 0.0;
            output.valid = true;
        } else if (!output.valid) {
            RCLCPP_ERROR(logger_, "Step cancellation controller failed; publishing zero NORMAL fallback");
            output.velocity = 0.0;
            output.omega = 0.0;
            output.valid = true;
        }
    }

    // held-command 分支可能跨越 release 或其他状态转换复用上一帧完整命令。
    // 一旦 FSM 不再处于 ARMED/COMMITTED，旧台阶提示便不再有执行权，必须独立于
    // 求解器是否运行而改写为 NORMAL，防止 release 后继续重发 STEP mode。
    const bool previous_step_mode_still_held = has_last_command_output_
        && is_step_mode(last_command_output_.mode);
    const bool step_mode_no_longer_authorized =
        !step_phase_activates_chassis_mode(control_fsm_->step_phase())
        && (is_step_mode(output.mode) || (!output.valid && previous_step_mode_still_held));
    if (step_mode_no_longer_authorized) {
        output.mode = chassis_mode::NORMAL;
        output.step_dist_cm = 0;
        if (!output.valid) {
            RCLCPP_ERROR(logger_, "Step release controller failed; publishing zero NORMAL fallback");
            output.velocity = 0.0;
            output.omega = 0.0;
            output.valid = true;
        }
    }

    output.motion_state = state;
    output.step_phase = control_fsm_->step_phase();
    output.goal_reached = fsm_output.goal_reached;
    output.executor_replan_event = fsm_output.executor_replan_event;
    output.mpc_lethal = mpc_lethal_pending_;
    mpc_lethal_pending_ = false;

    if (output.valid) {
        remember_command_output(output, input.observation.stamp);
        if (output.mpc_generated_command && !command_blocked) {
            const Eigen::Vector2d command(output.velocity, output.omega);
            mpc_command_rate_ = (command - mpc_command_state_) / MPC_DT;
            mpc_command_state_ = command;
            mpc_command_history_ = MpcCommandHistory::TRACKED;
            mpc_controller_->set_command_state(
                mpc_command_state_, mpc_command_rate_
            );
        } else {
            invalidate_mpc_command_history(ObserverResetReason::CONTROL_UNAVAILABLE);
            mpc_controller_->reset_warm_start();
        }
    } else {
        mpc_command_rate_.setZero();
        invalidate_mpc_command_history(ObserverResetReason::CONTROL_OUTPUT_INVALID);
        mpc_controller_->set_command_state(
            mpc_command_state_, mpc_command_rate_
        );
    }

    output.observer_diagnostics = mpc_controller_->observer_diagnostics();
    return output;
}

// ═══════════════════ 状态转移副作用 ══════════════════════════

void PathExecutor::on_state_transition(const MotionState prev, const MotionState next, const bool allow_warm_start_reset) {
    if (prev == next) return;

    const bool prev_follow_like = (prev == MotionState::FOLLOW) || (prev == MotionState::STEPPING);
    const bool next_follow_like = (next == MotionState::FOLLOW) || (next == MotionState::STEPPING);

    if (prev_follow_like != next_follow_like) {
        progress_monitor_.reset();
    }

    if (prev == MotionState::STUCK_REVERSE || prev == MotionState::HAZARD_RECOVERY) {
        safety_monitor_.reset_stuck();
    }

    if (prev_follow_like && !next_follow_like) {
        mpc_lethal_pending_ = false;
        if (allow_warm_start_reset) {
            mpc_controller_->reset_warm_start();
        }
    }

    if (prev == MotionState::HAZARD_RECOVERY && next != MotionState::HAZARD_RECOVERY) {
        safety_monitor_.reset_recovery();
    }

    const bool next_uses_hold = (next == MotionState::FIXED);
    const bool prev_uses_hold = (prev == MotionState::FIXED) || (prev == MotionState::HAZARD_RECOVERY);
    if (allow_warm_start_reset && next_uses_hold && !prev_uses_hold) {
        mpc_controller_->reset_warm_start();
    }
}

// ═══════════════════ 辅助（求解计时）══════════════════════════

namespace {

struct SolveTimer {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
};

void warn_if_slow_solve(rclcpp::Logger logger, const char* name, double solve_ms) {
    constexpr double threshold_ms = MPC_DT * 600.0;
    if (solve_ms > threshold_ms) {
        RCLCPP_WARN(logger, "MPCSolver(%s) solve time %.2f ms > %.2f ms", name, solve_ms, threshold_ms);
    }
}

MPCDiagnostics failed_diagnostics(
    const MPCSolverMode mode,
    const std::string& error,
    const ChassisMotionState& chassis_state,
    const Eigen::Vector2d& previous_command,
    const bool ancillary_enabled = false
) {
    MPCDiagnostics diagnostics;
    diagnostics.solver_mode = mode;
    diagnostics.solve_error = error;
    diagnostics.ancillary_enabled = ancillary_enabled;
    diagnostics.measured_velocity = {chassis_state.velocity, chassis_state.omega};
    diagnostics.previous_command = previous_command;
    return diagnostics;
}

void assign_hold_output(ExecutorOutput& out, const MPCSolver::SolveResult& result) {
    out.velocity = result.command.x();
    out.omega = result.command.y();
    out.mode = chassis_mode::NORMAL;
    out.mpc_path_map = result.diagnostics.applied_prediction.path_map;
    out.mpc_diagnostics = result.diagnostics;
    out.mpc_generated_command = true;
    out.valid = true;
}

} // anonymous namespace

// ═══════════════════ IDLE ════════════════════════════════════

ExecutorOutput PathExecutor::execute_idle() {
    ExecutorOutput out;
    out.velocity = 0.0;
    out.omega = 0.0;
    out.mode = chassis_mode::NORMAL;
    out.valid = true;
    return out;
}

// ═══════════════════ FOLLOW ══════════════════════════════════

ExecutorOutput PathExecutor::execute_follow(const ExecutorInput& input, bool check_lethal_status) {
    ExecutorOutput out;
    if (!input.intent.active_path || !input.environment.final_cost_map || !input.environment.masked_global_cost_map || !input.environment.masked_direction_map) {
        return out;
    }

    if (!input.route || input.route->status != RouteTrackingStatus::TRACKED) return out;
    const double path_progress = input.route->arc_length;
    const MincoTrajectory& path = input.intent.active_path->trajectory;

    step_controller_.tick_profile_blend();
    const SolveTimer timer;
    const auto result = mpc_controller_->solve_follow(
        path, input.intent.active_path->speed_profile,
        input.observation.chassis_pose_map, input.observation.chassis_state,
        input.route->arc_length, input.route->path_speed,
        *input.environment.final_cost_map, *input.environment.masked_global_cost_map, input.environment.per_step_cost_maps, input.environment.prediction_dt,
        step_controller_.effective_capability(),
        input.intent.active_path->step_constraint_schedule,
        check_lethal_status
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Follow) solve failed: %s", result.error().c_str());
        out.mpc_diagnostics = failed_diagnostics(
            MPCSolverMode::FOLLOW,
            result.error(),
            input.observation.chassis_state,
            mpc_command_state_,
            mpc_controller_->params().follow.ancillary_feedback.enable
        );
        return out;
    }
    warn_if_slow_solve(logger_, "Follow", timer.elapsed_ms());

    const auto& follow_result = *result;
    const auto& cmd = follow_result.command;
    const auto& diagnostics = follow_result.diagnostics;

    // rollout 命中致命障碍：该 rollout 已被求解器拒绝，本周期下发的是 STOP 命令。
    // 只有连续拒绝达到阈值才把 MPC_LETHAL 上报给顶层触发重规划。
    const bool rollout_rejected =
        follow_result.status != MPCSolver::FollowSolveStatus::FOLLOW;
    if (rollout_rejected) {
        const bool replan = follow_result.status
            == MPCSolver::FollowSolveStatus::STOP_AND_WAIT_REPLAN;
        if (replan) mpc_lethal_pending_ = true;
        if (follow_result.lethal_obstacle) {
            const auto& lethal = *follow_result.lethal_obstacle;
            RCLCPP_WARN(
                logger_,
                "Rejected Follow rollout: lethal obstacle at step %d (x=%.2f, y=%.2f, cost=%.1f); "
                "commanding stop%s",
                lethal.state_index, lethal.position_map.x(), lethal.position_map.y(),
                lethal.sampled_cost, replan ? " and flagging MPC_LETHAL" : ""
            );
        } else {
            RCLCPP_WARN(
                logger_, "Rejected Follow rollout: lethal obstacle; commanding stop%s",
                replan ? " and flagging MPC_LETHAL" : ""
            );
        }
    }

    out.velocity = cmd.x();
    out.omega = cmd.y();

    if (rollout_rejected) {
        out.mode = chassis_mode::NORMAL;
    } else if (const StepChassisCommand* const chassis_command = step_controller_.current_chassis_command(path_progress);
               chassis_command && step_phase_activates_chassis_mode(control_fsm_->step_phase())) {
        out.mode = chassis_command->mode;
    } else {
        out.mode = chassis_mode::NORMAL;
    }

    out.mpc_path_map = diagnostics.applied_prediction.path_map;
    out.mpc_diagnostics = diagnostics;
    out.mpc_generated_command = true;

    out.step_dist_cm = step_controller_.compute_step_distance_cm(path_progress);
    out.valid = true;

    return out;
}

// ═══════════════════ SPIN ════════════════════════════════════

ExecutorOutput PathExecutor::execute_spin(const ExecutorInput& input) {
    ExecutorOutput out;
    out.mode = input.intent.spin_fast ? chassis_mode::SPIN_FAST : chassis_mode::SPIN_SLOW;
    out.valid = true;
    return out;
}

// ═══════════════════ PREPARE_SPIN ════════════════════════════

ExecutorOutput PathExecutor::execute_prepare_spin(const ExecutorInput& input) {
    ExecutorOutput out;
    if (!input.environment.final_cost_map) return out;

    const SolveTimer timer;
    const auto result = mpc_controller_->solve_stop(
        input.observation.chassis_pose_map, input.observation.chassis_state, *input.environment.final_cost_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Stop) solve failed: %s", result.error().c_str());
        out.mpc_diagnostics = failed_diagnostics(
            MPCSolverMode::STOP,
            result.error(),
            input.observation.chassis_state,
            mpc_command_state_
        );
        return out;
    }
    warn_if_slow_solve(logger_, "Stop", timer.elapsed_ms());
    assign_hold_output(out, *result);
    return out;
}

// ═══════════════════ HAZARD_RECOVERY ═════════════════════════

ExecutorOutput PathExecutor::execute_recovery(const ExecutorInput& input) {
    ExecutorOutput out;
    if (!input.environment.final_cost_map || !input.environment.masked_direction_map) return out;

    if (input.environment.masked_global_cost_map) {
        safety_monitor_.update_recovery_goal_if_needed(
            input.observation.chassis_pose_map,
            *input.environment.masked_global_cost_map,
            *input.environment.masked_direction_map,
            input.environment.base_direction_map,
            input.observation.stamp
        );
    }

    const auto& recovery_goal = safety_monitor_.recovery_goal();
    if (!recovery_goal) {
        out.velocity = 0.0;
        out.omega = 0.0;
        out.mode = chassis_mode::NORMAL;
        out.valid = true;
        return out;
    }

    const SolveTimer timer;
    const auto result = mpc_controller_->solve_hold(
        *recovery_goal, input.observation.chassis_pose_map, input.observation.chassis_state, *input.environment.final_cost_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Recovery) solve failed: %s", result.error().c_str());
        out.mpc_diagnostics = failed_diagnostics(
            MPCSolverMode::HOLD,
            result.error(),
            input.observation.chassis_state,
            mpc_command_state_
        );
        return out;
    }
    warn_if_slow_solve(logger_, "Recovery", timer.elapsed_ms());
    assign_hold_output(out, *result);
    return out;
}

// ═══════════════════ STUCK_REVERSE ═══════════════════════════

ExecutorOutput PathExecutor::execute_stuck_reverse() {
    ExecutorOutput out;
    out.velocity = -fsm_params_.stuck.reverse_speed;
    out.omega = 0.0;
    out.mode = chassis_mode::NORMAL;
    out.valid = true;
    return out;
}

// ═══════════════════ FIXED ═══════════════════════════════════

ExecutorOutput PathExecutor::execute_fixed(const ExecutorInput& input) {
    ExecutorOutput out;
    if (!input.environment.final_cost_map || !input.environment.masked_direction_map || !input.intent.hold_goal) return out;

    const SolveTimer timer;
    const auto result = mpc_controller_->solve_hold(
        *input.intent.hold_goal, input.observation.chassis_pose_map, input.observation.chassis_state, *input.environment.final_cost_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Fixed) solve failed: %s", result.error().c_str());
        out.mpc_diagnostics = failed_diagnostics(
            MPCSolverMode::HOLD,
            result.error(),
            input.observation.chassis_state,
            mpc_command_state_
        );
        return out;
    }
    warn_if_slow_solve(logger_, "Fixed", timer.elapsed_ms());
    assign_hold_output(out, *result);
    return out;
}

} // namespace nav_executor
