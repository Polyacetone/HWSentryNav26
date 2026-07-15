#include <nav_executor/path_executor/path_executor.hpp>
#include <nav_executor/path_executor/recovery_helpers.hpp>
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

// ═══════════════════════ 辅助 ════════════════════════════════

void PathExecutor::sync_mpc_context(const ExecutorInput& input, const bool allow_observer_update) {
    mpc_controller_->set_last_cmd(last_cmd_);
    if (allow_observer_update) {
        mpc_controller_->update_observer(input.observation.chassis_state);
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

void PathExecutor::remember_command_output(const ExecutorOutput& output) {
    last_command_output_.velocity = output.velocity;
    last_command_output_.omega = output.omega;
    last_command_output_.mode = output.mode;
    last_command_output_.step_dist_cm = output.step_dist_cm;
    has_last_command_output_ = true;
}

// ═══════════════════════ 主更新接口 ══════════════════════════

ExecutorOutput PathExecutor::update(const ExecutorInput& input) {
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

        last_cmd_ = Eigen::Vector2d::Zero();
        mpc_controller_->reset_warm_start();
        mpc_controller_->reset_observer();
        safety_monitor_.reset_stuck();
        safety_monitor_.reset_recovery();
        remember_command_output(out);
        return out;
    }

    if (entered_controllable) {
        RCLCPP_DEBUG(logger_, "Chassis entered mature control state: resetting Luenberger observer");
        mpc_controller_->reset_observer();
    }
    if (command_blocked) {
        mpc_controller_->reset_observer();
    }

    const MotionState prev_state = last_motion_state_;
    sync_mpc_context(input, chassis_controllable);

    const bool has_bound_path = static_cast<bool>(input.intent.active_path);
    const bool has_path = has_bound_path && input.route
        && input.route->path == input.intent.active_path
        && input.route->status == RouteTrackingStatus::TRACKED;

    // ── path 绑定切换（新的不可变包 → 重置台阶/进度状态）──
    if (input.intent.active_path != bound_path_) {
        bound_path_ = input.intent.active_path;
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

    double current_u = 0.0;
    if (has_path) {
        current_u = input.route->observed_tau;
        if (prev_state == MotionState::FOLLOW || prev_state == MotionState::STEPPING) {
            step_controller_.update_active_segment(current_u);
        }
    }

    // ── stuck-like 检测：follow/stepping 无进度 + 卡住 ──
    bool no_progress_detected = false;
    if (!command_blocked && has_path) {
        if (prev_state == MotionState::FOLLOW) {
            no_progress_detected = progress_monitor_.update_and_check_no_progress(
                current_u, params_.follow_no_progress_guard, MotionState::FOLLOW, prev_state, input.observation.stamp
            );
        } else if (prev_state == MotionState::STEPPING) {
            no_progress_detected = progress_monitor_.update_and_check_no_progress(
                current_u, params_.stepping_no_progress_guard, MotionState::STEPPING, prev_state, input.observation.stamp
            );
        }
    }

    const bool endpoint_reached = has_path
        && ((input.observation.chassis_pose_map.head<2>() - bound_path_->trajectory.position(1.0)).norm() < params_.stop_threshold_dist);
    const bool progress_reached = has_path
        && input.route->remaining_length < params_.stop_threshold_remaining_distance;

    // ── 组装 FSM 输入 ──
    FsmInput fsm_input;
    fsm_input.has_path = has_path;
    fsm_input.has_hold_goal = input.intent.hold_goal.has_value();
    fsm_input.reach_goal = endpoint_reached && progress_reached;
    fsm_input.step_nonpreemptible = has_path && step_controller_.is_step_nonpreemptible(current_u);
    fsm_input.resumed_from_stopped = resumed_from_stopped;
    fsm_input.command_blocked = command_blocked;
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
    fsm_input.is_stuck = !command_blocked && safety_monitor_.check_stuck(
        input.observation.chassis_pose_map.head<2>(), last_cmd_.x(), input.observation.stamp
    );
    fsm_input.is_recovery_safe = !command_blocked
        && input.environment.final_cost_map && input.environment.masked_direction_map
        && safety_monitor_.check_recovery_safe(
            *input.environment.final_cost_map, *input.environment.masked_direction_map, input.observation.chassis_pose_map.head<2>(), input.observation.stamp
        );
    fsm_input.chassis_pos_map = input.observation.chassis_pose_map.head<2>();
    fsm_input.velocity = last_cmd_.x();
    fsm_input.omega = last_cmd_.y();
    fsm_input.stamp = input.observation.stamp;

    const FsmOutput fsm_output = control_fsm_->update(fsm_input);
    const MotionState state = fsm_output.state;
    on_state_transition(prev_state, state, !command_blocked);
    last_motion_state_ = state;

    ExecutorOutput output;
    if (command_blocked) {
        apply_held_command(output);
    } else {
        switch (state) {
            case MotionState::IDLE: output = execute_idle(); break;
            case MotionState::FOLLOW: output = execute_follow(input, true); break;
            case MotionState::SPIN: output = execute_spin(input); break;
            case MotionState::STOPPING: output = execute_stop(input); break;
            case MotionState::HAZARD_RECOVERY: output = execute_recovery(input); break;
            case MotionState::STUCK_REVERSE: output = execute_stuck_reverse(); break;
            case MotionState::FIXED: output = execute_fixed(input); break;
            case MotionState::STEPPING: output = execute_follow(input, false); break;
            case MotionState::DEAD: output = execute_idle(); break;
        }
    }

    output.motion_state = state;
    output.goal_reached = fsm_output.goal_reached;
    output.executor_replan_event = fsm_output.executor_replan_event;
    output.mpc_lethal = mpc_lethal_pending_;
    mpc_lethal_pending_ = false;

    if (output.valid) {
        last_cmd_ = Eigen::Vector2d(output.velocity, output.omega);
        mpc_controller_->set_last_cmd(last_cmd_);
        remember_command_output(output);
        const bool reset_warm_start = !command_blocked
            && (state == MotionState::IDLE || state == MotionState::SPIN || state == MotionState::STUCK_REVERSE);
        if (reset_warm_start) {
            mpc_controller_->reset_warm_start();
        }
    }

    return output;
}

// ═══════════════════ 状态转移副作用 ══════════════════════════

void PathExecutor::on_state_transition(const MotionState prev, const MotionState next, const bool allow_warm_start_reset) {
    if (prev == next) return;

    const bool prev_follow_like = (prev == MotionState::FOLLOW) || (prev == MotionState::STEPPING);
    const bool next_follow_like = (next == MotionState::FOLLOW) || (next == MotionState::STEPPING);

    if (prev_follow_like || next_follow_like) {
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

void assign_hold_output(ExecutorOutput& out, const std::tuple<Eigen::Vector2d, MPCPrediction>& result) {
    out.velocity = std::get<0>(result).x();
    out.omega = std::get<0>(result).y();
    out.mode = chassis_mode::NORMAL;
    out.mpc_path_map = std::get<1>(result).path_map;
    out.predicted_v = std::get<1>(result).v_pred;
    out.predicted_w = std::get<1>(result).w_pred;
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
    const double u0 = input.route->observed_tau;
    const MincoTrajectory& path = input.intent.active_path->trajectory;

    step_controller_.tick_profile_blend();
    const SolveTimer timer;
    const auto result = mpc_controller_->solve_follow(
        path, input.observation.chassis_pose_map, input.observation.chassis_state,
        input.route->phase_time, input.route->phase_rate,
        *input.environment.final_cost_map, *input.environment.masked_global_cost_map, input.environment.per_step_cost_maps, input.environment.prediction_dt,
        step_controller_.current_blended_profile(),
        input.intent.active_path->step_constraint_schedule,
        check_lethal_status
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Follow) solve failed: %s", result.error().c_str());
        return out;
    }
    warn_if_slow_solve(logger_, "Follow", timer.elapsed_ms());

    const auto& follow_result = *result;
    const auto& cmd = follow_result.command;
    const auto& prediction = follow_result.prediction;

    if (follow_result.status == MPCSolver::FollowSolveStatus::STOP_AND_WAIT_REPLAN) {
        // path invalid（MPC_LETHAL）：本周期输出减速指令，标记 one-shot 供顶层 replan。
        mpc_lethal_pending_ = true;
        if (follow_result.lethal_obstacle) {
            const auto& lethal = *follow_result.lethal_obstacle;
            RCLCPP_WARN(
                logger_,
                "Follow rollout entered lethal obstacle at step %d (x=%.2f, y=%.2f, cost=%.1f); flagging MPC_LETHAL",
                lethal.state_index, lethal.position_map.x(), lethal.position_map.y(), lethal.sampled_cost
            );
        } else {
            RCLCPP_WARN(logger_, "Follow rollout entered lethal obstacle; flagging MPC_LETHAL");
        }
    }

    out.velocity = cmd.x();
    out.omega = cmd.y();

    if (follow_result.status == MPCSolver::FollowSolveStatus::STOP_AND_WAIT_REPLAN) {
        out.mode = chassis_mode::NORMAL;
    } else if (const StepChassisCommand* const chassis_command = step_controller_.current_chassis_command(u0);
               chassis_command && step_controller_.should_activate_chassis_mode(u0)) {
        out.mode = chassis_command->mode;
    } else {
        out.mode = chassis_mode::NORMAL;
    }

    out.mpc_path_map = prediction.path_map;
    out.predicted_v = prediction.v_pred;
    out.predicted_w = prediction.w_pred;
    out.step_dist_cm = step_controller_.compute_step_distance_cm(path, u0);
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

// ═══════════════════ STOPPING ════════════════════════════════

ExecutorOutput PathExecutor::execute_stop(const ExecutorInput& input) {
    ExecutorOutput out;
    if (!input.environment.final_cost_map) return out;

    const SolveTimer timer;
    const auto result = mpc_controller_->solve_stop(
        input.observation.chassis_pose_map, input.observation.chassis_state, *input.environment.final_cost_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Stop) solve failed: %s", result.error().c_str());
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
        return out;
    }
    warn_if_slow_solve(logger_, "Fixed", timer.elapsed_ms());
    assign_hold_output(out, *result);
    return out;
}

} // namespace nav_executor
