#include <path_follower/main_controller.hpp>
#include <path_follower/utils.hpp>
#include <rclcpp/logging.hpp>

namespace path_follower {

// ═══════════════════════ 构造函数 ════════════════════════════

MainController::MainController(
    const NavigationParams& nav_params,
    const FsmParams& fsm_params,
    std::shared_ptr<MPCSolver> mpc_controller,
    const CapabilityProfile& normal_profile,
    const std::array<CapabilityProfile, 3>& capability_profiles,
    const ProfileBlendParams& blend_params,
    rclcpp::Logger logger
) : control_fsm_(std::make_unique<StateMachine>(fsm_params, logger)),
    mpc_controller_(std::move(mpc_controller)),
    step_controller_(nav_params.step_detection, nav_params.step_block_replan, nav_params.step_dist_offset, normal_profile, capability_profiles, blend_params, logger),
    progress_monitor_(logger),
    safety_monitor_(fsm_params, logger),
    logger_(logger),
    nav_params_(nav_params),
    fsm_params_(fsm_params) {
    last_fsm_state_ = control_fsm_->state();
}

void MainController::sync_mpc_context(const ControlInput& input, const bool allow_observer_update) {
    mpc_controller_->set_last_cmd(last_cmd_);

    if (allow_observer_update) {
        mpc_controller_->update_observer(input.chassis_state);
    }

    mpc_controller_->set_energy_state(input.remaining_energy, input.rfr_pwr_limit);
}

void MainController::reset_all_mpc_warm_start() {
    mpc_controller_->reset_warm_start();
}

void MainController::reset_all_mpc_observer() {
    mpc_controller_->reset_observer();
}

void MainController::apply_held_command(ControlOutput& output) const {
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

void MainController::remember_command_output(const ControlOutput& output) {
    last_command_output_.velocity = output.velocity;
    last_command_output_.omega = output.omega;
    last_command_output_.mode = output.mode;
    last_command_output_.step_dist_cm = output.step_dist_cm;
    has_last_command_output_ = true;
}

double MainController::project_path_u(const ControlInput& input, const SplinePath& path, const double seed_u) const {
    return path.project_extrapolated(
        input.chassis_pose_map.head<2>(),
        seed_u,
        mpc_controller_->params().follow.projection.proj_num_samples,
        mpc_controller_->params().follow.projection.proj_search_window,
        mpc_controller_->params().follow.projection.local_search_lazy_distance
    );
}

bool MainController::check_follow_projection_guard(const ControlInput& input, const SplinePath& path, const double current_u) const {
    const Eigen::Vector2d pos_map = input.chassis_pose_map.head<2>();
    const Eigen::Vector2d proj_map = path.position(current_u);
    const double proj_dist = (proj_map - pos_map).norm();
    if (nav_params_.follow_proj_guard.dist_max > 0.0 && proj_dist > nav_params_.follow_proj_guard.dist_max) {
        RCLCPP_WARN(
            logger_,
            "Follow replan: projection too far (dist=%.2f m > %.2f m)",
            proj_dist,
            nav_params_.follow_proj_guard.dist_max
        );
        return true;
    }

    if (!input.masked_global_cost_map || nav_params_.follow_proj_guard.cost_max < 0.0 || nav_params_.follow_proj_guard.cost_max >= 255.0) {
        return false;
    }

    const auto max_cost = recovery_helpers::max_cost_along_segment(
        *input.masked_global_cost_map,
        pos_map,
        proj_map,
        nav_params_.follow_proj_guard.cost_samples
    );
    if (!max_cost) {
        RCLCPP_WARN(logger_, "Follow replan: projection segment out of masked_global_cost_map bounds");
        return true;
    }

    if (*max_cost > nav_params_.follow_proj_guard.cost_max) {
        RCLCPP_WARN(
            logger_,
            "Follow replan: projection segment cost too high (max_cost=%.1f > %.1f)",
            *max_cost,
            nav_params_.follow_proj_guard.cost_max
        );
        return true;
    }

    return false;
}

// ═══════════════════════ 主更新接口 ══════════════════════════

ControlOutput MainController::update(const ControlInput& input) {
    const ChassisControlState chassis_control_state = classify_chassis_control_state(input.chassis_leg_mode, input.comp_stage);
    const bool chassis_dead = chassis_control_state == ChassisControlState::STOPPED;
    const bool command_blocked = chassis_control_state == ChassisControlState::BLOCKED;
    const bool chassis_controllable = chassis_control_state == ChassisControlState::NORMAL;
    const bool entered_controllable = chassis_controllable && !last_cycle_chassis_controllable_;
    last_cycle_chassis_controllable_ = chassis_controllable;

    // 全局中断优先：底盘 Dead 直接外部拦截，不进入 FSM。
    if (chassis_dead) {
        ControlOutput out;
        out.velocity = 0.0;
        out.omega = 0.0;
        out.mode = chassis_mode::NORMAL;
        out.step_dist_cm = 0;
        out.fsm_state = FsmState::DEAD;
        out.valid = true;

        last_cmd_ = Eigen::Vector2d::Zero();
        reset_all_mpc_warm_start();
        reset_all_mpc_observer();
        safety_monitor_.reset_stuck();
        safety_monitor_.reset_recovery();
        remember_command_output(out);
        return out;
    }

    if (entered_controllable) {
        RCLCPP_DEBUG(logger_, "Chassis entered mature control state: resetting Luenberger observer");
        reset_all_mpc_observer();
    }
    if (command_blocked) {
        reset_all_mpc_observer();
    }

    // ─── 路径锁存替换 ───
    ControlInput effective_input = input;
    bool had_deferred_update = false;
    const bool step_path_locked = step_controller_.is_path_locked();
    if (step_path_locked) {
        if (input.path_updated || !input.global_path.has_value()) {
            step_controller_.set_deferred_update();
        }
        effective_input.global_path = step_controller_.locked_path();
        effective_input.path_updated = false;
        effective_input.fixed_goal = step_controller_.locked_fixed_goal();
        effective_input.fixed_goal_pos = step_controller_.locked_fixed_goal_pos();
    } else {
        had_deferred_update = step_controller_.consume_deferred_update();
        effective_input.path_updated = input.path_updated || had_deferred_update;
    }

    const FsmState prev_state = last_fsm_state_;
    sync_mpc_context(effective_input, chassis_controllable);

    if (prev_state == FsmState::HAZARD_RECOVERY
        && effective_input.masked_global_cost_map
        && effective_input.masked_direction_map
    ) {
        safety_monitor_.update_recovery_goal_if_needed(
            effective_input.chassis_pose_map,
            *effective_input.masked_global_cost_map,
            *effective_input.masked_direction_map,
            effective_input.base_direction_map,
            effective_input.stamp
        );
    }

    const bool has_path = effective_input.global_path.has_value();
    const bool has_new_path = has_path && effective_input.path_updated;

    step_controller_.update_plan_for_path_change(
        has_new_path, effective_input.global_path, effective_input.base_direction_map
    );
    if (!has_path) {
        step_controller_.clear_plan();
    }

    if (has_new_path) {
        const double spacing = std::max(0.1, std::min(
            nav_params_.follow_no_progress_guard.landmark_spacing,
            nav_params_.stepping_no_progress_guard.landmark_spacing
        ));
        progress_monitor_.recompute_landmarks(*effective_input.global_path, spacing);
    } else if (!has_path) {
        progress_monitor_.clear();
    }

    const double current_u = has_path ? project_path_u(effective_input, *effective_input.global_path, last_reference_u_) : 0.0;
    if (has_path) {
        last_reference_u_ = current_u;
        if (prev_state == FsmState::FOLLOW || prev_state == FsmState::STEPPING) {
            step_controller_.update_active_segment(
                current_u,
                effective_input.global_path,
                effective_input.fixed_goal,
                effective_input.fixed_goal_pos
            );
        } else {
            step_controller_.clear_runtime_state();
        }
    }

    bool request_replan_now = follow_stop_and_wait_replan_pending_;
    if (request_replan_now) {
        follow_stop_and_wait_replan_pending_ = false;
    }
    if (!command_blocked && has_path && prev_state == FsmState::FOLLOW) {
        request_replan_now = request_replan_now
            || check_follow_projection_guard(effective_input, *effective_input.global_path, current_u)
            || step_controller_.check_block_replan(
                *effective_input.global_path,
                current_u,
                effective_input.masked_direction_map,
                effective_input.current_dynamic_cost_map,
                effective_input.per_step_dynamic_cost_maps
            );
    }

    // 无进度检测（Follow / Stepping 模式均使用统一的路标点方式，参数不同）
    bool no_progress_detected = false;
    if (!command_blocked && has_path) {
        if (prev_state == FsmState::FOLLOW) {
            no_progress_detected = progress_monitor_.check_no_progress(
                current_u, nav_params_.follow_no_progress_guard, FsmState::FOLLOW, prev_state, effective_input.stamp
            );
        } else if (prev_state == FsmState::STEPPING) {
            no_progress_detected = progress_monitor_.check_no_progress(
                current_u, nav_params_.stepping_no_progress_guard, FsmState::STEPPING, prev_state, effective_input.stamp
            );
        }
    }

    // stepping 期间若有路径更新被延迟锁存，stepping 结束后强制重规划
    if (!request_replan_now && had_deferred_update) {
        request_replan_now = true;
    }

    const bool dist_reached = has_path && ((effective_input.chassis_pose_map.head<2>() - effective_input.global_path->position(1.0)).norm() < nav_params_.stop_threshold_dist);
    const bool u_reached = has_path && (current_u > nav_params_.stop_threshold_u);

    FsmInput fsm_input;
    fsm_input.has_path = has_path;
    fsm_input.has_new_path = has_new_path;
    fsm_input.chassis_pos_map = effective_input.chassis_pose_map.head<2>();
    fsm_input.fixed_goal_flag = effective_input.fixed_goal;
    fsm_input.reach_goal = dist_reached || u_reached;
    fsm_input.step_active = step_controller_.is_step_active(current_u);
    fsm_input.replan_requested = request_replan_now;
    fsm_input.replan_failed = !has_path && effective_input.path_updated;
    fsm_input.command_blocked = command_blocked;
    fsm_input.spin_requested = effective_input.spin_requested;
    fsm_input.spin_high_priority = effective_input.spin_high_priority;
    const bool hazard_allowed = (prev_state == FsmState::IDLE) || (prev_state == FsmState::SPIN)
        || (prev_state == FsmState::HAZARD_RECOVERY)
        || (prev_state == FsmState::STUCK_REVERSE);
    if (no_progress_detected) {
        fsm_input.is_hazard = !command_blocked
            && effective_input.masked_global_cost_map
            && effective_input.masked_direction_map
            && safety_monitor_.compute_is_hazard(
                *effective_input.masked_global_cost_map,
                *effective_input.masked_direction_map,
                effective_input.chassis_pose_map.head<2>()
            );
    } else {
        fsm_input.is_hazard = !command_blocked && hazard_allowed
            && effective_input.masked_global_cost_map
            && effective_input.masked_direction_map
            && safety_monitor_.compute_is_hazard(
                *effective_input.masked_global_cost_map,
                *effective_input.masked_direction_map,
                effective_input.chassis_pose_map.head<2>()
            );
    }
    fsm_input.no_progress_detected = no_progress_detected;
    fsm_input.is_stuck = !command_blocked && safety_monitor_.check_stuck(
        effective_input.chassis_pose_map.head<2>(),
        last_cmd_.x(),
        effective_input.stamp
    );
    fsm_input.is_recovery_safe = !command_blocked
        && effective_input.final_cost_map
        && effective_input.masked_direction_map
        && safety_monitor_.update_recovery_safe_flag(
            *effective_input.final_cost_map,
            *effective_input.masked_direction_map,
            effective_input.chassis_pose_map.head<2>(),
            effective_input.stamp
        );
    fsm_input.velocity = last_cmd_.x();
    fsm_input.omega = last_cmd_.y();
    fsm_input.stamp = effective_input.stamp;

    const FsmOutput fsm_output = control_fsm_->update(fsm_input);
    const FsmState state = fsm_output.state;
    on_state_transition(prev_state, state, !command_blocked);
    last_fsm_state_ = state;

    ControlOutput output;
    const bool hold_last_output = command_blocked;
    if (hold_last_output) {
        apply_held_command(output);
    } else {
        switch (state) {
            case FsmState::IDLE: output = execute_idle(effective_input); break;
            case FsmState::FOLLOW: output = execute_follow(effective_input); break;
            case FsmState::SPIN: output = execute_spin(effective_input); break;
            case FsmState::STOPPING: output = execute_stop(effective_input); break;
            case FsmState::HAZARD_RECOVERY: output = execute_recovery(effective_input); break;
            case FsmState::STUCK_REVERSE: output = execute_stuck_reverse(effective_input); break;
            case FsmState::FIXED: output = execute_fixed(effective_input); break;
            case FsmState::WAIT_REPLAN: output = execute_stop(effective_input); break;
            case FsmState::STEPPING: output = execute_follow(effective_input); break;
            case FsmState::DEAD: output = execute_idle(effective_input); break;
        }
    }

    output.fsm_state = state;
    output.consume_global_path |= fsm_output.consume_global_path;
    output.request_replan = fsm_output.request_replan;
    output.keep_goal_on_path_consume = output.request_replan || state == FsmState::WAIT_REPLAN;
    if (output.consume_global_path) {
        last_reference_u_ = 0.0;
        step_controller_.clear_plan();
    }

    if (output.valid) {
        last_cmd_ = Eigen::Vector2d(output.velocity, output.omega);
        mpc_controller_->set_last_cmd(last_cmd_);
        remember_command_output(output);
        const bool reset_warm_start = !hold_last_output && (
            state == FsmState::IDLE || state == FsmState::SPIN || state == FsmState::STUCK_REVERSE
        );
        if (reset_warm_start) {
            reset_all_mpc_warm_start();
        }
    }

    return output;
}

FsmState MainController::fsm_state() const {
    return control_fsm_->state();
}

// ═══════════════════ IDLE: 保持静止 ══════════════════════════

ControlOutput MainController::execute_idle(const ControlInput& input) {
    (void)input;
    ControlOutput out;
    out.velocity = 0.0;
    out.omega = 0.0;
    out.mode = chassis_mode::NORMAL;
    out.valid = true;
    return out;
}

// ═══════════════════ FOLLOW: 跟随路径 ════════════════════════

ControlOutput MainController::execute_follow(const ControlInput& input) {
    ControlOutput out;
    if (!input.global_path || !input.final_cost_map || !input.masked_global_cost_map || !input.masked_direction_map) return out;

    const double u0 = last_reference_u_;

    step_controller_.tick_profile_blend();
    auto start_time = std::chrono::steady_clock::now();
    const auto result = mpc_controller_->solve_follow(
        *input.global_path, input.chassis_pose_map, input.chassis_state,
        *input.final_cost_map, *input.masked_global_cost_map, input.per_step_cost_maps, input.prediction_dt,
        *input.masked_direction_map,
        step_controller_.current_blended_profile(),
        step_controller_.current_active_step_mode(u0)
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Follow) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 600.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Follow) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 600.0);
    }

    const auto& follow_result = *result;
    const auto& cmd = follow_result.command;
    const auto& prediction = follow_result.prediction;

    if (follow_result.status == MPCSolver::FollowSolveStatus::STOP_AND_WAIT_REPLAN) {
        follow_stop_and_wait_replan_pending_ = true;
        if (follow_result.lethal_obstacle) {
            const auto& lethal = *follow_result.lethal_obstacle;
            RCLCPP_WARN(
                logger_,
                "Follow rollout entered lethal obstacle at step %d (x=%.2f, y=%.2f, cost=%.1f); outputting solve_stop and requesting wait_replan next cycle",
                lethal.state_index,
                lethal.position_map.x(),
                lethal.position_map.y(),
                lethal.sampled_cost
            );
        } else {
            RCLCPP_WARN(logger_, "Follow rollout entered lethal obstacle; outputting solve_stop and requesting wait_replan next cycle");
        }
        step_controller_.clear_plan();
    }

    out.velocity = cmd.x();
    out.omega = cmd.y();

    if (follow_result.status == MPCSolver::FollowSolveStatus::STOP_AND_WAIT_REPLAN) {
        out.mode = chassis_mode::NORMAL;
    } else if (const auto active_step_mode = step_controller_.current_active_step_mode(u0); active_step_mode && step_controller_.should_activate_step_mode(u0)) {
        out.mode = active_step_mode->mode;
    } else {
        out.mode = chassis_mode::NORMAL;
    }

    out.predicted_path_map = prediction.path_map;
    out.predicted_v = prediction.v_pred;
    out.predicted_w = prediction.w_pred;
    if (!prediction.rollout_paths.empty()) {
        out.mppi_rollouts = std::move(prediction.rollout_paths);
    }
    out.step_dist_cm = step_controller_.compute_step_distance_cm(*input.global_path, u0);
    out.valid = true;

    return out;
}

// ═══════════════════ SPIN: 小陀螺 ════════════════════════════

ControlOutput MainController::execute_spin(const ControlInput& input) {
    ControlOutput out;
    out.mode = input.spin_fast ? chassis_mode::SPIN_FAST : chassis_mode::SPIN_SLOW;
    out.valid = true;
    return out;
}

// ═══════════════════ STOPPING: 平滑减速 ══════════════════════

ControlOutput MainController::execute_stop(const ControlInput& input) {
    ControlOutput out;
    if (!input.final_cost_map) return out;

    auto start_time = std::chrono::steady_clock::now();
    const auto result = mpc_controller_->solve_stop(
        input.chassis_pose_map, input.chassis_state,
        *input.final_cost_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Stop) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 600.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Stop) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 600.0);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.mode = chassis_mode::NORMAL;
    out.predicted_path_map = std::get<1>(*result).path_map;
    out.predicted_v = std::get<1>(*result).v_pred;
    out.predicted_w = std::get<1>(*result).w_pred;
    out.valid = true;
    return out;
}

void MainController::on_state_transition(const FsmState prev, const FsmState next, const bool allow_warm_start_reset) {
    if (prev == next) return;

    const bool prev_follow_like = (prev == FsmState::FOLLOW) || (prev == FsmState::STEPPING);
    const bool next_follow_like = (next == FsmState::FOLLOW) || (next == FsmState::STEPPING);

    if (prev == FsmState::STUCK_REVERSE || prev == FsmState::HAZARD_RECOVERY) {
        safety_monitor_.reset_stuck();
    }

    if (prev_follow_like && !next_follow_like) {
        step_controller_.clear_runtime_state();
        last_reference_u_ = 0.0;
        progress_monitor_.reset();
        follow_stop_and_wait_replan_pending_ = false;
        if (allow_warm_start_reset) {
            mpc_controller_->reset_warm_start();
        }
    }

    if (next == FsmState::WAIT_REPLAN) {
        step_controller_.clear_plan();
        last_reference_u_ = 0.0;
        follow_stop_and_wait_replan_pending_ = false;
    }

    if (next_follow_like && !prev_follow_like) {
        last_reference_u_ = 0.0;
    }

    if (next == FsmState::HAZARD_RECOVERY) {
        safety_monitor_.reset_recovery();
    }
    if (prev == FsmState::HAZARD_RECOVERY && next != FsmState::HAZARD_RECOVERY) {
        safety_monitor_.reset_recovery();
    }

    const bool next_uses_hold = (next == FsmState::FIXED);
    const bool prev_uses_hold = (prev == FsmState::FIXED) || (prev == FsmState::HAZARD_RECOVERY);
    if (allow_warm_start_reset && next_uses_hold && !prev_uses_hold) {
        mpc_controller_->reset_warm_start();
    }
}

// ═══════════════════ HAZARD_RECOVERY: MPC 驱动恢复 ═══════════

ControlOutput MainController::execute_recovery(const ControlInput& input) {
    ControlOutput out;
    if (!input.final_cost_map || !input.masked_direction_map) return out;

    if (input.masked_global_cost_map) {
        safety_monitor_.update_recovery_goal_if_needed(
            input.chassis_pose_map,
            *input.masked_global_cost_map,
            *input.masked_direction_map,
            input.base_direction_map,
            input.stamp
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

    auto start_time = std::chrono::steady_clock::now();
    const auto result = mpc_controller_->solve_hold(
        *recovery_goal,
        input.chassis_pose_map, input.chassis_state,
        *input.final_cost_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Recovery) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 600.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Recovery) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 600.0);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.mode = chassis_mode::NORMAL;
    out.predicted_path_map = std::get<1>(*result).path_map;
    out.predicted_v = std::get<1>(*result).v_pred;
    out.predicted_w = std::get<1>(*result).w_pred;
    out.valid = true;
    return out;
}

// ═══════════════════ STUCK_REVERSE: 倒车脱困 ═════════════════

ControlOutput MainController::execute_stuck_reverse(const ControlInput& input) {
    (void)input;
    ControlOutput out;
    out.velocity = -fsm_params_.stuck.reverse_speed;
    out.omega = 0.0;
    out.mode = chassis_mode::NORMAL;
    out.valid = true;
    return out;
}

// ═══════════════════ FIXED: 固定在目标点 ═════════════════════

ControlOutput MainController::execute_fixed(const ControlInput& input) {
    ControlOutput out;
    if (!input.final_cost_map || !input.masked_direction_map) return out;

    auto start_time = std::chrono::steady_clock::now();
    const auto result = mpc_controller_->solve_hold(
        input.fixed_goal_pos,
        input.chassis_pose_map, input.chassis_state,
        *input.final_cost_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Fixed) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 600.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Fixed) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 600.0);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.mode = chassis_mode::NORMAL;
    out.predicted_path_map = std::get<1>(*result).path_map;
    out.predicted_v = std::get<1>(*result).v_pred;
    out.predicted_w = std::get<1>(*result).w_pred;
    out.valid = true;
    return out;
}

} // namespace path_follower
