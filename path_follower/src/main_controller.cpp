#include <path_follower/main_controller.hpp>
#include <rclcpp/logging.hpp>

namespace path_follower {
namespace {

enum class ChassisControlState : uint8_t {
    DEAD,
    COMMAND_BLOCKED,
    MATURE,
};

constexpr uint8_t LEG_MODE_DEAD = 0u;
constexpr uint8_t LEG_MODE_RECOVERY = 1u;
constexpr uint8_t LEG_MODE_FLIGHT = 2u;
constexpr uint8_t LEG_MODE_JUMP = 3u;
constexpr uint8_t LEG_MODE_MATURE = 4u;
constexpr uint8_t LEG_MODE_STEP = 5u;
constexpr uint8_t LEG_MODE_ABNORMAL = 6u;
constexpr uint8_t COMP_STAGE_MATCH = 4u;

using path_follower::mode_label;

ChassisControlState classify_chassis_control_state(const uint8_t leg_mode, const uint8_t comp_stage) {
    if (comp_stage != COMP_STAGE_MATCH) {
        return ChassisControlState::DEAD;
    }

    switch (leg_mode) {
        case LEG_MODE_DEAD:
        case LEG_MODE_RECOVERY:
        case LEG_MODE_ABNORMAL:
            return ChassisControlState::DEAD;
        case LEG_MODE_FLIGHT:
        case LEG_MODE_JUMP:
        case LEG_MODE_STEP:
            return ChassisControlState::COMMAND_BLOCKED;
        case LEG_MODE_MATURE:
        default:
            return ChassisControlState::MATURE;
    }
}

}

using path_follower::is_step_mode;

namespace {

struct FollowStepBlockSampleStats {
    int sample_count = 0;
    int step_sample_count = 0;
    int blocked_step_sample_count = 0;
};

std::optional<FollowStepBlockSampleStats> sample_step_block_replan_stats(
    const NavigationParams& params,
    const SplineD& path,
    const double start_u,
    const CostMap* const dynamic_cost_map,
    const std::vector<const CostMap*>& dynamic_prediction_maps,
    const DirectionMap& direction_map
) {
    const auto& p = params.step_block_replan;
    const double lookahead_distance = std::max(0.0, p.lookahead_distance);
    const double resolution = std::max(1e-3, p.sample_resolution);
    const bool using_predicted = !dynamic_prediction_maps.empty();

    FollowStepBlockSampleStats stats;
    const int samples = using_predicted
        ? static_cast<int>(dynamic_prediction_maps.size())
        : std::max(1, static_cast<int>(std::ceil(lookahead_distance / resolution)) + 1);
    if (samples <= 0) return stats;

    auto advance_path_u = [&](const double distance) {
        double u = std::clamp(start_u, 0.0, 1.0);
        double travelled = 0.0;
        while (u < 1.0 && travelled < distance) {
            const Eigen::Vector2d d1 = path.derivative(u, 1);
            const double speed = d1.norm();
            if (speed < 1e-12) {
                u = std::min(1.0, u + 1e-3);
                continue;
            }
            const double du = resolution / speed;
            const double next_u = std::min(1.0, u + du);
            travelled += (path.evaluate(next_u) - path.evaluate(u)).norm();
            u = next_u;
        }
        return u;
    };

    for (int i = 0; i < samples; ++i) {
        const double t = samples == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(samples - 1);
        const double u = advance_path_u(lookahead_distance * t);

        const Eigen::Vector2d pos = path.evaluate(u);
        const Eigen::Vector2d dir_grid = direction_map.map_coord_to_grid(pos);
        if (!direction_map.is_valid_coord(dir_grid)) return std::nullopt;

        const double step_norm = direction_map.interpolate(dir_grid).norm();
        const bool on_step = step_norm >= p.step_norm_threshold;

        bool blocked_dynamic = false;
        if (using_predicted) {
            const CostMap* const cost_map = dynamic_prediction_maps[static_cast<size_t>(i)];
            if (!cost_map) return std::nullopt;
            const Eigen::Vector2d cost_grid = cost_map->map_coord_to_grid(pos);
            if (!cost_map->is_valid_coord(cost_grid)) return std::nullopt;
            blocked_dynamic = cost_map->interpolate(cost_grid) >= p.obstacle_cost_threshold;
        } else if (dynamic_cost_map) {
            const Eigen::Vector2d cost_grid = dynamic_cost_map->map_coord_to_grid(pos);
            if (!dynamic_cost_map->is_valid_coord(cost_grid)) return std::nullopt;
            blocked_dynamic = dynamic_cost_map->interpolate(cost_grid) >= p.obstacle_cost_threshold;
        }

        stats.sample_count++;
        if (!on_step) continue;

        stats.step_sample_count++;
        if (blocked_dynamic) {
            stats.blocked_step_sample_count++;
        }
    }

    return stats;
}

}
}  // namespace path_follower

// ═══════════════════════ 构造函数 ════════════════════════════

namespace path_follower {

MainController::MainController(
    const NavigationParams& nav_params,
    const FsmParams& fsm_params,
    std::shared_ptr<MPCSolver> mpc_controller,
    rclcpp::Logger logger
) : control_fsm_(std::make_unique<StateMachine>(fsm_params, logger)),
    mpc_controller_(std::move(mpc_controller)),
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
        output.mode = ChassisMode::NORMAL;
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

double MainController::project_path_u(const ControlInput& input, const SplineD& path, const double seed_u) const {
    return project_to_spline_u_extrapolated(
        path,
        input.chassis_pose_map.head<2>(),
        seed_u,
        mpc_controller_->params().follow.projection.proj_num_samples,
        mpc_controller_->params().follow.projection.proj_search_window,
        mpc_controller_->params().follow.projection.local_search_lazy_distance
    );
}

bool MainController::check_follow_projection_guard(const ControlInput& input, const SplineD& path, const double current_u) const {
    const Eigen::Vector2d pos_map = input.chassis_pose_map.head<2>();
    const Eigen::Vector2d proj_map = path.evaluate(current_u);
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

bool MainController::check_step_block_replan(const ControlInput& input, const SplineD& path, const double current_u) const {
    const auto& p = nav_params_.step_block_replan;
    if (!p.enable || !input.masked_direction_map) return false;

    const auto stats = sample_step_block_replan_stats(
        nav_params_,
        path,
        current_u,
        input.current_dynamic_cost_map,
        input.per_step_dynamic_cost_maps,
        *input.masked_direction_map
    );
    if (!stats || stats->step_sample_count == 0) {
        return false;
    }

    if (input.per_step_dynamic_cost_maps.empty()) {
        if (stats->blocked_step_sample_count > 0) {
            RCLCPP_WARN(
                logger_,
                "Follow replan: detected blocked step ahead within %.2f m (blocked_step_samples=%d/%d)",
                p.lookahead_distance,
                stats->blocked_step_sample_count,
                stats->step_sample_count
            );
            return true;
        }
        return false;
    }

    const double blocked_ratio = static_cast<double>(stats->blocked_step_sample_count) / static_cast<double>(stats->step_sample_count);
    if (blocked_ratio >= p.predicted_obstacle_ratio_threshold) {
        RCLCPP_WARN(
            logger_,
            "Follow replan: predicted blocked step ahead within %.2f m (ratio=%.2f >= %.2f)",
            p.lookahead_distance,
            blocked_ratio,
            p.predicted_obstacle_ratio_threshold
        );
        return true;
    }

    return false;
}

// ═══════════════════════ 主更新接口 ══════════════════════════

ControlOutput MainController::update(const ControlInput& input) {
    const ChassisControlState chassis_control_state = classify_chassis_control_state(input.chassis_leg_mode, input.comp_stage);
    const bool chassis_dead = chassis_control_state == ChassisControlState::DEAD;
    const bool command_blocked = chassis_control_state == ChassisControlState::COMMAND_BLOCKED;
    const bool chassis_controllable = chassis_control_state == ChassisControlState::MATURE;
    const bool entered_controllable = chassis_controllable && !last_cycle_chassis_controllable_;
    last_cycle_chassis_controllable_ = chassis_controllable;

    // 全局中断优先：底盘 Dead 直接外部拦截，不进入 FSM。
    if (chassis_dead) {
        ControlOutput out;
        out.velocity = 0.0;
        out.omega = 0.0;
        out.mode = ChassisMode::NORMAL;
        out.step_dist_cm = 0;
        out.fsm_state = FsmState::DEAD;
        out.valid = true;

        last_cmd_ = Eigen::Vector2d::Zero();
        reset_all_mpc_warm_start();
        reset_all_mpc_observer();
        stuck_active_ = false;
        recovery_safe_since_ = std::nullopt;
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

    ControlInput effective_input = input;
    const bool step_path_locked = active_step_command_.has_value() && step_locked_path_.has_value();
    if (step_path_locked) {
        if (input.path_updated || !input.global_path.has_value()) {
            deferred_external_path_update_ = true;
        }
        effective_input.global_path = step_locked_path_;
        effective_input.path_updated = false;
        effective_input.fixed_goal = step_locked_fixed_goal_;
        effective_input.fixed_goal_pos = step_locked_fixed_goal_pos_;
    } else {
        effective_input.path_updated = input.path_updated || deferred_external_path_update_;
        deferred_external_path_update_ = false;
    }

    const FsmState prev_state = last_fsm_state_;
    sync_mpc_context(effective_input, chassis_controllable);

    if (prev_state == FsmState::HAZARD_RECOVERY) {
        update_recovery_goal_if_needed(effective_input);
    }

    const bool has_path = effective_input.global_path.has_value();
    const bool has_new_path = has_path && effective_input.path_updated;

    update_step_state_for_path_change(has_new_path);
    if (!has_path) {
        clear_step_state();
    }

    if (has_new_path) {
        recompute_follow_landmarks(*effective_input.global_path);
        follow_max_landmark_idx_ = -1;
    } else if (!has_path) {
        follow_landmarks_u_.clear();
        follow_max_landmark_idx_ = -1;
    }

    const double current_u = has_path ? project_path_u(effective_input, *effective_input.global_path, last_reference_u_) : 0.0;
    if (has_path) {
        last_reference_u_ = current_u;
        if (effective_input.masked_direction_map) {
            extend_active_step_exit(*effective_input.global_path, *effective_input.masked_direction_map);
        }
        update_step_release(*effective_input.global_path, current_u, effective_input.stamp);
    }

    bool request_replan_now = false;
    if (!command_blocked && has_path && prev_state == FsmState::FOLLOW) {
        request_replan_now = check_follow_projection_guard(effective_input, *effective_input.global_path, current_u)
            || check_no_progress(effective_input, current_u)
            || ((prev_state == FsmState::FOLLOW) && check_step_block_replan(effective_input, *effective_input.global_path, current_u));
    }

    if (!request_replan_now && has_path && prev_state == FsmState::FOLLOW) {
        request_replan_now = !command_blocked && prepare_follow_step_behavior(effective_input, *effective_input.global_path, current_u);
    }

    const bool dist_reached = has_path && ((effective_input.chassis_pose_map.head<2>() - effective_input.global_path->evaluate(1.0)).norm() < nav_params_.stop_threshold_dist);
    const bool u_reached = has_path && (current_u > nav_params_.stop_threshold_u);

    FsmInput fsm_input;
    fsm_input.has_path = has_path;
    fsm_input.has_new_path = has_new_path;
    fsm_input.step_ttl_expired = step_ttl_just_expired_;
    step_ttl_just_expired_ = false;
    fsm_input.chassis_pos_map = effective_input.chassis_pose_map.head<2>();
    fsm_input.fixed_goal_flag = effective_input.fixed_goal;
    fsm_input.reach_goal = dist_reached || u_reached;
    fsm_input.step_active = is_step_active();
    fsm_input.replan_requested = request_replan_now;
    fsm_input.command_blocked = command_blocked;
    fsm_input.spin_requested = effective_input.spin_requested;
    fsm_input.spin_high_priority = effective_input.spin_high_priority;
    const bool hazard_allowed = (prev_state == FsmState::IDLE) || (prev_state == FsmState::SPIN)
        || (prev_state == FsmState::HAZARD_RECOVERY);
    fsm_input.is_hazard = !command_blocked && hazard_allowed && compute_is_hazard(effective_input);
    fsm_input.is_stuck = !command_blocked && check_stuck(effective_input);
    fsm_input.is_recovery_safe = !command_blocked && update_recovery_safe_flag(effective_input);
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
        clear_step_state();
    }

    // 4. 同步已发布指令到 FSM / MPC；IDLE / SPIN / STUCK_REVERSE 无 track 连续性，
    //    进入这些状态后 MPC warm start 已无意义，直接重置。
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
    out.mode = ChassisMode::NORMAL;
    out.valid = true;
    return out;
}

// ═══════════════════ FOLLOW: 跟随路径 ════════════════════════

ControlOutput MainController::execute_follow(const ControlInput& input) {
    ControlOutput out;
    if (!input.global_path || !input.final_cost_map || !input.masked_direction_map) return out;

    const double u0 = last_reference_u_;
    last_reference_u_ = u0;

    auto start_time = std::chrono::steady_clock::now();
    const auto result = mpc_controller_->solve_follow(
        *input.global_path, input.chassis_pose_map, input.chassis_state,
        *input.final_cost_map, input.per_step_cost_maps, input.prediction_dt,
        *input.masked_direction_map,
        current_active_step_mode()
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Follow) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Follow) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    }

    const auto& [cmd, prediction] = *result;

    out.velocity = cmd.x();
    out.omega = cmd.y();

    if (active_step_command_) {
        out.mode = active_step_command_->mode;
    } else {
        out.mode = ChassisMode::NORMAL;
    }

    out.predicted_path_map = prediction.path_map;
    out.predicted_v = prediction.v_pred;
    out.predicted_w = prediction.w_pred;
    out.step_dist_cm = compute_step_distance_cm(input, u0, prediction);
    out.valid = true;

    return out;
}

// ═══════════════════ SPIN: 小陀螺 ════════════════════════════

ControlOutput MainController::execute_spin(const ControlInput& input) {
    ControlOutput out;
    out.mode = input.spin_fast ? ChassisMode::SPIN_FAST : ChassisMode::SPIN_SLOW;
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
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Stop) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.mode = ChassisMode::NORMAL;
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

    // 离开 STUCK_REVERSE 或 HAZARD_RECOVERY 时，必须重置卡死检测
    if (prev == FsmState::STUCK_REVERSE || prev == FsmState::HAZARD_RECOVERY) {
        stuck_active_ = false;
    }

    if (prev_follow_like && !next_follow_like) {
        clear_step_state();
        last_reference_u_ = 0.0;
        follow_max_landmark_idx_ = -1;
        if (allow_warm_start_reset) {
            mpc_controller_->reset_warm_start();
        }
    }

    if (next == FsmState::WAIT_REPLAN) {
        clear_step_state();
        last_reference_u_ = 0.0;
    }

    if (next_follow_like && !prev_follow_like) {
        last_reference_u_ = 0.0;
    }

    if (next == FsmState::HAZARD_RECOVERY) {
        recovery_goal_map_ = std::nullopt;
        recovery_goal_set_time_ = std::nullopt;
        recovery_safe_since_ = std::nullopt;
    }
    if (prev == FsmState::HAZARD_RECOVERY && next != FsmState::HAZARD_RECOVERY) {
        recovery_goal_map_ = std::nullopt;
        recovery_goal_set_time_ = std::nullopt;
        recovery_safe_since_ = std::nullopt;
    }

    // Hold 求解器由 HAZARD_RECOVERY / FIXED 共享，二者之间切换不应互相清空 warm start。
    const bool next_uses_hold = (next == FsmState::FIXED);
    const bool prev_uses_hold = (prev == FsmState::FIXED) || (prev == FsmState::HAZARD_RECOVERY);
    if (allow_warm_start_reset && next_uses_hold && !prev_uses_hold) {
        mpc_controller_->reset_warm_start();
    }
}

void MainController::update_recovery_goal_if_needed(const ControlInput& input) {
    const auto& p = fsm_params_.recovery;
    if (!input.masked_global_cost_map || !input.masked_direction_map) return;

    const bool need_new = (!recovery_goal_map_) || (!recovery_goal_set_time_) || (std::chrono::duration<double>(input.stamp - *recovery_goal_set_time_).count() >= p.goal_timeout);

    if (!need_new) return;

    recovery_goal_map_ = recovery_helpers::find_goal(
        p, *input.masked_global_cost_map, *input.masked_direction_map, input.chassis_pose_map,
        input.base_direction_map
    );
    recovery_goal_set_time_ = input.stamp;

    if (!recovery_goal_map_) {
        RCLCPP_ERROR(logger_, "HAZARD_RECOVERY failed to find a recovery goal");
        return;
    }

    const auto s = recovery_helpers::sample_fields(*input.masked_global_cost_map, *input.masked_direction_map, *recovery_goal_map_);
    if (!s) {
        RCLCPP_WARN(logger_, "HAZARD_RECOVERY new goal=(%.2f, %.2f) (field sample invalid)", recovery_goal_map_->x(), recovery_goal_map_->y());
        return;
    }

    if (recovery_helpers::is_safe_goal(p, *s)) {
        RCLCPP_WARN(logger_, "HAZARD_RECOVERY new goal=(%.2f, %.2f) (SAFE cost=%.1f step=%.3f)", recovery_goal_map_->x(), recovery_goal_map_->y(), s->cost, s->step_norm);
    } else {
        RCLCPP_WARN(logger_, "HAZARD_RECOVERY new goal=(%.2f, %.2f) (UNSAFE cost=%.1f step=%.3f)", recovery_goal_map_->x(), recovery_goal_map_->y(), s->cost, s->step_norm);
    }
}

bool MainController::check_stuck(const ControlInput& input) {
    const auto& p = fsm_params_.stuck;
    if (std::abs(last_cmd_.x()) < p.cmd_vel_threshold) {
        stuck_active_ = false;
        return false;
    }

    const Eigen::Vector2d pos = input.chassis_pose_map.head<2>();
    if (!stuck_active_) {
        stuck_active_ = true;
        stuck_start_time_ = input.stamp;
        stuck_start_pos_ = pos;
        return false;
    }

    const double dt = std::chrono::duration<double>(input.stamp - stuck_start_time_).count();
    const double disp = (pos - stuck_start_pos_).norm();
    if (disp > p.max_displacement) {
        stuck_start_time_ = input.stamp;
        stuck_start_pos_ = pos;
        return false;
    }

    return dt >= p.timeout;
}

bool MainController::compute_is_hazard(const ControlInput& input) const {
    const auto& p = fsm_params_.recovery;
    if (!input.masked_global_cost_map || !input.masked_direction_map) return false;

    // 注意：危险判断的 cost_map 使用的是 masked_global 而非 final，避免动态障碍物导致车进入危险恢复模式
    const auto sample = recovery_helpers::sample_fields(
        *input.masked_global_cost_map,
        *input.masked_direction_map,
        input.chassis_pose_map.head<2>()
    );
    if (!sample) return false;

    return (sample->cost >= p.hazard_cost_threshold) ||
        (sample->step_norm >= p.hazard_step_norm_threshold);
}

bool MainController::update_recovery_safe_flag(const ControlInput& input) {
    const auto& p = fsm_params_.recovery;
    if (!input.final_cost_map || !input.masked_direction_map || !recovery_goal_map_) {
        recovery_safe_since_ = std::nullopt;
        return false;
    }

    const auto sample = recovery_helpers::sample_fields(
        *input.final_cost_map,
        *input.masked_direction_map,
        input.chassis_pose_map.head<2>()
    );
    if (!sample) {
        recovery_safe_since_ = std::nullopt;
        return false;
    }

    const bool safe_now = recovery_helpers::is_safe_goal(p, *sample);
    if (!safe_now) {
        recovery_safe_since_ = std::nullopt;
        return false;
    }

    if (!recovery_safe_since_) {
        recovery_safe_since_ = input.stamp;
    }
    return std::chrono::duration<double>(input.stamp - *recovery_safe_since_).count() >= p.safe_hold_time;
}

// ═══════════════════ HAZARD_RECOVERY: MPC 驱动恢复 ═══════════

ControlOutput MainController::execute_recovery(const ControlInput& input) {
    ControlOutput out;
    if (!input.final_cost_map || !input.masked_direction_map) return out;

    update_recovery_goal_if_needed(input);
    if (!recovery_goal_map_) {
        out.velocity = 0.0;
        out.omega = 0.0;
        out.mode = ChassisMode::NORMAL;
        out.valid = true;
        return out;
    }

    auto start_time = std::chrono::steady_clock::now();
    const auto result = mpc_controller_->solve_hold(
        *recovery_goal_map_,
        input.chassis_pose_map, input.chassis_state,
        *input.final_cost_map, *input.masked_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Recovery) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Recovery) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.mode = ChassisMode::NORMAL;
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
    out.mode = ChassisMode::NORMAL;
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
        *input.final_cost_map, *input.masked_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Fixed) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Fixed) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.mode = ChassisMode::NORMAL;
    out.predicted_path_map = std::get<1>(*result).path_map;
    out.predicted_v = std::get<1>(*result).v_pred;
    out.predicted_w = std::get<1>(*result).w_pred;
    out.valid = true;
    return out;
}

// ═══════════════════ Follow 路标点无进度检测 ══════════════════

void MainController::recompute_follow_landmarks(const SplineD& path) {
    follow_landmarks_u_.clear();
    const double spacing = std::max(0.1, nav_params_.no_progress_guard.landmark_spacing);

    // 先粗略估计路径弧长，动态确定采样数
    constexpr int ESTIMATE_SAMPLES = 100;
    double est_length = 0.0;
    Eigen::Vector2d est_prev = path.evaluate(0.0);
    for (int i = 1; i <= ESTIMATE_SAMPLES; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(ESTIMATE_SAMPLES);
        const Eigen::Vector2d cur = path.evaluate(u);
        est_length += (cur - est_prev).norm();
        est_prev = cur;
    }

    // 确保每米弧长至少有足够的采样点来准确捕捉弧长
    const double sample_density = std::max(
        10.0, 1.0 / std::max(spacing, 0.01)
    );
    const int N = std::max(10, static_cast<int>(std::ceil(est_length * sample_density)));

    double arc = 0.0;
    double next_threshold = 0.0;
    Eigen::Vector2d prev = path.evaluate(0.0);

    for (int i = 0; i <= N; i++) {
        const double u = static_cast<double>(i) / static_cast<double>(N);
        const Eigen::Vector2d cur = path.evaluate(u);
        if (i > 0) arc += (cur - prev).norm();
        prev = cur;
        if (arc >= next_threshold) {
            follow_landmarks_u_.push_back(u);
            next_threshold += spacing;
        }
    }

    if (follow_landmarks_u_.empty() || follow_landmarks_u_.back() < 1.0) {
        follow_landmarks_u_.push_back(1.0);
    }
}

bool MainController::check_no_progress(const ControlInput& input, const double current_u) {
    if (last_fsm_state_ != FsmState::FOLLOW) return false;
    if (follow_landmarks_u_.empty()) return false;

    const int n = static_cast<int>(follow_landmarks_u_.size());

    // 安全性说明：
    //   follow_max_landmark_idx_ == -1 时下方条件恒成立，确保 FOLLOW 的首个
    //   周期必重置计时器。该索引在以下时机归 -1：
    //     - 构造函数默认值
    //     - has_new_path 时 (line 368)
    //     - !has_path 时 (line 371)
    //     - 离开 follow-like 状态时 (line 640)
    // 因此除首个 FOLLOW 周期外，每次新路径或重入 FOLLOW 后首 cycle 必重置计时。
    // follow_max_landmark_time_ 默认构造为 epoch，但首个 FOLLOW 周期
    // follow_max_landmark_idx_ < 0 保证其立即被覆盖，不会误触发超时。

    // 寻找当前 u 能覆盖的最高路标点索引
    int new_max = follow_max_landmark_idx_;
    for (int i = std::max(0, new_max + 1); i < n; i++) {
        if (current_u >= follow_landmarks_u_[static_cast<size_t>(i)]) {
            new_max = i;
        } else {
            break;
        }
    }

    // 路标点前进（包括首次初始化）：重置计时器
    if (follow_max_landmark_idx_ < 0 || new_max > follow_max_landmark_idx_) {
        follow_max_landmark_idx_ = new_max;
        follow_max_landmark_time_ = input.stamp;
        return false;
    }

    const double elapsed = std::chrono::duration<double>(input.stamp - follow_max_landmark_time_).count();
    if (elapsed >= nav_params_.no_progress_guard.timeout) {
        const double landmark_u = follow_max_landmark_idx_ >= 0 ? follow_landmarks_u_[static_cast<size_t>(follow_max_landmark_idx_)] : -1.0;
        RCLCPP_WARN(logger_,
            "Follow replan: no progress at landmark %d/%d (landmark_u=%.3f, progress_u=%.3f) for %.1fs",
            follow_max_landmark_idx_, n - 1, landmark_u, current_u, elapsed);
        return true;
    }
    return false;
}

}
