#include <path_follower/main_controller.hpp>
#include <rclcpp/logging.hpp>

namespace path_follower {

namespace {

constexpr double ANGLE_EPSILON = 1e-6;

const char* mode_label(ChassisMode m) {
    switch (m) {
        case ChassisMode::STEP_UP_LEG: return "LEG";
        case ChassisMode::STEP_UP_JUMP: return "JUMP";
        case ChassisMode::NORMAL: return "NORMAL";
        case ChassisMode::STEP_DOWN_LEG: return "DOWN_LEG";
        case ChassisMode::STEP_DOWN_JUMP: return "DOWN_JUMP";
        default: return "?";
    }
}

inline bool is_chassis_dead(const uint8_t leg_mode, const uint8_t comp_stage) {
    // leg_mode: 0:Dead, 1:Recovery, 6:Abnormal, 7:Transition → chassis dead
    // comp_stage: 4:比赛中
    return leg_mode == 0u || leg_mode == 1u || leg_mode == 6u || leg_mode == 7u || comp_stage != 4u;
}

inline bool is_step_mode(const ChassisMode mode) {
    switch (mode) {
        case ChassisMode::STEP_UP_LEG:
        case ChassisMode::STEP_UP_JUMP:
        case ChassisMode::STEP_DOWN_LEG:
        case ChassisMode::STEP_DOWN_JUMP:
            return true;
        default:
            return false;
    }
}

}

namespace {

std::optional<double> max_cost_along_segment(
    const CostMap& cost_map,
    const Eigen::Vector2d& a_map,
    const Eigen::Vector2d& b_map,
    const int samples
) {
    const int n = std::max(1, samples);
    double max_cost = 0.0;
    for (int i = 0; i <= n; i++) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        const Eigen::Vector2d pos = a_map + (b_map - a_map) * t;
        const Eigen::Vector2d g = cost_map.map_coord_to_grid(pos);
        if (!cost_map.is_valid_coord(g)) return std::nullopt;
        max_cost = std::max(max_cost, cost_map.interpolate(g));
    }
    return max_cost;
}

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
    const DirectionMap& direction_map,
    const bool using_predicted_cost_maps
) {
    const auto& p = params.step_block_replan;
    const double lookahead_distance = std::max(0.0, p.lookahead_distance);
    const double resolution = std::max(1e-3, p.sample_resolution);

    FollowStepBlockSampleStats stats;
    const int samples = using_predicted_cost_maps
        ? static_cast<int>(dynamic_prediction_maps.size())
        : std::max(1, static_cast<int>(std::ceil(lookahead_distance / resolution)) + 1);
    if (samples <= 0) return stats;

    auto advance_path_u = [&](const double distance) {
        double u = std::clamp(start_u, 0.0, 1.0);
        Eigen::Vector2d prev = path.evaluate(u);
        const int sub_steps = std::max(1, static_cast<int>(std::ceil(std::max(0.0, distance) / resolution)));
        double travelled = 0.0;
        for (int step = 1; step <= sub_steps && u < 1.0; ++step) {
            const double next_u = std::min(1.0, start_u + (1.0 - start_u) * static_cast<double>(step) / static_cast<double>(sub_steps));
            const Eigen::Vector2d cur = path.evaluate(next_u);
            travelled += (cur - prev).norm();
            prev = cur;
            u = next_u;
            if (travelled >= distance) break;
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
        if (using_predicted_cost_maps) {
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

namespace {

struct RecoveryGoalPlanner {
    struct FieldSample {
        double cost = 0.0;
        double step_norm = 0.0;
    };

    static std::optional<FieldSample> sample_fields(
        const CostMap& cost_map,
        const DirectionMap& dir_map,
        const Eigen::Vector2d& pos
    ) {
        const Eigen::Vector2d gc = cost_map.map_coord_to_grid(pos);
        if (!cost_map.is_valid_coord(gc)) return std::nullopt;
        const Eigen::Vector2d gd = dir_map.map_coord_to_grid(pos);
        if (!dir_map.is_valid_coord(gd)) return std::nullopt;

        FieldSample s;
        s.cost = cost_map.interpolate(gc);
        s.step_norm = dir_map.interpolate(gd).norm();
        return s;
    }

    static bool is_safe_goal(const RecoveryParams& p, const FieldSample& s) {
        return (s.cost < p.safe_cost_threshold) && (s.step_norm < p.safe_step_norm_threshold);
    }

    static double potential_cost(const FieldSample& s) {
        const double cost01 = std::clamp(s.cost / 255.0, 0.0, 1.0);
        return cost01 + s.step_norm;
    }

    struct PathScore {
        double score = std::numeric_limits<double>::infinity();
        bool end_safe = false;
        FieldSample end_sample;
    };

    static std::optional<PathScore> score_candidate_by_path_integral(
        const RecoveryParams& p,
        const CostMap& cost_map,
        const DirectionMap& dir_map,
        const Eigen::Vector2d& origin,
        const Eigen::Vector2d& goal,
        const double radius
    ) {
        double acc = 0.0;
        std::optional<FieldSample> end_s;
        const int n = std::max(1, static_cast<int>(radius / p.path_integral_resolution));
        for (int i = 0; i <= n; i++) {
            const double t = static_cast<double>(i) / static_cast<double>(n);
            const Eigen::Vector2d pos = origin + (goal - origin) * t;
            const auto s = sample_fields(cost_map, dir_map, pos);
            if (!s) return std::nullopt;
            acc += potential_cost(*s);
            if (i == n) end_s = s;
        }

        if (!end_s || !is_safe_goal(p, *end_s)) {
            return std::nullopt;
        }

        PathScore out;
        out.score = acc;
        out.end_sample = *end_s;
        out.end_safe = is_safe_goal(p, *end_s);
        return out;
    }

    static std::optional<Eigen::Vector2d> find_goal(
        const RecoveryParams& p,
        const CostMap& cost_map,
        const DirectionMap& dir_map,
        const Eigen::Vector3d& chassis_pose
    ) {
        const Eigen::Vector2d origin = chassis_pose.head<2>();

        const double r_min = std::min(p.radius_min, p.radius_max);
        const double r_max = std::max(p.radius_min, p.radius_max);
        const int r_n = std::max(1, p.radius_samples);
        const int a_n = std::max(1, p.angle_samples);

        std::optional<Eigen::Vector2d> best_pt;
        std::optional<PathScore> best_sc;

        for (int ri = 0; ri < r_n; ri++) {
            const double rt = (r_n == 1) ? 0.0 : (static_cast<double>(ri) / static_cast<double>(r_n - 1));
            const double r = r_min + (r_max - r_min) * rt;

            for (int ai = 0; ai < a_n; ai++) {
                const double a = 2.0 * std::numbers::pi * static_cast<double>(ai) / static_cast<double>(a_n);
                const Eigen::Vector2d pt = origin + Eigen::Vector2d(std::cos(a), std::sin(a)) * r;
                const auto field = sample_fields(cost_map, dir_map, pt);
                if (!field) continue;
                if (field->cost >= p.recovery_cost_threshold) continue;
                const auto sc = score_candidate_by_path_integral(p, cost_map, dir_map, origin, pt, r);
                if (!sc) continue;
                if (!best_sc || sc->score < best_sc->score) {
                    best_sc = *sc;
                    best_pt = pt;
                }
            }
        }

        return best_pt;
    }
};

Eigen::Vector2d rotate_vector(const Eigen::Vector2d& v, const double angle) {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return Eigen::Vector2d(c * v.x() - s * v.y(), s * v.x() + c * v.y());
}

std::optional<double> score_runup_path_integral(
    const NavigationParams& params,
    const CostMap& cost_map,
    const DirectionMap& dir_map,
    const Eigen::Vector2d& origin,
    const Eigen::Vector2d& goal
) {
    const double dist = (goal - origin).norm();
    const double resolution = std::max(1e-3, params.step_runup.search.path_integral_resolution);
    const int samples = std::max(1, static_cast<int>(std::ceil(dist / resolution)));
    double score = 0.0;
    for (int i = 0; i <= samples; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(samples);
        const Eigen::Vector2d pos = origin + (goal - origin) * t;
        const auto sample = RecoveryGoalPlanner::sample_fields(cost_map, dir_map, pos);
        if (!sample) return std::nullopt;
        score += params.step_runup.search.path_integral_cost_weight * std::clamp(sample->cost / 255.0, 0.0, 1.0)
            + params.step_runup.search.path_integral_step_weight * sample->step_norm;
    }
    return score;
}

}  // namespace

// ═══════════════════════ 构造函数 ════════════════════════════

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
    filtered_follow_rollout_length_ = nav_params_.step_detection.lookahead.min_distance;
    step_lookahead_distance_ = nav_params_.step_detection.lookahead.min_distance;
}

void MainController::sync_mpc_context(const ControlInput& input) {
    mpc_controller_->set_last_cmd(last_cmd_);

    mpc_controller_->update_observer(input.chassis_state);

    mpc_controller_->set_energy_state(input.remaining_energy, input.rfr_pwr_limit);
}

void MainController::reset_all_mpc_warm_start() {
    mpc_controller_->reset_warm_start();
}

void MainController::reset_all_mpc_observer() {
    mpc_controller_->reset_observer();
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

    const auto max_cost = max_cost_along_segment(
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
        *input.masked_direction_map,
        input.using_predicted_cost_maps
    );
    if (!stats || stats->step_sample_count == 0) {
        return false;
    }

    if (!input.using_predicted_cost_maps) {
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

bool MainController::prepare_follow_step_behavior(const ControlInput& input, const SplineD& path, const double current_u) {
    if (!input.masked_direction_map) return false;

    if (!active_step_target_ && !step_runup_request_pending_) {
        if (const auto step_target = try_latch_step_target(path, current_u, *input.masked_direction_map)) {
            const auto step_command = build_step_command(*step_target, *input.masked_direction_map);
            if (step_command) {
                const bool needs_runup = step_target->direction == StepDirection::UP;
                const auto runup = needs_runup
                    ? evaluate_step_runup(input, path, current_u, *step_target, *step_command)
                    : StepRunupDecision {};
                if (needs_runup && runup.debug_rollout_path_map) {
                    pending_step_rollout_path_map_ = runup.debug_rollout_path_map;
                }
                if (runup.request_replan) {
                    RCLCPP_WARN(
                        logger_,
                        "Follow replan: step run-up failed for target at (%.2f, %.2f)",
                        step_target->enter_pos_map.x(),
                        step_target->enter_pos_map.y()
                    );
                    return true;
                }

                if (runup.context) {
                    step_runup_context_ = runup.context;
                    step_runup_request_pending_ = true;
                    step_runup_goal_reached_ = false;
                    RCLCPP_DEBUG(
                        logger_,
                        "Step run-up requested: mode=%s deficit=%.2f target_v=%.2f arrival_v=%.2f current_v=%.2f enter_u=%.3f goal=(%.2f, %.2f)",
                        mode_label(step_command->mode),
                        runup.context->velocity_deficit,
                        step_command->target_velocity,
                        runup.context->predicted_arrival_velocity,
                        input.chassis_state.velocity,
                        step_target->enter_u,
                        runup.context->goal_map.x(),
                        runup.context->goal_map.y()
                    );
                    return false;
                }

                active_step_target_ = *step_target;
                active_step_command_ = *step_command;
                step_locked_path_ = input.global_path;
                step_locked_fixed_goal_ = input.fixed_goal;
                step_locked_fixed_goal_pos_ = input.fixed_goal_pos;
                RCLCPP_DEBUG(
                    logger_,
                    "Step command latched: dir=%s mode=%s target_v=%.2f at (%.2f, %.2f)",
                    step_target->direction == StepDirection::UP ? "UP" : "DOWN",
                    mode_label(step_command->mode),
                    step_command->target_velocity,
                    step_target->enter_pos_map.x(),
                    step_target->enter_pos_map.y()
                );
            } else {
                RCLCPP_WARN(logger_, "StepUp target latched but alpha mode forbids traversal");
            }
        }
    }

    if (active_step_command_ && !step_locked_path_ && input.global_path) {
        step_locked_path_ = input.global_path;
        step_locked_fixed_goal_ = input.fixed_goal;
        step_locked_fixed_goal_pos_ = input.fixed_goal_pos;
    }

    return false;
}

// ═══════════════════════ 主更新接口 ══════════════════════════

ControlOutput MainController::update(const ControlInput& input) {
    const bool chassis_dead = is_chassis_dead(input.chassis_leg_mode, input.comp_stage);

    // 检测 leg_mode 上升沿进入 4(Mature)，每次进入都重置龙伯格观测器
    const bool entered_mature = !chassis_dead && input.chassis_leg_mode == 4u && last_leg_mode_ != 4u;
    last_leg_mode_ = input.chassis_leg_mode;

    // 全局中断优先：底盘 Dead 直接外部拦截，不进入 FSM。
    if (chassis_dead) {
        last_cycle_chassis_dead_ = true;
        ControlOutput out;
        out.velocity = 0.0;
        out.omega = 0.0;
        out.fsm_state = FsmState::DEAD;
        out.valid = true;

        last_cmd_ = Eigen::Vector2d::Zero();
        reset_all_mpc_warm_start();
        reset_all_mpc_observer();
        stuck_active_ = false;
        recovery_safe_since_ = std::nullopt;
        return out;
    }

    // 每次进入 Mature(4) 重置观测器，确保从干净状态开始
    if (entered_mature) {
        RCLCPP_DEBUG(logger_, "Mature entered (leg_mode=4): resetting Luenberger observer");
        reset_all_mpc_observer();
    }

    last_cycle_chassis_dead_ = false;

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
    const bool had_step_runup_request = step_runup_request_pending_;
    sync_mpc_context(effective_input);

    if (prev_state == FsmState::HAZARD_RECOVERY) {
        update_recovery_goal_if_needed(effective_input);
    }

    const bool has_path = effective_input.global_path.has_value();
    const bool has_new_path = has_path && effective_input.path_updated;

    update_step_state_for_path_change(has_new_path);
    if (!has_path) {
        clear_step_state();
        clear_step_runup_state(true);
    }

    step_runup_goal_reached_ = (prev_state == FsmState::STEP_RUNUP) && is_step_runup_goal_reached(effective_input);

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
        update_step_release(*effective_input.global_path, current_u);
    }

    bool request_replan_now = false;
    if (has_path && (prev_state == FsmState::FOLLOW || prev_state == FsmState::STEP_RUNUP)) {
        request_replan_now = check_follow_projection_guard(effective_input, *effective_input.global_path, current_u)
            || check_no_progress(effective_input, current_u)
            || ((prev_state == FsmState::FOLLOW) && check_step_block_replan(effective_input, *effective_input.global_path, current_u));
    }

    if (!request_replan_now && has_path && prev_state == FsmState::FOLLOW) {
        request_replan_now = prepare_follow_step_behavior(effective_input, *effective_input.global_path, current_u);
    }

    const bool dist_reached = has_path && ((effective_input.chassis_pose_map.head<2>() - effective_input.global_path->evaluate(1.0)).norm() < nav_params_.stop_threshold_dist);
    const bool u_reached = has_path && (current_u > nav_params_.stop_threshold_u);

    FsmInput fsm_input;
    fsm_input.has_path = has_path;
    fsm_input.has_new_path = has_new_path;
    fsm_input.fixed_goal_flag = effective_input.fixed_goal;
    fsm_input.reach_goal = dist_reached || u_reached;
    fsm_input.step_active = is_step_active();
    fsm_input.step_runup_requested = step_runup_request_pending_;
    fsm_input.step_runup_completed = step_runup_goal_reached_;
    fsm_input.replan_requested = request_replan_now;
    fsm_input.spin_requested = effective_input.spin_requested;
    fsm_input.spin_high_priority = effective_input.spin_high_priority;
    const bool hazard_allowed = (prev_state == FsmState::IDLE) || (prev_state == FsmState::SPIN)
        || (prev_state == FsmState::HAZARD_RECOVERY) || (prev_state == FsmState::STEP_RUNUP);
    fsm_input.is_hazard = hazard_allowed && compute_is_hazard(effective_input);
    fsm_input.is_stuck = check_stuck(effective_input);
    fsm_input.is_recovery_safe = update_recovery_safe_flag(effective_input);
    fsm_input.velocity = last_cmd_.x();
    fsm_input.omega = last_cmd_.y();
    fsm_input.stamp = effective_input.stamp;

    const FsmOutput fsm_output = control_fsm_->update(fsm_input);
    const FsmState state = fsm_output.state;
    on_state_transition(prev_state, state);
    last_fsm_state_ = state;

    ControlOutput output;
    switch (state) {
        case FsmState::IDLE: output = execute_idle(effective_input); break;
        case FsmState::FOLLOW: output = execute_follow(effective_input); break;
        case FsmState::SPIN: output = execute_spin(effective_input); break;
        case FsmState::STOPPING: output = execute_stop(effective_input); break;
        case FsmState::HAZARD_RECOVERY: output = execute_recovery(effective_input); break;
        case FsmState::STUCK_REVERSE: output = execute_stuck_reverse(effective_input); break;
        case FsmState::FIXED: output = execute_fixed(effective_input); break;
        case FsmState::STEP_RUNUP: output = execute_step_runup(effective_input); break;
        case FsmState::WAIT_REPLAN: output = execute_stop(effective_input); break;
        case FsmState::STEPPING: output = execute_follow(effective_input); break;
        case FsmState::DEAD: output = execute_idle(effective_input); break;
    }

    if (had_step_runup_request && state != FsmState::STEP_RUNUP) {
        clear_step_runup_state();
    }

    output.fsm_state = state;
    output.consume_global_path |= fsm_output.consume_global_path;
    output.request_replan = fsm_output.request_replan;
    output.keep_goal_on_path_consume = output.request_replan || state == FsmState::WAIT_REPLAN;
    if (output.consume_global_path) {
        last_reference_u_ = 0.0;
        clear_step_state();
        clear_step_runup_state(state != FsmState::WAIT_REPLAN);
    }

    // 4. 同步已发布指令到 FSM / MPC，并在非 MPC 状态时重置 MPC 的 warm start
    if (output.valid) {
        last_cmd_ = Eigen::Vector2d(output.velocity, output.omega);
        mpc_controller_->set_last_cmd(last_cmd_);
        const bool non_mpc_state = (state == FsmState::IDLE) || (state == FsmState::SPIN) || (state == FsmState::STUCK_REVERSE);
        if (non_mpc_state) {
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

    if (pending_step_rollout_path_map_) {
        out.step_rollout_path_map = pending_step_rollout_path_map_;
        pending_step_rollout_path_map_ = std::nullopt;
    }

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
    update_step_lookahead_distance(prediction);

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

    if (active_step_target_) {
        RCLCPP_DEBUG(
            logger_,
            "step: mode=%s step_dist=%d u=%.3f enter=%.3f exit=%.3f inside=%d",
            mode_label(active_step_command_ ? active_step_command_->mode : ChassisMode::NORMAL),
            out.step_dist_cm, u0,
            active_step_target_->enter_u, active_step_target_->exit_u,
            (u0 >= active_step_target_->enter_u && u0 < active_step_target_->exit_u) ? 1 : 0
        );
    }

    return out;
}

ControlOutput MainController::execute_step_runup(const ControlInput& input) {
    ControlOutput out;
    out.mode = ChassisMode::NORMAL;
    if (!step_runup_context_ || !input.final_cost_map || !input.masked_direction_map) {
        out.velocity = 0.0;
        out.omega = 0.0;
        out.valid = true;
        return out;
    }

    auto start_time = std::chrono::steady_clock::now();
    const auto result = mpc_controller_->solve_hold(
        step_runup_context_->goal_map,
        input.chassis_pose_map,
        input.chassis_state,
        *input.final_cost_map,
        *input.masked_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(StepRunup) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(StepRunup) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.predicted_path_map = std::get<1>(*result).path_map;
    out.predicted_v = std::get<1>(*result).v_pred;
    out.predicted_w = std::get<1>(*result).w_pred;
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

void MainController::on_state_transition(const FsmState prev, const FsmState next) {
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
        mpc_controller_->reset_warm_start();
    }

    if (prev == FsmState::STEP_RUNUP && next != FsmState::STEP_RUNUP) {
        if (step_runup_goal_reached_ && step_runup_context_) {
            last_completed_step_runup_target_ = step_runup_context_->step_target;
        }
        clear_step_runup_state();
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

    // Hold 求解器由 HAZARD_RECOVERY / FIXED / STEP_RUNUP 共享，三者之间切换不应互相清空 warm start。
    const bool next_uses_hold = (next == FsmState::FIXED) || (next == FsmState::STEP_RUNUP);
    const bool prev_uses_hold = (prev == FsmState::FIXED) || (prev == FsmState::HAZARD_RECOVERY) || (prev == FsmState::STEP_RUNUP);
    if (next_uses_hold && !prev_uses_hold) {
        mpc_controller_->reset_warm_start();
    }
}

void MainController::update_recovery_goal_if_needed(const ControlInput& input) {
    const auto& p = fsm_params_.recovery;
    if (!input.masked_global_cost_map || !input.masked_direction_map) return;

    const bool need_new = (!recovery_goal_map_) || (!recovery_goal_set_time_) || (std::chrono::duration<double>(input.stamp - *recovery_goal_set_time_).count() >= p.goal_timeout);

    if (!need_new) return;

    recovery_goal_map_ = RecoveryGoalPlanner::find_goal(
        p, *input.masked_global_cost_map, *input.masked_direction_map, input.chassis_pose_map
    );
    recovery_goal_set_time_ = input.stamp;

    if (!recovery_goal_map_) {
        RCLCPP_ERROR(logger_, "HAZARD_RECOVERY failed to find a recovery goal");
        return;
    }

    const auto s = RecoveryGoalPlanner::sample_fields(*input.masked_global_cost_map, *input.masked_direction_map, *recovery_goal_map_);
    if (!s) {
        RCLCPP_WARN(logger_, "HAZARD_RECOVERY new goal=(%.2f, %.2f) (field sample invalid)", recovery_goal_map_->x(), recovery_goal_map_->y());
        return;
    }

    if (RecoveryGoalPlanner::is_safe_goal(p, *s)) {
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
    const auto sample = RecoveryGoalPlanner::sample_fields(
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

    const auto sample = RecoveryGoalPlanner::sample_fields(
        *input.final_cost_map,
        *input.masked_direction_map,
        input.chassis_pose_map.head<2>()
    );
    if (!sample) {
        recovery_safe_since_ = std::nullopt;
        return false;
    }

    const bool safe_now = RecoveryGoalPlanner::is_safe_goal(p, *sample);
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

double MainController::advance_path_u_by_distance(const SplineD& path, const double start_u, const double distance) const {
    const double resolution = std::max(1e-3, nav_params_.step_detection.path_sample_resolution);
    const int steps = std::max(1, static_cast<int>(std::ceil(std::max(0.0, distance) / resolution)));
    double u = std::clamp(start_u, 0.0, 1.0);
    Eigen::Vector2d prev = path.evaluate(u);
    double travelled = 0.0;

    for (int i = 1; i <= steps && u < 1.0; ++i) {
        const double next_u = std::min(1.0, start_u + (1.0 - start_u) * static_cast<double>(i) / static_cast<double>(steps));
        const Eigen::Vector2d cur = path.evaluate(next_u);
        travelled += (cur - prev).norm();
        prev = cur;
        u = next_u;
        if (travelled >= distance) break;
    }
    return u;
}

std::optional<MainController::StepTargetObservation> MainController::detect_step_target_on_rollout(
    const std::vector<Eigen::Vector2d>& rollout_path_map,
    const DirectionMap& direction_map
) const {
    if (rollout_path_map.empty()) return std::nullopt;

    double travelled = 0.0;
    Eigen::Vector2d prev = rollout_path_map.front();
    for (size_t i = 0; i < rollout_path_map.size(); ++i) {
        const Eigen::Vector2d& pos = rollout_path_map[i];
        if (i > 0) {
            travelled += (pos - prev).norm();
            prev = pos;
        }

        Eigen::Vector2d tangent = Eigen::Vector2d::Zero();
        if (i + 1 < rollout_path_map.size()) {
            tangent = rollout_path_map[i + 1] - pos;
        } else if (i > 0) {
            tangent = pos - rollout_path_map[i - 1];
        }
        if (tangent.norm() < ANGLE_EPSILON) continue;

        const Eigen::Vector2d g = direction_map.map_coord_to_grid(pos);
        if (!direction_map.is_valid_coord(Eigen::Vector2i(
            static_cast<int>(std::floor(g.x())),
            static_cast<int>(std::floor(g.y()))))
        ) continue;

        const Eigen::Vector2d dir = direction_map.interpolate(g);
        const double norm = dir.norm();
        if (norm < nav_params_.step_detection.detect_norm_threshold) continue;

        const double dot = dir.normalized().dot(tangent.normalized());
        if (std::abs(dot) <= nav_params_.step_detection.detect_dot_threshold) continue;

        return StepTargetObservation {
            .distance_from_start = travelled,
            .pos_map = pos,
            .dir_map = dir,
            .direction = dot > 0.0 ? StepDirection::UP : StepDirection::DOWN,
        };
    }

    return std::nullopt;
}

double MainController::prediction_path_length(const MPCPrediction& prediction) {
    double length = 0.0;
    for (size_t i = 1; i < prediction.path_map.size(); ++i) {
        length += (prediction.path_map[i] - prediction.path_map[i - 1]).norm();
    }
    return length;
}

void MainController::update_step_lookahead_distance(const MPCPrediction& prediction) {
    const double raw_length = prediction_path_length(prediction);
    const double alpha = std::clamp(nav_params_.step_detection.lookahead.rollout_length_filter_alpha, 0.0, 1.0);
    filtered_follow_rollout_length_ = alpha * raw_length + (1.0 - alpha) * filtered_follow_rollout_length_;
    step_lookahead_distance_ = std::max(
        nav_params_.step_detection.lookahead.min_distance,
        filtered_follow_rollout_length_ + nav_params_.step_detection.lookahead.fixed_extension_distance
    );
}

double MainController::current_step_lookahead_distance() const {
    return std::max(nav_params_.step_detection.lookahead.min_distance, step_lookahead_distance_);
}

std::optional<MainController::PathStepTarget> MainController::detect_step_target_on_path(
    const SplineD& path,
    const double start_u,
    const DirectionMap& direction_map
) const {
    const double lookahead_distance = current_step_lookahead_distance();
    const double lookahead_u = advance_path_u_by_distance(path, start_u, lookahead_distance);
    const double resolution = std::max(1e-3, nav_params_.step_detection.path_sample_resolution);
    const int samples = std::max(1, static_cast<int>(std::ceil(lookahead_distance / resolution)));
    bool inside_step_region = false;
    double enter_u = 0.0;
    Eigen::Vector2d enter_pos = Eigen::Vector2d::Zero();
    Eigen::Vector2d enter_dir = Eigen::Vector2d::Zero();
    StepDirection enter_direction = StepDirection::UP;

    for (int i = 0; i <= samples; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(samples);
        const double u = start_u + (lookahead_u - start_u) * t;
        const Eigen::Vector2d pos = path.evaluate(u);
        const Eigen::Vector2d tangent = path.derivative(u, 1);
        if (tangent.norm() < ANGLE_EPSILON) continue;

        const Eigen::Vector2d g = direction_map.map_coord_to_grid(pos);
        if (!direction_map.is_valid_coord(Eigen::Vector2i(
            static_cast<int>(std::floor(g.x())),
            static_cast<int>(std::floor(g.y()))))
        ) continue;

        const Eigen::Vector2d dir = direction_map.interpolate(g);
        const double norm = dir.norm();

        if (norm >= nav_params_.step_detection.detect_norm_threshold) {
            const double dot = dir.normalized().dot(tangent.normalized());
            if (std::abs(dot) <= nav_params_.step_detection.detect_dot_threshold) {
                if (!inside_step_region) continue;
                continue;
            }
            if (inside_step_region) continue;

            inside_step_region = true;
            enter_u = u;
            enter_pos = pos;
            enter_dir = dir;
            enter_direction = dot > 0.0 ? StepDirection::UP : StepDirection::DOWN;
            continue;
        }

        if (!inside_step_region) continue;

        PathStepTarget target;
        target.path_version = path_version_;
        target.enter_u = enter_u;
        target.exit_u = u;
        target.enter_pos_map = enter_pos;
        target.exit_pos_map = pos;
        target.dir_map = enter_dir;
        target.direction = enter_direction;
        return target;
    }

    if (inside_step_region) {
        PathStepTarget target;
        target.path_version = path_version_;
        target.enter_u = enter_u;
        target.exit_u = lookahead_u;
        target.enter_pos_map = enter_pos;
        target.exit_pos_map = path.evaluate(lookahead_u);
        target.dir_map = enter_dir;
        target.direction = enter_direction;
        return target;
    }

    return std::nullopt;
}

bool MainController::is_same_step_target(const PathStepTarget& lhs, const PathStepTarget& rhs) const {
    if (lhs.path_version != rhs.path_version) return false;
    if (lhs.direction != rhs.direction) return false;
    return (lhs.enter_pos_map - rhs.enter_pos_map).norm() <= nav_params_.step_detection.target_match_distance;
}

bool MainController::is_same_step_target(const PathStepTarget& target, const StepTargetObservation& observation) const {
    if (target.direction != observation.direction) return false;
    return (target.enter_pos_map - observation.pos_map).norm() <= nav_params_.step_detection.rollout_match_distance;
}

void MainController::clear_step_state() {
    const bool had_latch = active_step_target_.has_value();
    pending_step_target_detection_ = std::nullopt;
    pending_step_target_on_count_ = 0;
    active_step_target_ = std::nullopt;
    active_step_command_ = std::nullopt;
    step_locked_path_ = std::nullopt;
    step_locked_fixed_goal_ = false;
    step_locked_fixed_goal_pos_ = Eigen::Vector2d::Zero();
    if (had_latch) {
        RCLCPP_DEBUG(logger_, "Step decision cleared (had active latch)");
    }
}

void MainController::clear_step_runup_state(const bool clear_last_completed_target) {
    step_runup_request_pending_ = false;
    step_runup_goal_reached_ = false;
    step_runup_context_ = std::nullopt;
    if (clear_last_completed_target) {
        last_completed_step_runup_target_ = std::nullopt;
    }
}

void MainController::update_step_state_for_path_change(const bool has_new_path) {
    if (!has_new_path) return;
    path_version_++;
    clear_step_state();
    clear_step_runup_state(true);
}

void MainController::update_step_release(const SplineD& path, const double current_u) {
    (void)path;
    if (!active_step_target_) return;
    if (active_step_target_->path_version != path_version_) {
        RCLCPP_DEBUG(logger_, "Step released: path version changed (target_v=%d != cur_v=%d)", active_step_target_->path_version, path_version_);
        clear_step_state();
        return;
    }
    if (current_u >= active_step_target_->exit_u) {
        RCLCPP_DEBUG(
            logger_,
            "Step released: passed step (u=%.3f >= exit_u=%.3f)",
            current_u,
            active_step_target_->exit_u
        );
        clear_step_state();
    }
}

void MainController::extend_active_step_exit(const SplineD& path, const DirectionMap& direction_map) {
    if (!active_step_target_) return;
    if (path_version_ != active_step_target_->path_version) return;

    const double resolution = std::max(1e-3, nav_params_.step_detection.path_sample_resolution);
    const double lookahead_dist = current_step_lookahead_distance();
    const double scan_start_u = active_step_target_->exit_u + 1e-4;
    const double scan_end_u = advance_path_u_by_distance(path, scan_start_u, lookahead_dist);
    if (scan_end_u <= scan_start_u) return;

    const int samples = std::max(1, static_cast<int>(std::ceil(lookahead_dist / resolution)));
    double farthest_step_u = -1.0;
    Eigen::Vector2d farthest_step_pos = Eigen::Vector2d::Zero();

    for (int i = 0; i <= samples; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(samples);
        const double u = scan_start_u + (scan_end_u - scan_start_u) * t;
        if (u <= active_step_target_->exit_u) continue;

        const Eigen::Vector2d pos = path.evaluate(u);
        const Eigen::Vector2d tangent = path.derivative(u, 1);
        if (tangent.norm() < ANGLE_EPSILON) continue;

        const Eigen::Vector2d g = direction_map.map_coord_to_grid(pos);
        if (!direction_map.is_valid_coord(Eigen::Vector2i(
            static_cast<int>(std::floor(g.x())),
            static_cast<int>(std::floor(g.y()))))
        ) continue;

        const Eigen::Vector2d dir = direction_map.interpolate(g);
        if (dir.norm() < nav_params_.step_detection.detect_norm_threshold) continue;

        const double dot = dir.normalized().dot(tangent.normalized());
        if (std::abs(dot) <= nav_params_.step_detection.detect_dot_threshold) continue;

        const StepDirection sample_dir = dot > 0.0 ? StepDirection::UP : StepDirection::DOWN;
        if (sample_dir != active_step_target_->direction) continue;

        farthest_step_u = u;
        farthest_step_pos = pos;
    }

    if (farthest_step_u > active_step_target_->exit_u) {
        active_step_target_->exit_u = farthest_step_u;
        active_step_target_->exit_pos_map = farthest_step_pos;
    }
}

bool MainController::is_currently_inside_active_step(const double current_u) const {
    return active_step_target_ && current_u >= active_step_target_->enter_u && current_u < active_step_target_->exit_u;
}

uint8_t MainController::compute_step_distance_cm(
    const ControlInput& input,
    const double current_u,
    const MPCPrediction& prediction
) const {
    if (!active_step_command_ || !is_step_mode(active_step_command_->mode)) {
        return 0;
    }
    if (!input.masked_direction_map) {
        return 255;
    }
    if (is_currently_inside_active_step(current_u)) {
        return 0;
    }

    const auto observed = detect_step_target_on_rollout(prediction.path_map, *input.masked_direction_map);
    if (!observed) {
        return 255;
    }
    if (active_step_target_ && !is_same_step_target(*active_step_target_, *observed)) {
        return 255;
    }
    if (observed->distance_from_start > 2.55) {
        return 255;
    }

    const double adjusted_distance = observed->distance_from_start + nav_params_.step_dist_offset;
    const int64_t rounded_cm = std::lround(adjusted_distance * 100.0);
    return static_cast<uint8_t>(std::clamp<int64_t>(rounded_cm, 0, 255));
}

std::optional<MainController::PathStepTarget> MainController::try_latch_step_target(
    const SplineD& path,
    const double current_u,
    const DirectionMap& direction_map
) {
    const auto detected_target = detect_step_target_on_path(path, current_u, direction_map);
    if (!detected_target) {
        pending_step_target_detection_ = std::nullopt;
        pending_step_target_on_count_ = 0;
        return std::nullopt;
    }

    if (pending_step_target_detection_ && is_same_step_target(*pending_step_target_detection_, *detected_target)) {
        pending_step_target_on_count_++;
        pending_step_target_detection_ = detected_target;
    } else {
        pending_step_target_detection_ = detected_target;
        pending_step_target_on_count_ = 1;
    }

    if (pending_step_target_on_count_ < nav_params_.step_detection.latch_threshold) {
        return std::nullopt;
    }

    RCLCPP_DEBUG(
        logger_, "Step target latched: enter_u=%.3f exit_u=%.3f at (%.2f, %.2f) path_version=%d dir=%s",
        pending_step_target_detection_->enter_u,
        pending_step_target_detection_->exit_u,
        pending_step_target_detection_->enter_pos_map.x(),
        pending_step_target_detection_->enter_pos_map.y(),
        pending_step_target_detection_->path_version,
        pending_step_target_detection_->direction == StepDirection::UP ? "UP" : "DOWN"
    );

    PathStepTarget target = *pending_step_target_detection_;
    pending_step_target_detection_ = std::nullopt;
    pending_step_target_on_count_ = 0;
    return target;
}

MainController::StepRunupDecision MainController::evaluate_step_runup(
    const ControlInput& input,
    const SplineD& path,
    const double current_u,
    const PathStepTarget& target,
    const ActiveStepMode& step_command
) const {
    StepRunupDecision decision;
    const auto rollout = mpc_controller_->rollout_step_arrival(
        path,
        input.chassis_pose_map,
        input.chassis_state,
        current_u,
        step_command,
        target.enter_u
    );
    if (!rollout) {
        RCLCPP_WARN(logger_, "Step run-up rollout failed: %s", rollout.error().c_str());
        return decision;
    }
    decision.debug_rollout_path_map = rollout->prediction.path_map;
    if (!rollout->reached_target) {
        return decision;
    }
    if (rollout->deficit <= nav_params_.step_runup.velocity_deficit_threshold) {
        return decision;
    }

    const auto maybe_index = [&]() -> int {
        if (rollout->prediction.path_map.empty()) return -1;
        int best_idx = -1;
        double best_dist = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < rollout->prediction.path_map.size(); ++i) {
            const double d = (rollout->prediction.path_map[i] - target.enter_pos_map).squaredNorm();
            if (d < best_dist) {
                best_dist = d;
                best_idx = static_cast<int>(i);
            }
        }
        return best_idx;
    }();
    RCLCPP_DEBUG(
        logger_,
        "Step run-up rollout: idx=%d arrival_v=%.2f target_v=%.2f deficit=%.2f current_v=%.2f",
        maybe_index,
        rollout->arrival_velocity,
        rollout->target_velocity,
        rollout->deficit,
        input.chassis_state.velocity
    );

    if (last_completed_step_runup_target_ && is_same_step_target(*last_completed_step_runup_target_, target)) {
        RCLCPP_WARN(
            logger_,
            "Step run-up loop detected at (%.2f, %.2f), cancelling path",
            target.enter_pos_map.x(),
            target.enter_pos_map.y()
        );
        decision.request_replan = true;
        return decision;
    }

    const auto& profiles = mpc_controller_->params().follow.mode_profiles;
    const MPCMotionConstraints* motion = nullptr;
    switch (step_command.mode) {
        case ChassisMode::STEP_UP_LEG: motion = &profiles.leg_up.motion_constraints; break;
        case ChassisMode::STEP_UP_JUMP: motion = &profiles.jump_up.motion_constraints; break;
        default: motion = &profiles.normal.motion_constraints; break;
    }

    const double required_distance = std::max(
        0.0,
        (step_command.target_velocity * step_command.target_velocity - rollout->arrival_velocity * rollout->arrival_velocity)
            / std::max(2.0 * motion->acc_max, 1e-6)
    );
    const double preferred_backoff_distance = std::clamp(
        required_distance,
        nav_params_.step_runup.search.radius_min,
        nav_params_.step_runup.search.radius_max
    );
    const auto goal = find_step_runup_goal(input, target, preferred_backoff_distance);
    if (!goal) {
        RCLCPP_WARN(
            logger_,
            "Step run-up requested but no valid run-up goal found at target (%.2f, %.2f)",
            target.enter_pos_map.x(),
            target.enter_pos_map.y()
        );
        decision.request_replan = true;
        return decision;
    }

    decision.context = StepRunupContext {
        .step_target = target,
        .step_command = step_command,
        .goal_map = *goal,
        .preferred_backoff_distance = preferred_backoff_distance,
        .predicted_arrival_velocity = rollout->arrival_velocity,
        .velocity_deficit = rollout->deficit,
    };
    return decision;
}

std::optional<Eigen::Vector2d> MainController::find_step_runup_goal(
    const ControlInput& input,
    const PathStepTarget& target,
    const double preferred_backoff_distance
) const {
    if (!input.final_cost_map || !input.masked_direction_map) return std::nullopt;

    const double dir_norm = target.dir_map.norm();
    if (dir_norm < ANGLE_EPSILON) return std::nullopt;

    const Eigen::Vector2d origin = input.chassis_pose_map.head<2>();
    const Eigen::Vector2d backward_dir = -target.dir_map / dir_norm;
    const double radius_min = std::min(nav_params_.step_runup.search.radius_min, nav_params_.step_runup.search.radius_max);
    const double radius_max = std::max(nav_params_.step_runup.search.radius_min, nav_params_.step_runup.search.radius_max);
    const int radius_samples = std::max(1, nav_params_.step_runup.search.radius_samples);
    const int angle_samples = std::max(1, nav_params_.step_runup.search.angle_samples);

    std::optional<Eigen::Vector2d> best_goal;
    double best_score = std::numeric_limits<double>::infinity();

    for (int ri = 0; ri < radius_samples; ++ri) {
        const double rt = (radius_samples == 1) ? 0.0 : static_cast<double>(ri) / static_cast<double>(radius_samples - 1);
        const double radius = radius_min + (radius_max - radius_min) * rt;

        for (int ai = 0; ai < angle_samples; ++ai) {
            const double at = (angle_samples == 1) ? 0.5 : static_cast<double>(ai) / static_cast<double>(angle_samples - 1);
            const double angle = -nav_params_.step_runup.search.sector_half_angle_rad
                + 2.0 * nav_params_.step_runup.search.sector_half_angle_rad * at;
            const Eigen::Vector2d dir = rotate_vector(backward_dir, angle);
            const Eigen::Vector2d candidate = target.enter_pos_map + dir * radius;

            const auto field = RecoveryGoalPlanner::sample_fields(*input.final_cost_map, *input.masked_direction_map, candidate);
            if (!field) continue;
            if (field->cost > nav_params_.step_runup.search.candidate_cost_max) continue;
            if (field->step_norm > nav_params_.step_runup.search.safe_step_norm_threshold) continue;

            const auto line_cost = max_cost_along_segment(
                *input.final_cost_map,
                origin,
                candidate,
                nav_params_.follow_proj_guard.cost_samples
            );
            if (!line_cost || *line_cost > nav_params_.step_runup.search.line_cost_max) continue;

            const auto path_score = score_runup_path_integral(
                nav_params_,
                *input.final_cost_map,
                *input.masked_direction_map,
                origin,
                candidate
            );
            if (!path_score) continue;

            const double score = *path_score
                + nav_params_.step_runup.search.radius_preference_weight * std::abs(radius - preferred_backoff_distance);
            if (score < best_score) {
                best_score = score;
                best_goal = candidate;
            }
        }
    }

    return best_goal;
}

bool MainController::is_step_runup_goal_reached(const ControlInput& input) const {
    if (!step_runup_context_) return false;
    return (input.chassis_pose_map.head<2>() - step_runup_context_->goal_map).norm() <= nav_params_.step_runup.goal_tolerance;
}

double MainController::step_speed_from_level(const uint8_t speed_level) const {
    return mpc_controller_->params().follow.terrain_limits.step_speed_levels[std::min<size_t>(speed_level, 3)];
}

std::optional<ActiveStepMode> MainController::build_step_command(const PathStepTarget& target, const DirectionMap& direction_map) const {
    const auto mode_info = direction_map.step_mode_info_at(direction_map.map_coord_to_grid(target.enter_pos_map));
    const StepTraversalMode mode = target.direction == StepDirection::UP ? mode_info.up_mode : mode_info.down_mode;
    const uint8_t speed_level = target.direction == StepDirection::UP ? mode_info.up_speed_level : mode_info.down_speed_level;
    if (!is_step_traversal_allowed(mode)) {
        return std::nullopt;
    }

    return ActiveStepMode {
        .mode = [&]() {
            if (target.direction == StepDirection::UP) {
                return mode == StepTraversalMode::LEG ? ChassisMode::STEP_UP_LEG : ChassisMode::STEP_UP_JUMP;
            }
            return mode == StepTraversalMode::LEG ? ChassisMode::STEP_DOWN_LEG : ChassisMode::STEP_DOWN_JUMP;
        }(),
        .target_velocity = step_speed_from_level(speed_level),
    };
}

std::optional<ActiveStepMode> MainController::current_active_step_mode() const {
    return active_step_command_;
}

bool MainController::is_step_active() const {
    return active_step_command_.has_value();
}

// ═══════════════════ Follow 路标点无进度检测 ══════════════════

void MainController::recompute_follow_landmarks(const SplineD& path) {
    follow_landmarks_u_.clear();
    const int N = 500;
    const double spacing = std::max(0.1, nav_params_.no_progress_guard.landmark_spacing);
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
