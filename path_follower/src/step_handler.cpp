#include <path_follower/main_controller.hpp>
#include <rclcpp/logging.hpp>

namespace path_follower {

namespace {

constexpr double ANGLE_EPSILON = 1e-6;

using path_follower::mode_label;
using path_follower::is_step_mode;

} // anonymous namespace

// ═══════════════════ 工具函数 ══════════════════════════════

double MainController::advance_path_u_by_distance(const SplineD& path, const double start_u, const double distance) const {
    const double resolution = std::max(1e-3, nav_params_.step_detection.path_sample_resolution);
    double u = std::clamp(start_u, 0.0, 1.0);
    double travelled = 0.0;

    // 使用 B-spline 导数信息做弧长推进（而非等参数间隔）
    while (u < 1.0 && travelled < distance) {
        const Eigen::Vector2d d1 = path.derivative(u, 1);
        const double speed = d1.norm();
        if (speed < 1e-12) {
            // 零导数区域（退化情况），用最小步进
            u = std::min(1.0, u + 1e-3);
            continue;
        }
        // du  = 期望弧长 / 当前参数速度
        const double du = resolution / speed;
        const double next_u = std::min(1.0, u + du);
        const Eigen::Vector2d cur = path.evaluate(next_u);
        travelled += (cur - path.evaluate(u)).norm();
        u = next_u;
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

std::optional<MainController::PathStepTarget> MainController::detect_step_target_on_path(
    const SplineD& path,
    const double start_u,
    const DirectionMap& direction_map
) const {
    const double lookahead_distance = nav_params_.step_detection.lookahead_distance;
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
    step_latch_start_time_ = std::nullopt;
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

void MainController::update_step_release(const SplineD& path, const double current_u, const std::chrono::steady_clock::time_point stamp) {
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
        return;
    }

    // TTL 超时释放：台阶锁存超过 latch_ttl 仍未退出，认为卡死
    // 通过 step_ttl_just_expired_ 标志通知 FSM，由 STEPPING 状态直接转入 STUCK_REVERSE
    if (step_latch_start_time_) {
        const double elapsed = std::chrono::duration<double>(
            stamp - *step_latch_start_time_
        ).count();
        if (elapsed >= nav_params_.latch_ttl) {
            RCLCPP_WARN(
                logger_,
                "Step TTL expired (%.1f >= %.1f s), signaling direct STUCK_REVERSE transition",
                elapsed, nav_params_.latch_ttl
            );
            clear_step_state();
            step_ttl_just_expired_ = true;
        }
    }
}

void MainController::extend_active_step_exit(const SplineD& path, const DirectionMap& direction_map) {
    if (!active_step_target_) return;
    if (path_version_ != active_step_target_->path_version) return;

    const double resolution = std::max(1e-3, nav_params_.step_detection.path_sample_resolution);
    const double lookahead_dist = nav_params_.step_detection.lookahead_distance;
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
        case ChassisMode::STEP_UP_LEG_SHORT: motion = &profiles.up.short_leg.motion_constraints; break;
        case ChassisMode::STEP_UP_JUMP:      motion = &profiles.up.jump.motion_constraints; break;
        case ChassisMode::STEP_UP_LEG_LONG:  motion = &profiles.up.long_leg.motion_constraints; break;
        case ChassisMode::STEP_DOWN_LEG_SHORT: motion = &profiles.down.short_leg.motion_constraints; break;
        case ChassisMode::STEP_DOWN_JUMP:    motion = &profiles.down.jump.motion_constraints; break;
        default:                             motion = &profiles.normal.motion_constraints; break;
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
            const Eigen::Vector2d dir = recovery_helpers::rotate_vector(backward_dir, angle);
            const Eigen::Vector2d candidate = target.enter_pos_map + dir * radius;

            const auto field = recovery_helpers::sample_fields(*input.final_cost_map, *input.masked_direction_map, candidate);
            if (!field) continue;
            if (field->cost > nav_params_.step_runup.search.candidate_cost_max) continue;
            if (field->step_norm > nav_params_.step_runup.search.safe_step_norm_threshold) continue;

            const auto line_cost = recovery_helpers::max_cost_along_segment(
                *input.final_cost_map,
                origin,
                candidate,
                nav_params_.follow_proj_guard.cost_samples
            );
            if (!line_cost || *line_cost > nav_params_.step_runup.search.line_cost_max) continue;

            const auto path_score = recovery_helpers::score_runup_path_integral(
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
                switch (mode) {
                    case StepTraversalMode::LEG_LONG:  return ChassisMode::STEP_UP_LEG_LONG;
                    case StepTraversalMode::LEG_SHORT: return ChassisMode::STEP_UP_LEG_SHORT;
                    case StepTraversalMode::JUMP:      return ChassisMode::STEP_UP_JUMP;
                    default:                           return ChassisMode::NORMAL;
                }
            }
            switch (mode) {
                case StepTraversalMode::LEG_SHORT: return ChassisMode::STEP_DOWN_LEG_SHORT;
                case StepTraversalMode::JUMP:      return ChassisMode::STEP_DOWN_JUMP;
                default:                           return ChassisMode::NORMAL;
            }
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

bool MainController::prepare_follow_step_behavior(
    const ControlInput& input,
    const SplineD& path,
    const double current_u
) {
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
                step_latch_start_time_ = input.stamp;
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
        step_latch_start_time_ = input.stamp;
    }

    return false;
}

} // namespace path_follower
