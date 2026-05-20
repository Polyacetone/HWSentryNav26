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
    pending_step_release_count_ = 0;
    active_step_target_ = std::nullopt;
    active_step_command_ = std::nullopt;
    step_locked_path_ = std::nullopt;
    step_locked_fixed_goal_ = false;
    step_locked_fixed_goal_pos_ = Eigen::Vector2d::Zero();
    if (had_latch) {
        RCLCPP_DEBUG(logger_, "Step decision cleared (had active latch)");
    }
}

void MainController::update_step_state_for_path_change(const bool has_new_path) {
    if (!has_new_path) return;
    path_version_++;
    last_reference_u_ = 0.0;
    clear_step_state();
}

void MainController::update_step_release(const SplineD& path, const double current_u) {
    if (!active_step_target_) return;
    if (active_step_target_->path_version != path_version_) {
        RCLCPP_DEBUG(logger_, "Step released: path version changed (target_v=%d != cur_v=%d)", active_step_target_->path_version, path_version_);
        clear_step_state();
        return;
    }
    const double release_u = advance_path_u_by_distance(path, active_step_target_->exit_u, nav_params_.step_detection.exit_advance_distance);
    if (current_u >= release_u) {
        pending_step_release_count_++;
        if (pending_step_release_count_ >= nav_params_.step_detection.release_threshold) {
            RCLCPP_DEBUG(
                logger_,
                "Step released: passed exit (u=%.3f >= release_u=%.3f (exit_u=%.3f + adv=%.2fm), release_count=%d >= release_threshold=%d)",
                current_u,
                release_u,
                active_step_target_->exit_u,
                nav_params_.step_detection.exit_advance_distance,
                pending_step_release_count_,
                nav_params_.step_detection.release_threshold
            );
            clear_step_state();
        }
        return;
    }
    pending_step_release_count_ = 0;
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
        .step_entry_u = target.enter_u,
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

    if (!active_step_target_) {
        if (const auto step_target = try_latch_step_target(path, current_u, *input.masked_direction_map)) {
            const auto step_command = build_step_command(*step_target, *input.masked_direction_map);
            if (step_command) {
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
                RCLCPP_WARN(logger_, "Step target latched but alpha mode forbids traversal");
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

} // namespace path_follower
