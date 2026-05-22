#include <path_follower/main_controller.hpp>
#include <rclcpp/logging.hpp>

namespace path_follower {

namespace {

constexpr double ANGLE_EPSILON = 1e-6;
constexpr double U_EPSILON = 1e-6;

using path_follower::is_step_mode;
using path_follower::mode_label;

struct StepSample {
    double u = 0.0;
    Eigen::Vector2d pos_map = Eigen::Vector2d::Zero();
    Eigen::Vector2d dir_map = Eigen::Vector2d::Zero();
    StepDirection direction = StepDirection::UP;
};

bool step_direction_matches(
    const Eigen::Vector2d& tangent,
    const Eigen::Vector2d& dir,
    const StepDetectionParams& params,
    StepDirection* direction_out = nullptr
) {
    if (tangent.norm() < ANGLE_EPSILON) return false;
    if (dir.norm() < params.detect_norm_threshold) return false;

    const double dot = dir.normalized().dot(tangent.normalized());
    if (std::abs(dot) <= params.detect_dot_threshold) return false;

    if (direction_out) {
        *direction_out = dot > 0.0
            ? StepDirection::UP
            : StepDirection::DOWN;
    }
    return true;
}

} // anonymous namespace

double MainController::advance_path_u_by_distance(const SplineD& path, const double start_u, const double distance) const {
    const double resolution = std::max(1e-3, nav_params_.step_detection.path_sample_resolution);
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
}

double MainController::retreat_path_u_by_distance(const SplineD& path, const double start_u, const double distance) const {
    const double resolution = std::max(1e-3, nav_params_.step_detection.path_sample_resolution);
    double u = std::clamp(start_u, 0.0, 1.0);
    double travelled = 0.0;

    while (u > 0.0 && travelled < distance) {
        const Eigen::Vector2d d1 = path.derivative(u, 1);
        const double speed = d1.norm();
        if (speed < 1e-12) {
            u = std::max(0.0, u - 1e-3);
            continue;
        }
        const double du = resolution / speed;
        const double next_u = std::max(0.0, u - du);
        travelled += (path.evaluate(u) - path.evaluate(next_u)).norm();
        u = next_u;
    }
    return u;
}

std::vector<MainController::StepPlanSegment> MainController::build_step_plan(
    const SplineD& path,
    const DirectionMap& direction_map
) const {
    const double resolution = std::max(1e-3, nav_params_.step_detection.path_sample_resolution);

    double estimated_length = 0.0;
    Eigen::Vector2d prev = path.evaluate(0.0);
    constexpr int length_estimate_samples = 100;
    for (int i = 1; i <= length_estimate_samples; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(length_estimate_samples);
        const Eigen::Vector2d pos = path.evaluate(u);
        estimated_length += (pos - prev).norm();
        prev = pos;
    }

    const int sample_count = std::max(2, static_cast<int>(std::ceil(estimated_length / resolution)) + 1);
    std::vector<StepSample> step_samples;
    step_samples.reserve(static_cast<size_t>(sample_count));

    for (int i = 0; i < sample_count; ++i) {
        const double u = sample_count == 1
            ? 0.0
            : static_cast<double>(i) / static_cast<double>(sample_count - 1);
        const Eigen::Vector2d pos = path.evaluate(u);
        const Eigen::Vector2d tangent = path.derivative(u, 1);
        const Eigen::Vector2d g = direction_map.map_coord_to_grid(pos);
        if (!direction_map.is_valid_coord(g)) continue;

        StepDirection direction = StepDirection::UP;
        const Eigen::Vector2d dir = direction_map.interpolate(g);
        if (!step_direction_matches(tangent, dir, nav_params_.step_detection, &direction)) {
            continue;
        }

        step_samples.push_back(StepSample {
            .u = u,
            .pos_map = pos,
            .dir_map = dir,
            .direction = direction,
        });
    }

    std::vector<StepPlanSegment> plan;
    if (step_samples.empty()) return plan;

    size_t segment_begin = 0;
    while (segment_begin < step_samples.size()) {
        size_t segment_end = segment_begin + 1;
        while (segment_end < step_samples.size()) {
            const bool direction_changed = step_samples[segment_end].direction != step_samples[segment_begin].direction;
            const bool gap_too_large = quadratic_bspline_arc_length(
                path.getControlPoints(),
                step_samples[segment_end - 1].u,
                step_samples[segment_end].u
            ) > resolution * 1.5;
            if (direction_changed || gap_too_large) {
                break;
            }
            ++segment_end;
        }

        const StepSample& first = step_samples[segment_begin];
        const StepSample& last = step_samples[segment_end - 1];
        auto command = build_step_command(first.direction, first.pos_map, first.u, direction_map);
        if (command) {
            StepPlanSegment segment;
            segment.path_version = path_version_;
            segment.step_enter_u = first.u;
            segment.step_exit_u = last.u;
            segment.step_enter_pos_map = first.pos_map;
            segment.step_exit_pos_map = last.pos_map;
            segment.dir_map = first.dir_map;
            segment.direction = first.direction;
            segment.command = *command;
            plan.push_back(segment);
        } else {
            RCLCPP_WARN(
                logger_,
                "Step plan skipped forbidden segment at (%.2f, %.2f) dir=%s",
                first.pos_map.x(),
                first.pos_map.y(),
                first.direction == StepDirection::UP ? "UP" : "DOWN"
            );
        }

        segment_begin = segment_end;
    }

    if (plan.empty()) return plan;

    for (size_t i = 0; i < plan.size(); ++i) {
        StepPlanSegment& segment = plan[i];
        segment.prepare_u = retreat_path_u_by_distance(path, segment.step_enter_u, nav_params_.step_detection.prepare_distance);
        segment.active_u = retreat_path_u_by_distance(path, segment.step_enter_u, nav_params_.step_detection.active_distance);
        segment.release_u = advance_path_u_by_distance(path, segment.step_exit_u, nav_params_.step_detection.release_distance);
    }

    for (size_t i = 1; i < plan.size(); ++i) {
        StepPlanSegment& prev_segment = plan[i - 1];
        StepPlanSegment& cur_segment = plan[i];

        if (cur_segment.prepare_u < prev_segment.release_u) {
            cur_segment.prepare_u = prev_segment.release_u;
        }
        if (cur_segment.active_u < cur_segment.prepare_u) {
            cur_segment.active_u = cur_segment.prepare_u;
        }

        if (prev_segment.active_u > prev_segment.step_enter_u) {
            prev_segment.active_u = prev_segment.step_enter_u;
        }
        if (cur_segment.prepare_u > cur_segment.step_enter_u) {
            cur_segment.prepare_u = cur_segment.step_enter_u;
        }
        if (cur_segment.active_u > cur_segment.step_enter_u) {
            cur_segment.active_u = cur_segment.step_enter_u;
        }
    }

    for (StepPlanSegment& segment : plan) {
        segment.prepare_u = std::clamp(segment.prepare_u, 0.0, segment.step_enter_u);
        segment.active_u = std::clamp(segment.active_u, segment.prepare_u, segment.step_enter_u);
        segment.step_exit_u = std::max(segment.step_exit_u, segment.step_enter_u);
        segment.release_u = std::max(segment.release_u, segment.step_exit_u);
        segment.command.step_entry_u = segment.step_enter_u;
        segment.command.prepare_u = segment.prepare_u;
        segment.command.active_u = segment.active_u;
        segment.command.release_u = segment.release_u;
    }

    return plan;
}

void MainController::clear_step_runtime_state() {
    active_step_segment_index_ = std::nullopt;
    step_locked_path_ = std::nullopt;
    step_locked_fixed_goal_ = false;
    step_locked_fixed_goal_pos_ = Eigen::Vector2d::Zero();
}

void MainController::clear_step_plan() {
    clear_step_runtime_state();
    step_plan_.clear();
}

void MainController::update_step_plan_for_path_change(
    const bool has_new_path,
    const std::optional<SplineD>& path,
    const DirectionMap* const direction_map
) {
    if (!has_new_path) return;

    path_version_++;
    last_reference_u_ = 0.0;
    clear_step_plan();

    if (!path || !direction_map) {
        return;
    }

    step_plan_ = build_step_plan(*path, *direction_map);
    if (!step_plan_.empty()) {
        RCLCPP_DEBUG(logger_, "Built step plan with %zu segments for path_version=%d", step_plan_.size(), path_version_);
    }
}

std::optional<size_t> MainController::find_active_step_segment_index(const double current_u) const {
    for (size_t i = 0; i < step_plan_.size(); ++i) {
        const StepPlanSegment& segment = step_plan_[i];
        if (current_u + U_EPSILON < segment.prepare_u) {
            break;
        }
        if (current_u < segment.release_u) {
            return i;
        }
    }
    return std::nullopt;
}

const MainController::StepPlanSegment* MainController::active_step_segment(const double current_u) const {
    const auto index = find_active_step_segment_index(current_u);
    if (!index) return nullptr;
    return &step_plan_[*index];
}

const MainController::StepPlanSegment* MainController::current_step_command_segment(const double current_u) const {
    const StepPlanSegment* const segment = active_step_segment(current_u);
    if (!segment) return nullptr;
    if (!is_step_mode(segment->command.mode)) return nullptr;
    return segment;
}

void MainController::update_active_step_segment(const ControlInput& input, const double current_u) {
    const auto next_index = find_active_step_segment_index(current_u);
    if (next_index == active_step_segment_index_) {
        return;
    }

    active_step_segment_index_ = next_index;
    if (!active_step_segment_index_) {
        step_locked_path_ = std::nullopt;
        step_locked_fixed_goal_ = false;
        step_locked_fixed_goal_pos_ = Eigen::Vector2d::Zero();
        return;
    }

    if (input.global_path) {
        step_locked_path_ = input.global_path;
        step_locked_fixed_goal_ = input.fixed_goal;
        step_locked_fixed_goal_pos_ = input.fixed_goal_pos;
    }

    const StepPlanSegment& segment = step_plan_[*active_step_segment_index_];
    RCLCPP_DEBUG(
        logger_,
        "Activated step segment #%zu: prepare=[%.3f, %.3f) step=[%.3f, %.3f] active_u=%.3f mode=%s",
        *active_step_segment_index_,
        segment.prepare_u,
        segment.release_u,
        segment.step_exit_u,
        segment.active_u,
        segment.release_u,
        mode_label(segment.command.mode)
    );
}

uint8_t MainController::compute_step_distance_cm(const SplineD& path, const double current_u) const {
    const StepPlanSegment* const segment = current_step_command_segment(current_u);
    if (!segment) {
        return 0;
    }
    if (current_u >= segment->step_enter_u) {
        return 0;
    }

    const double distance = quadratic_bspline_arc_length(path.getControlPoints(), current_u, segment->step_enter_u);
    const double adjusted_distance = distance + nav_params_.step_dist_offset;
    const int64_t rounded_cm = std::lround(adjusted_distance * 100.0);
    return static_cast<uint8_t>(std::clamp<int64_t>(rounded_cm, 0, 255));
}

bool MainController::should_activate_step_mode(const double current_u) const {
    const StepPlanSegment* const segment = current_step_command_segment(current_u);
    if (!segment) return false;
    return current_u + U_EPSILON >= segment->active_u;
}

std::optional<ActiveStepMode> MainController::build_step_command(
    const StepDirection direction,
    const Eigen::Vector2d& step_enter_pos_map,
    const double step_enter_u,
    const DirectionMap& direction_map
) const {
    const auto mode_info = direction_map.step_mode_info_at(direction_map.map_coord_to_grid(step_enter_pos_map));
    const StepTraversalMode mode = direction == StepDirection::UP ? mode_info.up_mode : mode_info.down_mode;
    const uint8_t speed_level = direction == StepDirection::UP ? mode_info.up_speed_level : mode_info.down_speed_level;
    if (!is_step_traversal_allowed(mode)) {
        return std::nullopt;
    }

    return ActiveStepMode {
        .mode = [&]() {
            if (direction == StepDirection::UP) {
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
        .step_entry_u = step_enter_u,
    };
}

double MainController::step_speed_from_level(const uint8_t speed_level) const {
    return mpc_controller_->params().follow.terrain_limits.step_speed_levels[std::min<size_t>(speed_level, 3)];
}

std::optional<ActiveStepMode> MainController::current_active_step_mode(const double current_u) const {
    const StepPlanSegment* const segment = current_step_command_segment(current_u);
    if (!segment) return std::nullopt;
    return segment->command;
}

bool MainController::is_step_active(const double current_u) const {
    return active_step_segment(current_u) != nullptr;
}

} // namespace path_follower
