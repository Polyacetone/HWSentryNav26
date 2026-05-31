#include <path_follower/step_controller.hpp>
#include <rclcpp/logging.hpp>

namespace path_follower {

namespace {

constexpr double ANGLE_EPSILON = 1e-6;
constexpr double U_EPSILON = 1e-6;

struct StepSample {
    double u = 0.0;
    Eigen::Vector2d pos_map = Eigen::Vector2d::Zero();
    Eigen::Vector2d dir_map = Eigen::Vector2d::Zero();
    StepDirection direction = StepDirection::UP;
};

} // anonymous namespace

// ═══════════════════════ 构造函数 ════════════════════════════

StepController::StepController(
    const StepDetectionParams& step_detection,
    const StepBlockReplanParams& step_block_replan,
    const double step_dist_offset,
    rclcpp::Logger logger
) : step_detection_(step_detection),
    step_block_replan_(step_block_replan),
    step_dist_offset_(step_dist_offset),
    logger_(logger) {}

// ═══════════════════════ 路径工具 ═══════════════════════════

double StepController::advance_path_u_by_distance(const SplinePath& path, const double start_u, const double distance) const {
    const double resolution = std::max(1e-3, step_detection_.path_sample_resolution);
    double u = std::clamp(start_u, 0.0, 1.0);
    double travelled = 0.0;

    while (u < 1.0 && travelled < distance) {
        const Eigen::Vector2d d1 = path.tangent(u);
        const double speed = d1.norm();
        if (speed < 1e-12) {
            u = std::min(1.0, u + 1e-3);
            continue;
        }
        const double du = resolution / speed;
        const double next_u = std::min(1.0, u + du);
        travelled += (path.position(next_u) - path.position(u)).norm();
        u = next_u;
    }
    return u;
}

double StepController::retreat_path_u_by_distance(const SplinePath& path, const double start_u, const double distance) const {
    const double resolution = std::max(1e-3, step_detection_.path_sample_resolution);
    double u = std::clamp(start_u, 0.0, 1.0);
    double travelled = 0.0;

    while (u > 0.0 && travelled < distance) {
        const Eigen::Vector2d d1 = path.tangent(u);
        const double speed = d1.norm();
        if (speed < 1e-12) {
            u = std::max(0.0, u - 1e-3);
            continue;
        }
        const double du = resolution / speed;
        const double next_u = std::max(0.0, u - du);
        travelled += (path.position(u) - path.position(next_u)).norm();
        u = next_u;
    }

    if (travelled < distance) {
        const double speed = path.tangent(0.0).norm();
        if (speed > 1e-12) {
            u = -(distance - travelled) / speed;
        }
    }

    return u;
}

// ═══════════════════════ 台阶命令构建 ═══════════════════════

std::optional<ActiveStepMode> StepController::build_step_command(
    const StepDirection direction,
    const Eigen::Vector2d& step_enter_pos_map,
    const double step_enter_u,
    const DirectionMap& direction_map
) const {
    const Eigen::Vector2d g = direction_map.map_coord_to_grid(step_enter_pos_map);
    const uint8_t label = direction_map.terrain_at(g);
    const auto& rule = direction_map.rule_for_label(label, direction == StepDirection::UP);

    if (rule.chassis_mode == 0) {
        return std::nullopt;
    }

    return ActiveStepMode {
        .mode = rule.chassis_mode,
        .capability = rule.capability,
        .speed_min = rule.speed.min,
        .speed_max = rule.speed.max,
        .step_entry_u = step_enter_u,
    };
}

// ═══════════════════════ 台阶规划构建 ═══════════════════════

std::vector<StepController::StepPlanSegment> StepController::build_step_plan(
    const SplinePath& path,
    const DirectionMap& direction_map
) const {
    const double resolution = std::max(1e-3, step_detection_.path_sample_resolution);

    double estimated_length = 0.0;
    Eigen::Vector2d prev = path.position(0.0);
    constexpr int length_estimate_samples = 100;
    for (int i = 1; i <= length_estimate_samples; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(length_estimate_samples);
        const Eigen::Vector2d pos = path.position(u);
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
        const Eigen::Vector2d pos = path.position(u);
        const Eigen::Vector2d tangent = path.tangent(u);
        const Eigen::Vector2d g = direction_map.map_coord_to_grid(pos);
        if (!direction_map.is_valid_coord(g)) continue;

        const uint8_t label = direction_map.terrain_at(g);
        if (label < static_cast<uint8_t>(TerrainType::SLOPE)) continue;

        const Eigen::Vector2d dir = direction_map.interpolate(g);
        if (dir.norm() < ANGLE_EPSILON) continue;

        const double dot = dir.normalized().dot(tangent.normalized());
        if (std::abs(dot) <= step_detection_.detect_dot_threshold) continue;

        const StepDirection direction = dot > 0.0 ? StepDirection::UP : StepDirection::DOWN;

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
            const bool gap_too_large = path.arc_length(
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
        segment.prepare_u = retreat_path_u_by_distance(path, segment.step_enter_u, step_detection_.prepare_distance);
        segment.active_u = retreat_path_u_by_distance(path, segment.step_enter_u, step_detection_.active_distance);
        segment.release_u = advance_path_u_by_distance(path, segment.step_exit_u, step_detection_.release_distance);
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
        segment.prepare_u = std::min(segment.prepare_u, segment.step_enter_u);
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

void StepController::clear_runtime_state() {
    held_step_segment_index_ = std::nullopt;
    step_mode_blend_factor_ = 0.0;
    step_locked_path_ = std::nullopt;
    step_locked_fixed_goal_ = false;
    step_locked_fixed_goal_pos_ = Eigen::Vector2d::Zero();
}

void StepController::clear_plan() {
    clear_runtime_state();
    step_plan_.clear();
}

void StepController::update_plan_for_path_change(
    const bool has_new_path,
    const std::optional<SplinePath>& path,
    const DirectionMap* const direction_map
) {
    if (!has_new_path) return;

    path_version_++;
    clear_plan();

    if (!path || !direction_map) {
        return;
    }

    step_plan_ = build_step_plan(*path, *direction_map);
    if (!step_plan_.empty()) {
        RCLCPP_DEBUG(logger_, "Built step plan with %zu segments for path_version=%d", step_plan_.size(), path_version_);
    }
}

// ═══════════════════════ 台阶段查询 ═══════════════════════

std::optional<size_t> StepController::find_active_segment_index(const double current_u) const {
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

const StepController::StepPlanSegment* StepController::active_segment(const double current_u) const {
    if (held_step_segment_index_.has_value()) {
        const auto& segment = step_plan_[*held_step_segment_index_];
        if (current_u < segment.release_u) {
            return &segment;
        }
    }
    const auto index = find_active_segment_index(current_u);
    if (!index) return nullptr;
    return &step_plan_[*index];
}

const StepController::StepPlanSegment* StepController::current_command_segment(const double current_u) const {
    const StepPlanSegment* const segment = active_segment(current_u);
    if (!segment) return nullptr;
    if (!is_step_mode(segment->command.mode)) return nullptr;
    return segment;
}

// ═══════════════════════ 台阶激活/锁存 ═══════════════════════

void StepController::update_active_segment(
    const double current_u,
    const std::optional<SplinePath>& global_path,
    const bool fixed_goal,
    const Eigen::Vector2d& fixed_goal_pos
) {
    if (held_step_segment_index_.has_value()) {
        const auto& segment = step_plan_[*held_step_segment_index_];
        if (current_u >= segment.release_u) {
            RCLCPP_DEBUG(
                logger_,
                "Step segment #%zu released (current_u=%.3f >= release_u=%.3f)",
                *held_step_segment_index_, current_u, segment.release_u
            );
            held_step_segment_index_ = std::nullopt;
            step_mode_blend_factor_ = 0.0;
            step_locked_path_ = std::nullopt;
            step_locked_fixed_goal_ = false;
            step_locked_fixed_goal_pos_ = Eigen::Vector2d::Zero();
            return;
        }
        double raw = 0.0;
        if (current_u >= segment.active_u) {
            raw = 1.0;
        } else if (current_u > segment.prepare_u) {
            const double range = segment.active_u - segment.prepare_u;
            if (range > 1e-12) {
                raw = (current_u - segment.prepare_u) / range;
            }
        }
        step_mode_blend_factor_ = std::max(step_mode_blend_factor_, raw);
        return;
    }

    const auto next_index = find_active_segment_index(current_u);
    if (!next_index) {
        return;
    }

    held_step_segment_index_ = next_index;

    {
        const auto& segment = step_plan_[*held_step_segment_index_];
        if (current_u >= segment.active_u) {
            step_mode_blend_factor_ = 1.0;
        } else if (current_u > segment.prepare_u) {
            const double range = segment.active_u - segment.prepare_u;
            if (range > 1e-12) {
                step_mode_blend_factor_ = (current_u - segment.prepare_u) / range;
            }
        } else {
            step_mode_blend_factor_ = 0.0;
        }

        if (global_path) {
            step_locked_path_ = global_path;
            step_locked_fixed_goal_ = fixed_goal;
            step_locked_fixed_goal_pos_ = fixed_goal_pos;
        }

        RCLCPP_DEBUG(
            logger_,
            "Step segment #%zu acquired: prepare_u=%.3f active_u=%.3f step=[%.3f, %.3f) release_u=%.3f mode=%hhu blend=%.2f",
            *held_step_segment_index_,
            segment.prepare_u, segment.active_u,
            segment.step_enter_u, segment.step_exit_u, segment.release_u,
            segment.command.mode,
            step_mode_blend_factor_
        );
    }
}

// ═══════════════════════ 台阶模式查询 ═══════════════════════

std::optional<ActiveStepMode> StepController::current_active_step_mode(const double current_u) const {
    const StepPlanSegment* const segment = current_command_segment(current_u);
    if (!segment) return std::nullopt;
    auto mode = segment->command;
    mode.mode_blend_factor = step_mode_blend_factor_;
    return mode;
}

bool StepController::is_step_active(const double current_u) const {
    return active_segment(current_u) != nullptr;
}

bool StepController::should_activate_step_mode(const double current_u) const {
    const StepPlanSegment* const segment = current_command_segment(current_u);
    if (!segment) return false;
    return current_u + U_EPSILON >= segment->active_u;
}

uint8_t StepController::compute_step_distance_cm(const SplinePath& path, const double current_u) const {
    const StepPlanSegment* const segment = current_command_segment(current_u);
    if (!segment) {
        return 0;
    }
    if (current_u >= segment->step_enter_u) {
        return 0;
    }

    const double distance = path.arc_length(current_u, segment->step_enter_u);
    const double adjusted_distance = distance + step_dist_offset_;
    const int64_t rounded_cm = std::lround(adjusted_distance * 100.0);
    return static_cast<uint8_t>(std::clamp<int64_t>(rounded_cm, 0, 255));
}

// ═══════════════════════ 台阶阻塞重规划检测 ═══════════════════

/*static*/ std::optional<StepController::FollowStepBlockSampleStats> StepController::sample_block_replan_stats(
    const StepBlockReplanParams& p,
    const SplinePath& path,
    const double start_u,
    const CostMap* const dynamic_cost_map,
    const std::vector<const CostMap*>& dynamic_prediction_maps,
    const DirectionMap& direction_map
) {
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
            const Eigen::Vector2d d1 = path.tangent(u);
            const double speed = d1.norm();
            if (speed < 1e-12) {
                u = std::min(1.0, u + 1e-3);
                continue;
            }
            const double du = resolution / speed;
            const double next_u = std::min(1.0, u + du);
            travelled += (path.position(next_u) - path.position(u)).norm();
            u = next_u;
        }
        return u;
    };

    for (int i = 0; i < samples; ++i) {
        const double t = samples == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(samples - 1);
        const double u = advance_path_u(lookahead_distance * t);

        const Eigen::Vector2d pos = path.position(u);
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

bool StepController::check_block_replan(
    const SplinePath& path,
    const double current_u,
    const DirectionMap* const masked_direction_map,
    const CostMap* const current_dynamic_cost_map,
    const std::vector<const CostMap*>& per_step_dynamic_cost_maps
) const {
    const auto& p = step_block_replan_;
    if (!p.enable || !masked_direction_map) return false;

    const auto stats = sample_block_replan_stats(
        p,
        path,
        current_u,
        current_dynamic_cost_map,
        per_step_dynamic_cost_maps,
        *masked_direction_map
    );
    if (!stats || stats->step_sample_count == 0) {
        return false;
    }

    if (per_step_dynamic_cost_maps.empty()) {
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

} // namespace path_follower
