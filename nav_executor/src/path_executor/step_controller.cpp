#include <nav_executor/path_executor/step_controller.hpp>

#include <algorithm>

#include <rclcpp/logging.hpp>

namespace nav_executor {

namespace {
constexpr double U_EPSILON = 1e-6;

double approach(const double current, const double target, const double max_step) {
    return current + std::clamp(target - current, -max_step, max_step);
}

StepPhase phase_in_segment(const StepPlanSegment& segment, const double path_progress) {
    if (path_progress + U_EPSILON >= segment.commit_arc_length) return StepPhase::COMMITTED;
    if (path_progress + U_EPSILON >= segment.active_arc_length) return StepPhase::ARMED;
    return StepPhase::PREPARING;
}
} // anonymous namespace

StepPhaseObservation classify_step_phase(const AnnotatedPath& path, const double path_progress) {
    for (size_t i = 0; i < path.step_segments.size(); ++i) {
        const StepPlanSegment& segment = path.step_segments[i];
        if (path_progress + U_EPSILON < segment.prepare_arc_length) break;
        if (path_progress >= segment.release_arc_length) continue;
        if (!is_step_mode(segment.chassis_command.mode)) continue;
        return {
            .phase = phase_in_segment(segment, path_progress),
            .segment_index = i,
        };
    }
    return {};
}

StepController::StepController(
    const double step_dist_offset,
    const CapabilityProfile& normal_profile,
    const std::array<CapabilityProfile, 3>& capability_profiles,
    const ProfileBlendParams& blend_params,
    rclcpp::Logger logger
) : step_dist_offset_(step_dist_offset),
    blend_params_(blend_params),
    normal_profile_(normal_profile),
    capability_profiles_(capability_profiles),
    logger_(logger),
    current_profile_(normal_profile),
    target_profile_(normal_profile) {}

void StepController::clear() {
    path_.reset();
    next_step_segment_index_ = 0;
    held_step_segment_index_ = std::nullopt;
    current_profile_ = normal_profile_;
    target_profile_ = normal_profile_;
}

void StepController::set_path(AnnotatedPath::ConstPtr path) {
    next_step_segment_index_ = 0;
    held_step_segment_index_ = std::nullopt;
    target_profile_ = normal_profile_;
    path_ = std::move(path);
    if (path_ && !path_->step_segments.empty()) {
        RCLCPP_DEBUG(logger_, "StepController bound path with %zu step segments", path_->step_segments.size());
    }
}

// ═══════════════════════ 台阶段查询 ═══════════════════════

std::optional<size_t> StepController::find_active_segment_index(const double path_progress) const {
    if (!path_) return std::nullopt;
    for (size_t i = next_step_segment_index_; i < path_->step_segments.size(); ++i) {
        const StepPlanSegment& segment = path_->step_segments[i];
        if (path_progress + U_EPSILON < segment.prepare_arc_length) {
            break;
        }
        if (path_progress < segment.release_arc_length) {
            return i;
        }
    }
    return std::nullopt;
}

const StepPlanSegment* StepController::active_segment(const double path_progress) const {
    if (!path_) return nullptr;
    if (held_step_segment_index_.has_value()) {
        const auto& segment = path_->step_segments[*held_step_segment_index_];
        if (path_progress < segment.release_arc_length) {
            return &segment;
        }
    }
    const auto index = find_active_segment_index(path_progress);
    if (!index) return nullptr;
    return &path_->step_segments[*index];
}

const StepPlanSegment* StepController::current_command_segment(const double path_progress) const {
    const StepPlanSegment* const segment = active_segment(path_progress);
    if (!segment) return nullptr;
    if (!is_step_mode(segment->chassis_command.mode)) return nullptr;
    return segment;
}

// ═══════════════════════ 台阶段激活跟踪 ═══════════════════════

void StepController::update_active_segment(const double path_progress) {
    if (!path_) {
        next_step_segment_index_ = 0;
        held_step_segment_index_ = std::nullopt;
        target_profile_ = normal_profile_;
        return;
    }

    if (held_step_segment_index_.has_value()) {
        const auto& segment = path_->step_segments[*held_step_segment_index_];
        if (path_progress >= segment.release_arc_length) {
            RCLCPP_DEBUG(
                logger_,
                "Step segment #%zu released (progress=%.3f >= release=%.3f)",
                *held_step_segment_index_, path_progress, segment.release_arc_length
            );
            next_step_segment_index_ = *held_step_segment_index_ + 1;
            held_step_segment_index_ = std::nullopt;
            // fall through 尝试获取下一段
        } else {
            return;
        }
    }

    while (next_step_segment_index_ < path_->step_segments.size()
        && path_progress >= path_->step_segments[next_step_segment_index_].release_arc_length) {
        ++next_step_segment_index_;
    }

    const auto next_index = find_active_segment_index(path_progress);
    if (!next_index) {
        target_profile_ = normal_profile_;
        return;
    }

    held_step_segment_index_ = next_index;
    const auto& segment = path_->step_segments[*held_step_segment_index_];
    target_profile_ = capability_profiles_[static_cast<size_t>(segment.chassis_command.capability)];

    RCLCPP_DEBUG(
        logger_,
        "Step segment #%zu acquired: label=%hhu dir=%s "
        "prepare=%.3f active=%.3f commit=%.3f step=[%.3f,%.3f) release=%.3f mode=%hhu",
        *held_step_segment_index_,
        segment.terrain_label,
        segment.direction == StepDirection::UP ? "UP" : "DOWN",
        segment.prepare_arc_length, segment.active_arc_length, segment.commit_arc_length,
        segment.step_enter_arc_length, segment.step_exit_arc_length, segment.release_arc_length,
        segment.chassis_command.mode
    );
}

// ═══════════════════════ 时间域 profile 融合 ═══════════════

void StepController::tick_profile_blend() {
    auto& cur = current_profile_;
    const auto& tgt = target_profile_;

    {
        auto& c = cur.command_envelope;
        const auto& t = tgt.command_envelope;
        const double v_step = blend_params_.v_step;
        const double w_step = blend_params_.w_step;

        c.velocity.min = approach(c.velocity.min, t.velocity.min, v_step);
        c.velocity.max = approach(c.velocity.max, t.velocity.max, v_step);
        c.angular_velocity.min = approach(
            c.angular_velocity.min, t.angular_velocity.min, w_step
        );
        c.angular_velocity.max = approach(
            c.angular_velocity.max, t.angular_velocity.max, w_step
        );
    }

    {
        auto& c = cur.command_dynamics;
        const auto& t = tgt.command_dynamics;

        c.velocity_rate_max = approach(
            c.velocity_rate_max, t.velocity_rate_max, blend_params_.acc_step
        );
        c.angular_velocity_rate_max = approach(
            c.angular_velocity_rate_max,
            t.angular_velocity_rate_max,
            blend_params_.alpha_step
        );
        c.lateral_acceleration_max = approach(
            c.lateral_acceleration_max,
            t.lateral_acceleration_max,
            blend_params_.a_lat_step
        );
    }
}

// ═══════════════════════ 台阶模式查询 ═══════════════════════

const StepChassisCommand* StepController::current_chassis_command(const double path_progress) const {
    const StepPlanSegment* const segment = current_command_segment(path_progress);
    return segment ? &segment->chassis_command : nullptr;
}

StepPhaseObservation StepController::observe_step_phase(const double path_progress) const {
    if (!path_) return {};

    std::optional<size_t> index;
    if (held_step_segment_index_) {
        const StepPlanSegment& held = path_->step_segments[*held_step_segment_index_];
        if (path_progress < held.release_arc_length) index = held_step_segment_index_;
    }
    if (!index) index = find_active_segment_index(path_progress);
    if (!index) return {};

    const StepPlanSegment& segment = path_->step_segments[*index];
    if (!is_step_mode(segment.chassis_command.mode)) return {};
    return {
        .phase = phase_in_segment(segment, path_progress),
        .segment_index = index,
    };
}

uint8_t StepController::compute_step_distance_cm(const double path_progress) const {
    const StepPlanSegment* const segment = current_command_segment(path_progress);
    if (!segment) return 0;
    if (path_progress >= segment->step_enter_arc_length) return 0;

    const double distance = segment->step_enter_arc_length - path_progress;
    const double adjusted_distance = distance + step_dist_offset_;
    const int64_t rounded_cm = std::lround(adjusted_distance * 100.0);
    return static_cast<uint8_t>(std::clamp<int64_t>(rounded_cm, 0, 255));
}

} // namespace nav_executor
