#include <nav_executor/path_executor/step_controller.hpp>
#include <rclcpp/logging.hpp>

namespace nav_executor {

namespace {
constexpr double U_EPSILON = 1e-6;
} // anonymous namespace

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
    held_step_segment_index_ = std::nullopt;
    current_profile_ = normal_profile_;
    target_profile_ = normal_profile_;
}

void StepController::set_path(AnnotatedPath::ConstPtr path) {
    held_step_segment_index_ = std::nullopt;
    target_profile_ = normal_profile_;
    path_ = std::move(path);
    if (path_ && !path_->step_segments.empty()) {
        RCLCPP_DEBUG(logger_, "StepController bound path with %zu step segments", path_->step_segments.size());
    }
}

// ═══════════════════════ 台阶段查询 ═══════════════════════

std::optional<size_t> StepController::find_active_segment_index(const double current_u) const {
    if (!path_) return std::nullopt;
    for (size_t i = 0; i < path_->step_segments.size(); ++i) {
        const StepPlanSegment& segment = path_->step_segments[i];
        if (current_u + U_EPSILON < segment.prepare_u) {
            break;
        }
        if (current_u < segment.release_u) {
            return i;
        }
    }
    return std::nullopt;
}

const StepPlanSegment* StepController::active_segment(const double current_u) const {
    if (!path_) return nullptr;
    if (held_step_segment_index_.has_value()) {
        const auto& segment = path_->step_segments[*held_step_segment_index_];
        if (current_u < segment.release_u) {
            return &segment;
        }
    }
    const auto index = find_active_segment_index(current_u);
    if (!index) return nullptr;
    return &path_->step_segments[*index];
}

const StepPlanSegment* StepController::current_command_segment(const double current_u) const {
    const StepPlanSegment* const segment = active_segment(current_u);
    if (!segment) return nullptr;
    if (!is_step_mode(segment->chassis_command.mode)) return nullptr;
    return segment;
}

// ═══════════════════════ 台阶段激活跟踪 ═══════════════════════

void StepController::update_active_segment(const double current_u) {
    if (!path_) {
        held_step_segment_index_ = std::nullopt;
        target_profile_ = normal_profile_;
        return;
    }

    if (held_step_segment_index_.has_value()) {
        const auto& segment = path_->step_segments[*held_step_segment_index_];
        if (current_u >= segment.release_u) {
            RCLCPP_DEBUG(
                logger_,
                "Step segment #%zu released (current_u=%.3f >= release_u=%.3f)",
                *held_step_segment_index_, current_u, segment.release_u
            );
            held_step_segment_index_ = std::nullopt;
            // fall through 尝试获取下一段
        } else {
            return;
        }
    }

    const auto next_index = find_active_segment_index(current_u);
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
        "prepare=%.3f active=%.3f step=[%.3f,%.3f) release=%.3f mode=%hhu",
        *held_step_segment_index_,
        segment.terrain_label,
        segment.direction == StepDirection::UP ? "UP" : "DOWN",
        segment.prepare_u, segment.active_u,
        segment.step_enter_u, segment.step_exit_u, segment.release_u,
        segment.chassis_command.mode
    );
}

// ═══════════════════════ 时间域 profile 融合 ═══════════════

void StepController::tick_profile_blend() {
    auto& cur = current_profile_;
    const auto& tgt = target_profile_;

    {
        auto& c = cur.command_bounds;
        const auto& t = tgt.command_bounds;
        const double v_step = blend_params_.v_step;
        const double w_step = blend_params_.w_step;

        if (t.vel_max >= c.vel_max) c.vel_max = t.vel_max;
        else c.vel_max = std::max(t.vel_max, c.vel_max - v_step);
        if (t.vel_min >= c.vel_min) c.vel_min = t.vel_min;
        else c.vel_min = std::max(t.vel_min, c.vel_min - v_step);
        if (t.omega_max >= c.omega_max) c.omega_max = t.omega_max;
        else c.omega_max = std::max(t.omega_max, c.omega_max - w_step);
        if (t.omega_min >= c.omega_min) c.omega_min = t.omega_min;
        else c.omega_min = std::max(t.omega_min, c.omega_min - w_step);
    }

    {
        auto& c = cur.motion_constraints;
        const auto& t = tgt.motion_constraints;

        if (t.acc_max >= c.acc_max) c.acc_max = t.acc_max;
        else c.acc_max = std::max(t.acc_max, c.acc_max - blend_params_.acc_step);
        if (t.alpha_max >= c.alpha_max) c.alpha_max = t.alpha_max;
        else c.alpha_max = std::max(t.alpha_max, c.alpha_max - blend_params_.alpha_step);
        if (t.a_lat_max >= c.a_lat_max) c.a_lat_max = t.a_lat_max;
        else c.a_lat_max = std::max(t.a_lat_max, c.a_lat_max - blend_params_.a_lat_step);
    }
}

// ═══════════════════════ 台阶模式查询 ═══════════════════════

const StepChassisCommand* StepController::current_chassis_command(const double current_u) const {
    const StepPlanSegment* const segment = current_command_segment(current_u);
    return segment ? &segment->chassis_command : nullptr;
}

bool StepController::is_step_nonpreemptible(const double current_u) const {
    const StepPlanSegment* const segment = current_command_segment(current_u);
    return segment && current_u + U_EPSILON >= segment->active_u;
}

bool StepController::should_activate_chassis_mode(const double current_u) const {
    return is_step_nonpreemptible(current_u);
}

uint8_t StepController::compute_step_distance_cm(const SplinePath& path, const double current_u) const {
    const StepPlanSegment* const segment = current_command_segment(current_u);
    if (!segment) return 0;
    if (current_u >= segment->step_enter_u) return 0;

    const double distance = path.arc_length(current_u, segment->step_enter_u);
    const double adjusted_distance = distance + step_dist_offset_;
    const int64_t rounded_cm = std::lround(adjusted_distance * 100.0);
    return static_cast<uint8_t>(std::clamp<int64_t>(rounded_cm, 0, 255));
}

} // namespace nav_executor
