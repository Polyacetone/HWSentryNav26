#include <nav_executor/common/route_tracker.hpp>

#include <algorithm>
#include <cmath>

namespace nav_executor {

RouteTracker::RouteTracker(RouteTrackerParams params) : params_(params) {}

void RouteTracker::reset() {
    path_.reset();
    estimate_.reset();
    last_stamp_.reset();
    segment_cursor_ = 0;
}

std::optional<RouteEstimate> RouteTracker::update(
    AnnotatedPath::ConstPtr path,
    const Eigen::Vector3d& chassis_pose_map,
    const double chassis_velocity,
    const std::chrono::steady_clock::time_point stamp
) {
    if (!path) {
        reset();
        return std::nullopt;
    }

    if (path != path_) {
        path_ = std::move(path);
        estimate_.reset();
        last_stamp_.reset();
        segment_cursor_ = 0;
    }

    const MincoTrajectory& geometry = path_->trajectory;
    if (geometry.empty()) {
        reset();
        return std::nullopt;
    }
    const double total_len = geometry.length();
    double predicted_arc_length = 0.0;
    double path_speed = 0.0;
    double search_lo = 0.0;
    double search_hi = std::min(total_len, params_.initial_search_distance);
    if (!estimate_) {
        search_hi = std::min(total_len, std::max(0.0, params_.initial_search_distance));
        for (int segment = 0; segment + 1 < geometry.segment_count(); ++segment) {
            if (geometry.segment_gear(segment) != geometry.segment_gear(segment + 1)) {
                search_hi = std::min(
                    search_hi, geometry.segment_boundary_arc_length(segment + 1)
                );
                break;
            }
        }
    } else {
        constexpr double DIRECTION_SPEED_EPS = 0.05;
        if (segment_cursor_ + 1 < geometry.segment_count()) {
            const double current_gear = geometry.segment_gear(segment_cursor_);
            const double next_gear = geometry.segment_gear(segment_cursor_ + 1);
            const double boundary = geometry.segment_boundary_arc_length(segment_cursor_ + 1);
            const bool near_boundary = estimate_->arc_length
                >= boundary - params_.cusp_switch_distance;
            if (current_gear == next_gear && estimate_->arc_length >= boundary) {
                ++segment_cursor_;
            } else if (current_gear != next_gear && near_boundary
                && next_gear * chassis_velocity > DIRECTION_SPEED_EPS
                && current_gear * chassis_velocity < -DIRECTION_SPEED_EPS) {
                ++segment_cursor_;
            }
        }

        const double current_gear = geometry.segment_gear(segment_cursor_);
        const double measured_path_speed = std::max(0.0, current_gear * chassis_velocity);
        path_speed = std::lerp(
            estimate_->path_speed,
            measured_path_speed,
            params_.path_speed_filter_alpha
        );
        const double dt = last_stamp_
            ? std::clamp(
                std::chrono::duration<double>(stamp - *last_stamp_).count(),
                0.0,
                params_.prediction_time_limit
            )
            : 0.0;
        predicted_arc_length = estimate_->arc_length + path_speed * dt;

        const bool previous_boundary_is_cusp = segment_cursor_ > 0
            && geometry.segment_gear(segment_cursor_ - 1)
                != geometry.segment_gear(segment_cursor_);
        const int first_segment = previous_boundary_is_cusp
            ? segment_cursor_
            : std::max(0, segment_cursor_ - 1);
        const bool next_boundary_is_cusp = segment_cursor_ + 1 < geometry.segment_count()
            && geometry.segment_gear(segment_cursor_)
                != geometry.segment_gear(segment_cursor_ + 1);
        const int last_segment = next_boundary_is_cusp
            ? segment_cursor_
            : std::min(geometry.segment_count() - 1, segment_cursor_ + 1);
        const double topology_lo = geometry.segment_boundary_arc_length(first_segment);
        const double topology_hi = geometry.segment_boundary_arc_length(last_segment + 1);
        predicted_arc_length = std::clamp(
            predicted_arc_length, topology_lo, topology_hi
        );
        search_lo = std::max(
            topology_lo,
            predicted_arc_length - params_.projection.search_distance_backward
        );
        search_hi = std::min(
            topology_hi,
            predicted_arc_length + params_.projection.search_distance_forward
        );
    }

    const TrajectoryProjection projection = project_trajectory(
        geometry,
        chassis_pose_map.head<2>(),
        predicted_arc_length,
        search_lo,
        search_hi,
        params_.projection
    );
    const double arc_length = projection.arc_length;

    const double tau = geometry.tau_at_arc_length(arc_length);
    const TrajSample reference = geometry.eval_arc_length(arc_length);
    const Eigen::Vector2d reference_position = reference.p;
    const double tracking_error = projection.tracking_error;
    if (!estimate_) {
        segment_cursor_ = geometry.segment_index_at_arc_length(arc_length);
        if (segment_cursor_ > 0) {
            const double current_alignment = geometry.segment_gear(segment_cursor_)
                * chassis_velocity;
            const double previous_alignment = geometry.segment_gear(segment_cursor_ - 1)
                * chassis_velocity;
            if (previous_alignment > current_alignment + 0.05) --segment_cursor_;
        }
    }
    const double measured_path_speed = std::max(
        0.0, geometry.segment_gear(segment_cursor_) * chassis_velocity
    );
    if (!estimate_) {
        path_speed = measured_path_speed;
    } else if (geometry.segment_gear(estimate_->segment_index)
        != geometry.segment_gear(segment_cursor_)) {
        path_speed = measured_path_speed;
    }

    RouteEstimate estimate {
        .path = path_,
        .status = tracking_error <= params_.max_tracking_error
            ? RouteTrackingStatus::TRACKED
            : RouteTrackingStatus::LOST,
        .tau = tau,
        .arc_length = arc_length,
        .path_speed = path_speed,
        .segment_index = segment_cursor_,
        .remaining_length = std::max(0.0, total_len - arc_length),
        .reference_position = reference_position,
        .projected_position = reference_position,
        .tracking_error = tracking_error,
    };

    estimate_ = estimate;
    last_stamp_ = stamp;
    return estimate;
}

} // namespace nav_executor
