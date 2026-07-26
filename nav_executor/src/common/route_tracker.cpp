#include <nav_executor/common/route_tracker.hpp>

#include <algorithm>
#include <cmath>

namespace nav_executor {

RouteTracker::RouteTracker(RouteTrackerParams params) : params_(params) {}

void RouteTracker::reset() {
    path_.reset();
    estimate_.reset();
    last_stamp_.reset();
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
    if (estimate_) {
        const double measured_path_speed = std::max(0.0, chassis_velocity);
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

        search_lo = std::max(
            0.0,
            predicted_arc_length - params_.projection.search_distance_backward
        );
        search_hi = std::min(
            total_len,
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

    const TrajSample reference = geometry.eval_arc_length(arc_length);
    const Eigen::Vector2d reference_position = reference.p;
    const double tracking_error = projection.tracking_error;
    const double measured_path_speed = std::max(0.0, chassis_velocity);
    if (!estimate_) {
        path_speed = measured_path_speed;
    }

    RouteEstimate estimate {
        .path = path_,
        .status = tracking_error <= params_.max_tracking_error
            ? RouteTrackingStatus::TRACKED
            : RouteTrackingStatus::LOST,
        .arc_length = arc_length,
        .path_speed = path_speed,
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
