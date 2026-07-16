#include <nav_executor/common/route_tracker.hpp>

#include <algorithm>
#include <cmath>

namespace nav_executor {

RouteTracker::RouteTracker(RouteTrackerParams params) : params_(params) {}

void RouteTracker::reset() {
    path_.reset();
    estimate_.reset();
}

std::optional<RouteEstimate> RouteTracker::update(
    AnnotatedPath::ConstPtr path,
    const Eigen::Vector3d& chassis_pose_map,
    const double chassis_velocity
) {
    if (!path) {
        reset();
        return std::nullopt;
    }

    if (path != path_) {
        path_ = std::move(path);
        estimate_.reset();
    }

    const MincoTrajectory& geometry = path_->trajectory;
    const double total_len = geometry.length();
    const double total_time = geometry.total_time();

    TrajectoryProjection projection;
    double phase_time = 0.0;
    if (!estimate_) {
        const double initial_arc = std::min(total_len, std::max(0.0, params_.initial_search_distance));
        const double initial_time_hi = geometry.tau_at_arc_length(initial_arc) * total_time;
        projection = project_trajectory(
            geometry, chassis_pose_map, chassis_velocity, 0.0, initial_time_hi, params_.projection
        );
        phase_time = projection.time;
    } else {
        const double previous_time = estimate_->phase_time;
        projection = project_trajectory(
            geometry,
            chassis_pose_map,
            chassis_velocity,
            previous_time - params_.projection.projection_window_backward,
            previous_time + params_.projection.projection_window_forward,
            params_.projection
        );
        phase_time = std::clamp(
            projection.time,
            previous_time - params_.projection.max_backward_step,
            previous_time + params_.projection.max_forward_step
        );
        phase_time = std::clamp(phase_time, 0.0, total_time);
    }

    const double tau = total_time > 1e-9 ? phase_time / total_time : 0.0;
    const Eigen::Vector2d reference_position = geometry.eval_time(phase_time).p;
    const double arc_length = geometry.arc_length_at_tau(tau);
    const double tracking_error = (reference_position - chassis_pose_map.head<2>()).norm();

    RouteEstimate estimate {
        .path = path_,
        .status = tracking_error <= params_.max_tracking_error
            ? RouteTrackingStatus::TRACKED
            : RouteTrackingStatus::LOST,
        .phase_time = phase_time,
        .tau = tau,
        .arc_length = arc_length,
        .remaining_length = std::max(0.0, total_len - arc_length),
        .reference_position = reference_position,
        .projected_position = reference_position,
        .tracking_error = tracking_error,
    };

    estimate_ = estimate;
    return estimate;
}

} // namespace nav_executor
