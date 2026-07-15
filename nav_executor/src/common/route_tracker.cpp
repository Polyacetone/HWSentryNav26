#include <nav_executor/common/route_tracker.hpp>

#include <algorithm>
#include <cmath>

namespace nav_executor {

RouteTracker::RouteTracker(RouteTrackerParams params) : params_(params) {}

void RouteTracker::reset() {
    path_.reset();
    accepted_estimate_.reset();
    last_stamp_ = {};
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
        accepted_estimate_.reset();
        last_stamp_ = stamp;
    }

    const SplinePath& geometry = path_->spline;
    double search_min = 0.0;
    double search_max = std::min(geometry.length(), std::max(0.0, params_.initial_search_distance));

    if (accepted_estimate_) {
        const double raw_dt = std::chrono::duration<double>(stamp - last_stamp_).count();
        const double dt = std::clamp(raw_dt, 0.0, 0.5);
        const SplineEval reference = geometry.eval(accepted_estimate_->u);
        const Eigen::Vector2d heading(std::cos(chassis_pose_map.z()), std::sin(chassis_pose_map.z()));
        const Eigen::Vector2d tangent = reference.d1.norm() > 1e-9
            ? reference.d1.normalized()
            : heading;
        const double predicted_rate = std::clamp(
            chassis_velocity * heading.dot(tangent),
            -std::max(0.0, params_.max_progress_rate),
            std::max(0.0, params_.max_progress_rate)
        );
        const double predicted_arc = std::clamp(
            accepted_estimate_->arc_length + predicted_rate * dt,
            0.0,
            geometry.length()
        );
        const double radius = std::max(0.0, params_.progress_tolerance)
            + std::max(0.0, params_.max_progress_rate) * dt;
        search_min = std::max(0.0, predicted_arc - radius);
        search_max = std::min(geometry.length(), predicted_arc + radius);
    }

    const auto projection = geometry.project(chassis_pose_map.head<2>(), search_min, search_max);
    if (!projection) {
        return RouteEstimate {
            .path = path_,
            .status = RouteTrackingStatus::LOST,
            .remaining_length = geometry.length(),
        };
    }

    RouteEstimate estimate {
        .path = path_,
        .status = projection->distance <= params_.max_cross_track_error
            ? RouteTrackingStatus::TRACKED
            : RouteTrackingStatus::LOST,
        .u = projection->u,
        .arc_length = projection->arc_length,
        .remaining_length = std::max(0.0, geometry.length() - projection->arc_length),
        .projected_position = projection->position,
        .cross_track_error = projection->distance,
    };

    if (estimate.status == RouteTrackingStatus::TRACKED) {
        accepted_estimate_ = estimate;
        last_stamp_ = stamp;
    }
    return estimate;
}

} // namespace nav_executor
