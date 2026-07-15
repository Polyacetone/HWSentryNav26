#include <nav_executor/common/route_tracker.hpp>

#include <algorithm>
#include <cmath>

namespace nav_executor {

RouteTracker::RouteTracker(RouteTrackerParams params) : params_(params) {}

void RouteTracker::reset() {
    path_.reset();
    estimate_.reset();
    last_stamp_ = {};
}

std::optional<RouteEstimate> RouteTracker::update(
    AnnotatedPath::ConstPtr path,
    const Eigen::Vector3d& chassis_pose_map,
    const std::chrono::steady_clock::time_point stamp,
    const bool advance_clock
) {
    if (!path) {
        reset();
        return std::nullopt;
    }

    if (path != path_) {
        path_ = std::move(path);
        estimate_.reset();
        last_stamp_ = stamp;
    }

    const MincoTrajectory& geometry = path_->trajectory;
    const double total_len = geometry.length();

    double tau = 0.0;
    if (!estimate_) {
        // 新路径仅在起点附近投影一次。之后 tau 由执行时钟持久化，空间投影不再覆盖它。
        const double initial_arc = std::min(total_len, std::max(0.0, params_.initial_search_distance));
        tau = geometry.project(
            chassis_pose_map.head<2>(), 0.0, geometry.tau_at_arc_length(initial_arc)
        );
    } else {
        tau = estimate_->tau;
        const double raw_dt = std::chrono::duration<double>(stamp - last_stamp_).count();
        const double dt = std::clamp(raw_dt, 0.0, 0.5);
        if (advance_clock && geometry.total_time() > 1e-9) {
            const TrajSample reference = geometry.eval(tau);
            const double factor = governed_clock_factor(reference, chassis_pose_map, params_.governed_clock);
            tau = std::min(1.0, tau + dt * factor / geometry.total_time());
        }
    }

    const Eigen::Vector2d reference_position = geometry.position(tau);
    const double arc_length = geometry.arc_length_at_tau(tau);
    const double distance = (reference_position - chassis_pose_map.head<2>()).norm();

    RouteEstimate estimate {
        .path = path_,
        .status = distance <= params_.max_tracking_error
            ? RouteTrackingStatus::TRACKED
            : RouteTrackingStatus::LOST,
        .tau = tau,
        .arc_length = arc_length,
        .remaining_length = std::max(0.0, total_len - arc_length),
        .reference_position = reference_position,
        .tracking_error = distance,
    };

    estimate_ = estimate;
    last_stamp_ = stamp;
    return estimate;
}

} // namespace nav_executor
