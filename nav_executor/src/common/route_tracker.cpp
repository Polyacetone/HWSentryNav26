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
    const double chassis_velocity,
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
    const double total_time = geometry.total_time();

    TrajectoryPhaseState phase;
    double accumulated_delay = 0.0;
    TrajectoryPhaseProjection projection;
    if (!estimate_) {
        const double initial_arc = std::min(total_len, std::max(0.0, params_.initial_search_distance));
        const double initial_time_hi = geometry.tau_at_arc_length(initial_arc) * total_time;
        projection = project_trajectory_phase(
            geometry, chassis_pose_map, chassis_velocity, 0.0, initial_time_hi, params_.phase
        );
        phase.time = projection.time;
        phase.rate = trajectory_phase_rate_target(
            geometry.eval_time(phase.time), chassis_pose_map, params_.phase
        );
    } else {
        phase = {.time = estimate_->phase_time, .rate = estimate_->phase_rate};
        accumulated_delay = estimate_->accumulated_delay;
        const double raw_dt = std::chrono::duration<double>(stamp - last_stamp_).count();
        const double dt = std::clamp(raw_dt, 0.0, 0.5);
        if (advance_clock && total_time > 1e-9) {
            const double previous_time = phase.time;
            phase = advance_trajectory_phase(
                geometry, phase, chassis_pose_map, dt, params_.phase
            );
            accumulated_delay += std::max(0.0, dt - (phase.time - previous_time));
        } else {
            phase.rate = 0.0;
        }

        projection = project_trajectory_phase(
            geometry,
            chassis_pose_map,
            chassis_velocity,
            phase.time - params_.phase.projection_window_backward,
            phase.time + params_.phase.projection_window_forward,
            params_.phase
        );
        if (advance_clock) {
            const double innovation = projection.time - phase.time;
            const double correction = std::clamp(
                params_.phase.observation_gain * innovation,
                -params_.phase.max_observation_correction,
                params_.phase.max_observation_correction
            );
            phase.time = std::clamp(
                std::max(estimate_->phase_time, phase.time + correction), 0.0, total_time
            );
        }
    }

    const double tau = total_time > 1e-9 ? phase.time / total_time : 0.0;
    const double observed_tau = total_time > 1e-9 ? projection.time / total_time : 0.0;
    const Eigen::Vector2d reference_position = geometry.eval_time(phase.time).p;
    const double observed_arc_length = geometry.arc_length_at_tau(observed_tau);

    RouteEstimate estimate {
        .path = path_,
        .status = projection.position_error <= params_.max_tracking_error
            ? RouteTrackingStatus::TRACKED
            : RouteTrackingStatus::LOST,
        .phase_time = phase.time,
        .observed_phase_time = projection.time,
        .phase_rate = phase.rate,
        .accumulated_delay = accumulated_delay,
        .tau = tau,
        .observed_tau = observed_tau,
        .arc_length = observed_arc_length,
        .remaining_length = std::max(0.0, total_len - observed_arc_length),
        .reference_position = reference_position,
        .projected_position = projection.position,
        .tracking_error = projection.position_error,
    };

    estimate_ = estimate;
    last_stamp_ = stamp;
    return estimate;
}

} // namespace nav_executor
