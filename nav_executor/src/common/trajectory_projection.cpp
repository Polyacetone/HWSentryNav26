#include <nav_executor/common/trajectory_projection.hpp>

#include <algorithm>
#include <cmath>

namespace nav_executor {

namespace {

double wrap_angle(const double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

double projection_cost(
    const MincoTrajectory& trajectory,
    const Eigen::Vector3d& pose,
    const double velocity,
    const double time,
    const TrajectoryProjectionParams& params
) {
    const TrajSample sample = trajectory.eval_time(time);
    const Eigen::Vector2d position_error = pose.head<2>() - sample.p;
    const double heading_error = wrap_angle(pose.z() - sample.theta);
    const double velocity_error = velocity - trajectory.longitudinal_velocity(sample);
    const double weighted_heading = params.heading_weight * heading_error;
    const double weighted_velocity = params.velocity_weight * velocity_error;
    return position_error.squaredNorm()
        + weighted_heading * weighted_heading
        + weighted_velocity * weighted_velocity;
}

} // anonymous namespace

TrajectoryProjection project_trajectory(
    const MincoTrajectory& trajectory,
    const Eigen::Vector3d& chassis_pose_map,
    const double chassis_velocity,
    double time_lo,
    double time_hi,
    const TrajectoryProjectionParams& params
) {
    TrajectoryProjection result;
    if (trajectory.empty()) return result;

    time_lo = std::clamp(time_lo, 0.0, trajectory.total_time());
    time_hi = std::clamp(time_hi, 0.0, trajectory.total_time());
    if (time_lo > time_hi) std::swap(time_lo, time_hi);

    constexpr int COARSE_SAMPLES = 60;
    constexpr int REFINE_ITERS = 24;
    double best_time = time_lo;
    double best_cost = projection_cost(
        trajectory, chassis_pose_map, chassis_velocity, best_time, params
    );
    for (int i = 1; i <= COARSE_SAMPLES; ++i) {
        const double time = std::lerp(
            time_lo, time_hi, static_cast<double>(i) / static_cast<double>(COARSE_SAMPLES)
        );
        const double cost = projection_cost(
            trajectory, chassis_pose_map, chassis_velocity, time, params
        );
        if (cost < best_cost) {
            best_cost = cost;
            best_time = time;
        }
    }

    const double span = (time_hi - time_lo) / static_cast<double>(COARSE_SAMPLES);
    double left = std::max(time_lo, best_time - span);
    double right = std::min(time_hi, best_time + span);
    for (int i = 0; i < REFINE_ITERS; ++i) {
        const double m1 = left + (right - left) / 3.0;
        const double m2 = right - (right - left) / 3.0;
        if (projection_cost(trajectory, chassis_pose_map, chassis_velocity, m1, params)
            <= projection_cost(trajectory, chassis_pose_map, chassis_velocity, m2, params)) {
            right = m2;
        } else {
            left = m1;
        }
    }

    result.time = 0.5 * (left + right);
    return result;
}

} // namespace nav_executor
