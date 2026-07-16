#pragma once

#include <Eigen/Core>

#include <nav_executor/common/minco_trajectory.hpp>

namespace nav_executor {

struct TrajectoryProjectionParams {
    double heading_weight = 0.5;
    double velocity_weight = 0.2;
    double projection_window_backward = 0.5;
    double projection_window_forward = 1.0;
    double max_backward_step = 0.05;
    double max_forward_step = 0.15;
};

struct TrajectoryProjection {
    double time = 0.0;
};

[[nodiscard]] TrajectoryProjection project_trajectory(
    const MincoTrajectory& trajectory,
    const Eigen::Vector3d& chassis_pose_map,
    double chassis_velocity,
    double time_lo,
    double time_hi,
    const TrajectoryProjectionParams& params
);

} // namespace nav_executor
