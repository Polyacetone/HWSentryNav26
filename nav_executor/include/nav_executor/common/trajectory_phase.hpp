#pragma once

#include <Eigen/Core>

#include <nav_executor/common/minco_trajectory.hpp>

namespace nav_executor {

struct TrajectoryPhaseParams {
    double error_scale = 0.5;
    double heading_weight = 1.0;
    double projection_heading_weight = 0.5;
    double projection_velocity_weight = 0.2;
    double projection_window_backward = 0.5;
    double projection_window_forward = 1.0;
    double observation_gain = 0.25;
    double max_observation_correction = 0.05;
    double rate_accel = 1.0;
    double rate_decel = 4.0;
    double pause_error = 1.5;
};

struct TrajectoryPhaseState {
    double time = 0.0;
    double rate = 0.0;
};

struct TrajectoryPhaseProjection {
    double time = 0.0;
    double cost = 0.0;
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    double position_error = 0.0;
};

[[nodiscard]] double trajectory_phase_rate_target(
    const TrajSample& reference,
    const Eigen::Vector3d& chassis_pose_map,
    const TrajectoryPhaseParams& params
);

[[nodiscard]] TrajectoryPhaseState advance_trajectory_phase(
    const MincoTrajectory& trajectory,
    const TrajectoryPhaseState& state,
    const Eigen::Vector3d& chassis_pose_map,
    double dt,
    const TrajectoryPhaseParams& params
);

[[nodiscard]] TrajectoryPhaseProjection project_trajectory_phase(
    const MincoTrajectory& trajectory,
    const Eigen::Vector3d& chassis_pose_map,
    double chassis_velocity,
    double time_lo,
    double time_hi,
    const TrajectoryPhaseParams& params
);

} // namespace nav_executor
