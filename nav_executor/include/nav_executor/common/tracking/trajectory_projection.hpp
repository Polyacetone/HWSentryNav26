#pragma once

#include <Eigen/Core>

#include <nav_executor/common/trajectory/minco_trajectory.hpp>

namespace nav_executor {

struct TrajectoryProjectionParams {
    double prediction_weight = 1.0;
    double search_distance_backward = 0.2;
    double search_distance_forward = 0.4;
};

struct TrajectoryProjection {
    double arc_length = 0.0;
    double tracking_error = 0.0;
};

[[nodiscard]] TrajectoryProjection project_trajectory(
    const MincoTrajectory& trajectory,
    const Eigen::Vector2d& chassis_position_map,
    double predicted_arc_length,
    double arc_length_lo,
    double arc_length_hi,
    const TrajectoryProjectionParams& params
);

} // namespace nav_executor
