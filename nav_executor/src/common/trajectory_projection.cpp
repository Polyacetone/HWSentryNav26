#include <nav_executor/common/trajectory_projection.hpp>

#include <algorithm>
#include <cmath>

namespace nav_executor {

namespace {

double projection_cost(
    const MincoTrajectory& trajectory,
    const Eigen::Vector2d& position,
    const double predicted_arc_length,
    const double arc_length,
    const TrajectoryProjectionParams& params
) {
    const TrajSample sample = trajectory.eval_arc_length(arc_length);
    const Eigen::Vector2d position_error = position - sample.p;
    const double weighted_prediction = params.prediction_weight
        * (arc_length - predicted_arc_length);
    return position_error.squaredNorm()
        + weighted_prediction * weighted_prediction;
}

} // anonymous namespace

TrajectoryProjection project_trajectory(
    const MincoTrajectory& trajectory,
    const Eigen::Vector2d& chassis_position_map,
    const double predicted_arc_length,
    double arc_length_lo,
    double arc_length_hi,
    const TrajectoryProjectionParams& params
) {
    TrajectoryProjection result;
    if (trajectory.empty()) return result;

    arc_length_lo = std::clamp(arc_length_lo, 0.0, trajectory.total_arc_length());
    arc_length_hi = std::clamp(arc_length_hi, 0.0, trajectory.total_arc_length());
    if (arc_length_lo > arc_length_hi) arc_length_lo = arc_length_hi;

    constexpr int COARSE_SAMPLES = 60;
    constexpr int REFINE_ITERS = 24;
    double best_arc_length = arc_length_lo;
    double best_cost = projection_cost(
        trajectory, chassis_position_map, predicted_arc_length, best_arc_length, params
    );
    for (int i = 1; i <= COARSE_SAMPLES; ++i) {
        const double arc_length = std::lerp(
            arc_length_lo, arc_length_hi,
            static_cast<double>(i) / static_cast<double>(COARSE_SAMPLES)
        );
        const double cost = projection_cost(
            trajectory, chassis_position_map, predicted_arc_length, arc_length, params
        );
        if (cost < best_cost) {
            best_cost = cost;
            best_arc_length = arc_length;
        }
    }

    const double span = (arc_length_hi - arc_length_lo)
        / static_cast<double>(COARSE_SAMPLES);
    double left = std::max(arc_length_lo, best_arc_length - span);
    double right = std::min(arc_length_hi, best_arc_length + span);
    for (int i = 0; i < REFINE_ITERS; ++i) {
        const double m1 = left + (right - left) / 3.0;
        const double m2 = right - (right - left) / 3.0;
        if (projection_cost(trajectory, chassis_position_map, predicted_arc_length, m1, params)
            <= projection_cost(trajectory, chassis_position_map, predicted_arc_length, m2, params)) {
            right = m2;
        } else {
            left = m1;
        }
    }

    result.arc_length = 0.5 * (left + right);
    result.tracking_error = (
        trajectory.eval_arc_length(result.arc_length).p - chassis_position_map
    ).norm();
    return result;
}

} // namespace nav_executor
