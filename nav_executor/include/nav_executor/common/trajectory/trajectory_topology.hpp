#pragma once

#include <optional>

#include <nav_executor/common/trajectory/minco_trajectory.hpp>

namespace nav_executor {

struct TrajectoryTopologyParams {
    double flatness_tolerance;
    double max_edge_length;
};

struct TrajectorySelfIntersection {
    int first_segment;
    int second_segment;
};

// 基于带几何误差界的自适应折线，保守检测可能的自交。
[[nodiscard]] std::optional<TrajectorySelfIntersection> find_possible_self_intersection(
    const MincoTrajectory& trajectory,
    const TrajectoryTopologyParams& params
);

} // namespace nav_executor
