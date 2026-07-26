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

// 检测前向轨迹的一般自交。
[[nodiscard]] std::optional<TrajectorySelfIntersection> find_disallowed_self_intersection(
    const MincoTrajectory& trajectory,
    const TrajectoryTopologyParams& params
);

} // namespace nav_executor
