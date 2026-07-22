#pragma once

#include <optional>

#include <nav_executor/common/minco_trajectory.hpp>

namespace nav_executor {

struct TrajectoryTopologyParams {
    double flatness_tolerance;
    double max_edge_length;
    double cusp_retrace_alignment_threshold;
};

struct TrajectorySelfIntersection {
    int first_segment;
    int second_segment;
};

// 检测不属于冻结换向拓扑的自交。相反 gear 的重合只有在能够从中间 cusp 连续回描到
// 交点时才合法；同段回环、同 gear 交叉和与 cusp 不连通的二次交点均返回异常。
[[nodiscard]] std::optional<TrajectorySelfIntersection> find_disallowed_self_intersection(
    const MincoTrajectory& trajectory,
    const TrajectoryTopologyParams& params
);

} // namespace nav_executor
