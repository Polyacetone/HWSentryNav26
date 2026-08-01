#pragma once

#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/environment/nav_map.hpp>

namespace nav_executor {

struct GridCrossing {
    Eigen::Vector2i from = Eigen::Vector2i::Zero();
    Eigen::Vector2i to = Eigen::Vector2i::Zero();
    double fraction = 0.0;
};

[[nodiscard]] bool same_cell(
    const Eigen::Vector2i& lhs,
    const Eigen::Vector2i& rhs
);

[[nodiscard]] bool grid_cell_traversable(
    const CostMap& cost_map,
    const Eigen::Vector2i& cell,
    int occupied_threshold
);

[[nodiscard]] bool grid_edge_avoids_corner_cutting(
    const CostMap& cost_map,
    const Eigen::Vector2i& from,
    const Eigen::Vector2i& to,
    int occupied_threshold
);

// Directional terrain is a hard constraint only on physical body cells.
[[nodiscard]] bool directed_terrain_edge_allowed(
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const Eigen::Vector2i& from,
    const Eigen::Vector2i& to,
    double min_alignment_cosine,
    bool* going_up = nullptr
);

[[nodiscard]] std::vector<GridCrossing> trace_grid_crossings(
    const GridGeometry& geometry,
    const Eigen::Vector2d& from_map,
    const Eigen::Vector2d& to_map
);

} // namespace nav_executor
