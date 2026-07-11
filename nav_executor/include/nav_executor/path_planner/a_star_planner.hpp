#pragma once

#include <memory>
#include <expected>
#include <Eigen/Core>
#include <nav_executor/path_planner/nav_map.hpp>

namespace nav_executor {
class AStarPlanner {
public:
    using Ptr = std::shared_ptr<AStarPlanner>;
    using ConstPtr = std::shared_ptr<const AStarPlanner>;

    explicit AStarPlanner(
        const double step_alignment_weight,
        const double obstacle_weight,
        const double step_proximity_weight,
        const double step_mode_dot_threshold,
        const int downsampled_waypoint_max_interval,
        const int feasible_threshold
    );
    std::expected<std::vector<Eigen::Vector2i>, std::string> search_path(
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const TerrainTraversalConstraints& terrain_constraints,
        const Eigen::Vector2i& start_grid,
        const Eigen::Vector2i& goal_grid
    ) const;

private:
    double heuristic(const Eigen::Vector2i& s, const Eigen::Vector2i& t) const;
    bool is_valid(
        const CostMap& cost_map,
        const Eigen::Vector2i& coord
    ) const;
    const std::vector<Eigen::Vector2i> directions_ = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {1, -1}, {-1, -1}
    };
    const double step_alignment_weight_, obstacle_weight_, step_proximity_weight_, step_mode_dot_threshold_;
    const int downsampled_waypoint_max_interval_, feasible_threshold_;
};
} // namespace nav_executor
