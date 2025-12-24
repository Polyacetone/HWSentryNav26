#pragma once

#include <memory>
#include <Eigen/Core>
#include <path_planner/nav_map.hpp>

namespace path_planner {
class AStarPlanner {
public:
    using Ptr = std::shared_ptr<AStarPlanner>;
    using ConstPtr = std::shared_ptr<const AStarPlanner>;

    explicit AStarPlanner(
        const double direction_weight,
        const double obstacle_weight,
        const int downsampled_waypoint_max_interval,
        const int feasible_threshold
    );
    std::vector<Eigen::Vector2i> search_path(
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const Eigen::Vector2i& start_grid,
        const Eigen::Vector2i& goal_grid
    ) const;

private:
    double heuristic(const Eigen::Vector2i& s, const Eigen::Vector2i& t) const;
    bool is_line_safe(
        const CostMap& cost_map,
        const Eigen::Vector2i& s,
        const Eigen::Vector2i& t
    ) const;
    bool is_valid(
        const CostMap& cost_map,
        const Eigen::Vector2i& coord
    ) const;
    std::vector<Eigen::Vector2i> downsample_path(
        const std::vector<Eigen::Vector2i>& path,
        const int start, const int end
    ) const;

    const std::vector<Eigen::Vector2i> directions_ = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {1, -1}, {-1, -1}
    };
    const double direction_weight_, obstacle_weight_;
    const int downsampled_waypoint_max_interval_, feasible_threshold_;
};
} // namespace path_planner