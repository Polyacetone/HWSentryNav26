#pragma once

#include <memory>
#include <Eigen/Core>
#include <path_planner/costmap_2d.hpp>

namespace path_planner {

struct Node {
    using Ptr = std::shared_ptr<Node>;
    Eigen::Vector2i coord;
    double g, h;
    Node::Ptr parent;
    double f() const { return g + h; }
};

class AStarPlanner {
public:
    explicit AStarPlanner(
        const int downsampled_waypoint_max_interval,
        const int occupied_threshold
    );
    std::vector<Eigen::Vector2i> search_path(
        const Costmap2D& costmap,
        const Eigen::Vector2i& start_grid,
        const Eigen::Vector2i& goal_grid
    ) const;

private:
    double heuristic(const Eigen::Vector2i& s, const Eigen::Vector2i& t) const;
    bool is_line_safe(
        const Costmap2D& costmap,
        const Eigen::Vector2i& s,
        const Eigen::Vector2i& t
    ) const;
    bool is_valid(
        const Costmap2D& costmap,
        const Eigen::Vector2i& coord
    ) const;
    std::vector<Eigen::Vector2i> downsample_path(
        const std::vector<Eigen::Vector2i>& path,
        const int start, const int end
    ) const;

    const std::vector<Eigen::Vector2i> directions_ = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {1, -1}, {-1, -1}
    };
    const int downsampled_waypoint_max_interval_, occupied_threshold_;
};

} // namespace path_planner