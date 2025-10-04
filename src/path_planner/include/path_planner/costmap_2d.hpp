#pragma once

#include <Eigen/Core>
#include <nav_msgs/msg/occupancy_grid.hpp>

namespace path_planner {
class Costmap2D {
public:
    explicit Costmap2D(const nav_msgs::msg::OccupancyGrid& occupancy_grid);
    Eigen::Vector2d map_coord_to_grid(const Eigen::Vector2d& map_coord) const;
    Eigen::Vector2d grid_coord_to_map(const Eigen::Vector2d& grid_coord) const;

    bool is_valid_coord(const Eigen::Vector2i& grid_coord) const;
    bool is_valid_coord(const Eigen::Vector2d& grid_coord) const;
    int8_t at(const Eigen::Vector2i& grid_coord) const;
    double interpolate(const Eigen::Vector2d& grid_coord) const;
    Eigen::Vector2d gradient(const Eigen::Vector2d& grid_coord) const;

    unsigned width() const;
    unsigned height() const;
    unsigned size() const;

private:
    const unsigned width_, height_;
    const double resolution_, origin_x_, origin_y_;
    const std::vector<int8_t> data_;
};
} // namespace path_planner