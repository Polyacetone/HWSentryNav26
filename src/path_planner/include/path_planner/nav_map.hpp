#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

namespace path_planner {
class CostMap {
public:
    using Ptr = std::shared_ptr<CostMap>;
    explicit CostMap(const nav_msgs::msg::OccupancyGrid& occupancy_grid);
    Eigen::Vector2d map_coord_to_grid(const Eigen::Vector2d& map_coord) const;
    Eigen::Vector2d grid_coord_to_map(const Eigen::Vector2d& grid_coord) const;

    bool is_valid_coord(const Eigen::Vector2i& grid_coord) const;
    bool is_valid_coord(const Eigen::Vector2d& grid_coord) const;
    uint8_t at(const Eigen::Vector2i& grid_coord) const;
    double interpolate(const Eigen::Vector2d& grid_coord) const;
    Eigen::Vector2d gradient(const Eigen::Vector2d& grid_coord) const;

    const int width, height;
    const double resolution, origin_x, origin_y;
    const std::vector<uint8_t> data; // 占用网格数据，0-255
};

class DirectionMap {
public:
    using Ptr = std::shared_ptr<DirectionMap>;
    explicit DirectionMap(const cv::Mat& direction_map, double resolution, double origin_x, double origin_y);
    Eigen::Vector2d map_coord_to_grid(const Eigen::Vector2d& map_coord) const;
    Eigen::Vector2d grid_coord_to_map(const Eigen::Vector2d& grid_coord) const;

    bool is_valid_coord(const Eigen::Vector2i& grid_coord) const;
    bool is_valid_coord(const Eigen::Vector2d& grid_coord) const;
    Eigen::Vector2d at(const Eigen::Vector2i& grid_coord) const;
    Eigen::Vector2d interpolate(const Eigen::Vector2d& grid_coord) const;

    const int width, height;
    const double resolution, origin_x, origin_y;
    const std::vector<Eigen::Vector2d> data; // 归一化后的方向向量
};
} // namespace path_planner