#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

namespace mpc_controller {
class CostMap {
public:
    using Ptr = std::shared_ptr<CostMap>;
    using ConstPtr = std::shared_ptr<const CostMap>;

    explicit CostMap(int width, int height, double resolution, double origin_x, double origin_y, const std::vector<uint8_t>& data);
    explicit CostMap(const nav_msgs::msg::OccupancyGrid& occupancy_grid);

    CostMap merge(const CostMap& other) const;

    Eigen::Vector2d map_coord_to_grid(const Eigen::Vector2d& map_coord) const;
    Eigen::Vector2d grid_coord_to_map(const Eigen::Vector2d& grid_coord) const;

    bool is_valid_coord(const Eigen::Vector2i& grid_coord) const;
    bool is_valid_coord(const Eigen::Vector2d& grid_coord) const;

    uint8_t at(const Eigen::Vector2i& grid_coord) const;
    double interpolate(const Eigen::Vector2d& grid_coord) const;
    Eigen::Vector2d gradient(const Eigen::Vector2d& grid_coord) const;

    const int width, height;
    const double resolution, origin_x, origin_y;
    const std::vector<uint8_t> data;
};

class DirectionMap {
public:
    using Ptr = std::shared_ptr<DirectionMap>;
    using ConstPtr = std::shared_ptr<const DirectionMap>;

    explicit DirectionMap(const cv::Mat& direction_map, double resolution, double origin_x, double origin_y);
    explicit DirectionMap(int width, int height, double resolution, double origin_x, double origin_y, std::vector<Eigen::Vector2d> data);

    Eigen::Vector2d map_coord_to_grid(const Eigen::Vector2d& map_coord) const;
    Eigen::Vector2d grid_coord_to_map(const Eigen::Vector2d& grid_coord) const;

    bool is_valid_coord(const Eigen::Vector2i& grid_coord) const;
    bool is_valid_coord(const Eigen::Vector2d& grid_coord) const;

    Eigen::Vector2d at(const Eigen::Vector2i& grid_coord) const;
    Eigen::Vector2d interpolate(const Eigen::Vector2d& grid_coord) const;

    const int width, height;
    const double resolution, origin_x, origin_y;
    const std::vector<Eigen::Vector2d> data;
};
}