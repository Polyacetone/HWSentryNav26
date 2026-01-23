#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <memory>
#include <vector>

namespace path_planner {
class CostMap {
public:
    using Ptr = std::shared_ptr<CostMap>;
    using ConstPtr = std::shared_ptr<const CostMap>;
    explicit CostMap(const int width, const int height, const double resolution, const double origin_x, const double origin_y, const std::vector<uint8_t>& data);
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
    const std::vector<uint8_t> data; // 占用网格数据，0-255
};

class ESDFMap {
public:
    using Ptr = std::shared_ptr<ESDFMap>;
    using ConstPtr = std::shared_ptr<const ESDFMap>;

    explicit ESDFMap(
        int width,
        int height,
        double resolution,
        double origin_x,
        double origin_y,
        std::vector<float> signed_distance_m
    );

    static ESDFMap from_cost_map(const CostMap& cost_map, int obstacle_threshold);

    Eigen::Vector2d map_coord_to_grid(const Eigen::Vector2d& map_coord) const;
    Eigen::Vector2d grid_coord_to_map(const Eigen::Vector2d& grid_coord) const;

    bool is_valid_coord(const Eigen::Vector2i& grid_coord) const;
    bool is_valid_coord(const Eigen::Vector2d& grid_coord) const;

    // 返回格点坐标处的有符号距离（单位：米）
    float at(const Eigen::Vector2i& grid_coord) const;
    double interpolate(const Eigen::Vector2d& grid_coord) const;
    // 返回对格点坐标的梯度：∂d/∂(x,y)，单位为 (m / cell)
    Eigen::Vector2d gradient(const Eigen::Vector2d& grid_coord) const;

    const int width, height;
    const double resolution, origin_x, origin_y;
    const std::vector<float> signed_distance_m; // 行主序，size = width * height
};

class DirectionMap {
public:
    using Ptr = std::shared_ptr<DirectionMap>;
    using ConstPtr = std::shared_ptr<const DirectionMap>;
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