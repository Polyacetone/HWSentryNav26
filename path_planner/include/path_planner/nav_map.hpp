#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

namespace path_planner {

enum class TerrainType : uint8_t {
    FLAT = 0,
    OBSTACLE = 1,
    SLOPE = 2,
    STEP_L1 = 3,
    STEP_L2 = 4,
    FLY_SLOPE = 5,
    STEP_HIGH = 6
};

struct TerrainRule {
    bool forward_allowed = true;
    bool backward_allowed = true;
};

using TerrainRuleTable = std::array<TerrainRule, 7>;

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

class DirectionMap {
public:
    using Ptr = std::shared_ptr<DirectionMap>;
    using ConstPtr = std::shared_ptr<const DirectionMap>;
    explicit DirectionMap(const cv::Mat& direction_map, double resolution, double origin_x, double origin_y, const TerrainRuleTable& rules);
    explicit DirectionMap(int width, int height, double resolution, double origin_x, double origin_y, std::pair<std::vector<Eigen::Vector2d>, std::vector<uint8_t>> data_pair, const TerrainRuleTable& rules);
    Eigen::Vector2d map_coord_to_grid(const Eigen::Vector2d& map_coord) const;
    Eigen::Vector2d grid_coord_to_map(const Eigen::Vector2d& grid_coord) const;

    bool is_valid_coord(const Eigen::Vector2i& grid_coord) const;
    bool is_valid_coord(const Eigen::Vector2d& grid_coord) const;
    Eigen::Vector2d at(const Eigen::Vector2i& grid_coord) const;
    uint8_t terrain_at(const Eigen::Vector2i& grid_coord) const;
    Eigen::Vector2d interpolate(const Eigen::Vector2d& grid_coord) const;
    bool is_fully_prohibited(const Eigen::Vector2d& grid_coord) const;
    double prohibited_direction_score(const Eigen::Vector2i& grid_coord, const Eigen::Vector2d& move_dir, double dot_threshold) const;
    double prohibited_direction_score(const Eigen::Vector2d& grid_coord, const Eigen::Vector2d& move_dir, double dot_threshold) const;
    bool is_direction_prohibited(const Eigen::Vector2i& grid_coord, const Eigen::Vector2d& move_dir, double dot_threshold) const;

    const int width, height;
    const double resolution, origin_x, origin_y;
    const std::vector<Eigen::Vector2d> data; // 归一化后的方向向量（从 angle+mag 解码）
    const std::vector<uint8_t> terrain; // 原始地形标签 (0-6)

private:
    TerrainRuleTable rules_;
};

} // namespace path_planner
