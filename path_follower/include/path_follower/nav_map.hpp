#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

namespace path_follower {

enum class StepTraversalMode : uint8_t {
    FORBIDDEN = 0,
    JUMP = 1,
    LEG_SHORT = 2,
    LEG_LONG = 3,
};

struct StepModeInfo {
    StepTraversalMode up_mode;
    StepTraversalMode down_mode;
    uint8_t up_speed_level;
    uint8_t down_speed_level;
};

constexpr StepModeInfo decode_step_mode(uint8_t step_mode) {
    return {
        .up_mode = static_cast<StepTraversalMode>(step_mode & 0b11),
        .down_mode = static_cast<StepTraversalMode>((step_mode >> 2) & 0b11),
        .up_speed_level = static_cast<uint8_t>((step_mode >> 4) & 0b11),
        .down_speed_level = static_cast<uint8_t>((step_mode >> 6) & 0b11),
    };
}

constexpr bool is_step_traversal_allowed(StepTraversalMode mode) {
    return mode == StepTraversalMode::JUMP || mode == StepTraversalMode::LEG_SHORT || mode == StepTraversalMode::LEG_LONG;
}

static_assert(decode_step_mode(0b11100110).up_mode == StepTraversalMode::LEG_SHORT);
static_assert(decode_step_mode(0b11100110).down_mode == StepTraversalMode::JUMP);
static_assert(decode_step_mode(0b11100110).up_speed_level == 2);
static_assert(decode_step_mode(0b11100110).down_speed_level == 3);

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
    explicit DirectionMap(int width, int height, double resolution, double origin_x, double origin_y, std::pair<std::vector<Eigen::Vector2d>, std::vector<uint8_t>> data_pair);
    explicit DirectionMap(int width, int height, double resolution, double origin_x, double origin_y, std::vector<Eigen::Vector2d> data, std::vector<uint8_t> step_mode_data = {});

    Eigen::Vector2d map_coord_to_grid(const Eigen::Vector2d& map_coord) const;
    Eigen::Vector2d grid_coord_to_map(const Eigen::Vector2d& grid_coord) const;

    bool is_valid_coord(const Eigen::Vector2i& grid_coord) const;
    bool is_valid_coord(const Eigen::Vector2d& grid_coord) const;

    Eigen::Vector2d at(const Eigen::Vector2i& grid_coord) const;
    Eigen::Vector2d interpolate(const Eigen::Vector2d& grid_coord) const;
    uint8_t step_mode_at(const Eigen::Vector2i& grid_coord) const;
    uint8_t step_mode_at(const Eigen::Vector2d& grid_coord) const;
    StepModeInfo step_mode_info_at(const Eigen::Vector2i& grid_coord) const;
    StepModeInfo step_mode_info_at(const Eigen::Vector2d& grid_coord) const;

    const int width, height;
    const double resolution, origin_x, origin_y;
    const std::vector<Eigen::Vector2d> data;
    const std::vector<uint8_t> step_mode_data;
};
}
