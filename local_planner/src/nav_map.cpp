#include <local_planner/nav_map.hpp>

#include <algorithm>
#include <stdexcept>

namespace local_planner {
CostMap::CostMap(int width, int height, double resolution, double origin_x, double origin_y, const std::vector<uint8_t>& data):
    width(width), height(height), resolution(resolution), origin_x(origin_x), origin_y(origin_y), data(data) {}

CostMap::CostMap(const nav_msgs::msg::OccupancyGrid& occupancy_grid):
    width(static_cast<int>(occupancy_grid.info.width)),
    height(static_cast<int>(occupancy_grid.info.height)),
    resolution(occupancy_grid.info.resolution),
    origin_x(occupancy_grid.info.origin.position.x),
    origin_y(occupancy_grid.info.origin.position.y),
    data(occupancy_grid.data.begin(), occupancy_grid.data.end()) {}

CostMap CostMap::merge(const CostMap& other) const {
    if (width != other.width || height != other.height || resolution != other.resolution || origin_x != other.origin_x || origin_y != other.origin_y) {
        throw std::runtime_error("Cannot merge cost maps with different parameters");
    }

    std::vector<uint8_t> merged_data;
    merged_data.reserve(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        merged_data.push_back(std::max(data[i], other.data[i]));
    }
    return CostMap(width, height, resolution, origin_x, origin_y, merged_data);
}

Eigen::Vector2d CostMap::map_coord_to_grid(const Eigen::Vector2d& map_coord) const {
    return {(map_coord.x() - origin_x) / resolution, (map_coord.y() - origin_y) / resolution};
}

Eigen::Vector2d CostMap::grid_coord_to_map(const Eigen::Vector2d& grid_coord) const {
    return {grid_coord.x() * resolution + origin_x, grid_coord.y() * resolution + origin_y};
}

bool CostMap::is_valid_coord(const Eigen::Vector2i& grid_coord) const {
    return grid_coord.x() >= 0 && grid_coord.x() < width && grid_coord.y() >= 0 && grid_coord.y() < height;
}

bool CostMap::is_valid_coord(const Eigen::Vector2d& grid_coord) const {
    return grid_coord.x() >= 0 && grid_coord.x() + 1 <= width && grid_coord.y() >= 0 && grid_coord.y() + 1 <= height;
}

uint8_t CostMap::at(const Eigen::Vector2i& grid_coord) const {
    if (is_valid_coord(grid_coord)) {
        return data[static_cast<size_t>(grid_coord.y() * width + grid_coord.x())];
    }
    return 255;
}

double CostMap::interpolate(const Eigen::Vector2d& grid_coord) const {
    if (!is_valid_coord(grid_coord)) return 255;

    const int x0 = static_cast<int>(grid_coord.x());
    const int y0 = static_cast<int>(grid_coord.y());
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const double dx = grid_coord.x() - x0;
    const double dy = grid_coord.y() - y0;

    return (1 - dx) * (1 - dy) * at({x0, y0}) + dx * (1 - dy) * at({x1, y0}) + (1 - dx) * dy * at({x0, y1}) + dx * dy * at({x1, y1});
}

Eigen::Vector2d CostMap::gradient(const Eigen::Vector2d& grid_coord) const {
    constexpr int samples = 2;
    const int x = static_cast<int>(grid_coord.x());
    const int y = static_cast<int>(grid_coord.y());

    Eigen::Vector2d sum_grad(0, 0);
    for (int i = 1; i <= samples; i++) {
        sum_grad.x() += (at({x + i, y}) - at({x - i, y})) / (i * 2.0);
        sum_grad.y() += (at({x, y + i}) - at({x, y - i})) / (i * 2.0);
    }
    return sum_grad / samples;
}

std::vector<Eigen::Vector2d> convert_direction_map(const cv::Mat& mat) {
    if (mat.type() != CV_8UC2) {
        throw std::runtime_error("Direction map must be of type CV_8UC2");
    }

    std::vector<Eigen::Vector2d> vec;
    vec.reserve(static_cast<size_t>(mat.cols * mat.rows));
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            const cv::Vec2b val = mat.at<cv::Vec2b>(y, x);
            if ((val[0] == 0 && val[1] == 0) || (val[0] == 128 && val[1] == 128)) {
                vec.emplace_back(0, 0);
            } else {
                Eigen::Vector2d dir(val[0] - 128.0, val[1] - 128.0);
                vec.emplace_back(dir / 128.0);
            }
        }
    }
    return vec;
}

DirectionMap::DirectionMap(const cv::Mat& direction_map, double resolution, double origin_x, double origin_y):
    width(direction_map.cols),
    height(direction_map.rows),
    resolution(resolution),
    origin_x(origin_x),
    origin_y(origin_y),
    data(convert_direction_map(direction_map)) {}

DirectionMap::DirectionMap(int width, int height, double resolution, double origin_x, double origin_y, std::vector<Eigen::Vector2d> data):
    width(width),
    height(height),
    resolution(resolution),
    origin_x(origin_x),
    origin_y(origin_y),
    data(std::move(data)) {
    if (static_cast<int>(this->data.size()) != width * height) {
        throw std::runtime_error("DirectionMap data size does not match width*height");
    }
}

Eigen::Vector2d DirectionMap::map_coord_to_grid(const Eigen::Vector2d& map_coord) const {
    return {(map_coord.x() - origin_x) / resolution, (map_coord.y() - origin_y) / resolution};
}

Eigen::Vector2d DirectionMap::grid_coord_to_map(const Eigen::Vector2d& grid_coord) const {
    return {grid_coord.x() * resolution + origin_x, grid_coord.y() * resolution + origin_y};
}

bool DirectionMap::is_valid_coord(const Eigen::Vector2i& grid_coord) const {
    return grid_coord.x() >= 0 && grid_coord.x() < width && grid_coord.y() >= 0 && grid_coord.y() < height;
}

bool DirectionMap::is_valid_coord(const Eigen::Vector2d& grid_coord) const {
    return grid_coord.x() >= 0 && grid_coord.x() + 1 <= width && grid_coord.y() >= 0 && grid_coord.y() + 1 <= height;
}

Eigen::Vector2d DirectionMap::at(const Eigen::Vector2i& grid_coord) const {
    if (is_valid_coord(grid_coord)) {
        return data[static_cast<size_t>(grid_coord.y() * width + grid_coord.x())];
    }
    return {0, 0};
}

Eigen::Vector2d DirectionMap::interpolate(const Eigen::Vector2d& grid_coord) const {
    if (!is_valid_coord(grid_coord)) return {0, 0};

    const int x0 = static_cast<int>(grid_coord.x());
    const int y0 = static_cast<int>(grid_coord.y());
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const double dx = grid_coord.x() - x0;
    const double dy = grid_coord.y() - y0;

    return (1 - dx) * (1 - dy) * at({x0, y0}) + dx * (1 - dy) * at({x1, y0}) + (1 - dx) * dy * at({x0, y1}) + dx * dy * at({x1, y1});
}
}