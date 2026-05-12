#include <path_follower/nav_map.hpp>

#include <algorithm>
#include <stdexcept>

namespace {

bool is_step_direction_pixel(const cv::Vec3b& val) {
    const bool zero = val[0] == 0 && val[1] == 0;
    const bool neutral = val[0] == 128 && val[1] == 128;
    return !(zero || neutral);
}

std::pair<std::vector<Eigen::Vector2d>, std::vector<uint8_t>> convert_direction_map(const cv::Mat& mat) {
    if (mat.type() != CV_8UC3) {
        throw std::runtime_error("Direction map must be of type CV_8UC3");
    }

    std::vector<Eigen::Vector2d> vec;
    std::vector<uint8_t> step_mode_vec;
    vec.reserve(static_cast<size_t>(mat.cols * mat.rows));
    step_mode_vec.reserve(static_cast<size_t>(mat.cols * mat.rows));
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            const cv::Vec3b val = mat.at<cv::Vec3b>(y, x);
            if (is_step_direction_pixel(val)) {
                Eigen::Vector2d dir(val[0] - 128.0, val[1] - 128.0);
                vec.emplace_back(dir / 128.0);
                step_mode_vec.push_back(val[2]);
            } else {
                vec.emplace_back(0.0, 0.0);
                step_mode_vec.push_back(0);
            }
        }
    }
    return {std::move(vec), std::move(step_mode_vec)};
}

} // namespace

namespace path_follower {
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

DirectionMap::DirectionMap(const cv::Mat& direction_map, double resolution, double origin_x, double origin_y):
    DirectionMap(
        direction_map.cols,
        direction_map.rows,
        resolution,
        origin_x,
        origin_y,
        convert_direction_map(direction_map)
    ) {}

DirectionMap::DirectionMap(int width, int height, double resolution, double origin_x, double origin_y,
    std::pair<std::vector<Eigen::Vector2d>, std::vector<uint8_t>> data_pair):
    width(width),
    height(height),
    resolution(resolution),
    origin_x(origin_x),
    origin_y(origin_y),
    data(std::move(data_pair.first)),
    step_mode_data(data_pair.second.empty() ? std::vector<uint8_t>(static_cast<size_t>(width * height), 0) : std::move(data_pair.second)) {
    if (static_cast<int>(this->data.size()) != width * height) {
        throw std::runtime_error("DirectionMap data size does not match width*height");
    }
    if (static_cast<int>(this->step_mode_data.size()) != width * height) {
        throw std::runtime_error("DirectionMap step_mode_data size does not match width*height");
    }
}

DirectionMap::DirectionMap(int width, int height, double resolution, double origin_x, double origin_y,
    std::vector<Eigen::Vector2d> data, std::vector<uint8_t> step_mode_data):
    DirectionMap(width, height, resolution, origin_x, origin_y,
        std::pair{std::move(data), std::move(step_mode_data)}) {}

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

uint8_t DirectionMap::step_mode_at(const Eigen::Vector2i& grid_coord) const {
    if (is_valid_coord(grid_coord)) {
        return step_mode_data[static_cast<size_t>(grid_coord.y() * width + grid_coord.x())];
    }
    return 0;
}

uint8_t DirectionMap::step_mode_at(const Eigen::Vector2d& grid_coord) const {
    const Eigen::Array2i floored = grid_coord.array().floor().cast<int>();
    return step_mode_at(Eigen::Vector2i(floored.x(), floored.y()));
}

StepModeInfo DirectionMap::step_mode_info_at(const Eigen::Vector2i& grid_coord) const {
    return decode_step_mode(step_mode_at(grid_coord));
}

StepModeInfo DirectionMap::step_mode_info_at(const Eigen::Vector2d& grid_coord) const {
    return decode_step_mode(step_mode_at(grid_coord));
}
}
