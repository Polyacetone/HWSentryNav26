#include <path_planner/nav_map.hpp>

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

    std::vector<Eigen::Vector2d> dir_vec;
    std::vector<uint8_t> step_mode_vec;
    dir_vec.reserve(static_cast<size_t>(mat.cols) * static_cast<size_t>(mat.rows));
    step_mode_vec.reserve(static_cast<size_t>(mat.cols) * static_cast<size_t>(mat.rows));

    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            const cv::Vec3b val = mat.at<cv::Vec3b>(y, x);
            if (is_step_direction_pixel(val)) {
                const Eigen::Vector2d dir(val[0] - 128.0, val[1] - 128.0);
                dir_vec.emplace_back(dir / 128.0);
                step_mode_vec.push_back(val[2]);
            } else {
                dir_vec.emplace_back(0.0, 0.0);
                step_mode_vec.push_back(0);
            }
        }
    }

    return {std::move(dir_vec), std::move(step_mode_vec)};
}

} // namespace

namespace path_planner {
namespace {

double prohibited_direction_score_impl(const Eigen::Vector2d& step_dir, uint8_t step_mode, const Eigen::Vector2d& move_dir, double dot_threshold) {
    if (step_dir.squaredNorm() <= 1e-12 || move_dir.squaredNorm() <= 1e-12) {
        return 0.0;
    }

    const Eigen::Vector2d normalized_step_dir = step_dir.normalized();
    const Eigen::Vector2d normalized_move_dir = move_dir.normalized();
    const double alignment = normalized_move_dir.dot(normalized_step_dir);
    const double clamped_threshold = std::clamp(dot_threshold, 0.0, 1.0);
    const double denom = std::max(1.0 - clamped_threshold, 1e-6);

    if (alignment > clamped_threshold && extract_up_mode(step_mode) == STEP_MODE_FORBIDDEN) {
        return std::clamp((alignment - clamped_threshold) / denom, 0.0, 1.0);
    }

    if (alignment < -clamped_threshold && extract_down_mode(step_mode) == STEP_MODE_FORBIDDEN) {
        return std::clamp((-alignment - clamped_threshold) / denom, 0.0, 1.0);
    }

    return 0.0;
}

} // namespace
} // namespace path_planner

namespace path_planner {
CostMap::CostMap(
    const int width,
    const int height,
    const double resolution,
    const double origin_x,
    const double origin_y,
    const std::vector<uint8_t>& data
):  width(width),
    height(height),
    resolution(resolution),
    origin_x(origin_x),
    origin_y(origin_y),
    data(data) {}

CostMap::CostMap(const nav_msgs::msg::OccupancyGrid& occupancy_grid):
    width(static_cast<int>(occupancy_grid.info.width)),
    height(static_cast<int>(occupancy_grid.info.height)),
    resolution(occupancy_grid.info.resolution),
    origin_x(occupancy_grid.info.origin.position.x),
    origin_y(occupancy_grid.info.origin.position.y),
    data(occupancy_grid.data.begin(), occupancy_grid.data.end()) {}

CostMap CostMap::merge(const CostMap& other) const {
    if (width != other.width || height != other.height || resolution != other.resolution ||
        origin_x != other.origin_x || origin_y != other.origin_y) {
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
    if (is_valid_coord(grid_coord)) return data[static_cast<size_t>(grid_coord.y()) * static_cast<size_t>(width) + static_cast<size_t>(grid_coord.x())];
    else return 255;
}

double CostMap::interpolate(const Eigen::Vector2d& grid_coord) const {
    if (!is_valid_coord(grid_coord)) return 255;
    const int x0 = static_cast<int>(grid_coord.x()), y0 = static_cast<int>(grid_coord.y());
    const int x1 = x0 + 1, y1 = y0 + 1;
    const double dx = grid_coord.x() - x0, dy = grid_coord.y() - y0;
    return (1 - dx) * (1 - dy) * at({x0, y0}) + dx * (1 - dy) * at({x1, y0}) +
        (1 - dx) * dy * at({x0, y1}) + dx * dy * at({x1, y1});
}

Eigen::Vector2d CostMap::gradient(const Eigen::Vector2d& grid_coord) const {
    constexpr int samples = 2;
    const int x = static_cast<int>(grid_coord.x()), y = static_cast<int>(grid_coord.y());
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
    step_mode_data(std::move(data_pair.second)) {
    if (static_cast<int>(this->data.size()) != width * height) {
        throw std::runtime_error("DirectionMap data size does not match width*height");
    }
    if (static_cast<int>(this->step_mode_data.size()) != width * height) {
        throw std::runtime_error("DirectionMap step_mode_data size does not match width*height");
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
    if (is_valid_coord(grid_coord)) return data[static_cast<size_t>(grid_coord.y()) * static_cast<size_t>(width) + static_cast<size_t>(grid_coord.x())];
    else return {0, 0};
}

uint8_t DirectionMap::step_mode_at(const Eigen::Vector2i& grid_coord) const {
    if (is_valid_coord(grid_coord)) return step_mode_data[static_cast<size_t>(grid_coord.y()) * static_cast<size_t>(width) + static_cast<size_t>(grid_coord.x())];
    return 0;
}

uint8_t DirectionMap::step_mode_at(const Eigen::Vector2d& grid_coord) const {
    if (!is_valid_coord(grid_coord)) return 0;
    const Eigen::Array2i floored = grid_coord.array().floor().template cast<int>();
    return step_mode_at(Eigen::Vector2i(floored.x(), floored.y()));
}

Eigen::Vector2d DirectionMap::interpolate(const Eigen::Vector2d& grid_coord) const {
    if (!is_valid_coord(grid_coord)) return {0, 0};
    const int x0 = static_cast<int>(grid_coord.x()), y0 = static_cast<int>(grid_coord.y());
    const int x1 = x0 + 1, y1 = y0 + 1;
    const double dx = grid_coord.x() - x0, dy = grid_coord.y() - y0;
    return (1 - dx) * (1 - dy) * at({x0, y0}) + dx * (1 - dy) * at({x1, y0}) +
        (1 - dx) * dy * at({x0, y1}) + dx * dy * at({x1, y1});
}

// 检查插值点4个角是否完全被方向场覆盖且双向禁止。
// 跳过无方向场的角点（台阶边缘插值时可能只有部分角点在台阶内），
// 只要存在任一台阶角点双向禁止，则该点判定为不可通行。
bool DirectionMap::is_fully_prohibited(const Eigen::Vector2d& grid_coord) const {
    if (!is_valid_coord(grid_coord)) return false;

    const int x0 = static_cast<int>(grid_coord.x());
    const int y0 = static_cast<int>(grid_coord.y());
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    for (const Eigen::Vector2i &sample : {Eigen::Vector2i{x0, y0}, Eigen::Vector2i{x1, y0}, Eigen::Vector2i{x0, y1}, Eigen::Vector2i{x1, y1}}) {
        const Eigen::Vector2d dir = at(sample);
        if (dir.squaredNorm() <= 1e-12) continue; // 该角点无台阶方向场，跳过
        const uint8_t mode = step_mode_at(sample);
        if (extract_up_mode(mode) == STEP_MODE_FORBIDDEN && extract_down_mode(mode) == STEP_MODE_FORBIDDEN) {
            return true; // 台阶区域内双向禁止 → 该点不可通行
        }
    }

    return false;
}

double DirectionMap::prohibited_direction_score(const Eigen::Vector2i& grid_coord, const Eigen::Vector2d& move_dir, double dot_threshold) const {
    return prohibited_direction_score_impl(at(grid_coord), step_mode_at(grid_coord), move_dir, dot_threshold);
}

double DirectionMap::prohibited_direction_score(const Eigen::Vector2d& grid_coord, const Eigen::Vector2d& move_dir, double dot_threshold) const {
    if (!is_valid_coord(grid_coord) || move_dir.squaredNorm() <= 1e-12) return 0.0;

    const int x0 = static_cast<int>(grid_coord.x());
    const int y0 = static_cast<int>(grid_coord.y());
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const double dx = grid_coord.x() - x0;
    const double dy = grid_coord.y() - y0;

    // 标准双线性插值加权和（而非保守 max）
    const double w00 = (1 - dx) * (1 - dy);
    const double w10 = dx * (1 - dy);
    const double w01 = (1 - dx) * dy;
    const double w11 = dx * dy;

    const double total = w00 + w10 + w01 + w11;
    if (total <= 0.0) return 0.0;

    const double score = w00 * prohibited_direction_score(Eigen::Vector2i(x0, y0), move_dir, dot_threshold)
                       + w10 * prohibited_direction_score(Eigen::Vector2i(x1, y0), move_dir, dot_threshold)
                       + w01 * prohibited_direction_score(Eigen::Vector2i(x0, y1), move_dir, dot_threshold)
                       + w11 * prohibited_direction_score(Eigen::Vector2i(x1, y1), move_dir, dot_threshold);
    return std::clamp(score / total, 0.0, 1.0);
}

bool DirectionMap::is_direction_prohibited(const Eigen::Vector2i& grid_coord, const Eigen::Vector2d& move_dir, double dot_threshold) const {
    const Eigen::Vector2d step_dir = at(grid_coord);
    if (step_dir.squaredNorm() <= 1e-12 || move_dir.squaredNorm() <= 1e-12) {
        return false;
    }

    const Eigen::Vector2d normalized_step_dir = step_dir.normalized();
    const Eigen::Vector2d normalized_move_dir = move_dir.normalized();
    const double alignment = normalized_move_dir.dot(normalized_step_dir);
    const double clamped_threshold = std::clamp(dot_threshold, 0.0, 1.0);

    if (alignment > clamped_threshold && extract_up_mode(step_mode_at(grid_coord)) == STEP_MODE_FORBIDDEN) {
        return true;
    }

    if (alignment < -clamped_threshold && extract_down_mode(step_mode_at(grid_coord)) == STEP_MODE_FORBIDDEN) {
        return true;
    }

    return false;
}
}