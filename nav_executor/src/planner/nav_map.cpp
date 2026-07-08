#include <nav_executor/planner/nav_map.hpp>
#include <nav_executor/common/chassis_defs.hpp>

#include <algorithm>
#include <numbers>
#include <stdexcept>

namespace nav_executor {

// ════════════════════════════════════════════════════════════════
//  CostMap
// ════════════════════════════════════════════════════════════════

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
        return data[static_cast<size_t>(grid_coord.y()) * static_cast<size_t>(width) + static_cast<size_t>(grid_coord.x())];
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

// ════════════════════════════════════════════════════════════════
//  DirectionMap
// ════════════════════════════════════════════════════════════════

namespace {

double prohibited_direction_score_impl(const Eigen::Vector2d& step_dir, const TerrainRule& rule, const Eigen::Vector2d& move_dir, double dot_threshold) {
    if (step_dir.squaredNorm() <= 1e-12 || move_dir.squaredNorm() <= 1e-12) {
        return 0.0;
    }

    const Eigen::Vector2d normalized_step_dir = step_dir.normalized();
    const Eigen::Vector2d normalized_move_dir = move_dir.normalized();
    const double alignment = normalized_move_dir.dot(normalized_step_dir);
    const double clamped_threshold = std::clamp(dot_threshold, 0.0, 1.0);
    const double denom = std::max(1.0 - clamped_threshold, 1e-6);

    if (alignment > clamped_threshold && !rule.forward_allowed) {
        return std::clamp((alignment - clamped_threshold) / denom, 0.0, 1.0);
    }

    if (alignment < -clamped_threshold && !rule.backward_allowed) {
        return std::clamp((-alignment - clamped_threshold) / denom, 0.0, 1.0);
    }

    return 0.0;
}

} // anonymous namespace

/*static*/ std::pair<std::vector<Eigen::Vector2d>, std::vector<uint8_t>>
DirectionMap::decode_mat(const cv::Mat& mat) {
    if (mat.type() != CV_8UC3) {
        throw std::runtime_error("Direction map must be of type CV_8UC3");
    }

    std::vector<Eigen::Vector2d> dir;
    std::vector<uint8_t> terrain;
    dir.reserve(static_cast<size_t>(mat.cols) * static_cast<size_t>(mat.rows));
    terrain.reserve(static_cast<size_t>(mat.cols) * static_cast<size_t>(mat.rows));
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            const cv::Vec3b val = mat.at<cv::Vec3b>(y, x);
            if (val[1] == 0) {
                dir.emplace_back(0.0, 0.0);
            } else {
                const double angle = static_cast<double>(val[0]) / 255.0 * 2.0 * std::numbers::pi;
                const double mag = static_cast<double>(val[1]) / 255.0;
                dir.emplace_back(std::cos(angle) * mag, std::sin(angle) * mag);
            }
            terrain.push_back(val[2]);
        }
    }
    return {std::move(dir), std::move(terrain)};
}

DirectionMap::DirectionMap(
    const cv::Mat& direction_map, double resolution, double origin_x, double origin_y,
    const TerrainProfiles& profiles, const TerrainRuleTable& rules):
    DirectionMap(
        direction_map.cols, direction_map.rows, resolution, origin_x, origin_y,
        decode_mat(direction_map), profiles, rules
    ) {}

DirectionMap::DirectionMap(
    int width, int height, double resolution, double origin_x, double origin_y,
    std::pair<std::vector<Eigen::Vector2d>, std::vector<uint8_t>> decoded,
    const TerrainProfiles& profiles, const TerrainRuleTable& rules):
    DirectionMap(
        width, height, resolution, origin_x, origin_y,
        std::move(decoded.first), std::move(decoded.second), profiles, rules
    ) {}

DirectionMap::DirectionMap(
    int width, int height, double resolution, double origin_x, double origin_y,
    std::vector<Eigen::Vector2d> dir_data, std::vector<uint8_t> terrain_data,
    const TerrainProfiles& profiles, const TerrainRuleTable& rules):
    width(width),
    height(height),
    resolution(resolution),
    origin_x(origin_x),
    origin_y(origin_y),
    data(std::move(dir_data)),
    terrain(terrain_data.empty() ? std::vector<uint8_t>(static_cast<size_t>(width) * static_cast<size_t>(height), static_cast<uint8_t>(TerrainType::FLAT)) : std::move(terrain_data)),
    profiles_(profiles),
    rules_(rules) {
    if (static_cast<int>(this->data.size()) != width * height) {
        throw std::runtime_error("DirectionMap data size does not match width*height");
    }
    if (static_cast<int>(this->terrain.size()) != width * height) {
        throw std::runtime_error("DirectionMap terrain size does not match width*height");
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
        return data[static_cast<size_t>(grid_coord.y()) * static_cast<size_t>(width) + static_cast<size_t>(grid_coord.x())];
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

DirectionMap::DirectionSample DirectionMap::interpolate_with_gradient(const Eigen::Vector2d& grid_coord) const {
    if (!is_valid_coord(grid_coord)) return {{0, 0}, Eigen::Matrix2d::Zero()};
    const int x0 = static_cast<int>(grid_coord.x());
    const int y0 = static_cast<int>(grid_coord.y());
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const double dx = grid_coord.x() - x0;
    const double dy = grid_coord.y() - y0;

    const Eigen::Vector2d d00 = at({x0, y0}), d10 = at({x1, y0});
    const Eigen::Vector2d d01 = at({x0, y1}), d11 = at({x1, y1});

    DirectionSample result;
    result.value = (1 - dx) * (1 - dy) * d00 + dx * (1 - dy) * d10 + (1 - dx) * dy * d01 + dx * dy * d11;
    // d(value)/dx = (1-dy)*(d10-d00) + dy*(d11-d01)
    result.gradient.col(0) = (1 - dy) * (d10 - d00) + dy * (d11 - d01);
    // d(value)/dy = (1-dx)*(d01-d00) + dx*(d11-d10)
    result.gradient.col(1) = (1 - dx) * (d01 - d00) + dx * (d11 - d10);
    return result;
}

uint8_t DirectionMap::terrain_at(const Eigen::Vector2i& grid_coord) const {
    if (is_valid_coord(grid_coord)) {
        return terrain[static_cast<size_t>(grid_coord.y()) * static_cast<size_t>(width) + static_cast<size_t>(grid_coord.x())];
    }
    return static_cast<uint8_t>(TerrainType::OBSTACLE);
}

uint8_t DirectionMap::terrain_at(const Eigen::Vector2d& grid_coord) const {
    const Eigen::Array2i floored = grid_coord.array().floor().cast<int>();
    return terrain_at(Eigen::Vector2i(floored.x(), floored.y()));
}

bool DirectionMap::has_blocked_corner(const Eigen::Vector2d& grid_coord) const {
    if (!is_valid_coord(grid_coord)) return false;

    const int x0 = static_cast<int>(grid_coord.x());
    const int y0 = static_cast<int>(grid_coord.y());
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    for (const Eigen::Vector2i& sample : {Eigen::Vector2i{x0, y0}, Eigen::Vector2i{x1, y0}, Eigen::Vector2i{x0, y1}, Eigen::Vector2i{x1, y1}}) {
        const Eigen::Vector2d dir = at(sample);
        if (dir.squaredNorm() <= 1e-12) continue;
        const TerrainRule& rule = rules_[terrain_at(sample)];
        if (!rule.forward_allowed && !rule.backward_allowed) {
            return true;
        }
    }

    return false;
}

double DirectionMap::prohibited_direction_score(const Eigen::Vector2i& grid_coord, const Eigen::Vector2d& move_dir, double dot_threshold) const {
    return prohibited_direction_score_impl(at(grid_coord), rules_[terrain_at(grid_coord)], move_dir, dot_threshold);
}

double DirectionMap::prohibited_direction_score(const Eigen::Vector2d& grid_coord, const Eigen::Vector2d& move_dir, double dot_threshold) const {
    if (!is_valid_coord(grid_coord) || move_dir.squaredNorm() <= 1e-12) return 0.0;

    const int x0 = static_cast<int>(grid_coord.x());
    const int y0 = static_cast<int>(grid_coord.y());
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const double dx = grid_coord.x() - x0;
    const double dy = grid_coord.y() - y0;

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
    return prohibited_direction_score_impl(at(grid_coord), rules_[terrain_at(grid_coord)], move_dir, dot_threshold) > 0.0;
}

const TerrainStepRule& DirectionMap::rule_for_label(const uint8_t label, const bool is_up) const {
    if (label <= static_cast<uint8_t>(TerrainType::OBSTACLE)) {
        static const TerrainStepRule kFlatRule = {
            .chassis_mode = chassis_mode::NORMAL,
            .capability = CapabilityLevel::LOW,
            .speed = {0.0, 0.0},
        };
        return kFlatRule;
    }
    const auto& lr = profiles_.directional_labels[label - 2];
    return is_up ? lr.up : lr.down;
}

} // namespace nav_executor
