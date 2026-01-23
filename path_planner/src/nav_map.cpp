#include <path_planner/nav_map.hpp>

#include <algorithm>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

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
    width(occupancy_grid.info.width),
    height(occupancy_grid.info.height),
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
    if (is_valid_coord(grid_coord)) return data[grid_coord.y() * width + grid_coord.x()];
    else return 255;
}

double CostMap::interpolate(const Eigen::Vector2d& grid_coord) const {
    if (!is_valid_coord(grid_coord)) return 255;
    const int x0 = grid_coord.x(), y0 = grid_coord.y();
    const int x1 = x0 + 1, y1 = y0 + 1;
    const double dx = grid_coord.x() - x0, dy = grid_coord.y() - y0;
    return (1 - dx) * (1 - dy) * at({x0, y0}) + dx * (1 - dy) * at({x1, y0}) +
        (1 - dx) * dy * at({x0, y1}) + dx * dy * at({x1, y1});
}

Eigen::Vector2d CostMap::gradient(const Eigen::Vector2d& grid_coord) const {
    constexpr int samples = 2;
    const int x = grid_coord.x(), y = grid_coord.y();
    Eigen::Vector2d sum_grad(0, 0);
    for (int i = 1; i <= samples; i++) {
        sum_grad.x() += (at({x + i, y}) - at({x - i, y})) / (i * 2.0);
        sum_grad.y() += (at({x, y + i}) - at({x, y - i})) / (i * 2.0);
    }
    return sum_grad / samples;
}

ESDFMap::ESDFMap(
    const int width,
    const int height,
    const double resolution,
    const double origin_x,
    const double origin_y,
    std::vector<float> signed_distance_m
): width(width),
   height(height),
   resolution(resolution),
   origin_x(origin_x),
   origin_y(origin_y),
   signed_distance_m(std::move(signed_distance_m)) {
    if (static_cast<int>(this->signed_distance_m.size()) != width * height) {
        throw std::runtime_error("ESDFMap signed_distance_m size mismatch");
    }
}

ESDFMap ESDFMap::from_cost_map(const CostMap& cost_map, const int obstacle_threshold) {
    // OpenCV distanceTransform: 计算每个非零像素到最近零像素的距离（像素单位）
    // 我们将障碍物视为 0，自由空间视为 255，得到“到障碍物的距离”
    cv::Mat free_mask(cost_map.height, cost_map.width, CV_8U);
    cv::Mat occ_mask(cost_map.height, cost_map.width, CV_8U);

    for (int y = 0; y < cost_map.height; y++) {
        for (int x = 0; x < cost_map.width; x++) {
            const uint8_t v = cost_map.data[y * cost_map.width + x];
            const bool is_obstacle = v > obstacle_threshold;
            free_mask.at<uint8_t>(y, x) = is_obstacle ? 0 : 255;
            occ_mask.at<uint8_t>(y, x) = is_obstacle ? 255 : 0;
        }
    }

    cv::Mat dist_out_px, dist_in_px;
    cv::distanceTransform(free_mask, dist_out_px, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    cv::distanceTransform(occ_mask, dist_in_px, cv::DIST_L2, cv::DIST_MASK_PRECISE);

    std::vector<float> sdf_m;
    sdf_m.resize(static_cast<size_t>(cost_map.width) * static_cast<size_t>(cost_map.height));
    for (int y = 0; y < cost_map.height; y++) {
        for (int x = 0; x < cost_map.width; x++) {
            const float dout = dist_out_px.at<float>(y, x);
            const float din = dist_in_px.at<float>(y, x);
            const float sdf_px = dout - din; // outside: +, inside obstacle: -
            sdf_m[static_cast<size_t>(y) * cost_map.width + x] = sdf_px * static_cast<float>(cost_map.resolution);
        }
    }

    return ESDFMap(cost_map.width, cost_map.height, cost_map.resolution, cost_map.origin_x, cost_map.origin_y, std::move(sdf_m));
}

Eigen::Vector2d ESDFMap::map_coord_to_grid(const Eigen::Vector2d& map_coord) const {
    return {(map_coord.x() - origin_x) / resolution, (map_coord.y() - origin_y) / resolution};
}

Eigen::Vector2d ESDFMap::grid_coord_to_map(const Eigen::Vector2d& grid_coord) const {
    return {grid_coord.x() * resolution + origin_x, grid_coord.y() * resolution + origin_y};
}

bool ESDFMap::is_valid_coord(const Eigen::Vector2i& grid_coord) const {
    return grid_coord.x() >= 0 && grid_coord.x() < width && grid_coord.y() >= 0 && grid_coord.y() < height;
}

bool ESDFMap::is_valid_coord(const Eigen::Vector2d& grid_coord) const {
    // bilinear 插值需要 (x0,y0),(x1,y1) 都有效
    return grid_coord.x() >= 0 && grid_coord.x() + 1 <= width && grid_coord.y() >= 0 && grid_coord.y() + 1 <= height;
}

float ESDFMap::at(const Eigen::Vector2i& grid_coord) const {
    if (!is_valid_coord(grid_coord)) {
        // 越界按强碰撞处理：返回很大的负距离
        return -1e3f;
    }
    return signed_distance_m[static_cast<size_t>(grid_coord.y()) * width + grid_coord.x()];
}

double ESDFMap::interpolate(const Eigen::Vector2d& grid_coord) const {
    if (!is_valid_coord(grid_coord)) return static_cast<double>(-1e3);
    const int x0 = static_cast<int>(grid_coord.x());
    const int y0 = static_cast<int>(grid_coord.y());
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const double dx = grid_coord.x() - x0;
    const double dy = grid_coord.y() - y0;
    const double f00 = at({x0, y0});
    const double f10 = at({x1, y0});
    const double f01 = at({x0, y1});
    const double f11 = at({x1, y1});
    return (1 - dx) * (1 - dy) * f00 + dx * (1 - dy) * f10 + (1 - dx) * dy * f01 + dx * dy * f11;
}

Eigen::Vector2d ESDFMap::gradient(const Eigen::Vector2d& grid_coord) const {
    if (!is_valid_coord(grid_coord)) return {0.0, 0.0};
    const int x0 = static_cast<int>(grid_coord.x());
    const int y0 = static_cast<int>(grid_coord.y());
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const double dx = grid_coord.x() - x0;
    const double dy = grid_coord.y() - y0;

    const double f00 = at({x0, y0});
    const double f10 = at({x1, y0});
    const double f01 = at({x0, y1});
    const double f11 = at({x1, y1});

    // bilinear 插值对 (x,y) 的解析梯度（单位：m / cell）
    const double dfdx = (1 - dy) * (f10 - f00) + dy * (f11 - f01);
    const double dfdy = (1 - dx) * (f01 - f00) + dx * (f11 - f10);
    return {dfdx, dfdy};
}

std::vector<Eigen::Vector2d> convert_direction_map(const cv::Mat& mat) {
    if (mat.type() != CV_8UC2) {
        throw std::runtime_error("Direction map must be of type CV_8UC2");
    }
    std::vector<Eigen::Vector2d> vec;
    vec.reserve(mat.cols * mat.rows);
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            cv::Vec2b val = mat.at<cv::Vec2b>(y, x);
            if ((val[0] == 0 && val[1] == 0) || (val[0] == 128 && val[1] == 128)) {
                vec.emplace_back(0, 0);
            } else {
                Eigen::Vector2d dir(val[0] - 128, val[1] - 128);
                vec.emplace_back(dir.normalized());
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
    if (is_valid_coord(grid_coord)) return data[grid_coord.y() * width + grid_coord.x()];
    else return {0, 0};
}

Eigen::Vector2d DirectionMap::interpolate(const Eigen::Vector2d& grid_coord) const {
    if (!is_valid_coord(grid_coord)) return {0, 0};
    const int x0 = grid_coord.x(), y0 = grid_coord.y();
    const int x1 = x0 + 1, y1 = y0 + 1;
    const double dx = grid_coord.x() - x0, dy = grid_coord.y() - y0;
    return (1 - dx) * (1 - dy) * at({x0, y0}) + dx * (1 - dy) * at({x1, y0}) +
        (1 - dx) * dy * at({x0, y1}) + dx * dy * at({x1, y1});
}
}