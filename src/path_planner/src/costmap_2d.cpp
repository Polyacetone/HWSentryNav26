#include <path_planner/costmap_2d.hpp>

namespace path_planner {
Costmap2D::Costmap2D(const nav_msgs::msg::OccupancyGrid& occupancy_grid):
    width_(occupancy_grid.info.width),
    height_(occupancy_grid.info.height),
    resolution_(occupancy_grid.info.resolution),
    origin_x_(occupancy_grid.info.origin.position.x),
    origin_y_(occupancy_grid.info.origin.position.y),
    data_(occupancy_grid.data) {
    assert(data_.size() == size());
}

unsigned Costmap2D::width() const { return width_; }
unsigned Costmap2D::height() const { return height_; }
unsigned Costmap2D::size() const { return width_ * height_; }

Eigen::Vector2d Costmap2D::map_coord_to_grid(const Eigen::Vector2d& map_coord) const {
    return {(map_coord.x() - origin_x_) / resolution_, (map_coord.y() - origin_y_) / resolution_};
}

Eigen::Vector2d Costmap2D::grid_coord_to_map(const Eigen::Vector2d& grid_coord) const {
    return {grid_coord.x() * resolution_ + origin_x_, grid_coord.y() * resolution_ + origin_y_};
}

bool Costmap2D::is_valid_coord(const Eigen::Vector2i& grid_coord) const {
    return grid_coord.x() >= 0 && grid_coord.x() < width_ &&
        grid_coord.y() >= 0 && grid_coord.y() < height_;
}

bool Costmap2D::is_valid_coord(const Eigen::Vector2d& grid_coord) const {
    return grid_coord.x() >= 0 && grid_coord.x() + 1 <= width_ &&
        grid_coord.y() >= 0 && grid_coord.y() + 1 <= height_;
}

int8_t Costmap2D::at(const Eigen::Vector2i& grid_coord) const {
    if (is_valid_coord(grid_coord)) return data_[grid_coord.y() * width_ + grid_coord.x()];
    else return 100;
}

double Costmap2D::interpolate(const Eigen::Vector2d& grid_coord) const {
    if (!is_valid_coord(grid_coord)) return 100;
    const int x0 = grid_coord.x(), y0 = grid_coord.y();
    const int x1 = x0 + 1, y1 = y0 + 1;
    const double dx = grid_coord.x() - x0, dy = grid_coord.y() - y0;
    return (1 - dx) * (1 - dy) * at({x0, y0}) + dx * (1 - dy) * at({x1, y0}) +
        (1 - dx) * dy * at({x0, y1}) + dx * dy * at({x1, y1});
}

Eigen::Vector2d Costmap2D::gradient(const Eigen::Vector2d& grid_coord) const {
    constexpr int samples = 3;
    const int x = grid_coord.x(), y = grid_coord.y();
    Eigen::Vector2d sum_grad(0, 0);
    for (int i = 1; i <= samples; i++) {
        sum_grad.x() += (at({x + i, y}) - at({x - i, y})) / (i * 2.0);
        sum_grad.y() += (at({x, y + i}) - at({x, y - i})) / (i * 2.0);
    }
    return sum_grad / samples;
}
}