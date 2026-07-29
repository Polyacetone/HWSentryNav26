#include <nav_executor/path_planner/search/grid_utils.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nav_executor {

bool same_cell(const Eigen::Vector2i& lhs, const Eigen::Vector2i& rhs) {
    return (lhs.array() == rhs.array()).all();
}

bool grid_cell_traversable(
    const CostMap& cost_map,
    const Eigen::Vector2i& cell,
    const int occupied_threshold
) {
    return cost_map.geometry.contains_cell(cell)
        && static_cast<int>(cost_map.raw_cost_at_cell(cell)) < occupied_threshold;
}

bool grid_edge_avoids_corner_cutting(
    const CostMap& cost_map,
    const Eigen::Vector2i& from,
    const Eigen::Vector2i& to,
    const int occupied_threshold
) {
    const Eigen::Vector2i delta = to - from;
    if (std::abs(delta.x()) != 1 || std::abs(delta.y()) != 1) return true;
    return grid_cell_traversable(
        cost_map, {from.x() + delta.x(), from.y()}, occupied_threshold
    ) && grid_cell_traversable(
        cost_map, {from.x(), from.y() + delta.y()}, occupied_threshold
    );
}

std::vector<GridCrossing> trace_grid_crossings(
    const GridGeometry& geometry,
    const Eigen::Vector2d& from_map,
    const Eigen::Vector2d& to_map
) {
    const Eigen::Vector2d from = geometry.map_point_to_boundary_grid(from_map);
    const Eigen::Vector2d to = geometry.map_point_to_boundary_grid(to_map);
    Eigen::Vector2i cell = from.array().floor().cast<int>();
    const Eigen::Vector2i target = to.array().floor().cast<int>();
    std::vector<GridCrossing> crossings;
    if (same_cell(cell, target)) return crossings;

    const Eigen::Vector2d delta = to - from;
    const int step_x = delta.x() > 0.0 ? 1 : (delta.x() < 0.0 ? -1 : 0);
    const int step_y = delta.y() > 0.0 ? 1 : (delta.y() < 0.0 ? -1 : 0);
    const double infinity = std::numeric_limits<double>::infinity();
    const double t_delta_x = step_x == 0 ? infinity : std::abs(1.0 / delta.x());
    const double t_delta_y = step_y == 0 ? infinity : std::abs(1.0 / delta.y());
    double t_max_x = step_x == 0 ? infinity : (
        (step_x > 0 ? std::floor(from.x()) + 1.0 : std::floor(from.x())) - from.x()
    ) / delta.x();
    double t_max_y = step_y == 0 ? infinity : (
        (step_y > 0 ? std::floor(from.y()) + 1.0 : std::floor(from.y())) - from.y()
    ) / delta.y();
    t_max_x = std::max(t_max_x, 0.0);
    t_max_y = std::max(t_max_y, 0.0);

    while (!same_cell(cell, target)) {
        const Eigen::Vector2i previous = cell;
        double fraction = 0.0;
        if (std::abs(t_max_x - t_max_y) <= 1e-12) {
            fraction = t_max_x;
            cell.x() += step_x;
            cell.y() += step_y;
            t_max_x += t_delta_x;
            t_max_y += t_delta_y;
        } else if (t_max_x < t_max_y) {
            fraction = t_max_x;
            cell.x() += step_x;
            t_max_x += t_delta_x;
        } else {
            fraction = t_max_y;
            cell.y() += step_y;
            t_max_y += t_delta_y;
        }
        crossings.push_back({previous, cell, std::clamp(fraction, 0.0, 1.0)});
        if (crossings.size() > static_cast<size_t>(geometry.width() + geometry.height())) {
            break;
        }
    }
    return crossings;
}

} // namespace nav_executor
