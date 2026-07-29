#include <nav_executor/path_planner/search/time_to_goal_heuristic.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <queue>

#include <nav_executor/path_planner/search/grid_utils.hpp>

namespace nav_executor {

namespace {

const std::array<Eigen::Vector2i, 8> NEIGHBORS {{
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
}};

struct OpenEntry {
    double time = 0.0;
    Eigen::Vector2i cell = Eigen::Vector2i::Zero();

    bool operator>(const OpenEntry& other) const { return time > other.time; }
};

double terrain_speed_cap(
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& constraints,
    const Eigen::Vector2i& cell,
    const double velocity_max
) {
    if (!direction_map.is_terrain_body_cell(cell)) return velocity_max;
    const uint8_t label = direction_map.terrain_label_at_cell(cell);
    double cap = 0.0;
    for (const bool going_up : {false, true}) {
        if (const TraversalMode* mode = constraints.selected_mode(label, going_up)) {
            cap = std::max(cap, mode->velocity_window.max);
        }
    }
    return cap > 0.0 ? std::min(cap, velocity_max) : velocity_max;
}

} // anonymous namespace

void TimeToGoalHeuristic::build(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const Eigen::Vector2i& goal_cell,
    const int occupied_threshold,
    const double velocity_max
) {
    geometry_ = cost_map.geometry;
    time_.assign(
        static_cast<size_t>(cost_map.geometry.width())
            * static_cast<size_t>(cost_map.geometry.height()),
        UNREACHABLE
    );
    if (!cost_map.geometry.same_geometry(direction_map.geometry)
        || !grid_cell_traversable(cost_map, goal_cell, occupied_threshold)
        || !std::isfinite(velocity_max) || velocity_max <= 0.0) {
        return;
    }

    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<>> open;
    time_[index(goal_cell)] = 0.0;
    open.push({0.0, goal_cell});
    while (!open.empty()) {
        const OpenEntry current = open.top();
        open.pop();
        if (current.time > time_[index(current.cell)] + 1e-12) continue;
        for (const Eigen::Vector2i& delta : NEIGHBORS) {
            const Eigen::Vector2i predecessor = current.cell - delta;
            if (!grid_cell_traversable(cost_map, predecessor, occupied_threshold)
                || !grid_edge_avoids_corner_cutting(
                    cost_map, predecessor, current.cell, occupied_threshold
                )) {
                continue;
            }
            const double speed_cap = terrain_speed_cap(
                direction_map, terrain_constraints, current.cell, velocity_max
            );
            const double length = delta.cast<double>().norm()
                * cost_map.geometry.resolution();
            const double candidate = current.time + length / speed_cap;
            const size_t predecessor_index = index(predecessor);
            if (candidate + 1e-12 >= time_[predecessor_index]) continue;
            time_[predecessor_index] = candidate;
            open.push({candidate, predecessor});
        }
    }
}

size_t TimeToGoalHeuristic::index(const Eigen::Vector2i& cell) const {
    return static_cast<size_t>(cell.y())
        * static_cast<size_t>(geometry_->width())
        + static_cast<size_t>(cell.x());
}

double TimeToGoalHeuristic::at_cell(const Eigen::Vector2i& cell) const {
    if (!geometry_ || !geometry_->contains_cell(cell)) return UNREACHABLE;
    return time_[index(cell)];
}

double TimeToGoalHeuristic::at_map(const Eigen::Vector2d& point_map) const {
    if (!geometry_) return UNREACHABLE;
    const auto cell = geometry_->containing_cell(point_map);
    return cell ? at_cell(*cell) : UNREACHABLE;
}

} // namespace nav_executor
