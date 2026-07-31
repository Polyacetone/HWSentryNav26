#include <nav_executor/path_planner/search/spatial_grid_astar.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>

#include <nav_executor/path_planner/search/grid_utils.hpp>

namespace nav_executor {

namespace {

const std::array<Eigen::Vector2i, 8> NEIGHBORS {{
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
}};

size_t index_of(const Eigen::Vector2i& cell, const int width) {
    return static_cast<size_t>(cell.y()) * static_cast<size_t>(width)
        + static_cast<size_t>(cell.x());
}

double octile_distance(
    const Eigen::Vector2i& from,
    const Eigen::Vector2i& to,
    const double resolution
) {
    const Eigen::Vector2i delta = (to - from).cwiseAbs();
    const int diagonal = std::min(delta.x(), delta.y());
    const int straight = std::max(delta.x(), delta.y()) - diagonal;
    return resolution * (std::sqrt(2.0) * static_cast<double>(diagonal)
        + static_cast<double>(straight));
}

struct OpenEntry {
    double f = 0.0;
    double g = 0.0;
    Eigen::Vector2i cell = Eigen::Vector2i::Zero();

    bool operator>(const OpenEntry& other) const {
        if (f != other.f) return f > other.f;
        if (g != other.g) return g > other.g;
        if (cell.y() != other.cell.y()) return cell.y() > other.cell.y();
        return cell.x() > other.cell.x();
    }
};

bool extract_passages(
    SpatialRoute& route,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const double detect_dot_threshold,
    std::string& error
) {
    size_t cursor = 0;
    while (cursor < route.raw_path.size()) {
        if (!direction_map.is_terrain_body_cell(route.raw_path[cursor])) {
            ++cursor;
            continue;
        }
        const size_t first = cursor;
        while (cursor + 1 < route.raw_path.size()
            && direction_map.is_terrain_body_cell(route.raw_path[cursor + 1])) {
            ++cursor;
        }
        const size_t last = cursor;
        if (first == 0 || last + 1 >= route.raw_path.size()) {
            error = "spatial path starts or ends inside directional terrain";
            return false;
        }
        const uint8_t label = direction_map.terrain_label_at_cell(route.raw_path[first]);
        for (size_t i = first; i <= last; ++i) {
            if (direction_map.terrain_label_at_cell(route.raw_path[i]) != label) {
                error = "spatial path changes terrain label without a flat cell";
                return false;
            }
        }
        bool entry_going_up = true;
        bool exit_going_up = true;
        if (!directed_terrain_edge_allowed(
                direction_map, terrain_constraints,
                route.raw_path[first - 1], route.raw_path[first],
                detect_dot_threshold, &entry_going_up
            ) || !directed_terrain_edge_allowed(
                direction_map, terrain_constraints,
                route.raw_path[last], route.raw_path[last + 1],
                detect_dot_threshold, &exit_going_up
            ) || entry_going_up != exit_going_up) {
            error = "spatial path contains an inconsistent terrain passage";
            return false;
        }
        route.passages.push_back({
            .label = label,
            .going_up = entry_going_up,
            .first_body_index = first,
            .last_body_index = last,
            .entry_flat_index = first - 1,
            .exit_flat_index = last + 1,
        });
        ++cursor;
    }
    return true;
}

} // anonymous namespace

SpatialGridAstar::Result SpatialGridAstar::search(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const Eigen::Vector2i& start,
    const Eigen::Vector2i& goal,
    const int occupied_threshold,
    const double detect_dot_threshold
) const {
    Result result;
    if (!cost_map.geometry.same_geometry(direction_map.geometry)
        || params_.obstacle_weight < 0.0 || params_.max_expansions <= 0
        || occupied_threshold <= 0
        || !grid_cell_traversable(cost_map, start, occupied_threshold)
        || !grid_cell_traversable(cost_map, goal, occupied_threshold)) {
        result.error = "spatial grid A* configuration or endpoints are invalid";
        return result;
    }

    const int width = cost_map.geometry.width();
    const size_t cell_count = static_cast<size_t>(width)
        * static_cast<size_t>(cost_map.geometry.height());
    std::vector<double> g_score(cell_count, std::numeric_limits<double>::infinity());
    std::vector<Eigen::Vector2i> parent(cell_count, {-1, -1});
    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<>> open;
    g_score[index_of(start, width)] = 0.0;
    open.push({octile_distance(start, goal, cost_map.geometry.resolution()), 0.0, start});

    while (!open.empty()) {
        result.route.open_peak = std::max(result.route.open_peak, open.size());
        const OpenEntry current = open.top();
        open.pop();
        if (current.g > g_score[index_of(current.cell, width)] + 1e-12) continue;
        if (same_cell(current.cell, goal)) {
            for (Eigen::Vector2i cell = goal;; cell = parent[index_of(cell, width)]) {
                result.route.raw_path.push_back(cell);
                if (same_cell(cell, start)) break;
                if (parent[index_of(cell, width)].x() < 0) {
                    result.error = "spatial grid A* parent chain is incomplete";
                    return result;
                }
            }
            std::reverse(result.route.raw_path.begin(), result.route.raw_path.end());
            if (!extract_passages(
                    result.route, direction_map, terrain_constraints,
                    detect_dot_threshold, result.error
                )) {
                return result;
            }
            result.success = true;
            return result;
        }
        if (result.route.expansions >= params_.max_expansions) {
            result.error = "spatial grid A* expansion limit reached";
            return result;
        }
        ++result.route.expansions;

        for (const Eigen::Vector2i& delta : NEIGHBORS) {
            const Eigen::Vector2i next = current.cell + delta;
            if (!grid_cell_traversable(cost_map, next, occupied_threshold)
                || !grid_edge_avoids_corner_cutting(
                    cost_map, current.cell, next, occupied_threshold
                ) || !directed_terrain_edge_allowed(
                    direction_map, terrain_constraints,
                    current.cell, next, detect_dot_threshold
                )) {
                continue;
            }
            const double length = delta.cast<double>().norm()
                * cost_map.geometry.resolution();
            const double normalized_cost = static_cast<double>(std::max(
                cost_map.raw_cost_at_cell(current.cell),
                cost_map.raw_cost_at_cell(next)
            )) / static_cast<double>(occupied_threshold);
            const double candidate = current.g + length
                * (1.0 + params_.obstacle_weight * normalized_cost);
            const size_t next_index = index_of(next, width);
            if (candidate + 1e-12 >= g_score[next_index]) continue;
            g_score[next_index] = candidate;
            parent[next_index] = current.cell;
            open.push({
                candidate + octile_distance(
                    next, goal, cost_map.geometry.resolution()
                ),
                candidate,
                next,
            });
        }
    }
    result.error = "spatial grid A* found no path through hard planner obstacles";
    return result;
}

} // namespace nav_executor
