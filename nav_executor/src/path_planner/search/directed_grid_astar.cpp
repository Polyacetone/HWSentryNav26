#include <nav_executor/path_planner/search/directed_grid_astar.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>

namespace nav_executor {

namespace {

const std::array<Eigen::Vector2i, 8> NEIGHBORS {{
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
}};

struct OpenEntry {
    double f = 0.0;
    double g = 0.0;
    Eigen::Vector2i cell = Eigen::Vector2i::Zero();
    bool operator>(const OpenEntry& other) const { return f > other.f; }
};

size_t index_of(const Eigen::Vector2i& cell, const int width) {
    return static_cast<size_t>(cell.y()) * static_cast<size_t>(width)
        + static_cast<size_t>(cell.x());
}

double heuristic(
    const Eigen::Vector2i& from,
    const Eigen::Vector2i& to,
    const double resolution
) {
    const Eigen::Vector2i delta = (to - from).cwiseAbs();
    const int diagonal = std::min(delta.x(), delta.y());
    const int straight = std::max(delta.x(), delta.y()) - diagonal;
    return resolution * (
        std::sqrt(2.0) * static_cast<double>(diagonal)
        + static_cast<double>(straight)
    );
}

template <typename EdgePredicate>
std::expected<DirectedGridAstar::Result, std::string> run_astar(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const Eigen::Vector2i& start,
    const Eigen::Vector2i& goal,
    const int occupied_threshold,
    const DirectedGridAstar::Params& params,
    const EdgePredicate& edge_allowed
) {
    if (!grid_cell_traversable(cost_map, start, occupied_threshold)
        || !grid_cell_traversable(cost_map, goal, occupied_threshold)) {
        return std::unexpected("grid A* start or goal is not traversable");
    }
    const int width = cost_map.geometry.width();
    const size_t cell_count = static_cast<size_t>(width)
        * static_cast<size_t>(cost_map.geometry.height());
    std::vector<double> g_score(cell_count, std::numeric_limits<double>::infinity());
    std::vector<Eigen::Vector2i> parent(cell_count, Eigen::Vector2i(-1, -1));
    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<>> open;
    g_score[index_of(start, width)] = 0.0;
    open.push({heuristic(start, goal, cost_map.geometry.resolution()), 0.0, start});

    DirectedGridAstar::Result result;
    while (!open.empty()) {
        result.open_peak = std::max(result.open_peak, open.size());
        const OpenEntry current = open.top();
        open.pop();
        const size_t current_index = index_of(current.cell, width);
        if (current.g > g_score[current_index] + 1e-12) continue;
        if (same_cell(current.cell, goal)) {
            for (Eigen::Vector2i cell = goal; cell.x() >= 0;
                 cell = parent[index_of(cell, width)]) {
                result.raw_path.push_back(cell);
                if (same_cell(cell, start)) break;
            }
            if (result.raw_path.empty() || !same_cell(result.raw_path.back(), start)) {
                return std::unexpected("grid A* parent chain is incomplete");
            }
            std::reverse(result.raw_path.begin(), result.raw_path.end());
            return result;
        }
        if (result.expansions >= params.max_expansions) {
            return std::unexpected("grid A* expansion limit reached");
        }
        ++result.expansions;

        for (const Eigen::Vector2i& delta : NEIGHBORS) {
            const Eigen::Vector2i next = current.cell + delta;
            if (!grid_cell_traversable(cost_map, next, occupied_threshold)
                || !grid_edge_avoids_corner_cutting(
                    cost_map, current.cell, next, occupied_threshold
                ) || !edge_allowed(current.cell, next)) {
                continue;
            }
            const double length = delta.cast<double>().norm() * cost_map.geometry.resolution();
            const double occupancy = static_cast<double>(cost_map.raw_cost_at_cell(next)) / 255.0;
            const Eigen::Vector2d raw_direction = direction_map.raw_direction_at_cell(next);
            double alignment_cost = 0.0;
            if (raw_direction.squaredNorm() > 1e-12) {
                alignment_cost = params.alignment_weight * length
                    * (1.0 - std::abs(
                        delta.cast<double>().normalized().dot(raw_direction.normalized())
                    ));
            }
            const double edge_cost = length
                * (1.0 + params.obstacle_weight * occupancy)
                + alignment_cost
                + params.terrain_proximity_weight * length * raw_direction.norm();
            const double candidate = current.g + edge_cost;
            const size_t next_index = index_of(next, width);
            if (candidate + 1e-12 >= g_score[next_index]) continue;
            g_score[next_index] = candidate;
            parent[next_index] = current.cell;
            open.push({
                candidate + heuristic(next, goal, cost_map.geometry.resolution()),
                candidate,
                next,
            });
        }
    }
    return std::unexpected("grid A* found no directed path");
}

} // anonymous namespace

bool DirectedGridAstar::TerrainReachability::contains(
    const Eigen::Vector2i& cell
) const {
    if (cell.x() < 0 || cell.x() >= width || width <= 0 || cell.y() < 0) return false;
    const size_t index = static_cast<size_t>(cell.y()) * static_cast<size_t>(width)
        + static_cast<size_t>(cell.x());
    return index < can_reach_exit.size() && can_reach_exit[index] != 0;
}

std::expected<DirectedGridAstar::Result, std::string> DirectedGridAstar::search_global(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const Eigen::Vector2i& start,
    const Eigen::Vector2i& goal,
    const int occupied_threshold,
    const double detect_dot_threshold
) const {
    return run_astar(
        cost_map,
        direction_map,
        start,
        goal,
        occupied_threshold,
        params_,
        [&](const Eigen::Vector2i& from, const Eigen::Vector2i& to) {
            return directed_terrain_edge_allowed(
                direction_map,
                terrain_constraints,
                from,
                to,
                detect_dot_threshold
            );
        }
    );
}

std::expected<DirectedGridAstar::Result, std::string>
DirectedGridAstar::search_terrain_region(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const TerrainRegionIndex& regions,
    const TerrainRegionId region,
    const StepDirection direction,
    const Eigen::Vector2i& start,
    const Eigen::Vector2i& goal,
    const int occupied_threshold,
    const double detect_dot_threshold
) const {
    if (regions.region_at(start) != region || regions.region_at(goal) != region) {
        return std::unexpected("terrain-local A* endpoints do not belong to the requested region");
    }
    return run_astar(
        cost_map,
        direction_map,
        start,
        goal,
        occupied_threshold,
        params_,
        [&](const Eigen::Vector2i& from, const Eigen::Vector2i& to) {
            StepDirection edge_direction;
            return regions.region_at(to) == region
                && directed_terrain_edge_allowed(
                    direction_map,
                    terrain_constraints,
                    from,
                    to,
                    detect_dot_threshold,
                    &edge_direction
                ) && edge_direction == direction;
        }
    );
}

std::expected<DirectedGridAstar::TerrainReachability, std::string>
DirectedGridAstar::build_terrain_reachability(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const TerrainRegionIndex& regions,
    const TerrainRegionId region,
    const StepDirection direction,
    const Eigen::Vector2i& exit,
    const int occupied_threshold,
    const double detect_dot_threshold
) const {
    if (regions.region_at(exit) != region
        || !grid_cell_traversable(cost_map, exit, occupied_threshold)) {
        return std::unexpected("terrain reachability exit is not traversable in the region");
    }
    TerrainReachability result;
    result.width = cost_map.geometry.width();
    result.can_reach_exit.assign(
        static_cast<size_t>(cost_map.geometry.width())
            * static_cast<size_t>(cost_map.geometry.height()), 0
    );
    std::queue<Eigen::Vector2i> open;
    open.push(exit);
    result.can_reach_exit[index_of(exit, cost_map.geometry.width())] = 1;
    while (!open.empty()) {
        result.open_peak = std::max(result.open_peak, open.size());
        const Eigen::Vector2i current = open.front();
        open.pop();
        ++result.expansions;
        for (const Eigen::Vector2i& delta : NEIGHBORS) {
            const Eigen::Vector2i predecessor = current - delta;
            if (regions.region_at(predecessor) != region
                || !grid_cell_traversable(cost_map, predecessor, occupied_threshold)
                || !grid_edge_avoids_corner_cutting(
                    cost_map, predecessor, current, occupied_threshold
                )) {
                continue;
            }
            StepDirection edge_direction;
            if (!directed_terrain_edge_allowed(
                direction_map,
                terrain_constraints,
                predecessor,
                current,
                detect_dot_threshold,
                &edge_direction
            ) || edge_direction != direction) {
                continue;
            }
            const size_t predecessor_index = index_of(predecessor, cost_map.geometry.width());
            if (result.can_reach_exit[predecessor_index]) continue;
            result.can_reach_exit[predecessor_index] = 1;
            open.push(predecessor);
        }
    }
    return result;
}

} // namespace nav_executor
