#include <nav_executor/path_planner/search/terrain_topology.hpp>

#include <array>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace nav_executor {

namespace {

const std::array<Eigen::Vector2i, 8> NEIGHBORS {{
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
}};

bool transition_adjacent(
    const BoundaryTransition& lhs,
    const BoundaryTransition& rhs,
    const CostMap& cost_map,
    const int occupied_threshold
) {
    const Eigen::Vector2i flat_delta = (lhs.flat_cell - rhs.flat_cell).cwiseAbs();
    const Eigen::Vector2i body_delta = (lhs.body_cell - rhs.body_cell).cwiseAbs();
    return flat_delta.maxCoeff() <= 1 && body_delta.maxCoeff() <= 1
        && grid_edge_avoids_corner_cutting(
            cost_map, lhs.flat_cell, rhs.flat_cell, occupied_threshold
        )
        && grid_edge_avoids_corner_cutting(
            cost_map, lhs.body_cell, rhs.body_cell, occupied_threshold
        );
}

std::expected<DirectedPortal, std::string> build_entry_portal(
    const TerrainPassage& passage,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const TerrainRegionIndex& regions,
    const int occupied_threshold,
    const double detect_dot_threshold
) {
    std::vector<BoundaryTransition> candidates;
    for (const Eigen::Vector2i& body : regions.cells(passage.region)) {
        for (const Eigen::Vector2i& delta : NEIGHBORS) {
            const Eigen::Vector2i flat = body - delta;
            if (!direction_map.geometry.contains_cell(flat)
                || direction_map.is_terrain_body_cell(flat)
                || !grid_cell_traversable(cost_map, flat, occupied_threshold)
                || !grid_cell_traversable(cost_map, body, occupied_threshold)
                || !grid_edge_avoids_corner_cutting(
                    cost_map, flat, body, occupied_threshold
                )) {
                continue;
            }
            StepDirection direction;
            if (!directed_terrain_edge_allowed(
                direction_map,
                terrain_constraints,
                flat,
                body,
                detect_dot_threshold,
                &direction
            ) || direction != passage.direction) {
                continue;
            }
            candidates.push_back({.flat_cell = flat, .body_cell = body});
        }
    }

    size_t selected = candidates.size();
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (same_cell(candidates[i].flat_cell, passage.selected_entry.flat_cell)
            && same_cell(candidates[i].body_cell, passage.selected_entry.body_cell)) {
            selected = i;
            break;
        }
    }
    if (selected == candidates.size()) {
        return std::unexpected("global A* entry edge is absent from its directed portal candidates");
    }

    std::vector<uint8_t> visited(candidates.size(), 0);
    std::queue<size_t> open;
    open.push(selected);
    visited[selected] = 1;
    DirectedPortal portal {
        .region = passage.region,
        .direction = passage.direction,
        .transitions = {},
    };
    while (!open.empty()) {
        const size_t current = open.front();
        open.pop();
        portal.transitions.push_back(candidates[current]);
        for (size_t next = 0; next < candidates.size(); ++next) {
            if (visited[next] || !transition_adjacent(
                candidates[current], candidates[next], cost_map, occupied_threshold
            )) {
                continue;
            }
            visited[next] = 1;
            open.push(next);
        }
    }
    if (portal.transitions.empty()) {
        return std::unexpected("directed entry portal is empty");
    }
    return portal;
}

} // anonymous namespace

bool same_cell(const Eigen::Vector2i& lhs, const Eigen::Vector2i& rhs) {
    return (lhs.array() == rhs.array()).all();
}

TerrainRegionIndex::TerrainRegionIndex(
    const DirectionMap& direction_map,
    const CostMap& planning_cost_map,
    const int occupied_threshold
)
    : width_(direction_map.geometry.width()),
      height_(direction_map.geometry.height()),
      region_by_cell_(
          static_cast<size_t>(width_) * static_cast<size_t>(height_),
          INVALID_TERRAIN_REGION
      ) {
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Eigen::Vector2i root(x, y);
            if (!direction_map.is_terrain_body_cell(root)
                || !grid_cell_traversable(
                    planning_cost_map, root, occupied_threshold
                )
                || region_at(root) != INVALID_TERRAIN_REGION) {
                continue;
            }
            const TerrainRegionId id = static_cast<TerrainRegionId>(cells_by_region_.size());
            cells_by_region_.emplace_back();
            const uint8_t label = direction_map.terrain_label_at_cell(root);
            std::queue<Eigen::Vector2i> open;
            open.push(root);
            region_by_cell_[index(root)] = id;
            while (!open.empty()) {
                const Eigen::Vector2i current = open.front();
                open.pop();
                cells_by_region_.back().push_back(current);
                for (const Eigen::Vector2i& delta : NEIGHBORS) {
                    const Eigen::Vector2i next = current + delta;
                    if (!direction_map.geometry.contains_cell(next)
                        || !direction_map.is_terrain_body_cell(next)
                        || direction_map.terrain_label_at_cell(next) != label
                        || !grid_cell_traversable(
                            planning_cost_map, next, occupied_threshold
                        )
                        || region_at(next) != INVALID_TERRAIN_REGION
                        || !grid_edge_avoids_corner_cutting(
                            planning_cost_map,
                            current,
                            next,
                            occupied_threshold
                        )) {
                        continue;
                    }
                    region_by_cell_[index(next)] = id;
                    open.push(next);
                }
            }
        }
    }
}

size_t TerrainRegionIndex::index(const Eigen::Vector2i& cell) const {
    return static_cast<size_t>(cell.y()) * static_cast<size_t>(width_)
        + static_cast<size_t>(cell.x());
}

TerrainRegionId TerrainRegionIndex::region_at(const Eigen::Vector2i& cell) const {
    if (cell.x() < 0 || cell.x() >= width_ || cell.y() < 0 || cell.y() >= height_) {
        return INVALID_TERRAIN_REGION;
    }
    return region_by_cell_[index(cell)];
}

const std::vector<Eigen::Vector2i>& TerrainRegionIndex::cells(
    const TerrainRegionId region
) const {
    if (region >= cells_by_region_.size()) {
        throw std::out_of_range("invalid terrain region id");
    }
    return cells_by_region_[region];
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

bool directed_terrain_edge_allowed(
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const Eigen::Vector2i& from,
    const Eigen::Vector2i& to,
    const double detect_dot_threshold,
    StepDirection* const direction
) {
    const bool destination_body = direction_map.is_terrain_body_cell(to);
    const bool source_body = direction_map.is_terrain_body_cell(from);
    if (!destination_body && !source_body) return true;
    const Eigen::Vector2i terrain_cell = destination_body ? to : from;
    const Eigen::Vector2d raw_direction = direction_map.raw_direction_at_cell(terrain_cell);
    const Eigen::Vector2d displacement = (to - from).cast<double>();
    if (raw_direction.squaredNorm() <= 1e-12 || displacement.squaredNorm() <= 1e-12) {
        return false;
    }
    const double alignment = displacement.normalized().dot(raw_direction.normalized());
    if (std::abs(alignment) <= detect_dot_threshold) return false;
    const bool going_up = alignment > 0.0;
    if (!terrain_constraints.selected_mode(
        direction_map.terrain_label_at_cell(terrain_cell), going_up
    )) {
        return false;
    }
    if (direction) *direction = going_up ? StepDirection::UP : StepDirection::DOWN;
    return true;
}

std::expected<std::vector<TerrainPassage>, std::string> extract_terrain_passages(
    const std::vector<Eigen::Vector2i>& raw_path,
    const CostMap& planning_cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const TerrainRegionIndex& regions,
    const int occupied_threshold,
    const double detect_dot_threshold
) {
    std::vector<TerrainPassage> passages;
    size_t cursor = 0;
    while (cursor < raw_path.size()) {
        if (!direction_map.is_terrain_body_cell(raw_path[cursor])) {
            ++cursor;
            continue;
        }
        const size_t first_body = cursor;
        while (cursor + 1 < raw_path.size()
            && direction_map.is_terrain_body_cell(raw_path[cursor + 1])) {
            ++cursor;
        }
        const size_t last_body = cursor;
        if (first_body == 0 || last_body + 1 >= raw_path.size()) {
            return std::unexpected("global path starts or ends inside a terrain body");
        }

        TerrainPassage passage;
        passage.region = regions.region_at(raw_path[first_body]);
        passage.label = direction_map.terrain_label_at_cell(raw_path[first_body]);
        passage.selected_entry = {
            .flat_cell = raw_path[first_body - 1],
            .body_cell = raw_path[first_body],
        };
        passage.selected_exit = {
            .flat_cell = raw_path[last_body + 1],
            .body_cell = raw_path[last_body],
        };
        for (size_t i = first_body; i <= last_body; ++i) {
            if (regions.region_at(raw_path[i]) != passage.region) {
                return std::unexpected(
                    "global path switches terrain region without passing through flat space"
                );
            }
            passage.initial_body_path.push_back(raw_path[i]);
        }
        if (!directed_terrain_edge_allowed(
            direction_map,
            terrain_constraints,
            passage.selected_entry.flat_cell,
            passage.selected_entry.body_cell,
            detect_dot_threshold,
            &passage.direction
        )) {
            return std::unexpected("global path contains an invalid terrain entry edge");
        }
        StepDirection exit_direction;
        if (!directed_terrain_edge_allowed(
            direction_map,
            terrain_constraints,
            passage.selected_exit.body_cell,
            passage.selected_exit.flat_cell,
            detect_dot_threshold,
            &exit_direction
        ) || exit_direction != passage.direction) {
            return std::unexpected("global path contains an inconsistent terrain exit edge");
        }
        auto portal = build_entry_portal(
            passage,
            planning_cost_map,
            direction_map,
            terrain_constraints,
            regions,
            occupied_threshold,
            detect_dot_threshold
        );
        if (!portal) return std::unexpected(portal.error());
        passage.entry_portal = std::move(*portal);
        passages.push_back(std::move(passage));
        ++cursor;
    }
    return passages;
}

} // namespace nav_executor
