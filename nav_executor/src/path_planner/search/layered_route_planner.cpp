#include <nav_executor/path_planner/search/layered_route_planner.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace nav_executor {

namespace {

double wrap_angle(const double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

bool append_route(SpatialRoute& destination, const SpatialRoute& source) {
    if (source.edges.empty()) return false;
    if (destination.edges.empty()) {
        destination.start = source.start;
    } else if ((destination.edges.back().to.position - source.start.position).norm() > 1e-5) {
        return false;
    }
    destination.edges.insert(
        destination.edges.end(), source.edges.begin(), source.edges.end()
    );
    return true;
}

std::vector<std::vector<Eigen::Vector2d>> extract_flat_references(
    const std::vector<Eigen::Vector2i>& raw_path,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const Eigen::Vector2d& exact_start,
    const Eigen::Vector2d& exact_goal
) {
    std::vector<std::vector<Eigen::Vector2d>> references;
    std::vector<Eigen::Vector2d> current;
    for (const Eigen::Vector2i& cell : raw_path) {
        if (direction_map.is_terrain_body_cell(cell)) {
            if (!current.empty()) {
                references.push_back(std::move(current));
                current.clear();
            }
            continue;
        }
        current.push_back(cost_map.geometry.cell_center(cell));
    }
    if (!current.empty()) references.push_back(std::move(current));
    if (!references.empty()) {
        references.front().front() = exact_start;
        references.back().back() = exact_goal;
    }
    return references;
}

std::optional<SpeedSquaredInterval> directional_speed_limit(
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& constraints,
    const Eigen::Vector2i& terrain_cell,
    const Eigen::Vector2d& movement,
    const StepDirection required_direction,
    const double detect_dot_threshold
) {
    const Eigen::Vector2d raw_direction = direction_map.raw_direction_at_cell(terrain_cell);
    if (raw_direction.squaredNorm() <= 1e-12 || movement.squaredNorm() <= 1e-12) {
        return std::nullopt;
    }
    const double alignment = movement.normalized().dot(raw_direction.normalized());
    if (std::abs(alignment) <= detect_dot_threshold) return std::nullopt;
    const StepDirection direction = alignment > 0.0
        ? StepDirection::UP : StepDirection::DOWN;
    if (direction != required_direction) return std::nullopt;
    const TraversalMode* mode = constraints.selected_mode(
        direction_map.terrain_label_at_cell(terrain_cell), direction == StepDirection::UP
    );
    if (!mode) return std::nullopt;
    return SpeedSquaredInterval {
        .min = mode->velocity_window.min * mode->velocity_window.min,
        .max = mode->velocity_window.max * mode->velocity_window.max,
    };
}

std::optional<std::pair<SpatialRoute, SpeedSquaredInterval>> build_terrain_route(
    const SpatialPose& entry_pose,
    const SpeedSquaredInterval& entry_speed,
    const std::vector<Eigen::Vector2i>& body_path,
    const BoundaryTransition& exit,
    const StepDirection direction,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& constraints,
    const double detect_dot_threshold,
    const SpeedReachability& speed,
    std::string& error
) {
    if (body_path.empty()) {
        error = "terrain-local A* returned an empty body path";
        return std::nullopt;
    }
    struct TerrainRouteTarget {
        Eigen::Vector2d position;
        Eigen::Vector2i terrain_cell;
    };
    std::vector<TerrainRouteTarget> targets;
    targets.reserve(body_path.size() + 1);
    for (size_t i = 0; i < body_path.size(); ++i) {
        const Eigen::Vector2i& cell = body_path[i];
        const Eigen::Vector2d center = cost_map.geometry.cell_center(cell);
        if ((entry_pose.position - center).norm() <= 1e-8) continue;
        if (!targets.empty() && (targets.back().position - center).norm() <= 1e-8) continue;
        targets.push_back({
            .position = center,
            .terrain_cell = cell,
        });
    }
    targets.push_back({
        .position = cost_map.geometry.cell_center(exit.flat_cell),
        .terrain_cell = exit.body_cell,
    });

    SpatialRoute route;
    route.start = entry_pose;
    SpatialPose current = entry_pose;
    SpeedSquaredInterval current_speed = entry_speed;
    for (size_t target_index = 0; target_index < targets.size(); ++target_index) {
        const TerrainRouteTarget& target = targets[target_index];
        const Eigen::Vector2d& position = target.position;
        const Eigen::Vector2i& terrain_cell = target.terrain_cell;
        const Eigen::Vector2d movement = position - current.position;
        const double length = movement.norm();
        if (length <= 1e-8) continue;
        const double edge_heading = std::atan2(movement.y(), movement.x());
        // 地形栅格折线只提供空间 seed；格点切向跳变不代表可执行曲率，
        // 连续曲率与完整动力学约束由后续 MINCO 负责。
        const double curvature = 0.0;
        const auto limit = directional_speed_limit(
            direction_map,
            constraints,
            terrain_cell,
            movement,
            direction,
            detect_dot_threshold
        );
        if (!limit) {
            error = "terrain route violates direction or mode constraints";
            return std::nullopt;
        }
        SpatialPose next {.position = position, .heading = edge_heading};
        const auto propagated = speed.propagate(
            current_speed, length, curvature, *limit, false
        );
        if (!propagated) {
            error = "terrain route has an empty forward speed interval at edge "
                + std::to_string(target_index)
                + " (length=" + std::to_string(length)
                + ", curvature=" + std::to_string(curvature)
                + ", input=[" + std::to_string(current_speed.min)
                + "," + std::to_string(current_speed.max)
                + "], limit=[" + std::to_string(limit->min)
                + "," + std::to_string(limit->max) + "])";
            return std::nullopt;
        }
        route.edges.push_back({
            .from = current,
            .to = next,
            .length = length,
            .curvature = curvature,
            .endpoint_speed_limit = *limit,
        });
        current = next;
        current_speed = *propagated;
    }
    if (route.edges.empty()) {
        error = "terrain route produced no spatial edge";
        return std::nullopt;
    }
    return std::pair {std::move(route), current_speed};
}

} // anonymous namespace

LayeredRoutePlanner::Result LayeredRoutePlanner::search(
    const Eigen::Vector2d& start_map,
    const double start_yaw,
    const double start_velocity,
    const Eigen::Vector2d& goal_map,
    const CostMap& planning_cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints
) const {
    Result result;
    if (!planning_cost_map.geometry.same_geometry(direction_map.geometry)) {
        result.error = "planning cost map and direction map geometry mismatch";
        return result;
    }
    const auto start_cell = planning_cost_map.geometry.containing_cell(start_map);
    const auto goal_cell = planning_cost_map.geometry.containing_cell(goal_map);
    if (!start_cell || !goal_cell) {
        result.error = "layered route endpoint is outside the planning map";
        return result;
    }
    if (!primitive_library_.valid()) {
        result.error = "motion primitive generation failed: " + primitive_library_.error();
        return result;
    }

    const TerrainRegionIndex regions(
        direction_map, planning_cost_map, params_.occupied_threshold
    );
    const DirectedGridAstar grid_astar(params_.grid_astar);
    const auto global = grid_astar.search_global(
        planning_cost_map,
        direction_map,
        terrain_constraints,
        *start_cell,
        *goal_cell,
        params_.occupied_threshold,
        params_.detect_dot_threshold
    );
    if (!global) {
        result.error = "global directed A* failed: " + global.error();
        return result;
    }
    result.diagnostics.global_expansions = global->expansions;
    result.diagnostics.global_open_peak = global->open_peak;
    result.global_raw_path.reserve(global->raw_path.size());
    for (const Eigen::Vector2i& cell : global->raw_path) {
        result.global_raw_path.push_back(planning_cost_map.geometry.cell_center(cell));
    }

    auto passages = extract_terrain_passages(
        global->raw_path,
        planning_cost_map,
        direction_map,
        terrain_constraints,
        regions,
        params_.occupied_threshold,
        params_.detect_dot_threshold
    );
    if (!passages) {
        result.error = "terrain passage extraction failed: " + passages.error();
        return result;
    }
    const auto flat_references = extract_flat_references(
        global->raw_path,
        planning_cost_map,
        direction_map,
        start_map,
        goal_map
    );
    if (flat_references.size() != passages->size() + 1) {
        result.error = "global raw path produced an inconsistent number of flat segments";
        return result;
    }
    result.diagnostics.passage_count = passages->size();
    for (const TerrainPassage& passage : *passages) {
        result.diagnostics.portal_sizes.push_back(passage.entry_portal.transitions.size());
        result.diagnostics.terrain_regions.push_back(passage.region);
        result.diagnostics.terrain_labels.push_back(passage.label);
    }

    const double clamped_start_speed = std::clamp(
        std::max(start_velocity, 0.0),
        0.0,
        params_.state_lattice.dynamics.velocity_max
    );
    result.initial_speed = {
        clamped_start_speed * clamped_start_speed,
        clamped_start_speed * clamped_start_speed,
    };
    SpeedSquaredInterval segment_root_speed = result.initial_speed;
    SpatialPose segment_root {.position = start_map, .heading = wrap_angle(start_yaw)};
    bool first_flat_segment = true;
    size_t flat_segment_index = 0;
    const StateLatticeAstar lattice(params_.state_lattice, primitive_library_);

    const auto run_flat_segment = [&](
        const SearchTarget& target,
        const bool allow_yaw_relaxation,
        const std::vector<Eigen::Vector2d>& reference_path
    ) -> std::optional<StateLatticeAstar::Result> {
        const LatticeFrame frame {
            .origin_map = segment_root.position,
            .base_heading = segment_root.heading,
            .xy_resolution = params_.lattice_xy_resolution,
            .heading_bins = params_.lattice_heading_bins,
        };
        std::vector<StateLatticeAstar::SearchRoot> roots {{
            .key = {0, 0, 0},
            .reachable_speed = segment_root_speed,
            .initial_cost = 0.0,
            .relaxed = false,
        }};
        if (allow_yaw_relaxation
            && std::abs(start_velocity)
                <= params_.start_yaw_relaxation.speed_threshold) {
            for (int heading = 1; heading < params_.lattice_heading_bins; ++heading) {
                const double yaw_change = std::abs(wrap_angle(
                    2.0 * std::numbers::pi * static_cast<double>(heading)
                    / static_cast<double>(params_.lattice_heading_bins)
                ));
                roots.push_back({
                    .key = {0, 0, heading},
                    .reachable_speed = segment_root_speed,
                    .initial_cost = params_.start_yaw_relaxation.root_penalty
                        + params_.start_yaw_relaxation.yaw_penalty * yaw_change,
                    .relaxed = true,
                });
            }
        }
        StateLatticeAstar::Result flat = lattice.search(
            frame,
            roots,
            target,
            planning_cost_map,
            direction_map,
            terrain_constraints,
            params_.occupied_threshold,
            params_.detect_dot_threshold,
            reference_path
        );
        result.diagnostics.lattice_expansions += flat.expansions;
        result.diagnostics.lattice_labels += flat.generated_labels;
        result.diagnostics.lattice_dominated += flat.dominated_labels;
        result.diagnostics.lattice_transition_checks += flat.transition_checks;
        result.diagnostics.lattice_terminal_attempts += flat.terminal_attempts;
        result.diagnostics.rejected_portal_terminals
            += flat.rejected_portal_terminals;
        result.diagnostics.lattice_open_peak = std::max(
            result.diagnostics.lattice_open_peak, flat.open_peak
        );
        result.diagnostics.lattice_anchor_queue_peak = std::max(
            result.diagnostics.lattice_anchor_queue_peak, flat.anchor_queue_peak
        );
        result.diagnostics.lattice_pending_focal_queue_peak = std::max(
            result.diagnostics.lattice_pending_focal_queue_peak,
            flat.pending_focal_queue_peak
        );
        result.diagnostics.lattice_focal_queue_peak = std::max(
            result.diagnostics.lattice_focal_queue_peak, flat.focal_queue_peak
        );
        result.diagnostics.lattice_stale_queue_entries += flat.stale_queue_entries;
        if (!flat.success) {
            result.error = "flat state-lattice A* failed: " + flat.error;
            return std::nullopt;
        }
        result.diagnostics.lattice_search_cost += flat.selected_search_cost;
        if (first_flat_segment) {
            result.diagnostics.selected_relaxed_root = flat.selected_relaxed_root;
            result.diagnostics.selected_root_cost = flat.selected_root_cost;
        }
        return flat;
    };

    for (const TerrainPassage& passage : *passages) {
        const auto local_reachability = grid_astar.build_terrain_reachability(
            planning_cost_map,
            direction_map,
            terrain_constraints,
            regions,
            passage.region,
            passage.direction,
            passage.selected_exit.body_cell,
            params_.occupied_threshold,
            params_.detect_dot_threshold
        );
        if (!local_reachability) {
            result.error = "terrain reverse reachability failed: "
                + local_reachability.error();
            return result;
        }
        result.diagnostics.terrain_reachability_expansions
            += local_reachability->expansions;
        result.diagnostics.terrain_reachability_open_peak = std::max(
            result.diagnostics.terrain_reachability_open_peak,
            local_reachability->open_peak
        );
        DirectedPortal reachable_portal = passage.entry_portal;
        std::erase_if(
            reachable_portal.transitions,
            [&](const BoundaryTransition& transition) {
                return !local_reachability->contains(transition.body_cell);
            }
        );
        result.diagnostics.unreachable_portal_transitions +=
            passage.entry_portal.transitions.size() - reachable_portal.transitions.size();
        if (reachable_portal.transitions.empty()) {
            result.error = "directed portal has no transition that can reach its fixed exit";
            return result;
        }
        auto flat = run_flat_segment(
            PassingPortalTarget {
                .portal = std::move(reachable_portal),
            },
            first_flat_segment,
            flat_references[flat_segment_index++]
        );
        if (!flat) return result;
        if (!flat->portal_transition) {
            result.error = "passing portal search returned no crossing transition";
            return result;
        }
        if (!append_route(result.route, flat->route)) {
            result.error = "failed to concatenate a flat route segment";
            return result;
        }
        first_flat_segment = false;

        const auto terrain_path = grid_astar.search_terrain_region(
            planning_cost_map,
            direction_map,
            terrain_constraints,
            regions,
            passage.region,
            passage.direction,
            flat->portal_transition->body_cell,
            passage.selected_exit.body_cell,
            params_.occupied_threshold,
            params_.detect_dot_threshold
        );
        if (!terrain_path) {
            result.error = "terrain-local directed A* failed for region "
                + std::to_string(passage.region)
                + " from flat (" + std::to_string(flat->portal_transition->flat_cell.x())
                + "," + std::to_string(flat->portal_transition->flat_cell.y())
                + ") into body (" + std::to_string(flat->portal_transition->body_cell.x())
                + "," + std::to_string(flat->portal_transition->body_cell.y())
                + ") to (" + std::to_string(passage.selected_exit.body_cell.x())
                + "," + std::to_string(passage.selected_exit.body_cell.y())
                + "): " + terrain_path.error();
            return result;
        }
        result.diagnostics.terrain_expansions += terrain_path->expansions;
        std::string terrain_error;
        const auto terrain_route = build_terrain_route(
            flat->route.edges.back().to,
            flat->terminal_speed,
            terrain_path->raw_path,
            passage.selected_exit,
            passage.direction,
            planning_cost_map,
            direction_map,
            terrain_constraints,
            params_.detect_dot_threshold,
            speed_,
            terrain_error
        );
        if (!terrain_route) {
            result.error = "terrain route construction failed: " + terrain_error;
            return result;
        }
        if (!append_route(result.route, terrain_route->first)) {
            result.error = "failed to concatenate a terrain route segment";
            return result;
        }
        segment_root = result.route.edges.back().to;
        segment_root_speed = terrain_route->second;
    }

    auto final_flat = run_flat_segment(
        PointStopTarget {.position_map = goal_map},
        first_flat_segment,
        flat_references[flat_segment_index]
    );
    if (!final_flat) return result;
    if (!append_route(result.route, final_flat->route)) {
        result.error = "failed to concatenate the final flat route segment";
        return result;
    }
    result.success = true;
    return result;
}

} // namespace nav_executor
