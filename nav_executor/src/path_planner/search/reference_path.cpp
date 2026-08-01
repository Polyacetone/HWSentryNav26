#include <nav_executor/path_planner/search/reference_path.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include <nav_executor/path_planner/search/grid_utils.hpp>
#include <nav_executor/path_planner/trajectory/runup_gate.hpp>

namespace nav_executor {

namespace {

constexpr double EPS = 1e-9;

double wrap_angle(const double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

double path_length(const std::vector<Eigen::Vector2d>& points) {
    double length = 0.0;
    for (size_t i = 1; i < points.size(); ++i) {
        length += (points[i] - points[i - 1]).norm();
    }
    return length;
}

bool flat_line_of_sight(
    const Eigen::Vector2d& from,
    const Eigen::Vector2d& to,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const int occupied_threshold
) {
    const auto from_cell = cost_map.geometry.containing_cell(from);
    const auto to_cell = cost_map.geometry.containing_cell(to);
    if (!from_cell || !to_cell
        || direction_map.is_terrain_body_cell(*from_cell)
        || direction_map.is_terrain_body_cell(*to_cell)) {
        return false;
    }
    const double length = (to - from).norm();
    const int samples = std::max(1, static_cast<int>(std::ceil(
        length / (0.5 * cost_map.geometry.resolution())
    )));
    for (int i = 0; i <= samples; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(samples);
        const Eigen::Vector2d point = from + t * (to - from);
        const auto cost = cost_map.sample_map(point);
        const auto cell = direction_map.geometry.containing_cell(point);
        if (!cost || cost->value >= static_cast<double>(occupied_threshold)
            || !cell || direction_map.is_terrain_body_cell(*cell)) {
            return false;
        }
    }
    Eigen::Vector2i cell = *from_cell;
    for (const GridCrossing& crossing : trace_grid_crossings(
            cost_map.geometry, from, to
        )) {
        if (!grid_cell_traversable(cost_map, crossing.to, occupied_threshold)
            || !grid_edge_avoids_corner_cutting(
                cost_map, crossing.from, crossing.to, occupied_threshold
            ) || direction_map.is_terrain_body_cell(crossing.to)) {
            return false;
        }
        cell = crossing.to;
    }
    return same_cell(cell, *to_cell);
}

Eigen::Vector2d point_at_arc_length(
    const std::vector<Eigen::Vector2d>& points,
    const std::vector<double>& cumulative,
    const double arc_length
) {
    if (arc_length <= 0.0) return points.front();
    if (arc_length >= cumulative.back()) return points.back();
    const auto upper = std::upper_bound(cumulative.begin(), cumulative.end(), arc_length);
    const size_t next = static_cast<size_t>(upper - cumulative.begin());
    const size_t previous = next - 1;
    const double segment_length = cumulative[next] - cumulative[previous];
    const double t = segment_length > EPS
        ? (arc_length - cumulative[previous]) / segment_length : 0.0;
    return points[previous] + t * (points[next] - points[previous]);
}

} // anonymous namespace

ReferencePathBuilder::Result ReferencePathBuilder::build(
    const SpatialRoute& route,
    const Eigen::Vector2d& start_map,
    const Eigen::Vector2d& goal_map,
    const double start_speed,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const ShapingDynamicsLimits& dynamics,
    const int occupied_threshold
) const {
    Result result;
    if (route.raw_path.size() < 2 || params_.resample_spacing <= 0.0
        || params_.tangent_lookahead <= 0.0
        || params_.runup_transition_distance < 0.0
        || !cost_map.geometry.same_geometry(direction_map.geometry)
        || dynamics.velocity_max <= 0.0
        || dynamics.tangential_acceleration_max <= 0.0) {
        result.error = "reference path configuration or spatial route is invalid";
        return result;
    }

    std::vector<Eigen::Vector2d> raw_points;
    raw_points.reserve(route.raw_path.size());
    for (const Eigen::Vector2i& cell : route.raw_path) {
        raw_points.push_back(cost_map.geometry.cell_center(cell));
    }
    raw_points.front() = start_map;
    raw_points.back() = goal_map;
    result.path.raw_length = path_length(raw_points);

    std::vector<int8_t> passage_direction(
        static_cast<size_t>(cost_map.geometry.width())
            * static_cast<size_t>(cost_map.geometry.height()),
        0
    );
    const auto cell_index = [&](const Eigen::Vector2i& cell) {
        return static_cast<size_t>(cell.y())
            * static_cast<size_t>(cost_map.geometry.width())
            + static_cast<size_t>(cell.x());
    };
    for (const TerrainPassage& passage : route.passages) {
        for (size_t i = passage.first_body_index;
             i <= passage.last_body_index; ++i) {
            passage_direction[cell_index(route.raw_path[i])] =
                passage.going_up ? 1 : -1;
        }
    }

    std::vector<size_t> anchors {0, route.raw_path.size() - 1};
    for (const TerrainPassage& passage : route.passages) {
        anchors.insert(anchors.end(), {
            passage.entry_flat_index,
            passage.first_body_index,
            passage.last_body_index,
            passage.exit_flat_index,
        });
    }
    std::ranges::sort(anchors);
    anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());

    std::vector<Eigen::Vector2d>& smoothed = result.path.smoothed_path;
    smoothed.push_back(raw_points.front());
    for (size_t segment = 0; segment + 1 < anchors.size(); ++segment) {
        const size_t begin = anchors[segment];
        const size_t end = anchors[segment + 1];
        bool contains_body = false;
        for (size_t i = begin; i <= end; ++i) {
            contains_body = contains_body
                || direction_map.is_terrain_body_cell(route.raw_path[i]);
        }
        if (contains_body) {
            for (size_t i = begin + 1; i <= end; ++i) smoothed.push_back(raw_points[i]);
            continue;
        }
        size_t current = begin;
        while (current < end) {
            size_t furthest = current + 1;
            for (size_t candidate = end; candidate > current + 1; --candidate) {
                if (flat_line_of_sight(
                        raw_points[current], raw_points[candidate],
                        cost_map, direction_map, occupied_threshold
                    )) {
                    furthest = candidate;
                    break;
                }
            }
            smoothed.push_back(raw_points[furthest]);
            current = furthest;
        }
    }
    if (smoothed.size() < 2) {
        result.error = "reference path smoothing produced no segments";
        return result;
    }

    std::vector<double> cumulative(smoothed.size(), 0.0);
    for (size_t i = 1; i < smoothed.size(); ++i) {
        cumulative[i] = cumulative[i - 1] + (smoothed[i] - smoothed[i - 1]).norm();
    }
    result.path.smoothed_length = cumulative.back();
    if (result.path.smoothed_length <= EPS) {
        result.error = "reference path has zero length";
        return result;
    }

    std::vector<double> samples = cumulative;
    for (double s = params_.resample_spacing; s < cumulative.back();
         s += params_.resample_spacing) {
        samples.push_back(s);
    }
    std::ranges::sort(samples);
    samples.erase(std::unique(samples.begin(), samples.end(), [](const double a, const double b) {
        return std::abs(a - b) <= 1e-9;
    }), samples.end());

    result.path.points.resize(samples.size());
    std::vector<double> speed_cap(samples.size(), dynamics.velocity_max);
    for (size_t i = 0; i < samples.size(); ++i) {
        ReferencePoint& point = result.path.points[i];
        point.position = point_at_arc_length(smoothed, cumulative, samples[i]);
        const Eigen::Vector2d before = point_at_arc_length(
            smoothed, cumulative, std::max(0.0, samples[i] - params_.tangent_lookahead)
        );
        const Eigen::Vector2d after = point_at_arc_length(
            smoothed, cumulative,
            std::min(cumulative.back(), samples[i] + params_.tangent_lookahead)
        );
        Eigen::Vector2d tangent = after - before;
        if (tangent.squaredNorm() <= EPS) {
            result.error = "reference path contains an undefined tangent";
            return result;
        }
        tangent.normalize();

        const auto cell = direction_map.geometry.containing_cell(point.position);
        if (!cell) {
            result.error = "reference path leaves the direction map";
            return result;
        }
        if (direction_map.is_terrain_body_cell(*cell)) {
            Eigen::Vector2d terrain_direction = direction_map.raw_direction_at_cell(*cell);
            const int8_t direction_sign = passage_direction[cell_index(*cell)];
            if (terrain_direction.squaredNorm() <= EPS || direction_sign == 0) {
                result.error = "reference path entered terrain outside its recorded passage";
                return result;
            }
            terrain_direction.normalize();
            const bool going_up = direction_sign > 0;
            tangent = going_up ? terrain_direction : -terrain_direction;
            const TraversalMode* mode = terrain_constraints.selected_mode(
                direction_map.terrain_label_at_cell(*cell), going_up
            );
            if (!mode) {
                result.error = "reference path entered prohibited directional terrain";
                return result;
            }
            speed_cap[i] = std::min(speed_cap[i], mode->velocity_window.max);
        }
        point.heading = std::atan2(tangent.y(), tangent.x());
    }

    for (size_t i = 0; i < samples.size(); ++i) {
        const size_t previous = i > 0 ? i - 1 : i;
        const size_t next = i + 1 < samples.size() ? i + 1 : i;
        const double ds = samples[next] - samples[previous];
        if (ds <= EPS) continue;
        const double curvature = std::abs(wrap_angle(
            result.path.points[next].heading - result.path.points[previous].heading
        )) / ds;
        if (curvature <= EPS) continue;
        speed_cap[i] = std::min(
            speed_cap[i], dynamics.angular_velocity_max / curvature
        );
        speed_cap[i] = std::min(
            speed_cap[i], std::sqrt(dynamics.lateral_acceleration_max / curvature)
        );
    }

    result.path.points.front().speed = std::min(
        std::clamp(start_speed, 0.0, dynamics.velocity_max), speed_cap.front()
    );
    for (size_t i = 1; i < samples.size(); ++i) {
        const double ds = samples[i] - samples[i - 1];
        result.path.points[i].speed = std::min(speed_cap[i], std::sqrt(std::max(
            result.path.points[i - 1].speed * result.path.points[i - 1].speed
                + 2.0 * dynamics.tangential_acceleration_max * ds,
            0.0
        )));
    }
    result.path.points.back().speed = 0.0;
    for (size_t i = samples.size() - 1; i > 0; --i) {
        const double ds = samples[i] - samples[i - 1];
        result.path.points[i - 1].speed = std::min(
            result.path.points[i - 1].speed,
            std::sqrt(std::max(
                result.path.points[i].speed * result.path.points[i].speed
                    + 2.0 * dynamics.tangential_acceleration_max * ds,
                0.0
            ))
        );
    }

    const TraversalMode* downstream_mode = nullptr;
    Eigen::Vector2d downstream_direction = Eigen::Vector2d::Zero();
    double body_entry_arc = 0.0;
    for (size_t reverse = samples.size(); reverse > 0; --reverse) {
        const size_t i = reverse - 1;
        const auto cell = direction_map.geometry.containing_cell(
            result.path.points[i].position
        );
        if (!cell) {
            result.error = "reference path leaves the direction map during approach annotation";
            return result;
        }
        if (direction_map.is_terrain_body_cell(*cell)) {
            Eigen::Vector2d terrain_direction = direction_map.raw_direction_at_cell(*cell);
            const int8_t direction_sign = passage_direction[cell_index(*cell)];
            if (terrain_direction.squaredNorm() <= EPS || direction_sign == 0) {
                result.error = "terrain approach has no recorded passage direction";
                return result;
            }
            terrain_direction.normalize();
            const bool going_up = direction_sign > 0;
            downstream_mode = terrain_constraints.selected_mode(
                direction_map.terrain_label_at_cell(*cell), going_up
            );
            if (!downstream_mode) {
                result.error = "terrain approach references a prohibited traversal mode";
                return result;
            }
            downstream_direction = going_up ? terrain_direction : -terrain_direction;
            body_entry_arc = samples[i];
            continue;
        }
        if (!downstream_mode) continue;
        const double distance = std::max(body_entry_arc - samples[i], 0.0);
        const double gate = runup_distance_gate(
            distance,
            downstream_mode->run_up,
            params_.runup_transition_distance
        ).first;
        if (gate <= 0.0) {
            downstream_mode = nullptr;
            downstream_direction.setZero();
            continue;
        }
        result.path.points[i].approach = {
            .direction = downstream_direction,
            .velocity_window = downstream_mode->velocity_window,
            .gate = gate,
        };
    }

    result.path.points.back().time_to_goal = 0.0;
    for (size_t i = samples.size() - 1; i > 0; --i) {
        const double ds = samples[i] - samples[i - 1];
        const double speed_sum = result.path.points[i - 1].speed
            + result.path.points[i].speed;
        const double duration = speed_sum > EPS
            ? 2.0 * ds / speed_sum : ds / dynamics.velocity_max;
        result.path.points[i - 1].time_to_goal =
            result.path.points[i].time_to_goal + duration;
    }
    result.success = true;
    return result;
}

} // namespace nav_executor
