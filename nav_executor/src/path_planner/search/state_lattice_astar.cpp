#include <nav_executor/path_planner/search/state_lattice_astar.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <numbers>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <ceres/ceres.h>

namespace nav_executor {

namespace {

constexpr double EPS = 1e-9;

double wrap_angle(const double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

int wrap_heading(const int heading, const int bins) {
    const int result = heading % bins;
    return result < 0 ? result + bins : result;
}

SpatialPose advance_pose(
    const SpatialPose& from,
    const double curvature,
    const double length
) {
    SpatialPose to = from;
    const Eigen::Vector2d tangent(std::cos(from.heading), std::sin(from.heading));
    const Eigen::Vector2d normal(-tangent.y(), tangent.x());
    if (std::abs(curvature) <= EPS) {
        to.position += length * tangent;
    } else {
        const double angle = curvature * length;
        to.position += std::sin(angle) / curvature * tangent
            + (1.0 - std::cos(angle)) / curvature * normal;
        to.heading = wrap_angle(from.heading + angle);
    }
    return to;
}

template <typename T>
void advance_components(
    T& x,
    T& y,
    T& heading,
    const T& curvature,
    const T& length
) {
    const T angle = curvature * length;
    const T half = angle * T(0.5);
    const T half_squared = half * half;
    const T sinc_half = T(1.0) - half_squared / T(6.0)
        + half_squared * half_squared / T(120.0)
        - half_squared * half_squared * half_squared / T(5040.0)
        + half_squared * half_squared * half_squared * half_squared / T(362880.0)
        - half_squared * half_squared * half_squared * half_squared * half_squared
            / T(39916800.0)
        + half_squared * half_squared * half_squared * half_squared * half_squared
            * half_squared / T(6227020800.0);
    const T chord = length * sinc_half;
    x += chord * ceres::cos(heading + half);
    y += chord * ceres::sin(heading + half);
    heading += angle;
}

struct PrimitiveResidual {
    double start_heading;
    double target_x;
    double target_y;
    double target_heading;
    double nominal_curvature;
    double nominal_length;

    template <typename T>
    bool operator()(const T* const curvature, const T* const length, T* residual) const {
        T x = T(0.0);
        T y = T(0.0);
        T heading = T(start_heading);
        for (int i = 0; i < 3; ++i) {
            advance_components(x, y, heading, curvature[i], length[i]);
        }
        residual[0] = T(10000.0) * (x - T(target_x));
        residual[1] = T(10000.0) * (y - T(target_y));
        residual[2] = T(10000.0) * (heading - T(target_heading));
        residual[3] = T(1.0) * (
            length[0] + length[1] + length[2] - T(nominal_length)
        );
        for (int i = 0; i < 3; ++i) {
            residual[4 + i] = T(0.01) * (curvature[i] - T(nominal_curvature));
            residual[7 + i] = T(0.01) * (length[i] - T(nominal_length / 3.0));
        }
        return true;
    }
};

struct PrimitiveClosureResidual {
    double start_heading;
    double target_x;
    double target_y;
    double target_heading;
    std::array<double, 3> length;

    template <typename T>
    bool operator()(const T* const curvature, T* residual) const {
        T x = T(0.0);
        T y = T(0.0);
        T heading = T(start_heading);
        for (int i = 0; i < 3; ++i) {
            advance_components(x, y, heading, curvature[i], T(length[static_cast<size_t>(i)]));
        }
        residual[0] = x - T(target_x);
        residual[1] = y - T(target_y);
        residual[2] = heading - T(target_heading);
        return true;
    }
};

struct PrimitiveCandidate {
    LatticeKey endpoint;
    double target_heading = 0.0;
    double nominal_error = 0.0;
};

std::vector<double> nominal_curvatures(const MotionPrimitiveLibrary::Params& params) {
    if (params.nominal_curvature_samples <= 1) return {0.0};
    std::vector<double> result;
    result.reserve(static_cast<size_t>(params.nominal_curvature_samples));
    for (int i = 0; i < params.nominal_curvature_samples; ++i) {
        const double fraction = static_cast<double>(i)
            / static_cast<double>(params.nominal_curvature_samples - 1);
        const double centered = 2.0 * fraction - 1.0;
        result.push_back(params.curvature_max * centered * centered * centered);
    }
    return result;
}

struct LatticeKeyHash {
    size_t operator()(const LatticeKey& key) const {
        size_t seed = std::hash<int>{}(key.x);
        seed ^= std::hash<int>{}(key.y) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(key.heading) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
        return seed;
    }
};

uint64_t transition_key(const BoundaryTransition& transition) {
    const auto pack_cell = [](const Eigen::Vector2i& cell) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(cell.x())) << 32)
            | static_cast<uint32_t>(cell.y());
    };
    const uint64_t flat = pack_cell(transition.flat_cell);
    const uint64_t body = pack_cell(transition.body_cell);
    return flat ^ (body + 0x9e3779b97f4a7c15ULL + (flat << 6) + (flat >> 2));
}

struct GridCrossing {
    Eigen::Vector2i from;
    Eigen::Vector2i to;
    double fraction = 0.0;
};

double distance_to_transition(
    const CostMap& map,
    const Eigen::Vector2d& position_map,
    const BoundaryTransition& transition
) {
    const Eigen::Vector2d position = map.map_coord_to_grid(position_map);
    const Eigen::Vector2i delta = transition.body_cell - transition.flat_cell;
    Eigen::Vector2d closest;
    if (delta.x() != 0 && delta.y() == 0) {
        closest.x() = static_cast<double>(std::max(
            transition.flat_cell.x(), transition.body_cell.x()
        ));
        closest.y() = std::clamp(
            position.y(),
            static_cast<double>(transition.flat_cell.y()),
            static_cast<double>(transition.flat_cell.y() + 1)
        );
    } else if (delta.x() == 0 && delta.y() != 0) {
        closest.x() = std::clamp(
            position.x(),
            static_cast<double>(transition.flat_cell.x()),
            static_cast<double>(transition.flat_cell.x() + 1)
        );
        closest.y() = static_cast<double>(std::max(
            transition.flat_cell.y(), transition.body_cell.y()
        ));
    } else {
        closest = Eigen::Vector2d(
            std::max(transition.flat_cell.x(), transition.body_cell.x()),
            std::max(transition.flat_cell.y(), transition.body_cell.y())
        );
    }
    return (position - closest).norm() * map.resolution;
}

std::vector<GridCrossing> trace_grid_crossings(
    const CostMap& map,
    const Eigen::Vector2d& from_map,
    const Eigen::Vector2d& to_map
) {
    const Eigen::Vector2d from = map.map_coord_to_grid(from_map);
    const Eigen::Vector2d to = map.map_coord_to_grid(to_map);
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
        double fraction;
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
        if (crossings.size() > static_cast<size_t>(map.width + map.height)) break;
    }
    return crossings;
}

struct PropagationResult {
    SpatialPose pose;
    SpeedSquaredInterval speed;
    SpatialRoute route;
    std::optional<BoundaryTransition> transition;
};

std::optional<SpeedSquaredInterval> terrain_speed_limit(
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& constraints,
    const BoundaryTransition& transition,
    const Eigen::Vector2d& actual_tangent,
    const StepDirection required_direction,
    const double detect_dot_threshold
) {
    const Eigen::Vector2d raw_direction = direction_map.at(transition.body_cell);
    if (raw_direction.squaredNorm() <= 1e-12 || actual_tangent.squaredNorm() <= 1e-12) {
        return std::nullopt;
    }
    const double alignment = actual_tangent.normalized().dot(raw_direction.normalized());
    if (std::abs(alignment) <= detect_dot_threshold) return std::nullopt;
    const StepDirection direction = alignment > 0.0
        ? StepDirection::UP : StepDirection::DOWN;
    if (direction != required_direction) return std::nullopt;
    const bool going_up = direction == StepDirection::UP;
    const TraversalMode* mode = constraints.selected_mode(
        direction_map.terrain_at(transition.body_cell), going_up
    );
    if (!mode) return std::nullopt;
    return SpeedSquaredInterval {
        .min = mode->velocity_window.min * mode->velocity_window.min,
        .max = mode->velocity_window.max * mode->velocity_window.max,
    };
}

struct PrimitivePropagation {
    bool valid = false;
    bool terminal = false;
    PropagationResult result;
};

PrimitivePropagation propagate_segments(
    const SpatialPose& start,
    const SpeedSquaredInterval& start_speed,
    const std::vector<PrimitiveSegment>& segments,
    const std::optional<SpatialPose>& canonical_endpoint,
    const PassingPortalTarget* const portal_target,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const int occupied_threshold,
    const double detect_dot_threshold,
    const double collision_resolution,
    const SpeedReachability& speed,
    int& transition_checks,
    int& rejected_portal_terminals,
    const bool terminal_stop
) {
    std::unordered_set<uint64_t> portal_edges;
    if (portal_target) {
        for (const BoundaryTransition& transition : portal_target->portal.transitions) {
            portal_edges.insert(transition_key(transition));
        }
    }

    PropagationResult propagated;
    propagated.pose = start;
    propagated.speed = start_speed;
    propagated.route.start = start;
    for (size_t segment_index = 0; segment_index < segments.size(); ++segment_index) {
        const PrimitiveSegment& segment = segments[segment_index];
        const int substeps = std::max(
            1,
            static_cast<int>(std::ceil(segment.length / collision_resolution))
        );
        const double substep_length = segment.length / static_cast<double>(substeps);
        for (int substep = 0; substep < substeps; ++substep) {
            const SpatialPose previous = propagated.pose;
            SpatialPose next = advance_pose(previous, segment.curvature, substep_length);
            ++transition_checks;
            const Eigen::Vector2d next_grid = cost_map.map_coord_to_grid(next.position);
            if (!cost_map.is_valid_coord(next_grid)
                || cost_map.interpolate(next_grid) >= static_cast<double>(occupied_threshold)) {
                return {};
            }
            const std::vector<GridCrossing> crossings = trace_grid_crossings(
                cost_map, previous.position, next.position
            );
            for (const GridCrossing& crossing : crossings) {
                if (!grid_cell_traversable(cost_map, crossing.to, occupied_threshold)
                    || !grid_edge_avoids_corner_cutting(
                        cost_map, crossing.from, crossing.to, occupied_threshold
                    )) {
                    return {};
                }
                if (!direction_map.is_terrain_body_at(crossing.to)) continue;
                const BoundaryTransition transition {
                    .flat_cell = crossing.from,
                    .body_cell = crossing.to,
                };
                if (!portal_target
                    || direction_map.is_terrain_body_at(crossing.from)
                    || !portal_edges.contains(transition_key(transition))) {
                    return {};
                }
                const double partial_length = std::max(
                    substep_length * crossing.fraction, 1e-6
                );
                SpatialPose terminal_pose = advance_pose(
                    previous, segment.curvature, partial_length
                );
                const Eigen::Vector2d actual_tangent(
                    std::cos(terminal_pose.heading), std::sin(terminal_pose.heading)
                );
                const auto limit = terrain_speed_limit(
                    direction_map,
                    terrain_constraints,
                    transition,
                    actual_tangent,
                    portal_target->portal.direction,
                    detect_dot_threshold
                );
                if (!limit) {
                    ++rejected_portal_terminals;
                    return {};
                }
                const auto terminal_speed = speed.propagate(
                    propagated.speed,
                    partial_length,
                    segment.curvature,
                    *limit,
                    false
                );
                if (!terminal_speed) {
                    ++rejected_portal_terminals;
                    return {};
                }
                propagated.route.edges.push_back({
                    .from = previous,
                    .to = terminal_pose,
                    .length = partial_length,
                    .curvature = segment.curvature,
                    .endpoint_speed_limit = *limit,
                });
                propagated.pose = terminal_pose;
                propagated.speed = *terminal_speed;
                propagated.transition = transition;
                return {.valid = true, .terminal = true, .result = std::move(propagated)};
            }

            const bool final_edge = terminal_stop
                && segment_index + 1 == segments.size()
                && substep + 1 == substeps;
            const auto next_speed = speed.propagate(
                propagated.speed,
                substep_length,
                segment.curvature,
                speed.unrestricted_limit(),
                final_edge
            );
            if (!next_speed) return {};
            if (canonical_endpoint && segment_index + 1 == segments.size()
                && substep + 1 == substeps) {
                next = *canonical_endpoint;
            }
            propagated.route.edges.push_back({
                .from = previous,
                .to = next,
                .length = substep_length,
                .curvature = segment.curvature,
                .endpoint_speed_limit = speed.unrestricted_limit(),
                .terminal_stop = final_edge,
            });
            propagated.pose = next;
            propagated.speed = *next_speed;
        }
    }
    return {.valid = true, .terminal = false, .result = std::move(propagated)};
}

std::optional<PrimitiveSegment> goal_connection(
    const SpatialPose& start,
    const Eigen::Vector2d& goal,
    const double max_length,
    const double max_curvature
) {
    const Eigen::Vector2d displacement = goal - start.position;
    if (displacement.norm() <= EPS) return std::nullopt;
    const Eigen::Vector2d tangent(std::cos(start.heading), std::sin(start.heading));
    const Eigen::Vector2d normal(-tangent.y(), tangent.x());
    const double local_x = displacement.dot(tangent);
    const double local_y = displacement.dot(normal);
    PrimitiveSegment connection;
    if (std::abs(local_y) <= 1e-8) {
        if (local_x <= 0.0) return std::nullopt;
        connection.length = local_x;
    } else {
        connection.curvature = 2.0 * local_y / displacement.squaredNorm();
        const double heading_change = 2.0 * std::atan2(local_y, local_x);
        connection.length = heading_change / connection.curvature;
    }
    if (!std::isfinite(connection.length) || connection.length <= EPS
        || connection.length > max_length
        || std::abs(connection.curvature) > max_curvature + EPS) {
        return std::nullopt;
    }
    return connection;
}

struct SearchNode {
    LatticeKey key;
    SpeedSquaredInterval speed;
    double g = 0.0;
    double anchor = 0.0;
    double guide = 0.0;
    int parent = -1;
    int incoming_primitive = -1;
    bool relaxed_root = false;
    bool active = true;
    bool open = true;
    bool in_focal = false;
};

struct AnchorEntry {
    double anchor = 0.0;
    int node = -1;
    bool operator>(const AnchorEntry& other) const {
        if (anchor != other.anchor) return anchor > other.anchor;
        return node > other.node;
    }
};

struct GuideEntry {
    double guide = 0.0;
    double anchor = 0.0;
    int node = -1;
    bool operator>(const GuideEntry& other) const {
        if (guide != other.guide) return guide > other.guide;
        if (anchor != other.anchor) return anchor > other.anchor;
        return node > other.node;
    }
};

class ReferencePathGuide {
public:
    explicit ReferencePathGuide(const std::vector<Eigen::Vector2d>& raw_path) {
        if (raw_path.empty()) return;
        constexpr double MIN_GUIDE_SPACING = 0.2;
        points_.push_back(raw_path.front());
        for (size_t i = 1; i + 1 < raw_path.size(); ++i) {
            if ((raw_path[i] - points_.back()).norm() >= MIN_GUIDE_SPACING) {
                points_.push_back(raw_path[i]);
            }
        }
        if ((raw_path.back() - points_.back()).norm() > 1e-9) {
            points_.push_back(raw_path.back());
        }
        remaining_.assign(points_.size(), 0.0);
        for (size_t i = points_.size(); i > 1; --i) {
            remaining_[i - 2] = remaining_[i - 1]
                + (points_[i - 1] - points_[i - 2]).norm();
        }
    }

    [[nodiscard]] double score(const Eigen::Vector2d& position) const {
        if (points_.empty()) return 0.0;
        if (points_.size() == 1) return (position - points_.front()).norm();
        double best_distance_squared = std::numeric_limits<double>::infinity();
        double best_remaining = remaining_.front();
        for (size_t i = 0; i + 1 < points_.size(); ++i) {
            const Eigen::Vector2d segment = points_[i + 1] - points_[i];
            const double length_squared = segment.squaredNorm();
            const double fraction = length_squared <= EPS
                ? 0.0
                : std::clamp(
                    (position - points_[i]).dot(segment) / length_squared,
                    0.0,
                    1.0
                );
            const Eigen::Vector2d projection = points_[i] + fraction * segment;
            const double distance_squared = (position - projection).squaredNorm();
            const double remaining = remaining_[i] - fraction * std::sqrt(length_squared);
            if (distance_squared < best_distance_squared - EPS
                || (std::abs(distance_squared - best_distance_squared) <= EPS
                    && remaining < best_remaining)) {
                best_distance_squared = distance_squared;
                best_remaining = remaining;
            }
        }
        return std::sqrt(std::max(best_distance_squared, 0.0)) + best_remaining;
    }

private:
    std::vector<Eigen::Vector2d> points_;
    std::vector<double> remaining_;
};

double route_length(const SpatialRoute& route) {
    double length = 0.0;
    for (const SpatialRouteEdge& edge : route.edges) length += edge.length;
    return length;
}

struct TerminalCandidate {
    int parent_node = -1;
    int action_order = -1;
    double cost = std::numeric_limits<double>::infinity();
    PrimitivePropagation propagation;
};

bool terminal_precedes(
    const TerminalCandidate& candidate,
    const TerminalCandidate& incumbent
) {
    if (candidate.cost < incumbent.cost - EPS) return true;
    if (candidate.cost > incumbent.cost + EPS) return false;
    const auto transition_rank = [](const PrimitivePropagation& propagation) {
        if (!propagation.result.transition) {
            return std::array<int, 4> {
                std::numeric_limits<int>::max(),
                std::numeric_limits<int>::max(),
                std::numeric_limits<int>::max(),
                std::numeric_limits<int>::max(),
            };
        }
        const BoundaryTransition& transition = *propagation.result.transition;
        return std::array<int, 4> {
            transition.flat_cell.x(),
            transition.flat_cell.y(),
            transition.body_cell.x(),
            transition.body_cell.y(),
        };
    };
    const auto candidate_rank = transition_rank(candidate.propagation);
    const auto incumbent_rank = transition_rank(incumbent.propagation);
    if (candidate_rank != incumbent_rank) return candidate_rank < incumbent_rank;
    if (candidate.parent_node != incumbent.parent_node) {
        return candidate.parent_node < incumbent.parent_node;
    }
    return candidate.action_order < incumbent.action_order;
}

} // anonymous namespace

MotionPrimitiveLibrary::MotionPrimitiveLibrary(Params params)
    : params_(params),
      primitives_by_heading_(static_cast<size_t>(std::max(params.heading_bins, 0))) {
    if (params_.xy_resolution <= 0.0 || params_.heading_bins <= 0
        || params_.nominal_curvature_samples <= 0
        || params_.nominal_curvature_samples % 2 == 0
        || params_.curvature_max <= 0.0 || params_.primitive_length <= 0.0) {
        error_ = "invalid motion primitive generation parameters";
        return;
    }

    const double heading_resolution = 2.0 * std::numbers::pi
        / static_cast<double>(params_.heading_bins);
    const std::vector<double> nominal_actions = nominal_curvatures(params_);
    struct HeadingGenerationStats {
        double max_position_residual = 0.0;
        double max_heading_residual = 0.0;
        std::string error;
    };
    std::vector<HeadingGenerationStats> stats(
        static_cast<size_t>(params_.heading_bins)
    );
    std::atomic<int> next_heading {0};
    const auto generate_heading = [&](const int heading_bin) {
        HeadingGenerationStats& heading_stats = stats[static_cast<size_t>(heading_bin)];
        const double start_heading = heading_resolution * static_cast<double>(heading_bin);
        std::unordered_map<LatticeKey, MotionPrimitive, LatticeKeyHash> unique;
        for (const double nominal : nominal_actions) {
            SpatialPose nominal_end {{0.0, 0.0}, start_heading};
            nominal_end = advance_pose(nominal_end, nominal, params_.primitive_length);
            const int center_x = static_cast<int>(std::llround(
                nominal_end.position.x() / params_.xy_resolution
            ));
            const int center_y = static_cast<int>(std::llround(
                nominal_end.position.y() / params_.xy_resolution
            ));
            const int center_heading = static_cast<int>(std::llround(
                nominal_end.heading / heading_resolution
            ));

            std::vector<PrimitiveCandidate> candidates;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dh = -1; dh <= 1; ++dh) {
                        const int target_heading_bin = wrap_heading(
                            center_heading + dh, params_.heading_bins
                        );
                        double target_heading = heading_resolution
                            * static_cast<double>(target_heading_bin);
                        target_heading += 2.0 * std::numbers::pi * std::round(
                            (start_heading + nominal * params_.primitive_length
                                - target_heading)
                            / (2.0 * std::numbers::pi)
                        );
                        const LatticeKey endpoint {
                            .x = center_x + dx,
                            .y = center_y + dy,
                            .heading = target_heading_bin,
                        };
                        const Eigen::Vector2d target_position = params_.xy_resolution
                            * Eigen::Vector2d(endpoint.x, endpoint.y);
                        const double progress = target_position.dot(
                            Eigen::Vector2d(std::cos(start_heading), std::sin(start_heading))
                        );
                        if (progress <= params_.xy_resolution * 0.5) continue;
                        candidates.push_back({
                            .endpoint = endpoint,
                            .target_heading = target_heading,
                            .nominal_error = (target_position - nominal_end.position).squaredNorm()
                                + std::pow(target_heading - (
                                    start_heading + nominal * params_.primitive_length
                                ), 2),
                        });
                    }
                }
            }
            std::ranges::sort(candidates, {}, &PrimitiveCandidate::nominal_error);

            for (const PrimitiveCandidate& candidate : candidates) {
                double curvature[3] {nominal, nominal, nominal};
                double length[3] {
                    params_.primitive_length / 3.0,
                    params_.primitive_length / 3.0,
                    params_.primitive_length / 3.0,
                };
                ceres::Problem problem;
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PrimitiveResidual, 10, 3, 3>(
                        new PrimitiveResidual {
                            .start_heading = start_heading,
                            .target_x = candidate.endpoint.x * params_.xy_resolution,
                            .target_y = candidate.endpoint.y * params_.xy_resolution,
                            .target_heading = candidate.target_heading,
                            .nominal_curvature = nominal,
                            .nominal_length = params_.primitive_length,
                        }
                    ),
                    nullptr,
                    curvature,
                    length
                );
                for (int i = 0; i < 3; ++i) {
                    problem.SetParameterLowerBound(curvature, i, -params_.curvature_max);
                    problem.SetParameterUpperBound(curvature, i, params_.curvature_max);
                    problem.SetParameterLowerBound(length, i, params_.primitive_length * 0.05);
                    problem.SetParameterUpperBound(length, i, params_.primitive_length * 0.9);
                }
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 40;
                options.num_threads = 1;
                options.logging_type = ceres::SILENT;
                options.function_tolerance = 1e-10;
                options.gradient_tolerance = 1e-10;
                options.parameter_tolerance = 1e-10;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                if (!summary.IsSolutionUsable()) continue;

                ceres::Problem closure_problem;
                closure_problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PrimitiveClosureResidual, 3, 3>(
                        new PrimitiveClosureResidual {
                            .start_heading = start_heading,
                            .target_x = candidate.endpoint.x * params_.xy_resolution,
                            .target_y = candidate.endpoint.y * params_.xy_resolution,
                            .target_heading = candidate.target_heading,
                            .length = {length[0], length[1], length[2]},
                        }
                    ),
                    nullptr,
                    curvature
                );
                for (int i = 0; i < 3; ++i) {
                    closure_problem.SetParameterLowerBound(
                        curvature, i, -params_.curvature_max
                    );
                    closure_problem.SetParameterUpperBound(
                        curvature, i, params_.curvature_max
                    );
                }
                ceres::Solver::Options closure_options;
                closure_options.linear_solver_type = ceres::DENSE_QR;
                closure_options.max_num_iterations = 60;
                closure_options.num_threads = 1;
                closure_options.logging_type = ceres::SILENT;
                closure_options.function_tolerance = 1e-14;
                closure_options.gradient_tolerance = 1e-14;
                closure_options.parameter_tolerance = 1e-14;
                ceres::Solver::Summary closure_summary;
                ceres::Solve(closure_options, &closure_problem, &closure_summary);
                if (!closure_summary.IsSolutionUsable()) continue;

                SpatialPose endpoint {{0.0, 0.0}, start_heading};
                double total_length = 0.0;
                MotionPrimitive primitive {
                    .endpoint_delta = candidate.endpoint,
                    .segments = {},
                    .nominal_curvature = nominal,
                };
                for (int i = 0; i < 3; ++i) {
                    primitive.segments.push_back({curvature[i], length[i]});
                    endpoint = advance_pose(endpoint, curvature[i], length[i]);
                    total_length += length[i];
                }
                const Eigen::Vector2d target_position = params_.xy_resolution
                    * Eigen::Vector2d(candidate.endpoint.x, candidate.endpoint.y);
                const double position_residual = (endpoint.position - target_position).norm();
                const double heading_residual = std::abs(wrap_angle(
                    endpoint.heading - candidate.target_heading
                ));
                if (position_residual > params_.endpoint_position_tolerance
                    || heading_residual > params_.endpoint_heading_tolerance
                    || total_length < params_.primitive_length * 0.5
                    || total_length > params_.primitive_length * 1.5) {
                    continue;
                }
                const auto incumbent = unique.find(candidate.endpoint);
                const auto score = [](const MotionPrimitive& value) {
                    double result = 0.0;
                    for (const PrimitiveSegment& segment : value.segments) {
                        result += std::pow(
                            segment.curvature - value.nominal_curvature, 2
                        );
                    }
                    return result;
                };
                if (incumbent == unique.end() || score(primitive) < score(incumbent->second)) {
                    unique[candidate.endpoint] = std::move(primitive);
                }
                heading_stats.max_position_residual = std::max(
                    heading_stats.max_position_residual, position_residual
                );
                heading_stats.max_heading_residual = std::max(
                    heading_stats.max_heading_residual, heading_residual
                );
                break;
            }
        }
        auto& output = primitives_by_heading_[static_cast<size_t>(heading_bin)];
        output.reserve(unique.size());
        for (auto& [key, primitive] : unique) {
            static_cast<void>(key);
            output.push_back(std::move(primitive));
        }
        std::ranges::sort(output, [](const MotionPrimitive& lhs, const MotionPrimitive& rhs) {
            if (lhs.endpoint_delta.x != rhs.endpoint_delta.x) {
                return lhs.endpoint_delta.x < rhs.endpoint_delta.x;
            }
            if (lhs.endpoint_delta.y != rhs.endpoint_delta.y) {
                return lhs.endpoint_delta.y < rhs.endpoint_delta.y;
            }
            if (lhs.endpoint_delta.heading != rhs.endpoint_delta.heading) {
                return lhs.endpoint_delta.heading < rhs.endpoint_delta.heading;
            }
            return lhs.nominal_curvature < rhs.nominal_curvature;
        });
        if (output.empty()) {
            heading_stats.error = "motion primitive generation produced no action for heading bin "
                + std::to_string(heading_bin);
        }
    };
    const auto generate_headings = [&] {
        while (true) {
            const int heading_bin = next_heading.fetch_add(1);
            if (heading_bin >= params_.heading_bins) return;
            generate_heading(heading_bin);
        }
    };
    const unsigned int available_threads = std::max(1U, std::thread::hardware_concurrency());
    const size_t worker_count = std::min<size_t>(8, available_threads);
    std::vector<std::jthread> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back(generate_headings);
    }
    workers.clear();
    for (int heading_bin = 0; heading_bin < params_.heading_bins; ++heading_bin) {
        const HeadingGenerationStats& heading_stats = stats[static_cast<size_t>(heading_bin)];
        if (!heading_stats.error.empty()) {
            error_ = heading_stats.error;
            return;
        }
        primitive_count_ += primitives_by_heading_[static_cast<size_t>(heading_bin)].size();
        max_position_residual_ = std::max(
            max_position_residual_, heading_stats.max_position_residual
        );
        max_heading_residual_ = std::max(
            max_heading_residual_, heading_stats.max_heading_residual
        );
    }
}

const std::vector<MotionPrimitive>& MotionPrimitiveLibrary::for_heading(
    const int heading
) const {
    return primitives_by_heading_.at(static_cast<size_t>(
        wrap_heading(heading, params_.heading_bins)
    ));
}

SpatialPose StateLatticeAstar::pose_of(
    const LatticeFrame& frame,
    const LatticeKey& key
) {
    const double c = std::cos(frame.base_heading);
    const double s = std::sin(frame.base_heading);
    const Eigen::Matrix2d rotation = (Eigen::Matrix2d() << c, -s, s, c).finished();
    return {
        .position = frame.origin_map + rotation * (
            frame.xy_resolution * Eigen::Vector2d(key.x, key.y)
        ),
        .heading = wrap_angle(
            frame.base_heading + 2.0 * std::numbers::pi
                * static_cast<double>(wrap_heading(key.heading, frame.heading_bins))
                / static_cast<double>(frame.heading_bins)
        ),
    };
}

StateLatticeAstar::Result StateLatticeAstar::search(
    const LatticeFrame& frame,
    const std::vector<SearchRoot>& roots,
    const SearchTarget& target,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const int occupied_threshold,
    const double detect_dot_threshold,
    const std::vector<Eigen::Vector2d>& reference_path
) const {
    Result result;
    if (!primitive_library_.valid()) {
        result.error = "motion primitive library is invalid: " + primitive_library_.error();
        return result;
    }
    if (roots.empty() || frame.heading_bins <= 0 || frame.xy_resolution <= 0.0
        || !std::isfinite(params_.focal_suboptimality)
        || params_.focal_suboptimality < 1.0) {
        result.error = "state-lattice search configuration or roots are invalid";
        return result;
    }

    const auto heuristic = [&](
        const SpatialPose& pose,
        const SpeedSquaredInterval& reachable_speed
    ) {
        if (const auto* point = std::get_if<PointStopTarget>(&target)) {
            const double stopping_distance = reachable_speed.min
                / (2.0 * params_.dynamics.tangential_acceleration_max);
            return std::max(
                (point->position_map - pose.position).norm(), stopping_distance
            );
        }
        const auto& portal = std::get<PassingPortalTarget>(target).portal;
        double best = std::numeric_limits<double>::infinity();
        const double heading_tolerance = std::acos(std::clamp(
            detect_dot_threshold, 0.0, 1.0
        ));
        for (const BoundaryTransition& transition : portal.transitions) {
            const Eigen::Vector2d terrain_direction = direction_map.at(
                transition.body_cell
            ).normalized() * (portal.direction == StepDirection::UP ? 1.0 : -1.0);
            const double required_heading = std::atan2(
                terrain_direction.y(), terrain_direction.x()
            );
            const double heading_error = std::max(
                0.0,
                std::abs(wrap_angle(pose.heading - required_heading))
                    - heading_tolerance
            );
            const double heading_distance = heading_error / params_.curvature_max;
            const TraversalMode* mode = terrain_constraints.selected_mode(
                direction_map.terrain_at(transition.body_cell),
                portal.direction == StepDirection::UP
            );
            const double acceleration_distance = mode
                ? std::max(
                    0.0,
                    mode->velocity_window.min * mode->velocity_window.min
                        - reachable_speed.max
                ) / (2.0 * params_.dynamics.tangential_acceleration_max)
                : std::numeric_limits<double>::infinity();
            best = std::min(
                best,
                std::max({
                    distance_to_transition(cost_map, pose.position, transition),
                    heading_distance,
                    acceleration_distance,
                })
            );
        }
        return best;
    };

    std::vector<SearchNode> nodes;
    nodes.reserve(4096);
    std::unordered_map<LatticeKey, std::vector<int>, LatticeKeyHash> labels;
    std::unordered_map<LatticeKey, double, LatticeKeyHash> guide_by_key;
    std::priority_queue<AnchorEntry, std::vector<AnchorEntry>, std::greater<>> anchor_open;
    std::priority_queue<AnchorEntry, std::vector<AnchorEntry>, std::greater<>> pending_focal;
    std::priority_queue<GuideEntry, std::vector<GuideEntry>, std::greater<>> focal;
    const ReferencePathGuide reference_guide(reference_path);
    std::optional<TerminalCandidate> best_terminal;
    size_t open_count = 0;
    const auto enqueue = [&](
        const LatticeKey& key,
        const SpeedSquaredInterval& reachable_speed,
        const double g,
        const int parent,
        const int incoming,
        const bool relaxed
    ) {
        auto& key_labels = labels[key];
        for (const int index : key_labels) {
            const SearchNode& incumbent = nodes[static_cast<size_t>(index)];
            if (incumbent.active && incumbent.g <= g + EPS
                && speed_interval_contains(incumbent.speed, reachable_speed)) {
                ++result.dominated_labels;
                return false;
            }
        }
        for (const int index : key_labels) {
            SearchNode& incumbent = nodes[static_cast<size_t>(index)];
            if (incumbent.active && g <= incumbent.g + EPS
                && speed_interval_contains(reachable_speed, incumbent.speed)) {
                incumbent.active = false;
                if (incumbent.open) --open_count;
                ++result.dominated_labels;
            }
        }
        const int index = static_cast<int>(nodes.size());
        const SpatialPose pose = pose_of(frame, key);
        const double anchor = g + heuristic(pose, reachable_speed);
        if (best_terminal
            && anchor >= best_terminal->cost / params_.focal_suboptimality - EPS) {
            return false;
        }
        const auto [guide_iterator, inserted] = guide_by_key.try_emplace(key, 0.0);
        if (inserted) guide_iterator->second = reference_guide.score(pose.position);
        const double guide = guide_iterator->second;
        nodes.push_back({
            .key = key,
            .speed = reachable_speed,
            .g = g,
            .anchor = anchor,
            .guide = guide,
            .parent = parent,
            .incoming_primitive = incoming,
            .relaxed_root = relaxed,
            .active = true,
            .open = true,
            .in_focal = false,
        });
        key_labels.push_back(index);
        anchor_open.push({anchor, index});
        pending_focal.push({anchor, index});
        result.anchor_queue_peak = std::max(result.anchor_queue_peak, anchor_open.size());
        result.pending_focal_queue_peak = std::max(
            result.pending_focal_queue_peak, pending_focal.size()
        );
        ++open_count;
        ++result.generated_labels;
        result.open_peak = std::max(result.open_peak, open_count);
        return true;
    };

    for (const SearchRoot& root : roots) {
        if (root.initial_cost < 0.0 || root.reachable_speed.min < 0.0
            || root.reachable_speed.min > root.reachable_speed.max) {
            result.error = "state-lattice root is invalid";
            return result;
        }
        LatticeKey key = root.key;
        key.heading = wrap_heading(key.heading, frame.heading_bins);
        enqueue(key, root.reachable_speed, root.initial_cost, -1, -1, root.relaxed);
    }

    const auto consider_terminal = [&](TerminalCandidate candidate) {
        if (!best_terminal || terminal_precedes(candidate, *best_terminal)) {
            best_terminal = std::move(candidate);
        }
    };
    const auto node_is_open = [&](const int index) {
        return index >= 0
            && static_cast<size_t>(index) < nodes.size()
            && nodes[static_cast<size_t>(index)].active
            && nodes[static_cast<size_t>(index)].open;
    };
    bool terminal_bound_satisfied = false;
    while (true) {
        while (!anchor_open.empty() && !node_is_open(anchor_open.top().node)) {
            anchor_open.pop();
            ++result.stale_queue_entries;
        }
        if (anchor_open.empty()) {
            terminal_bound_satisfied = best_terminal.has_value();
            break;
        }
        const double minimum_anchor = anchor_open.top().anchor;
        if (best_terminal
            && best_terminal->cost <= params_.focal_suboptimality * minimum_anchor + EPS) {
            terminal_bound_satisfied = true;
            break;
        }
        if (result.expansions >= params_.max_expansions) {
            result.error = "state-lattice expansion limit reached";
            break;
        }
        const double focal_bound = params_.focal_suboptimality * minimum_anchor;
        while (!pending_focal.empty() && pending_focal.top().anchor <= focal_bound + EPS) {
            const AnchorEntry candidate = pending_focal.top();
            pending_focal.pop();
            if (!node_is_open(candidate.node)) {
                ++result.stale_queue_entries;
                continue;
            }
            SearchNode& node = nodes[static_cast<size_t>(candidate.node)];
            if (node.in_focal) continue;
            node.in_focal = true;
            focal.push({node.guide, node.anchor, candidate.node});
            result.focal_queue_peak = std::max(result.focal_queue_peak, focal.size());
        }
        while (!focal.empty() && !node_is_open(focal.top().node)) {
            focal.pop();
            ++result.stale_queue_entries;
        }
        if (focal.empty()) continue;

        const GuideEntry entry = focal.top();
        focal.pop();
        SearchNode& selected = nodes[static_cast<size_t>(entry.node)];
        selected.open = false;
        selected.in_focal = false;
        --open_count;
        if (best_terminal
            && selected.anchor >= best_terminal->cost / params_.focal_suboptimality - EPS) {
            continue;
        }
        const SearchNode current = selected;
        const SpatialPose current_pose = pose_of(frame, current.key);
        ++result.expansions;

        if (const auto* point = std::get_if<PointStopTarget>(&target);
            point && (point->position_map - current_pose.position).norm()
                <= params_.goal_tolerance) {
            ++result.terminal_attempts;
            const auto connection = goal_connection(
                current_pose,
                point->position_map,
                params_.goal_connection_max_length,
                params_.curvature_max
            );
            if (connection) {
                const SpatialPose endpoint = advance_pose(
                    current_pose, connection->curvature, connection->length
                );
                PrimitivePropagation propagated = propagate_segments(
                    current_pose,
                    current.speed,
                    {*connection},
                    endpoint,
                    nullptr,
                    cost_map,
                    direction_map,
                    terrain_constraints,
                    occupied_threshold,
                    detect_dot_threshold,
                    params_.collision_check_resolution,
                    speed_,
                    result.transition_checks,
                    result.rejected_portal_terminals,
                    true
                );
                if (propagated.valid
                    && (propagated.result.pose.position - point->position_map).norm() <= 1e-5) {
                    propagated.result.pose.position = point->position_map;
                    if (!propagated.result.route.edges.empty()) {
                        propagated.result.route.edges.back().to.position = point->position_map;
                    }
                    consider_terminal({
                        .parent_node = entry.node,
                        .action_order = -1,
                        .cost = current.g + route_length(propagated.result.route),
                        .propagation = std::move(propagated),
                    });
                }
            }
        }

        const auto* portal = std::get_if<PassingPortalTarget>(&target);
        const auto& primitives = primitive_library_.for_heading(current.key.heading);
        for (size_t primitive_index = 0; primitive_index < primitives.size(); ++primitive_index) {
            const MotionPrimitive& primitive = primitives[primitive_index];
            LatticeKey successor_key {
                .x = current.key.x + primitive.endpoint_delta.x,
                .y = current.key.y + primitive.endpoint_delta.y,
                .heading = primitive.endpoint_delta.heading,
            };
            const SpatialPose canonical = pose_of(frame, successor_key);
            PrimitivePropagation propagated = propagate_segments(
                current_pose,
                current.speed,
                primitive.segments,
                canonical,
                portal,
                cost_map,
                direction_map,
                terrain_constraints,
                occupied_threshold,
                detect_dot_threshold,
                params_.collision_check_resolution,
                speed_,
                result.transition_checks,
                result.rejected_portal_terminals,
                false
            );
            if (!propagated.valid) continue;
            if (propagated.terminal) {
                ++result.terminal_attempts;
                consider_terminal({
                    .parent_node = entry.node,
                    .action_order = static_cast<int>(primitive_index),
                    .cost = current.g + route_length(propagated.result.route),
                    .propagation = std::move(propagated),
                });
                continue;
            }
            double primitive_length = 0.0;
            for (const PrimitiveSegment& segment : primitive.segments) {
                primitive_length += segment.length;
            }
            enqueue(
                successor_key,
                propagated.result.speed,
                current.g + primitive_length,
                entry.node,
                static_cast<int>(primitive_index),
                false
            );
        }
    }

    if (!terminal_bound_satisfied || !best_terminal) {
        if (result.error.empty()) result.error = "state lattice found no feasible path";
        return result;
    }
    const int selected_node = best_terminal->parent_node;
    result.selected_search_cost = best_terminal->cost;
    PrimitivePropagation selected_terminal = std::move(best_terminal->propagation);

    std::vector<int> chain;
    for (int index = selected_node; index >= 0;
         index = nodes[static_cast<size_t>(index)].parent) {
        chain.push_back(index);
    }
    std::reverse(chain.begin(), chain.end());
    const SearchNode& selected_root = nodes[static_cast<size_t>(chain.front())];
    result.route.start = pose_of(frame, selected_root.key);
    result.selected_relaxed_root = selected_root.relaxed_root;
    result.selected_root_cost = selected_root.g;
    SpeedSquaredInterval replay_speed = selected_root.speed;
    for (size_t i = 1; i < chain.size(); ++i) {
        const SearchNode& parent = nodes[static_cast<size_t>(chain[i - 1])];
        const SearchNode& child = nodes[static_cast<size_t>(chain[i])];
        const MotionPrimitive& primitive = primitive_library_.for_heading(
            parent.key.heading
        ).at(static_cast<size_t>(child.incoming_primitive));
        const PrimitivePropagation replay = propagate_segments(
            pose_of(frame, parent.key),
            replay_speed,
            primitive.segments,
            pose_of(frame, child.key),
            nullptr,
            cost_map,
            direction_map,
            terrain_constraints,
            occupied_threshold,
            detect_dot_threshold,
            params_.collision_check_resolution,
            speed_,
            result.transition_checks,
            result.rejected_portal_terminals,
            false
        );
        if (!replay.valid || replay.terminal) {
            result.error = "state-lattice route replay failed";
            return result;
        }
        result.route.edges.insert(
            result.route.edges.end(),
            replay.result.route.edges.begin(),
            replay.result.route.edges.end()
        );
        replay_speed = replay.result.speed;
    }
    result.route.edges.insert(
        result.route.edges.end(),
        selected_terminal.result.route.edges.begin(),
        selected_terminal.result.route.edges.end()
    );
    result.terminal_speed = selected_terminal.result.speed;
    result.portal_transition = selected_terminal.result.transition;
    result.success = true;
    return result;
}

} // namespace nav_executor
