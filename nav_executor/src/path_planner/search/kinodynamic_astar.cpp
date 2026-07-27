#include <nav_executor/path_planner/search/kinodynamic_astar.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <optional>
#include <queue>
#include <sstream>
#include <unordered_map>

namespace nav_executor {

namespace {

constexpr double EPS = 1e-9;
constexpr double FORWARD_SPEED_EPS = 1e-4;

struct SpeedSquaredInterval {
    double min = 0.0;
    double max = 0.0;
};

struct PoseKey {
    uint64_t packed = 0;
    bool operator==(const PoseKey& other) const { return packed == other.packed; }
};

struct PoseKeyHash {
    size_t operator()(const PoseKey& key) const {
        return std::hash<uint64_t>{}(key.packed);
    }
};

struct SpatialPrimitive {
    double curvature = 0.0;
    double length = 0.0;
};

struct SearchNode {
    KinodynamicAstar::Pose pose;
    SpeedSquaredInterval reachable_speed;
    double g = 0.0;
    int parent = -1;
    SpatialPrimitive incoming;
    bool relaxed_root = false;
    bool active = true;
};

struct OpenEntry {
    double f = 0.0;
    int node_index = -1;
    bool operator>(const OpenEntry& other) const { return f > other.f; }
};

struct PropagationResult {
    KinodynamicAstar::Pose pose;
    SpeedSquaredInterval reachable_speed;
};

struct RouteEdge {
    double curvature = 0.0;
    double length = 0.0;
    KinodynamicAstar::SpeedRange endpoint_speed_range;
    bool terminal_stop = false;
};

enum class SpeedPropagationFailure {
    NONE,
    INVALID_ENDPOINT_RANGE,
    INITIAL_DYNAMICALLY_INFEASIBLE,
    EMPTY_REACHABLE_INTERVAL,
    ENDPOINT_RANGE_UNREACHABLE,
    TERMINAL_STOP_UNREACHABLE,
};

struct SpeedPropagationTrace {
    SpeedPropagationFailure failure = SpeedPropagationFailure::NONE;
    double dynamic_speed_squared_cap = 0.0;
    double endpoint_speed_squared_cap = 0.0;
    SpeedSquaredInterval admissible_initial;
    SpeedSquaredInterval dynamically_reachable;
    SpeedSquaredInterval endpoint_allowed;
};

double wrap_positive(double angle) {
    angle = std::fmod(angle, 2.0 * M_PI);
    if (angle < 0.0) angle += 2.0 * M_PI;
    return angle;
}

int64_t quantize_component(const double value, const double resolution) {
    return static_cast<int64_t>(std::llround(value / std::max(resolution, EPS)));
}

PoseKey make_key(const KinodynamicAstar::Pose& pose, const KinodynamicAstar::Params& params) {
    const int64_t ix = quantize_component(pose.position.x(), params.dedup_xy);
    const int64_t iy = quantize_component(pose.position.y(), params.dedup_xy);
    const int64_t theta_bins = std::max<int64_t>(
        1, std::llround(2.0 * M_PI / std::max(params.dedup_theta, EPS))
    );
    const int64_t it = quantize_component(
        wrap_positive(pose.theta), params.dedup_theta
    ) % theta_bins;
    return {
        .packed = (static_cast<uint64_t>(ix) & 0xFFFFFFULL) << 40
            | (static_cast<uint64_t>(iy) & 0xFFFFFFULL) << 16
            | (static_cast<uint64_t>(it) & 0xFFFFULL),
    };
}

std::optional<SpeedSquaredInterval> intersect_intervals(
    const SpeedSquaredInterval& lhs,
    const SpeedSquaredInterval& rhs
) {
    SpeedSquaredInterval result {
        .min = std::max(lhs.min, rhs.min),
        .max = std::min(lhs.max, rhs.max),
    };
    if (result.min > result.max + EPS) return std::nullopt;
    if (result.min > result.max) result.min = result.max;
    return result;
}

bool contains_interval(
    const SpeedSquaredInterval& outer,
    const SpeedSquaredInterval& inner
) {
    return outer.min <= inner.min + EPS && outer.max + EPS >= inner.max;
}

bool contains_value(const SpeedSquaredInterval& interval, const double value) {
    return value >= interval.min - EPS && value <= interval.max + EPS;
}

const char* propagation_failure_string(const SpeedPropagationFailure failure) {
    switch (failure) {
        case SpeedPropagationFailure::NONE:
            return "none";
        case SpeedPropagationFailure::INVALID_ENDPOINT_RANGE:
            return "invalid endpoint speed range";
        case SpeedPropagationFailure::INITIAL_DYNAMICALLY_INFEASIBLE:
            return "input speed exceeds curvature-dependent dynamic cap";
        case SpeedPropagationFailure::EMPTY_REACHABLE_INTERVAL:
            return "spatial dynamics produced an empty reachable interval";
        case SpeedPropagationFailure::ENDPOINT_RANGE_UNREACHABLE:
            return "reachable interval does not intersect endpoint speed range";
        case SpeedPropagationFailure::TERMINAL_STOP_UNREACHABLE:
            return "zero terminal speed is unreachable over this edge";
    }
    return "unknown";
}

std::string speed_interval_string(const SpeedSquaredInterval& interval) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3)
           << '[' << std::sqrt(std::max(interval.min, 0.0))
           << ',' << std::sqrt(std::max(interval.max, 0.0)) << "] m/s";
    return stream.str();
}

std::string speed_range_string(const KinodynamicAstar::SpeedRange& range) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3)
           << '[' << range.min << ',' << range.max << "] m/s";
    return stream.str();
}

KinodynamicAstar::Pose advance_pose(
    const KinodynamicAstar::Pose& from,
    const double curvature,
    const double length
) {
    KinodynamicAstar::Pose to = from;
    const Eigen::Vector2d tangent(std::cos(from.theta), std::sin(from.theta));
    const Eigen::Vector2d normal(-tangent.y(), tangent.x());
    if (std::abs(curvature) <= EPS) {
        to.position += length * tangent;
        return to;
    }
    const double angle = curvature * length;
    to.position += std::sin(angle) / curvature * tangent
        + (1.0 - std::cos(angle)) / curvature * normal;
    to.theta = std::atan2(std::sin(from.theta + angle), std::cos(from.theta + angle));
    return to;
}

double dynamic_speed_squared_cap(
    const double curvature,
    const ShapingDynamicsLimits& limits
) {
    double cap = limits.velocity_max * limits.velocity_max;
    const double abs_curvature = std::abs(curvature);
    if (abs_curvature <= EPS) return cap;
    cap = std::min(
        cap,
        limits.angular_velocity_max * limits.angular_velocity_max
            / (abs_curvature * abs_curvature)
    );
    cap = std::min(cap, limits.lateral_acceleration_max / abs_curvature);
    return std::max(cap, 0.0);
}

double tangential_acceleration_cap(
    const double curvature,
    const ShapingDynamicsLimits& limits
) {
    double cap = limits.tangential_acceleration_max;
    const double abs_curvature = std::abs(curvature);
    if (abs_curvature > EPS) {
        // 恒曲率原语内 alpha=kappa*a_t；曲率跳变由后续连续 MINCO 负责平滑。
        cap = std::min(cap, limits.angular_acceleration_max / abs_curvature);
    }
    return cap;
}

// 标量速度的切向加速度传播：z=v²，dz/ds=2a_t。
double propagate_speed_extreme(
    const double initial_speed_squared,
    const double length,
    const double tangential_acceleration_max,
    const double speed_squared_cap,
    const bool accelerate
) {
    const double initial = std::clamp(initial_speed_squared, 0.0, speed_squared_cap);
    return std::clamp(
        initial + (accelerate ? 1.0 : -1.0)
            * 2.0 * tangential_acceleration_max * length,
        0.0,
        speed_squared_cap
    );
}

std::optional<SpeedSquaredInterval> propagate_speed_interval(
    const SpeedSquaredInterval& initial,
    const double length,
    const double curvature,
    const KinodynamicAstar::SpeedRange& environmental_speed,
    const ShapingDynamicsLimits& limits,
    const bool terminal_stop,
    SpeedPropagationTrace* const trace = nullptr
) {
    const double dynamic_cap = dynamic_speed_squared_cap(curvature, limits);
    const double acceleration_cap = tangential_acceleration_cap(curvature, limits);
    const double environmental_max = environmental_speed.max;
    const double endpoint_cap = std::min(
        dynamic_cap, std::max(environmental_max, 0.0) * std::max(environmental_max, 0.0)
    );
    if (trace) {
        trace->dynamic_speed_squared_cap = dynamic_cap;
        trace->endpoint_speed_squared_cap = endpoint_cap;
    }
    if (!std::isfinite(environmental_speed.min)
        || !std::isfinite(environmental_speed.max)
        || environmental_speed.min > environmental_speed.max
        || environmental_speed.max < 0.0
        || (endpoint_cap <= 0.0 && !terminal_stop)) {
        if (trace) trace->failure = SpeedPropagationFailure::INVALID_ENDPOINT_RANGE;
        return std::nullopt;
    }

    const auto admissible_initial = intersect_intervals(
        initial, SpeedSquaredInterval {0.0, dynamic_cap}
    );
    if (!admissible_initial) {
        if (trace) trace->failure = SpeedPropagationFailure::INITIAL_DYNAMICALLY_INFEASIBLE;
        return std::nullopt;
    }
    if (trace) trace->admissible_initial = *admissible_initial;

    SpeedSquaredInterval reachable {
        .min = propagate_speed_extreme(
            admissible_initial->min,
            length,
            acceleration_cap,
            dynamic_cap,
            false
        ),
        .max = propagate_speed_extreme(
            admissible_initial->max,
            length,
            acceleration_cap,
            dynamic_cap,
            true
        ),
    };
    if (trace) trace->dynamically_reachable = reachable;
    if (reachable.min > reachable.max + EPS) {
        if (trace) trace->failure = SpeedPropagationFailure::EMPTY_REACHABLE_INTERVAL;
        return std::nullopt;
    }

    const double minimum_speed = terminal_stop
        ? 0.0 : std::max(environmental_speed.min, FORWARD_SPEED_EPS);
    const SpeedSquaredInterval allowed {
        .min = minimum_speed * minimum_speed,
        .max = endpoint_cap,
    };
    if (trace) trace->endpoint_allowed = allowed;
    auto intersection = intersect_intervals(reachable, allowed);
    if (!intersection) {
        if (trace) trace->failure = SpeedPropagationFailure::ENDPOINT_RANGE_UNREACHABLE;
        return std::nullopt;
    }
    if (!terminal_stop) return intersection;
    if (!contains_value(*intersection, 0.0)) {
        if (trace) trace->failure = SpeedPropagationFailure::TERMINAL_STOP_UNREACHABLE;
        return std::nullopt;
    }
    return SpeedSquaredInterval {0.0, 0.0};
}

std::optional<PropagationResult> propagate_primitive(
    const KinodynamicAstar::Pose& start_pose,
    const SpeedSquaredInterval& start_speed,
    const SpatialPrimitive& primitive,
    const KinodynamicAstar::Params& params,
    const KinodynamicAstar::TransitionConstraintFn& transition_constraint,
    KinodynamicAstar::Result& diagnostics,
    const bool terminal_stop
) {
    const int substeps = std::max(
        1,
        static_cast<int>(std::ceil(
            primitive.length / params.collision_check_resolution
        ))
    );
    const double substep_length = primitive.length / static_cast<double>(substeps);
    KinodynamicAstar::Pose pose = start_pose;
    SpeedSquaredInterval speed = start_speed;
    for (int substep = 0; substep < substeps; ++substep) {
        const KinodynamicAstar::Pose previous = pose;
        pose = advance_pose(previous, primitive.curvature, substep_length);
        ++diagnostics.transition_checks;
        const auto environmental_speed = transition_constraint(previous, pose);
        if (!environmental_speed) return std::nullopt;
        const bool final_substep = terminal_stop && substep + 1 == substeps;
        const auto propagated = propagate_speed_interval(
            speed,
            substep_length,
            primitive.curvature,
            *environmental_speed,
            params.dynamics,
            final_substep
        );
        if (!propagated) return std::nullopt;
        speed = *propagated;
    }
    return PropagationResult {.pose = pose, .reachable_speed = speed};
}

std::optional<SpatialPrimitive> make_goal_connection(
    const KinodynamicAstar::Pose& start,
    const Eigen::Vector2d& goal,
    const KinodynamicAstar::Params& params
) {
    const Eigen::Vector2d displacement = goal - start.position;
    const double distance = displacement.norm();
    if (distance <= EPS) return std::nullopt;

    const Eigen::Vector2d tangent(std::cos(start.theta), std::sin(start.theta));
    const Eigen::Vector2d normal(-tangent.y(), tangent.x());
    const double local_x = displacement.dot(tangent);
    const double local_y = displacement.dot(normal);

    SpatialPrimitive connection;
    if (std::abs(local_y) <= 1e-8) {
        if (local_x <= 0.0) return std::nullopt;
        connection.length = local_x;
    } else {
        connection.curvature = 2.0 * local_y / displacement.squaredNorm();
        const double heading_change = 2.0 * std::atan2(local_y, local_x);
        connection.length = heading_change / connection.curvature;
    }
    if (!std::isfinite(connection.length) || connection.length <= EPS
        || connection.length > params.goal_connection_max_length
        || std::abs(connection.curvature) > params.curvature_max + EPS) {
        return std::nullopt;
    }
    return connection;
}

std::vector<double> curvature_samples(const KinodynamicAstar::Params& params) {
    if (params.curvature_samples <= 1) return {0.0};
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(params.curvature_samples));
    for (int index = 0; index < params.curvature_samples; ++index) {
        const double fraction = static_cast<double>(index)
            / static_cast<double>(params.curvature_samples - 1);
        const double centered = 2.0 * fraction - 1.0;
        samples.push_back(params.curvature_max * centered * centered * centered);
    }
    return samples;
}

std::optional<SpeedSquaredInterval> predecessor_interval(
    const SpeedSquaredInterval& target,
    const RouteEdge& edge,
    const ShapingDynamicsLimits& limits
) {
    const double cap = dynamic_speed_squared_cap(edge.curvature, limits);
    const double acceleration_cap = tangential_acceleration_cap(edge.curvature, limits);
    SpeedSquaredInterval result {
        .min = propagate_speed_extreme(
            target.min, edge.length, acceleration_cap, cap, false
        ),
        .max = propagate_speed_extreme(
            target.max, edge.length, acceleration_cap, cap, true
        ),
    };
    if (result.min > result.max + EPS) return std::nullopt;
    return result;
}

std::string route_edge_context(
    const size_t edge_index,
    const size_t edge_count,
    const KinodynamicAstar::Pose& from,
    const KinodynamicAstar::Pose& to,
    const RouteEdge& edge
) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3)
           << "edge=" << edge_index << '/' << edge_count
           << " from=(" << from.position.x() << ',' << from.position.y() << ')'
           << " to=(" << to.position.x() << ',' << to.position.y() << ')'
           << " length=" << edge.length << " m"
           << " curvature=" << edge.curvature << " 1/m"
           << " endpoint_range=" << speed_range_string(edge.endpoint_speed_range)
           << " terminal=" << edge.terminal_stop;
    return stream.str();
}

std::string propagation_trace_string(const SpeedPropagationTrace& trace) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3)
           << "reason=" << propagation_failure_string(trace.failure)
           << " dynamic_cap="
           << std::sqrt(std::max(trace.dynamic_speed_squared_cap, 0.0)) << " m/s"
           << " endpoint_cap="
           << std::sqrt(std::max(trace.endpoint_speed_squared_cap, 0.0)) << " m/s"
           << " admissible_input=" << speed_interval_string(trace.admissible_initial)
           << " dynamic_reachable=" << speed_interval_string(trace.dynamically_reachable)
           << " endpoint_allowed=" << speed_interval_string(trace.endpoint_allowed);
    return stream.str();
}

} // anonymous namespace

KinodynamicAstar::Result KinodynamicAstar::search(
    const std::vector<SearchRoot>& roots,
    const Eigen::Vector2d& goal_position,
    const DijkstraCostToGoal& dijkstra,
    const TransitionConstraintFn& transition_constraint
) const {
    Result result;
    if (!dijkstra.ready()) {
        result.error = "Dijkstra field not built";
        return result;
    }
    if (roots.empty()) {
        result.error = "no search roots provided";
        return result;
    }
    for (const SearchRoot& root : roots) {
        const double speed = root.state.velocity.norm();
        if (!root.state.position.allFinite() || !root.state.velocity.allFinite()
            || !std::isfinite(root.initial_cost) || root.initial_cost < 0.0
            || speed < FORWARD_SPEED_EPS
            || speed > params_.dynamics.velocity_max + EPS) {
            result.error = "search root is not a feasible forward state";
            return result;
        }
    }

    const auto heuristic = [&](const Pose& pose) {
        const double cost = dijkstra.at_map(pose.position);
        return std::isinf(cost)
            ? DijkstraCostToGoal::UNREACHABLE
            : params_.heuristic_weight * cost;
    };

    std::vector<SearchNode> nodes;
    nodes.reserve(4096);
    std::unordered_map<PoseKey, std::vector<int>, PoseKeyHash> labels_by_pose;
    labels_by_pose.reserve(4096);
    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<>> open;

    const auto enqueue_label = [&nodes, &labels_by_pose, &open, &result, this](
        const Pose& pose,
        const SpeedSquaredInterval& reachable_speed,
        const double g,
        const double h,
        const int parent,
        const SpatialPrimitive& incoming,
        const bool relaxed_root
    ) {
        const PoseKey key = make_key(pose, params_);
        auto& pose_labels = labels_by_pose[key];
        for (const int label_index : pose_labels) {
            const SearchNode& incumbent = nodes[static_cast<size_t>(label_index)];
            if (incumbent.active && incumbent.g <= g + EPS
                && contains_interval(incumbent.reachable_speed, reachable_speed)) {
                ++result.dominated_labels;
                return false;
            }
        }
        for (const int label_index : pose_labels) {
            SearchNode& incumbent = nodes[static_cast<size_t>(label_index)];
            if (incumbent.active && g <= incumbent.g + EPS
                && contains_interval(reachable_speed, incumbent.reachable_speed)) {
                incumbent.active = false;
                ++result.dominated_labels;
            }
        }

        const int node_index = static_cast<int>(nodes.size());
        nodes.push_back({
            .pose = pose,
            .reachable_speed = reachable_speed,
            .g = g,
            .parent = parent,
            .incoming = incoming,
            .relaxed_root = relaxed_root,
        });
        pose_labels.push_back(node_index);
        open.push({g + h, node_index});
        ++result.generated_labels;
        result.open_peak = std::max(result.open_peak, open.size());
        return true;
    };

    for (const SearchRoot& root : roots) {
        const double speed = root.state.velocity.norm();
        const Pose pose {
            .position = root.state.position,
            .theta = std::atan2(root.state.velocity.y(), root.state.velocity.x()),
        };
        const SpeedSquaredInterval speed_interval {
            speed * speed, speed * speed,
        };
        enqueue_label(
            pose,
            speed_interval,
            root.initial_cost,
            heuristic(pose),
            -1,
            {},
            root.relaxed
        );
    }
    if (open.empty()) {
        result.error = "all search roots were dominated";
        return result;
    }

    const std::vector<double> curvatures = curvature_samples(params_);
    int goal_node_index = -1;
    SpatialPrimitive goal_connection;

    while (!open.empty()) {
        if (result.expansions >= params_.max_expansions) {
            result.error = "expansion limit reached";
            break;
        }
        const OpenEntry entry = open.top();
        open.pop();
        if (entry.node_index < 0
            || !nodes[static_cast<size_t>(entry.node_index)].active) {
            continue;
        }
        const SearchNode current = nodes[static_cast<size_t>(entry.node_index)];
        ++result.expansions;

        if ((goal_position - current.pose.position).norm() <= params_.goal_tolerance) {
            ++result.goal_connection_attempts;
            const auto connection = make_goal_connection(
                current.pose, goal_position, params_
            );
            if (connection) {
                const auto propagated = propagate_primitive(
                    current.pose,
                    current.reachable_speed,
                    *connection,
                    params_,
                    transition_constraint,
                    result,
                    true
                );
                if (propagated
                    && (propagated->pose.position - goal_position).norm() <= 1e-6) {
                    goal_node_index = entry.node_index;
                    goal_connection = *connection;
                    result.success = true;
                    break;
                }
            }
        }

        for (const double curvature : curvatures) {
            const SpatialPrimitive primitive {
                .curvature = curvature,
                .length = params_.primitive_length,
            };
            const auto propagated = propagate_primitive(
                current.pose,
                current.reachable_speed,
                primitive,
                params_,
                transition_constraint,
                result,
                false
            );
            if (!propagated) continue;
            const double h = heuristic(propagated->pose);
            if (std::isinf(h)) continue;

            const double new_g = current.g + primitive.length;
            enqueue_label(
                propagated->pose,
                propagated->reachable_speed,
                new_g,
                h,
                entry.node_index,
                primitive,
                false
            );
        }
    }

    if (goal_node_index < 0) {
        if (result.error.empty()) result.error = "no feasible spatial-kinodynamic path";
        result.success = false;
        return result;
    }

    std::vector<int> reversed_indices;
    for (int index = goal_node_index; index >= 0;
         index = nodes[static_cast<size_t>(index)].parent) {
        reversed_indices.push_back(index);
    }
    std::reverse(reversed_indices.begin(), reversed_indices.end());

    const SearchNode& selected_root = nodes[static_cast<size_t>(reversed_indices.front())];
    const Pose start_pose = selected_root.pose;
    const SpeedSquaredInterval start_interval = selected_root.reachable_speed;
    const double start_speed = std::sqrt(std::max(start_interval.min, 0.0));
    result.selected_relaxed_root = selected_root.relaxed_root;
    result.selected_root_cost = selected_root.g;

    std::vector<SpatialPrimitive> primitives;
    primitives.reserve(reversed_indices.size());
    for (size_t i = 1; i < reversed_indices.size(); ++i) {
        primitives.push_back(
            nodes[static_cast<size_t>(reversed_indices[i])].incoming
        );
    }
    primitives.push_back(goal_connection);

    std::vector<Pose> route_poses {start_pose};
    std::vector<RouteEdge> route_edges;
    std::vector<SpeedSquaredInterval> forward_intervals {start_interval};
    Pose replay_pose = start_pose;
    SpeedSquaredInterval replay_speed = start_interval;
    bool reconstruction_failed = false;
    for (size_t primitive_index = 0;
         primitive_index < primitives.size() && !reconstruction_failed;
         ++primitive_index) {
        const SpatialPrimitive& primitive = primitives[primitive_index];
        const int substeps = std::max(
            1,
            static_cast<int>(std::ceil(
                primitive.length / params_.collision_check_resolution
            ))
        );
        const double substep_length = primitive.length / static_cast<double>(substeps);
        for (int substep = 0; substep < substeps; ++substep) {
            const Pose previous = replay_pose;
            replay_pose = advance_pose(previous, primitive.curvature, substep_length);
            const auto environmental_speed = transition_constraint(previous, replay_pose);
            const bool terminal_stop = primitive_index + 1 == primitives.size()
                && substep + 1 == substeps;
            if (!environmental_speed) {
                reconstruction_failed = true;
                break;
            }
            const auto propagated = propagate_speed_interval(
                replay_speed,
                substep_length,
                primitive.curvature,
                *environmental_speed,
                params_.dynamics,
                terminal_stop
            );
            if (!propagated) {
                reconstruction_failed = true;
                break;
            }
            replay_speed = *propagated;
            route_edges.push_back({
                .curvature = primitive.curvature,
                .length = substep_length,
                .endpoint_speed_range = *environmental_speed,
                .terminal_stop = terminal_stop,
            });
            route_poses.push_back(replay_pose);
            forward_intervals.push_back(replay_speed);
        }
    }
    if (reconstruction_failed || route_edges.empty()) {
        result.success = false;
        result.error = "failed to reconstruct spatial-kinodynamic route";
        return result;
    }
    route_poses.back().position = goal_position;

    const size_t sample_count = route_poses.size();
    std::vector<SpeedSquaredInterval> backward_intervals(sample_count);
    backward_intervals.back() = {0.0, 0.0};
    for (size_t reverse_index = sample_count - 1; reverse_index > 0; --reverse_index) {
        const size_t edge_index = reverse_index - 1;
        const auto predecessor = predecessor_interval(
            backward_intervals[reverse_index],
            route_edges[edge_index],
            params_.dynamics
        );
        if (!predecessor) {
            result.success = false;
            result.error = "backward speed reachability failed: "
                + route_edge_context(
                    edge_index,
                    route_edges.size(),
                    route_poses[edge_index],
                    route_poses[edge_index + 1],
                    route_edges[edge_index]
                )
                + " target=" + speed_interval_string(backward_intervals[reverse_index]);
            return result;
        }
        const auto feasible = intersect_intervals(
            *predecessor, forward_intervals[edge_index]
        );
        if (!feasible) {
            result.success = false;
            result.error = "forward/backward speed envelopes do not intersect: "
                + route_edge_context(
                    edge_index,
                    route_edges.size(),
                    route_poses[edge_index],
                    route_poses[edge_index + 1],
                    route_edges[edge_index]
                )
                + " forward=" + speed_interval_string(forward_intervals[edge_index])
                + " predecessor=" + speed_interval_string(*predecessor)
                + " target=" + speed_interval_string(backward_intervals[reverse_index]);
            return result;
        }
        backward_intervals[edge_index] = *feasible;
    }
    if (!contains_value(backward_intervals.front(), start_speed * start_speed)) {
        result.success = false;
        result.error = "selected route cannot preserve the requested start speed";
        return result;
    }

    std::vector<double> selected_speed_squared(sample_count, 0.0);
    selected_speed_squared.front() = start_speed * start_speed;
    for (size_t edge_index = 0; edge_index < route_edges.size(); ++edge_index) {
        const RouteEdge& edge = route_edges[edge_index];
        SpeedPropagationTrace trace;
        auto reachable = propagate_speed_interval(
            {selected_speed_squared[edge_index], selected_speed_squared[edge_index]},
            edge.length,
            edge.curvature,
            edge.endpoint_speed_range,
            params_.dynamics,
            edge.terminal_stop,
            &trace
        );
        if (!reachable) {
            result.success = false;
            result.error = "speed profile reconstruction failed: "
                + route_edge_context(
                    edge_index,
                    route_edges.size(),
                    route_poses[edge_index],
                    route_poses[edge_index + 1],
                    edge
                )
                + " selected_input=" + speed_interval_string({
                    selected_speed_squared[edge_index],
                    selected_speed_squared[edge_index],
                })
                + " forward=" + speed_interval_string(forward_intervals[edge_index])
                + " backward_current=" + speed_interval_string(
                    backward_intervals[edge_index]
                )
                + " backward_next=" + speed_interval_string(
                    backward_intervals[edge_index + 1]
                )
                + ' ' + propagation_trace_string(trace);
            return result;
        }
        reachable = intersect_intervals(*reachable, backward_intervals[edge_index + 1]);
        if (!reachable) {
            result.success = false;
            result.error = "speed profile has no backward-feasible successor: "
                + route_edge_context(
                    edge_index,
                    route_edges.size(),
                    route_poses[edge_index],
                    route_poses[edge_index + 1],
                    edge
                )
                + " selected_input=" + speed_interval_string({
                    selected_speed_squared[edge_index],
                    selected_speed_squared[edge_index],
                })
                + " dynamic_endpoint=" + speed_interval_string(trace.dynamically_reachable)
                + " endpoint_allowed=" + speed_interval_string(trace.endpoint_allowed)
                + " backward_next=" + speed_interval_string(
                    backward_intervals[edge_index + 1]
                );
            return result;
        }
        selected_speed_squared[edge_index + 1] = reachable->max;
    }
    selected_speed_squared.back() = 0.0;

    result.witness_states.reserve(sample_count);
    result.witness_durations.reserve(route_edges.size());
    for (size_t sample = 0; sample < sample_count; ++sample) {
        const double speed = std::sqrt(std::max(selected_speed_squared[sample], 0.0));
        result.witness_states.push_back({
            .position = route_poses[sample].position,
            .velocity = speed * Eigen::Vector2d(
                std::cos(route_poses[sample].theta),
                std::sin(route_poses[sample].theta)
            ),
        });
        if (sample == 0) continue;
        const double previous_speed = std::sqrt(
            std::max(selected_speed_squared[sample - 1], 0.0)
        );
        const double speed_sum = previous_speed + speed;
        if (speed_sum <= FORWARD_SPEED_EPS) {
            result.success = false;
            result.error = "speed profile contains a zero-speed spatial edge";
            result.witness_states.clear();
            result.witness_durations.clear();
            return result;
        }
        result.witness_durations.push_back(
            2.0 * route_edges[sample - 1].length / speed_sum
        );
    }
    return result;
}

} // namespace nav_executor
