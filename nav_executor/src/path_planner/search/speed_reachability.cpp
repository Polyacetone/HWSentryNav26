#include <nav_executor/path_planner/search/speed_reachability.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace nav_executor {

namespace {

constexpr double EPS = 1e-9;
constexpr double FORWARD_SPEED_EPS = 1e-4;

bool valid_interval(const SpeedSquaredInterval& interval) {
    return std::isfinite(interval.min) && std::isfinite(interval.max)
        && interval.min >= 0.0 && interval.min <= interval.max + EPS;
}

double propagate_extreme(
    const double value,
    const double length,
    const double acceleration,
    const double cap,
    const bool accelerate
) {
    return std::clamp(
        value + (accelerate ? 1.0 : -1.0) * 2.0 * acceleration * length,
        0.0,
        cap
    );
}

} // anonymous namespace

std::optional<SpeedSquaredInterval> intersect_speed_intervals(
    const SpeedSquaredInterval& lhs,
    const SpeedSquaredInterval& rhs
) {
    SpeedSquaredInterval intersection {
        .min = std::max(lhs.min, rhs.min),
        .max = std::min(lhs.max, rhs.max),
    };
    if (intersection.min > intersection.max + EPS) return std::nullopt;
    if (intersection.min > intersection.max) intersection.min = intersection.max;
    return intersection;
}

bool speed_interval_contains(
    const SpeedSquaredInterval& outer,
    const SpeedSquaredInterval& inner
) {
    return outer.min <= inner.min + EPS && outer.max + EPS >= inner.max;
}

SpeedSquaredInterval SpeedReachability::unrestricted_limit() const {
    return {0.0, limits_.velocity_max * limits_.velocity_max};
}

double SpeedReachability::dynamic_speed_squared_cap(const double curvature) const {
    double cap = limits_.velocity_max * limits_.velocity_max;
    const double abs_curvature = std::abs(curvature);
    if (abs_curvature <= EPS) return cap;
    cap = std::min(
        cap,
        limits_.angular_velocity_max * limits_.angular_velocity_max
            / (abs_curvature * abs_curvature)
    );
    cap = std::min(cap, limits_.lateral_acceleration_max / abs_curvature);
    return std::max(cap, 0.0);
}

double SpeedReachability::tangential_acceleration_cap(const double curvature) const {
    double cap = limits_.tangential_acceleration_max;
    if (std::abs(curvature) > EPS) {
        cap = std::min(cap, limits_.angular_acceleration_max / std::abs(curvature));
    }
    return cap;
}

std::optional<SpeedSquaredInterval> SpeedReachability::propagate(
    const SpeedSquaredInterval& initial,
    const double length,
    const double curvature,
    const SpeedSquaredInterval& endpoint_speed_limit,
    const bool terminal_stop
) const {
    if (!valid_interval(initial) || !valid_interval(endpoint_speed_limit)
        || !std::isfinite(length) || length <= EPS || !std::isfinite(curvature)) {
        return std::nullopt;
    }

    const double dynamic_cap = dynamic_speed_squared_cap(curvature);
    const auto admissible_initial = intersect_speed_intervals(initial, {0.0, dynamic_cap});
    if (!admissible_initial) return std::nullopt;

    const double acceleration = tangential_acceleration_cap(curvature);
    SpeedSquaredInterval reachable {
        .min = propagate_extreme(
            admissible_initial->min, length, acceleration, dynamic_cap, false
        ),
        .max = propagate_extreme(
            admissible_initial->max, length, acceleration, dynamic_cap, true
        ),
    };
    SpeedSquaredInterval allowed {
        .min = terminal_stop
            ? 0.0 : std::max(endpoint_speed_limit.min, FORWARD_SPEED_EPS * FORWARD_SPEED_EPS),
        .max = std::min(endpoint_speed_limit.max, dynamic_cap),
    };
    auto result = intersect_speed_intervals(reachable, allowed);
    if (!result) return std::nullopt;
    if (!terminal_stop) return result;
    if (result->min > EPS || result->max < -EPS) return std::nullopt;
    return SpeedSquaredInterval {0.0, 0.0};
}

std::optional<SpeedSquaredInterval> SpeedReachability::propagate_route(
    const SpeedSquaredInterval& initial,
    const SpatialRoute& route
) const {
    SpeedSquaredInterval interval = initial;
    for (const SpatialRouteEdge& edge : route.edges) {
        const auto next = propagate(
            interval,
            edge.length,
            edge.curvature,
            edge.endpoint_speed_limit,
            edge.terminal_stop
        );
        if (!next) return std::nullopt;
        interval = *next;
    }
    return interval;
}

std::optional<SpeedSquaredInterval> SpeedReachability::predecessor(
    const SpeedSquaredInterval& target,
    const SpatialRouteEdge& edge
) const {
    const double cap = dynamic_speed_squared_cap(edge.curvature);
    SpeedSquaredInterval endpoint_allowed = edge.endpoint_speed_limit;
    endpoint_allowed.max = std::min(endpoint_allowed.max, cap);
    if (!edge.terminal_stop) {
        endpoint_allowed.min = std::max(
            endpoint_allowed.min, FORWARD_SPEED_EPS * FORWARD_SPEED_EPS
        );
    } else {
        endpoint_allowed.min = 0.0;
    }
    const auto admissible_target = intersect_speed_intervals(target, endpoint_allowed);
    if (!admissible_target) return std::nullopt;

    const double delta = 2.0 * tangential_acceleration_cap(edge.curvature) * edge.length;
    return SpeedSquaredInterval {
        .min = std::max(0.0, admissible_target->min - delta),
        .max = std::min(cap, admissible_target->max + delta),
    };
}

std::optional<SpeedWitness> SpeedReachability::reconstruct_witness(
    const SpeedSquaredInterval& initial,
    const SpatialRoute& route,
    std::string& error
) const {
    error.clear();
    if (route.edges.empty() || !valid_interval(initial)) {
        error = "speed witness route is empty or has an invalid root interval";
        return std::nullopt;
    }

    const size_t state_count = route.edges.size() + 1;
    std::vector<SpeedSquaredInterval> forward(state_count);
    forward.front() = initial;
    for (size_t i = 0; i < route.edges.size(); ++i) {
        const auto next = propagate(
            forward[i],
            route.edges[i].length,
            route.edges[i].curvature,
            route.edges[i].endpoint_speed_limit,
            route.edges[i].terminal_stop
        );
        if (!next) {
            error = "forward speed reachability failed at edge " + std::to_string(i);
            return std::nullopt;
        }
        forward[i + 1] = *next;
    }

    std::vector<SpeedSquaredInterval> backward(state_count);
    backward.back() = route.edges.back().terminal_stop
        ? SpeedSquaredInterval {0.0, 0.0}
        : forward.back();
    for (size_t reverse = route.edges.size(); reverse > 0; --reverse) {
        const size_t edge_index = reverse - 1;
        const auto predecessor_interval = predecessor(
            backward[edge_index + 1], route.edges[edge_index]
        );
        if (!predecessor_interval) {
            error = "backward speed reachability failed at edge "
                + std::to_string(edge_index);
            return std::nullopt;
        }
        const auto feasible = intersect_speed_intervals(
            *predecessor_interval, forward[edge_index]
        );
        if (!feasible) {
            error = "forward/backward speed envelopes do not intersect at edge "
                + std::to_string(edge_index);
            return std::nullopt;
        }
        backward[edge_index] = *feasible;
    }

    const auto feasible_initial = intersect_speed_intervals(initial, backward.front());
    if (!feasible_initial) {
        error = "route cannot preserve the requested root speed interval";
        return std::nullopt;
    }

    std::vector<double> selected(state_count, 0.0);
    selected.front() = feasible_initial->max;
    for (size_t i = 0; i < route.edges.size(); ++i) {
        auto reachable = propagate(
            {selected[i], selected[i]},
            route.edges[i].length,
            route.edges[i].curvature,
            route.edges[i].endpoint_speed_limit,
            route.edges[i].terminal_stop
        );
        if (!reachable) {
            error = "scalar witness propagation failed at edge " + std::to_string(i);
            return std::nullopt;
        }
        reachable = intersect_speed_intervals(*reachable, backward[i + 1]);
        if (!reachable) {
            error = "no backward-feasible scalar successor at edge " + std::to_string(i);
            return std::nullopt;
        }
        selected[i + 1] = reachable->max;
    }
    if (route.edges.back().terminal_stop) selected.back() = 0.0;

    SpeedWitness witness;
    witness.positions.reserve(state_count);
    witness.velocities.reserve(state_count);
    witness.durations.reserve(route.edges.size());
    witness.positions.push_back(route.start.position);
    witness.velocities.push_back(
        std::sqrt(std::max(selected.front(), 0.0))
        * Eigen::Vector2d(std::cos(route.start.heading), std::sin(route.start.heading))
    );
    for (size_t i = 0; i < route.edges.size(); ++i) {
        const SpatialRouteEdge& edge = route.edges[i];
        const double from_speed = std::sqrt(std::max(selected[i], 0.0));
        const double to_speed = std::sqrt(std::max(selected[i + 1], 0.0));
        if (from_speed + to_speed <= FORWARD_SPEED_EPS) {
            error = "speed witness contains a zero-speed spatial edge";
            return std::nullopt;
        }
        witness.positions.push_back(edge.to.position);
        witness.velocities.push_back(
            to_speed * Eigen::Vector2d(
                std::cos(edge.to.heading), std::sin(edge.to.heading)
            )
        );
        witness.durations.push_back(2.0 * edge.length / (from_speed + to_speed));
    }
    return witness;
}

} // namespace nav_executor
