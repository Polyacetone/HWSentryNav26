#pragma once

#include <utility>

namespace nav_executor {

// Returns the C1 gate and its derivative with respect to distance.
inline std::pair<double, double> runup_distance_gate(
    const double distance,
    const double radius,
    const double transition
) {
    if (distance <= radius) return {1.0, 0.0};
    if (transition <= 0.0 || distance >= radius + transition) return {0.0, 0.0};
    const double x = (radius + transition - distance) / transition;
    return {
        x * x * (3.0 - 2.0 * x),
        -6.0 * x * (1.0 - x) / transition,
    };
}

} // namespace nav_executor
