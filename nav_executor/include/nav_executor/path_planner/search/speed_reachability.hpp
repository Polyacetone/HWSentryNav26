#pragma once

#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/path_planner/trajectory/shaping_dynamics.hpp>

namespace nav_executor {

struct SpeedSquaredInterval {
    double min = 0.0;
    double max = 0.0;
};

struct SpatialPose {
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    double heading = 0.0;
};

struct SpatialRouteEdge {
    SpatialPose from;
    SpatialPose to;
    double length = 0.0;
    double curvature = 0.0;
    SpeedSquaredInterval endpoint_speed_limit;
    bool terminal_stop = false;
};

struct SpatialRoute {
    SpatialPose start;
    std::vector<SpatialRouteEdge> edges;
};

struct SpeedWitness {
    std::vector<Eigen::Vector2d> positions;
    std::vector<Eigen::Vector2d> velocities;
    std::vector<double> durations;
};

class SpeedReachability {
public:
    explicit SpeedReachability(ShapingDynamicsLimits limits) : limits_(limits) {}

    [[nodiscard]] SpeedSquaredInterval unrestricted_limit() const;

    [[nodiscard]] std::optional<SpeedSquaredInterval> propagate(
        const SpeedSquaredInterval& initial,
        double length,
        double curvature,
        const SpeedSquaredInterval& endpoint_speed_limit,
        bool terminal_stop = false
    ) const;

    [[nodiscard]] std::optional<SpeedSquaredInterval> propagate_route(
        const SpeedSquaredInterval& initial,
        const SpatialRoute& route
    ) const;

    [[nodiscard]] std::optional<SpeedWitness> reconstruct_witness(
        const SpeedSquaredInterval& initial,
        const SpatialRoute& route,
        std::string& error
    ) const;

private:
    [[nodiscard]] double dynamic_speed_squared_cap(double curvature) const;
    [[nodiscard]] double tangential_acceleration_cap(double curvature) const;
    [[nodiscard]] std::optional<SpeedSquaredInterval> predecessor(
        const SpeedSquaredInterval& target,
        const SpatialRouteEdge& edge
    ) const;

    ShapingDynamicsLimits limits_;
};

[[nodiscard]] std::optional<SpeedSquaredInterval> intersect_speed_intervals(
    const SpeedSquaredInterval& lhs,
    const SpeedSquaredInterval& rhs
);

[[nodiscard]] bool speed_interval_contains(
    const SpeedSquaredInterval& outer,
    const SpeedSquaredInterval& inner
);

} // namespace nav_executor
