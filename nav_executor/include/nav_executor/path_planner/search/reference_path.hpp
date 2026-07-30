#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/path_planner/search/spatial_grid_astar.hpp>
#include <nav_executor/path_planner/trajectory/shaping_dynamics.hpp>

namespace nav_executor {

struct ReferencePoint {
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    double heading = 0.0;
    double speed = 0.0;
    double time_to_goal = 0.0;
};

struct ReferencePath {
    std::vector<ReferencePoint> points;
    std::vector<Eigen::Vector2d> smoothed_path;
    double raw_length = 0.0;
    double smoothed_length = 0.0;
};

class ReferencePathBuilder {
public:
    struct Params {
        double resample_spacing = 0.1;
        double tangent_lookahead = 0.3;
    };

    struct Result {
        ReferencePath path;
        bool success = false;
        std::string error;
    };

    explicit ReferencePathBuilder(Params params) : params_(params) {}

    [[nodiscard]] Result build(
        const SpatialRoute& route,
        const Eigen::Vector2d& start_map,
        const Eigen::Vector2d& goal_map,
        double start_speed,
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const TerrainTraversalConstraints& terrain_constraints,
        const ShapingDynamicsLimits& dynamics,
        int occupied_threshold
    ) const;

private:
    Params params_;
};

} // namespace nav_executor
