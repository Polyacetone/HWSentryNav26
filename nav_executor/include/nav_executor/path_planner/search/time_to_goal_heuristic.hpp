#pragma once

#include <limits>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/environment/nav_map.hpp>

namespace nav_executor {

class TimeToGoalHeuristic {
public:
    static constexpr double UNREACHABLE = std::numeric_limits<double>::infinity();

    void build(
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const TerrainTraversalConstraints& terrain_constraints,
        const Eigen::Vector2i& goal_cell,
        int occupied_threshold,
        double velocity_max
    );

    [[nodiscard]] bool ready() const { return geometry_.has_value(); }
    [[nodiscard]] double at_cell(const Eigen::Vector2i& cell) const;
    [[nodiscard]] double at_map(const Eigen::Vector2d& point_map) const;

private:
    [[nodiscard]] size_t index(const Eigen::Vector2i& cell) const;

    std::optional<GridGeometry> geometry_;
    std::vector<double> time_;
};

} // namespace nav_executor
