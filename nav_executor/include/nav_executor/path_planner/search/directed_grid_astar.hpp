#pragma once

#include <expected>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/path_planner/search/terrain_topology.hpp>

namespace nav_executor {

class DirectedGridAstar {
public:
    struct Params {
        double obstacle_weight = 0.02;
        double alignment_weight = 1.0;
        double terrain_proximity_weight = 0.1;
        int max_expansions = 1000000;
    };

    struct Result {
        std::vector<Eigen::Vector2i> raw_path;
        int expansions = 0;
        size_t open_peak = 0;
    };

    struct TerrainReachability {
        std::vector<uint8_t> can_reach_exit;
        int width = 0;
        int expansions = 0;
        size_t open_peak = 0;

        [[nodiscard]] bool contains(const Eigen::Vector2i& cell) const;
    };

    explicit DirectedGridAstar(Params params) : params_(params) {}

    [[nodiscard]] std::expected<Result, std::string> search_global(
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const TerrainTraversalConstraints& terrain_constraints,
        const Eigen::Vector2i& start,
        const Eigen::Vector2i& goal,
        int occupied_threshold,
        double detect_dot_threshold
    ) const;

    [[nodiscard]] std::expected<Result, std::string> search_terrain_region(
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const TerrainTraversalConstraints& terrain_constraints,
        const TerrainRegionIndex& regions,
        TerrainRegionId region,
        StepDirection direction,
        const Eigen::Vector2i& start,
        const Eigen::Vector2i& goal,
        int occupied_threshold,
        double detect_dot_threshold
    ) const;

    [[nodiscard]] std::expected<TerrainReachability, std::string>
    build_terrain_reachability(
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const TerrainTraversalConstraints& terrain_constraints,
        const TerrainRegionIndex& regions,
        TerrainRegionId region,
        StepDirection direction,
        const Eigen::Vector2i& exit,
        int occupied_threshold,
        double detect_dot_threshold
    ) const;

private:
    Params params_;
};

} // namespace nav_executor
