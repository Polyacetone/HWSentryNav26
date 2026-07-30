#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/environment/nav_map.hpp>

namespace nav_executor {

struct TerrainPassage {
    uint8_t label = 0;
    bool going_up = true;
    size_t first_body_index = 0;
    size_t last_body_index = 0;
    size_t entry_flat_index = 0;
    size_t exit_flat_index = 0;
};

struct SpatialRoute {
    std::vector<Eigen::Vector2i> raw_path;
    std::vector<TerrainPassage> passages;
    int expansions = 0;
    size_t open_peak = 0;
};

class SpatialGridAstar {
public:
    struct Params {
        double obstacle_weight = 1.0;
        int max_expansions = 1000000;
    };

    struct Result {
        SpatialRoute route;
        bool success = false;
        std::string error;
    };

    explicit SpatialGridAstar(Params params) : params_(params) {}

    [[nodiscard]] Result search(
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const TerrainTraversalConstraints& terrain_constraints,
        const Eigen::Vector2i& start,
        const Eigen::Vector2i& goal,
        int occupied_threshold,
        double detect_dot_threshold
    ) const;

private:
    Params params_;
};

} // namespace nav_executor
