#pragma once

#include <cstdint>
#include <expected>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/chassis_defs.hpp>
#include <nav_executor/common/environment/nav_map.hpp>

namespace nav_executor {

using TerrainRegionId = uint32_t;
inline constexpr TerrainRegionId INVALID_TERRAIN_REGION =
    std::numeric_limits<TerrainRegionId>::max();

struct BoundaryTransition {
    Eigen::Vector2i flat_cell = Eigen::Vector2i::Zero();
    Eigen::Vector2i body_cell = Eigen::Vector2i::Zero();
};

struct DirectedPortal {
    TerrainRegionId region = INVALID_TERRAIN_REGION;
    StepDirection direction = StepDirection::UP;
    std::vector<BoundaryTransition> transitions;
};

struct TerrainPassage {
    TerrainRegionId region = INVALID_TERRAIN_REGION;
    uint8_t label = 0;
    StepDirection direction = StepDirection::UP;
    BoundaryTransition selected_entry;
    BoundaryTransition selected_exit;
    DirectedPortal entry_portal;
    std::vector<Eigen::Vector2i> initial_body_path;
};

class TerrainRegionIndex {
public:
    TerrainRegionIndex(
        const DirectionMap& direction_map,
        const CostMap& planning_cost_map,
        int occupied_threshold
    );

    [[nodiscard]] TerrainRegionId region_at(const Eigen::Vector2i& cell) const;
    [[nodiscard]] const std::vector<Eigen::Vector2i>& cells(TerrainRegionId region) const;
    [[nodiscard]] size_t region_count() const { return cells_by_region_.size(); }

private:
    [[nodiscard]] size_t index(const Eigen::Vector2i& cell) const;

    int width_ = 0;
    int height_ = 0;
    std::vector<TerrainRegionId> region_by_cell_;
    std::vector<std::vector<Eigen::Vector2i>> cells_by_region_;
};

[[nodiscard]] bool same_cell(
    const Eigen::Vector2i& lhs,
    const Eigen::Vector2i& rhs
);

[[nodiscard]] bool grid_cell_traversable(
    const CostMap& cost_map,
    const Eigen::Vector2i& cell,
    int occupied_threshold
);

[[nodiscard]] bool grid_edge_avoids_corner_cutting(
    const CostMap& cost_map,
    const Eigen::Vector2i& from,
    const Eigen::Vector2i& to,
    int occupied_threshold
);

[[nodiscard]] bool directed_terrain_edge_allowed(
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const Eigen::Vector2i& from,
    const Eigen::Vector2i& to,
    double detect_dot_threshold,
    StepDirection* direction = nullptr
);

[[nodiscard]] std::expected<std::vector<TerrainPassage>, std::string>
extract_terrain_passages(
    const std::vector<Eigen::Vector2i>& raw_path,
    const CostMap& planning_cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const TerrainRegionIndex& regions,
    int occupied_threshold,
    double detect_dot_threshold
);

} // namespace nav_executor
