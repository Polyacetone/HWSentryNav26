#pragma once

#include <memory>
#include <vector>

#include <nav_executor/common/trajectory/annotated_path.hpp>
#include <nav_executor/common/environment/nav_map.hpp>

namespace nav_executor {

struct CostLayers {
    CostMap::ConstPtr global;
    CostMap::ConstPtr current_dynamic;
    CostMap::ConstPtr planner_merged;
    std::vector<CostMap::ConstPtr> prediction_dynamic;
};

struct DirectionLayers {
    DirectionMap::ConstPtr global;
};

struct RouteContext {
    CostMap::ConstPtr masked_global;
    CostMap::ConstPtr control_final;
    DirectionMap::ConstPtr masked_direction;
    std::vector<CostMap::ConstPtr> prediction_with_step_mask;

    std::vector<const CostMap*> prediction_with_step_mask_ptrs;
    std::vector<const CostMap*> prediction_dynamic_ptrs;
};

[[nodiscard]] RouteContext build_route_context(
    const CostLayers& cost_layers,
    const DirectionLayers& direction_layers,
    const AnnotatedPath::ConstPtr& active_path
);

} // namespace nav_executor
