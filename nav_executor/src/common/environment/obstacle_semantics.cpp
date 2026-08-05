#include <nav_executor/common/environment/obstacle_semantics.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nav_executor {

namespace {

void require_same_geometry(const CostMap& map, const GridGeometry& geometry) {
    if (!map.geometry.same_geometry(geometry)) {
        throw std::invalid_argument("Obstacle layer geometry mismatch");
    }
}

CostMap::ConstPtr apply_terrain_influence(
    const CostMap& dynamic_cost,
    const CostMap& terrain_cost
) {
    require_same_geometry(dynamic_cost, terrain_cost.geometry);
    std::vector<uint8_t> weighted(dynamic_cost.data.size(), 0);
    for (size_t index = 0; index < weighted.size(); ++index) {
        const unsigned int product = static_cast<unsigned int>(dynamic_cost.data[index])
            * static_cast<unsigned int>(terrain_cost.data[index]);
        weighted[index] = static_cast<uint8_t>((product + 127U) / 255U);
    }
    return std::make_shared<const CostMap>(terrain_cost.geometry, std::move(weighted));
}

} // anonymous namespace

PlannerObstacleView build_planner_obstacle_view(
    const ObstacleLayers& layers,
    const double prediction_dt,
    const double prediction_horizon_seconds
) {
    PlannerObstacleView view;
    view.global_static = layers.global_static;
    if (!layers.global_static || !layers.base_terrain_cost) return view;
    require_same_geometry(*layers.base_terrain_cost, layers.global_static->geometry);

    if (!layers.dynamic_current) {
        view.hard_cost = layers.global_static;
        return view;
    }

    const GridGeometry& geometry = layers.global_static->geometry;
    view.terrain_dynamic_timeline.reserve(layers.dynamic_predictions.size() + 1);
    view.terrain_dynamic_timeline.push_back(apply_terrain_influence(
        *layers.dynamic_current, *layers.base_terrain_cost
    ));
    for (const CostMap::ConstPtr& prediction : layers.dynamic_predictions) {
        if (!prediction) throw std::invalid_argument("Null dynamic obstacle prediction frame");
        view.terrain_dynamic_timeline.push_back(apply_terrain_influence(
            *prediction, *layers.base_terrain_cost
        ));
    }

    size_t planning_frame_count = 1;
    if (prediction_horizon_seconds > 0.0 && prediction_dt > 0.0) {
        planning_frame_count += std::min(
            layers.dynamic_predictions.size(),
            static_cast<size_t>(std::floor(prediction_horizon_seconds / prediction_dt))
        );
    }
    std::vector<uint8_t> terrain_dynamic(layers.global_static->data.size(), 0);
    for (size_t index = 0; index < terrain_dynamic.size(); ++index) {
        for (size_t frame = 0; frame < planning_frame_count; ++frame) {
            terrain_dynamic[index] = std::max(
                terrain_dynamic[index],
                view.terrain_dynamic_timeline[frame]->data[index]
            );
        }
    }

    const CostMap dynamic_union(geometry, std::move(terrain_dynamic));
    view.hard_cost = std::make_shared<CostMap>(layers.global_static->merge(dynamic_union));
    return view;
}

FollowerObstacleView build_follower_obstacle_view(
    const ObstacleLayers& layers,
    const std::vector<CostMap::ConstPtr>& terrain_dynamic_timeline,
    const CostMap::ConstPtr& step_cost_layer,
    const DirectionMap::ConstPtr& route_direction,
    const double prediction_dt,
    const int occupied_threshold
) {
    FollowerObstacleView view;
    view.base_direction = layers.base_direction;
    view.route_direction = route_direction ? route_direction : layers.base_direction;
    view.prediction_dt = prediction_dt;
    view.occupied_threshold = occupied_threshold;

    view.terrain_dynamic_timeline = terrain_dynamic_timeline;
    if (!layers.global_static) return view;

    view.hard_route_cost = step_cost_layer
        ? std::make_shared<CostMap>(layers.global_static->merge(*step_cost_layer))
        : layers.global_static;
    view.soft_current_cost = layers.dynamic_current
        ? std::make_shared<CostMap>(view.hard_route_cost->merge(*layers.dynamic_current))
        : view.hard_route_cost;

    view.soft_prediction_costs.reserve(layers.dynamic_predictions.size());
    view.soft_prediction_cost_ptrs.reserve(layers.dynamic_predictions.size());
    for (const CostMap::ConstPtr& prediction : layers.dynamic_predictions) {
        if (!prediction) throw std::invalid_argument("Null dynamic obstacle prediction frame");
        view.soft_prediction_costs.push_back(
            std::make_shared<CostMap>(view.hard_route_cost->merge(*prediction))
        );
        view.soft_prediction_cost_ptrs.push_back(view.soft_prediction_costs.back().get());
    }
    return view;
}

} // namespace nav_executor
