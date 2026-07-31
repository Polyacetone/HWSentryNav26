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

} // anonymous namespace

PlannerObstacleView build_planner_obstacle_view(
    const ObstacleLayers& layers,
    const double prediction_dt,
    const double prediction_horizon_seconds
) {
    PlannerObstacleView view;
    view.global_static = layers.global_static;
    if (!layers.global_static || !layers.base_direction) return view;
    if (!layers.global_static->geometry.same_geometry(layers.base_direction->geometry)) {
        throw std::invalid_argument("Global cost and direction map geometry mismatch");
    }

    std::vector<CostMap::ConstPtr> frames;
    if (layers.dynamic_current) frames.push_back(layers.dynamic_current);

    if (prediction_horizon_seconds > 0.0 && prediction_dt > 0.0) {
        const size_t prediction_count = std::min(
            layers.dynamic_predictions.size(),
            static_cast<size_t>(std::floor(prediction_horizon_seconds / prediction_dt))
        );
        frames.insert(
            frames.end(), layers.dynamic_predictions.begin(),
            layers.dynamic_predictions.begin() + static_cast<std::ptrdiff_t>(prediction_count)
        );
    }

    if (frames.empty()) {
        view.hard_cost = layers.global_static;
        return view;
    }

    const GridGeometry& geometry = layers.global_static->geometry;
    for (const CostMap::ConstPtr& frame : frames) {
        if (!frame) throw std::invalid_argument("Null dynamic obstacle prediction frame");
        require_same_geometry(*frame, geometry);
    }

    std::vector<uint8_t> terrain_dynamic(layers.global_static->data.size(), 0);
    for (int y = 0; y < geometry.height(); ++y) {
        for (int x = 0; x < geometry.width(); ++x) {
            const Eigen::Vector2i cell {x, y};
            if (!layers.base_direction->is_terrain_body_cell(cell)) continue;
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(geometry.width())
                + static_cast<size_t>(x);
            for (const CostMap::ConstPtr& frame : frames) {
                terrain_dynamic[index] = std::max(terrain_dynamic[index], frame->data[index]);
            }
        }
    }

    const CostMap dynamic_union(geometry, std::move(terrain_dynamic));
    view.hard_cost = std::make_shared<CostMap>(layers.global_static->merge(dynamic_union));
    return view;
}

FollowerObstacleView build_follower_obstacle_view(
    const ObstacleLayers& layers,
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

    if (layers.dynamic_current) view.dynamic_timeline.push_back(layers.dynamic_current);
    view.dynamic_timeline.insert(
        view.dynamic_timeline.end(),
        layers.dynamic_predictions.begin(), layers.dynamic_predictions.end()
    );
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
