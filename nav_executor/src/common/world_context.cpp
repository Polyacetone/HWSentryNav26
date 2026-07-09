#include <nav_executor/common/world_context.hpp>

namespace nav_executor {

RouteContext build_route_context(
    const CostLayers& cost_layers,
    const DirectionLayers& direction_layers,
    const AnnotatedPath* active_path
) {
    RouteContext context;

    const CostMap* step_cost_layer = active_path ? active_path->step_cost_layer.get() : nullptr;
    context.masked_direction = active_path && active_path->masked_direction_map
        ? active_path->masked_direction_map
        : direction_layers.global;

    if (cost_layers.global) {
        context.masked_global = step_cost_layer
            ? std::make_shared<CostMap>(cost_layers.global->merge(*step_cost_layer))
            : cost_layers.global;

        context.control_final = cost_layers.current_dynamic
            ? std::make_shared<CostMap>(context.masked_global->merge(*cost_layers.current_dynamic))
            : context.masked_global;
    }

    context.prediction_with_step_mask.reserve(cost_layers.prediction_dynamic.size());
    context.prediction_with_step_mask_ptrs.reserve(cost_layers.prediction_dynamic.size());
    context.prediction_dynamic_ptrs.reserve(cost_layers.prediction_dynamic.size());

    if (context.masked_global) {
        for (const auto& prediction : cost_layers.prediction_dynamic) {
            if (!prediction) continue;
            context.prediction_with_step_mask.push_back(
                std::make_shared<CostMap>(context.masked_global->merge(*prediction))
            );
            context.prediction_with_step_mask_ptrs.push_back(context.prediction_with_step_mask.back().get());
            context.prediction_dynamic_ptrs.push_back(prediction.get());
        }
    }

    return context;
}

} // namespace nav_executor
