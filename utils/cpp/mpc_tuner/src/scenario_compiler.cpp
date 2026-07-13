#include <mpc_tuner/scenario_compiler.hpp>

#include <stdexcept>
#include <thread>

#include <nav_executor/common/world_context.hpp>
#include <nav_executor/path_planner/a_star_planner.hpp>

namespace mpc_tuner {

ScenarioPlanner::ScenarioPlanner(
    const RuntimeConfig& config,
    nav_executor::CostMap::ConstPtr global_cost_map,
    nav_executor::DirectionMap::ConstPtr direction_map,
    rclcpp::Logger logger
):
    global_cost_map_(std::move(global_cost_map)),
    direction_map_(std::move(direction_map)),
    logger_(std::move(logger))
{
    if (!global_cost_map_ || !direction_map_) {
        throw std::invalid_argument("ScenarioPlanner requires both cost and direction maps");
    }
    if (global_cost_map_->width != direction_map_->width
        || global_cost_map_->height != direction_map_->height
        || global_cost_map_->resolution != direction_map_->resolution
        || global_cost_map_->origin_x != direction_map_->origin_x
        || global_cost_map_->origin_y != direction_map_->origin_y) {
        throw std::invalid_argument("ScenarioPlanner map geometry mismatch");
    }

    terrain_constraints_ = nav_executor::build_terrain_traversal_constraints(
        *direction_map_, config.terrain_profiles, performance_
    );
    step_mask_ = std::make_shared<nav_executor::StepRoutingMask>(config.step_mask);
    step_mask_->initialize(*global_cost_map_, direction_map_);
    const auto& a = config.a_star;
    auto a_star = std::make_shared<nav_executor::AStarPlanner>(
        a.step_alignment_weight, a.obstacle_weight, a.step_proximity_weight,
        a.step_mode_dot_threshold, a.downsampled_waypoint_max_interval, a.feasible_threshold
    );
    auto optimizer = std::make_shared<nav_executor::BSplineOptimizer>(config.path_optimizer);
    planner_ = std::make_unique<nav_executor::PathPlanner>(
        config.planner, std::move(a_star), std::move(optimizer), step_mask_, logger_
    );
    planner_->start();
}

ScenarioPlanner::~ScenarioPlanner() {
    if (planner_) planner_->stop();
}

CompiledScenario ScenarioPlanner::plan(
    const ScenarioSpec& spec,
    const std::chrono::seconds timeout
) {
    nav_executor::PlanRequest request;
    request.goal = {.id = next_goal_id_++, .position_map = spec.goal, .fixed = false};
    request.current_pos_map = spec.start_pose.head<2>();
    request.current_yaw = spec.start_pose.z();
    request.current_velocity = 0.0;
    request.global_cost_map = global_cost_map_;
    request.merged_cost_map = global_cost_map_;
    request.direction_map = direction_map_;
    request.terrain_constraints = terrain_constraints_;
    request.performance = performance_;
    planner_->submit(request);

    std::optional<nav_executor::PlanResult> planned;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (auto candidate = planner_->try_take_result()) {
            planned = std::move(candidate);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (!planned || planned->kind != nav_executor::PlanResult::Kind::PATH || !planned->path) {
        throw std::runtime_error("Failed to plan tuning route: " + spec.name);
    }

    const nav_executor::CostLayers cost_layers {
        .global = global_cost_map_,
        .current_dynamic = nullptr,
        .planner_merged = nullptr,
        .prediction_dynamic = {},
    };
    const nav_executor::DirectionLayers direction_layers {.global = direction_map_};
    auto context = nav_executor::build_route_context(cost_layers, direction_layers, planned->path);
    if (!context.masked_global || !context.control_final) {
        throw std::runtime_error("Failed to build route context for route: " + spec.name);
    }

    RCLCPP_INFO(
        logger_, "Planned route '%s': %.2f m, %zu step segments",
        spec.name.c_str(), planned->path->spline.arc_length(0.0, 1.0, 100),
        planned->path->step_segments.size()
    );
    return {
        .spec = spec,
        .path = planned->path,
        .global_cost_map = global_cost_map_,
        .control_cost_map = std::move(context.control_final),
        .direction_map = direction_map_,
    };
}

} // namespace mpc_tuner
