#include <nav_executor/nav_executor_node.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <common_utils/convert.hpp>

namespace nav_executor {

void NavExecutorNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    chassis_state_.velocity = msg->velocity;
    chassis_state_.omega = msg->omega;
    chassis_state_.leg_h = msg->leg_h;
    chassis_state_.leg_psi = msg->leg_psi;
    chassis_leg_mode_ = msg->leg_mode;
    rfr_pwr_limit_ = static_cast<double>(msg->rfr_pwr_limit);
    remaining_energy_supercap_filtered_ = remaining_energy_filter_alpha_ * static_cast<double>(msg->remaining_energy_supercap)
        + (1.0 - remaining_energy_filter_alpha_) * remaining_energy_supercap_filtered_;
    remaining_energy_buffercap_filtered_ = remaining_energy_filter_alpha_ * static_cast<double>(msg->remaining_energy_buffercap)
        + (1.0 - remaining_energy_filter_alpha_) * remaining_energy_buffercap_filtered_;
}

void NavExecutorNode::spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg) {
    switch (msg->spin_mode) {
        case 0: spin_state_ = SpinState::STOP; break;
        case 1: spin_state_ = SpinState::SPIN_SLOW; break;
        case 2: spin_state_ = SpinState::SPIN_FAST; break;
        default: RCLCPP_ERROR(get_logger(), "Invalid spin_mode: %d", msg->spin_mode); return;
    }
    spin_high_priority_ = msg->high_priority;
}

void NavExecutorNode::local_cost_maps_callback(const interfaces::msg::CostMaps::SharedPtr msg) {
    if (!global_cost_map_ || msg->maps.empty()) return;
    const int w = global_cost_map_->width;
    const int h = global_cost_map_->height;
    const auto total = static_cast<size_t>(w * h);
    if (msg->maps[0].data.size() != total) return;
    prediction_dt_ = msg->prediction_dt;

    const auto to_cost_map = [&](const nav_msgs::msg::OccupancyGrid& grid) {
        std::vector<uint8_t> data(total);
        for (size_t j = 0; j < total; j++) data[j] = static_cast<uint8_t>(grid.data[j]);
        return std::make_shared<CostMap>(w, h, global_cost_map_->resolution, global_cost_map_->origin_x, global_cost_map_->origin_y, data);
    };

    current_cost_map_ = to_cost_map(msg->maps[0]);

    prediction_maps_.clear();
    for (size_t i = 1; i < msg->maps.size(); i++) {
        if (msg->maps[i].data.size() != total) continue;
        prediction_maps_.push_back(to_cost_map(msg->maps[i]));
    }

    // planner 用：global + 时域融合动态
    CostMap::ConstPtr fused_dynamic;
    if (prediction_horizon_seconds_ <= 0.0 || msg->maps.size() <= 1 || msg->prediction_dt <= 0.0) {
        fused_dynamic = current_cost_map_;
    } else {
        const size_t n = std::min(msg->maps.size(), static_cast<size_t>(std::ceil(prediction_horizon_seconds_ / msg->prediction_dt)) + 1);
        const double inv_denom = n > 1 ? 1.0 / static_cast<double>(n - 1) : 0.0;
        std::vector<double> frame_weights(n);
        double total_weight = 0.0;
        for (size_t i = 0; i < n; i++) {
            frame_weights[i] = std::max(0.0, 1.0 - prediction_weight_decay_ * static_cast<double>(i) * inv_denom);
            total_weight += frame_weights[i];
        }
        if (total_weight <= 0.0) {
            fused_dynamic = current_cost_map_;
        } else {
            std::vector<double> accum(total, 0.0);
            for (size_t i = 0; i < n; i++) {
                const double weight = frame_weights[i];
                if (weight <= 0.0) continue;
                const auto& frame = msg->maps[i];
                for (size_t j = 0; j < total; j++) accum[j] += static_cast<double>(frame.data[j]) * weight;
            }
            std::vector<uint8_t> result(total);
            for (size_t j = 0; j < total; j++) {
                const uint32_t u = static_cast<uint32_t>(accum[j] / total_weight + 0.5);
                result[j] = u > 255u ? 255u : static_cast<uint8_t>(u);
            }
            fused_dynamic = std::make_shared<CostMap>(w, h, global_cost_map_->resolution, global_cost_map_->origin_x, global_cost_map_->origin_y, result);
        }
    }

    try {
        merged_prediction_cost_map_ = std::make_shared<CostMap>(global_cost_map_->merge(*fused_dynamic));
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to merge planner cost map: %s", e.what());
        merged_prediction_cost_map_ = global_cost_map_;
    }
}

// ROS callbacks update cached facts; task and motion state changes happen here.

void NavExecutorNode::control_tick() {
    if (!global_cost_map_ || !global_direction_map_ || !step_mask_ready_) return;

    Eigen::Vector3d chassis_pose_map;
    if (!get_chassis_pose(chassis_pose_map)) return;

    const auto stamp = std::chrono::steady_clock::now();

    previous_motion_feedback_.preemptible = executor_->preemptible();
    previous_motion_feedback_.motion_state = executor_->motion_state();

    const CostLayers cost_layers {
        .global = global_cost_map_,
        .current_dynamic = current_cost_map_,
        .planner_merged = merged_prediction_cost_map_,
        .prediction_dynamic = prediction_maps_,
    };
    const DirectionLayers direction_layers { .global = global_direction_map_ };
    const PerformanceState performance {
        .high_performance = remaining_energy_buffercap_filtered_ >= terrain_profiles_.high_performance_buffercap_threshold
            && remaining_energy_supercap_filtered_ >= terrain_profiles_.high_performance_supercap_threshold
            && rfr_pwr_limit_ >= terrain_profiles_.high_performance_rfr_pwr_limit_threshold
    };
    const TerrainTraversalConstraints terrain_constraints = build_terrain_traversal_constraints(
        *global_direction_map_, terrain_profiles_, performance
    );

    const AnnotatedPath::ConstPtr active_path_before_update = task_->active_path();
    RouteContext route_context = build_route_context(cost_layers, direction_layers, active_path_before_update);
    if (!route_context.masked_global || !route_context.control_final || !route_context.masked_direction) return;

    TaskUpdateInput task_input;
    task_input.incoming_goal = pending_goal_;
    task_input.feedback = previous_motion_feedback_;
    task_input.stamp = stamp;
    pending_goal_.reset();

    if (merged_prediction_cost_map_) {
        task_input.plan_snapshot.current_pos_map = chassis_pose_map.head<2>();
        task_input.plan_snapshot.current_yaw = chassis_pose_map.z();
        task_input.plan_snapshot.current_velocity = chassis_state_.velocity;
        task_input.plan_snapshot.global_cost_map = global_cost_map_;
        task_input.plan_snapshot.merged_cost_map = merged_prediction_cost_map_;
        task_input.plan_snapshot.direction_map = global_direction_map_;
        task_input.plan_snapshot.terrain_constraints = terrain_constraints;
        task_input.plan_snapshot.performance = performance;
    }

    if (active_path_before_update && previous_motion_feedback_.motion_state == MotionState::FOLLOW) {
        RouteMonitorInput rm;
        rm.active_path = active_path_before_update;
        rm.current_u = previous_motion_feedback_.route_u;
        rm.chassis_pos_map = chassis_pose_map.head<2>();
        rm.masked_global_cost_map = route_context.masked_global.get();
        rm.current_dynamic_cost_map = current_cost_map_.get();
        rm.per_step_dynamic_cost_maps = route_context.prediction_dynamic_ptrs;
        rm.masked_direction_map = route_context.masked_direction.get();
        rm.proj_guard = proj_guard_params_;
        rm.step_block = step_block_params_;
        rm.performance = performance_replan_params_;
        rm.current_performance = performance;
        task_input.route_monitor = std::move(rm);
    }

    const TaskUpdateOutput task_output = task_->update(task_input);

    route_context = build_route_context(cost_layers, direction_layers, task_output.command.active_path);
    if (!route_context.masked_global || !route_context.control_final || !route_context.masked_direction) return;

    ExecutorInput ein;
    ein.intent.active_path = task_output.command.active_path;
    ein.intent.hold_goal = task_output.command.hold_goal;
    ein.intent.spin_requested = (spin_state_ != SpinState::STOP);
    ein.intent.spin_high_priority = spin_high_priority_;
    ein.intent.spin_fast = (spin_state_ == SpinState::SPIN_FAST);
    ein.observation.chassis_pose_map = chassis_pose_map;
    ein.observation.chassis_state = chassis_state_;
    ein.observation.remaining_energy = remaining_energy_supercap_filtered_;
    ein.observation.rfr_pwr_limit = rfr_pwr_limit_;
    ein.observation.chassis_leg_mode = chassis_leg_mode_;
    ein.observation.comp_stage = comp_stage_;
    ein.observation.stamp = stamp;
    ein.environment.final_cost_map = route_context.control_final.get();
    ein.environment.masked_global_cost_map = route_context.masked_global.get();
    ein.environment.masked_direction_map = route_context.masked_direction.get();
    ein.environment.base_direction_map = global_direction_map_.get();
    ein.environment.terrain_constraints = &terrain_constraints;
    ein.environment.current_dynamic_cost_map = current_cost_map_.get();
    ein.environment.per_step_cost_maps = std::move(route_context.prediction_with_step_mask_ptrs);
    ein.environment.per_step_dynamic_cost_maps = std::move(route_context.prediction_dynamic_ptrs);
    ein.environment.prediction_dt = prediction_dt_;

    const ExecutorOutput out = executor_->update(ein);

    previous_motion_feedback_.goal_reached = out.goal_reached;
    previous_motion_feedback_.executor_replan_event = out.executor_replan_event;
    previous_motion_feedback_.route_u = out.current_u;
    previous_motion_feedback_.motion_state = out.motion_state;

    if (out.valid) {
        interfaces::msg::ChassisCmd cmd;
        cmd.velocity = static_cast<float>(out.velocity);
        cmd.omega = static_cast<float>(out.omega);
        cmd.mode = out.mode;
        cmd.step_dist = out.step_dist_cm;
        chassis_cmd_pub_->publish(cmd);

        if (enable_debug_) {
            if (out.mpc_path_map) debug_mpc_path_pub_->publish(path_to_nav_msg(*out.mpc_path_map));
            if (out.predicted_v && out.predicted_w && !out.predicted_v->empty() && !out.predicted_w->empty()) {
                std_msgs::msg::Float64 v_msg, w_msg;
                v_msg.data = (*out.predicted_v)[0];
                w_msg.data = (*out.predicted_w)[0];
                debug_v_pred_pub_->publish(v_msg);
                debug_w_pred_pub_->publish(w_msg);
            }
            if (out.search_path && debug_search_path_pub_) debug_search_path_pub_->publish(path_to_nav_msg(*out.search_path));
        }
    }

    publish_diagnostics(task_output.diagnostics, out.motion_state);

    if (enable_debug_ && debug_final_cost_map_pub_) {
        nav_msgs::msg::OccupancyGrid grid_msg;
        grid_msg.header.stamp = now();
        grid_msg.header.frame_id = "map";
        grid_msg.info.width = static_cast<uint32_t>(route_context.control_final->width);
        grid_msg.info.height = static_cast<uint32_t>(route_context.control_final->height);
        grid_msg.info.resolution = static_cast<float>(route_context.control_final->resolution);
        grid_msg.info.origin.position.x = route_context.control_final->origin_x;
        grid_msg.info.origin.position.y = route_context.control_final->origin_y;
        grid_msg.data.resize(route_context.control_final->data.size());
        for (size_t idx = 0; idx < route_context.control_final->data.size(); idx++) {
            grid_msg.data[idx] = static_cast<int8_t>(route_context.control_final->data[idx]);
        }
        debug_final_cost_map_pub_->publish(grid_msg);
    }
}

// ═══════════════════════ 工具 ════════════════════════════════

bool NavExecutorNode::get_chassis_pose(Eigen::Vector3d& chassis_pose) const {
    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = tf_buffer_->lookupTransform("map", "chassis_link", tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(get_logger(), "Could not transform chassis_link to map: %s", ex.what());
        return false;
    }
    chassis_pose.head<2>() = utils::convert_to<Eigen::Vector3d>(tf.transform.translation).head<2>();
    const Eigen::Quaterniond q = utils::convert_to<Eigen::Quaterniond>(tf.transform.rotation);
    const Eigen::Vector2d x_axis = (q * Eigen::Vector3d::UnitX()).head<2>();
    if (x_axis.norm() < 1e-6) {
        RCLCPP_ERROR(get_logger(), "Invalid chassis_link orientation");
        return false;
    }
    chassis_pose.z() = std::atan2(x_axis.y(), x_axis.x());
    return true;
}

void NavExecutorNode::publish_diagnostics(const TaskDiagnostics& diag, const MotionState motion_state) {
    interfaces::msg::NavExecutorState msg;
    msg.motion_state = static_cast<uint8_t>(motion_state);
    msg.has_goal = diag.has_goal;
    msg.has_path = diag.has_path;
    msg.has_hold_goal = diag.has_hold_goal;
    msg.planner_state = static_cast<uint8_t>(diag.planner_state);
    msg.last_replan_reason = static_cast<uint8_t>(diag.last_replan_reason);
    global_path_pub_->publish(path_to_nav_msg(diag.global_path));
    state_pub_->publish(msg);

    if (enable_debug_) {
        if (!diag.debug_rough_path.empty()) debug_rough_path_pub_->publish(path_to_nav_msg(diag.debug_rough_path));
        if (!diag.debug_warmup_path.empty()) debug_warmup_path_pub_->publish(path_to_nav_msg(diag.debug_warmup_path));
    }
}

nav_msgs::msg::Path NavExecutorNode::path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const {
    nav_msgs::msg::Path msg;
    msg.header.stamp = now();
    msg.header.frame_id = "map";
    for (const auto& p : path) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = msg.header;
        ps.pose.position.x = p.x();
        ps.pose.position.y = p.y();
        ps.pose.position.z = 0.0;
        msg.poses.push_back(ps);
    }
    return msg;
}

} // namespace nav_executor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nav_executor::NavExecutorNode)
