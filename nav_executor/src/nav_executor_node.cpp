#include <nav_executor/nav_executor_node.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <common_utils/convert.hpp>

#include <algorithm>
#include <cmath>

namespace {

std_msgs::msg::ColorRGBA velocity_color(const double velocity, const double velocity_min, const double velocity_max) {
    const double range = velocity_max - velocity_min;
    const double normalized_velocity = range > 1e-6 && std::isfinite(velocity)
        ? std::clamp((velocity - velocity_min) / range, 0.0, 1.0)
        : 0.0;

    std_msgs::msg::ColorRGBA color;
    color.r = static_cast<float>(std::clamp(1.5 - std::abs(4.0 * normalized_velocity - 3.0), 0.0, 1.0));
    color.g = static_cast<float>(std::clamp(1.5 - std::abs(4.0 * normalized_velocity - 2.0), 0.0, 1.0));
    color.b = static_cast<float>(std::clamp(1.5 - std::abs(4.0 * normalized_velocity - 1.0), 0.0, 1.0));
    color.a = 1.0F;
    return color;
}

} // anonymous namespace

namespace nav_executor {

void NavExecutorNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    ++chassis_state_sequence_;
    if (!std::isfinite(msg->velocity) || !std::isfinite(msg->omega)
        || !std::isfinite(msg->leg_h) || !std::isfinite(msg->leg_psi)) {
        RCLCPP_ERROR(get_logger(), "Ignoring non-finite chassis status");
        chassis_state_valid_ = false;
        return;
    }
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
    chassis_state_valid_ = true;
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
    if (dynamic_prediction_horizon_seconds_ <= 0.0 || msg->maps.size() <= 1 || msg->prediction_dt <= 0.0) {
        fused_dynamic = current_cost_map_;
    } else {
        const size_t n = std::min(msg->maps.size(), static_cast<size_t>(std::ceil(dynamic_prediction_horizon_seconds_ / msg->prediction_dt)) + 1);
        const double inv_denom = n > 1 ? 1.0 / static_cast<double>(n - 1) : 0.0;
        std::vector<double> frame_weights(n);
        double total_weight = 0.0;
        for (size_t i = 0; i < n; i++) {
            frame_weights[i] = std::max(0.0, 1.0 - dynamic_prediction_weight_decay_ * static_cast<double>(i) * inv_denom);
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

// ROS 回调只写缓存；任务/运动状态转移在 control_tick 中完成。

void NavExecutorNode::control_tick() {
    if (!global_cost_map_ || !global_direction_map_ || !step_mask_ready_
        || !chassis_state_valid_
        || chassis_state_sequence_ == 0
        || chassis_state_sequence_ == last_control_chassis_state_sequence_) return;

    Eigen::Vector3d chassis_pose_map;
    if (!get_chassis_pose(chassis_pose_map)) return;

    const auto stamp = std::chrono::steady_clock::now();

    const CostLayers cost_layers {
        .global = global_cost_map_,
        .current_dynamic = current_cost_map_,
        .planner_merged = merged_prediction_cost_map_,
        .prediction_dynamic = prediction_maps_,
    };
    const DirectionLayers direction_layers { .global = global_direction_map_ };
    const PerformanceState performance {
        .high_performance = remaining_energy_buffercap_filtered_ >= traversal_configuration_.high_performance_buffercap_threshold
            && remaining_energy_supercap_filtered_ >= traversal_configuration_.high_performance_supercap_threshold
            && rfr_pwr_limit_ >= traversal_configuration_.high_performance_rfr_pwr_limit_threshold
    };
    const TerrainTraversalConstraints terrain_constraints = build_terrain_traversal_constraints(
        *global_direction_map_, traversal_configuration_, performance
    );

    const AnnotatedPath::ConstPtr active_path_before_update = task_->active_path();
    std::optional<RouteEstimate> route_estimate = route_tracker_->update(
        active_path_before_update,
        chassis_pose_map,
        chassis_state_.velocity,
        stamp
    );

    // TaskManager 必须使用本周期 route 位置判断 commit，而不能只依赖上一周期 FSM。
    // 这样规划结果恰好在跨过 commit 弧长的控制周期到达时，也不会获得路径执行权。
    previous_motion_feedback_.motion_state = executor_->motion_state();
    const bool route_tracked = active_path_before_update && route_estimate
        && route_estimate->path == active_path_before_update
        && route_estimate->status == RouteTrackingStatus::TRACKED;
    const StepExecutionPreview step_preview = executor_->preview_step_execution(
        active_path_before_update,
        route_tracked ? route_estimate->arc_length : 0.0,
        route_tracked
    );
    previous_motion_feedback_.step_phase = step_preview.phase;
    previous_motion_feedback_.preemptible = step_preview.preemptible;

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

    const MotionState previous_state = previous_motion_feedback_.motion_state;
    const bool pending_mpc_lethal = previous_motion_feedback_.mpc_lethal
        && previous_motion_feedback_.lethal_path == active_path_before_update;
    const bool route_monitoring_state = pending_mpc_lethal
        || (previous_motion_feedback_.preemptible
            && (previous_state == MotionState::FOLLOW
                || previous_state == MotionState::PREPARE_SPIN
                || previous_state == MotionState::IDLE
                || previous_state == MotionState::STEPPING));
    if (active_path_before_update && route_estimate && route_monitoring_state) {
        RouteMonitorInput rm;
        rm.active_path = active_path_before_update;
        rm.route = *route_estimate;
        rm.chassis_pos_map = chassis_pose_map.head<2>();
        rm.masked_global_cost_map = route_context.masked_global.get();
        rm.current_dynamic_cost_map = current_cost_map_.get();
        rm.per_step_dynamic_cost_maps = route_context.prediction_dynamic_ptrs;
        rm.prediction_dt = prediction_dt_;
        rm.base_direction_map = global_direction_map_.get();
        rm.proj_guard = proj_guard_params_;
        rm.step_block = step_block_params_;
        rm.performance = performance_replan_params_;
        rm.current_performance = performance;
        rm.mpc_lethal = previous_motion_feedback_.mpc_lethal && previous_motion_feedback_.lethal_path == active_path_before_update;
        task_input.route_monitor = std::move(rm);
    }

    const TaskUpdateOutput task_output = task_->update(task_input);

    if (task_output.command.active_path != active_path_before_update) {
        route_estimate = route_tracker_->update(
            task_output.command.active_path,
            chassis_pose_map,
            chassis_state_.velocity,
            stamp
        );
    }

    route_context = build_route_context(cost_layers, direction_layers, task_output.command.active_path);
    if (!route_context.masked_global || !route_context.control_final || !route_context.masked_direction) return;

    ExecutorInput ein;
    ein.intent.active_path = task_output.command.active_path;
    ein.intent.hold_goal = task_output.command.hold_goal;
    ein.intent.spin_requested = (spin_state_ != SpinState::STOP);
    ein.intent.spin_high_priority = spin_high_priority_;
    ein.intent.spin_fast = (spin_state_ == SpinState::SPIN_FAST);
    ein.route = route_estimate;
    ein.observation.chassis_pose_map = chassis_pose_map;
    ein.observation.chassis_state = chassis_state_;
    ein.observation.chassis_state_sequence = chassis_state_sequence_;
    ein.observation.chassis_leg_mode = chassis_leg_mode_;
    ein.observation.comp_stage = comp_stage_;
    ein.observation.stamp = stamp;
    ein.environment.final_cost_map = route_context.control_final.get();
    ein.environment.masked_global_cost_map = route_context.masked_global.get();
    ein.environment.masked_direction_map = route_context.masked_direction.get();
    ein.environment.base_direction_map = global_direction_map_.get();
    ein.environment.current_dynamic_cost_map = current_cost_map_.get();
    ein.environment.per_step_cost_maps = std::move(route_context.prediction_with_step_mask_ptrs);
    ein.environment.per_step_dynamic_cost_maps = std::move(route_context.prediction_dynamic_ptrs);
    ein.environment.prediction_dt = prediction_dt_;

    last_control_chassis_state_sequence_ = chassis_state_sequence_;
    ExecutorOutput out = executor_->update(ein);

    previous_motion_feedback_.goal_reached = out.goal_reached;
    previous_motion_feedback_.goal_reached_path = out.goal_reached ? task_output.command.active_path : nullptr;
    previous_motion_feedback_.executor_replan_event = out.executor_replan_event;
    previous_motion_feedback_.mpc_lethal = out.mpc_lethal;
    previous_motion_feedback_.lethal_path = out.mpc_lethal ? task_output.command.active_path : nullptr;
    previous_motion_feedback_.motion_state = out.motion_state;
    previous_motion_feedback_.step_phase = out.step_phase;

    if (out.motion_state == MotionState::IDLE) {
        out.mode = idle_chassis_mode_override_;
    }

    if (out.valid) {
        interfaces::msg::ChassisCmd cmd;
        cmd.velocity = static_cast<float>(out.velocity);
        cmd.omega = static_cast<float>(out.omega);
        cmd.mode = out.mode;
        cmd.step_dist = out.step_dist_cm;
        chassis_cmd_pub_->publish(cmd);
    }

    publish_diagnostics(task_output.diagnostics, out, task_output.command.active_path);

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

void NavExecutorNode::publish_diagnostics(
    const TaskDiagnostics& diag,
    const ExecutorOutput& executor_output,
    const AnnotatedPath::ConstPtr& active_path
) {
    if (active_path) global_path_pub_->publish(trajectory_to_nav_msg(active_path->trajectory));
    if (!enable_debug_) return;

    if (!diag.debug_rough_path.empty()) {
        debug_rough_path_pub_->publish(path_to_nav_msg(diag.debug_rough_path));
    }
    if (active_path) {
        debug_minco_trajectory_pub_->publish(trajectory_to_marker(*active_path));
    }
    if (executor_output.mpc_path_map) {
        debug_mpc_path_pub_->publish(path_to_nav_msg(*executor_output.mpc_path_map));
    }

    interfaces::msg::NavExecutorDiag msg;

    msg.motion_state = static_cast<uint8_t>(executor_output.motion_state);
    msg.step_phase = static_cast<uint8_t>(executor_output.step_phase);
    msg.has_goal = diag.has_goal;
    msg.has_path = diag.has_path;
    msg.has_hold_goal = diag.has_hold_goal;
    msg.planner_state = static_cast<uint8_t>(diag.planner_state);
    msg.last_replan_reason = static_cast<uint8_t>(diag.last_replan_reason);

    msg.command_published = executor_output.valid;
    msg.command_velocity = executor_output.velocity;
    msg.command_angular_velocity = executor_output.omega;
    msg.command_mode = executor_output.mode;
    msg.command_step_distance = executor_output.step_dist_cm;

    const ObserverDiagnostics& observer = executor_output.observer_diagnostics;
    msg.observer_event = static_cast<uint8_t>(observer.event);
    msg.observer_last_reset_reason = static_cast<uint8_t>(observer.last_reset_reason);
    msg.observer_initialized = observer.initialized;
    msg.observer_validated = observer.validated;
    msg.observer_prediction_available = observer.prediction_available;
    msg.observer_auxiliary_prediction_available = observer.auxiliary_prediction_available;
    msg.observer_velocity_correction_clipped = observer.velocity_correction_clipped;
    msg.observer_state_sequence = observer.state_sequence;
    msg.observer_reset_count = observer.reset_count;
    msg.observer_active_run_length = observer.active_run_length;
    msg.observer_revalidation_latency_updates = observer.revalidation_latency_updates;
    msg.observer_hidden_state = observer.hidden_state_estimate;
    msg.observer_predicted_hidden_state = observer.predicted_hidden_state;
    msg.observer_predicted_velocity = observer.predicted_velocity;
    msg.observer_velocity_innovation = observer.velocity_innovation;
    msg.observer_predicted_angular_velocity = observer.predicted_angular_velocity;
    msg.observer_angular_velocity_innovation = observer.angular_velocity_innovation;
    msg.observer_predicted_leg_psi = observer.predicted_leg_psi;
    msg.observer_leg_psi_innovation = observer.leg_psi_innovation;
    msg.observer_input_command_velocity = observer.input_command_velocity;
    msg.observer_input_command_angular_velocity = observer.input_command_angular_velocity;

    msg.mpc_attempted = executor_output.mpc_diagnostics.has_value();
    if (executor_output.mpc_diagnostics) {
        const MPCDiagnostics& mpc = *executor_output.mpc_diagnostics;
        msg.mpc_mode = static_cast<uint8_t>(mpc.solver_mode);
        msg.mpc_succeeded = mpc.solve_succeeded;
        msg.mpc_error = mpc.solve_error;

        msg.ancillary_enabled = mpc.ancillary_enabled;
        msg.ancillary_active = mpc.ancillary_active;
        msg.ancillary_reanchored = mpc.nominal_reanchored;
        msg.ancillary_tube_feasible = mpc.first_command_tube_feasible;

        msg.measured_velocity = mpc.measured_velocity.x();
        msg.measured_angular_velocity = mpc.measured_velocity.y();
        msg.previous_command_velocity = mpc.previous_command.x();
        msg.previous_command_angular_velocity = mpc.previous_command.y();
        msg.mpc_nominal_command_velocity = mpc.nominal_command.x();
        msg.mpc_nominal_command_angular_velocity = mpc.nominal_command.y();
        msg.mpc_nominal_command_velocity_rate = mpc.nominal_command_rate.x();
        msg.mpc_nominal_command_angular_velocity_rate = mpc.nominal_command_rate.y();
        msg.mpc_applied_command_velocity_rate = mpc.applied_command_rate.x();
        msg.mpc_applied_command_angular_velocity_rate = mpc.applied_command_rate.y();

        msg.mpc_reference_path_progress = mpc.reference_path_progress;
        msg.mpc_reference_path_speed = mpc.reference_path_speed;
        msg.trajectory_nominal_velocity = mpc.trajectory_nominal_velocity;
        msg.trajectory_nominal_angular_velocity = mpc.trajectory_nominal_angular_velocity;
        msg.mpc_reference_velocity = mpc.reference_velocity;
        msg.mpc_reference_angular_velocity = mpc.reference_angular_velocity;
        msg.mpc_nominal_prediction_velocity = mpc.nominal_prediction.v_pred;
        msg.mpc_nominal_prediction_angular_velocity = mpc.nominal_prediction.w_pred;
        msg.mpc_applied_prediction_velocity = mpc.applied_prediction.v_pred;
        msg.mpc_applied_prediction_angular_velocity = mpc.applied_prediction.w_pred;
    }

    debug_diag_pub_->publish(msg);
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

nav_msgs::msg::Path NavExecutorNode::trajectory_to_nav_msg(const MincoTrajectory& trajectory) const {
    nav_msgs::msg::Path msg;
    msg.header.stamp = now();
    msg.header.frame_id = "map";
    for (double arc_length = 0.0; arc_length < trajectory.total_arc_length(); arc_length += path_publish_sample_resolution_) {
        const TrajSample sample = trajectory.eval_arc_length(arc_length);
        geometry_msgs::msg::PoseStamped pose;
        pose.header = msg.header;
        pose.pose.position.x = sample.p.x();
        pose.pose.position.y = sample.p.y();
        pose.pose.position.z = 0.0;
        pose.pose.orientation = tf2::toMsg(tf2::Quaternion(0.0, 0.0, std::sin(sample.theta / 2.0), std::cos(sample.theta / 2.0)));
        msg.poses.push_back(pose);
    }
    const TrajSample final_sample = trajectory.eval(1.0);
    geometry_msgs::msg::PoseStamped final_pose;
    final_pose.header = msg.header;
    final_pose.pose.position.x = final_sample.p.x();
    final_pose.pose.position.y = final_sample.p.y();
    final_pose.pose.position.z = 0.0;
    final_pose.pose.orientation = tf2::toMsg(tf2::Quaternion(0.0, 0.0, std::sin(final_sample.theta / 2.0), std::cos(final_sample.theta / 2.0)));
    msg.poses.push_back(final_pose);
    return msg;
}

visualization_msgs::msg::Marker NavExecutorNode::trajectory_to_marker(const AnnotatedPath& path) const {
    visualization_msgs::msg::Marker msg;
    msg.header.frame_id = "map";
    msg.header.stamp = now();
    msg.ns = "minco_trajectory";
    msg.id = 0;
    msg.type = visualization_msgs::msg::Marker::LINE_STRIP;
    msg.action = visualization_msgs::msg::Marker::ADD;
    msg.pose.orientation.w = 1.0;
    msg.scale.x = 0.15;

    const MincoTrajectory& trajectory = path.trajectory;
    const auto append_sample = [&](const TrajSample& sample, const double progress) {
        geometry_msgs::msg::Point point;
        point.x = sample.p.x();
        point.y = sample.p.y();
        msg.points.push_back(point);
        msg.colors.push_back(velocity_color(
            path.speed_profile.eval_arc_length(progress).velocity,
            debug_velocity_color_min_, debug_velocity_color_max_
        ));
    };
    for (double arc_length = 0.0; arc_length < trajectory.total_arc_length(); arc_length += path_publish_sample_resolution_) {
        append_sample(trajectory.eval_arc_length(arc_length), arc_length);
    }
    append_sample(trajectory.eval(1.0), trajectory.total_arc_length());
    return msg;
}


} // namespace nav_executor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nav_executor::NavExecutorNode)
