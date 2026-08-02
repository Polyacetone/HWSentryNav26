#include <nav_executor/nav_executor_node.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <common_utils/convert.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

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

uint8_t chassis_control_state_value(const uint8_t leg_mode, const uint8_t comp_stage) {
    if (comp_stage != nav_executor::COMP_STAGE_MATCH
        || leg_mode > static_cast<uint8_t>(nav_executor::LegMode::ABNORMAL)) {
        return static_cast<uint8_t>(nav_executor::ChassisControlState::STOPPED);
    }
    return static_cast<uint8_t>(nav_executor::classify_chassis_control_state(leg_mode, comp_stage));
}

bool navigation_planning_suspended(const nav_executor::MotionState motion_state) {
    return motion_state == nav_executor::MotionState::HAZARD_RECOVERY
        || motion_state == nav_executor::MotionState::STUCK_REVERSE
        || motion_state == nav_executor::MotionState::DEAD;
}

} // anonymous namespace

namespace nav_executor {

void NavExecutorNode::record_input_rejection(const uint8_t reason) {
    last_input_rejection_reason_ = reason;
    ++input_rejection_count_;
}

void NavExecutorNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    ++chassis_state_sequence_;
    if (!std::isfinite(msg->velocity) || !std::isfinite(msg->omega)
        || !std::isfinite(msg->leg_h) || !std::isfinite(msg->leg_psi)
        || msg->leg_mode > static_cast<uint8_t>(LegMode::ABNORMAL)) {
        record_input_rejection(interfaces::msg::NavExecutorDiag::INPUT_REJECTION_CHASSIS_INVALID);
        RCLCPP_ERROR(get_logger(), "Ignoring invalid chassis status");
        chassis_state_valid_ = false;
        return;
    }
    const auto receive_stamp = std::chrono::steady_clock::now();
    if (last_chassis_status_time_ != std::chrono::steady_clock::time_point{}
        && receive_stamp - last_chassis_status_time_ >= chassis_status_timeout_) {
        chassis_state_history_discontinuous_ = true;
    }

    ChassisMotionState received_state;
    received_state.velocity = msg->velocity;
    received_state.omega = msg->omega;
    received_state.leg_h = msg->leg_h;
    received_state.leg_psi = msg->leg_psi;
    chassis_state_ = received_state;
    chassis_leg_mode_ = msg->leg_mode;
    rfr_pwr_limit_ = static_cast<double>(msg->rfr_pwr_limit);
    remaining_energy_supercap_filtered_ = remaining_energy_filter_alpha_ * static_cast<double>(msg->remaining_energy_supercap)
        + (1.0 - remaining_energy_filter_alpha_) * remaining_energy_supercap_filtered_;
    remaining_energy_buffercap_filtered_ = remaining_energy_filter_alpha_ * static_cast<double>(msg->remaining_energy_buffercap)
        + (1.0 - remaining_energy_filter_alpha_) * remaining_energy_buffercap_filtered_;

    if (pending_chassis_state_samples_.size() >= chassis_state_queue_capacity_) {
        pending_chassis_state_samples_.pop_front();
        chassis_state_history_discontinuous_ = true;
        ++chassis_state_queue_overflow_count_;
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "Chassis state queue overflowed (%zu slots, %lu total); dropping oldest sample",
            chassis_state_queue_capacity_,
            static_cast<unsigned long>(chassis_state_queue_overflow_count_)
        );
    }
    pending_chassis_state_samples_.push_back(ChassisStateSample {
        .state = received_state,
        .sequence = chassis_state_sequence_,
    });
    chassis_state_valid_ = true;
    last_chassis_status_time_ = receive_stamp;
}

void NavExecutorNode::spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg) {
    switch (msg->spin_mode) {
        case 0: spin_state_ = SpinState::STOP; break;
        case 1: spin_state_ = SpinState::SPIN_SLOW; break;
        case 2: spin_state_ = SpinState::SPIN_FAST; break;
        default:
            record_input_rejection(interfaces::msg::NavExecutorDiag::INPUT_REJECTION_SPIN_MODE_INVALID);
            RCLCPP_ERROR(get_logger(), "Invalid spin_mode: %d", msg->spin_mode);
            return;
    }
    spin_high_priority_ = msg->high_priority;
}

void NavExecutorNode::local_cost_maps_callback(const interfaces::msg::CostMaps::SharedPtr msg) {
    if (!global_cost_map_) {
        record_input_rejection(interfaces::msg::NavExecutorDiag::INPUT_REJECTION_LOCAL_COST_MAP_BEFORE_GLOBAL_MAP);
        return;
    }
    if (msg->maps.empty()) {
        record_input_rejection(interfaces::msg::NavExecutorDiag::INPUT_REJECTION_LOCAL_COST_MAP_EMPTY);
        return;
    }
    std::vector<CostMap::Ptr> received_maps;
    received_maps.reserve(msg->maps.size());
    for (size_t i = 0; i < msg->maps.size(); ++i) {
        try {
            auto map = std::make_shared<CostMap>(msg->maps[i]);
            if (!map->geometry.same_geometry(global_cost_map_->geometry)) {
                throw std::invalid_argument("geometry differs from the global cost map");
            }
            received_maps.push_back(std::move(map));
        } catch (const std::exception& error) {
            record_input_rejection(i == 0
                ? interfaces::msg::NavExecutorDiag::INPUT_REJECTION_LOCAL_COST_MAP_SIZE_MISMATCH
                : interfaces::msg::NavExecutorDiag::INPUT_REJECTION_PREDICTION_MAP_SIZE_MISMATCH);
            RCLCPP_ERROR(
                get_logger(), "Rejected invalid %s cost map at index %zu: %s",
                i == 0 ? "current" : "predicted", i, error.what()
            );
            return;
        }
    }
    if (!std::isfinite(msg->prediction_dt) || msg->prediction_dt < 0.0
        || (msg->maps.size() > 1 && msg->prediction_dt <= 0.0)) {
        record_input_rejection(interfaces::msg::NavExecutorDiag::INPUT_REJECTION_PREDICTION_DT_INVALID);
        return;
    }
    prediction_dt_ = msg->prediction_dt;

    current_cost_map_ = received_maps.front();

    prediction_maps_.clear();
    for (size_t i = 1; i < received_maps.size(); ++i) {
        prediction_maps_.push_back(received_maps[i]);
    }

    refresh_planner_obstacles();
}

// ROS 回调只写缓存；任务/运动状态转移在 control_tick 中完成。

void NavExecutorNode::control_tick() {
    const auto control_stamp = std::chrono::steady_clock::now();
    const rclcpp::Time diagnostic_stamp = now();
    ++control_cycle_;

    const auto publish_gate = [&](const uint8_t cycle_result) {
        publish_diagnostics(
            cycle_result,
            diagnostic_stamp,
            task_->diagnostics(control_stamp),
            std::nullopt,
            nullptr,
            task_->active_path()
        );
    };

    if (!global_cost_map_) {
        publish_gate(interfaces::msg::NavExecutorDiag::CYCLE_WAITING_GLOBAL_COST_MAP);
        return;
    }
    if (!global_direction_map_) {
        publish_gate(interfaces::msg::NavExecutorDiag::CYCLE_WAITING_DIRECTION_MAP);
        return;
    }
    if (!route_terrain_mask_ready_) {
        publish_gate(interfaces::msg::NavExecutorDiag::CYCLE_WAITING_STEP_MASK);
        return;
    }
    if (!chassis_state_valid_) {
        publish_gate(interfaces::msg::NavExecutorDiag::CYCLE_INVALID_CHASSIS_STATE);
        return;
    }
    // 控制 tick 与状态流解耦：没有新样本时沿用最新状态，有多个新样本时由
    // PathExecutor 按 FIFO 顺序推进 observer。这里只负责断流 fail-safe。
    if (control_stamp - last_chassis_status_time_ >= chassis_status_timeout_) {
        if (!chassis_state_stale_) {
            executor_->notify_chassis_state_unavailable();
            pending_chassis_state_samples_.clear();
            chassis_state_history_discontinuous_ = true;
            chassis_state_stale_ = true;
        }
        publish_gate(interfaces::msg::NavExecutorDiag::CYCLE_NO_NEW_CHASSIS_STATE);
        return;
    }
    chassis_state_stale_ = false;

    Eigen::Vector3d chassis_pose_map;
    if (!get_chassis_pose(chassis_pose_map)) {
        publish_gate(interfaces::msg::NavExecutorDiag::CYCLE_CHASSIS_POSE_UNAVAILABLE);
        return;
    }

    const auto stamp = control_stamp;

    const ObstacleLayers obstacle_layers {
        .global_static = global_cost_map_,
        .dynamic_current = current_cost_map_,
        .dynamic_predictions = prediction_maps_,
        .base_direction = global_direction_map_,
    };
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
    const MotionState current_motion_state = executor_->motion_state();
    const bool route_tracked = active_path_before_update && route_estimate
        && route_estimate->path == active_path_before_update
        && route_estimate->status == RouteTrackingStatus::TRACKED;
    const StepExecutionPreview step_preview = executor_->preview_step_execution(
        active_path_before_update,
        route_tracked ? route_estimate->arc_length : 0.0,
        route_tracked
    );
    const ControlOwner owner_before_task = arbitrate_control({
        .navigation_ready = route_tracked || task_->hold_goal().has_value(),
        .spin_requested = spin_state_ != SpinState::STOP,
        .spin_high_priority = spin_high_priority_,
    });
    const NavigationAccess navigation_access = arbitrate_navigation_access(
        owner_before_task,
        step_preview.phase == StepPhase::COMMITTED,
        navigation_planning_suspended(current_motion_state)
    );

    FollowerObstacleView follower_obstacles = build_follower_obstacle_view(
        obstacle_layers,
        active_path_before_update ? active_path_before_update->step_cost_layer : nullptr,
        active_path_before_update ? active_path_before_update->masked_direction_map : nullptr,
        prediction_dt_, obstacle_occupied_threshold_
    );
    if (!follower_obstacles.hard_route_cost || !follower_obstacles.soft_current_cost
        || !follower_obstacles.route_direction) {
        publish_diagnostics(
            interfaces::msg::NavExecutorDiag::CYCLE_ROUTE_CONTEXT_UNAVAILABLE_BEFORE_TASK,
            diagnostic_stamp,
            task_->diagnostics(control_stamp),
            route_estimate,
            nullptr,
            active_path_before_update
        );
        return;
    }

    TaskUpdateInput task_input;
    task_input.incoming_goal = pending_goal_;
    task_input.feedback = previous_motion_feedback_;
    task_input.navigation_access = navigation_access;
    task_input.stamp = stamp;
    pending_goal_.reset();

    if (planner_obstacles_.hard_cost) {
        task_input.plan_snapshot.current_pos_map = chassis_pose_map.head<2>();
        task_input.plan_snapshot.current_yaw = chassis_pose_map.z();
        task_input.plan_snapshot.current_velocity = chassis_state_.velocity;
        task_input.plan_snapshot.obstacles = planner_obstacles_;
        task_input.plan_snapshot.direction_map = global_direction_map_;
        task_input.plan_snapshot.terrain_constraints = terrain_constraints;
        task_input.plan_snapshot.performance = performance;
    }

    const bool pending_mpc_lethal = previous_motion_feedback_.mpc_lethal
        && previous_motion_feedback_.lethal_path == active_path_before_update;
    const bool route_monitoring_state = pending_mpc_lethal
        || (navigation_access == NavigationAccess::AVAILABLE
            && (current_motion_state == MotionState::FOLLOW
                || current_motion_state == MotionState::PREPARE_SPIN
                || current_motion_state == MotionState::IDLE
                || current_motion_state == MotionState::STEPPING));
    if (active_path_before_update && route_estimate && route_monitoring_state) {
        RouteMonitorInput rm;
        rm.active_path = active_path_before_update;
        rm.route = *route_estimate;
        rm.chassis_pos_map = chassis_pose_map.head<2>();
        rm.obstacles = &follower_obstacles;
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

    follower_obstacles = build_follower_obstacle_view(
        obstacle_layers,
        task_output.command.active_path ? task_output.command.active_path->step_cost_layer : nullptr,
        task_output.command.active_path ? task_output.command.active_path->masked_direction_map : nullptr,
        prediction_dt_, obstacle_occupied_threshold_
    );
    if (!follower_obstacles.hard_route_cost || !follower_obstacles.soft_current_cost
        || !follower_obstacles.route_direction) {
        publish_diagnostics(
            interfaces::msg::NavExecutorDiag::CYCLE_ROUTE_CONTEXT_UNAVAILABLE_AFTER_TASK,
            diagnostic_stamp,
            task_output.diagnostics,
            route_estimate,
            nullptr,
            task_output.command.active_path
        );
        return;
    }

    ExecutorInput ein;
    ein.intent.active_path = task_output.command.active_path;
    ein.intent.hold_goal = task_output.command.hold_goal;
    const bool updated_route_tracked = task_output.command.active_path && route_estimate
        && route_estimate->path == task_output.command.active_path
        && route_estimate->status == RouteTrackingStatus::TRACKED;
    ein.intent.requested_owner = arbitrate_control({
        .navigation_ready = updated_route_tracked || task_output.command.hold_goal.has_value(),
        .spin_requested = spin_state_ != SpinState::STOP,
        .spin_high_priority = spin_high_priority_,
    });
    ein.intent.spin_fast = (spin_state_ == SpinState::SPIN_FAST);
    ein.route = route_estimate;
    ein.observation.chassis_pose_map = chassis_pose_map;
    ein.observation.chassis_state = chassis_state_;
    ein.observation.pending_chassis_state_samples.assign(
        pending_chassis_state_samples_.begin(), pending_chassis_state_samples_.end()
    );
    pending_chassis_state_samples_.clear();
    ein.observation.chassis_state_history_discontinuous =
        std::exchange(chassis_state_history_discontinuous_, false);
    ein.observation.chassis_leg_mode = chassis_leg_mode_;
    ein.observation.comp_stage = comp_stage_;
    ein.observation.stamp = stamp;
    ein.environment.obstacles = &follower_obstacles;

    ExecutorOutput out = executor_->update(ein);

    previous_motion_feedback_.goal_reached = out.goal_reached;
    previous_motion_feedback_.goal_reached_path = out.goal_reached ? task_output.command.active_path : nullptr;
    previous_motion_feedback_.executor_replan_event = out.executor_replan_event;
    previous_motion_feedback_.mpc_lethal = out.mpc_lethal;
    previous_motion_feedback_.lethal_path = out.mpc_lethal ? task_output.command.active_path : nullptr;

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

    publish_diagnostics(
        interfaces::msg::NavExecutorDiag::CYCLE_EXECUTED,
        diagnostic_stamp,
        task_output.diagnostics,
        route_estimate,
        &out,
        task_output.command.active_path
    );

    if (enable_debug_ && debug_final_cost_map_pub_) {
        nav_msgs::msg::OccupancyGrid grid_msg;
        grid_msg.header.stamp = now();
        grid_msg.header.frame_id = "map";
        const GridGeometry& geometry = follower_obstacles.soft_current_cost->geometry;
        grid_msg.info.width = static_cast<uint32_t>(geometry.width());
        grid_msg.info.height = static_cast<uint32_t>(geometry.height());
        grid_msg.info.resolution = static_cast<float>(geometry.resolution());
        grid_msg.info.origin.position.x = geometry.origin().x();
        grid_msg.info.origin.position.y = geometry.origin().y();
        grid_msg.info.origin.orientation.w = 1.0;
        grid_msg.data.resize(follower_obstacles.soft_current_cost->data.size());
        for (size_t idx = 0; idx < follower_obstacles.soft_current_cost->data.size(); idx++) {
            grid_msg.data[idx] = static_cast<int8_t>(follower_obstacles.soft_current_cost->data[idx]);
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
    const uint8_t cycle_result,
    const rclcpp::Time& stamp,
    const TaskDiagnostics& diag,
    const std::optional<RouteEstimate>& route,
    const ExecutorOutput* const executor_output,
    const AnnotatedPath::ConstPtr& active_path
) {
    if (cycle_result == interfaces::msg::NavExecutorDiag::CYCLE_EXECUTED
        && active_path && global_path_pub_) {
        global_path_pub_->publish(trajectory_to_nav_msg(active_path->trajectory));
    }
    if (!enable_debug_ || !debug_diag_pub_) return;

    if (cycle_result == interfaces::msg::NavExecutorDiag::CYCLE_EXECUTED) {
        if (!diag.debug_spatial_path.empty() && debug_spatial_path_pub_) {
            debug_spatial_path_pub_->publish(path_to_nav_msg(diag.debug_spatial_path));
        }
        if (!diag.debug_smoothed_spatial_path.empty()
            && debug_smoothed_spatial_path_pub_) {
            debug_smoothed_spatial_path_pub_->publish(
                path_to_nav_msg(diag.debug_smoothed_spatial_path)
            );
        }
        if (!diag.debug_kino_path.empty() && debug_kino_path_pub_) {
            debug_kino_path_pub_->publish(path_to_nav_msg(diag.debug_kino_path));
        }
        if (active_path && debug_minco_trajectory_pub_) {
            debug_minco_trajectory_pub_->publish(trajectory_to_marker(*active_path));
        }
        if (executor_output && executor_output->mpc_path_map && debug_mpc_path_pub_) {
            debug_mpc_path_pub_->publish(path_to_nav_msg(*executor_output->mpc_path_map));
        }
    }

    interfaces::msg::NavExecutorDiag msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "map";
    msg.control_cycle = control_cycle_;
    msg.cycle_result = cycle_result;

    msg.last_input_rejection_reason = last_input_rejection_reason_;
    msg.input_rejection_count = input_rejection_count_;

    msg.chassis_state_sequence = chassis_state_sequence_;
    msg.chassis_state_valid = chassis_state_valid_;
    msg.chassis_velocity = chassis_state_.velocity;
    msg.chassis_angular_velocity = chassis_state_.omega;
    msg.chassis_leg_height = chassis_state_.leg_h;
    msg.chassis_leg_psi = chassis_state_.leg_psi;
    msg.chassis_leg_mode = chassis_leg_mode_;
    msg.comp_stage = comp_stage_;
    msg.chassis_control_state = chassis_control_state_value(chassis_leg_mode_, comp_stage_);
    msg.remaining_energy_supercap = remaining_energy_supercap_filtered_;
    msg.remaining_energy_buffercap = remaining_energy_buffercap_filtered_;
    msg.rfr_power_limit = rfr_pwr_limit_;
    msg.high_performance_available =
        remaining_energy_buffercap_filtered_ >= traversal_configuration_.high_performance_buffercap_threshold
        && remaining_energy_supercap_filtered_ >= traversal_configuration_.high_performance_supercap_threshold
        && rfr_pwr_limit_ >= traversal_configuration_.high_performance_rfr_pwr_limit_threshold;
    msg.spin_requested = spin_state_ != SpinState::STOP;
    msg.spin_high_priority = spin_high_priority_;
    msg.spin_fast = spin_state_ == SpinState::SPIN_FAST;
    msg.current_cost_map_available = static_cast<bool>(current_cost_map_);
    msg.prediction_map_count = static_cast<uint32_t>(prediction_maps_.size());
    msg.prediction_dt = prediction_dt_;

    msg.goal_id = diag.goal_id;
    msg.goal_x = diag.goal_position.x();
    msg.goal_y = diag.goal_position.y();
    msg.goal_fixed = diag.goal_fixed;
    msg.active_path_goal_id = diag.active_path_goal_id;
    msg.has_hold_goal = diag.has_hold_goal;
    msg.hold_goal_x = diag.hold_goal_position.x();
    msg.hold_goal_y = diag.hold_goal_position.y();
    msg.plan_generation = diag.plan_generation;
    msg.needs_plan = diag.needs_plan;
    msg.planner_state = static_cast<uint8_t>(diag.planner_state);
    msg.planner_cooldown_remaining = diag.planner_cooldown_remaining;
    msg.planner_last_result = static_cast<uint8_t>(diag.planner_last_result);
    msg.planner_last_failure_reason = diag.planner_last_failure_reason;
    msg.last_replan_reason = static_cast<uint8_t>(diag.last_replan_reason);
    msg.replan_count = diag.replan_count;

    if (route) {
        msg.route_status = route->status == RouteTrackingStatus::TRACKED
            ? interfaces::msg::NavExecutorDiag::ROUTE_TRACKED
            : interfaces::msg::NavExecutorDiag::ROUTE_LOST;
        msg.route_progress = route->arc_length;
        msg.route_path_speed = route->path_speed;
        msg.route_remaining_length = route->remaining_length;
        msg.route_tracking_error = route->tracking_error;
    } else {
        msg.route_status = interfaces::msg::NavExecutorDiag::ROUTE_NONE;
    }

    if (executor_output) {
        msg.motion_state = static_cast<uint8_t>(executor_output->motion_state);
        msg.step_phase = static_cast<uint8_t>(executor_output->step_phase);
        msg.command_status = static_cast<uint8_t>(executor_output->command_status);
        msg.command_velocity = executor_output->velocity;
        msg.command_angular_velocity = executor_output->omega;
        msg.command_mode = executor_output->mode;
        msg.command_step_distance = executor_output->step_dist_cm;
    } else {
        msg.motion_state = static_cast<uint8_t>(executor_->motion_state());
        msg.step_phase = static_cast<uint8_t>(executor_->step_phase());
        msg.command_status = interfaces::msg::NavExecutorDiag::COMMAND_NOT_EVALUATED;
    }

    const ObserverDiagnostics observer = executor_output
        ? executor_output->observer_diagnostics
        : ObserverDiagnostics {};
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

    msg.mpc_attempted = executor_output && executor_output->mpc_diagnostics.has_value();
    if (executor_output && executor_output->mpc_diagnostics) {
        const MPCDiagnostics& mpc = *executor_output->mpc_diagnostics;
        msg.mpc_mode = static_cast<uint8_t>(mpc.solver_mode);
        msg.mpc_succeeded = mpc.solve_succeeded;
        msg.mpc_error = mpc.solve_error;
        msg.mpc_solve_time_ms = mpc.solve_time_ms;

        msg.ancillary_enabled = mpc.ancillary_enabled;
        msg.ancillary_active = mpc.ancillary_active;
        msg.ancillary_reanchored = mpc.nominal_reanchored;
        msg.ancillary_tube_feasible = mpc.first_command_tube_feasible;

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

        msg.mpc_solver_iters = mpc.solver_termination.iters;
        msg.mpc_solver_converged = mpc.solver_termination.converged;
        msg.mpc_solver_cost = mpc.solver_termination.cost;

        msg.rejected_follow_rollout = mpc.rejected_follow_rollout.has_value();
        if (mpc.rejected_follow_rollout) {
            const RejectedFollowRollout& rejected = *mpc.rejected_follow_rollout;
            msg.rejected_lethal_state_index = rejected.lethal.state_index;
            msg.rejected_lethal_position_x = rejected.lethal.position_map.x();
            msg.rejected_lethal_position_y = rejected.lethal.position_map.y();
            msg.rejected_lethal_sampled_cost = rejected.lethal.sampled_cost;
            msg.rejected_consecutive_count = rejected.consecutive_count;
            msg.rejected_replan_requested = rejected.replan_requested;
            msg.rejected_prediction_x.reserve(rejected.prediction.path_map.size());
            msg.rejected_prediction_y.reserve(rejected.prediction.path_map.size());
            for (const Eigen::Vector2d& point : rejected.prediction.path_map) {
                msg.rejected_prediction_x.push_back(point.x());
                msg.rejected_prediction_y.push_back(point.y());
            }
        }
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
