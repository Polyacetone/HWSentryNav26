#include <path_follower/navigation_controller.hpp>

#include <chrono>
#include <rclcpp/logging.hpp>

namespace path_follower {

// ═══════════════════════ 构造函数 ════════════════════════════

NavigationController::NavigationController(
    const NavigationParams& nav_params,
    const FsmParams& fsm_params,
    std::shared_ptr<MPCController> mpc_controller,
    rclcpp::Logger logger
) : control_fsm_(std::make_unique<ControlFsm>(fsm_params, logger)),
    mpc_controller_(std::move(mpc_controller)),
    logger_(logger),
    nav_params_(nav_params) {
    last_fsm_state_ = control_fsm_->state();
}

// ═══════════════════════ 主更新接口 ══════════════════════════

ControlOutput NavigationController::update(const ControlInput& input) {
    // 1. 组装 FSM 输入
    FsmInput fsm_input;
    fsm_input.has_path = input.global_path.has_value();
    fsm_input.spin_requested = input.spin_requested;
    fsm_input.spin_high_priority = input.spin_high_priority;
    fsm_input.velocity = input.chassis_status.x();
    fsm_input.omega = input.chassis_status.y();
    fsm_input.chassis_pose_map = input.chassis_pose_map;
    fsm_input.merged_cost_map = input.merged_cost_map;
    fsm_input.global_direction_map = input.global_direction_map;
    fsm_input.chassis_theta_imu_world = input.chassis_theta_imu_world;
    fsm_input.stamp = input.stamp;

    // 2. FSM 状态决策
    const FsmOutput fsm_output = control_fsm_->update(fsm_input);
    const FsmState state = fsm_output.state;

    // 3. 日志：状态变化
    if (state != last_fsm_state_) {
        static constexpr const char* names[] = {"IDLE", "FOLLOW", "SPIN", "STOPPING", "HAZARD_RECOVERY", "STUCK_REVERSE"};
        RCLCPP_INFO(logger_, "Control FSM -> %s", names[static_cast<int>(state)]);
        last_fsm_state_ = state;
    }

    // 4. 根据 FSM 状态，执行对应的控制逻辑
    ControlOutput output;
    switch (state) {
        case FsmState::IDLE: output = execute_idle(input); break;
        case FsmState::FOLLOW: output = execute_follow(input); break;
        case FsmState::SPIN: output = execute_spin(input); break;
        case FsmState::STOPPING: output = execute_stop(input); break;
        case FsmState::HAZARD_RECOVERY: output = execute_recovery(input, fsm_output); break;
        case FsmState::STUCK_REVERSE: output = execute_stuck_reverse(input, fsm_output); break;
    }

    output.fsm_state = state;
    output.path_cleared |= fsm_output.clear_global_path;

    // 5. 非 IDLE 状态清除保持角度
    if (state != FsmState::IDLE) {
        theta_keep_imu_world_ = std::nullopt;
    }

    // 6. 回调卡住检测
    if (output.valid) {
        control_fsm_->on_chassis_cmd_published(output.velocity, output.omega, input.stamp);
    }

    return output;
}

FsmState NavigationController::fsm_state() const {
    return control_fsm_->state();
}

// ═══════════════════ IDLE: 保持静止 ══════════════════════════

ControlOutput NavigationController::execute_idle(const ControlInput& input) {
    ControlOutput out;

    if (!theta_keep_imu_world_) theta_keep_imu_world_ = input.chassis_theta_imu_world;
    if (!theta_keep_imu_world_) return out;

    out.velocity = 0.0;
    out.theta_imu_world = *theta_keep_imu_world_;
    out.omega = 0.0;
    out.valid = true;
    return out;
}

// ═══════════════════ FOLLOW: 跟随路径 ════════════════════════

ControlOutput NavigationController::execute_follow(const ControlInput& input) {
    ControlOutput out;

    if (!input.global_path || !input.merged_cost_map || !input.global_direction_map) return out;
    if (!input.map_to_imu_world_yaw) return out;

    // 投影当前位置到样条
    const double u0 = project_to_spline_u(
        *input.global_path, input.chassis_pose_map.head<2>(), last_reference_u_,
        mpc_controller_->params().follow_projection.proj_num_samples,
        mpc_controller_->params().follow_projection.proj_search_window,
        mpc_controller_->params().follow_projection.max_correspondence_distance
    );
    last_reference_u_ = u0;

    // 调用 MPC
    auto start_time = std::chrono::high_resolution_clock::now();
    const auto result = mpc_controller_->follow_path(
        *input.global_path, input.chassis_pose_map, input.chassis_status,
        *input.merged_cost_map, *input.global_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCController(Follow) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    const double dt = mpc_controller_->params().dt;
    if (solve_ms > dt * 500.0) {
        RCLCPP_WARN(logger_, "MPCController(Follow) solve time %.2f ms > %.2f ms", solve_ms, dt * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCController(Follow) solve time: %.2f ms", solve_ms);
    }

    // 到达目标点 → 标记清除路径
    if ((input.chassis_pose_map.head<2>() - input.global_path->evaluate(1.0)).norm() < nav_params_.stop_threshold_dist || u0 > nav_params_.stop_threshold_u) {
        RCLCPP_INFO(logger_, "Reached goal, currently at (%.2f, %.2f)", input.chassis_pose_map.x(), input.chassis_pose_map.y());
        out.path_cleared = true;
        last_reference_u_ = 0.0;
    }

    // 坐标变换: map theta → imu_world theta
    const double theta_imu_world = wrap_pi(std::get<0>(*result).y() - *input.map_to_imu_world_yaw);

    // 台阶检测
    const auto [step_up, step_down] = detect_steps_on_spline(input, u0);

    out.velocity = std::get<0>(*result).x();
    out.theta_imu_world = theta_imu_world;
    out.omega = std::get<0>(*result).z();
    out.step_up_ahead = step_up;
    out.step_down_ahead = step_down;
    out.predicted_path_map = std::get<1>(*result);
    out.valid = true;
    return out;
}

// ═══════════════════ SPIN: 小陀螺 ════════════════════════════

ControlOutput NavigationController::execute_spin(const ControlInput& input) {
    ControlOutput out;
    out.slow_spin = input.spin_slow;
    out.fast_spin = input.spin_fast;
    out.valid = true;
    return out;
}

// ═══════════════════ STOPPING: 平滑减速 ══════════════════════

ControlOutput NavigationController::execute_stop(const ControlInput& input) {
    ControlOutput out;

    if (!input.merged_cost_map || !input.global_direction_map) return out;
    if (!input.map_to_imu_world_yaw) return out;

    auto start_time = std::chrono::high_resolution_clock::now();
    const auto result = mpc_controller_->stop(
        input.chassis_pose_map, input.chassis_status,
        *input.merged_cost_map, *input.global_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCController(Stop) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    const double dt = mpc_controller_->params().dt;
    if (solve_ms > dt * 500.0) {
        RCLCPP_WARN(logger_, "MPCController(Stop) solve time %.2f ms > %.2f ms", solve_ms, dt * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCController(Stop) solve time: %.2f ms", solve_ms);
    }

    const double theta_imu_world = wrap_pi(std::get<0>(*result).y() - *input.map_to_imu_world_yaw);

    out.velocity = std::get<0>(*result).x();
    out.theta_imu_world = theta_imu_world;
    out.omega = std::get<0>(*result).z();
    out.predicted_path_map = std::get<1>(*result);
    out.valid = true;
    return out;
}

// ═══════════════════ HAZARD_RECOVERY: MPC 驱动恢复 ═══════════

ControlOutput NavigationController::execute_recovery(const ControlInput& input, const FsmOutput& fsm_output) {
    ControlOutput out;

    if (!fsm_output.recovery_goal_map) return out;
    if (!input.merged_cost_map || !input.global_direction_map) return out;
    if (!input.map_to_imu_world_yaw) return out;

    auto start_time = std::chrono::high_resolution_clock::now();
    const auto result = mpc_controller_->recover_to_point(
        *fsm_output.recovery_goal_map,
        input.chassis_pose_map, input.chassis_status,
        *input.merged_cost_map, *input.global_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCController(Recovery) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    const double dt = mpc_controller_->params().dt;
    if (solve_ms > dt * 500.0) {
        RCLCPP_WARN(logger_, "MPCController(Recovery) solve time %.2f ms > %.2f ms", solve_ms, dt * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCController(Recovery) solve time: %.2f ms", solve_ms);
    }

    const double theta_imu_world = wrap_pi(std::get<0>(*result).y() - *input.map_to_imu_world_yaw);

    out.velocity = std::get<0>(*result).x();
    out.theta_imu_world = theta_imu_world;
    out.omega = std::get<0>(*result).z();
    out.predicted_path_map = std::get<1>(*result);
    out.valid = true;
    return out;
}

// ═══════════════════ STUCK_REVERSE: 倒车脱困 ═════════════════

ControlOutput NavigationController::execute_stuck_reverse(const ControlInput& input, const FsmOutput& fsm_output) {
    (void)input;
    ControlOutput out;

    if (!fsm_output.reverse_cmd) return out;

    out.velocity = fsm_output.reverse_cmd->velocity;
    out.theta_imu_world = fsm_output.reverse_cmd->theta_imu_world;
    out.omega = 0.0;
    out.valid = true;
    return out;
}

// ═══════════════════ 台阶检测 ════════════════════════════════

std::tuple<bool, bool> NavigationController::detect_steps_on_spline(const ControlInput& input, const double u0) const {
    constexpr double dir_norm_threshold = 0.5;
    constexpr double dot_threshold = 0.5;

    if (!input.global_direction_map || !input.global_path) return {false, false};
    if (nav_params_.step_check_front <= 0.0 && nav_params_.step_check_back <= 0.0) return {false, false};
    if (nav_params_.step_check_sample_step <= 1e-6) return {false, false};

    const Eigen::Vector2d heading(
        std::cos(input.chassis_pose_map.z()),
        std::sin(input.chassis_pose_map.z())
    );
    bool step_up = false, step_down = false;

    const auto& path = *input.global_path;
    const auto& dir_map = *input.global_direction_map;

    const auto sample_at_u = [&](double u) {
        const Eigen::Vector2d p_map = path.evaluate(u);
        const Eigen::Vector2d g = dir_map.map_coord_to_grid(p_map);
        const Eigen::Vector2d dir = dir_map.interpolate(g);
        const double n = dir.norm();
        if (n < dir_norm_threshold) return;
        const double dot = dir.normalized().dot(heading);
        if (dot > dot_threshold) step_up = true;
        if (dot < -dot_threshold) step_down = true;
    };

    sample_at_u(u0);

    // 向前采样
    double u_fwd = u0, dist_fwd = 0.0;
    const double target_fwd = std::max(0.0, nav_params_.step_check_front);
    while (dist_fwd + 1e-9 < target_fwd && u_fwd < 1.0 - 1e-9 && !(step_up && step_down)) {
        const Eigen::Vector2d d1 = path.derivative(u_fwd, 1);
        const double dsdu = std::max(1e-6, d1.norm());
        const double du = nav_params_.step_check_sample_step / dsdu;
        u_fwd = std::min(1.0, u_fwd + du);
        dist_fwd += nav_params_.step_check_sample_step;
        sample_at_u(u_fwd);
    }

    // 向后采样
    double u_bwd = u0, dist_bwd = 0.0;
    const double target_bwd = std::max(0.0, nav_params_.step_check_back);
    while (dist_bwd + 1e-9 < target_bwd && u_bwd > 1e-9 && !(step_up && step_down)) {
        const Eigen::Vector2d d1 = path.derivative(u_bwd, 1);
        const double dsdu = std::max(1e-6, d1.norm());
        const double du = nav_params_.step_check_sample_step / dsdu;
        u_bwd = std::max(0.0, u_bwd - du);
        dist_bwd += nav_params_.step_check_sample_step;
        sample_at_u(u_bwd);
    }

    return {step_up, step_down};
}

}