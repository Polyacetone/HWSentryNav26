#include <path_follower/main_controller.hpp>

#include <chrono>
#include <algorithm>
#include <limits>
#include <numbers>
#include <rclcpp/logging.hpp>

namespace path_follower {

namespace {

inline bool is_chassis_dead(const uint8_t leg_mode, const uint8_t comp_status) {
    // leg_mode: 0:死亡, 1:恢复, 6:异常
    // comp_status: 4:比赛中
    return leg_mode == 0u || leg_mode == 1u || leg_mode == 6u || comp_status != 4u;
}

}

namespace {

struct RecoveryGoalPlanner {
    struct FieldSample {
        double cost = 0.0;
        double step_norm = 0.0;
    };

    static std::optional<FieldSample> sample_fields(
        const CostMap& cost_map,
        const DirectionMap& dir_map,
        const Eigen::Vector2d& pos
    ) {
        const Eigen::Vector2d gc = cost_map.map_coord_to_grid(pos);
        if (!cost_map.is_valid_coord(gc)) return std::nullopt;
        const Eigen::Vector2d gd = dir_map.map_coord_to_grid(pos);
        if (!dir_map.is_valid_coord(gd)) return std::nullopt;

        FieldSample s;
        s.cost = cost_map.interpolate(gc);
        s.step_norm = dir_map.interpolate(gd).norm();
        return s;
    }

    static bool is_safe_goal(const RecoveryParams& p, const FieldSample& s) {
        return (s.cost < p.safe_cost_threshold) && (s.step_norm < p.safe_step_norm_threshold);
    }

    static double potential_cost(const FieldSample& s) {
        const double cost01 = std::clamp(s.cost / 255.0, 0.0, 1.0);
        return cost01 + s.step_norm;
    }

    struct PathScore {
        double score = std::numeric_limits<double>::infinity();
        bool end_safe = false;
        FieldSample end_sample;
    };

    static std::optional<PathScore> score_candidate_by_path_integral(
        const RecoveryParams& p,
        const CostMap& cost_map,
        const DirectionMap& dir_map,
        const Eigen::Vector2d& origin,
        const Eigen::Vector2d& goal
    ) {
        double acc = 0.0;
        std::optional<FieldSample> end_s;
        for (int i = 0; i <= p.circ_radius_samples; i++) {
            const double t = static_cast<double>(i) / static_cast<double>(p.circ_radius_samples);
            const Eigen::Vector2d pos = origin + (goal - origin) * t;
            const auto s = sample_fields(cost_map, dir_map, pos);
            if (!s) return std::nullopt;
            acc += potential_cost(*s);
            if (i == p.circ_radius_samples) end_s = s;
        }

        PathScore out;
        out.score = acc;
        out.end_sample = *end_s;
        out.end_safe = is_safe_goal(p, *end_s);
        return out;
    }

    static std::optional<Eigen::Vector2d> find_goal(
        const RecoveryParams& p,
        const CostMap& cost_map,
        const DirectionMap& dir_map,
        const Eigen::Vector3d& chassis_pose
    ) {
        const Eigen::Vector2d origin = chassis_pose.head<2>();

        std::optional<Eigen::Vector2d> best_pt;
        std::optional<PathScore> best_sc;

        for (int i = 0; i < p.circ_angle_samples; i++) {
            const double a = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(p.circ_angle_samples);
            const Eigen::Vector2d pt = origin + Eigen::Vector2d(std::cos(a), std::sin(a)) * p.circ_radius;
            const auto sc = score_candidate_by_path_integral(p, cost_map, dir_map, origin, pt);
            if (!sc) continue;
            if (!best_sc || sc->score < best_sc->score) {
                best_sc = *sc;
                best_pt = pt;
            }
        }

        return best_pt;
    }
};

}  // namespace

// ═══════════════════════ 构造函数 ════════════════════════════

MainController::MainController(
    const NavigationParams& nav_params,
    const FsmParams& fsm_params,
    std::shared_ptr<MPCSolver> mpc_controller,
    rclcpp::Logger logger
) : control_fsm_(std::make_unique<StateMachine>(fsm_params, logger)),
    mpc_controller_(std::move(mpc_controller)),
    logger_(logger),
    nav_params_(nav_params),
    fsm_params_(fsm_params) {
    last_fsm_state_ = control_fsm_->state();
}

// ═══════════════════════ 主更新接口 ══════════════════════════

ControlOutput MainController::update(const ControlInput& input) {
    const FsmState prev_state = last_fsm_state_;
    mpc_controller_->set_last_cmd(last_cmd_);

    // 1. 组装 FSM 输入
    FsmInput fsm_input;
    fsm_input.has_path = input.global_path.has_value();
    fsm_input.spin_requested = input.spin_requested;
    fsm_input.spin_high_priority = input.spin_high_priority;
    fsm_input.chassis_dead = is_chassis_dead(input.chassis_leg_mode, input.comp_stage);
    fsm_input.velocity = input.chassis_status.x();
    fsm_input.omega = input.chassis_status.y();
    fsm_input.chassis_pose_map = input.chassis_pose_map;
    fsm_input.merged_cost_map = input.merged_cost_map;
    fsm_input.global_direction_map = input.global_direction_map;
    fsm_input.recovery_goal_map = recovery_goal_map_;
    fsm_input.stamp = input.stamp;

    // 2. FSM 状态决策
    const FsmOutput fsm_output = control_fsm_->update(fsm_input);
    const FsmState state = fsm_output.state;
    on_state_transition(prev_state, state);
    last_fsm_state_ = state;

    // 3. 根据 FSM 状态，执行对应的控制逻辑
    ControlOutput output;
    switch (state) {
        case FsmState::DEAD: output = execute_dead(input); break;
        case FsmState::IDLE: output = execute_idle(input); break;
        case FsmState::FOLLOW: output = execute_follow(input); break;
        case FsmState::SPIN: output = execute_spin(input); break;
        case FsmState::STOPPING: output = execute_stop(input); break;
        case FsmState::HAZARD_RECOVERY: output = execute_recovery(input); break;
        case FsmState::STUCK_REVERSE: output = execute_stuck_reverse(input); break;
    }

    output.fsm_state = state;
    output.path_cleared |= fsm_output.clear_global_path;

    // 4. 同步已发布指令到 FSM / MPC，并在非 MPC 状态时重置 MPC 的 warm start
    if (output.valid) {
        last_cmd_ = Eigen::Vector2d(output.velocity, output.omega);
        control_fsm_->on_chassis_cmd_published(last_cmd_.x(), last_cmd_.y(), input.stamp);
        mpc_controller_->set_last_cmd(last_cmd_);
        const bool non_mpc_state = (state == FsmState::DEAD) || (state == FsmState::IDLE) || (state == FsmState::SPIN) || (state == FsmState::STUCK_REVERSE);
        if (non_mpc_state) {
            mpc_controller_->reset_warm_start();
        }
    }

    return output;
}

// ═══════════════════ DEAD: 失效保持静止 ═════════════════════

ControlOutput MainController::execute_dead(const ControlInput& input) {
    (void)input;
    ControlOutput out;
    out.velocity = 0.0;
    out.omega = 0.0;
    out.valid = true;
    return out;
}

FsmState MainController::fsm_state() const {
    return control_fsm_->state();
}

// ═══════════════════ IDLE: 保持静止 ══════════════════════════

ControlOutput MainController::execute_idle(const ControlInput& input) {
    (void)input;
    ControlOutput out;
    out.velocity = 0.0;
    out.omega = 0.0;
    out.valid = true;
    return out;
}

// ═══════════════════ FOLLOW: 跟随路径 ════════════════════════

ControlOutput MainController::execute_follow(const ControlInput& input) {
    ControlOutput out;
    if (!input.global_path || !input.merged_cost_map || !input.global_direction_map) return out;

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
        RCLCPP_ERROR(logger_, "MPCSolver(Follow) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    const double dt = mpc_controller_->params().dt;
    if (solve_ms > dt * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Follow) solve time %.2f ms > %.2f ms", solve_ms, dt * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCSolver(Follow) solve time: %.2f ms", solve_ms);
    }

    // 到达目标点 → 标记清除路径
    if ((input.chassis_pose_map.head<2>() - input.global_path->evaluate(1.0)).norm() < nav_params_.stop_threshold_dist || u0 > nav_params_.stop_threshold_u) {
        RCLCPP_INFO(logger_, "Reached goal, currently at (%.2f, %.2f)", input.chassis_pose_map.x(), input.chassis_pose_map.y());
        out.path_cleared = true;
        last_reference_u_ = 0.0;
    }

    // 台阶检测
    const auto [step_up, step_down] = detect_steps_on_spline(input, u0);

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.step_up_ahead = step_up;
    out.step_down_ahead = step_down;
    out.predicted_path_map = std::get<1>(*result);
    out.valid = true;
    return out;
}

// ═══════════════════ SPIN: 小陀螺 ════════════════════════════

ControlOutput MainController::execute_spin(const ControlInput& input) {
    ControlOutput out;
    out.slow_spin = input.spin_slow;
    out.fast_spin = input.spin_fast;
    out.valid = true;
    return out;
}

// ═══════════════════ STOPPING: 平滑减速 ══════════════════════

ControlOutput MainController::execute_stop(const ControlInput& input) {
    ControlOutput out;
    if (!input.merged_cost_map || !input.global_direction_map) return out;

    auto start_time = std::chrono::high_resolution_clock::now();
    const auto result = mpc_controller_->stop(
        input.chassis_pose_map, input.chassis_status,
        *input.merged_cost_map, *input.global_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Stop) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    const double dt = mpc_controller_->params().dt;
    if (solve_ms > dt * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Stop) solve time %.2f ms > %.2f ms", solve_ms, dt * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCSolver(Stop) solve time: %.2f ms", solve_ms);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.predicted_path_map = std::get<1>(*result);
    out.valid = true;
    return out;
}

void MainController::on_state_transition(const FsmState prev, const FsmState next) {
    if (prev == next) return;

    // DEAD 只是临时冻结；不要让它打断 HAZARD_RECOVERY 的恢复目标/计时
    if (next == FsmState::HAZARD_RECOVERY && prev != FsmState::DEAD) {
        recovery_goal_map_ = std::nullopt;
        recovery_goal_set_time_ = rclcpp::Time{0, 0, RCL_ROS_TIME};
    }
    if (prev == FsmState::HAZARD_RECOVERY && next != FsmState::HAZARD_RECOVERY && next != FsmState::DEAD) {
        recovery_goal_map_ = std::nullopt;
        recovery_goal_set_time_ = rclcpp::Time{0, 0, RCL_ROS_TIME};
    }
}

void MainController::update_recovery_goal_if_needed(const ControlInput& input) {
    const auto& p = fsm_params_.recovery;
    if (!p.enable) {
        recovery_goal_map_ = std::nullopt;
        return;
    }

    if (!input.merged_cost_map || !input.global_direction_map) return;

    const bool need_new = (!recovery_goal_map_) || (recovery_goal_set_time_.nanoseconds() == 0) || ((input.stamp - recovery_goal_set_time_).seconds() >= p.goal_timeout);

    if (!need_new) return;

    recovery_goal_map_ = RecoveryGoalPlanner::find_goal(
        p, *input.merged_cost_map, *input.global_direction_map, input.chassis_pose_map
    );
    recovery_goal_set_time_ = input.stamp;

    if (!recovery_goal_map_) {
        RCLCPP_ERROR(logger_, "HAZARD_RECOVERY failed to find a recovery goal");
        return;
    }

    const auto s = RecoveryGoalPlanner::sample_fields(*input.merged_cost_map, *input.global_direction_map, *recovery_goal_map_);
    if (!s) {
        RCLCPP_WARN(logger_, "HAZARD_RECOVERY new goal=(%.2f, %.2f) (field sample invalid)", recovery_goal_map_->x(), recovery_goal_map_->y());
        return;
    }

    if (RecoveryGoalPlanner::is_safe_goal(p, *s)) {
        RCLCPP_WARN(logger_, "HAZARD_RECOVERY new goal=(%.2f, %.2f) (SAFE cost=%.1f step=%.3f)", recovery_goal_map_->x(), recovery_goal_map_->y(), s->cost, s->step_norm);
    } else {
        RCLCPP_WARN(logger_, "HAZARD_RECOVERY new goal=(%.2f, %.2f) (UNSAFE cost=%.1f step=%.3f)", recovery_goal_map_->x(), recovery_goal_map_->y(), s->cost, s->step_norm);
    }
}

// ═══════════════════ HAZARD_RECOVERY: MPC 驱动恢复 ═══════════

ControlOutput MainController::execute_recovery(const ControlInput& input) {
    ControlOutput out;
    if (!input.merged_cost_map || !input.global_direction_map) return out;

    update_recovery_goal_if_needed(input);
    if (!recovery_goal_map_) return out;

    auto start_time = std::chrono::high_resolution_clock::now();
    const auto result = mpc_controller_->recover_to_point(
        *recovery_goal_map_,
        input.chassis_pose_map, input.chassis_status,
        *input.merged_cost_map, *input.global_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Recovery) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    const double dt = mpc_controller_->params().dt;
    if (solve_ms > dt * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Recovery) solve time %.2f ms > %.2f ms", solve_ms, dt * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCSolver(Recovery) solve time: %.2f ms", solve_ms);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.predicted_path_map = std::get<1>(*result);
    out.valid = true;
    return out;
}

// ═══════════════════ STUCK_REVERSE: 倒车脱困 ═════════════════

ControlOutput MainController::execute_stuck_reverse(const ControlInput& input) {
    (void)input;
    ControlOutput out;
    out.velocity = -fsm_params_.stuck.reverse_speed;
    out.omega = 0.0;
    out.valid = true;
    return out;
}

// ═══════════════════ 台阶检测 ════════════════════════════════

std::tuple<bool, bool> MainController::detect_steps_on_spline(const ControlInput& input, const double u0) const {
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