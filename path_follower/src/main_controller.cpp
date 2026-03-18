#include <path_follower/main_controller.hpp>

#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numbers>
#include <rclcpp/logging.hpp>

namespace path_follower {

namespace {

inline bool is_chassis_dead(const uint8_t leg_mode, const uint8_t comp_status) {
    // leg_mode: 4:Mature
    // comp_status: 4:比赛中
    return leg_mode != 4u || comp_status != 4u;
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
    const bool chassis_dead = is_chassis_dead(input.chassis_leg_mode, input.comp_stage);

    // 全局中断优先：底盘 Dead 直接外部拦截，不进入 FSM。
    if (chassis_dead) {
        ControlOutput out;
        out.velocity = 0.0;
        out.omega = 0.0;
        out.fsm_state = FsmState::DEAD;
        out.valid = true;

        last_cmd_ = Eigen::Vector2d::Zero();
        mpc_controller_->set_last_cmd(last_cmd_);
        mpc_controller_->reset_warm_start();
        stuck_active_ = false;
        recovery_safe_since_ = std::nullopt;
        return out;
    }

    const FsmState prev_state = last_fsm_state_;
    mpc_controller_->set_last_cmd(last_cmd_);

    // 0. 更新隐藏状态观测器（在 MPC 求解之前）
    mpc_controller_->update_observer(input.chassis_status.x(), input.chassis_status.y());

    // 0.5 更新能量状态
    mpc_controller_->set_energy_state(input.remaining_energy, input.rfr_pwr_limit);

    if (prev_state == FsmState::HAZARD_RECOVERY) {
        update_recovery_goal_if_needed(input);
    }

    const bool has_path = input.global_path.has_value();
    const bool has_new_path = has_path && input.path_updated;
    const bool dist_reached = has_path && ((input.chassis_pose_map.head<2>() - input.global_path->evaluate(1.0)).norm() < nav_params_.stop_threshold_dist);
    const bool u_reached = has_path && (last_reference_u_ > nav_params_.stop_threshold_u);

    // 1. 组装 FSM 输入
    FsmInput fsm_input;
    fsm_input.has_path = has_path;
    fsm_input.has_new_path = has_new_path;
    fsm_input.fixed_goal_flag = input.fixed_goal;
    fsm_input.reach_goal = dist_reached || u_reached;
    fsm_input.spin_requested = input.spin_requested;
    fsm_input.spin_high_priority = input.spin_high_priority;
    fsm_input.is_hazard = compute_is_hazard(input);
    fsm_input.is_stuck = check_stuck(input);
    fsm_input.is_recovery_safe = update_recovery_safe_flag(input);
    // STOPPING 退出判定按“控制指令是否收敛到零”进行，而不是底盘实速
    fsm_input.velocity = last_cmd_.x();
    fsm_input.omega = last_cmd_.y();
    fsm_input.stamp = input.stamp;

    // 2. FSM 状态决策
    const FsmOutput fsm_output = control_fsm_->update(fsm_input);
    const FsmState state = fsm_output.state;
    on_state_transition(prev_state, state);
    last_fsm_state_ = state;

    // 3. 根据 FSM 状态，执行对应的控制逻辑
    ControlOutput output;
    switch (state) {
        case FsmState::IDLE: output = execute_idle(input); break;
        case FsmState::FOLLOW: output = execute_follow(input); break;
        case FsmState::SPIN: output = execute_spin(input); break;
        case FsmState::STOPPING: output = execute_stop(input); break;
        case FsmState::HAZARD_RECOVERY: output = execute_recovery(input); break;
        case FsmState::STUCK_REVERSE: output = execute_stuck_reverse(input); break;
        case FsmState::FIXED: output = execute_fixed(input); break;
        case FsmState::DEAD: output = execute_idle(input); break;
    }

    output.fsm_state = state;
    output.consume_global_path |= fsm_output.consume_global_path;
    if (output.consume_global_path) {
        last_reference_u_ = 0.0;
    }

    // 4. 同步已发布指令到 FSM / MPC，并在非 MPC 状态时重置 MPC 的 warm start
    if (output.valid) {
        last_cmd_ = Eigen::Vector2d(output.velocity, output.omega);
        mpc_controller_->set_last_cmd(last_cmd_);
        const bool non_mpc_state = (state == FsmState::IDLE) || (state == FsmState::SPIN) || (state == FsmState::STUCK_REVERSE);
        if (non_mpc_state) {
            mpc_controller_->reset_warm_start();
        }
    }

    return output;
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
    if (!input.global_path || !input.final_cost_map || !input.masked_direction_map) return out;

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
        *input.final_cost_map, *input.masked_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Follow) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Follow) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCSolver(Follow) solve time: %.2f ms", solve_ms);
    }

    // 台阶检测（基于 MPC 预测轨迹）
    const auto& [cmd, prediction] = *result;
    const auto [step_up, step_down] = detect_steps_on_prediction(prediction, *input.masked_direction_map);

    out.velocity = cmd.x();
    out.omega = cmd.y();
    out.step_up_ahead = step_up;
    out.step_down_ahead = step_down;
    out.predicted_path_map = prediction.path_map;
    out.predicted_v = prediction.v_pred;
    out.predicted_w = prediction.w_pred;
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
    if (!input.final_cost_map || !input.masked_direction_map) return out;

    auto start_time = std::chrono::high_resolution_clock::now();
    const auto result = mpc_controller_->stop(
        input.chassis_pose_map, input.chassis_status,
        *input.final_cost_map, *input.masked_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Stop) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Stop) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCSolver(Stop) solve time: %.2f ms", solve_ms);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.predicted_path_map = std::get<1>(*result).path_map;
    out.predicted_v = std::get<1>(*result).v_pred;
    out.predicted_w = std::get<1>(*result).w_pred;
    out.valid = true;
    return out;
}

void MainController::on_state_transition(const FsmState prev, const FsmState next) {
    if (prev == next) return;

    // 离开 STUCK_REVERSE 或 HAZARD_RECOVERY 时，必须重置卡死检测
    if (prev == FsmState::STUCK_REVERSE || prev == FsmState::HAZARD_RECOVERY) {
        stuck_active_ = false;
    }

    // 离开 FOLLOW 时重置台阶检测防抖状态
    if (prev == FsmState::FOLLOW && next != FsmState::FOLLOW) {
        step_up_on_count_ = step_up_off_count_ = 0;
        step_down_on_count_ = step_down_off_count_ = 0;
        step_up_flag_ = step_down_flag_ = false;
        last_reference_u_ = 0.0;
        mpc_controller_->reset_warm_start();
    }

    if (next == FsmState::FOLLOW && prev != FsmState::FOLLOW) {
        last_reference_u_ = 0.0;
    }

    if (next == FsmState::HAZARD_RECOVERY) {
        recovery_goal_map_ = std::nullopt;
        recovery_goal_set_time_ = rclcpp::Time{0, 0, RCL_ROS_TIME};
        recovery_safe_since_ = std::nullopt;
    }
    if (prev == FsmState::HAZARD_RECOVERY && next != FsmState::HAZARD_RECOVERY) {
        recovery_goal_map_ = std::nullopt;
        recovery_goal_set_time_ = rclcpp::Time{0, 0, RCL_ROS_TIME};
        recovery_safe_since_ = std::nullopt;
    }

    // 进入 FIXED 时重置 warm start（从其他 MPC 模式切换）
    if (next == FsmState::FIXED && prev != FsmState::FIXED) {
        mpc_controller_->reset_warm_start();
    }
}

void MainController::update_recovery_goal_if_needed(const ControlInput& input) {
    const auto& p = fsm_params_.recovery;
    if (!p.enable) {
        recovery_goal_map_ = std::nullopt;
        return;
    }

    if (!input.final_cost_map || !input.masked_direction_map) return;

    const bool need_new = (!recovery_goal_map_) || (recovery_goal_set_time_.nanoseconds() == 0) || ((input.stamp - recovery_goal_set_time_).seconds() >= p.goal_timeout);

    if (!need_new) return;

    recovery_goal_map_ = RecoveryGoalPlanner::find_goal(
        p, *input.final_cost_map, *input.masked_direction_map, input.chassis_pose_map
    );
    recovery_goal_set_time_ = input.stamp;

    if (!recovery_goal_map_) {
        RCLCPP_ERROR(logger_, "HAZARD_RECOVERY failed to find a recovery goal");
        return;
    }

    const auto s = RecoveryGoalPlanner::sample_fields(*input.final_cost_map, *input.masked_direction_map, *recovery_goal_map_);
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

bool MainController::check_stuck(const ControlInput& input) {
    const auto& p = fsm_params_.stuck;
    if (!p.enable) return false;

    if (std::abs(last_cmd_.x()) < p.cmd_vel_threshold) {
        stuck_active_ = false;
        return false;
    }

    const Eigen::Vector2d pos = input.chassis_pose_map.head<2>();
    if (!stuck_active_) {
        stuck_active_ = true;
        stuck_start_time_ = input.stamp;
        stuck_start_pos_ = pos;
        return false;
    }

    const double dt = (input.stamp - stuck_start_time_).seconds();
    const double disp = (pos - stuck_start_pos_).norm();
    if (disp > p.max_displacement) {
        stuck_start_time_ = input.stamp;
        stuck_start_pos_ = pos;
        return false;
    }

    return dt >= p.timeout;
}

bool MainController::compute_is_hazard(const ControlInput& input) const {
    const auto& p = fsm_params_.recovery;
    if (!p.enable || !input.masked_global_cost_map || !input.masked_direction_map) return false;

    // 注意：危险判断的 cost_map 使用的是 masked_global 而非 final，避免动态障碍物导致车进入危险恢复模式
    const auto sample = RecoveryGoalPlanner::sample_fields(
        *input.masked_global_cost_map,
        *input.masked_direction_map,
        input.chassis_pose_map.head<2>()
    );
    if (!sample) return false;

    return (sample->cost >= p.hazard_cost_threshold) ||
        (sample->step_norm >= p.hazard_step_norm_threshold);
}

bool MainController::update_recovery_safe_flag(const ControlInput& input) {
    const auto& p = fsm_params_.recovery;
    if (!p.enable || !input.final_cost_map || !input.masked_direction_map || !recovery_goal_map_) {
        recovery_safe_since_ = std::nullopt;
        return false;
    }

    const auto sample = RecoveryGoalPlanner::sample_fields(
        *input.final_cost_map,
        *input.masked_direction_map,
        input.chassis_pose_map.head<2>()
    );
    if (!sample) {
        recovery_safe_since_ = std::nullopt;
        return false;
    }

    const bool safe_now = RecoveryGoalPlanner::is_safe_goal(p, *sample);
    if (!safe_now) {
        recovery_safe_since_ = std::nullopt;
        return false;
    }

    if (!recovery_safe_since_) {
        recovery_safe_since_ = input.stamp;
    }
    return (input.stamp - *recovery_safe_since_).seconds() >= p.safe_hold_time;
}

// ═══════════════════ HAZARD_RECOVERY: MPC 驱动恢复 ═══════════

ControlOutput MainController::execute_recovery(const ControlInput& input) {
    ControlOutput out;
    if (!input.final_cost_map || !input.masked_direction_map) return out;

    update_recovery_goal_if_needed(input);
    if (!recovery_goal_map_) return out;

    auto start_time = std::chrono::high_resolution_clock::now();
    const auto result = mpc_controller_->recover_to_point(
        *recovery_goal_map_,
        input.chassis_pose_map, input.chassis_status,
        *input.final_cost_map, *input.masked_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Recovery) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Recovery) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCSolver(Recovery) solve time: %.2f ms", solve_ms);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.predicted_path_map = std::get<1>(*result).path_map;
    out.predicted_v = std::get<1>(*result).v_pred;
    out.predicted_w = std::get<1>(*result).w_pred;
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

// ═══════════════════ FIXED: 固定在目标点 ═════════════════════

ControlOutput MainController::execute_fixed(const ControlInput& input) {
    ControlOutput out;
    if (!input.final_cost_map || !input.masked_direction_map) return out;

    auto start_time = std::chrono::high_resolution_clock::now();
    const auto result = mpc_controller_->hold_at_point(
        input.fixed_goal_pos,
        input.chassis_pose_map, input.chassis_status,
        *input.final_cost_map, *input.masked_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Fixed) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Fixed) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCSolver(Fixed) solve time: %.2f ms", solve_ms);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.predicted_path_map = std::get<1>(*result).path_map;
    out.predicted_v = std::get<1>(*result).v_pred;
    out.predicted_w = std::get<1>(*result).w_pred;
    out.valid = true;
    return out;
}

// ═══════════════════ 台阶检测（基于 MPC 预测轨迹 + 防抖） ════════════════════

std::tuple<bool, bool> MainController::detect_steps_on_prediction(
    const MPCPrediction& prediction,
    const DirectionMap& direction_map
) {
    bool raw_step_up = false;
    bool raw_step_down = false;

    for (size_t i = 0; i < prediction.path_map.size(); ++i) {
        const Eigen::Vector2d& pos = prediction.path_map[i];
        const double theta = prediction.headings[i];

        const Eigen::Vector2d g = direction_map.map_coord_to_grid(pos);
        if (!direction_map.is_valid_coord(Eigen::Vector2i(
            static_cast<int>(std::floor(g.x())),
            static_cast<int>(std::floor(g.y())))
        )) continue;

        const Eigen::Vector2d dir = direction_map.interpolate(g);
        const double norm = dir.norm();
        if (norm < nav_params_.step_detect_norm_threshold) continue;

        const Eigen::Vector2d heading(std::cos(theta), std::sin(theta));
        const double dot = dir.normalized().dot(heading);
        if (dot > nav_params_.step_detect_dot_threshold) raw_step_up = true;
        if (dot < -nav_params_.step_detect_dot_threshold) raw_step_down = true;
    }

    // 防抖：连续多次检测到才设置标志位，连续多次未检测到才取消
    if (raw_step_up) {
        step_up_on_count_++;
        step_up_off_count_ = 0;
        if (step_up_on_count_ >= nav_params_.step_on_count_threshold) step_up_flag_ = true;
    } else {
        step_up_off_count_++;
        step_up_on_count_ = 0;
        if (step_up_off_count_ >= nav_params_.step_off_count_threshold) step_up_flag_ = false;
    }

    if (raw_step_down) {
        step_down_on_count_++;
        step_down_off_count_ = 0;
        if (step_down_on_count_ >= nav_params_.step_on_count_threshold) step_down_flag_ = true;
    } else {
        step_down_off_count_++;
        step_down_on_count_ = 0;
        if (step_down_off_count_ >= nav_params_.step_off_count_threshold) step_down_flag_ = false;
    }

    return {step_up_flag_, step_down_flag_};
}

}