#include <path_follower/main_controller.hpp>

#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numbers>
#include <rclcpp/logging.hpp>

namespace path_follower {

namespace {

constexpr double ANGLE_EPSILON = 1e-6;

inline bool is_chassis_dead(const uint8_t leg_mode, const uint8_t comp_status) {
    // leg_mode: 2:Flight, 4:Mature
    // comp_status: 4:比赛中
    return (leg_mode != 2u && leg_mode != 4u) || comp_status != 4u;
}

}

namespace {

std::optional<double> max_cost_along_segment(
    const CostMap& cost_map,
    const Eigen::Vector2d& a_map,
    const Eigen::Vector2d& b_map,
    const int samples
) {
    const int n = std::max(1, samples);
    double max_cost = 0.0;
    for (int i = 0; i <= n; i++) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        const Eigen::Vector2d pos = a_map + (b_map - a_map) * t;
        const Eigen::Vector2d g = cost_map.map_coord_to_grid(pos);
        if (!cost_map.is_valid_coord(g)) return std::nullopt;
        max_cost = std::max(max_cost, cost_map.interpolate(g));
    }
    return max_cost;
}

std::optional<double> path_integral_cost01(
    const CostMap& cost_map,
    const Eigen::Vector2d& from_map,
    const Eigen::Vector2d& to_map,
    const double resolution
) {
    const double dist = (to_map - from_map).norm();
    const double res = std::max(1e-3, resolution);
    const int n = std::max(1, static_cast<int>(std::ceil(dist / res)));

    double acc = 0.0;
    for (int i = 0; i <= n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        const Eigen::Vector2d pos = from_map + (to_map - from_map) * t;
        const Eigen::Vector2d g = cost_map.map_coord_to_grid(pos);
        if (!cost_map.is_valid_coord(g)) {
            return std::nullopt;
        }
        acc += std::clamp(cost_map.interpolate(g) / 255.0, 0.0, 1.0);
    }
    return acc;
}

Eigen::Vector2d rotate_vec(const Eigen::Vector2d& v, const double angle) {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return Eigen::Vector2d(c * v.x() - s * v.y(), s * v.x() + c * v.y());
}

double angle_between(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return std::atan2(a.x() * b.y() - a.y() * b.x(), a.dot(b));
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
        const Eigen::Vector2d& goal,
        const double radius
    ) {
        double acc = 0.0;
        std::optional<FieldSample> end_s;
        const int n = std::max(1, static_cast<int>(radius / p.path_integral_resolution));
        for (int i = 0; i <= n; i++) {
            const double t = static_cast<double>(i) / static_cast<double>(n);
            const Eigen::Vector2d pos = origin + (goal - origin) * t;
            const auto s = sample_fields(cost_map, dir_map, pos);
            if (!s) return std::nullopt;
            acc += potential_cost(*s);
            if (i == n) end_s = s;
        }

        if (!end_s || !is_safe_goal(p, *end_s)) {
            return std::nullopt;
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

        const double r_min = std::min(p.radius_min, p.radius_max);
        const double r_max = std::max(p.radius_min, p.radius_max);
        const int r_n = std::max(1, p.radius_samples);
        const int a_n = std::max(1, p.angle_samples);

        std::optional<Eigen::Vector2d> best_pt;
        std::optional<PathScore> best_sc;

        for (int ri = 0; ri < r_n; ri++) {
            const double rt = (r_n == 1) ? 0.0 : (static_cast<double>(ri) / static_cast<double>(r_n - 1));
            const double r = r_min + (r_max - r_min) * rt;

            for (int ai = 0; ai < a_n; ai++) {
                const double a = 2.0 * std::numbers::pi * static_cast<double>(ai) / static_cast<double>(a_n);
                const Eigen::Vector2d pt = origin + Eigen::Vector2d(std::cos(a), std::sin(a)) * r;
                const auto field = sample_fields(cost_map, dir_map, pt);
                if (!field) continue;
                if (field->cost >= p.recovery_cost_threshold) continue;
                const auto sc = score_candidate_by_path_integral(p, cost_map, dir_map, origin, pt, r);
                if (!sc) continue;
                if (!best_sc || sc->score < best_sc->score) {
                    best_sc = *sc;
                    best_pt = pt;
                }
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
        last_cycle_chassis_dead_ = true;
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

    const bool just_revived = last_cycle_chassis_dead_;
    last_cycle_chassis_dead_ = false;

    const FsmState prev_state = last_fsm_state_;
    mpc_controller_->set_last_cmd(last_cmd_);

    // 0. 更新隐藏状态观测器（在 MPC 求解之前）
    mpc_controller_->update_observer(input.chassis_state);

    // 0.5 更新能量状态
    mpc_controller_->set_energy_state(input.remaining_energy, input.rfr_pwr_limit);

    if (prev_state == FsmState::HAZARD_RECOVERY) {
        update_recovery_goal_if_needed(input);
    }

    const bool has_path = input.global_path.has_value();
    const bool has_new_path = has_path && input.path_updated;

    clear_step_up_attempt_history_if_needed(has_path, has_new_path);

    // 路标点重算：新路径到达时重新分割；路径消失时清空
    if (has_new_path) {
        recompute_follow_landmarks(*input.global_path);
        follow_max_landmark_idx_ = -1;
    } else if (!has_path) {
        follow_landmarks_u_.clear();
        follow_max_landmark_idx_ = -1;
    }

    bool cancel_follow_task_now = false;
    if (pending_cancel_follow_task_) {
        cancel_follow_task_now = true;
        pending_cancel_follow_task_ = false;
    }
    // 路标点无进度检测：在 FOLLOW 模式下，若超时未推进最高路标点则取消任务
    if (!cancel_follow_task_now && check_no_progress(input)) {
        cancel_follow_task_now = true;
    }

    // Dead->Mature 常发生在上台阶失败后的恢复循环中。
    // 若复活时仍处于“上台阶标志位已拉高”状态，则按一次“拉高事件”计入兜底。
    if (just_revived && !has_new_path && last_fsm_state_ == FsmState::FOLLOW && has_path && step_up_flag_) {
        if (register_step_up_attempt_and_should_cancel(input.chassis_pose_map.head<2>())) {
            cancel_follow_task_now = true;
        }
    }
    const bool dist_reached = has_path && ((input.chassis_pose_map.head<2>() - input.global_path->evaluate(1.0)).norm() < nav_params_.stop_threshold_dist);
    const bool u_reached = has_path && (last_reference_u_ > nav_params_.stop_threshold_u);

    // 1. 组装 FSM 输入
    FsmInput fsm_input;
    fsm_input.has_path = cancel_follow_task_now ? false : has_path;
    fsm_input.has_new_path = cancel_follow_task_now ? false : has_new_path;
    fsm_input.fixed_goal_flag = cancel_follow_task_now ? false : input.fixed_goal;
    fsm_input.reach_goal = dist_reached || u_reached;
    fsm_input.step_runup_requested = pending_step_runup_;
    fsm_input.step_runup_done = step_runup_done_;
    fsm_input.spin_requested = input.spin_requested;
    fsm_input.spin_high_priority = input.spin_high_priority;
    const bool hazard_allowed = (prev_state == FsmState::IDLE) || (prev_state == FsmState::SPIN) || (prev_state == FsmState::HAZARD_RECOVERY);
    fsm_input.is_hazard = hazard_allowed && compute_is_hazard(input);
    fsm_input.is_stuck = check_stuck(input);
    fsm_input.is_recovery_safe = update_recovery_safe_flag(input);
    // STOPPING 退出判定按“控制指令是否收敛到零”进行，而不是底盘实速
    fsm_input.velocity = last_cmd_.x();
    fsm_input.omega = last_cmd_.y();
    fsm_input.stamp = input.stamp;

    // 2. FSM 状态决策
    const FsmOutput fsm_output = control_fsm_->update(fsm_input);
    const FsmState state = fsm_output.state;
    pending_step_runup_ = false;
    step_runup_done_ = false;
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
        case FsmState::STEP_RUNUP: output = execute_step_runup(input); break;
        case FsmState::DEAD: output = execute_idle(input); break;
    }

    output.fsm_state = state;
    output.consume_global_path |= fsm_output.consume_global_path;
    if (cancel_follow_task_now) {
        output.consume_global_path = true;
    }
    if (output.consume_global_path) {
        last_reference_u_ = 0.0;
        clear_step_up_attempt_history();
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
    out.mode = ChassisMode::NORMAL;
    out.valid = true;
    return out;
}

// ═══════════════════ FOLLOW: 跟随路径 ════════════════════════

ControlOutput MainController::execute_follow(const ControlInput& input) {
    ControlOutput out;
    if (!input.global_path || !input.final_cost_map || !input.masked_direction_map) return out;

    // 投影当前位置到样条
    const double u0 = project_to_spline_u_extrapolated(
        *input.global_path, input.chassis_pose_map.head<2>(), last_reference_u_,
        mpc_controller_->params().follow.projection.proj_num_samples,
        mpc_controller_->params().follow.projection.proj_search_window,
        mpc_controller_->params().follow.projection.local_search_lazy_distance
    );
    last_reference_u_ = u0;

    // 路径跟随任务取消条件（投影守卫）
    // 1) 投影距离过大：说明路径与当前位置严重不一致
    // 2) 当前位置到投影点的连线最大障碍物代价过高：说明“接回路径”的直线路径可能不可通行
    const Eigen::Vector2d pos_map = input.chassis_pose_map.head<2>();
    const Eigen::Vector2d proj_map = input.global_path->evaluate(u0);
    const double proj_dist = (proj_map - pos_map).norm();
    if (nav_params_.follow_proj_dist_max > 0.0 && proj_dist > nav_params_.follow_proj_dist_max) {
        RCLCPP_WARN(
            logger_,
            "Follow cancel: projection too far (dist=%.2f m > %.2f m)",
            proj_dist, nav_params_.follow_proj_dist_max
        );
        out.velocity = 0.0;
        out.omega = 0.0;
        out.consume_global_path = true;
        out.valid = true;
        mpc_controller_->reset_warm_start();
        return out;
    }

    if (input.masked_global_cost_map && nav_params_.follow_proj_cost_max >= 0.0 && nav_params_.follow_proj_cost_max < 255.0) {
        const auto max_cost = max_cost_along_segment(
            *input.masked_global_cost_map,
            pos_map,
            proj_map,
            nav_params_.follow_proj_cost_samples
        );

        const double c = max_cost.value_or(255.0);
        if (!max_cost) {
            RCLCPP_ERROR(logger_, "Follow cancel: projection segment out of cost map bounds");
            out.velocity = 0.0;
            out.omega = 0.0;
            out.consume_global_path = true;
            out.valid = true;
            mpc_controller_->reset_warm_start();
            return out;
        }

        if (c > nav_params_.follow_proj_cost_max) {
            RCLCPP_WARN(
                logger_,
                "Follow cancel: projection segment cost too high (max_cost=%.1f > %.1f) using masked_global_cost_map",
                c, nav_params_.follow_proj_cost_max
            );
            out.velocity = 0.0;
            out.omega = 0.0;
            out.consume_global_path = true;
            out.valid = true;
            mpc_controller_->reset_warm_start();
            return out;
        }
    }

    // 调用 MPC
    auto start_time = std::chrono::steady_clock::now();
    const auto result = mpc_controller_->solve_follow(
        *input.global_path, input.chassis_pose_map, input.chassis_state,
        *input.final_cost_map, input.per_step_cost_maps, input.prediction_dt,
        *input.masked_direction_map,
        latched_step_up_mode_
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Follow) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Follow) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCSolver(Follow) solve time: %.2f ms", solve_ms);
    }

    // 台阶检测（基于 MPC 预测轨迹）
    const auto& [cmd, prediction] = *result;
    const auto step = detect_steps_on_prediction_with_edges(prediction, *input.masked_direction_map);
    const auto step_edge = find_first_step_edge_on_prediction(prediction, *input.masked_direction_map);

    if (step.step_up_rising) {
        const Eigen::Vector2d pos = input.chassis_pose_map.head<2>();
        if (register_step_up_attempt_and_should_cancel(pos)) {
            RCLCPP_ERROR(logger_, "StepUp failsafe: cancel current follow task");
            pending_cancel_follow_task_ = true;
        }
    }

    out.velocity = cmd.x();
    out.omega = cmd.y();

    // Step-up mode latching: decide only once when step_up is first detected (debounced rising),
    // and keep the decision until step_up flag goes low.
    if (step.step_up_rising) {
        step_up_latch_armed_ = true;
        latched_step_up_mode_ = std::nullopt;
        latched_step_up_runup_ = false;
    }

    // If we enter FOLLOW while step_up is already active (e.g., revive), arm once as well.
    if (step.step_up && !step_up_latch_armed_ && !latched_step_up_mode_ && !latched_step_up_runup_) {
        step_up_latch_armed_ = true;
    }

    if (!step.step_up) {
        step_up_latch_armed_ = false;
        latched_step_up_mode_ = std::nullopt;
        latched_step_up_runup_ = false;
    }

    if (step.step_up) {
        if (latched_step_up_mode_) {
            out.mode = *latched_step_up_mode_;
        } else if (latched_step_up_runup_) {
            out.mode = ChassisMode::NORMAL;
        } else {
            // If rising edge happened but step edge info wasn't available that cycle,
            // latch on the first cycle that has a valid edge while step_up stays active.
            if (step_up_latch_armed_ && step_edge) {
                const double v_edge = step_edge->v_pred;
                if (v_edge >= nav_params_.step_up_vel_jump_threshold) {
                    latched_step_up_mode_ = ChassisMode::STEP_UP_JUMP;
                    out.mode = *latched_step_up_mode_;
                } else if (v_edge >= nav_params_.step_up_vel_leg_threshold) {
                    latched_step_up_mode_ = ChassisMode::STEP_UP_LEG;
                    out.mode = *latched_step_up_mode_;
                } else  {
                    latched_step_up_runup_ = true;
                    pending_step_runup_ = true;
                    pending_step_runup_edge_ = step_edge;
                    out.mode = ChassisMode::NORMAL;
                }

                // Disarm after first decision.
                step_up_latch_armed_ = false;
            } else {
                out.mode = ChassisMode::NORMAL;
            }
        }
    } else if (step.step_down) {
        out.mode = ChassisMode::STEP_DOWN;
    } else {
        out.mode = ChassisMode::NORMAL;
    }
    out.predicted_path_map = prediction.path_map;
    out.predicted_v = prediction.v_pred;
    out.predicted_w = prediction.w_pred;
    out.valid = true;
    return out;
}

// ═══════════════════ SPIN: 小陀螺 ════════════════════════════

ControlOutput MainController::execute_spin(const ControlInput& input) {
    ControlOutput out;
    out.mode = input.spin_fast ? ChassisMode::SPIN_FAST : ChassisMode::SPIN_SLOW;
    out.valid = true;
    return out;
}

// ═══════════════════ STOPPING: 平滑减速 ══════════════════════

ControlOutput MainController::execute_stop(const ControlInput& input) {
    ControlOutput out;
    if (!input.final_cost_map || !input.masked_direction_map) return out;

    auto start_time = std::chrono::steady_clock::now();
    const auto result = mpc_controller_->solve_stop(
        input.chassis_pose_map, input.chassis_state,
        *input.final_cost_map, *input.masked_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Stop) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Stop) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCSolver(Stop) solve time: %.2f ms", solve_ms);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.mode = ChassisMode::NORMAL;
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

    // 离开 FOLLOW/STEP_RUNUP 时重置台阶检测防抖状态 + 无进度检测索引
    if ((prev == FsmState::FOLLOW || prev == FsmState::STEP_RUNUP) && next != FsmState::FOLLOW && next != FsmState::STEP_RUNUP) {
        step_up_on_count_ = step_up_off_count_ = 0;
        step_down_on_count_ = step_down_off_count_ = 0;
        step_up_flag_ = step_down_flag_ = false;
        last_reference_u_ = 0.0;
        follow_max_landmark_idx_ = -1;
        mpc_controller_->reset_warm_start();
    }

    if (next == FsmState::FOLLOW && prev != FsmState::FOLLOW) {
        last_reference_u_ = 0.0;
    }

    if (next == FsmState::STEP_RUNUP && prev != FsmState::STEP_RUNUP) {
        step_runup_goal_map_ = std::nullopt;

        // STEP_RUNUP does not use step flags; reset debouncing + latching so we can re-detect after runup.
        step_up_on_count_ = step_up_off_count_ = 0;
        step_down_on_count_ = step_down_off_count_ = 0;
        step_up_flag_ = step_down_flag_ = false;
        step_up_latch_armed_ = false;
        latched_step_up_mode_ = std::nullopt;
        latched_step_up_runup_ = false;
    }
    if (prev == FsmState::STEP_RUNUP && next != FsmState::STEP_RUNUP) {
        step_runup_goal_map_ = std::nullopt;
        pending_step_runup_edge_ = std::nullopt;
    }

    if (next == FsmState::HAZARD_RECOVERY) {
        recovery_goal_map_ = std::nullopt;
        recovery_goal_set_time_ = std::nullopt;
        recovery_safe_since_ = std::nullopt;
    }
    if (prev == FsmState::HAZARD_RECOVERY && next != FsmState::HAZARD_RECOVERY) {
        recovery_goal_map_ = std::nullopt;
        recovery_goal_set_time_ = std::nullopt;
        recovery_safe_since_ = std::nullopt;
    }

    // Hold 求解器由 HAZARD_RECOVERY / FIXED 共享，二者之间切换不应互相清空 warm start。
    if (next == FsmState::FIXED && prev != FsmState::FIXED && prev != FsmState::HAZARD_RECOVERY) {
        mpc_controller_->reset_warm_start();
    }

    if (next == FsmState::FOLLOW && prev == FsmState::STEP_RUNUP) {
        mpc_controller_->reset_warm_start();
    }
}

void MainController::update_recovery_goal_if_needed(const ControlInput& input) {
    const auto& p = fsm_params_.recovery;
    if (!p.enable) {
        recovery_goal_map_ = std::nullopt;
        return;
    }

    if (!input.masked_global_cost_map || !input.masked_direction_map) return;

    const bool need_new = (!recovery_goal_map_) || (!recovery_goal_set_time_) || (std::chrono::duration<double>(input.stamp - *recovery_goal_set_time_).count() >= p.goal_timeout);

    if (!need_new) return;

    recovery_goal_map_ = RecoveryGoalPlanner::find_goal(
        p, *input.masked_global_cost_map, *input.masked_direction_map, input.chassis_pose_map
    );
    recovery_goal_set_time_ = input.stamp;

    if (!recovery_goal_map_) {
        RCLCPP_ERROR(logger_, "HAZARD_RECOVERY failed to find a recovery goal");
        return;
    }

    const auto s = RecoveryGoalPlanner::sample_fields(*input.masked_global_cost_map, *input.masked_direction_map, *recovery_goal_map_);
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

    const double dt = std::chrono::duration<double>(input.stamp - stuck_start_time_).count();
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
    return std::chrono::duration<double>(input.stamp - *recovery_safe_since_).count() >= p.safe_hold_time;
}

// ═══════════════════ HAZARD_RECOVERY: MPC 驱动恢复 ═══════════

ControlOutput MainController::execute_recovery(const ControlInput& input) {
    ControlOutput out;
    if (!input.final_cost_map || !input.masked_direction_map) return out;

    update_recovery_goal_if_needed(input);
    if (!recovery_goal_map_) {
        out.velocity = 0.0;
        out.omega = 0.0;
        out.mode = ChassisMode::NORMAL;
        out.valid = true;
        return out;
    }

    auto start_time = std::chrono::steady_clock::now();
    const auto result = mpc_controller_->solve_hold(
        *recovery_goal_map_,
        input.chassis_pose_map, input.chassis_state,
        *input.final_cost_map, *input.masked_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Recovery) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Recovery) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCSolver(Recovery) solve time: %.2f ms", solve_ms);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.mode = ChassisMode::NORMAL;
    out.predicted_path_map = std::get<1>(*result).path_map;
    out.predicted_v = std::get<1>(*result).v_pred;
    out.predicted_w = std::get<1>(*result).w_pred;
    out.valid = true;
    return out;
}

ControlOutput MainController::execute_step_runup(const ControlInput& input) {
    ControlOutput out;
    if (!input.final_cost_map || !input.masked_direction_map || !input.masked_global_cost_map) return out;

    if (!step_runup_goal_map_) {
        step_runup_goal_map_ = select_step_runup_point(input);
        if (!step_runup_goal_map_) {
            RCLCPP_ERROR(logger_, "STEP_RUNUP failed to find a feasible runup point, cancelling follow task");
            out.velocity = 0.0;
            out.omega = 0.0;
            out.mode = ChassisMode::NORMAL;
            out.consume_global_path = true;
            out.valid = true;
            return out;
        }
    }

    auto start_time = std::chrono::steady_clock::now();
    const auto result = mpc_controller_->solve_hold(
        *step_runup_goal_map_,
        input.chassis_pose_map, input.chassis_state,
        *input.final_cost_map, *input.masked_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(StepRunup) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(StepRunup) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCSolver(StepRunup) solve time: %.2f ms", solve_ms);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.mode = ChassisMode::NORMAL;
    out.predicted_path_map = std::get<1>(*result).path_map;
    out.predicted_v = std::get<1>(*result).v_pred;
    out.predicted_w = std::get<1>(*result).w_pred;
    out.valid = true;

    const double dist_to_goal = (input.chassis_pose_map.head<2>() - *step_runup_goal_map_).norm();
    if (dist_to_goal <= mpc_controller_->params().hold.goal_weights.goal_deadzone) {
        step_runup_done_ = true;
    }
    return out;
}

// ═══════════════════ STUCK_REVERSE: 倒车脱困 ═════════════════

ControlOutput MainController::execute_stuck_reverse(const ControlInput& input) {
    (void)input;
    ControlOutput out;
    out.velocity = -fsm_params_.stuck.reverse_speed;
    out.omega = 0.0;
    out.mode = ChassisMode::NORMAL;
    out.valid = true;
    return out;
}

// ═══════════════════ FIXED: 固定在目标点 ═════════════════════

ControlOutput MainController::execute_fixed(const ControlInput& input) {
    ControlOutput out;
    if (!input.final_cost_map || !input.masked_direction_map) return out;

    auto start_time = std::chrono::steady_clock::now();
    const auto result = mpc_controller_->solve_hold(
        input.fixed_goal_pos,
        input.chassis_pose_map, input.chassis_state,
        *input.final_cost_map, *input.masked_direction_map
    );
    if (!result) {
        RCLCPP_ERROR(logger_, "MPCSolver(Fixed) solve failed: %s", result.error().c_str());
        return out;
    }

    const double solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    if (solve_ms > MPC_DT * 500.0) {
        RCLCPP_WARN(logger_, "MPCSolver(Fixed) solve time %.2f ms > %.2f ms", solve_ms, MPC_DT * 500.0);
    } else {
        RCLCPP_DEBUG(logger_, "MPCSolver(Fixed) solve time: %.2f ms", solve_ms);
    }

    out.velocity = std::get<0>(*result).x();
    out.omega = std::get<0>(*result).y();
    out.mode = ChassisMode::NORMAL;
    out.predicted_path_map = std::get<1>(*result).path_map;
    out.predicted_v = std::get<1>(*result).v_pred;
    out.predicted_w = std::get<1>(*result).w_pred;
    out.valid = true;
    return out;
}

// ═══════════════════ 台阶检测（基于 MPC 预测轨迹 + 防抖） ════════════════════

MainController::StepDetectResult MainController::detect_steps_on_prediction_with_edges(
    const MPCPrediction& prediction,
    const DirectionMap& direction_map
) {
    std::optional<size_t> nearest_step_up_idx;
    std::optional<size_t> nearest_step_down_idx;

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
        if (dot > nav_params_.step_detect_dot_threshold && !nearest_step_up_idx) {
            nearest_step_up_idx = i;
        }
        if (dot < -nav_params_.step_detect_dot_threshold && !nearest_step_down_idx) {
            nearest_step_down_idx = i;
        }
    }

    bool raw_step_up = false;
    bool raw_step_down = false;
    if (nearest_step_up_idx && nearest_step_down_idx) {
        if (*nearest_step_up_idx <= *nearest_step_down_idx) {
            raw_step_up = true;
        } else {
            raw_step_down = true;
        }
    } else {
        raw_step_up = nearest_step_up_idx.has_value();
        raw_step_down = nearest_step_down_idx.has_value();
    }

    const bool prev_up = step_up_flag_;
    const bool prev_down = step_down_flag_;

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

    StepDetectResult out;
    out.step_up = step_up_flag_;
    out.step_down = step_down_flag_;
    out.step_up_rising = (!prev_up) && step_up_flag_;
    out.step_down_rising = (!prev_down) && step_down_flag_;
    return out;
}

std::optional<MainController::StepEdgeInfo> MainController::find_first_step_edge_on_prediction(
    const MPCPrediction& prediction,
    const DirectionMap& direction_map
) const {
    const size_t n = std::min(prediction.path_map.size(), prediction.v_pred.size());
    for (size_t i = 0; i < n; ++i) {
        const Eigen::Vector2d g = direction_map.map_coord_to_grid(prediction.path_map[i]);
        if (!direction_map.is_valid_coord(Eigen::Vector2i(
            static_cast<int>(std::floor(g.x())),
            static_cast<int>(std::floor(g.y()))))
        ) continue;

        const Eigen::Vector2d dir = direction_map.interpolate(g);
        if (dir.norm() < nav_params_.step_edge_norm_threshold) continue;

        StepEdgeInfo edge;
        edge.pos_map = prediction.path_map[i];
        edge.dir_map = dir;
        edge.v_pred = prediction.v_pred[i];
        return edge;
    }
    return std::nullopt;
}

void MainController::clear_step_up_attempt_history() {
    step_up_attempt_positions_.clear();
}

void MainController::clear_step_up_attempt_history_if_needed(const bool has_path, const bool has_new_path) {
    if (!nav_params_.step_up_failsafe_enable) {
        step_up_attempt_positions_.clear();
        pending_cancel_follow_task_ = false;
        return;
    }

    if (!has_path || has_new_path) {
        step_up_attempt_positions_.clear();
        pending_cancel_follow_task_ = false;
    }
}

bool MainController::register_step_up_attempt_and_should_cancel(const Eigen::Vector2d& pos_map) {
    if (!nav_params_.step_up_failsafe_enable) return false;

    const size_t n = static_cast<size_t>(nav_params_.step_up_failsafe_similar_attempts);
    step_up_attempt_positions_.push_back(pos_map);
    while (step_up_attempt_positions_.size() > n) {
        step_up_attempt_positions_.erase(step_up_attempt_positions_.begin());
    }

    if (step_up_attempt_positions_.size() < n) {
        RCLCPP_INFO(logger_, "StepUp attempt recorded (%zu/%zu) at (%.2f, %.2f)", step_up_attempt_positions_.size(), n, pos_map.x(), pos_map.y());
        return false;
    }

    const Eigen::Vector2d& ref = step_up_attempt_positions_.front();
    double max_dist = 0.0;
    for (size_t i = 1; i < step_up_attempt_positions_.size(); ++i) {
        max_dist = std::max(max_dist, (step_up_attempt_positions_[i] - ref).norm());
    }

    const bool similar = max_dist <= nav_params_.step_up_failsafe_similar_dist;
    if (similar) {
        RCLCPP_WARN(
            logger_,
            "StepUp failsafe triggered: %zu attempts within %.2f m (ref=(%.2f, %.2f), max_dist=%.2f)",
            step_up_attempt_positions_.size(),
            nav_params_.step_up_failsafe_similar_dist,
            ref.x(), ref.y(),
            max_dist
        );
    }
    return similar;
}

bool MainController::is_step_runup_segment_feasible(
    const CostMap& cost_map,
    const Eigen::Vector2d& from_map,
    const Eigen::Vector2d& to_map
) const {
    const int n = std::max(1, nav_params_.step_runup_line_check_samples);
    for (int i = 0; i <= n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        const Eigen::Vector2d pos = from_map + (to_map - from_map) * t;
        const Eigen::Vector2d g = cost_map.map_coord_to_grid(pos);
        if (!cost_map.is_valid_coord(g)) return false;
        if (cost_map.interpolate(g) >= nav_params_.step_runup_cost_threshold) return false;
    }
    return true;
}

std::optional<Eigen::Vector2d> MainController::select_step_runup_point(const ControlInput& input) const {
    if (!pending_step_runup_edge_ || !input.masked_global_cost_map) {
        return std::nullopt;
    }

    const Eigen::Vector2d step_pos = pending_step_runup_edge_->pos_map;
    const Eigen::Vector2d step_dir = pending_step_runup_edge_->dir_map;
    if (step_dir.norm() < ANGLE_EPSILON) return std::nullopt;

    const Eigen::Vector2d robot_pos = input.chassis_pose_map.head<2>();
    Eigen::Vector2d approach_dir = step_dir.normalized();
    const Eigen::Vector2d to_robot = robot_pos - step_pos;
    if (approach_dir.dot(to_robot) <= 0.0) {
        approach_dir = -approach_dir;
    }

    std::optional<Eigen::Vector2d> best_point;
    double best_score = std::numeric_limits<double>::infinity();

    const double r_min = std::min(nav_params_.step_runup_radius_min, nav_params_.step_runup_radius_max);
    const double r_max = std::max(nav_params_.step_runup_radius_min, nav_params_.step_runup_radius_max);
    const int radius_samples = std::max(1, nav_params_.step_runup_radius_samples);
    const int angle_samples = std::max(1, nav_params_.step_runup_angle_samples);

    for (int ri = 0; ri < radius_samples; ++ri) {
        const double rt = (radius_samples == 1) ? 0.0 : static_cast<double>(ri) / static_cast<double>(radius_samples - 1);
        const double radius = r_min + (r_max - r_min) * rt;

        for (int ai = 0; ai < angle_samples; ++ai) {
            const double at = (angle_samples == 1) ? 0.0 : static_cast<double>(ai) / static_cast<double>(angle_samples - 1);
            const double angle = -nav_params_.step_runup_angle_half_range + 2.0 * nav_params_.step_runup_angle_half_range * at;
            const Eigen::Vector2d ray = rotate_vec(approach_dir, angle);
            const Eigen::Vector2d candidate = step_pos + ray * radius;

            const Eigen::Vector2d g = input.masked_global_cost_map->map_coord_to_grid(candidate);
            if (!input.masked_global_cost_map->is_valid_coord(g)) continue;

            const double cost = input.masked_global_cost_map->interpolate(g);
            if (cost >= nav_params_.step_runup_cost_threshold) continue;
            if (!is_step_runup_segment_feasible(*input.masked_global_cost_map, candidate, step_pos)) continue;

            const auto robot_path_cost = path_integral_cost01(
                *input.masked_global_cost_map,
                robot_pos,
                candidate,
                nav_params_.step_runup_path_integral_resolution
            );
            if (!robot_path_cost) continue;

            const double angle_error = std::abs(angle_between(approach_dir, ray));
            const double score =
                nav_params_.step_runup_cost_weight * (cost / 255.0) +
                nav_params_.step_runup_step_dist_weight * (1.0 / std::max(radius, 0.1)) +
                nav_params_.step_runup_angle_weight * angle_error +
                nav_params_.step_runup_robot_dist_weight * (candidate - robot_pos).norm() +
                nav_params_.step_runup_robot_path_cost_weight * (*robot_path_cost);

            if (score < best_score) {
                best_score = score;
                best_point = candidate;
            }
        }
    }

    return best_point;
}

// ═══════════════════ Follow 路标点无进度检测 ══════════════════

void MainController::recompute_follow_landmarks(const SplineD& path) {
    follow_landmarks_u_.clear();
    if (!nav_params_.no_progress_enable) return;

    const int N = 500;
    const double spacing = std::max(0.1, nav_params_.no_progress_landmark_spacing);
    double arc = 0.0;
    double next_threshold = 0.0;
    Eigen::Vector2d prev = path.evaluate(0.0);

    for (int i = 0; i <= N; i++) {
        const double u = static_cast<double>(i) / static_cast<double>(N);
        const Eigen::Vector2d cur = path.evaluate(u);
        if (i > 0) arc += (cur - prev).norm();
        prev = cur;
        if (arc >= next_threshold) {
            follow_landmarks_u_.push_back(u);
            next_threshold += spacing;
        }
    }

    if (follow_landmarks_u_.empty() || follow_landmarks_u_.back() < 1.0) {
        follow_landmarks_u_.push_back(1.0);
    }
}

bool MainController::check_no_progress(const ControlInput& input) {
    if (!nav_params_.no_progress_enable) return false;
    if (last_fsm_state_ != FsmState::FOLLOW) return false;
    if (follow_landmarks_u_.empty()) return false;

    if (input.global_path) {
        last_reference_u_ = project_to_spline_u_extrapolated(
            *input.global_path,
            input.chassis_pose_map.head<2>(),
            last_reference_u_,
            mpc_controller_->params().follow.projection.proj_num_samples,
            mpc_controller_->params().follow.projection.proj_search_window,
            mpc_controller_->params().follow.projection.local_search_lazy_distance
        );
    }

    const int n = static_cast<int>(follow_landmarks_u_.size());

    // 寻找当前 u 能覆盖的最高路标点索引
    int new_max = follow_max_landmark_idx_;
    for (int i = std::max(0, new_max + 1); i < n; i++) {
        if (last_reference_u_ >= follow_landmarks_u_[static_cast<size_t>(i)]) {
            new_max = i;
        } else {
            break;
        }
    }

    // 路标点前进（包括首次初始化）：重置计时器
    if (follow_max_landmark_idx_ < 0 || new_max > follow_max_landmark_idx_) {
        follow_max_landmark_idx_ = new_max;
        follow_max_landmark_time_ = input.stamp;
        return false;
    }

    const double elapsed = std::chrono::duration<double>(input.stamp - follow_max_landmark_time_).count();
    if (elapsed >= nav_params_.no_progress_timeout) {
        const double landmark_u = follow_max_landmark_idx_ >= 0 ? follow_landmarks_u_[static_cast<size_t>(follow_max_landmark_idx_)] : -1.0;
        RCLCPP_WARN(logger_,
            "Follow no-progress: stuck at landmark %d/%d (landmark_u=%.3f, progress_u=%.3f) for %.1fs, cancelling task",
            follow_max_landmark_idx_, n - 1, landmark_u, last_reference_u_, elapsed);
        return true;
    }
    return false;
}

}
