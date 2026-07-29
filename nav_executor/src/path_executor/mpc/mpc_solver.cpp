#include <nav_executor/path_executor/mpc/mpc_solver.hpp>
#include <nav_executor/path_executor/mpc/mpc_utils.hpp>
#include <nav_executor/path_executor/mpc/lpv_model.hpp>

namespace nav_executor {

// ── Solver 辅助模板 ──

namespace {

CapabilityProfile tightened_nominal_capability(
    const CapabilityProfile& capability,
    const MPCFollowAncillaryFeedbackParams& feedback
) {
    CapabilityProfile nominal = capability;
    nominal.command_envelope.velocity.min += feedback.velocity_command_margin;
    nominal.command_envelope.velocity.max -= feedback.velocity_command_margin;
    nominal.command_dynamics.velocity_rate_max -= feedback.velocity_command_rate_margin;
    return nominal;
}

struct LongitudinalNominalInitialState {
    StateVec state;
    bool reanchored;
};

LongitudinalNominalInitialState longitudinal_nominal_initial_state(
    const StateVec& measured,
    const std::optional<StateVec>& predicted_nominal,
    const MPCFollowAncillaryFeedbackParams& feedback
) {
    if (!predicted_nominal || !predicted_nominal->allFinite()) {
        return {.state = measured, .reanchored = true};
    }
    if (std::abs(measured(ix::V) - (*predicted_nominal)(ix::V))
        > feedback.velocity_error_reanchor_threshold) {
        return {.state = measured, .reanchored = true};
    }

    StateVec nominal = measured;
    for (const int index : {ix::XH, ix::V, ix::V_CMD, ix::V_CMD_RATE}) {
        nominal(index) = (*predicted_nominal)(index);
    }
    return {.state = nominal, .reanchored = false};
}

struct AncillaryControlResult {
    double command_rate;
    bool command_tube_feasible;
};

AncillaryControlResult ancillary_velocity_command_rate(
    const StateVec& actual,
    const StateVec& nominal,
    const double nominal_command_rate,
    const MPCControlBounds& applied_bounds,
    const MPCFollowAncillaryFeedbackParams& feedback
) {
    // LPV 隐状态没有跨对象统一的物理标度，辅助环只反馈可测速度和实际命令偏差。
    const double correction =
        -feedback.velocity_error_gain * (actual(ix::V) - nominal(ix::V))
        -feedback.command_error_gain * (actual(ix::V_CMD) - nominal(ix::V_CMD));
    const double desired_rate = nominal_command_rate + std::clamp(
        correction,
        -feedback.velocity_command_rate_margin,
        feedback.velocity_command_rate_margin
    );

    const double command_error = actual(ix::V_CMD) - nominal(ix::V_CMD);
    const double tube_rate_lower = nominal_command_rate
        + (-feedback.velocity_command_margin - command_error) / MPC_DT;
    const double tube_rate_upper = nominal_command_rate
        + (feedback.velocity_command_margin - command_error) / MPC_DT;
    const double lower = std::max(
        applied_bounds.lower(iu::V_CMD_RATE),
        tube_rate_lower
    );
    const double upper = std::min(
        applied_bounds.upper(iu::V_CMD_RATE),
        tube_rate_upper
    );
    if (lower <= upper) {
        return {
            .command_rate = std::clamp(desired_rate, lower, upper),
            .command_tube_feasible = true,
        };
    }
    return {
        .command_rate = std::clamp(
            desired_rate,
            applied_bounds.lower(iu::V_CMD_RATE),
            applied_bounds.upper(iu::V_CMD_RATE)
        ),
        .command_tube_feasible = false,
    };
}

// 纯追踪曲率 κ = 2·y_body/d²，既含参考曲率前馈，也回正当前横向与航向误差，
// 因此 seed rollout 会收敛到已认证的全局路径上，而不是与之平行偏移。
double reference_tracking_omega(
    const StateVec& x,
    const MincoTrajectory& trajectory,
    const double path_progress,
    const double velocity,
    const MPCFollowReferenceSeedParams& seed
) {
    const double lookahead = std::clamp(
        velocity * seed.lookahead_time,
        seed.lookahead_distance_min,
        seed.lookahead_distance_max
    );
    const Eigen::Vector2d target =
        trajectory.eval_arc_length(path_progress + lookahead).p;
    const Eigen::Vector2d delta = target - Eigen::Vector2d(x(ix::X), x(ix::Y));
    const double distance_squared = delta.squaredNorm();
    if (distance_squared < 1e-6) return 0.0;

    const double theta = x(ix::THETA);
    const double lateral = -std::sin(theta) * delta.x() + std::cos(theta) * delta.y();
    return 2.0 * lateral * velocity / distance_squared;
}

template<typename SolverT>
struct RolloutStates {
    std::array<StateVec, SolverT::N + 1> xs {};
    size_t valid_steps = SolverT::N;
};

template<typename SolverT>
struct AncillaryRollout {
    RolloutStates<SolverT> rollout;
    ControlVec first_control = ControlVec::Zero();
    bool first_command_tube_feasible = false;
};

template<typename ProblemT, typename SolverT>
AncillaryRollout<SolverT> rollout_ancillary_feedback(
    const ProblemT& problem,
    const SolverT& solver,
    const StateVec& actual_x0,
    const RolloutStates<SolverT>& nominal_rollout,
    const CapabilityProfile& effective_capability,
    const MPCStartCommandLimits& start_command,
    const MPCFollowAncillaryFeedbackParams& feedback
) {
    AncillaryRollout<SolverT> result;
    result.rollout.xs[0] = actual_x0;
    for (size_t k = 0; k < SolverT::N; ++k) {
        ControlVec applied_control = solver.us[k];
        const MPCControlBounds applied_bounds = command_rate_control_bounds(
            result.rollout.xs[k],
            effective_capability,
            k == 0 ? &start_command : nullptr
        );
        const AncillaryControlResult ancillary = ancillary_velocity_command_rate(
            result.rollout.xs[k],
            nominal_rollout.xs[k],
            solver.us[k](iu::V_CMD_RATE),
            applied_bounds,
            feedback
        );
        applied_control(iu::V_CMD_RATE) = ancillary.command_rate;
        if (k == 0) {
            result.first_control = applied_control;
            result.first_command_tube_feasible = ancillary.command_tube_feasible;
        }
        result.rollout.xs[k + 1] = problem.dynamics(
            static_cast<int>(k), result.rollout.xs[k], applied_control
        );
        if (!result.rollout.xs[k + 1].allFinite()) {
            result.rollout.valid_steps = k;
            return result;
        }
    }
    return result;
}

} // namespace

template<typename SolverT>
void shift_warm_start(SolverT& solver) {
    const auto xs_prev = solver.xs;
    const auto us_prev = solver.us;

    for (size_t k = 0; k + 1 < SolverT::N; ++k) {
        solver.us[k] = us_prev[k + 1];
    }
    solver.us[SolverT::N - 1] = us_prev[SolverT::N - 1];

    for (size_t k = 1; k < SolverT::N; ++k) {
        solver.xs[k] = xs_prev[k + 1];
    }
    solver.xs[SolverT::N] = xs_prev[SolverT::N];
}

template<typename SolverT>
void fill_solver_controls(SolverT& solver, const ControlVec& u) {
    for (size_t k = 0; k < SolverT::N; ++k) {
        solver.us[k] = u;
    }
}

// 沿已认证全局路径构造 FOLLOW 冷启动初值。零变化率初值会让 rollout 保持当前
// 角速度命令画圆弧，在弯道处直接穿入走廊外侧障碍；这里改为跟踪律前推，使 seed
// 从一开始就位于正确的同伦类中。
template<typename ProblemT, typename SolverT>
void fill_reference_tracking_seed(
    SolverT& solver,
    const ProblemT& problem,
    const StateVec& x0,
    const MincoTrajectory& trajectory,
    const PathSpeedProfile& speed_profile,
    const CapabilityProfile& capability,
    const MPCFollowReferenceSeedParams& seed
) {
    const double velocity_max = capability.command_envelope.velocity.max;
    const double velocity_rate_max = capability.command_dynamics.velocity_rate_max;
    const double angular_rate_max = capability.command_dynamics.angular_velocity_rate_max;
    const auto& angular_envelope = capability.command_envelope.angular_velocity;

    StateVec x = x0;
    for (size_t k = 0; k < SolverT::N; ++k) {
        // PATH_PROGRESS 由 problem.dynamics 按 MPC_DT · PATH_SPEED_CMD 前向积分。
        const double path_progress = x(ix::PATH_PROGRESS);
        const double path_speed = std::clamp(
            speed_profile.eval_arc_length(path_progress).velocity, 0.0, velocity_max
        );
        const double omega_reference = std::clamp(
            reference_tracking_omega(x, trajectory, path_progress, path_speed, seed),
            angular_envelope.min,
            angular_envelope.max
        );

        ControlVec& u = solver.us[k];
        u(iu::PATH_SPEED_CMD) = path_speed;
        u(iu::V_CMD_RATE) = std::clamp(
            (path_speed - x(ix::V_CMD)) / MPC_DT, -velocity_rate_max, velocity_rate_max
        );
        u(iu::W_CMD_RATE) = std::clamp(
            (omega_reference - x(ix::W_CMD)) / MPC_DT, -angular_rate_max, angular_rate_max
        );

        const StateVec next = problem.dynamics(static_cast<int>(k), x, u);
        if (!next.allFinite()) {
            // 前推失效时保持剩余控制为该拍的值，硬边界由 solver 的投影负责收紧。
            for (size_t rest = k + 1; rest < SolverT::N; ++rest) solver.us[rest] = u;
            return;
        }
        x = next;
    }
}

template<typename ProblemT, typename StateContainerT>
std::optional<RolloutLethalObstacleInfo> detect_rollout_lethal_obstacle(
    const ProblemT& prob,
    const StateContainerT& xs,
    size_t state_count
) {
    for (size_t i = 0; i < state_count; ++i) {
        if (const auto lethal = prob.detect_lethal_obstacle(static_cast<int>(i), xs[i])) {
            return lethal;
        }
    }
    return std::nullopt;
}

template<typename ProblemT, typename SolverT>
RolloutStates<SolverT> rollout_states(const ProblemT& prob, const SolverT& solver, const StateVec& x0) {
    RolloutStates<SolverT> rollout;
    rollout.xs[0] = x0;
    for (size_t k = 0; k < SolverT::N; ++k) {
        rollout.xs[k + 1] = prob.dynamics(static_cast<int>(k), rollout.xs[k], solver.us[k]);
        if (!rollout.xs[k + 1].allFinite()) {
            rollout.valid_steps = k;
            return rollout;
        }
    }
    return rollout;
}

template<typename SolverT>
MPCPrediction prediction_from_rollout(const RolloutStates<SolverT>& rollout) {
    MPCPrediction pred;
    const size_t sz = rollout.valid_steps + 1;
    pred.path_map.reserve(sz);
    pred.headings.reserve(sz);
    pred.v_pred.reserve(sz);
    pred.w_pred.reserve(sz);
    pred.path_progress_pred.reserve(sz);
    pred.path_speed_pred.reserve(sz);
    for (size_t i = 0; i <= rollout.valid_steps; ++i) {
        const auto& x = rollout.xs[i];
        pred.path_map.emplace_back(x(ix::X), x(ix::Y));
        pred.headings.push_back(x(ix::THETA));
        pred.v_pred.push_back(x(ix::V));
        pred.w_pred.push_back(x(ix::W));
        pred.path_progress_pred.push_back(x(ix::PATH_PROGRESS));
        pred.path_speed_pred.push_back(x(ix::PATH_SPEED));
    }
    return pred;
}

template<typename ProblemT, typename SolverT>
MPCPrediction rollout_prediction(const ProblemT& prob, const SolverT& solver, const StateVec& x0) {
    return prediction_from_rollout(rollout_states(prob, solver, x0));
}

MPCDiagnostics initial_diagnostics(
    const MPCSolverMode mode,
    const StateVec& measured_x0
) {
    MPCDiagnostics diagnostics;
    diagnostics.solver_mode = mode;
    diagnostics.measured_velocity = {
        measured_x0(ix::V), measured_x0(ix::W)
    };
    diagnostics.previous_command = {
        measured_x0(ix::V_CMD), measured_x0(ix::W_CMD)
    };
    return diagnostics;
}

// ── MPCSolver 方法 ──

MPCSolver::MPCSolver(const MPCParams& params, rclcpp::Logger logger)
    : params_(params),
      logger_(std::move(logger)),
      observer_(params.kinematic_model, logger_.get_child("observer")) {}

MPCSolver::~MPCSolver() = default;

void MPCSolver::set_command_state(
    const Eigen::Vector2d& command,
    const Eigen::Vector2d& command_rate
) {
    last_cmd_ = command;
    last_cmd_rate_ = command_rate;
}

void MPCSolver::reset_warm_start() {
    // 清除 warm 标记即可：三个 solver 都会在下次求解的冷启动分支重填整条控制序列。
    follow_warm_ = false;
    stop_warm_ = false;
    hold_warm_ = false;
    follow_nominal_longitudinal_state_.reset();
    follow_lethal_consecutive_count_ = 0;
}

void MPCSolver::update_observer(
    const ChassisMotionState& chassis_state,
    const uint64_t state_sequence
) {
    const ObserverDiagnostics& diagnostics = observer_.update(
        chassis_state,
        last_cmd_,
        state_sequence
    );
    if (diagnostics.event == ObserverUpdateEvent::RESET) {
        follow_nominal_longitudinal_state_.reset();
    }
}

void MPCSolver::reset_observer(const ObserverResetReason reason) {
    observer_.reset(reason);
    follow_nominal_longitudinal_state_.reset();
}

StateVec MPCSolver::make_initial_state(
    const Eigen::Vector3d& pose,
    const ChassisMotionState& chassis_state,
    const Eigen::Vector2d& current_command,
    const Eigen::Vector2d& current_command_rate,
    const double path_progress,
    const double path_speed
) const {
    StateVec x0;
    x0(ix::X) = pose.x();
    x0(ix::Y) = pose.y();
    x0(ix::THETA) = pose.z();
    x0(ix::XH) = observer_.validated() ? observer_.hidden_state_estimate() : 0.0;
    x0(ix::V) = chassis_state.velocity;
    x0(ix::W) = chassis_state.omega;
    x0(ix::V_CMD) = current_command.x();
    x0(ix::W_CMD) = current_command.y();
    x0(ix::V_CMD_RATE) = current_command_rate.x();
    x0(ix::W_CMD_RATE) = current_command_rate.y();
    x0(ix::PATH_PROGRESS) = path_progress;
    x0(ix::PATH_SPEED) = path_speed;
    return x0;
}

std::expected<MPCSolver::FollowSolveResult, std::string> MPCSolver::solve_follow(
    const MincoTrajectory& global_trajectory,
    const PathSpeedProfile& speed_profile,
    const Eigen::Vector3d& chassis_pose_map,
    const ChassisMotionState& chassis_state,
    const double current_path_progress,
    const double current_path_speed,
    const CostMap& cost_map,
    const CostMap& masked_global_map,
    const std::vector<const CostMap*>& per_step_cost_maps,
    double prediction_dt,
    const CapabilityProfile& effective_capability,
    std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule,
    bool check_lethal_status
) {
    if (!step_constraint_schedule) {
        step_constraint_schedule = std::make_shared<const StepConstraintSchedule>(
            std::vector<StepTraversalConstraint> {}
        );
    }

    // 有向进度估计是外部观测；虚拟进度速度允许从零开始，因此下界是 0 而非命令下限。
    const double path_progress0 = std::clamp(
        current_path_progress, 0.0, global_trajectory.total_arc_length()
    );
    const double path_speed0 = std::clamp(
        current_path_speed, 0.0, effective_capability.command_envelope.velocity.max
    );

    const double schedule_rho = select_follow_schedule_rho(
        chassis_state,
        params_.kinematic_model
    );

    // per_step_cost_maps 是 t0 + (i+1)·prediction_dt 的预测帧；当前帧必须作为
    // 时域向量的第 0 项，否则 stage 0 会错用未来 +prediction_dt 的地图。
    const bool prediction_usable = !per_step_cost_maps.empty()
        && std::isfinite(prediction_dt) && prediction_dt > 0.0;
    std::vector<const CostMap*> step_cost_maps;
    step_cost_maps.reserve(prediction_usable ? per_step_cost_maps.size() + 1 : 1);
    step_cost_maps.push_back(&cost_map);
    if (prediction_usable) {
        for (const auto* cm : per_step_cost_maps) {
            step_cost_maps.push_back(cm);
        }
    }
    const double pred_dt = prediction_usable ? prediction_dt : MPC_DT;

    const StateVec measured_x0 = make_initial_state(
        chassis_pose_map, chassis_state, last_cmd_, last_cmd_rate_, path_progress0, path_speed0
    );
    if (!measured_x0.allFinite()) {
        follow_nominal_longitudinal_state_.reset();
        follow_warm_ = false;
        return std::unexpected("Follow MPC received a non-finite initial state");
    }

    const auto& feedback = params_.follow.ancillary_feedback;
    const bool feedback_active = feedback.enable && observer_.validated();
    if (!feedback_active) follow_nominal_longitudinal_state_.reset();
    const LongitudinalNominalInitialState nominal_initial = feedback_active
        ? longitudinal_nominal_initial_state(
            measured_x0, follow_nominal_longitudinal_state_, feedback
        )
        : LongitudinalNominalInitialState {.state = measured_x0, .reanchored = true};
    const StateVec& x0 = nominal_initial.state;
    const CapabilityProfile nominal_capability = feedback_active
        ? tightened_nominal_capability(effective_capability, feedback)
        : effective_capability;

    const FollowProblem problem(
        global_trajectory, speed_profile, params_, step_cost_maps, masked_global_map,
        pred_dt, schedule_rho, nominal_capability, step_constraint_schedule
    );

    fddp::SolverOptions opts;
    opts.max_iters = params_.follow.max_iters;
    opts.tol_grad = SOLVER_TOL_GRAD;

    // 纵向重锚只替换 x0 的纵向分量，与控制序列初值是否可用无关，因此不影响
    // warm start 的有效性；尤其不能丢弃上一帧已收敛的角向命令变化率序列。
    if (follow_warm_) {
        shift_warm_start(follow_solver_);
    } else {
        fill_reference_tracking_seed(
            follow_solver_, problem, x0, global_trajectory, speed_profile,
            nominal_capability, params_.follow.reference_seed
        );
    }
    follow_solver_.xs[0] = x0;
    const auto solver_result = follow_solver_.solve(problem, opts);
    if (!solver_result.feasible) {
        follow_nominal_longitudinal_state_.reset();
        follow_warm_ = false;
        return std::unexpected(
            "Follow MPC hard command bounds are infeasible from the current command"
        );
    }
    follow_warm_ = true;

    const auto nominal_rollout = rollout_states(problem, follow_solver_, x0);
    if (nominal_rollout.valid_steps != fddp::Solver<FollowProblem>::N) {
        follow_nominal_longitudinal_state_.reset();
        follow_warm_ = false;
        return std::unexpected("Follow MPC produced a non-finite nominal rollout");
    }
    ControlVec applied_control = follow_solver_.us[0];
    bool first_command_tube_feasible = true;
    auto applied_rollout = nominal_rollout;
    if (feedback_active) {
        auto ancillary_rollout = rollout_ancillary_feedback(
            problem,
            follow_solver_,
            measured_x0,
            nominal_rollout,
            effective_capability,
            params_.follow.start_command,
            feedback
        );
        applied_control = ancillary_rollout.first_control;
        first_command_tube_feasible = ancillary_rollout.first_command_tube_feasible;
        applied_rollout = std::move(ancillary_rollout.rollout);
    }
    if (applied_rollout.valid_steps != fddp::Solver<FollowProblem>::N
        || !applied_control.allFinite()) {
        follow_nominal_longitudinal_state_.reset();
        follow_warm_ = false;
        return std::unexpected("Follow MPC produced a non-finite applied rollout");
    }
    MPCDiagnostics diagnostics = initial_diagnostics(MPCSolverMode::FOLLOW, measured_x0);
    diagnostics.solve_succeeded = true;
    diagnostics.solver_termination = {
        .iters = solver_result.iters,
        .converged = solver_result.converged,
        .cost = solver_result.cost,
    };
    diagnostics.ancillary_enabled = feedback.enable;
    diagnostics.ancillary_active = feedback_active;
    diagnostics.nominal_reanchored = feedback_active && nominal_initial.reanchored;
    diagnostics.first_command_tube_feasible = feedback_active
        && first_command_tube_feasible;
    diagnostics.nominal_command = command_after_control(x0, follow_solver_.us[0]);
    diagnostics.nominal_command_rate = {
        follow_solver_.us[0](iu::V_CMD_RATE),
        follow_solver_.us[0](iu::W_CMD_RATE)
    };
    diagnostics.applied_command_rate = {
        applied_control(iu::V_CMD_RATE), applied_control(iu::W_CMD_RATE)
    };
    diagnostics.nominal_prediction = prediction_from_rollout(nominal_rollout);
    diagnostics.applied_prediction = prediction_from_rollout(applied_rollout);

    diagnostics.reference_path_progress.reserve(MPC_HORIZON);
    diagnostics.reference_path_speed.reserve(MPC_HORIZON);
    diagnostics.trajectory_nominal_velocity.reserve(MPC_HORIZON);
    diagnostics.trajectory_nominal_angular_velocity.reserve(MPC_HORIZON);
    diagnostics.reference_velocity.reserve(MPC_HORIZON);
    diagnostics.reference_angular_velocity.reserve(MPC_HORIZON);
    for (size_t k = 0; k < MPC_HORIZON; ++k) {
        const double path_progress = nominal_rollout.xs[k](ix::PATH_PROGRESS);
        const double path_speed = follow_solver_.us[k](iu::PATH_SPEED_CMD);
        const TrajSample sample = global_trajectory.eval_arc_length(path_progress);
        const double nominal_velocity =
            speed_profile.eval_arc_length(path_progress).velocity;
        diagnostics.reference_path_progress.push_back(path_progress);
        diagnostics.reference_path_speed.push_back(path_speed);
        diagnostics.trajectory_nominal_velocity.push_back(nominal_velocity);
        diagnostics.trajectory_nominal_angular_velocity.push_back(
            sample.kappa * nominal_velocity
        );
        diagnostics.reference_velocity.push_back(path_speed);
        diagnostics.reference_angular_velocity.push_back(sample.kappa * path_speed);
    }

    if (check_lethal_status) {
        const auto lethal = detect_rollout_lethal_obstacle(
            problem, applied_rollout.xs, applied_rollout.valid_steps + 1
        );
        if (lethal.has_value()) {
            // 命令安全不做消抖：命中致命障碍的 rollout 一次也不下发。坏解同时失去
            // warm start 资格，下一帧从参考跟踪 seed 重启，不再从坏盆地继续。
            ++follow_lethal_consecutive_count_;
            follow_warm_ = false;
            follow_nominal_longitudinal_state_.reset();

            auto stop_result = solve_stop(chassis_pose_map, chassis_state, cost_map);
            if (!stop_result) {
                return std::unexpected(stop_result.error());
            }

            const bool replan_requested = follow_lethal_consecutive_count_
                >= params_.follow.rollout_safety.lethal_replan_consecutive_threshold;
            stop_result->diagnostics.rejected_follow_rollout = RejectedFollowRollout {
                .lethal = *lethal,
                .prediction = std::move(diagnostics.applied_prediction),
                .consecutive_count = follow_lethal_consecutive_count_,
                .replan_requested = replan_requested,
            };

            FollowSolveResult out;
            out.command = stop_result->command;
            out.diagnostics = std::move(stop_result->diagnostics);
            out.status = replan_requested
                ? FollowSolveStatus::STOP_AND_WAIT_REPLAN
                : FollowSolveStatus::STOP;
            out.lethal_obstacle = lethal;
            return out;
        }
        follow_lethal_consecutive_count_ = 0;
    } else {
        follow_lethal_consecutive_count_ = 0;
    }

    if (feedback_active) {
        if (first_command_tube_feasible) {
            follow_nominal_longitudinal_state_ = nominal_rollout.xs[1];
        } else {
            follow_nominal_longitudinal_state_.reset();
        }
    }

    const Eigen::Vector2d cmd = command_after_control(measured_x0, applied_control);
    if (!cmd.allFinite()) {
        follow_nominal_longitudinal_state_.reset();
        follow_warm_ = false;
        return std::unexpected("Follow MPC produced a non-finite command");
    }
    last_cmd_rate_.x() = applied_control(iu::V_CMD_RATE);
    last_cmd_rate_.y() = applied_control(iu::W_CMD_RATE);
    last_cmd_ = cmd;

    FollowSolveResult out;
    out.command = cmd;
    out.diagnostics = std::move(diagnostics);
    out.status = FollowSolveStatus::FOLLOW;
    return out;
}

std::expected<MPCSolver::SolveResult, std::string> MPCSolver::solve_stop(
    const Eigen::Vector3d& chassis_pose_map,
    const ChassisMotionState& chassis_state,
    const CostMap& cost_map
) {
    follow_nominal_longitudinal_state_.reset();
    const double schedule_rho = schedule_rho_from_state(chassis_state, params_.kinematic_model);

    const StateVec x0 = make_initial_state(
        chassis_pose_map, chassis_state, last_cmd_, last_cmd_rate_, 0.0, 0.0
    );
    if (!x0.allFinite()) {
        stop_warm_ = false;
        return std::unexpected("Stop MPC received a non-finite initial state");
    }

    StopProblem prob(params_, cost_map, schedule_rho);
    if (stop_warm_) {
        shift_warm_start(stop_solver_);
    } else {
        fill_solver_controls(stop_solver_, ControlVec::Zero());
    }
    fddp::SolverOptions opts;
    opts.max_iters = params_.stop.max_iters;
    opts.tol_grad = SOLVER_TOL_GRAD;
    stop_solver_.xs[0] = x0;
    const auto solver_result = stop_solver_.solve(prob, opts);
    if (!solver_result.feasible) {
        return std::unexpected(
            "Stop MPC hard command bounds are infeasible from the current command"
        );
    }
    stop_warm_ = true;

    const Eigen::Vector2d cmd = command_after_control(x0, stop_solver_.us[0]);
    if (!cmd.allFinite()) {
        stop_warm_ = false;
        return std::unexpected("Stop MPC produced a non-finite command");
    }
    last_cmd_rate_.x() = stop_solver_.us[0](iu::V_CMD_RATE);
    last_cmd_rate_.y() = stop_solver_.us[0](iu::W_CMD_RATE);
    last_cmd_ = cmd;
    MPCDiagnostics diagnostics = initial_diagnostics(MPCSolverMode::STOP, x0);
    diagnostics.solve_succeeded = true;
    diagnostics.solver_termination = {
        .iters = solver_result.iters,
        .converged = solver_result.converged,
        .cost = solver_result.cost,
    };
    diagnostics.nominal_command = cmd;
    diagnostics.nominal_command_rate = {
        stop_solver_.us[0](iu::V_CMD_RATE), stop_solver_.us[0](iu::W_CMD_RATE)
    };
    diagnostics.applied_command_rate = diagnostics.nominal_command_rate;
    diagnostics.nominal_prediction = rollout_prediction(prob, stop_solver_, x0);
    diagnostics.applied_prediction = diagnostics.nominal_prediction;
    return SolveResult {.command = cmd, .diagnostics = std::move(diagnostics)};
}

std::expected<MPCSolver::SolveResult, std::string> MPCSolver::solve_hold(
    const Eigen::Vector2d& goal_map,
    const Eigen::Vector3d& chassis_pose_map,
    const ChassisMotionState& chassis_state,
    const CostMap& cost_map
) {
    follow_nominal_longitudinal_state_.reset();
    const double schedule_rho = schedule_rho_from_state(chassis_state, params_.kinematic_model);

    const StateVec x0 = make_initial_state(
        chassis_pose_map, chassis_state, last_cmd_, last_cmd_rate_, 0.0, 0.0
    );
    if (!x0.allFinite()) {
        hold_warm_ = false;
        return std::unexpected("Hold MPC received a non-finite initial state");
    }

    HoldProblem prob(goal_map, params_, cost_map, schedule_rho);
    if (hold_warm_) {
        shift_warm_start(hold_solver_);
    } else {
        fill_solver_controls(hold_solver_, ControlVec::Zero());
    }
    fddp::SolverOptions opts;
    opts.max_iters = params_.hold.max_iters;
    opts.tol_grad = SOLVER_TOL_GRAD;
    hold_solver_.xs[0] = x0;
    const auto solver_result = hold_solver_.solve(prob, opts);
    if (!solver_result.feasible) {
        return std::unexpected(
            "Hold MPC hard command bounds are infeasible from the current command"
        );
    }
    hold_warm_ = true;

    const Eigen::Vector2d cmd = command_after_control(x0, hold_solver_.us[0]);
    if (!cmd.allFinite()) {
        hold_warm_ = false;
        return std::unexpected("Hold MPC produced a non-finite command");
    }
    last_cmd_rate_.x() = hold_solver_.us[0](iu::V_CMD_RATE);
    last_cmd_rate_.y() = hold_solver_.us[0](iu::W_CMD_RATE);
    last_cmd_ = cmd;
    MPCDiagnostics diagnostics = initial_diagnostics(MPCSolverMode::HOLD, x0);
    diagnostics.solve_succeeded = true;
    diagnostics.solver_termination = {
        .iters = solver_result.iters,
        .converged = solver_result.converged,
        .cost = solver_result.cost,
    };
    diagnostics.nominal_command = cmd;
    diagnostics.nominal_command_rate = {
        hold_solver_.us[0](iu::V_CMD_RATE), hold_solver_.us[0](iu::W_CMD_RATE)
    };
    diagnostics.applied_command_rate = diagnostics.nominal_command_rate;
    diagnostics.nominal_prediction = rollout_prediction(prob, hold_solver_, x0);
    diagnostics.applied_prediction = diagnostics.nominal_prediction;
    return SolveResult {.command = cmd, .diagnostics = std::move(diagnostics)};
}

} // namespace nav_executor
