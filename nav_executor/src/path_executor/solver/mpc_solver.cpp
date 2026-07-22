#include <nav_executor/path_executor/solver/mpc_solver.hpp>
#include <nav_executor/path_executor/solver/mpc_utils.hpp>
#include <nav_executor/path_executor/solver/bilinear_sampling.hpp>
#include <nav_executor/path_executor/solver/lpv_model.hpp>

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
    const MPCFollowAncillaryFeedbackParams& feedback,
    const double phase_rate_min,
    const double phase_rate_max
) {
    AncillaryRollout<SolverT> result;
    result.rollout.xs[0] = actual_x0;
    for (size_t k = 0; k < SolverT::N; ++k) {
        ControlVec applied_control = solver.us[k];
        const MPCControlBounds applied_bounds = command_rate_control_bounds(
            result.rollout.xs[k],
            effective_capability,
            phase_rate_min,
            phase_rate_max,
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

template<typename ProblemT, typename SolverT>
MPCPrediction rollout_prediction(const ProblemT& prob, const SolverT& solver, const StateVec& x0) {
    const auto rollout = rollout_states(prob, solver, x0);

    MPCPrediction pred;
    const size_t sz = rollout.valid_steps + 1;
    pred.path_map.reserve(sz);
    pred.headings.reserve(sz);
    pred.v_pred.reserve(sz);
    pred.w_pred.reserve(sz);
    pred.phase_time_pred.reserve(sz);
    pred.phase_rate_pred.reserve(sz);
    for (size_t i = 0; i <= rollout.valid_steps; ++i) {
        const auto& x = rollout.xs[i];
        pred.path_map.emplace_back(x(ix::X), x(ix::Y));
        pred.headings.push_back(x(ix::THETA));
        pred.v_pred.push_back(x(ix::V));
        pred.w_pred.push_back(x(ix::W));
        pred.phase_time_pred.push_back(x(ix::PHASE_TIME));
        pred.phase_rate_pred.push_back(x(ix::PHASE_RATE));
    }
    return pred;
}

// ── MPCSolver 方法 ──

MPCSolver::MPCSolver(const MPCParams& params, rclcpp::Logger logger)
    : params_(params),
      logger_(std::move(logger)),
      last_phase_rate_(params.follow.phase.nominal_rate) {}

MPCSolver::~MPCSolver() = default;

void MPCSolver::set_command_state(
    const Eigen::Vector2d& command,
    const Eigen::Vector2d& command_rate
) {
    last_cmd_ = command;
    last_cmd_rate_ = command_rate;
}

void MPCSolver::reset_warm_start() {
    follow_warm_ = false;
    stop_warm_ = false;
    hold_warm_ = false;
    follow_nominal_longitudinal_state_.reset();
    fddp_lethal_consecutive_count_ = 0;
    last_phase_rate_ = params_.follow.phase.nominal_rate;
    for (size_t k = 0; k < MPC_HORIZON; ++k) {
        follow_solver_.us[k].setZero();
        stop_solver_.us[k].setZero();
        hold_solver_.us[k].setZero();
    }
}

void MPCSolver::update_observer(const ChassisMotionState& chassis_state) {
    if (!std::isfinite(chassis_state.velocity) || !std::isfinite(chassis_state.omega)
        || !std::isfinite(chassis_state.leg_h) || !std::isfinite(chassis_state.leg_psi)) {
        reset_observer();
        return;
    }
    const double v_act = chassis_state.velocity;
    const double w_act = chassis_state.omega;
    const double rho_cur = schedule_rho_from_state(chassis_state, params_.kinematic_model);
    if (!observer_initialized_) {
        const double initial_xh = params_.kinematic_model.xh0_bias
            + params_.kinematic_model.xh0_psi * chassis_state.leg_psi
            + params_.kinematic_model.xh0_v * v_act;
        if (!std::isfinite(rho_cur) || !std::isfinite(initial_xh)) {
            reset_observer();
            return;
        }
        x_h_hat_ = initial_xh;
        prev_v_act_ = v_act;
        prev_w_act_ = w_act;
        prev_schedule_rho_ = rho_cur;
        observer_input_command_ = last_cmd_;
        observer_initialized_ = true;
        return;
    }
    const auto model = build_lpv_discrete_model(params_.kinematic_model, 0.5 * (prev_schedule_rho_ + rho_cur));
    const auto nl_eval = evaluate_lpv_nonlinear(prev_v_act_, prev_w_act_, model);
    // 辨识模型包含一拍输入延迟，因此当前观测应由上次 observer 更新时捕获的命令预测。
    const double xh_pred = model.ad00 * x_h_hat_ + model.ad01 * prev_v_act_ + model.bd0 * observer_input_command_.x() + model.gd0 * nl_eval.nl;
    const double v_pred = model.ad10 * x_h_hat_ + model.ad11 * prev_v_act_ + model.bd1 * observer_input_command_.x() + model.gd1 * nl_eval.nl;
    const double w_pred = model.alpha_w * prev_w_act_
        + model.beta_w * observer_input_command_.y() - model.gamma_w * nl_eval.sw;
    const double psi_proxy_pred = params_.kinematic_model.psi_bias + params_.kinematic_model.psi_gain * xh_pred
        + params_.kinematic_model.psi_v * v_pred;
    const double velocity_innovation = v_act - v_pred;
    const double angular_velocity_innovation = w_act - w_pred;
    const double leg_psi_innovation = chassis_state.leg_psi - psi_proxy_pred;
    const double corrected_xh = xh_pred
        + params_.kinematic_model.obs_lv * velocity_innovation
        + params_.kinematic_model.obs_lpsi * leg_psi_innovation;
    if (!std::isfinite(xh_pred)
        || !std::isfinite(v_pred)
        || !std::isfinite(w_pred)
        || !std::isfinite(psi_proxy_pred)
        || !std::isfinite(velocity_innovation)
        || !std::isfinite(angular_velocity_innovation)
        || !std::isfinite(leg_psi_innovation)
        || !std::isfinite(corrected_xh)
        || std::abs(velocity_innovation) > params_.kinematic_model.obs_v_innovation_max
        || std::abs(angular_velocity_innovation) > params_.kinematic_model.obs_omega_innovation_max
        || std::abs(leg_psi_innovation) > params_.kinematic_model.obs_psi_innovation_max) {
        const bool should_log = !observer_rejection_active_;
        reset_observer();
        observer_rejection_active_ = true;
        if (should_log) {
            RCLCPP_WARN(
                logger_,
                "Rejecting LPV observer innovation: velocity=%.3f m/s, omega=%.3f rad/s, leg_psi=%.3f rad",
                velocity_innovation,
                angular_velocity_innovation,
                leg_psi_innovation
            );
        }
        return;
    }
    x_h_hat_ = corrected_xh;
    prev_v_act_ = v_act;
    prev_w_act_ = w_act;
    prev_schedule_rho_ = rho_cur;
    observer_input_command_ = last_cmd_;
    observer_validated_ = true;
    observer_rejection_active_ = false;
}

void MPCSolver::reset_observer() {
    x_h_hat_ = 0.0;
    prev_v_act_ = 0.0;
    prev_w_act_ = 0.0;
    observer_input_command_.setZero();
    observer_initialized_ = false;
    observer_validated_ = false;
    observer_rejection_active_ = false;
    follow_nominal_longitudinal_state_.reset();
}

StateVec MPCSolver::make_initial_state(
    const Eigen::Vector3d& pose,
    const ChassisMotionState& chassis_state,
    const Eigen::Vector2d& current_command,
    const Eigen::Vector2d& current_command_rate,
    const double phase_time,
    const double phase_rate
) const {
    StateVec x0;
    x0(ix::X) = pose.x();
    x0(ix::Y) = pose.y();
    x0(ix::THETA) = pose.z();
    x0(ix::XH) = observer_validated_ ? x_h_hat_ : 0.0;
    x0(ix::V) = chassis_state.velocity;
    x0(ix::W) = chassis_state.omega;
    x0(ix::V_CMD) = current_command.x();
    x0(ix::W_CMD) = current_command.y();
    x0(ix::V_CMD_RATE) = current_command_rate.x();
    x0(ix::W_CMD_RATE) = current_command_rate.y();
    x0(ix::PHASE_TIME) = phase_time;
    x0(ix::PHASE_RATE) = phase_rate;
    return x0;
}

std::expected<MPCSolver::FollowSolveResult, std::string> MPCSolver::solve_follow(
    const MincoTrajectory& global_trajectory,
    const Eigen::Vector3d& chassis_pose_map,
    const ChassisMotionState& chassis_state,
    const double current_phase_time,
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

    const double phase_time0 = std::clamp(current_phase_time, 0.0, global_trajectory.total_time());
    const double phase_rate0 = std::clamp(
        last_phase_rate_, params_.follow.phase.rate_min, params_.follow.phase.rate_max
    );

    const double schedule_rho = select_follow_schedule_rho(
        chassis_state,
        params_.kinematic_model
    );

    std::vector<CostMapGridView> step_cost_grids;
    if (per_step_cost_maps.empty()) {
        step_cost_grids.reserve(1);
    } else {
        step_cost_grids.reserve(per_step_cost_maps.size());
    }
    if (per_step_cost_maps.empty()) {
        step_cost_grids.emplace_back(cost_map);
    } else {
        for (const auto* cm : per_step_cost_maps) {
            step_cost_grids.emplace_back(*cm);
        }
    }
    const double pred_dt = per_step_cost_maps.empty() ? MPC_DT : prediction_dt;

    const GridInfo ci = make_grid_info(cost_map);
    const CostMapGridView masked_global_grid(masked_global_map);
    const StateVec measured_x0 = make_initial_state(
        chassis_pose_map, chassis_state, last_cmd_, last_cmd_rate_, phase_time0, phase_rate0
    );
    if (!measured_x0.allFinite()) {
        follow_nominal_longitudinal_state_.reset();
        follow_warm_ = false;
        return std::unexpected("Follow MPC received a non-finite initial state");
    }

    const auto& feedback = params_.follow.ancillary_feedback;
    const bool feedback_active = feedback.enable && observer_validated_;
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
        global_trajectory, params_, step_cost_grids, ci, masked_global_grid, pred_dt, schedule_rho,
        nominal_capability, step_constraint_schedule
    );

    fddp::SolverOptions opts;
    opts.max_iters = params_.follow.max_iters;
    opts.tol_grad = SOLVER_TOL_GRAD;

    if (feedback_active && nominal_initial.reanchored) follow_warm_ = false;
    if (follow_warm_) {
        shift_warm_start(follow_solver_);
    } else {
        ControlVec initial_control = ControlVec::Zero();
        initial_control(iu::PHASE_RATE_CMD) = params_.follow.phase.nominal_rate;
        fill_solver_controls(follow_solver_, initial_control);
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
            feedback,
            params_.follow.phase.rate_min,
            params_.follow.phase.rate_max
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
    MPCPrediction prediction;
    {
        const size_t sz = applied_rollout.valid_steps + 1;
        prediction.path_map.reserve(sz);
        prediction.headings.reserve(sz);
        prediction.v_pred.reserve(sz);
        prediction.w_pred.reserve(sz);
        prediction.phase_time_pred.reserve(sz);
        prediction.phase_rate_pred.reserve(sz);
        for (size_t i = 0; i <= applied_rollout.valid_steps; ++i) {
            const auto& x = applied_rollout.xs[i];
            prediction.path_map.emplace_back(x(ix::X), x(ix::Y));
            prediction.headings.push_back(x(ix::THETA));
            prediction.v_pred.push_back(x(ix::V));
            prediction.w_pred.push_back(x(ix::W));
            prediction.phase_time_pred.push_back(x(ix::PHASE_TIME));
            prediction.phase_rate_pred.push_back(x(ix::PHASE_RATE));
        }
    }

    if (check_lethal_status) {
        const auto& safety = params_.follow.rollout_safety;
        const auto lethal = detect_rollout_lethal_obstacle(
            problem, applied_rollout.xs, applied_rollout.valid_steps + 1
        );

        if (lethal.has_value()) {
            ++fddp_lethal_consecutive_count_;
        } else {
            fddp_lethal_consecutive_count_ = 0;
        }

        const int threshold = safety.fddp_lethal_consecutive_threshold;
        if (lethal.has_value() && fddp_lethal_consecutive_count_ >= threshold) {
            auto stop_result = solve_stop(chassis_pose_map, chassis_state, cost_map);
            if (!stop_result) {
                return std::unexpected(stop_result.error());
            }

            FollowSolveResult out;
            out.command = std::get<0>(*stop_result);
            out.prediction = std::get<1>(*stop_result);
            out.status = FollowSolveStatus::STOP_AND_WAIT_REPLAN;
            out.lethal_obstacle = lethal;
            return out;
        }
    } else {
        fddp_lethal_consecutive_count_ = 0;
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
    last_phase_rate_ = follow_solver_.us[0](iu::PHASE_RATE_CMD);
    last_cmd_rate_.x() = applied_control(iu::V_CMD_RATE);
    last_cmd_rate_.y() = applied_control(iu::W_CMD_RATE);
    last_cmd_ = cmd;

    FollowSolveResult out;
    out.command = cmd;
    out.prediction = std::move(prediction);
    out.status = FollowSolveStatus::FOLLOW;
    return out;
}

std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> MPCSolver::solve_stop(
    const Eigen::Vector3d& chassis_pose_map,
    const ChassisMotionState& chassis_state,
    const CostMap& cost_map
) {
    follow_nominal_longitudinal_state_.reset();
    const double schedule_rho = schedule_rho_from_state(chassis_state, params_.kinematic_model);

    const CostMapGridView cg(cost_map);
    const GridInfo ci = make_grid_info(cost_map);
    const StateVec x0 = make_initial_state(
        chassis_pose_map, chassis_state, last_cmd_, last_cmd_rate_, 0.0, 0.0
    );
    if (!x0.allFinite()) {
        stop_warm_ = false;
        return std::unexpected("Stop MPC received a non-finite initial state");
    }

    StopProblem prob(params_, cg, ci, schedule_rho);
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
    return std::tuple {cmd, rollout_prediction(prob, stop_solver_, x0)};
}

std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> MPCSolver::solve_hold(
    const Eigen::Vector2d& goal_map,
    const Eigen::Vector3d& chassis_pose_map,
    const ChassisMotionState& chassis_state,
    const CostMap& cost_map
) {
    follow_nominal_longitudinal_state_.reset();
    const double schedule_rho = schedule_rho_from_state(chassis_state, params_.kinematic_model);

    const CostMapGridView cg(cost_map);
    const GridInfo ci = make_grid_info(cost_map);
    const StateVec x0 = make_initial_state(
        chassis_pose_map, chassis_state, last_cmd_, last_cmd_rate_, 0.0, 0.0
    );
    if (!x0.allFinite()) {
        hold_warm_ = false;
        return std::unexpected("Hold MPC received a non-finite initial state");
    }

    HoldProblem prob(goal_map, params_, cg, ci, schedule_rho);
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
    return std::tuple {cmd, rollout_prediction(prob, hold_solver_, x0)};
}

} // namespace nav_executor
