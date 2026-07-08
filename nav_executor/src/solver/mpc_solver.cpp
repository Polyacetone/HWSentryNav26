#include <nav_executor/solver/mpc_solver.hpp>
#include <nav_executor/solver/mpc_utils.hpp>
#include <nav_executor/solver/bilinear_sampling.hpp>
#include <nav_executor/solver/lpv_model.hpp>
#include <nav_executor/solver/mppi_sampler.hpp>

namespace nav_executor {

// ── Solver 辅助模板 ──

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

template<typename SolverT, typename ProblemT>
void rollout_solver_states(SolverT& solver, const ProblemT& prob, const StateVec& x0) {
    solver.xs[0] = x0;
    for (size_t k = 0; k < SolverT::N; ++k) {
        solver.xs[k + 1] = prob.dynamics(static_cast<int>(k), solver.xs[k], solver.us[k]);
    }
}

template<typename SolverT>
void fill_solver_controls(SolverT& solver, const ControlVec& u) {
    for (size_t k = 0; k < SolverT::N; ++k) {
        solver.us[k] = u;
    }
}

template<typename SolverT, typename ProblemT>
void scale_solver_controls(SolverT& solver, const ProblemT& prob) {
    const auto u_lo = prob.u_lower();
    const auto u_hi = prob.u_upper();
    for (size_t k = 0; k < SolverT::N; ++k) {
        auto& u = solver.us[k];
        double scale = 1.0;
        for (int i = 0; i < SolverT::NU; ++i) {
            if (u(i) > u_hi(i) && u_hi(i) != 0.0) {
                scale = std::min(scale, u_hi(i) / u(i));
            } else if (u(i) < u_lo(i) && u_lo(i) != 0.0) {
                scale = std::min(scale, u_lo(i) / u(i));
            }
        }
        if (scale < 1.0) {
            u *= scale;
        }
        for (int i = 0; i < SolverT::NU; ++i) {
            u(i) = std::clamp(u(i), u_lo(i), u_hi(i));
        }
    }
}

template<typename SolverT>
std::array<ControlVec, SolverT::N> copy_solver_controls(const SolverT& solver) {
    return solver.us;
}

template<typename SolverT>
void seed_solver_from_sampling_result(SolverT& solver, const MPPIFollowSamplingResult& sample) {
    solver.xs = sample.xs;
    solver.us = sample.us;
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

template<typename SolverT>
struct RolloutStates {
    std::array<StateVec, SolverT::N + 1> xs {};
    size_t valid_steps = SolverT::N;
};

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
    for (size_t i = 0; i <= rollout.valid_steps; ++i) {
        const auto& x = rollout.xs[i];
        pred.path_map.emplace_back(x(ix::X), x(ix::Y));
        pred.headings.push_back(x(ix::THETA));
        pred.v_pred.push_back(x(ix::V));
        pred.w_pred.push_back(x(ix::W));
    }
    return pred;
}

// ── MPCSolver 方法 ──

MPCSolver::MPCSolver(const MPCParams& params): params_(params) {
    step_cost_grids_cache_.reserve(MPC_HORIZON + 1);
}

void MPCSolver::set_last_cmd(const Eigen::Vector2d& cmd) {
    last_cmd_ = cmd;
}

void MPCSolver::reset_warm_start() {
    follow_warm_ = false;
    stop_warm_ = false;
    hold_warm_ = false;
    last_u_ = 0.0;
    fddp_lethal_consecutive_count_ = 0;
    for (size_t k = 0; k < MPC_HORIZON; ++k) {
        follow_solver_.us[k].setZero();
        stop_solver_.us[k].setZero();
        hold_solver_.us[k].setZero();
    }
}

void MPCSolver::set_energy_state(double remaining_energy, double rfr_pwr_limit) {
    remaining_energy_ = remaining_energy;
    rfr_pwr_limit_ = rfr_pwr_limit;
}

void MPCSolver::update_observer(const ChassisMotionState& chassis_state) {
    const double v_act = chassis_state.velocity;
    const double w_act = chassis_state.omega;
    const double rho_cur = schedule_rho_from_state(chassis_state, params_.kinematic_model);
    if (!observer_initialized_) {
        x_h_hat_ = params_.kinematic_model.xh0_bias + params_.kinematic_model.xh0_psi * chassis_state.leg_psi
            + params_.kinematic_model.xh0_v * v_act;
        prev_v_act_ = v_act;
        prev_w_act_ = w_act;
        prev_schedule_rho_ = rho_cur;
        observer_initialized_ = true;
        return;
    }
    const auto model = build_lpv_discrete_model(params_.kinematic_model, 0.5 * (prev_schedule_rho_ + rho_cur));
    const auto nl_eval = evaluate_lpv_nonlinear(prev_v_act_, prev_w_act_, model);
    const double xh_pred = model.ad00 * x_h_hat_ + model.ad01 * prev_v_act_ + model.bd0 * last_cmd_.x() + model.gd0 * nl_eval.nl;
    const double v_pred = model.ad10 * x_h_hat_ + model.ad11 * prev_v_act_ + model.bd1 * last_cmd_.x() + model.gd1 * nl_eval.nl;
    const double psi_proxy_pred = params_.kinematic_model.psi_bias + params_.kinematic_model.psi_gain * xh_pred
        + params_.kinematic_model.psi_v * v_pred;
    x_h_hat_ = xh_pred + params_.kinematic_model.obs_lv * (v_act - v_pred)
        + params_.kinematic_model.obs_lpsi * (chassis_state.leg_psi - psi_proxy_pred);
    prev_v_act_ = v_act;
    prev_w_act_ = w_act;
    prev_schedule_rho_ = rho_cur;
}

void MPCSolver::reset_observer() {
    x_h_hat_ = 0.0;
    prev_v_act_ = 0.0;
    prev_w_act_ = 0.0;
    observer_initialized_ = false;
}

StateVec MPCSolver::make_initial_state(
    const Eigen::Vector3d& pose,
    const ChassisMotionState& chassis_state,
    const Eigen::Vector2d& cmd_clamped,
    double path_u
) const {
    StateVec x0;
    x0(ix::X) = pose.x();
    x0(ix::Y) = pose.y();
    x0(ix::THETA) = pose.z();
    x0(ix::XH) = x_h_hat_;
    x0(ix::V) = chassis_state.velocity;
    x0(ix::W) = chassis_state.omega;
    x0(ix::DV) = cmd_clamped.x();
    x0(ix::DW) = cmd_clamped.y();
    x0(ix::PATH_U) = path_u;
    x0(ix::ENERGY) = remaining_energy_;
    return x0;
}

std::expected<MPCSolver::FollowSolveResult, std::string> MPCSolver::solve_follow(
    const SplinePath& global_path,
    const Eigen::Vector3d& chassis_pose_map,
    const ChassisMotionState& chassis_state,
    const CostMap& cost_map,
    const CostMap& masked_global_map,
    const std::vector<const CostMap*>& per_step_cost_maps,
    double prediction_dt,
    const DirectionMap& direction_map,
    const CapabilityProfile& blended_profile,
    std::optional<ActiveStepMode> active_step_mode
) {
    const bool path_changed = !(prev_ref_control_points_ && *prev_ref_control_points_ == global_path);
    const double projection_hint = path_changed ? 0.0 : std::clamp(last_u_, 0.0, 1.0);
    const double u0 = global_path.project_extrapolated(
        chassis_pose_map.head<2>(),
        projection_hint,
        params_.follow.projection.proj_num_samples,
        params_.follow.projection.proj_search_window,
        params_.follow.projection.local_search_lazy_distance
    );
    if (path_changed) {
        follow_warm_ = false;
    }
    last_u_ = u0;

    const Eigen::Vector2d cmd0(
        clamp_prev_cmd(
            last_cmd_.x(),
            chassis_state.velocity,
            params_.follow.start_command.vel_cmd_act_gap_max,
            blended_profile.motion_constraints.acc_max,
            MPC_DT
        ),
        clamp_prev_cmd(
            last_cmd_.y(),
            chassis_state.omega,
            params_.follow.start_command.omega_cmd_act_gap_max,
            blended_profile.motion_constraints.alpha_max,
            MPC_DT
        )
    );
    const double schedule_rho = select_follow_schedule_rho(
        chassis_state,
        params_.kinematic_model
    );

    prev_ref_control_points_ = global_path;

    step_cost_grids_cache_.clear();
    if (per_step_cost_maps.empty()) {
        step_cost_grids_cache_.reserve(1);
    } else {
        step_cost_grids_cache_.reserve(std::max(step_cost_grids_cache_.capacity(), per_step_cost_maps.size()));
    }
    if (per_step_cost_maps.empty()) {
        step_cost_grids_cache_.emplace_back(cost_map);
    } else {
        for (const auto* cm : per_step_cost_maps) {
            step_cost_grids_cache_.emplace_back(*cm);
        }
    }
    const double pred_dt = per_step_cost_maps.empty() ? MPC_DT : prediction_dt;

    const GridInfo ci = make_grid_info(cost_map);
    const CostMapGridView masked_global_grid(masked_global_map);
    const DirectionMapGridView dg(direction_map);
    const GridInfo di = make_grid_info(direction_map);
    const StateVec x0 = make_initial_state(chassis_pose_map, chassis_state, cmd0, u0);

    const FollowProblem problem(
        global_path, params_, step_cost_grids_cache_, ci, masked_global_grid, pred_dt, schedule_rho,
        dg, di, remaining_energy_, rfr_pwr_limit_, blended_profile, active_step_mode, u0
    );

    fddp::SolverOptions opts;
    opts.max_iters = params_.follow.max_iters;
    opts.tol_grad = SOLVER_TOL_GRAD;
    opts.tol_cost = SOLVER_TOL_COST;

    if (follow_warm_) {
        shift_warm_start(follow_solver_);
    } else {
        fill_solver_controls(follow_solver_, cmd0);
    }
    scale_solver_controls(follow_solver_, problem);

    MPPIFollowSamplingResult mppi_result;
    bool seeded_by_mppi = false;
    if (params_.follow.mppi.enable) {
        MPPIFollowSampler sampler(params_.follow.mppi);
        mppi_result = sampler.optimize(problem, x0, copy_solver_controls(follow_solver_));
        if (mppi_result.valid) {
            seed_solver_from_sampling_result(follow_solver_, mppi_result);
            seeded_by_mppi = true;
        }
    }

    if (!seeded_by_mppi) {
        rollout_solver_states(follow_solver_, problem, x0);
    }
    follow_solver_.solve(problem, opts);
    follow_warm_ = true;

    const auto solved_rollout = rollout_states(problem, follow_solver_, x0);
    MPCPrediction prediction;
    {
        const size_t sz = solved_rollout.valid_steps + 1;
        prediction.path_map.reserve(sz);
        prediction.headings.reserve(sz);
        prediction.v_pred.reserve(sz);
        prediction.w_pred.reserve(sz);
        for (size_t i = 0; i <= solved_rollout.valid_steps; ++i) {
            const auto& x = solved_rollout.xs[i];
            prediction.path_map.emplace_back(x(ix::X), x(ix::Y));
            prediction.headings.push_back(x(ix::THETA));
            prediction.v_pred.push_back(x(ix::V));
            prediction.w_pred.push_back(x(ix::W));
        }
    }
    prediction.rollout_paths = std::move(mppi_result.rollout_paths);

    {
        const auto& safety = params_.follow.rollout_safety;
        const auto lethal = detect_rollout_lethal_obstacle(
            problem, solved_rollout.xs, solved_rollout.valid_steps + 1
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
    }

    const Eigen::Vector2d cmd(follow_solver_.us[0](0), follow_solver_.us[0](1));
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
    const Eigen::Vector2d cmd0(
        clamp_prev_cmd(
            last_cmd_.x(),
            chassis_state.velocity,
            params_.stop.start_command.vel_cmd_act_gap_max,
            params_.stop.motion_constraints.acc_max,
            MPC_DT
        ),
        clamp_prev_cmd(
            last_cmd_.y(),
            chassis_state.omega,
            params_.stop.start_command.omega_cmd_act_gap_max,
            params_.stop.motion_constraints.alpha_max,
            MPC_DT
        )
    );
    const double schedule_rho = schedule_rho_from_state(chassis_state, params_.kinematic_model);

    const CostMapGridView cg(cost_map);
    const GridInfo ci = make_grid_info(cost_map);
    const StateVec x0 = make_initial_state(chassis_pose_map, chassis_state, cmd0, 0.0);

    StopProblem prob(params_, cg, ci, schedule_rho, remaining_energy_, rfr_pwr_limit_);
    if (stop_warm_) {
        shift_warm_start(stop_solver_);
    } else {
        fill_solver_controls(stop_solver_, cmd0);
    }
    scale_solver_controls(stop_solver_, prob);
    rollout_solver_states(stop_solver_, prob, x0);

    fddp::SolverOptions opts;
    opts.max_iters = params_.stop.max_iters;
    opts.tol_grad = SOLVER_TOL_GRAD;
    opts.tol_cost = SOLVER_TOL_COST;
    stop_solver_.solve(prob, opts);
    stop_warm_ = true;

    const Eigen::Vector2d cmd(stop_solver_.us[0](0), stop_solver_.us[0](1));
    last_cmd_ = cmd;
    return std::tuple {cmd, rollout_prediction(prob, stop_solver_, x0)};
}

std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> MPCSolver::solve_hold(
    const Eigen::Vector2d& goal_map,
    const Eigen::Vector3d& chassis_pose_map,
    const ChassisMotionState& chassis_state,
    const CostMap& cost_map
) {
    const Eigen::Vector2d cmd0(
        clamp_prev_cmd(
            last_cmd_.x(),
            chassis_state.velocity,
            params_.hold.start_command.vel_cmd_act_gap_max,
            params_.hold.motion_constraints.acc_max,
            MPC_DT
        ),
        clamp_prev_cmd(
            last_cmd_.y(),
            chassis_state.omega,
            params_.hold.start_command.omega_cmd_act_gap_max,
            params_.hold.motion_constraints.alpha_max,
            MPC_DT
        )
    );
    const double schedule_rho = schedule_rho_from_state(chassis_state, params_.kinematic_model);

    const CostMapGridView cg(cost_map);
    const GridInfo ci = make_grid_info(cost_map);
    const StateVec x0 = make_initial_state(chassis_pose_map, chassis_state, cmd0, 0.0);

    HoldProblem prob(goal_map, params_, cg, ci, schedule_rho, remaining_energy_, rfr_pwr_limit_);
    if (hold_warm_) {
        shift_warm_start(hold_solver_);
    } else {
        fill_solver_controls(hold_solver_, cmd0);
    }
    scale_solver_controls(hold_solver_, prob);
    rollout_solver_states(hold_solver_, prob, x0);

    fddp::SolverOptions opts;
    opts.max_iters = params_.hold.max_iters;
    opts.tol_grad = SOLVER_TOL_GRAD;
    opts.tol_cost = SOLVER_TOL_COST;
    hold_solver_.solve(prob, opts);
    hold_warm_ = true;

    const Eigen::Vector2d cmd(hold_solver_.us[0](0), hold_solver_.us[0](1));
    last_cmd_ = cmd;
    return std::tuple {cmd, rollout_prediction(prob, hold_solver_, x0)};
}

} // namespace nav_executor
