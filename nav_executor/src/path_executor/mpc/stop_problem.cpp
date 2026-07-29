#include <nav_executor/path_executor/mpc/stop_problem.hpp>
#include <nav_executor/path_executor/mpc/mpc_utils.hpp>

namespace nav_executor {

StopProblem::StopProblem(
    const MPCParams& params,
    const CostMap& cost_map,
    double schedule_rho
):
    p_(params),
    cost_map_(cost_map),
    model_(build_lpv_discrete_model(params.kinematic_model, schedule_rho)) {}

StateVec StopProblem::dynamics(int, const StateVec& x, const ControlVec& u) const {
    return mpc_dynamics(x, u, model_);
}

void StopProblem::dynamics_jacobians(int, const StateVec& x, const ControlVec& u, MatXX& dfx, MatXU& dfu) const {
    mpc_dynamics_jacobians(x, u, model_, dfx, dfu);
}

MPCControlBounds StopProblem::control_bounds(const int k, const StateVec& x) const {
    return command_rate_control_bounds(
        x, p_.stop.profile, k == 0 ? &p_.stop.start_command : nullptr
    );
}

constexpr int STOP_RESIDUAL_DIM = 8;
using StopResidualVec = Eigen::Matrix<double, STOP_RESIDUAL_DIM, 1>;

StopResidualVec stop_residual_impl(
    const StateVec& x,
    const ControlVec& u,
    const MPCParams& p,
    const CostMap& cost_map
) {
    const auto& stop = p.stop;
    const auto& motion_lim = stop.profile.command_dynamics;
    const auto& command_w = stop.command_weights;
    const auto& dynamics_w = stop.command_dynamics_weights;
    const auto& env_w = stop.environment_weights;

    StopResidualVec r = StopResidualVec::Zero();
    const double px = x(ix::X), py = x(ix::Y);
    const Eigen::Vector2d next_command = command_after_control(x, u);
    const double v_cmd = next_command.x(), w_cmd = next_command.y();
    const double dv_cmd = MPC_DT * u(iu::V_CMD_RATE);
    const double dw_cmd = MPC_DT * u(iu::W_CMD_RATE);

    const auto cs = cost_map.sample_map({px, py}).value_or(CostMap::CostSample {
        .value = 255.0,
        .gradient = Eigen::Vector2d::Zero(),
    });

    const double a_lat = std::abs(v_cmd * w_cmd);
    r(0) = command_w.r_v * v_cmd;
    r(1) = command_w.r_omega * w_cmd;
    r(2) = command_w.r_dv * dv_cmd;
    r(3) = command_w.r_domega * dw_cmd;
    r(4) = command_w.r_jerk_v
        * (u(iu::V_CMD_RATE) - x(ix::V_CMD_RATE)) / MPC_DT;
    r(5) = command_w.r_jerk_omega
        * (u(iu::W_CMD_RATE) - x(ix::W_CMD_RATE)) / MPC_DT;
    r(6) = dynamics_w.lateral_acceleration
        * positive_part(a_lat - motion_lim.lateral_acceleration_max);
    r(7) = env_w.obstacle * cs.value / 255.0;

    return r;
}

constexpr int STOP_TERMINAL_RESIDUAL_DIM = 1;
using StopTerminalResidualVec = Eigen::Matrix<double, STOP_TERMINAL_RESIDUAL_DIM, 1>;

StopTerminalResidualVec stop_terminal_residual_impl(
    const StateVec& x,
    const MPCParams& p,
    const CostMap& cost_map
) {
    StopTerminalResidualVec r = StopTerminalResidualVec::Zero();
    const auto& w = p.stop.terminal_weights;
    const auto cs = cost_map.sample_map({x(ix::X), x(ix::Y)}).value_or(
        CostMap::CostSample {.value = 255.0, .gradient = Eigen::Vector2d::Zero()}
    );
    r(0) = w.obstacle_terminal * cs.value / 255.0;
    return r;
}

double StopProblem::running_cost(int k, const StateVec& x, const ControlVec& u) const {
    (void)k;
    return residual_cost(stop_residual_impl(x, u, p_, cost_map_));
}

void StopProblem::running_cost_derivatives(
    int k,
    const StateVec& x,
    const ControlVec& u,
    StateVec& lx,
    ControlVec& lu,
    MatXX& lxx,
    Eigen::Matrix<double, MPC_NU, MPC_NX>& lux,
    Eigen::Matrix<double, MPC_NU, MPC_NU>& luu
) const {
    (void)k;
    auto residual_fn = [&](const StateVec& xv, const ControlVec& uv) {
        return stop_residual_impl(xv, uv, p_, cost_map_);
    };
    gauss_newton_running_derivatives<STOP_RESIDUAL_DIM>(residual_fn, x, u, lx, lu, lxx, lux, luu);
}

double StopProblem::terminal_cost(const StateVec& x) const {
    return residual_cost(stop_terminal_residual_impl(x, p_, cost_map_));
}

void StopProblem::terminal_cost_derivatives(const StateVec& x, StateVec& lfx, MatXX& lfxx) const {
    auto residual_fn = [&](const StateVec& xv) {
        return stop_terminal_residual_impl(xv, p_, cost_map_);
    };
    gauss_newton_terminal_derivatives<STOP_TERMINAL_RESIDUAL_DIM>(residual_fn, x, lfx, lfxx);
}

} // namespace nav_executor
