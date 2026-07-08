#include <nav_executor/solver/stop_problem.hpp>
#include <nav_executor/solver/mpc_utils.hpp>
#include <nav_executor/solver/bilinear_sampling.hpp>

namespace nav_executor {

StopProblem::StopProblem(
    const MPCParams& params,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info,
    double schedule_rho,
    double remaining_energy,
    double rfr_pwr_limit
):
    p_(params),
    cost_grid_(cost_grid),
    cost_info_(cost_info),
    model_(build_lpv_discrete_model(params.kinematic_model, schedule_rho)),
    remaining_energy_(remaining_energy),
    rfr_pwr_limit_(rfr_pwr_limit) {}

StateVec StopProblem::dynamics(int, const StateVec& x, const ControlVec& u) const {
    StateVec xn = mpc_dynamics(x, u, model_);
    apply_capacitor_energy_dynamics(xn, x, p_.power_model, rfr_pwr_limit_);
    return xn;
}

void StopProblem::dynamics_jacobians(int, const StateVec& x, const ControlVec& u, MatXX& dfx, MatXU& dfu) const {
    mpc_dynamics_jacobians(x, u, model_, dfx, dfu);
    const StateVec xn = mpc_dynamics(x, u, model_);
    apply_capacitor_energy_jacobian(dfx, dfu, x, xn, p_.power_model);
}

ControlVec StopProblem::u_lower() const {
    return ControlVec(p_.stop.command_bounds.vel_min, p_.stop.command_bounds.omega_min);
}

ControlVec StopProblem::u_upper() const {
    return ControlVec(p_.stop.command_bounds.vel_max, p_.stop.command_bounds.omega_max);
}

constexpr int STOP_RESIDUAL_DIM = 8;
using StopResidualVec = Eigen::Matrix<double, STOP_RESIDUAL_DIM, 1>;

StopResidualVec stop_residual_impl(
    const StateVec& x,
    const ControlVec& u,
    const MPCParams& p,
    const CostMapGridView& cg,
    const GridInfo& ci,
    double /*rfr_pwr_limit*/
) {
    const auto& stop = p.stop;
    const auto& motion_lim = stop.motion_constraints;
    const auto& command_w = stop.command_weights;
    const auto& motion_w = stop.motion_constraint_weights;
    const auto& env_w = stop.environment_weights;

    StopResidualVec r = StopResidualVec::Zero();
    const double px = x(ix::X), py = x(ix::Y);
    const double v_cmd = u(0), w_cmd = u(1);
    const double dv_cmd = v_cmd - x(ix::DV);
    const double dw_cmd = w_cmd - x(ix::DW);

    const auto cs = eval_cost_bilinear(cg, ci, px, py);

    const double dv_lim = motion_lim.acc_max * MPC_DT;
    const double dw_lim = motion_lim.alpha_max * MPC_DT;
    const double a_lat = std::abs(v_cmd * w_cmd);
    r(0) = command_w.r_v * v_cmd;
    r(1) = command_w.r_omega * w_cmd;
    r(2) = command_w.r_dv * dv_cmd;
    r(3) = command_w.r_domega * dw_cmd;
    r(4) = motion_w.acc_limit * positive_part(std::abs(dv_cmd) - dv_lim);
    r(5) = motion_w.alpha_limit * positive_part(std::abs(dw_cmd) - dw_lim);
    r(6) = motion_w.lat_acc * positive_part(a_lat - motion_lim.a_lat_max);
    r(7) = env_w.obstacle * cs.value / 255.0;

    return r;
}

constexpr int STOP_TERMINAL_RESIDUAL_DIM = 1;
using StopTerminalResidualVec = Eigen::Matrix<double, STOP_TERMINAL_RESIDUAL_DIM, 1>;

StopTerminalResidualVec stop_terminal_residual_impl(
    const StateVec& x,
    const MPCParams& p,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info
) {
    StopTerminalResidualVec r = StopTerminalResidualVec::Zero();
    const auto& w = p.stop.terminal_weights;
    const auto cs = eval_cost_bilinear(cost_grid, cost_info, x(ix::X), x(ix::Y));
    r(0) = w.obstacle_terminal * cs.value / 255.0;
    return r;
}

double StopProblem::running_cost(int k, const StateVec& x, const ControlVec& u) const {
    (void)k;
    return residual_cost(stop_residual_impl(x, u, p_, cost_grid_, cost_info_, rfr_pwr_limit_))
        + energy_hinge_cost(p_.energy, x(ix::ENERGY));
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
        return stop_residual_impl(xv, uv, p_, cost_grid_, cost_info_, rfr_pwr_limit_);
    };
    gauss_newton_running_derivatives<STOP_RESIDUAL_DIM>(residual_fn, x, u, lx, lu, lxx, lux, luu);
    add_energy_hinge_gradient(p_.energy, x, lx);
}

double StopProblem::terminal_cost(const StateVec& x) const {
    return residual_cost(stop_terminal_residual_impl(x, p_, cost_grid_, cost_info_));
}

void StopProblem::terminal_cost_derivatives(const StateVec& x, StateVec& lfx, MatXX& lfxx) const {
    auto residual_fn = [&](const StateVec& xv) {
        return stop_terminal_residual_impl(xv, p_, cost_grid_, cost_info_);
    };
    gauss_newton_terminal_derivatives<STOP_TERMINAL_RESIDUAL_DIM>(residual_fn, x, lfx, lfxx);
}

} // namespace nav_executor
