#include <nav_executor/path_executor/solver/hold_problem.hpp>
#include <nav_executor/path_executor/solver/mpc_utils.hpp>
#include <nav_executor/path_executor/solver/bilinear_sampling.hpp>

namespace nav_executor {

HoldProblem::HoldProblem(
    const Eigen::Vector2d& goal_map,
    const MPCParams& params,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info,
    double schedule_rho
):
    goal_(goal_map),
    p_(params),
    cost_grid_(cost_grid),
    cost_info_(cost_info),
    model_(build_lpv_discrete_model(params.kinematic_model, schedule_rho)) {}

StateVec HoldProblem::dynamics(int, const StateVec& x, const ControlVec& u) const {
    return mpc_dynamics(x, u, model_);
}

void HoldProblem::dynamics_jacobians(int, const StateVec& x, const ControlVec& u, MatXX& dfx, MatXU& dfu) const {
    mpc_dynamics_jacobians(x, u, model_, dfx, dfu);
}

ControlVec HoldProblem::u_lower() const {
    return ControlVec(p_.hold.command_bounds.vel_min, p_.hold.command_bounds.omega_min);
}

ControlVec HoldProblem::u_upper() const {
    return ControlVec(p_.hold.command_bounds.vel_max, p_.hold.command_bounds.omega_max);
}

constexpr int HOLD_RESIDUAL_DIM = 11;
using HoldResidualVec = Eigen::Matrix<double, HOLD_RESIDUAL_DIM, 1>;

HoldResidualVec hold_residual_impl(
    const StateVec& x,
    const ControlVec& u,
    const Eigen::Vector2d& goal,
    const MPCParams& p,
    const CostMapGridView& cg,
    const GridInfo& ci
) {
    const auto& hold = p.hold;
    const auto& goal_w = hold.goal_weights;
    const auto& command_w = hold.command_weights;
    const auto& motion_w = hold.motion_constraint_weights;
    const auto& env_w = hold.environment_weights;
    const auto& motion_lim = hold.motion_constraints;

    HoldResidualVec r = HoldResidualVec::Zero();
    const double px = x(ix::X), py = x(ix::Y), theta = x(ix::THETA);
    const double v_cmd = u(0), w_cmd = u(1);
    const double dv_cmd = v_cmd - x(ix::DV);
    const double dw_cmd = w_cmd - x(ix::DW);

    const Eigen::Vector2d goal_delta = apply_goal_deadzone(
        Eigen::Vector2d(px - goal.x(), py - goal.y()),
        goal_w.goal_deadzone
    );
    const double ddx = goal_delta.x(), ddy = goal_delta.y();
    const double desired_theta = std::atan2(goal.y() - py, goal.x() - px);
    const double heading_sin = std::sin(theta - desired_theta);

    const auto cs = eval_cost_bilinear(cg, ci, px, py);

    const double dv_lim = motion_lim.acc_max * MPC_DT;
    const double dw_lim = motion_lim.alpha_max * MPC_DT;
    const double a_lat = std::abs(v_cmd * w_cmd);
    r(0) = goal_w.q_goal_xy * ddx;
    r(1) = goal_w.q_goal_xy * ddy;
    r(2) = goal_w.q_goal_theta * std::abs(heading_sin);
    r(3) = command_w.r_v * v_cmd;
    r(4) = command_w.r_omega * w_cmd;
    r(5) = command_w.r_dv * dv_cmd;
    r(6) = command_w.r_domega * dw_cmd;
    r(7) = motion_w.acc_limit * positive_part(std::abs(dv_cmd) - dv_lim);
    r(8) = motion_w.alpha_limit * positive_part(std::abs(dw_cmd) - dw_lim);
    r(9) = motion_w.lat_acc * positive_part(a_lat - motion_lim.a_lat_max);
    r(10) = env_w.obstacle * cs.value / 255.0;

    return r;
}

constexpr int HOLD_TERMINAL_RESIDUAL_DIM = 3;
using HoldTerminalResidualVec = Eigen::Matrix<double, HOLD_TERMINAL_RESIDUAL_DIM, 1>;

HoldTerminalResidualVec hold_terminal_residual_impl(
    const StateVec& x,
    const Eigen::Vector2d& goal,
    const MPCParams& p,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info
) {
    HoldTerminalResidualVec r = HoldTerminalResidualVec::Zero();
    const auto& goal_w = p.hold.goal_weights;
    const auto& terminal_w = p.hold.terminal_weights;
    const Eigen::Vector2d terminal_delta = apply_goal_deadzone(
        Eigen::Vector2d(x(ix::X) - goal.x(), x(ix::Y) - goal.y()),
        goal_w.goal_deadzone
    );
    const auto cs = eval_cost_bilinear(cost_grid, cost_info, x(ix::X), x(ix::Y));
    r(0) = terminal_w.q_goal_xy_terminal * terminal_delta.x();
    r(1) = terminal_w.q_goal_xy_terminal * terminal_delta.y();
    r(2) = terminal_w.obstacle_terminal * cs.value / 255.0;
    return r;
}

double HoldProblem::running_cost(int k, const StateVec& x, const ControlVec& u) const {
    (void)k;
    return residual_cost(hold_residual_impl(x, u, goal_, p_, cost_grid_, cost_info_));
}

void HoldProblem::running_cost_derivatives(
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
        return hold_residual_impl(xv, uv, goal_, p_, cost_grid_, cost_info_);
    };
    gauss_newton_running_derivatives<HOLD_RESIDUAL_DIM>(residual_fn, x, u, lx, lu, lxx, lux, luu);
}

double HoldProblem::terminal_cost(const StateVec& x) const {
    return residual_cost(hold_terminal_residual_impl(x, goal_, p_, cost_grid_, cost_info_));
}

void HoldProblem::terminal_cost_derivatives(const StateVec& x, StateVec& lfx, MatXX& lfxx) const {
    auto residual_fn = [&](const StateVec& xv) {
        return hold_terminal_residual_impl(xv, goal_, p_, cost_grid_, cost_info_);
    };
    gauss_newton_terminal_derivatives<HOLD_TERMINAL_RESIDUAL_DIM>(residual_fn, x, lfx, lfxx);
}

} // namespace nav_executor
