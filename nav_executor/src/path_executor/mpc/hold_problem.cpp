#include <nav_executor/path_executor/mpc/hold_problem.hpp>
#include <nav_executor/path_executor/mpc/mpc_utils.hpp>

namespace nav_executor {

HoldProblem::HoldProblem(
    const Eigen::Vector2d& goal_map,
    const MPCParams& params,
    const CostMap& cost_map,
    double schedule_rho
):
    goal_(goal_map),
    p_(params),
    cost_map_(cost_map),
    model_(build_lpv_discrete_model(params.kinematic_model, schedule_rho)) {}

StateVec HoldProblem::dynamics(int, const StateVec& x, const ControlVec& u) const {
    return mpc_dynamics(x, u, model_);
}

void HoldProblem::dynamics_jacobians(int, const StateVec& x, const ControlVec& u, MatXX& dfx, MatXU& dfu) const {
    mpc_dynamics_jacobians(x, u, model_, dfx, dfu);
}

MPCControlBounds HoldProblem::control_bounds(const int k, const StateVec& x) const {
    return command_rate_control_bounds(
        x, p_.hold.profile, k == 0 ? &p_.hold.start_command : nullptr
    );
}

constexpr int HOLD_RESIDUAL_DIM = 11;
using HoldResidualVec = Eigen::Matrix<double, HOLD_RESIDUAL_DIM, 1>;

HoldResidualVec hold_residual_impl(
    const StateVec& x,
    const ControlVec& u,
    const Eigen::Vector2d& goal,
    const MPCParams& p,
    const CostMap& cost_map
) {
    const auto& hold = p.hold;
    const auto& goal_w = hold.goal_weights;
    const auto& command_w = hold.command_weights;
    const auto& dynamics_w = hold.command_dynamics_weights;
    const auto& env_w = hold.environment_weights;
    const auto& motion_lim = hold.profile.command_dynamics;

    HoldResidualVec r = HoldResidualVec::Zero();
    const double px = x(ix::X), py = x(ix::Y), theta = x(ix::THETA);
    const Eigen::Vector2d next_command = command_after_control(x, u);
    const double v_cmd = next_command.x(), w_cmd = next_command.y();
    const double dv_cmd = MPC_DT * u(iu::V_CMD_RATE);
    const double dw_cmd = MPC_DT * u(iu::W_CMD_RATE);

    const Eigen::Vector2d goal_delta = apply_goal_deadzone(
        Eigen::Vector2d(px - goal.x(), py - goal.y()),
        goal_w.goal_deadzone
    );
    const double ddx = goal_delta.x(), ddy = goal_delta.y();
    const double desired_theta = std::atan2(goal.y() - py, goal.x() - px);
    const double heading_sin = std::sin(theta - desired_theta);

    const auto cs = cost_map.sample_map({px, py}).value_or(CostMap::CostSample {
        .value = 255.0,
        .gradient = Eigen::Vector2d::Zero(),
    });

    const double a_lat = std::abs(v_cmd * w_cmd);
    r(0) = goal_w.q_goal_xy * ddx;
    r(1) = goal_w.q_goal_xy * ddy;
    r(2) = goal_w.q_goal_theta * std::abs(heading_sin);
    r(3) = command_w.r_v * v_cmd;
    r(4) = command_w.r_omega * w_cmd;
    r(5) = command_w.r_dv * dv_cmd;
    r(6) = command_w.r_domega * dw_cmd;
    r(7) = command_w.r_jerk_v
        * (u(iu::V_CMD_RATE) - x(ix::V_CMD_RATE)) / MPC_DT;
    r(8) = command_w.r_jerk_omega
        * (u(iu::W_CMD_RATE) - x(ix::W_CMD_RATE)) / MPC_DT;
    r(9) = dynamics_w.lateral_acceleration
        * positive_part(a_lat - motion_lim.lateral_acceleration_max);
    r(10) = env_w.obstacle * cs.value / 255.0;

    return r;
}

constexpr int HOLD_TERMINAL_RESIDUAL_DIM = 3;
using HoldTerminalResidualVec = Eigen::Matrix<double, HOLD_TERMINAL_RESIDUAL_DIM, 1>;

HoldTerminalResidualVec hold_terminal_residual_impl(
    const StateVec& x,
    const Eigen::Vector2d& goal,
    const MPCParams& p,
    const CostMap& cost_map
) {
    HoldTerminalResidualVec r = HoldTerminalResidualVec::Zero();
    const auto& goal_w = p.hold.goal_weights;
    const auto& terminal_w = p.hold.terminal_weights;
    const Eigen::Vector2d terminal_delta = apply_goal_deadzone(
        Eigen::Vector2d(x(ix::X) - goal.x(), x(ix::Y) - goal.y()),
        goal_w.goal_deadzone
    );
    const auto cs = cost_map.sample_map({x(ix::X), x(ix::Y)}).value_or(
        CostMap::CostSample {.value = 255.0, .gradient = Eigen::Vector2d::Zero()}
    );
    r(0) = terminal_w.q_goal_xy_terminal * terminal_delta.x();
    r(1) = terminal_w.q_goal_xy_terminal * terminal_delta.y();
    r(2) = terminal_w.obstacle_terminal * cs.value / 255.0;
    return r;
}

double HoldProblem::running_cost(int k, const StateVec& x, const ControlVec& u) const {
    (void)k;
    return residual_cost(hold_residual_impl(x, u, goal_, p_, cost_map_));
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
        return hold_residual_impl(xv, uv, goal_, p_, cost_map_);
    };
    gauss_newton_running_derivatives<HOLD_RESIDUAL_DIM>(residual_fn, x, u, lx, lu, lxx, lux, luu);
}

double HoldProblem::terminal_cost(const StateVec& x) const {
    return residual_cost(hold_terminal_residual_impl(x, goal_, p_, cost_map_));
}

void HoldProblem::terminal_cost_derivatives(const StateVec& x, StateVec& lfx, MatXX& lfxx) const {
    auto residual_fn = [&](const StateVec& xv) {
        return hold_terminal_residual_impl(xv, goal_, p_, cost_map_);
    };
    gauss_newton_terminal_derivatives<HOLD_TERMINAL_RESIDUAL_DIM>(residual_fn, x, lfx, lfxx);
}

} // namespace nav_executor
