#pragma once

#include <cmath>
#include <nav_executor/path_executor/solver/mpc_types.hpp>

namespace nav_executor {

// ─── 基本数学工具 ───

inline double positive_part(double x) {
    return std::max(x, 0.0);
}

inline double positive_part_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

inline double sign_or_zero(double x) {
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 0.0;
}

inline double clamp_derivative_piecewise(double x, double lo, double hi) {
    return (x > lo && x < hi) ? 1.0 : 0.0;
}

inline double wrap_pi(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

inline double clamp_lpv_rho(double rho, double rho_clip) {
    return std::clamp(rho, -rho_clip, rho_clip);
}

// 约束窗软门控：沿 u 的梯形窗，在 [gate_start_u, commit_u] 平滑 0→1，
// 在 [commit_u, exit_u] 恒 1，在 [exit_u, gate_end_u] 平滑 1→0。
//
// 门控值随预测 path_u 逐步取值（因此约束跟随轨迹前移），但其对 path_u 的梯度被刻意
// 冻结（不并入雅可比）：门控只调度「约束是否施加」，速度窗/航向对齐的梯度分别落在
// V / THETA 上，避免优化器为逃出门控而移动 path_u 造成的病态与不连续。
inline double step_window_gate(double uc, const StepTraversalConstraint& m) {
    if (uc <= m.gate_start_u || uc >= m.gate_end_u) return 0.0;
    if (uc >= m.commit_u && uc <= m.exit_u) return 1.0;
    auto smoothstep = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };
    if (uc < m.commit_u) {
        const double w = m.commit_u - m.gate_start_u;
        return w > 1e-9 ? smoothstep((uc - m.gate_start_u) / w) : 1.0;
    }
    const double w = m.gate_end_u - m.exit_u;
    return w > 1e-9 ? smoothstep((m.gate_end_u - uc) / w) : 1.0;
}

inline double lpv_schedule_z(double leg_h, double leg_psi) {
    return leg_h * std::cos(leg_psi);
}

inline Eigen::Vector2d command_after_control(const StateVec& x, const ControlVec& u) {
    return {
        x(ix::V_CMD) + MPC_DT * u(iu::V_CMD_RATE),
        x(ix::W_CMD) + MPC_DT * u(iu::W_CMD_RATE),
    };
}

// ─── 中等大小工具函数（定义在 mpc_utils.cpp）───

double smooth_sgn(double x, double eps);
double smooth_sgn_deriv(double x, double eps);

struct LPVNonlinearEval {
    double nl;
    double dnl_dv;
    double dnl_dw;
    double sw;
    double dsw_dw;
};

LPVNonlinearEval evaluate_lpv_nonlinear(double v, double w, const LPVDiscreteModel& model);

Eigen::Vector2d apply_goal_deadzone(const Eigen::Vector2d& delta, double deadzone);

void zoh_v_matrices(
    double a00, double a01, double a10, double a11,
    double b0, double b1, double g0, double g1,
    double dt,
    double& ad00, double& ad01, double& ad10, double& ad11,
    double& bd0, double& bd1, double& gd0, double& gd1
);

LPVDiscreteModel build_lpv_discrete_model(const LPVKinematicModelParams& params, double rho);

MPCControlBounds command_rate_control_bounds(
    const StateVec& x,
    const CapabilityProfile& capability,
    double path_speed_min,
    double path_speed_max,
    const MPCStartCommandLimits* start_command = nullptr
);

double schedule_rho_from_state(const ChassisMotionState& chassis_state, const LPVKinematicModelParams& model_params);
double select_follow_schedule_rho(const ChassisMotionState& chassis_state, const LPVKinematicModelParams& model_params);

// ─── Gauss-Newton 模板 ───

template<typename ResidualVec>
inline double residual_cost(const ResidualVec& r) {
    return 0.5 * r.squaredNorm();
}

template<int NR, typename ResidualFn>
void gauss_newton_running_derivatives(
    ResidualFn&& residual_fn,
    const StateVec& x,
    const ControlVec& u,
    StateVec& lx,
    ControlVec& lu,
    MatXX& lxx,
    Eigen::Matrix<double, MPC_NU, MPC_NX>& lux,
    Eigen::Matrix<double, MPC_NU, MPC_NU>& luu
) {
    using ResidualVec = Eigen::Matrix<double, NR, 1>;
    constexpr double EPS_REL = 1e-6;
    constexpr double EPS_ABS = 1e-4;

    const ResidualVec r0 = residual_fn(x, u);
    Eigen::Matrix<double, NR, MPC_NX> jx;
    Eigen::Matrix<double, NR, MPC_NU> ju;

    for (int i = 0; i < MPC_NX; ++i) {
        const double eps = std::max(EPS_REL * std::abs(x(i)), EPS_ABS);
        StateVec xp = x;
        xp(i) += eps;
        StateVec xm = x;
        xm(i) -= eps;
        jx.col(i) = (residual_fn(xp, u) - residual_fn(xm, u)) / (2.0 * eps);
    }
    for (int i = 0; i < MPC_NU; ++i) {
        const double eps = std::max(EPS_REL * std::abs(u(i)), EPS_ABS);
        ControlVec up = u;
        up(i) += eps;
        ControlVec um = u;
        um(i) -= eps;
        ju.col(i) = (residual_fn(x, up) - residual_fn(x, um)) / (2.0 * eps);
    }

    lx = jx.transpose() * r0;
    lu = ju.transpose() * r0;
    lxx = (jx.transpose() * jx).eval();
    lux = (ju.transpose() * jx).eval();
    luu = (ju.transpose() * ju).eval();
    lxx = (lxx + lxx.transpose()).eval() * 0.5;
    luu = (luu + luu.transpose()).eval() * 0.5;
    for (int i = 0; i < MPC_NU; ++i) {
        luu(i, i) = std::max(luu(i, i), 1e-8);
    }
}

template<int NR, typename ResidualFn>
void gauss_newton_terminal_derivatives(ResidualFn&& residual_fn, const StateVec& x, StateVec& lx, MatXX& lxx) {
    using ResidualVec = Eigen::Matrix<double, NR, 1>;
    constexpr double EPS_REL = 1e-6;
    constexpr double EPS_ABS = 1e-4;

    const ResidualVec r0 = residual_fn(x);
    Eigen::Matrix<double, NR, MPC_NX> jx;

    for (int i = 0; i < MPC_NX; ++i) {
        const double eps = std::max(EPS_REL * std::abs(x(i)), EPS_ABS);
        StateVec xp = x;
        xp(i) += eps;
        StateVec xm = x;
        xm(i) -= eps;
        jx.col(i) = (residual_fn(xp) - residual_fn(xm)) / (2.0 * eps);
    }

    lx = jx.transpose() * r0;
    lxx = (jx.transpose() * jx).eval();
    lxx = (lxx + lxx.transpose()).eval() * 0.5;
}

} // namespace nav_executor
