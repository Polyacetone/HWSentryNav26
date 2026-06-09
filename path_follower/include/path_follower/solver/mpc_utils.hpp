#pragma once

#include <cmath>
#include <path_follower/solver/mpc_types.hpp>

namespace path_follower {

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

inline double smooth_abs(double x) {
    constexpr double EPS = 0.05;
    return std::sqrt(x * x + EPS * EPS);
}

inline double smooth_abs_derivative(double x) {
    return x / smooth_abs(x);
}

inline double clamp_derivative_piecewise(double x, double lo, double hi) {
    return (x > lo && x < hi) ? 1.0 : 0.0;
}

inline double wrap_pi(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

inline double brake_speed_limit(double remaining_s, const MPCFollowTerminalWeights& w) {
    return std::sqrt(std::max(0.0, 2.0 * w.a_brake * remaining_s) + w.slow_down_target_vel * w.slow_down_target_vel);
}

inline double energy_hinge_cost(const EnergyParams& energy, double remaining_energy) {
    if (!energy.enable) {
        return 0.0;
    }
    return energy.weight * positive_part(energy.threshold - remaining_energy);
}

inline void add_energy_hinge_gradient(const EnergyParams& energy, const StateVec& x, StateVec& lx) {
    if (energy.enable && x(ix::ENERGY) < energy.threshold) {
        lx(ix::ENERGY) -= energy.weight;
    }
}

inline double clamp_lpv_rho(double rho, double rho_clip) {
    return std::clamp(rho, -rho_clip, rho_clip);
}

inline bool is_active_follow_step_mode(std::optional<ActiveStepMode> active_step_mode) {
    return active_step_mode.has_value() && active_step_mode->mode != chassis_mode::NORMAL;
}

inline double lpv_schedule_z(double leg_h, double leg_psi) {
    return leg_h * std::cos(leg_psi);
}

inline double clamp_prev_cmd(double cmd_prev, double status, double cmd_act_diff_max, double rate_max, double dt) {
    return std::max(
        std::min(cmd_prev, status + cmd_act_diff_max - rate_max * dt),
        status - cmd_act_diff_max + rate_max * dt
    );
}

inline double predict_power(const PowerModelParams& power_model, double v, double w, double a, double alpha) {
    const auto& c = power_model.coeffs;
    return c[0] + c[1] * v * a + c[2] * w * alpha + c[3] * a * a + c[4] * alpha * alpha
        + c[5] * smooth_abs(v) + c[6] * smooth_abs(w) + c[7] * v * v + c[8] * w * w + c[9] * smooth_abs(a)
        + c[10] * smooth_abs(alpha) + c[11] * smooth_abs(v * w);
}

struct PowerEval {
    double value;
    double dv;
    double dw;
    double da;
    double dalpha;
};

inline PowerEval predict_power_eval(const PowerModelParams& power_model, double v, double w, double a, double alpha) {
    const auto& c = power_model.coeffs;
    const double vw = v * w;
    const double ds_v = smooth_abs_derivative(v);
    const double ds_w = smooth_abs_derivative(w);
    const double ds_a = smooth_abs_derivative(a);
    const double ds_alpha = smooth_abs_derivative(alpha);
    const double ds_vw = smooth_abs_derivative(vw);

    PowerEval out {};
    out.value = predict_power(power_model, v, w, a, alpha);
    out.dv = c[1] * a + c[5] * ds_v + 2.0 * c[7] * v + c[11] * ds_vw * w;
    out.dw = c[2] * alpha + c[6] * ds_w + 2.0 * c[8] * w + c[11] * ds_vw * v;
    out.da = c[1] * v + 2.0 * c[3] * a + c[9] * ds_a;
    out.dalpha = c[2] * w + 2.0 * c[4] * alpha + c[10] * ds_alpha;
    return out;
}

inline void apply_capacitor_energy_dynamics(
    StateVec& xn,
    const StateVec& x,
    const PowerModelParams& power_model,
    double charge_power
) {
    const double a = (xn(ix::V) - x(ix::V)) / MPC_DT;
    const double alpha = (xn(ix::W) - x(ix::W)) / MPC_DT;
    const auto pwr = predict_power_eval(power_model, xn(ix::V), xn(ix::W), a, alpha);
    xn(ix::ENERGY) = x(ix::ENERGY) + (charge_power - pwr.value) * MPC_DT;
}

inline void apply_capacitor_energy_jacobian(
    MatXX& fx,
    MatXU& fu,
    const StateVec& x,
    const StateVec& xn,
    const PowerModelParams& power_model
) {
    const double a = (xn(ix::V) - x(ix::V)) / MPC_DT;
    const double alpha = (xn(ix::W) - x(ix::W)) / MPC_DT;
    const auto pwr = predict_power_eval(power_model, xn(ix::V), xn(ix::W), a, alpha);

    const Eigen::Matrix<double, 1, MPC_NX> dv_next_dx = fx.row(ix::V);
    const Eigen::Matrix<double, 1, MPC_NX> dw_next_dx = fx.row(ix::W);
    const Eigen::Matrix<double, 1, MPC_NU> dv_next_du = fu.row(ix::V);
    const Eigen::Matrix<double, 1, MPC_NU> dw_next_du = fu.row(ix::W);
    Eigen::Matrix<double, 1, MPC_NX> da_dx = dv_next_dx;
    Eigen::Matrix<double, 1, MPC_NX> dalpha_dx = dw_next_dx;
    da_dx(ix::V) -= 1.0;
    dalpha_dx(ix::W) -= 1.0;
    da_dx /= MPC_DT;
    dalpha_dx /= MPC_DT;
    const Eigen::Matrix<double, 1, MPC_NU> da_du = dv_next_du / MPC_DT;
    const Eigen::Matrix<double, 1, MPC_NU> dalpha_du = dw_next_du / MPC_DT;

    const Eigen::Matrix<double, 1, MPC_NX> dp_dx = pwr.dv * dv_next_dx + pwr.dw * dw_next_dx
        + pwr.da * da_dx + pwr.dalpha * dalpha_dx;
    const Eigen::Matrix<double, 1, MPC_NU> dp_du = pwr.dv * dv_next_du + pwr.dw * dw_next_du
        + pwr.da * da_du + pwr.dalpha * dalpha_du;

    fx.row(ix::ENERGY).setZero();
    fu.row(ix::ENERGY).setZero();
    fx(ix::ENERGY, ix::ENERGY) = 1.0;
    fx.row(ix::ENERGY) -= MPC_DT * dp_dx;
    fu.row(ix::ENERGY) -= MPC_DT * dp_du;
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

} // namespace path_follower
