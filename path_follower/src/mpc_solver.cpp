#include <path_follower/mpc_solver.hpp>
#include <path_follower/mppi_sampler.hpp>
#include <algorithm>
#include <array>
#include <cmath>

namespace path_follower {

// ════════════════════════════════════════════════════════════════
//  工具函数
// ════════════════════════════════════════════════════════════════

namespace {

constexpr double COST_EPS = 1e-9;
constexpr double REACHABILITY_EPS = 1e-6;

const MPCFollowModeProfile& select_follow_mode_profile(
    const MPCFollowParams& params,
    std::optional<ActiveStepMode> active_step_mode
) {
    if (!active_step_mode) {
        return params.mode_profiles.normal;
    }
    switch (active_step_mode->mode) {
        case ChassisMode::STEP_UP_LEG_SHORT: return params.mode_profiles.up.short_leg;
        case ChassisMode::STEP_UP_JUMP: return params.mode_profiles.up.jump;
        case ChassisMode::STEP_UP_LEG_LONG: return params.mode_profiles.up.long_leg;
        case ChassisMode::STEP_DOWN_LEG_SHORT: return params.mode_profiles.down.short_leg;
        case ChassisMode::STEP_DOWN_JUMP: return params.mode_profiles.down.jump;
        default: return params.mode_profiles.normal;
    }
}

bool is_active_follow_step_mode(std::optional<ActiveStepMode> active_step_mode) {
    return active_step_mode.has_value() && active_step_mode->mode != ChassisMode::NORMAL;
}



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

double brake_speed_limit(double remaining_s, const MPCFollowTerminalWeights& w) {
    return std::sqrt(std::max(0.0, 2.0 * w.a_brake * remaining_s) + w.slow_down_target_vel * w.slow_down_target_vel);
}

inline double smooth_sgn(double x, double eps) {
    return std::tanh(x / std::max(eps, 1e-6));
}

inline double smooth_sgn_deriv(double x, double eps) {
    const double s = smooth_sgn(x, eps);
    return (1.0 - s * s) / std::max(eps, 1e-6);
}

inline double clamp_lpv_rho(double rho, double rho_clip) {
    return std::clamp(rho, -rho_clip, rho_clip);
}

inline double lpv_schedule_z(double leg_h, double leg_psi) {
    return leg_h * std::cos(leg_psi);
}

double schedule_rho_from_state(const ChassisMotionState& chassis_state, const LPVKinematicModelParams& model_params) {
    const double z = lpv_schedule_z(chassis_state.leg_h, chassis_state.leg_psi);
    return clamp_lpv_rho((z - model_params.z_ref) / std::max(model_params.z_scale, 1e-6), model_params.rho_clip);
}

double select_follow_schedule_rho(
    const ChassisMotionState& chassis_state,
    const LPVKinematicModelParams& model_params
) {
    return schedule_rho_from_state(chassis_state, model_params);
}

void zoh_v_matrices(
    double a00,
    double a01,
    double a10,
    double a11,
    double b0,
    double b1,
    double g0,
    double g1,
    double dt,
    double& ad00,
    double& ad01,
    double& ad10,
    double& ad11,
    double& bd0,
    double& bd1,
    double& gd0,
    double& gd1
) {
    const double m00 = a00 * dt;
    const double m01 = a01 * dt;
    const double m10 = a10 * dt;
    const double m11 = a11 * dt;
    const double tr_m = m00 + m11;
    const double det_m = m00 * m11 - m01 * m10;
    const double disc = tr_m * tr_m - 4.0 * det_m;
    constexpr double REG_EPS = 1e-8;

    double alpha = 0.0;
    double beta = 0.0;
    if (disc > REG_EPS) {
        const double s = std::sqrt(disc);
        const double lam1 = 0.5 * (tr_m + s);
        const double lam2 = 0.5 * (tr_m - s);
        const double delta = lam1 - lam2;
        const double el2 = std::exp(lam2);
        beta = el2 * std::expm1(delta) / delta;
        alpha = std::exp(lam1) - beta * lam1;
    } else if (disc < -REG_EPS) {
        const double p = 0.5 * tr_m;
        const double q = 0.5 * std::sqrt(-disc);
        const double ep = std::exp(p);
        beta = ep * std::sin(q) / q;
        alpha = ep * (std::cos(q) - p * std::sin(q) / q);
    } else {
        const double lam = 0.5 * tr_m;
        const double el = std::exp(lam);
        beta = el;
        alpha = el * (1.0 - lam);
    }

    ad00 = alpha + beta * m00;
    ad01 = beta * m01;
    ad10 = beta * m10;
    ad11 = alpha + beta * m11;

    const double det_a = a00 * a11 - a01 * a10;
    const double c = alpha - 1.0;
    double g00 = 0.0;
    double g01 = 0.0;
    double g10 = 0.0;
    double g11 = 0.0;
    constexpr double DET_EPS = 1e-6;
    if (std::abs(det_a) > DET_EPS) {
        const double inv_det = 1.0 / det_a;
        g00 = c * a11 * inv_det + beta * dt;
        g01 = c * (-a01) * inv_det;
        g10 = c * (-a10) * inv_det;
        g11 = c * a00 * inv_det + beta * dt;
    } else {
        // Near-singular A: second-order Taylor for G = ∫₀ᵈᵗ exp(A τ) dτ
        g00 = dt + 0.5 * dt * dt * a00;
        g01 = 0.5 * dt * dt * a01;
        g10 = 0.5 * dt * dt * a10;
        g11 = dt + 0.5 * dt * dt * a11;
    }

    bd0 = g00 * b0 + g01 * b1;
    bd1 = g10 * b0 + g11 * b1;
    gd0 = g00 * g0 + g01 * g1;
    gd1 = g10 * g0 + g11 * g1;
}

LPVDiscreteModel build_lpv_discrete_model(const LPVKinematicModelParams& params, double rho) {
    LPVDiscreteModel model;
    model.rho = clamp_lpv_rho(rho, params.rho_clip);

    const double a00 = params.ca00 + model.rho * params.dca00;
    const double a01 = params.ca01 + model.rho * params.dca01;
    const double a10 = params.ca10 + model.rho * params.dca10;
    const double a11 = params.ca11 + model.rho * params.dca11;
    const double b0 = params.cb0 + model.rho * params.dcb0;
    const double b1 = params.cb1 + model.rho * params.dcb1;

    zoh_v_matrices(
        a00, a01, a10, a11, b0, b1, params.gxh, params.gv, MPC_DT,
        model.ad00, model.ad01, model.ad10, model.ad11,
        model.bd0, model.bd1, model.gd0, model.gd1
    );

    const double lam = std::max(params.w_lam0 + model.rho * params.w_lam1, 1e-5);
    const double kw = params.w_k0 + model.rho * params.w_k1;
    const double cf = params.w_cf0 + model.rho * params.w_cf1;
    model.alpha_w = std::exp(-lam * MPC_DT);
    const double integ_w = (1.0 - model.alpha_w) / lam;
    model.beta_w = integ_w * kw;
    model.gamma_w = integ_w * cf;
    model.sgn_eps = params.sgn_eps;
    model.cf1 = params.cf1;
    model.cf2 = params.cf2;
    return model;
}

struct LPVNonlinearEval {
    double nl;
    double dnl_dv;
    double dnl_dw;
    double sw;
    double dsw_dw;
};

LPVNonlinearEval evaluate_lpv_nonlinear(double v, double w, const LPVDiscreteModel& model) {
    const double sv = smooth_sgn(v, model.sgn_eps);
    const double dsv = smooth_sgn_deriv(v, model.sgn_eps);
    const double sw = smooth_sgn(w, model.sgn_eps);
    const double dsw = smooth_sgn_deriv(w, model.sgn_eps);
    const double absw = std::abs(w);
    const double sabsw = smooth_sgn(w, model.sgn_eps);
    return {
        .nl = model.cf1 * sv + model.cf2 * v * absw,
        .dnl_dv = model.cf1 * dsv + model.cf2 * absw,
        .dnl_dw = model.cf2 * v * sabsw,
        .sw = sw,
        .dsw_dw = dsw,
    };
}

Eigen::Vector2d apply_goal_deadzone(const Eigen::Vector2d& delta, double deadzone) {
    if (deadzone <= 0.0) return delta;

    const double dist = delta.norm();
    if (dist <= 0.0) return Eigen::Vector2d::Zero();

    const double mag = positive_part(dist - deadzone);
    if (mag <= 0.0) {
        return Eigen::Vector2d::Zero();
    }
    return delta * (mag / dist);
}

inline double wrap_pi(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

inline double predict_power(const PowerModelParams& power_model, double v, double w, double a, double alpha) {
    const auto& c = power_model.coeffs;
    return c[0] + c[1] * v * a + c[2] * w * alpha + c[3] * a * a + c[4] * alpha * alpha
        + c[5] * std::abs(v) + c[6] * std::abs(w) + c[7] * v * v + c[8] * w * w + c[9] * std::abs(a)
        + c[10] * std::abs(alpha) + c[11] * std::abs(v * w);
}

struct PowerEval {
    double value;
    double dv;
    double dw;
};

inline PowerEval predict_power_eval_vw(const PowerModelParams& power_model, double v, double w) {
    const auto& c = power_model.coeffs;
    const double vw = v * w;
    const double sign_v = sign_or_zero(v);
    const double sign_w = sign_or_zero(w);
    const double sign_vw = sign_or_zero(vw);

    PowerEval out {};
    out.value = c[0] + c[5] * std::abs(v) + c[6] * std::abs(w) + c[7] * v * v + c[8] * w * w + c[11] * std::abs(vw);

    out.dv = c[5] * sign_v + 2.0 * c[7] * v + c[11] * sign_vw * w;
    out.dw = c[6] * sign_w + 2.0 * c[8] * w + c[11] * sign_vw * v;
    return out;
}

inline double clamp_prev_cmd(double cmd_prev, double status, double cmd_act_diff_max, double rate_max, double dt) {
    return std::max(
        std::min(cmd_prev, status + cmd_act_diff_max - rate_max * dt),
        status - cmd_act_diff_max + rate_max * dt
    );
}

double advance_u_progress(double u_cur, const StateVec& x, const std::vector<Eigen::Vector2d>& ref_cps);
double advance_u_progress_extrapolated(double u_cur, const StateVec& x, const std::vector<Eigen::Vector2d>& ref_cps);

struct AdvanceUProgressEval {
    double u_next_extrap;
    StateVec du_next_dx;
};

AdvanceUProgressEval
advance_u_progress_extrapolated_with_jacobian(double u_cur, const StateVec& x, const std::vector<Eigen::Vector2d>& ref_cps);

inline bool same_cps(const std::vector<Eigen::Vector2d>& a, const std::vector<Eigen::Vector2d>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].x() != b[i].x() || a[i].y() != b[i].y()) {
            return false;
        }
    }
    return true;
}

// ─── Gauss-Newton 残差辅助 ───

template<typename ResidualVec>
double residual_cost(const ResidualVec& r) {
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

} // anonymous namespace

// ════════════════════════════════════════════════════════════════
//  代价地图双线性采样（保留函数名以兼容现有调用）
// ════════════════════════════════════════════════════════════════

CostSample eval_cost_bilinear(const CostMapGridView& grid, const GridInfo& info, double x_map, double y_map) {
    CostSample s {255.0, 0.0, 0.0};
    if (!std::isfinite(x_map) || !std::isfinite(y_map)) return s;
    if (info.width < 2 || info.height < 2) return s;

    const double gx = (x_map - info.origin_x) * info.inv_resolution;
    const double gy = (y_map - info.origin_y) * info.inv_resolution;
    if (gx < 0.0 || gy < 0.0 || gx >= static_cast<double>(info.width - 1)
        || gy >= static_cast<double>(info.height - 1)) {
        return s;
    }

    const int ix0 = static_cast<int>(std::floor(gx));
    const int iy0 = static_cast<int>(std::floor(gy));
    const double tx = gx - static_cast<double>(ix0);
    const double ty = gy - static_cast<double>(iy0);

    const double f00 = grid.value_at_clamped(iy0, ix0);
    const double f10 = grid.value_at_clamped(iy0, ix0 + 1);
    const double f01 = grid.value_at_clamped(iy0 + 1, ix0);
    const double f11 = grid.value_at_clamped(iy0 + 1, ix0 + 1);

    const double w00 = (1.0 - tx) * (1.0 - ty);
    const double w10 = tx * (1.0 - ty);
    const double w01 = (1.0 - tx) * ty;
    const double w11 = tx * ty;

    s.value = w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11;

    const double dvdgx = (1.0 - ty) * (f10 - f00) + ty * (f11 - f01);
    const double dvdgy = (1.0 - tx) * (f01 - f00) + tx * (f11 - f10);
    s.dx = dvdgx * info.inv_resolution;
    s.dy = dvdgy * info.inv_resolution;
    return s;
}

DirSample eval_dir_bilinear(const DirectionMapGridView& grid, const GridInfo& info, double x_map, double y_map) {
    DirSample s {Eigen::Vector2d::Zero(), Eigen::Matrix2d::Zero()};
    if (!std::isfinite(x_map) || !std::isfinite(y_map)) return s;
    if (info.width < 2 || info.height < 2) return s;

    const double gx = (x_map - info.origin_x) * info.inv_resolution;
    const double gy = (y_map - info.origin_y) * info.inv_resolution;
    if (gx < 0.0 || gy < 0.0 || gx >= static_cast<double>(info.width - 1)
        || gy >= static_cast<double>(info.height - 1)) {
        return s;
    }

    const int ix0 = static_cast<int>(std::floor(gx));
    const int iy0 = static_cast<int>(std::floor(gy));
    const double tx = gx - static_cast<double>(ix0);
    const double ty = gy - static_cast<double>(iy0);

    const Eigen::Vector2d f00 = grid.value_at_clamped(iy0, ix0);
    const Eigen::Vector2d f10 = grid.value_at_clamped(iy0, ix0 + 1);
    const Eigen::Vector2d f01 = grid.value_at_clamped(iy0 + 1, ix0);
    const Eigen::Vector2d f11 = grid.value_at_clamped(iy0 + 1, ix0 + 1);

    const double w00 = (1.0 - tx) * (1.0 - ty);
    const double w10 = tx * (1.0 - ty);
    const double w01 = (1.0 - tx) * ty;
    const double w11 = tx * ty;

    s.value = w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11;

    const Eigen::Vector2d dvdgx = (1.0 - ty) * (f10 - f00) + ty * (f11 - f01);
    const Eigen::Vector2d dvdgy = (1.0 - tx) * (f01 - f00) + tx * (f11 - f10);
    s.J.col(0) = dvdgx * info.inv_resolution;
    s.J.col(1) = dvdgy * info.inv_resolution;
    return s;
}

// ════════════════════════════════════════════════════════════════
//  共享动力学模型
// ════════════════════════════════════════════════════════════════

StateVec mpc_dynamics(const StateVec& x, const ControlVec& u, const LPVDiscreteModel& model) {
    const double theta = x(ix::THETA);
    const double xh = x(ix::XH), v = x(ix::V), w = x(ix::W);
    const double dv = x(ix::DV), dw = x(ix::DW);

    const auto nl_eval = evaluate_lpv_nonlinear(v, w, model);

    const double xh1 = model.ad00 * xh + model.ad01 * v + model.bd0 * dv + model.gd0 * nl_eval.nl;
    const double v1 = model.ad10 * xh + model.ad11 * v + model.bd1 * dv + model.gd1 * nl_eval.nl;
    const double w1 = model.alpha_w * w + model.beta_w * dw - model.gamma_w * nl_eval.sw;

    const double dt = MPC_DT;
    const double theta1 = theta + (w + w1) * (dt * 0.5);
    const double ct0 = std::cos(theta), st0 = std::sin(theta);
    const double ct1 = std::cos(theta1), st1 = std::sin(theta1);

    StateVec xn;
    xn(ix::X) = x(ix::X) + (v * ct0 + v1 * ct1) * (dt * 0.5);
    xn(ix::Y) = x(ix::Y) + (v * st0 + v1 * st1) * (dt * 0.5);
    xn(ix::THETA) = theta1;
    xn(ix::XH) = xh1;
    xn(ix::V) = v1;
    xn(ix::W) = w1;
    xn(ix::DV) = u(0);
    xn(ix::DW) = u(1);
    xn(ix::PATH_U) = x(ix::PATH_U);
    return xn;
}

void mpc_dynamics_jacobians(const StateVec& x, const ControlVec& /*u*/, const LPVDiscreteModel& model, MatXX& fx, MatXU& fu) {
    const double theta = x(ix::THETA);
    const double xh = x(ix::XH), v = x(ix::V), w = x(ix::W);
    const double dv = x(ix::DV), dw = x(ix::DW);
    const double dt = MPC_DT, h = dt * 0.5;

    const auto nl_eval = evaluate_lpv_nonlinear(v, w, model);

    const double v1 = model.ad10 * xh + model.ad11 * v + model.bd1 * dv + model.gd1 * nl_eval.nl;
    const double w1 = model.alpha_w * w + model.beta_w * dw - model.gamma_w * nl_eval.sw;
    const double theta1 = theta + (w + w1) * h;

    const double dvn_dxh = model.ad10;
    const double dvn_dv = model.ad11 + model.gd1 * nl_eval.dnl_dv;
    const double dvn_dw = model.gd1 * nl_eval.dnl_dw;
    const double dvn_ddv = model.bd1;

    const double dwn_dw = model.alpha_w - model.gamma_w * nl_eval.dsw_dw;
    const double dwn_ddw = model.beta_w;

    const double dth1_dth = 1.0;
    const double dth1_dw = (1.0 + dwn_dw) * h;
    const double dth1_ddw = dwn_ddw * h;

    const double ct0 = std::cos(theta), st0 = std::sin(theta);
    const double ct1 = std::cos(theta1), st1 = std::sin(theta1);

    fx.setZero();
    fu.setZero();

    // ∂X/∂state
    fx(ix::X, ix::X) = 1.0;
    fx(ix::X, ix::THETA) = (-v * st0 - v1 * st1 * dth1_dth) * h;
    fx(ix::X, ix::XH) = dvn_dxh * ct1 * h;
    fx(ix::X, ix::V) = (ct0 + dvn_dv * ct1) * h;
    fx(ix::X, ix::W) = (dvn_dw * ct1 - v1 * st1 * dth1_dw) * h;
    fx(ix::X, ix::DV) = dvn_ddv * ct1 * h;
    fx(ix::X, ix::DW) = -v1 * st1 * dth1_ddw * h;

    fx(ix::Y, ix::Y) = 1.0;
    fx(ix::Y, ix::THETA) = (v * ct0 + v1 * ct1 * dth1_dth) * h;
    fx(ix::Y, ix::XH) = dvn_dxh * st1 * h;
    fx(ix::Y, ix::V) = (st0 + dvn_dv * st1) * h;
    fx(ix::Y, ix::W) = (dvn_dw * st1 + v1 * ct1 * dth1_dw) * h;
    fx(ix::Y, ix::DV) = dvn_ddv * st1 * h;
    fx(ix::Y, ix::DW) = v1 * ct1 * dth1_ddw * h;

    fx(ix::THETA, ix::THETA) = dth1_dth;
    fx(ix::THETA, ix::W) = dth1_dw;
    fx(ix::THETA, ix::DW) = dth1_ddw;

    fx(ix::XH, ix::XH) = model.ad00;
    fx(ix::XH, ix::V) = model.ad01 + model.gd0 * nl_eval.dnl_dv;
    fx(ix::XH, ix::W) = model.gd0 * nl_eval.dnl_dw;
    fx(ix::XH, ix::DV) = model.bd0;

    fx(ix::V, ix::XH) = dvn_dxh;
    fx(ix::V, ix::V) = dvn_dv;
    fx(ix::V, ix::W) = dvn_dw;
    fx(ix::V, ix::DV) = dvn_ddv;

    fx(ix::W, ix::W) = dwn_dw;
    fx(ix::W, ix::DW) = dwn_ddw;

    fx(ix::PATH_U, ix::PATH_U) = 1.0;

    // ∂X/∂u: only DV_next = v_cmd, DW_next = w_cmd
    fu(ix::DV, 0) = 1.0;
    fu(ix::DW, 1) = 1.0;
}

// ════════════════════════════════════════════════════════════════
//  FollowProblem
// ════════════════════════════════════════════════════════════════

template<int Horizon>
FollowProblemT<Horizon>::FollowProblemT(
    const std::vector<Eigen::Vector2d>& ref_control_points,
    const MPCParams& params,
    const std::vector<CostMapGridView>& per_step_cost_grids,
    const GridInfo& cost_info,
    const CostMapGridView& masked_global_grid,
    double prediction_dt,
    double schedule_rho,
    const DirectionMapGridView& dir_grid,
    const GridInfo& dir_info,
    double remaining_energy,
    double rfr_pwr_limit,
    std::optional<ActiveStepMode> active_step_mode
):
    ref_cps_(ref_control_points),
    p_(params),
    step_cost_grids_(per_step_cost_grids),
    cost_info_(cost_info),
    masked_global_grid_(masked_global_grid),
    prediction_dt_(prediction_dt),
    model_(build_lpv_discrete_model(params.kinematic_model, schedule_rho)),
    dir_grid_(dir_grid),
    dir_info_(dir_info),
    remaining_energy_(remaining_energy),
    rfr_pwr_limit_(rfr_pwr_limit),
    active_step_mode_(active_step_mode) {
    Eigen::Vector2d p;
    eval_quadratic_bspline2_extrapolated(ref_control_points, 1.0, &p, nullptr, nullptr);
    goal_xy_ = p;
}

template<int Horizon>
StateVec FollowProblemT<Horizon>::dynamics(int, const StateVec& x, const ControlVec& u) const {
    StateVec xn = mpc_dynamics(x, u, model_);
    xn(ix::PATH_U) = advance_u_progress(x(ix::PATH_U), x, ref_cps_);
    return xn;
}

template<int Horizon>
void FollowProblemT<Horizon>::dynamics_jacobians(int, const StateVec& x, const ControlVec& u, MatXX& dfx, MatXU& dfu) const {
    mpc_dynamics_jacobians(x, u, model_, dfx, dfu);

    const auto adv = advance_u_progress_extrapolated_with_jacobian(x(ix::PATH_U), x, ref_cps_);
    const double dout_din = clamp_derivative_piecewise(adv.u_next_extrap, PATH_U_EXTRAP_MIN, PATH_U_EXTRAP_MAX);
    dfx.row(ix::PATH_U) = (dout_din * adv.du_next_dx).transpose();
    dfu.row(ix::PATH_U).setZero();
}

template<int Horizon>
ControlVec FollowProblemT<Horizon>::u_lower() const {
    const auto& command_bounds = select_follow_mode_profile(p_.follow, active_step_mode_).command_bounds;
    return ControlVec(command_bounds.vel_min, command_bounds.omega_min);
}

template<int Horizon>
ControlVec FollowProblemT<Horizon>::u_upper() const {
    const auto& command_bounds = select_follow_mode_profile(p_.follow, active_step_mode_).command_bounds;
    return ControlVec(command_bounds.vel_max, command_bounds.omega_max);
}

template<int Horizon>
const std::vector<Eigen::Vector2d>& FollowProblemT<Horizon>::ref_control_points() const {
    return ref_cps_;
}

template<int Horizon>
const MPCParams& FollowProblemT<Horizon>::params() const {
    return p_;
}

template<int Horizon>
FollowProblemT<Horizon> FollowProblemT<Horizon>::with_reference_path(const std::vector<Eigen::Vector2d>& ref_control_points) const {
    return FollowProblemT<Horizon>(
        ref_control_points,
        p_,
        step_cost_grids_,
        cost_info_,
        masked_global_grid_,
        prediction_dt_,
        model_.rho,
        dir_grid_,
        dir_info_,
        remaining_energy_,
        rfr_pwr_limit_,
        active_step_mode_
    );
}

namespace {

constexpr int FOLLOW_RESIDUAL_DIM = 16;
using FollowResidualVec = Eigen::Matrix<double, FOLLOW_RESIDUAL_DIM, 1>;

struct FollowResidualLinearization {
    FollowResidualVec r;
    Eigen::Matrix<double, FOLLOW_RESIDUAL_DIM, MPC_NX> jx;
    Eigen::Matrix<double, FOLLOW_RESIDUAL_DIM, MPC_NU> ju;
};

FollowResidualLinearization follow_residual_linearized_impl(
    const StateVec& x,
    const ControlVec& u,
    const std::vector<Eigen::Vector2d>& ref_cps,
    const MPCParams& p,
    const CostMapGridView& cg,
    const GridInfo& ci,
    const DirectionMapGridView& dg,
    const GridInfo& di,
    double rfr_pwr_limit,
    std::optional<ActiveStepMode> active_step_mode
) {
    const auto& follow = p.follow;
    const auto& tracking_w = follow.tracking_weights;
    const auto& command_w = follow.command_weights;
    const auto& motion_w = follow.motion_constraint_weights;
    const auto& terrain_w = follow.terrain_weights;
    const auto& env_w = follow.environment_weights;
    const auto& motion_lim = select_follow_mode_profile(follow, active_step_mode).motion_constraints;

    FollowResidualLinearization out;
    out.r.setZero();
    out.jx.setZero();
    out.ju.setZero();

    const double px = x(ix::X), py = x(ix::Y), theta = x(ix::THETA);
    const double v_act = x(ix::V), w_act = x(ix::W);
    const double v_cmd = u(0), w_cmd = u(1);
    const double dv_cmd = v_cmd - x(ix::DV);
    const double dw_cmd = w_cmd - x(ix::DW);

    const double uc_raw = x(ix::PATH_U);
    const double uc = clamp_path_u_extrapolated(uc_raw);
    const double duc_dpathu = clamp_derivative_piecewise(uc_raw, PATH_U_EXTRAP_MIN, PATH_U_EXTRAP_MAX);

    Eigen::Vector2d pr, d1, d2;
    eval_quadratic_bspline2_extrapolated(ref_cps, uc, &pr, &d1, &d2);

    const double thetar = std::atan2(d1.y(), d1.x());
    const double sin_r = std::sin(thetar);
    const double cos_r = std::cos(thetar);

    const double d1_norm2 = d1.squaredNorm();
    const double dtheta_du = (d1.x() * d2.y() - d1.y() * d2.x()) / std::max(d1_norm2, 1e-12);

    const double ex = px - pr.x(), ey_w = py - pr.y();
    const double ey = -ex * sin_r + ey_w * cos_r;
    const double dey_dpx = -sin_r;
    const double dey_dpy = cos_r;
    const double dey_du = sin_r * d1.x() - cos_r * d1.y() - dtheta_du * (ex * cos_r + ey_w * sin_r);
    const double dey_dpathu = dey_du * duc_dpathu;

    const double etheta = wrap_pi(theta - thetar);
    const double detheta_dpathu = -dtheta_du * duc_dpathu;

    const auto cs = eval_cost_bilinear(cg, ci, px, py);
    const auto ds = eval_dir_bilinear(dg, di, px, py);

    const Eigen::Vector2d dir = ds.value;
    const double dir_norm_sq = dir.squaredNorm();
    const double dir_norm = std::sqrt(ds.value.squaredNorm() + 1e-10);
    const double inv_dir_norm = 1.0 / dir_norm;
    const double cos_t = std::cos(theta);
    const double sin_t = std::sin(theta);
    const Eigen::Vector2d heading(cos_t, sin_t);

    const Eigen::Vector2d ddir_dx = ds.J.col(0);
    const Eigen::Vector2d ddir_dy = ds.J.col(1);
    const double dnorm_dx = dir.dot(ddir_dx) * inv_dir_norm;
    const double dnorm_dy = dir.dot(ddir_dy) * inv_dir_norm;

    const double cross = heading.x() * dir.y() - heading.y() * dir.x();
    const double dcross_dtheta = -sin_t * dir.y() - cos_t * dir.x();
    const double dcross_dx = cos_t * ddir_dx.y() - sin_t * ddir_dx.x();
    const double dcross_dy = cos_t * ddir_dy.y() - sin_t * ddir_dy.x();

    const double a_lat = std::abs(v_act * w_act);

    const double dv_lim = motion_lim.acc_max * MPC_DT;
    const double dw_lim = motion_lim.alpha_max * MPC_DT;

    out.r(0) = tracking_w.q_y * ey;
    out.jx(0, ix::X) = tracking_w.q_y * dey_dpx;
    out.jx(0, ix::Y) = tracking_w.q_y * dey_dpy;
    out.jx(0, ix::PATH_U) = tracking_w.q_y * dey_dpathu;

    out.r(1) = tracking_w.q_theta * etheta;
    out.jx(1, ix::THETA) = tracking_w.q_theta;
    out.jx(1, ix::PATH_U) = tracking_w.q_theta * detheta_dpathu;

    out.r(2) = command_w.r_v * v_cmd;
    out.ju(2, 0) = command_w.r_v;

    out.r(3) = command_w.r_omega * w_cmd;
    out.ju(3, 1) = command_w.r_omega;

    out.r(4) = command_w.r_dv * dv_cmd;
    out.ju(4, 0) = command_w.r_dv;
    out.jx(4, ix::DV) = -command_w.r_dv;

    out.r(5) = command_w.r_domega * dw_cmd;
    out.ju(5, 1) = command_w.r_domega;
    out.jx(5, ix::DW) = -command_w.r_domega;

    const double abs_dv_cmd = std::abs(dv_cmd);
    const double relu_dv = positive_part(abs_dv_cmd - dv_lim);
    out.r(6) = motion_w.acc_limit * relu_dv;
    const double coeff_dv = motion_w.acc_limit * positive_part_derivative(abs_dv_cmd - dv_lim) * sign_or_zero(dv_cmd);
    out.ju(6, 0) = coeff_dv;
    out.jx(6, ix::DV) = -coeff_dv;

    const double abs_dw_cmd = std::abs(dw_cmd);
    const double relu_dw = positive_part(abs_dw_cmd - dw_lim);
    out.r(7) = motion_w.alpha_limit * relu_dw;
    const double coeff_dw = motion_w.alpha_limit * positive_part_derivative(abs_dw_cmd - dw_lim) * sign_or_zero(dw_cmd);
    out.ju(7, 1) = coeff_dw;
    out.jx(7, ix::DW) = -coeff_dw;

    const double sign_lat = sign_or_zero(v_act * w_act);
    const double relu_lat = positive_part(a_lat - motion_lim.a_lat_max);
    out.r(8) = motion_w.lat_acc * relu_lat;
    const double coeff_lat = motion_w.lat_acc * positive_part_derivative(a_lat - motion_lim.a_lat_max);
    out.jx(8, ix::V) = coeff_lat * sign_lat * w_act;
    out.jx(8, ix::W) = coeff_lat * sign_lat * v_act;

    out.r(9) = env_w.obstacle * cs.value / 255.0;
    out.jx(9, ix::X) = env_w.obstacle * cs.dx / 255.0;
    out.jx(9, ix::Y) = env_w.obstacle * cs.dy / 255.0;

    const double sign_cross = sign_or_zero(cross);
    out.r(10) = terrain_w.direction * std::abs(cross);
    out.jx(10, ix::X) = terrain_w.direction * sign_cross * dcross_dx;
    out.jx(10, ix::Y) = terrain_w.direction * sign_cross * dcross_dy;
    out.jx(10, ix::THETA) = terrain_w.direction * sign_cross * dcross_dtheta;

    if (is_active_follow_step_mode(active_step_mode)) {
        const double target_vel = active_step_mode->target_velocity;
        const double v_min = target_vel - follow.terrain_limits.step_vel_deadzone;
        const double v_max = target_vel + follow.terrain_limits.step_vel_deadzone;
        const double v_center = 0.5 * (v_min + v_max);
        const double v_err = v_act - v_center;
        const double abs_v_err = std::abs(v_err);
        const double relu_vstep = positive_part(abs_v_err - follow.terrain_limits.step_vel_deadzone);

        if (dir_norm_sq > 1e-10) {
            out.r(11) = terrain_w.step_vel_weight * dir_norm * relu_vstep;
            out.jx(11, ix::X) = terrain_w.step_vel_weight * dnorm_dx * relu_vstep;
            out.jx(11, ix::Y) = terrain_w.step_vel_weight * dnorm_dy * relu_vstep;
            out.jx(11, ix::V) = terrain_w.step_vel_weight * dir_norm
                * positive_part_derivative(abs_v_err - follow.terrain_limits.step_vel_deadzone) * sign_or_zero(v_err);
        }

        if (active_step_mode->step_entry_u.has_value()) {
            const double entry_u = std::clamp(*active_step_mode->step_entry_u, 0.0, 1.0);
            const double path_u = std::clamp(uc, 0.0, 1.0);
            if (path_u < entry_u) {
                const double d = quadratic_bspline_arc_length(ref_cps, path_u, entry_u);

                Eigen::Vector2d path_d1;
                eval_quadratic_bspline2_extrapolated(ref_cps, path_u, nullptr, &path_d1, nullptr);
                const double ds_du = path_d1.norm() * duc_dpathu;

                const double a_guide = std::max(follow.terrain_limits.step_reachability_guide_acc, REACHABILITY_EPS);
                const double r_lo_expr = std::max(0.0, v_act) * std::max(0.0, v_act) + 2.0 * a_guide * d;
                const double r_hi_expr = v_act * v_act - 2.0 * a_guide * d;
                const double r_lo_arg = std::max(REACHABILITY_EPS, r_lo_expr);
                const double r_hi_arg = std::max(REACHABILITY_EPS, r_hi_expr);
                const double r_lo = std::sqrt(r_lo_arg);
                const double r_hi = std::sqrt(r_hi_arg);

                const double relu_lo = positive_part(v_min - r_lo);
                out.r(12) = std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * relu_lo;
                if (relu_lo > 0.0) {
                    const bool lo_active = r_lo_expr > REACHABILITY_EPS;
                    const double drlo_dv = (lo_active && v_act > 0.0) ? (v_act / r_lo) : 0.0;
                    const double drlo_du = (lo_active) ? (-a_guide * ds_du / r_lo) : 0.0;
                    out.jx(12, ix::V) = -std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * drlo_dv;
                    out.jx(12, ix::PATH_U) = -std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * drlo_du;
                }

                const double relu_hi = positive_part(r_hi - v_max);
                out.r(13) = std::sqrt(std::max(terrain_w.step_reachability_hi, 0.0)) * relu_hi;
                if (relu_hi > 0.0) {
                    const bool hi_active = r_hi_expr > REACHABILITY_EPS;
                    const double drhi_dv = hi_active ? (v_act / r_hi) : 0.0;
                    const double drhi_du = (hi_active) ? (a_guide * ds_du / r_hi) : 0.0;
                    out.jx(13, ix::V) = std::sqrt(std::max(terrain_w.step_reachability_hi, 0.0)) * drhi_dv;
                    out.jx(13, ix::PATH_U) = std::sqrt(std::max(terrain_w.step_reachability_hi, 0.0)) * drhi_du;
                }
            }
        }
    }

    const auto& terminal_w = follow.terminal_weights;
    const double uc_clamped_brake = std::clamp(uc, 0.0, 1.0);
    const double s_remaining_approx = (1.0 - uc_clamped_brake) * d1.norm();
    const double v_allowed = brake_speed_limit(s_remaining_approx, terminal_w);
    const double v_excess = v_act - v_allowed;
    const double relu_brake = positive_part(v_excess);
    out.r(14) = terminal_w.q_v_final * relu_brake;
    if (v_excess > 0.0) {
        out.jx(14, ix::V) = terminal_w.q_v_final;
        const double ds_dpathu = -d1.norm() * duc_dpathu;
        const double dv_allowed_ds = terminal_w.a_brake / std::max(v_allowed, 1e-6);
        out.jx(14, ix::PATH_U) = -terminal_w.q_v_final * dv_allowed_ds * ds_dpathu;
    }

    if (p.energy.enable) {
        const auto pwr = predict_power_eval_vw(p.power_model, v_act, w_act);
        const double thr = std::max(p.energy.threshold, 1.0);
        const double excess = (pwr.value - rfr_pwr_limit) / thr;
        if (excess > 0.0) {
            out.r(15) = p.energy.weight * excess;
            const double common = p.energy.weight / thr;
            out.jx(15, ix::V) = common * pwr.dv;
            out.jx(15, ix::W) = common * pwr.dw;
        }
    }

    return out;
}

double follow_running_cost_value_only_impl(
    const StateVec& x,
    const ControlVec& u,
    const std::vector<Eigen::Vector2d>& ref_cps,
    const MPCParams& p,
    const CostMapGridView& cg,
    const GridInfo& ci,
    const DirectionMapGridView& dg,
    const GridInfo& di,
    double rfr_pwr_limit,
    std::optional<ActiveStepMode> active_step_mode
) {
    const auto& follow = p.follow;
    const auto& tracking_w = follow.tracking_weights;
    const auto& command_w = follow.command_weights;
    const auto& motion_w = follow.motion_constraint_weights;
    const auto& terrain_w = follow.terrain_weights;
    const auto& env_w = follow.environment_weights;
    const auto& motion_lim = select_follow_mode_profile(follow, active_step_mode).motion_constraints;

    const double px = x(ix::X);
    const double py = x(ix::Y);
    const double theta = x(ix::THETA);
    const double v_act = x(ix::V);
    const double w_act = x(ix::W);
    const double v_cmd = u(0);
    const double w_cmd = u(1);
    const double dv_cmd = v_cmd - x(ix::DV);
    const double dw_cmd = w_cmd - x(ix::DW);

    const double uc_raw = x(ix::PATH_U);
    const double uc = clamp_path_u_extrapolated(uc_raw);

    Eigen::Vector2d pr;
    Eigen::Vector2d d1;
    Eigen::Vector2d d2;
    eval_quadratic_bspline2_extrapolated(ref_cps, uc, &pr, &d1, &d2);

    const double thetar = std::atan2(d1.y(), d1.x());
    const double sin_r = std::sin(thetar);
    const double cos_r = std::cos(thetar);

    const double ex = px - pr.x();
    const double ey_w = py - pr.y();
    const double ey = -ex * sin_r + ey_w * cos_r;
    const double etheta = wrap_pi(theta - thetar);

    const auto cs = eval_cost_bilinear(cg, ci, px, py);
    const auto ds = eval_dir_bilinear(dg, di, px, py);

    const Eigen::Vector2d dir = ds.value;
    const double dir_norm_sq = dir.squaredNorm();
    const double cross = std::cos(theta) * dir.y() - std::sin(theta) * dir.x();
    const double a_lat = std::abs(v_act * w_act);

    const double dv_lim = motion_lim.acc_max * MPC_DT;
    const double dw_lim = motion_lim.alpha_max * MPC_DT;

    double cost = 0.0;

    cost += 0.5 * std::pow(tracking_w.q_y * ey, 2);
    cost += 0.5 * std::pow(tracking_w.q_theta * etheta, 2);
    cost += 0.5 * std::pow(command_w.r_v * v_cmd, 2);
    cost += 0.5 * std::pow(command_w.r_omega * w_cmd, 2);
    cost += 0.5 * std::pow(command_w.r_dv * dv_cmd, 2);
    cost += 0.5 * std::pow(command_w.r_domega * dw_cmd, 2);
    cost += 0.5 * std::pow(motion_w.acc_limit * positive_part(std::abs(dv_cmd) - dv_lim), 2);
    cost += 0.5 * std::pow(motion_w.alpha_limit * positive_part(std::abs(dw_cmd) - dw_lim), 2);
    cost += 0.5 * std::pow(motion_w.lat_acc * positive_part(a_lat - motion_lim.a_lat_max), 2);
    cost += 0.5 * std::pow(env_w.obstacle * cs.value / 255.0, 2);
    cost += 0.5 * std::pow(terrain_w.direction * std::abs(cross), 2);

    if (is_active_follow_step_mode(active_step_mode)) {
        const double target_vel = active_step_mode->target_velocity;
        const double deadzone = follow.terrain_limits.step_vel_deadzone;
        const double v_center = target_vel;
        const double relu_vstep = positive_part(std::abs(v_act - v_center) - deadzone);

        if (dir_norm_sq > 1e-10) {
            cost += 0.5 * std::pow(terrain_w.step_vel_weight * std::sqrt(dir_norm_sq) * relu_vstep, 2);
        }

        if (active_step_mode->step_entry_u.has_value()) {
            const double entry_u = std::clamp(*active_step_mode->step_entry_u, 0.0, 1.0);
            const double path_u = std::clamp(uc, 0.0, 1.0);
            if (path_u < entry_u) {
                const double d = quadratic_bspline_arc_length(ref_cps, path_u, entry_u);
                const double a_guide = std::max(follow.terrain_limits.step_reachability_guide_acc, REACHABILITY_EPS);
                const double v_min = target_vel - deadzone;
                const double v_max = target_vel + deadzone;
                const double r_lo = std::sqrt(std::max(REACHABILITY_EPS, std::max(0.0, v_act) * std::max(0.0, v_act) + 2.0 * a_guide * d));
                const double r_hi = std::sqrt(std::max(REACHABILITY_EPS, v_act * v_act - 2.0 * a_guide * d));

                cost += 0.5 * std::pow(std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * positive_part(v_min - r_lo), 2);
                cost += 0.5 * std::pow(std::sqrt(std::max(terrain_w.step_reachability_hi, 0.0)) * positive_part(r_hi - v_max), 2);
            }
        }
    }

    if (p.energy.enable) {
        const auto pwr = predict_power_eval_vw(p.power_model, v_act, w_act);
        const double thr = std::max(p.energy.threshold, 1.0);
        const double excess = (pwr.value - rfr_pwr_limit) / thr;
        cost += 0.5 * std::pow(p.energy.weight * positive_part(excess), 2);
    }

    const double uc_clamped = std::clamp(uc, 0.0, 1.0);
    const double s_remaining = quadratic_bspline_arc_length(ref_cps, uc_clamped, 1.0);

    if (uc < 1.0) {
        cost += (p.follow.tracking_weights.q_u / MPC_HORIZON) * s_remaining;
    }

    const auto& terminal_w = follow.terminal_weights;
    const double v_allowed = brake_speed_limit(s_remaining, terminal_w);
    cost += 0.5 * std::pow(terminal_w.q_v_final * positive_part(v_act - v_allowed), 2);
    return cost;
}

AdvanceUProgressEval
advance_u_progress_extrapolated_with_jacobian(double u_cur, const StateVec& x, const std::vector<Eigen::Vector2d>& ref_cps) {
    AdvanceUProgressEval out {};
    out.du_next_dx.setZero();

    const double uc = clamp_path_u_extrapolated(u_cur);
    const double duc_dpathu = clamp_derivative_piecewise(x(ix::PATH_U), PATH_U_EXTRAP_MIN, PATH_U_EXTRAP_MAX);

    Eigen::Vector2d pr, d1, d2;
    eval_quadratic_bspline2_extrapolated(ref_cps, uc, &pr, &d1, &d2);

    const double d1_norm2 = d1.squaredNorm();
    const double d1_norm = std::sqrt(d1_norm2 + 0.01);
    const double dsdu = d1_norm + 1e-6;
    const double inv_dsdu = 1.0 / dsdu;

    const double cross12 = d1.x() * d2.y() - d1.y() * d2.x();
    const double kappa = cross12 / (dsdu * dsdu * dsdu);

    const double thetar = std::atan2(d1.y(), d1.x());
    const double sin_r = std::sin(thetar);
    const double cos_r = std::cos(thetar);
    const double dtheta_du = cross12 / std::max(d1_norm2, 1e-12);

    const double ex = x(ix::X) - pr.x();
    const double ey_w = x(ix::Y) - pr.y();
    const double ey = -ex * sin_r + ey_w * cos_r;

    const double dey_dpx = -sin_r;
    const double dey_dpy = cos_r;
    const double dey_du = sin_r * d1.x() - cos_r * d1.y() - dtheta_du * (ex * cos_r + ey_w * sin_r);

    const double etheta = wrap_pi(x(ix::THETA) - thetar);
    const double cos_e = std::cos(etheta);
    const double sin_e = std::sin(etheta);

    const double num = x(ix::V) * cos_e;
    const double dnum_dv = cos_e;
    const double dnum_dtheta = -x(ix::V) * sin_e;
    const double dnum_du = x(ix::V) * (-sin_e) * (-dtheta_du);

    const double denom_raw = 1.0 - kappa * ey;
    // Frenet 进度公式 ds/dt = v*cos(e) / (1 - κ*ey) 仅在 denom_raw > 0 时物理有效。
    // 当 1 - κ*ey <= 0（车在急弯内侧且横向误差过大），Frenet 框架退化：
    //   - 使用 copysign 保留负号会导致 v>0 时 ds/dt<0（正速度倒推进度），
    //     FDDP 反向传播会误导求解器选择负速度作为"增加 u"的手段。
    //   - 修复：denom 永远取正模长，退化区退化为 |1-κ*ey| 的正则化形式。
    constexpr double DENOM_EPS = 0.1;
    const double denom = std::sqrt(denom_raw * denom_raw + DENOM_EPS * DENOM_EPS);
    const double denom_grad = denom_raw / denom;

    const double ddenom_dpx = -kappa * dey_dpx * denom_grad;
    const double ddenom_dpy = -kappa * dey_dpy * denom_grad;
    const double ddenom_du = -kappa * dey_du * denom_grad;

    const double inv_denom = 1.0 / denom;
    const double inv_denom2 = inv_denom * inv_denom;
    const double dsdt = num * inv_denom;

    const double ddsdt_dpx = -num * ddenom_dpx * inv_denom2;
    const double ddsdt_dpy = -num * ddenom_dpy * inv_denom2;
    const double ddsdt_dtheta = dnum_dtheta * inv_denom;
    const double ddsdt_dv = dnum_dv * inv_denom;
    const double ddsdt_du = (dnum_du * denom - num * ddenom_du) * inv_denom2;

    const double ddsdu_du = (d1_norm > 1e-12) ? (d1.dot(d2) / d1_norm) : 0.0;
    const double d_inv_dsdu_dpathu = -ddsdu_du * duc_dpathu / (dsdu * dsdu);

    out.u_next_extrap = uc + MPC_DT * dsdt * inv_dsdu;

    out.du_next_dx(ix::X) = MPC_DT * ddsdt_dpx * inv_dsdu;
    out.du_next_dx(ix::Y) = MPC_DT * ddsdt_dpy * inv_dsdu;
    out.du_next_dx(ix::THETA) = MPC_DT * ddsdt_dtheta * inv_dsdu;
    out.du_next_dx(ix::V) = MPC_DT * ddsdt_dv * inv_dsdu;

    const double ddsdt_dpathu = ddsdt_du * duc_dpathu;
    out.du_next_dx(ix::PATH_U) = duc_dpathu + MPC_DT * (ddsdt_dpathu * inv_dsdu + dsdt * d_inv_dsdu_dpathu);

    return out;
}

double advance_u_progress_extrapolated(double u_cur, const StateVec& x, const std::vector<Eigen::Vector2d>& ref_cps) {
    return advance_u_progress_extrapolated_with_jacobian(u_cur, x, ref_cps).u_next_extrap;
}

/// 更新 Frenet 进度
double advance_u_progress(double u_cur, const StateVec& x, const std::vector<Eigen::Vector2d>& ref_cps) {
    return clamp_path_u_extrapolated(advance_u_progress_extrapolated(u_cur, x, ref_cps));
}

} // anonymous namespace

template<int Horizon>
const CostMapGridView& FollowProblemT<Horizon>::cost_grid_for_step(int k) const {
    if (step_cost_grids_.size() <= 1) return step_cost_grids_[0];
    int idx = static_cast<int>(static_cast<double>(k) * MPC_DT / prediction_dt_);
    return step_cost_grids_[static_cast<size_t>(std::min(idx, static_cast<int>(step_cost_grids_.size()) - 1))];
}

template<int Horizon>
std::optional<RolloutLethalObstacleInfo> FollowProblemT<Horizon>::detect_lethal_obstacle(int state_index, const StateVec& x) const {
    const auto& safety = p_.follow.rollout_safety;
    if (!safety.enable_lethal_obstacle_check) {
        return std::nullopt;
    }

    const auto sample = eval_cost_bilinear(
        masked_global_grid_,
        cost_info_,
        x(ix::X),
        x(ix::Y)
    );
    if (sample.value + COST_EPS < safety.lethal_obstacle_threshold) {
        return std::nullopt;
    }

    return RolloutLethalObstacleInfo {
        .state_index = state_index,
        .position_map = Eigen::Vector2d(x(ix::X), x(ix::Y)),
        .sampled_cost = sample.value,
    };
}

template<int Horizon>
double FollowProblemT<Horizon>::running_cost(int k, const StateVec& x, const ControlVec& u) const {
    return running_cost_value_only(k, x, u);
}

template<int Horizon>
double FollowProblemT<Horizon>::running_cost_value_only(int k, const StateVec& x, const ControlVec& u) const {
    return follow_running_cost_value_only_impl(
        x, u, ref_cps_, p_,
        cost_grid_for_step(k), cost_info_, dir_grid_, dir_info_,
        rfr_pwr_limit_, active_step_mode_
    );
}

template<int Horizon>
void FollowProblemT<Horizon>::running_cost_derivatives(
    int k,
    const StateVec& x,
    const ControlVec& u,
    StateVec& lx,
    ControlVec& lu,
    MatXX& lxx,
    Eigen::Matrix<double, MPC_NU, MPC_NX>& lux,
    Eigen::Matrix<double, MPC_NU, MPC_NU>& luu
) const {
    const auto& cg = cost_grid_for_step(k);
    const auto lin = follow_residual_linearized_impl(
        x, u, ref_cps_, p_, cg, cost_info_, dir_grid_, dir_info_,
        rfr_pwr_limit_, active_step_mode_
    );

    lx = lin.jx.transpose() * lin.r;
    lu = lin.ju.transpose() * lin.r;

    {
        const double uc_raw = x(ix::PATH_U);
        const double uc = clamp_path_u_extrapolated(uc_raw);
        if (uc < 1.0) {
            const double duc_dpathu = clamp_derivative_piecewise(uc_raw, PATH_U_EXTRAP_MIN, PATH_U_EXTRAP_MAX);
            Eigen::Vector2d d1;
            eval_quadratic_bspline2_extrapolated(ref_cps_, uc, nullptr, &d1, nullptr);
            const double ds_du = d1.norm();
            lx(ix::PATH_U) -= (p_.follow.tracking_weights.q_u / MPC_HORIZON) * ds_du * duc_dpathu;
        }
    }

    lxx = (lin.jx.transpose() * lin.jx).eval();
    lux = (lin.ju.transpose() * lin.jx).eval();
    luu = (lin.ju.transpose() * lin.ju).eval();
    lxx = (lxx + lxx.transpose()).eval() * 0.5;
    luu = (luu + luu.transpose()).eval() * 0.5;
    for (int i = 0; i < MPC_NU; ++i) {
        luu(i, i) = std::max(luu(i, i), 1e-8);
    }
}

template<int Horizon>
double FollowProblemT<Horizon>::terminal_cost(const StateVec& x) const {
    (void)x;
    return 0.0;
}

template<int Horizon>
void FollowProblemT<Horizon>::terminal_cost_derivatives(const StateVec& x, StateVec& lfx, MatXX& lfxx) const {
    (void)x;
    lfx.setZero();
    lfxx.setZero();
}

template class FollowProblemT<MPC_HORIZON>;

// ════════════════════════════════════════════════════════════════
//  StopProblem
// ════════════════════════════════════════════════════════════════

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
    return mpc_dynamics(x, u, model_);
}

void StopProblem::dynamics_jacobians(int, const StateVec& x, const ControlVec& u, MatXX& dfx, MatXU& dfu) const {
    mpc_dynamics_jacobians(x, u, model_, dfx, dfu);
}

ControlVec StopProblem::u_lower() const {
    return ControlVec(p_.stop.command_bounds.vel_min, p_.stop.command_bounds.omega_min);
}

ControlVec StopProblem::u_upper() const {
    return ControlVec(p_.stop.command_bounds.vel_max, p_.stop.command_bounds.omega_max);
}

namespace {

constexpr int STOP_RESIDUAL_DIM = 11;
using StopResidualVec = Eigen::Matrix<double, STOP_RESIDUAL_DIM, 1>;

StopResidualVec stop_residual_impl(
    const StateVec& x,
    const ControlVec& u,
    const MPCParams& p,
    const CostMapGridView& cg,
    const GridInfo& ci,
    double rfr_pwr_limit
) {
    const auto& stop = p.stop;
    const auto& motion_lim = stop.motion_constraints;
    const auto& command_w = stop.command_weights;
    const auto& motion_w = stop.motion_constraint_weights;
    const auto& env_w = stop.environment_weights;

    StopResidualVec r = StopResidualVec::Zero();
    const double px = x(ix::X), py = x(ix::Y);
    const double v_act = x(ix::V), w_act = x(ix::W);
    const double v_cmd = u(0), w_cmd = u(1);
    const double dv_cmd = v_cmd - x(ix::DV);
    const double dw_cmd = w_cmd - x(ix::DW);

    const auto cs = eval_cost_bilinear(cg, ci, px, py);
    const double a_lat = std::abs(v_act * w_act);

    const double dv_lim = motion_lim.acc_max * MPC_DT;
    const double dw_lim = motion_lim.alpha_max * MPC_DT;
    r(0) = command_w.q_v * v_cmd;
    r(1) = command_w.q_omega * w_cmd;
    r(2) = command_w.r_dv * dv_cmd;
    r(3) = command_w.r_domega * dw_cmd;
    r(4) = motion_w.acc_limit * positive_part(std::abs(dv_cmd) - dv_lim);
    r(5) = motion_w.alpha_limit * positive_part(std::abs(dw_cmd) - dw_lim);
    r(6) = motion_w.lat_acc * positive_part(a_lat - motion_lim.a_lat_max);
    r(7) = env_w.obstacle * cs.value / 255.0;
    r(8) = motion_w.acc_limit * std::abs(v_act);
    r(9) = motion_w.alpha_limit * std::abs(w_act);

    if (p.energy.enable) {
        const double pwr = predict_power(p.power_model, v_act, w_act, 0.0, 0.0);
        const double thr = std::max(p.energy.threshold, 1.0);
        const double excess = (pwr - rfr_pwr_limit) / thr;
        r(10) = p.energy.weight * positive_part(excess);
    }

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

} // anonymous namespace

double StopProblem::running_cost(int k, const StateVec& x, const ControlVec& u) const {
    (void)k;
    return residual_cost(stop_residual_impl(x, u, p_, cost_grid_, cost_info_, rfr_pwr_limit_));
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

// ════════════════════════════════════════════════════════════════
//  HoldProblem
// ════════════════════════════════════════════════════════════════

HoldProblem::HoldProblem(
    const Eigen::Vector2d& goal_map,
    const MPCParams& params,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info,
    double schedule_rho,
    double remaining_energy,
    double rfr_pwr_limit
):
    goal_(goal_map),
    p_(params),
    cost_grid_(cost_grid),
    cost_info_(cost_info),
    model_(build_lpv_discrete_model(params.kinematic_model, schedule_rho)),
    remaining_energy_(remaining_energy),
    rfr_pwr_limit_(rfr_pwr_limit) {}

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

namespace {

constexpr int HOLD_RESIDUAL_DIM = 12;
using HoldResidualVec = Eigen::Matrix<double, HOLD_RESIDUAL_DIM, 1>;

HoldResidualVec hold_residual_impl(
    const StateVec& x,
    const ControlVec& u,
    const Eigen::Vector2d& goal,
    const MPCParams& p,
    const CostMapGridView& cg,
    const GridInfo& ci,
    double rfr_pwr_limit
) {
    const auto& hold = p.hold;
    const auto& goal_w = hold.goal_weights;
    const auto& command_w = hold.command_weights;
    const auto& motion_w = hold.motion_constraint_weights;
    const auto& env_w = hold.environment_weights;
    const auto& motion_lim = hold.motion_constraints;

    HoldResidualVec r = HoldResidualVec::Zero();
    const double px = x(ix::X), py = x(ix::Y), theta = x(ix::THETA);
    const double v_act = x(ix::V), w_act = x(ix::W);
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
    const double a_lat = std::abs(v_act * w_act);

    const double dv_lim = motion_lim.acc_max * MPC_DT;
    const double dw_lim = motion_lim.alpha_max * MPC_DT;
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

    if (p.energy.enable) {
        const double pwr = predict_power(p.power_model, v_act, w_act, 0.0, 0.0);
        const double thr = std::max(p.energy.threshold, 1.0);
        const double excess = (pwr - rfr_pwr_limit) / thr;
        r(11) = p.energy.weight * positive_part(excess);
    }

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

} // anonymous namespace

double HoldProblem::running_cost(int k, const StateVec& x, const ControlVec& u) const {
    (void)k;
    return residual_cost(
        hold_residual_impl(x, u, goal_, p_, cost_grid_, cost_info_, rfr_pwr_limit_)
    );
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
        return hold_residual_impl(xv, uv, goal_, p_, cost_grid_, cost_info_, rfr_pwr_limit_);
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

// ════════════════════════════════════════════════════════════════
//  MPCSolver
// ════════════════════════════════════════════════════════════════

namespace {

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
void zero_solver_controls(SolverT& solver) {
    for (size_t k = 0; k < SolverT::N; ++k) {
        solver.us[k].setZero();
    }
}

template<typename SolverT>
void fill_solver_controls(SolverT& solver, const ControlVec& u) {
    for (size_t k = 0; k < SolverT::N; ++k) {
        solver.us[k] = u;
    }
}

template<typename SolverT, typename ProblemT>
void clamp_solver_controls(SolverT& solver, const ProblemT& prob) {
    const auto u_lo = prob.u_lower();
    const auto u_hi = prob.u_upper();
    for (size_t k = 0; k < SolverT::N; ++k) {
        solver.us[k] = solver.us[k].cwiseMax(u_lo).cwiseMin(u_hi);
    }
}

template<typename SolverT, typename ProblemT>
void initialize_primal_trajectory(SolverT& solver, const ProblemT& prob, const StateVec& x0, bool use_warm_start) {
    if (use_warm_start) {
        shift_warm_start(solver);
    } else {
        zero_solver_controls(solver);
    }
    clamp_solver_controls(solver, prob);
    rollout_solver_states(solver, prob, x0);
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

} // anonymous namespace

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
    return x0;
}

std::expected<MPCSolver::FollowSolveResult, std::string> MPCSolver::solve_follow(
    const SplineD& global_path,
    const Eigen::Vector3d& chassis_pose_map,
    const ChassisMotionState& chassis_state,
    const CostMap& cost_map,
    const CostMap& masked_global_map,
    const std::vector<const CostMap*>& per_step_cost_maps,
    double prediction_dt,
    const DirectionMap& direction_map,
    std::optional<ActiveStepMode> active_step_mode
) {
    const auto& ref_cps = global_path.getControlPoints();
    const bool path_changed = !same_cps(prev_ref_control_points_, ref_cps);
    const double projection_hint = path_changed ? 0.0 : std::clamp(last_u_, 0.0, 1.0);
    const double u0 = project_to_spline_u_extrapolated(
        global_path,
        chassis_pose_map.head<2>(),
        projection_hint,
        params_.follow.projection.proj_num_samples,
        params_.follow.projection.proj_search_window,
        params_.follow.projection.local_search_lazy_distance,
        PATH_U_EXTRAP_MIN,
        PATH_U_EXTRAP_MAX
    );
    if (path_changed) {
        follow_warm_ = false;
    }
    last_u_ = u0;

    const auto& follow_mode_profile = select_follow_mode_profile(params_.follow, active_step_mode);
    const Eigen::Vector2d cmd0(
        clamp_prev_cmd(
            last_cmd_.x(),
            chassis_state.velocity,
            params_.follow.start_command.vel_cmd_act_gap_max,
            follow_mode_profile.motion_constraints.acc_max,
            MPC_DT
        ),
        clamp_prev_cmd(
            last_cmd_.y(),
            chassis_state.omega,
            params_.follow.start_command.omega_cmd_act_gap_max,
            follow_mode_profile.motion_constraints.alpha_max,
            MPC_DT
        )
    );
    const double schedule_rho = select_follow_schedule_rho(
        chassis_state,
        params_.kinematic_model
    );

    prev_ref_control_points_ = ref_cps;

    // ── 构建每个预测步的 cost grid view（复用缓存，避免重复分配） ──
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
        ref_cps, params_, step_cost_grids_cache_, ci, masked_global_grid, pred_dt, schedule_rho,
        dg, di, remaining_energy_, rfr_pwr_limit_, active_step_mode
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
    clamp_solver_controls(follow_solver_, problem);

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
    initialize_primal_trajectory(stop_solver_, prob, x0, stop_warm_);

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
    initialize_primal_trajectory(hold_solver_, prob, x0, hold_warm_);

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

} // namespace path_follower
