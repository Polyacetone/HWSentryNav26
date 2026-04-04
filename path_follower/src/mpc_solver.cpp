#include <path_follower/mpc_solver.hpp>
#include <algorithm>
#include <array>
#include <cmath>

namespace path_follower {

// ════════════════════════════════════════════════════════════════
//  工具函数
// ════════════════════════════════════════════════════════════════

namespace {

inline double sq(double x) {
    return x * x;
}

inline double softplus(double x) {
    if (x > 20.0) return x;
    if (x < -20.0) return std::exp(x);
    return std::log(1.0 + std::exp(x));
}

Eigen::Vector2d apply_goal_deadzone(const Eigen::Vector2d& delta, double deadzone) {
    if (deadzone <= 0.0) return delta;

    const double dist = delta.norm();
    if (dist <= 0.0) return Eigen::Vector2d::Zero();

    const double mag = softplus(dist - deadzone) - softplus(0.0);
    if (mag <= 0.0) {
        return Eigen::Vector2d::Zero();
    }
    return delta * (mag / dist);
}

inline double sabs(double x) {
    return std::sqrt(x * x + PWR_EPS2);
}

inline double smooth_sgn(double x, double eps) {
    return std::tanh(x / std::max(eps, 1e-6));
}

inline double smooth_sgn_deriv(double x, double eps) {
    const double s = smooth_sgn(x, eps);
    return (1.0 - s * s) / std::max(eps, 1e-6);
}

inline double wrap_pi(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

inline double relu(double x) {
    // replace hard ReLU with a softplus-based deadzone so costs grow
    // smoothly. subtract the baseline softplus(0) so relu(0) == 0.
    double y = softplus(x) - softplus(0.0);
    return (y > 0.0) ? y : 0.0;
}

inline double predict_power(double v, double w, double a, double alpha) {
    return PWR_C[0] + PWR_C[1] * v * a + PWR_C[2] * w * alpha + PWR_C[3] * a * a + PWR_C[4] * alpha * alpha
        + PWR_C[5] * sabs(v) + PWR_C[6] * sabs(w) + PWR_C[7] * v * v + PWR_C[8] * w * w + PWR_C[9] * sabs(a)
        + PWR_C[10] * sabs(alpha) + PWR_C[11] * sabs(v * w);
}

inline double clamp_prev_cmd(double cmd_prev, double status, double cmd_act_diff_max, double rate_max, double dt) {
    return std::max(
        std::min(cmd_prev, status + cmd_act_diff_max - rate_max * dt),
        status - cmd_act_diff_max + rate_max * dt
    );
}

double advance_u_progress(double u_cur, const StateVec& x, const std::vector<Eigen::Vector2d>& ref_cps);

// ─── B-spline 求值 ───

inline void eval_bspline2(
    const std::vector<Eigen::Vector2d>& cps,
    double u_in,
    Eigen::Vector2d* p,
    Eigen::Vector2d* d1,
    Eigen::Vector2d* d2
) {
    const int n = static_cast<int>(cps.size());
    if (n < 3) {
        if (p) *p = Eigen::Vector2d::Zero();
        if (d1) *d1 = Eigen::Vector2d::Zero();
        if (d2) *d2 = Eigen::Vector2d::Zero();
        return;
    }
    const double scale = static_cast<double>(n - 2);
    const double u = std::clamp(u_in, 0.0, 1.0);
    const double bx = u * scale;
    const int xi = std::clamp(static_cast<int>(std::floor(bx)), 0, n - 3);
    const double t = bx - static_cast<double>(xi);
    const double omt = 1.0 - t;
    const auto& p0 = cps[static_cast<size_t>(xi)];
    const auto& p1 = cps[static_cast<size_t>(xi + 1)];
    const auto& p2 = cps[static_cast<size_t>(xi + 2)];
    if (p) *p = 0.5 * omt * omt * p0 + 0.5 * (-2 * t * t + 2 * t + 1) * p1 + 0.5 * t * t * p2;
    if (d1) *d1 = (-omt * p0 + (-2 * t + 1) * p1 + t * p2) * scale;
    if (d2) *d2 = (p0 - 2 * p1 + p2) * (scale * scale);
}

inline double estimate_arclength(const std::vector<Eigen::Vector2d>& cps, double u_in, int ns) {
    const double u = std::clamp(u_in, 0.0, 1.0);
    const double rest = 1.0 - u;
    if (ns <= 1) {
        Eigen::Vector2d d1;
        eval_bspline2(cps, u, nullptr, &d1, nullptr);
        return rest * std::sqrt(d1.squaredNorm() + 1e-12);
    }
    double len = 0.0, u_prev = u;
    Eigen::Vector2d d1;
    eval_bspline2(cps, u, nullptr, &d1, nullptr);
    double dsdu_prev = std::sqrt(d1.squaredNorm() + 1e-12);
    for (int i = 1; i <= ns; i++) {
        const double ui = u + rest * (static_cast<double>(i) / static_cast<double>(ns));
        eval_bspline2(cps, ui, nullptr, &d1, nullptr);
        const double dsdu = std::sqrt(d1.squaredNorm() + 1e-12);
        len += (dsdu_prev + dsdu) * 0.5 * (ui - u_prev);
        u_prev = ui;
        dsdu_prev = dsdu;
    }
    return len;
}

inline bool same_cps(const std::vector<Eigen::Vector2d>& a, const std::vector<Eigen::Vector2d>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].x() != b[i].x() || a[i].y() != b[i].y()) {
            return false;
        }
    }
    return true;
}

inline ArclengthTable build_arclength_table(const std::vector<Eigen::Vector2d>& cps, int ns) {
    ArclengthTable t {};
    for (int i = 0; i <= ARCLENGTH_TABLE_SIZE; i++) {
        const double u = static_cast<double>(i) / static_cast<double>(ARCLENGTH_TABLE_SIZE);
        t[static_cast<size_t>(i)] = estimate_arclength(cps, u, ns);
    }
    return t;
}

inline double lookup_arclength(const ArclengthTable& table, double u) {
    u = std::clamp(u, 0.0, 1.0);
    const double idx = u * static_cast<double>(ARCLENGTH_TABLE_SIZE);
    const int i0 = std::min(static_cast<int>(std::floor(idx)), ARCLENGTH_TABLE_SIZE - 1);
    const double t = idx - static_cast<double>(i0);
    return (1.0 - t) * table[static_cast<size_t>(i0)] + t * table[static_cast<size_t>(i0 + 1)];
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
    constexpr double eps = 1e-5;

    const ResidualVec r0 = residual_fn(x, u);
    Eigen::Matrix<double, NR, MPC_NX> jx;
    Eigen::Matrix<double, NR, MPC_NU> ju;

    for (int i = 0; i < MPC_NX; ++i) {
        StateVec xp = x;
        xp(i) += eps;
        StateVec xm = x;
        xm(i) -= eps;
        jx.col(i) = (residual_fn(xp, u) - residual_fn(xm, u)) / (2.0 * eps);
    }
    for (int i = 0; i < MPC_NU; ++i) {
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
    constexpr double eps = 1e-5;

    const ResidualVec r0 = residual_fn(x);
    Eigen::Matrix<double, NR, MPC_NX> jx;

    for (int i = 0; i < MPC_NX; ++i) {
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
//  代价地图 Bicubic 采样（Catmull-Rom, C¹ 连续梯度）
// ════════════════════════════════════════════════════════════════

namespace {

    // Catmull-Rom 基函数及其导数
    inline void catmull_rom_weights(double t, double w[4], double dw[4]) {
        const double t2 = t * t, t3 = t2 * t;
        w[0] = -0.5 * t3 + t2 - 0.5 * t;
        w[1] = 1.5 * t3 - 2.5 * t2 + 1.0;
        w[2] = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
        w[3] = 0.5 * t3 - 0.5 * t2;
        dw[0] = -1.5 * t2 + 2.0 * t - 0.5;
        dw[1] = 4.5 * t2 - 5.0 * t;
        dw[2] = -4.5 * t2 + 4.0 * t + 0.5;
        dw[3] = 1.5 * t2 - t;
    }

} // anonymous namespace

CostSample eval_cost_bicubic(const CostMapGridView& grid, const GridInfo& info, double x_map, double y_map) {
    CostSample s {255.0, 0.0, 0.0};
    if (!std::isfinite(x_map) || !std::isfinite(y_map)) return s;
    if (info.width < 4 || info.height < 4) return s;

    const double gx = (x_map - info.origin_x) * info.inv_resolution;
    const double gy = (y_map - info.origin_y) * info.inv_resolution;
    if (gx < 1.0 || gy < 1.0 || gx >= info.width - 2 || gy >= info.height - 2) return s;

    const int ix0 = static_cast<int>(std::floor(gx));
    const int iy0 = static_cast<int>(std::floor(gy));
    const double tx = gx - ix0, ty = gy - iy0;

    double wx[4], dwx[4], wy[4], dwy[4];
    catmull_rom_weights(tx, wx, dwx);
    catmull_rom_weights(ty, wy, dwy);

    double val = 0.0, dvdgx = 0.0, dvdgy = 0.0;
    for (int j = -1; j <= 2; ++j) {
        for (int i = -1; i <= 2; ++i) {
            const double f = grid.value_at_clamped(iy0 + j, ix0 + i);
            val += wx[i + 1] * wy[j + 1] * f;
            dvdgx += dwx[i + 1] * wy[j + 1] * f;
            dvdgy += wx[i + 1] * dwy[j + 1] * f;
        }
    }

    s.value = val;
    s.dx = dvdgx * info.inv_resolution;
    s.dy = dvdgy * info.inv_resolution;
    return s;
}

DirSample eval_dir_bicubic(const DirectionMapGridView& grid, const GridInfo& info, double x_map, double y_map) {
    DirSample s {Eigen::Vector2d::Zero(), Eigen::Matrix2d::Zero()};
    if (!std::isfinite(x_map) || !std::isfinite(y_map)) return s;
    if (info.width < 4 || info.height < 4) return s;

    const double gx = (x_map - info.origin_x) * info.inv_resolution;
    const double gy = (y_map - info.origin_y) * info.inv_resolution;
    if (gx < 1.0 || gy < 1.0 || gx >= info.width - 2 || gy >= info.height - 2) return s;

    const int ix0 = static_cast<int>(std::floor(gx));
    const int iy0 = static_cast<int>(std::floor(gy));
    const double tx = gx - ix0, ty = gy - iy0;

    double wx[4], dwx[4], wy[4], dwy[4];
    catmull_rom_weights(tx, wx, dwx);
    catmull_rom_weights(ty, wy, dwy);

    Eigen::Vector2d val = Eigen::Vector2d::Zero();
    Eigen::Vector2d dvdgx = Eigen::Vector2d::Zero();
    Eigen::Vector2d dvdgy = Eigen::Vector2d::Zero();
    for (int j = -1; j <= 2; ++j) {
        for (int i = -1; i <= 2; ++i) {
            const auto f = grid.value_at_clamped(iy0 + j, ix0 + i);
            val += wx[i + 1] * wy[j + 1] * f;
            dvdgx += dwx[i + 1] * wy[j + 1] * f;
            dvdgy += wx[i + 1] * dwy[j + 1] * f;
        }
    }

    s.value = val;
    s.J.col(0) = dvdgx * info.inv_resolution;
    s.J.col(1) = dvdgy * info.inv_resolution;
    return s;
}

// ════════════════════════════════════════════════════════════════
//  共享动力学模型
// ════════════════════════════════════════════════════════════════

StateVec mpc_dynamics(const StateVec& x, const ControlVec& u) {
    const double theta = x(ix::THETA);
    const double xh = x(ix::XH), v = x(ix::V), w = x(ix::W);
    const double dv = x(ix::DV), dw = x(ix::DW);

    const double sv = smooth_sgn(v, SGN_EPS);
    const double sw = smooth_sgn(w, SGN_EPS);
    const double nl = CF1 * sv + CF2 * v * std::abs(w);

    const double xh1 = A00 * xh + A01 * v + A03 * dv + GNL_XH * nl;
    const double v1 = A10 * xh + A11 * v + A13 * dv + GNL_V * nl;
    const double w1 = A22 * w + A24 * dw - GAMMA_W * CF3 * sw;

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

void mpc_dynamics_jacobians(const StateVec& x, const ControlVec& /*u*/, MatXX& fx, MatXU& fu) {
    const double theta = x(ix::THETA);
    const double xh = x(ix::XH), v = x(ix::V), w = x(ix::W);
    const double dv = x(ix::DV), dw = x(ix::DW);
    const double dt = MPC_DT, h = dt * 0.5;

    const double sv = smooth_sgn(v, SGN_EPS);
    const double dsv = smooth_sgn_deriv(v, SGN_EPS);
    const double sw = smooth_sgn(w, SGN_EPS);
    const double dsw = smooth_sgn_deriv(w, SGN_EPS);
    const double absw = std::abs(w);
    const double dabsw = (w >= 0) ? 1.0 : -1.0;

    const double nl = CF1 * sv + CF2 * v * absw;
    const double dnl_dv = CF1 * dsv + CF2 * absw;
    const double dnl_dw = CF2 * v * dabsw;

    // Greybox next states
    const double v1 = A10 * xh + A11 * v + A13 * dv + GNL_V * nl;
    const double w1 = A22 * w + A24 * dw - GAMMA_W * CF3 * sw;
    const double theta1 = theta + (w + w1) * h;

    // Derivatives of greybox model
    const double dvn_dxh = A10;
    const double dvn_dv = A11 + GNL_V * dnl_dv;
    const double dvn_dw = GNL_V * dnl_dw;
    const double dvn_ddv = A13;

    const double dwn_dw = A22 - GAMMA_W * CF3 * dsw;
    const double dwn_ddw = A24;

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

    fx(ix::XH, ix::XH) = A00;
    fx(ix::XH, ix::V) = A01 + GNL_XH * dnl_dv;
    fx(ix::XH, ix::W) = GNL_XH * dnl_dw;
    fx(ix::XH, ix::DV) = A03;

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

FollowProblem::FollowProblem(
    const std::vector<Eigen::Vector2d>& ref_control_points,
    const MPCParams& params,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info,
    const DirectionMapGridView& dir_grid,
    const GridInfo& dir_info,
    const ArclengthTable& arclength_table,
    double remaining_energy,
    double rfr_pwr_limit
):
    ref_cps_(ref_control_points),
    p_(params),
    cost_grid_(cost_grid),
    cost_info_(cost_info),
    dir_grid_(dir_grid),
    dir_info_(dir_info),
    arc_table_(arclength_table),
    remaining_energy_(remaining_energy),
    rfr_pwr_limit_(rfr_pwr_limit) {}

StateVec FollowProblem::dynamics(int, const StateVec& x, const ControlVec& u) const {
    StateVec xn = mpc_dynamics(x, u);
    xn(ix::PATH_U) = advance_u_progress(x(ix::PATH_U), x, ref_cps_);
    return xn;
}

void FollowProblem::dynamics_jacobians(int, const StateVec& x, const ControlVec& u, MatXX& dfx, MatXU& dfu) const {
    mpc_dynamics_jacobians(x, u, dfx, dfu);

    constexpr double eps = 1e-5;
    for (int i = 0; i < MPC_NX; ++i) {
        StateVec xp = x;
        xp(i) += eps;
        StateVec xm = x;
        xm(i) -= eps;
        dfx(ix::PATH_U, i) =
            (advance_u_progress(xp(ix::PATH_U), xp, ref_cps_) - advance_u_progress(xm(ix::PATH_U), xm, ref_cps_))
            / (2.0 * eps);
    }
    dfu.row(ix::PATH_U).setZero();
}

ControlVec FollowProblem::u_lower() const {
    return ControlVec(p_.follow_limits.vel_min, p_.follow_limits.omega_min);
}

ControlVec FollowProblem::u_upper() const {
    return ControlVec(p_.follow_limits.vel_max, p_.follow_limits.omega_max);
}

namespace {

    constexpr int FOLLOW_RESIDUAL_DIM = 15;
    using FollowResidualVec = Eigen::Matrix<double, FOLLOW_RESIDUAL_DIM, 1>;

    FollowResidualVec follow_residual_impl(
        const StateVec& x,
        const ControlVec& u,
        const std::vector<Eigen::Vector2d>& ref_cps,
        const ArclengthTable& arc_table,
        const MPCParams& p,
        const CostMapGridView& cg,
        const GridInfo& ci,
        const DirectionMapGridView& dg,
        const GridInfo& di,
        double rfr_pwr_limit
    ) {
        const auto& w = p.follow_weights;
        const auto& lim = p.follow_limits;

        FollowResidualVec r = FollowResidualVec::Zero();
        const double px = x(ix::X), py = x(ix::Y), theta = x(ix::THETA);
        const double v_act = x(ix::V), w_act = x(ix::W);
        const double v_cmd = u(0), w_cmd = u(1);
        const double dv_cmd = v_cmd - x(ix::DV);
        const double dw_cmd = w_cmd - x(ix::DW);

        const double uc = std::clamp(x(ix::PATH_U), 0.0, 1.0);
        Eigen::Vector2d pr, d1, d2;
        eval_bspline2(ref_cps, uc, &pr, &d1, &d2);

        const double thetar = std::atan2(d1.y(), d1.x());
        const double ex = px - pr.x(), ey_w = py - pr.y();
        const double ey = -ex * std::sin(thetar) + ey_w * std::cos(thetar);
        const double etheta = wrap_pi(theta - thetar);

        const double s_remain = lookup_arclength(arc_table, uc);
        const auto cs = eval_cost_bicubic(cg, ci, px, py);
        const auto ds = eval_dir_bicubic(dg, di, px, py);
        const double dir_norm = std::sqrt(ds.value.squaredNorm() + 1e-10);
        const Eigen::Vector2d dir_unit = ds.value / dir_norm;
        const Eigen::Vector2d heading(std::cos(theta), std::sin(theta));
        const double cross = heading.x() * ds.value.y() - heading.y() * ds.value.x();
        const double cos_th = heading.dot(dir_unit);
        const double weight_up = (cos_th + 1.0) / 2.0;
        const double target_vel = weight_up * lim.vel_step_up + (1.0 - weight_up) * lim.vel_step_down;
        const double v_dec = std::sqrt(2.0 * lim.slow_down_deceleration * s_remain + sq(lim.slow_down_target_vel));
        const double a_lat = std::abs(v_act * w_act);

        const double dv_lim = lim.acc_max * MPC_DT;
        const double dw_lim = lim.alpha_max * MPC_DT;
        r(0) = w.q_y * ey;
        r(1) = w.q_theta * etheta;
        r(2) = w.q_u * (1.0 - uc);
        r(3) = w.r_v * v_cmd;
        r(4) = w.r_omega * w_cmd;
        r(5) = w.r_dv * dv_cmd;
        r(6) = w.r_domega * dw_cmd;
        r(7) = w.acc_limit * relu(std::abs(dv_cmd) - dv_lim);
        r(8) = w.alpha_limit * relu(std::abs(dw_cmd) - dw_lim);
        r(9) = w.lat_acc * relu(a_lat - lim.a_lat_max);
        r(10) = w.obstacle * cs.value / 255.0;
        r(11) = w.direction * std::abs(cross);
        r(12) = w.vel_on_step * dir_norm * std::abs(v_act - target_vel);
        r(13) = w.q_v_final * relu(v_act - v_dec);

        if (p.energy.enable) {
            const double pwr = predict_power(v_act, w_act, 0.0, 0.0);
            const double thr = std::max(p.energy.threshold, 1.0);
            const double beta = std::max(p.energy.softplus_beta, 1e-6);
            const double excess = (pwr - rfr_pwr_limit) / thr;
            const double sp = softplus(beta * excess) / beta - softplus(0.0) / beta;
            r(14) = p.energy.weight * std::max(0.0, sp);
        }

        return r;
    }

    /// 更新 Frenet 进度
    double advance_u_progress(double u_cur, const StateVec& x, const std::vector<Eigen::Vector2d>& ref_cps) {
        u_cur = std::clamp(u_cur, 0.0, 1.0);
        Eigen::Vector2d pr, d1, d2;
        eval_bspline2(ref_cps, u_cur, &pr, &d1, &d2);
        const double dsdu = std::sqrt(d1.squaredNorm()) + 1e-6;
        const double kappa = (d1.x() * d2.y() - d1.y() * d2.x()) / (dsdu * dsdu * dsdu);
        const double thetar = std::atan2(d1.y(), d1.x());
        const double ex = x(ix::X) - pr.x(), ey_w = x(ix::Y) - pr.y();
        const double ey = -ex * std::sin(thetar) + ey_w * std::cos(thetar);
        const double etheta = wrap_pi(x(ix::THETA) - thetar);
        double denom = 1.0 - kappa * ey;
        denom = (denom > 0 ? 1.0 : -1.0) * std::max(std::abs(denom), 0.1);
        const double dsdt = x(ix::V) * std::cos(etheta) / denom;
        return std::clamp(u_cur + (dsdt / dsdu) * MPC_DT, 0.0, 1.0);
    }

} // anonymous namespace

double FollowProblem::running_cost(int k, const StateVec& x, const ControlVec& u) const {
    (void)k;
    return residual_cost(follow_residual_impl(
        x,
        u,
        ref_cps_,
        arc_table_,
        p_,
        cost_grid_,
        cost_info_,
        dir_grid_,
        dir_info_,
        rfr_pwr_limit_
    ));
}

void FollowProblem::running_cost_derivatives(
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
        return follow_residual_impl(
            xv,
            uv,
            ref_cps_,
            arc_table_,
            p_,
            cost_grid_,
            cost_info_,
            dir_grid_,
            dir_info_,
            rfr_pwr_limit_
        );
    };
    gauss_newton_running_derivatives<FOLLOW_RESIDUAL_DIM>(residual_fn, x, u, lx, lu, lxx, lux, luu);
}

double FollowProblem::terminal_cost(const StateVec&) const {
    return 0.0;
}
void FollowProblem::terminal_cost_derivatives(const StateVec&, StateVec& lfx, MatXX& lfxx) const {
    lfx.setZero();
    lfxx.setZero();
}

// ════════════════════════════════════════════════════════════════
//  StopProblem
// ════════════════════════════════════════════════════════════════

StopProblem::StopProblem(
    const MPCParams& params,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info,
    const DirectionMapGridView& dir_grid,
    const GridInfo& dir_info,
    double remaining_energy,
    double rfr_pwr_limit
):
    p_(params),
    cost_grid_(cost_grid),
    cost_info_(cost_info),
    dir_grid_(dir_grid),
    dir_info_(dir_info),
    remaining_energy_(remaining_energy),
    rfr_pwr_limit_(rfr_pwr_limit) {}

StateVec StopProblem::dynamics(int, const StateVec& x, const ControlVec& u) const {
    return mpc_dynamics(x, u);
}

void StopProblem::dynamics_jacobians(int, const StateVec& x, const ControlVec& u, MatXX& dfx, MatXU& dfu) const {
    mpc_dynamics_jacobians(x, u, dfx, dfu);
}

ControlVec StopProblem::u_lower() const {
    return ControlVec(0.0, p_.stop_limits.omega_min);
}

ControlVec StopProblem::u_upper() const {
    return ControlVec(p_.stop_limits.vel_max, p_.stop_limits.omega_max);
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
    const DirectionMapGridView& dg,
    const GridInfo& di,
    double rfr_pwr_limit
) {
    const auto& w = p.stop_weights;
    const auto& lim = p.stop_limits;

    StopResidualVec r = StopResidualVec::Zero();
    const double px = x(ix::X), py = x(ix::Y), theta = x(ix::THETA);
    const double v_act = x(ix::V), w_act = x(ix::W);
    const double v_cmd = u(0), w_cmd = u(1);
    const double dv_cmd = v_cmd - x(ix::DV);
    const double dw_cmd = w_cmd - x(ix::DW);

    const auto cs = eval_cost_bicubic(cg, ci, px, py);
    const auto ds = eval_dir_bicubic(dg, di, px, py);
    const double dir_norm = std::sqrt(ds.value.squaredNorm() + 1e-10);
    const Eigen::Vector2d dir_unit = ds.value / dir_norm;
    const Eigen::Vector2d heading(std::cos(theta), std::sin(theta));
    const double cross = heading.x() * ds.value.y() - heading.y() * ds.value.x();
    const double cos_th = heading.dot(dir_unit);
    const double weight_up = (cos_th + 1.0) / 2.0;
    const double target_vel = weight_up * lim.vel_step_up + (1.0 - weight_up) * lim.vel_step_down;
    const double a_lat = std::abs(v_act * w_act);

    const double dv_lim = lim.acc_max * MPC_DT;
    const double dw_lim = lim.alpha_max * MPC_DT;
    r(0) = w.q_v * v_cmd;
    r(1) = w.q_omega * w_cmd;
    r(2) = w.r_dv * dv_cmd;
    r(3) = w.r_domega * dw_cmd;
    r(4) = w.acc_limit * relu(std::abs(dv_cmd) - dv_lim);
    r(5) = w.alpha_limit * relu(std::abs(dw_cmd) - dw_lim);
    r(6) = w.lat_acc * relu(a_lat - lim.a_lat_max);
    r(7) = w.obstacle * cs.value / 255.0;
    r(8) = w.direction * dir_norm * std::abs(cross);
    r(9) = w.vel_on_step * dir_norm * dir_norm * std::abs(v_act - target_vel);

    if (p.energy.enable) {
        const double pwr = predict_power(v_act, w_act, 0.0, 0.0);
        const double thr = std::max(p.energy.threshold, 1.0);
        const double beta = std::max(p.energy.softplus_beta, 1e-6);
        const double excess = (pwr - rfr_pwr_limit) / thr;
        const double sp = softplus(beta * excess) / beta - softplus(0.0) / beta;
        r(10) = p.energy.weight * std::max(0.0, sp);
    }

    return r;
}

constexpr int STOP_TERMINAL_RESIDUAL_DIM = 2;
using StopTerminalResidualVec = Eigen::Matrix<double, STOP_TERMINAL_RESIDUAL_DIM, 1>;

StopTerminalResidualVec stop_terminal_residual_impl(
    const StateVec& x,
    const MPCParams& p,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info,
    const DirectionMapGridView& dir_grid,
    const GridInfo& dir_info
) {
    StopTerminalResidualVec r = StopTerminalResidualVec::Zero();
    const auto& w = p.stop_weights;
    const auto cs = eval_cost_bicubic(cost_grid, cost_info, x(ix::X), x(ix::Y));
    const auto ds = eval_dir_bicubic(dir_grid, dir_info, x(ix::X), x(ix::Y));
    const double dir_norm = std::sqrt(ds.value.squaredNorm() + 1e-10);
    r(0) = w.obstacle_terminal * cs.value / 255.0;
    r(1) = w.step_terminal * dir_norm;
    return r;
}

} // anonymous namespace

double StopProblem::running_cost(int k, const StateVec& x, const ControlVec& u) const {
    (void)k;
    return residual_cost(stop_residual_impl(x, u, p_, cost_grid_, cost_info_, dir_grid_, dir_info_, rfr_pwr_limit_));
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
        return stop_residual_impl(xv, uv, p_, cost_grid_, cost_info_, dir_grid_, dir_info_, rfr_pwr_limit_);
    };
    gauss_newton_running_derivatives<STOP_RESIDUAL_DIM>(residual_fn, x, u, lx, lu, lxx, lux, luu);
}

double StopProblem::terminal_cost(const StateVec& x) const {
    return residual_cost(stop_terminal_residual_impl(x, p_, cost_grid_, cost_info_, dir_grid_, dir_info_));
}

void StopProblem::terminal_cost_derivatives(const StateVec& x, StateVec& lfx, MatXX& lfxx) const {
    auto residual_fn = [&](const StateVec& xv) {
        return stop_terminal_residual_impl(xv, p_, cost_grid_, cost_info_, dir_grid_, dir_info_);
    };
    gauss_newton_terminal_derivatives<STOP_TERMINAL_RESIDUAL_DIM>(residual_fn, x, lfx, lfxx);
}

// ════════════════════════════════════════════════════════════════
//  RecoveryProblem
// ════════════════════════════════════════════════════════════════

RecoveryProblem::RecoveryProblem(
    const Eigen::Vector2d& goal_map,
    const MPCParams& params,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info,
    const DirectionMapGridView& dir_grid,
    const GridInfo& dir_info,
    double remaining_energy,
    double rfr_pwr_limit
):
    goal_(goal_map),
    p_(params),
    cost_grid_(cost_grid),
    cost_info_(cost_info),
    dir_grid_(dir_grid),
    dir_info_(dir_info),
    remaining_energy_(remaining_energy),
    rfr_pwr_limit_(rfr_pwr_limit) {}

StateVec RecoveryProblem::dynamics(int, const StateVec& x, const ControlVec& u) const {
    return mpc_dynamics(x, u);
}

void RecoveryProblem::dynamics_jacobians(int, const StateVec& x, const ControlVec& u, MatXX& dfx, MatXU& dfu) const {
    mpc_dynamics_jacobians(x, u, dfx, dfu);
}

ControlVec RecoveryProblem::u_lower() const {
    return ControlVec(p_.recovery_limits.vel_min, p_.recovery_limits.omega_min);
}

ControlVec RecoveryProblem::u_upper() const {
    return ControlVec(p_.recovery_limits.vel_max, p_.recovery_limits.omega_max);
}

namespace {

constexpr int RECOVERY_RESIDUAL_DIM = 13;
using RecoveryResidualVec = Eigen::Matrix<double, RECOVERY_RESIDUAL_DIM, 1>;

RecoveryResidualVec recovery_residual_impl(
    const StateVec& x,
    const ControlVec& u,
    const Eigen::Vector2d& goal,
    const MPCParams& p,
    const CostMapGridView& cg,
    const GridInfo& ci,
    const DirectionMapGridView& dg,
    const GridInfo& di,
    double rfr_pwr_limit
) {
    const auto& w = p.recovery_weights;
    const auto& lim = p.recovery_limits;

    RecoveryResidualVec r = RecoveryResidualVec::Zero();
    const double px = x(ix::X), py = x(ix::Y), theta = x(ix::THETA);
    const double v_act = x(ix::V), w_act = x(ix::W);
    const double v_cmd = u(0), w_cmd = u(1);
    const double dv_cmd = v_cmd - x(ix::DV);
    const double dw_cmd = w_cmd - x(ix::DW);

    const Eigen::Vector2d goal_delta(px - goal.x(), py - goal.y());
    const double ddx = goal_delta.x(), ddy = goal_delta.y();
    const double desired_theta = std::atan2(goal.y() - py, goal.x() - px);
    const double heading_sin = std::sin(theta - desired_theta);

    const auto cs = eval_cost_bicubic(cg, ci, px, py);
    const auto ds = eval_dir_bicubic(dg, di, px, py);
    const double dir_norm = std::sqrt(ds.value.squaredNorm() + 1e-10);
    const double a_lat = std::abs(v_act * w_act);

    const double dv_lim = lim.acc_max * MPC_DT;
    const double dw_lim = lim.alpha_max * MPC_DT;
    r(0) = w.q_goal_xy * ddx;
    r(1) = w.q_goal_xy * ddy;
    r(2) = w.q_goal_theta * std::abs(heading_sin);
    r(3) = w.r_v * v_cmd;
    r(4) = w.r_omega * w_cmd;
    r(5) = w.r_dv * dv_cmd;
    r(6) = w.r_domega * dw_cmd;
    r(7) = w.acc_limit * relu(std::abs(dv_cmd) - dv_lim);
    r(8) = w.alpha_limit * relu(std::abs(dw_cmd) - dw_lim);
    r(9) = w.lat_acc * relu(a_lat - lim.a_lat_max);
    r(10) = w.obstacle * cs.value / 255.0;
    r(11) = w.step * dir_norm;

    if (p.energy.enable) {
        const double pwr = predict_power(v_act, w_act, 0.0, 0.0);
        const double thr = std::max(p.energy.threshold, 1.0);
        const double beta = std::max(p.energy.softplus_beta, 1e-6);
        const double excess = (pwr - rfr_pwr_limit) / thr;
        const double sp = softplus(beta * excess) / beta - softplus(0.0) / beta;
        r(12) = p.energy.weight * std::max(0.0, sp);
    }

    return r;
}

constexpr int RECOVERY_TERMINAL_RESIDUAL_DIM = 4;
using RecoveryTerminalResidualVec = Eigen::Matrix<double, RECOVERY_TERMINAL_RESIDUAL_DIM, 1>;

RecoveryTerminalResidualVec recovery_terminal_residual_impl(
    const StateVec& x,
    const Eigen::Vector2d& goal,
    const MPCParams& p,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info,
    const DirectionMapGridView& dir_grid,
    const GridInfo& dir_info
) {
    RecoveryTerminalResidualVec r = RecoveryTerminalResidualVec::Zero();
    const auto& w = p.recovery_weights;
    const double dxT = x(ix::X) - goal.x();
    const double dyT = x(ix::Y) - goal.y();
    const auto cs = eval_cost_bicubic(cost_grid, cost_info, x(ix::X), x(ix::Y));
    const auto ds = eval_dir_bicubic(dir_grid, dir_info, x(ix::X), x(ix::Y));
    const double dir_norm = std::sqrt(ds.value.squaredNorm() + 1e-10);
    r(0) = w.q_goal_xy_terminal * dxT;
    r(1) = w.q_goal_xy_terminal * dyT;
    r(2) = w.obstacle_terminal * cs.value / 255.0;
    r(3) = w.step_terminal * dir_norm;
    return r;
}

} // anonymous namespace

double RecoveryProblem::running_cost(int k, const StateVec& x, const ControlVec& u) const {
    (void)k;
    return residual_cost(
        recovery_residual_impl(x, u, goal_, p_, cost_grid_, cost_info_, dir_grid_, dir_info_, rfr_pwr_limit_)
    );
}

void RecoveryProblem::running_cost_derivatives(
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
        return recovery_residual_impl(xv, uv, goal_, p_, cost_grid_, cost_info_, dir_grid_, dir_info_, rfr_pwr_limit_);
    };
    gauss_newton_running_derivatives<RECOVERY_RESIDUAL_DIM>(residual_fn, x, u, lx, lu, lxx, lux, luu);
}

double RecoveryProblem::terminal_cost(const StateVec& x) const {
    return residual_cost(recovery_terminal_residual_impl(x, goal_, p_, cost_grid_, cost_info_, dir_grid_, dir_info_));
}

void RecoveryProblem::terminal_cost_derivatives(const StateVec& x, StateVec& lfx, MatXX& lfxx) const {
    auto residual_fn = [&](const StateVec& xv) {
        return recovery_terminal_residual_impl(xv, goal_, p_, cost_grid_, cost_info_, dir_grid_, dir_info_);
    };
    gauss_newton_terminal_derivatives<RECOVERY_TERMINAL_RESIDUAL_DIM>(residual_fn, x, lfx, lfxx);
}

// ════════════════════════════════════════════════════════════════
//  FixedProblem
// ════════════════════════════════════════════════════════════════

FixedProblem::FixedProblem(
    const Eigen::Vector2d& goal_map,
    const MPCParams& params,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info,
    const DirectionMapGridView& dir_grid,
    const GridInfo& dir_info,
    double remaining_energy,
    double rfr_pwr_limit
):
    goal_(goal_map),
    p_(params),
    cost_grid_(cost_grid),
    cost_info_(cost_info),
    dir_grid_(dir_grid),
    dir_info_(dir_info),
    remaining_energy_(remaining_energy),
    rfr_pwr_limit_(rfr_pwr_limit) {}

StateVec FixedProblem::dynamics(int, const StateVec& x, const ControlVec& u) const {
    return mpc_dynamics(x, u);
}

void FixedProblem::dynamics_jacobians(int, const StateVec& x, const ControlVec& u, MatXX& dfx, MatXU& dfu) const {
    mpc_dynamics_jacobians(x, u, dfx, dfu);
}

ControlVec FixedProblem::u_lower() const {
    return ControlVec(p_.fixed_limits.vel_min, p_.fixed_limits.omega_min);
}

ControlVec FixedProblem::u_upper() const {
    return ControlVec(p_.fixed_limits.vel_max, p_.fixed_limits.omega_max);
}

namespace {

constexpr int FIXED_RESIDUAL_DIM = 13;
using FixedResidualVec = Eigen::Matrix<double, FIXED_RESIDUAL_DIM, 1>;

FixedResidualVec fixed_residual_impl(
    const StateVec& x,
    const ControlVec& u,
    const Eigen::Vector2d& goal,
    const MPCParams& p,
    const CostMapGridView& cg,
    const GridInfo& ci,
    const DirectionMapGridView& dg,
    const GridInfo& di,
    double rfr_pwr_limit
) {
    const auto& w = p.fixed_weights;
    const auto& lim = p.fixed_limits;

    FixedResidualVec r = FixedResidualVec::Zero();
    const double px = x(ix::X), py = x(ix::Y), theta = x(ix::THETA);
    const double v_act = x(ix::V), w_act = x(ix::W);
    const double v_cmd = u(0), w_cmd = u(1);
    const double dv_cmd = v_cmd - x(ix::DV);
    const double dw_cmd = w_cmd - x(ix::DW);

    const Eigen::Vector2d goal_delta = apply_goal_deadzone(
        Eigen::Vector2d(px - goal.x(), py - goal.y()),
        w.goal_deadzone
    );
    const double ddx = goal_delta.x(), ddy = goal_delta.y();
    const double desired_theta = std::atan2(goal.y() - py, goal.x() - px);
    const double heading_sin = std::sin(theta - desired_theta);

    const auto cs = eval_cost_bicubic(cg, ci, px, py);
    const auto ds = eval_dir_bicubic(dg, di, px, py);
    const double dir_norm = std::sqrt(ds.value.squaredNorm() + 1e-10);
    const double a_lat = std::abs(v_act * w_act);

    const double dv_lim = lim.acc_max * MPC_DT;
    const double dw_lim = lim.alpha_max * MPC_DT;
    r(0) = w.q_goal_xy * ddx;
    r(1) = w.q_goal_xy * ddy;
    r(2) = w.q_goal_theta * std::abs(heading_sin);
    r(3) = w.r_v * v_cmd;
    r(4) = w.r_omega * w_cmd;
    r(5) = w.r_dv * dv_cmd;
    r(6) = w.r_domega * dw_cmd;
    r(7) = w.acc_limit * relu(std::abs(dv_cmd) - dv_lim);
    r(8) = w.alpha_limit * relu(std::abs(dw_cmd) - dw_lim);
    r(9) = w.lat_acc * relu(a_lat - lim.a_lat_max);
    r(10) = w.obstacle * cs.value / 255.0;
    r(11) = w.step * dir_norm;

    if (p.energy.enable) {
        const double pwr = predict_power(v_act, w_act, 0.0, 0.0);
        const double thr = std::max(p.energy.threshold, 1.0);
        const double beta = std::max(p.energy.softplus_beta, 1e-6);
        const double excess = (pwr - rfr_pwr_limit) / thr;
        const double sp = softplus(beta * excess) / beta - softplus(0.0) / beta;
        r(12) = p.energy.weight * std::max(0.0, sp);
    }

    return r;
}

constexpr int FIXED_TERMINAL_RESIDUAL_DIM = 4;
using FixedTerminalResidualVec = Eigen::Matrix<double, FIXED_TERMINAL_RESIDUAL_DIM, 1>;

FixedTerminalResidualVec fixed_terminal_residual_impl(
    const StateVec& x,
    const Eigen::Vector2d& goal,
    const MPCParams& p,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info,
    const DirectionMapGridView& dir_grid,
    const GridInfo& dir_info
) {
    FixedTerminalResidualVec r = FixedTerminalResidualVec::Zero();
    const auto& w = p.fixed_weights;
    const Eigen::Vector2d terminal_delta = apply_goal_deadzone(
        Eigen::Vector2d(x(ix::X) - goal.x(), x(ix::Y) - goal.y()),
        w.goal_deadzone
    );
    const auto cs = eval_cost_bicubic(cost_grid, cost_info, x(ix::X), x(ix::Y));
    const auto ds = eval_dir_bicubic(dir_grid, dir_info, x(ix::X), x(ix::Y));
    const double dir_norm = std::sqrt(ds.value.squaredNorm() + 1e-10);
    r(0) = w.q_goal_xy_terminal * terminal_delta.x();
    r(1) = w.q_goal_xy_terminal * terminal_delta.y();
    r(2) = w.obstacle_terminal * cs.value / 255.0;
    r(3) = w.step_terminal * dir_norm;
    return r;
}

} // anonymous namespace

double FixedProblem::running_cost(int k, const StateVec& x, const ControlVec& u) const {
    (void)k;
    return residual_cost(
        fixed_residual_impl(x, u, goal_, p_, cost_grid_, cost_info_, dir_grid_, dir_info_, rfr_pwr_limit_)
    );
}

void FixedProblem::running_cost_derivatives(
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
        return fixed_residual_impl(xv, uv, goal_, p_, cost_grid_, cost_info_, dir_grid_, dir_info_, rfr_pwr_limit_);
    };
    gauss_newton_running_derivatives<FIXED_RESIDUAL_DIM>(residual_fn, x, u, lx, lu, lxx, lux, luu);
}

double FixedProblem::terminal_cost(const StateVec& x) const {
    return residual_cost(fixed_terminal_residual_impl(x, goal_, p_, cost_grid_, cost_info_, dir_grid_, dir_info_));
}

void FixedProblem::terminal_cost_derivatives(const StateVec& x, StateVec& lfx, MatXX& lfxx) const {
    auto residual_fn = [&](const StateVec& xv) {
        return fixed_terminal_residual_impl(xv, goal_, p_, cost_grid_, cost_info_, dir_grid_, dir_info_);
    };
    gauss_newton_terminal_derivatives<FIXED_TERMINAL_RESIDUAL_DIM>(residual_fn, x, lfx, lfxx);
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
void initialize_primal_trajectory(SolverT& solver, const ProblemT& prob, const StateVec& x0, bool use_warm_start) {
    if (use_warm_start) {
        shift_warm_start(solver);
    } else {
        for (size_t k = 0; k < SolverT::N; ++k) {
            solver.us[k].setZero();
        }
    }

    solver.xs[0] = x0;
    for (size_t k = 0; k < SolverT::N; ++k) {
        solver.xs[k + 1] = prob.dynamics(static_cast<int>(k), solver.xs[k], solver.us[k]);
    }
}

template<typename ProblemT, typename SolverT>
MPCPrediction rollout_prediction(const ProblemT& prob, const SolverT& solver, const StateVec& x0) {
    std::array<StateVec, MPC_HORIZON + 1> xs_pred;
    xs_pred[0] = x0;
    int valid_steps = MPC_HORIZON;
    for (int k = 0; k < MPC_HORIZON; ++k) {
        xs_pred[static_cast<size_t>(k + 1)] =
            prob.dynamics(k, xs_pred[static_cast<size_t>(k)], solver.us[static_cast<size_t>(k)]);
        if (!xs_pred[static_cast<size_t>(k + 1)].allFinite()) {
            valid_steps = k;
            break;
        }
    }

    MPCPrediction pred;
    const auto sz = static_cast<size_t>(valid_steps + 1);
    pred.path_map.reserve(sz);
    pred.headings.reserve(sz);
    pred.v_pred.reserve(sz);
    pred.w_pred.reserve(sz);
    for (int i = 0; i <= valid_steps; ++i) {
        const auto& x = xs_pred[static_cast<size_t>(i)];
        pred.path_map.emplace_back(x(ix::X), x(ix::Y));
        pred.headings.push_back(x(ix::THETA));
        pred.v_pred.push_back(x(ix::V));
        pred.w_pred.push_back(x(ix::W));
    }
    return pred;
}

} // anonymous namespace

MPCSolver::MPCSolver(const MPCParams& params): params_(params) {}

void MPCSolver::set_last_cmd(const Eigen::Vector2d& cmd) {
    last_cmd_ = cmd;
}

void MPCSolver::reset_warm_start() {
    follow_warm_ = false;
    stop_warm_ = false;
    recovery_warm_ = false;
    fixed_warm_ = false;
    last_u_ = 0.0;
    for (size_t k = 0; k < MPC_HORIZON; ++k) {
        follow_solver_.us[k].setZero();
        follow_solver_left_.us[k].setZero();
        follow_solver_right_.us[k].setZero();
        stop_solver_.us[k].setZero();
        recovery_solver_.us[k].setZero();
        fixed_solver_.us[k].setZero();
    }
}

void MPCSolver::set_energy_state(double remaining_energy, double rfr_pwr_limit) {
    remaining_energy_ = remaining_energy;
    rfr_pwr_limit_ = rfr_pwr_limit;
}

void MPCSolver::update_observer(double v_act, double w_act) {
    if (!observer_initialized_) {
        x_h_hat_ = XH0;
        prev_v_act_ = v_act;
        prev_w_act_ = w_act;
        observer_initialized_ = true;
        return;
    }
    const double sv_prev = std::tanh(prev_v_act_ / SGN_EPS);
    const double nl_prev = CF1 * sv_prev + CF2 * prev_v_act_ * std::abs(prev_w_act_);
    const double xh_pred = A00 * x_h_hat_ + A01 * prev_v_act_ + A03 * last_cmd_.x() + GNL_XH * nl_prev;
    const double v_pred = A10 * x_h_hat_ + A11 * prev_v_act_ + A13 * last_cmd_.x() + GNL_V * nl_prev;
    x_h_hat_ = xh_pred + OBS_L * (v_act - v_pred);
    prev_v_act_ = v_act;
    prev_w_act_ = w_act;
}

StateVec MPCSolver::make_initial_state(
    const Eigen::Vector3d& pose,
    const Eigen::Vector2d& status,
    const Eigen::Vector2d& cmd_clamped,
    double path_u
) const {
    StateVec x0;
    x0(ix::X) = pose.x();
    x0(ix::Y) = pose.y();
    x0(ix::THETA) = pose.z();
    x0(ix::XH) = x_h_hat_;
    x0(ix::V) = status.x();
    x0(ix::W) = status.y();
    x0(ix::DV) = cmd_clamped.x();
    x0(ix::DW) = cmd_clamped.y();
    x0(ix::PATH_U) = path_u;
    return x0;
}

MPCPrediction MPCSolver::extract_prediction(const StateVec* xs, int n) {
    MPCPrediction pred;
    const auto sz = static_cast<size_t>(n);
    pred.path_map.reserve(sz);
    pred.headings.reserve(sz);
    pred.v_pred.reserve(sz);
    pred.w_pred.reserve(sz);
    for (int i = 0; i < n; ++i) {
        pred.path_map.emplace_back(xs[i](ix::X), xs[i](ix::Y));
        pred.headings.push_back(xs[i](ix::THETA));
        pred.v_pred.push_back(xs[i](ix::V));
        pred.w_pred.push_back(xs[i](ix::W));
    }
    return pred;
}

void MPCSolver::generate_lateral_hypothesis(
    fddp::Solver<FollowProblem>& solver,
    const fddp::Solver<FollowProblem>& base_solver,
    const FollowProblem& prob,
    const StateVec& x0,
    const std::vector<Eigen::Vector2d>& ref_cps,
    int keep_steps,
    double lateral_offset
) {
    // 复制基准 solver 的完整轨迹（已 shift 过的 warm start）
    solver.xs = base_solver.xs;
    solver.us = base_solver.us;

    const int K = keep_steps;
    const int N = static_cast<int>(MPC_HORIZON);
    if (K >= N - 1 || ref_cps.size() < 3) return;

    // 获取 splice point（第 K 步）状态
    const auto& x_splice = base_solver.xs[static_cast<size_t>(K)];
    const double splice_v = x_splice(ix::V);

    // 参考路径在末端的位置和方向（目标纵向位置）
    const auto& x_end = base_solver.xs[static_cast<size_t>(N)];
    const double end_u = std::clamp(x_end(ix::PATH_U), 0.0, 1.0);
    Eigen::Vector2d pr_end, d1_end;
    eval_bspline2(ref_cps, end_u, &pr_end, &d1_end, nullptr);
    const double end_theta_ref = std::atan2(d1_end.y(), d1_end.x());

    // 在 Frenet 坐标中，构造五次多项式使横向偏移从 0 平滑过渡到 lateral_offset
    // s ∈ [0, 1]（归一化时间），ey(0)=0, ey'(0)=0, ey''(0)=0，ey(1)=offset, ey'(1)=0, ey''(1)=0
    // 五次多项式: ey(s) = offset * (10*s³ - 15*s⁴ + 6*s⁵)
    const int steps_remaining = N - K;
    const double dt_inv = 1.0 / static_cast<double>(steps_remaining);

    // 目标位置：沿参考方向移动，向法线方向偏移
    const Eigen::Vector2d tang(std::cos(end_theta_ref), std::sin(end_theta_ref));
    const Eigen::Vector2d normal(-tang.y(), tang.x());

    for (int i = 0; i <= steps_remaining; ++i) {
        const double s = static_cast<double>(i) * dt_inv;
        const double s3 = s * s * s, s4 = s3 * s, s5 = s4 * s;
        const double ey = lateral_offset * (10.0 * s3 - 15.0 * s4 + 6.0 * s5);
        const double dey_ds = lateral_offset * (30.0 * s * s - 60.0 * s3 + 30.0 * s4);

        // 线性插值 base 轨迹位置作为纵向参考
        const auto& x_base = base_solver.xs[static_cast<size_t>(K + i)];
        const double base_x = x_base(ix::X), base_y = x_base(ix::Y);

        // 横向偏移
        auto& xs_k = solver.xs[static_cast<size_t>(K + i)];
        xs_k = x_base;
        xs_k(ix::X) = base_x + normal.x() * ey;
        xs_k(ix::Y) = base_y + normal.y() * ey;

        // 微分平坦反推角度：偏移后的航向考虑横向运动
        if (i < steps_remaining) {
            const double theta_corr = std::atan2(dey_ds * dt_inv * normal.y(), splice_v * tang.x() + dey_ds * dt_inv * normal.x());
            xs_k(ix::THETA) = x_base(ix::THETA) + theta_corr * 0.3;
        }
    }

    // 用 modified 的状态通过动力学反推控制序列
    for (int k = K; k < N; ++k) {
        // 简化：使用 base solver 的控制量，但调整 omega 以匹配修改后的航向
        solver.us[static_cast<size_t>(k)] = base_solver.us[static_cast<size_t>(k)];
        if (k + 1 <= N) {
            const double theta_cur = solver.xs[static_cast<size_t>(k)](ix::THETA);
            const double theta_next = solver.xs[static_cast<size_t>(k + 1)](ix::THETA);
            const double dtheta = wrap_pi(theta_next - theta_cur);
            // 粗略估算所需角速度指令
            const double w_est = dtheta / MPC_DT;
            solver.us[static_cast<size_t>(k)](1) = std::clamp(w_est, prob.u_lower()(1), prob.u_upper()(1));
        }
    }

    // 从 x0 开始重新前向 rollout 以保证动力学一致性
    solver.xs[0] = x0;
    for (int k = 0; k < N; ++k) {
        solver.xs[static_cast<size_t>(k + 1)] = prob.dynamics(k, solver.xs[static_cast<size_t>(k)], solver.us[static_cast<size_t>(k)]);
    }
}

std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> MPCSolver::follow_path(
    const SplineD& global_path,
    const Eigen::Vector3d& chassis_pose_map,
    const Eigen::Vector2d& chassis_status,
    const CostMap& cost_map,
    const DirectionMap& direction_map
) {
    const auto& ref_cps = global_path.getControlPoints();
    const bool path_changed = !same_cps(prev_ref_control_points_, ref_cps);
    const double projection_hint = path_changed ? 0.0 : last_u_;
    const double u0 = project_to_spline_u(
        global_path,
        chassis_pose_map.head<2>(),
        projection_hint,
        params_.follow_projection.proj_num_samples,
        params_.follow_projection.proj_search_window,
        params_.follow_projection.local_search_lazy_distance
    );
    if (path_changed) {
        follow_warm_ = false;
    }
    last_u_ = u0;

    const Eigen::Vector2d cmd0(
        clamp_prev_cmd(
            last_cmd_.x(),
            chassis_status.x(),
            params_.follow_limits.start_vel_cmd_act_diff_max,
            params_.follow_limits.acc_max,
            MPC_DT
        ),
        clamp_prev_cmd(
            last_cmd_.y(),
            chassis_status.y(),
            params_.follow_limits.start_omega_cmd_act_diff_max,
            params_.follow_limits.alpha_max,
            MPC_DT
        )
    );

    const CostMapGridView cg(cost_map);
    const GridInfo ci = make_grid_info(cost_map);
    const DirectionMapGridView dg(direction_map);
    const GridInfo di = make_grid_info(direction_map);
    const StateVec x0 = make_initial_state(chassis_pose_map, chassis_status, cmd0, u0);

    const auto& arc = [&]() -> const ArclengthTable& {
        const int ns = params_.follow_limits.slow_down_num_samples;
        if (prev_arc_samples_ != ns || !same_cps(prev_ref_control_points_, ref_cps)) {
            prev_arclength_table_ = build_arclength_table(ref_cps, ns);
            prev_ref_control_points_ = ref_cps;
            prev_arc_samples_ = ns;
        }
        return prev_arclength_table_;
    }();

    // ── 局部障碍物感知：前瞻检测并动态缩放跟踪权重 ──
    const auto& oat = params_.obstacle_aware_tracking;
    MPCParams effective_params = params_;
    {
        double max_cost = 0.0;
        Eigen::Vector2d prev_pos;
        eval_bspline2(ref_cps, u0, &prev_pos, nullptr, nullptr);
        double cum_dist = 0.0;
        const int n = std::max(1, oat.num_samples);
        const double u_remain = 1.0 - u0;
        for (int i = 1; i <= n; i++) {
            const double u = u0 + u_remain * static_cast<double>(i) / static_cast<double>(n);
            Eigen::Vector2d pos;
            eval_bspline2(ref_cps, u, &pos, nullptr, nullptr);
            cum_dist += (pos - prev_pos).norm();
            prev_pos = pos;
            if (cum_dist > oat.lookahead_distance) break;
            max_cost = std::max(max_cost, eval_cost_bicubic(cg, ci, pos.x(), pos.y()).value);
        }
        if (max_cost > oat.cost_threshold) {
            const double t = std::clamp(
                (max_cost - oat.cost_threshold) / (255.0 - oat.cost_threshold), 0.0, 1.0
            );
            const double scale = 1.0 - t * (1.0 - oat.min_weight_scale);
            effective_params.follow_weights.q_y *= scale;
            effective_params.follow_weights.q_theta *= scale;
        }
    }

    FollowProblem prob(ref_cps, effective_params, cg, ci, dg, di, arc, remaining_energy_, rfr_pwr_limit_);

    fddp::SolverOptions opts;
    opts.max_iters = 25;
    opts.tol_grad = 1e-4;
    opts.tol_cost = 1e-6;

    // ── 初始化中心假设 ──
    initialize_primal_trajectory(follow_solver_, prob, x0, follow_warm_);

    if (params_.mh_params.enable && follow_warm_) {
        // ── 生成左右偏移假设 ──
        generate_lateral_hypothesis(
            follow_solver_left_,
            follow_solver_,
            prob,
            x0,
            ref_cps,
            params_.mh_params.keep_steps,
            -params_.mh_params.lateral_offset
        );
        generate_lateral_hypothesis(
            follow_solver_right_,
            follow_solver_,
            prob,
            x0,
            ref_cps,
            params_.mh_params.keep_steps,
            +params_.mh_params.lateral_offset
        );

        // ── 并行求解 3 个假设 ──
        std::array<double, 3> costs {};
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                auto r = follow_solver_.solve(prob, opts);
                costs[0] = r.cost;
            }
            #pragma omp section
            {
                auto r = follow_solver_left_.solve(prob, opts);
                costs[1] = r.cost;
            }
            #pragma omp section
            {
                auto r = follow_solver_right_.solve(prob, opts);
                costs[2] = r.cost;
            }
        }

        // ── 选取最优 ──
        const int best = static_cast<int>(std::min_element(costs.begin(), costs.end()) - costs.begin());
        auto* best_solver = &follow_solver_;
        // 将最优解传播回主 solver，使下一帧 warm start 使用最优轨迹
        if (best == 1) {
            best_solver = &follow_solver_left_;
            follow_solver_.xs = best_solver->xs;
            follow_solver_.us = best_solver->us;
        } else if (best == 2) {
            best_solver = &follow_solver_right_;
            follow_solver_.xs = best_solver->xs;
            follow_solver_.us = best_solver->us;
        }
    } else {
        // 单假设求解
        follow_solver_.solve(prob, opts);
    }

    follow_warm_ = true;
    const Eigen::Vector2d cmd(follow_solver_.us[0](0), follow_solver_.us[0](1));
    last_cmd_ = cmd;
    return std::tuple {cmd, rollout_prediction(prob, follow_solver_, x0)};
}

std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> MPCSolver::stop(
    const Eigen::Vector3d& chassis_pose_map,
    const Eigen::Vector2d& chassis_status,
    const CostMap& cost_map,
    const DirectionMap& direction_map
) {
    const Eigen::Vector2d cmd0(
        clamp_prev_cmd(
            last_cmd_.x(),
            chassis_status.x(),
            params_.stop_limits.start_vel_cmd_act_diff_max,
            params_.stop_limits.acc_max,
            MPC_DT
        ),
        clamp_prev_cmd(
            last_cmd_.y(),
            chassis_status.y(),
            params_.stop_limits.start_omega_cmd_act_diff_max,
            params_.stop_limits.alpha_max,
            MPC_DT
        )
    );

    const CostMapGridView cg(cost_map);
    const GridInfo ci = make_grid_info(cost_map);
    const DirectionMapGridView dg(direction_map);
    const GridInfo di = make_grid_info(direction_map);
    const StateVec x0 = make_initial_state(chassis_pose_map, chassis_status, cmd0, 0.0);

    StopProblem prob(params_, cg, ci, dg, di, remaining_energy_, rfr_pwr_limit_);
    initialize_primal_trajectory(stop_solver_, prob, x0, stop_warm_);

    fddp::SolverOptions opts;
    opts.max_iters = 25;
    opts.tol_grad = 1e-4;
    opts.tol_cost = 1e-6;
    stop_solver_.solve(prob, opts);
    stop_warm_ = true;

    const Eigen::Vector2d cmd(stop_solver_.us[0](0), stop_solver_.us[0](1));
    last_cmd_ = cmd;
    return std::tuple {cmd, rollout_prediction(prob, stop_solver_, x0)};
}

std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> MPCSolver::recover_to_point(
    const Eigen::Vector2d& goal_map,
    const Eigen::Vector3d& chassis_pose_map,
    const Eigen::Vector2d& chassis_status,
    const CostMap& cost_map,
    const DirectionMap& direction_map
) {
    const Eigen::Vector2d cmd0(
        clamp_prev_cmd(
            last_cmd_.x(),
            chassis_status.x(),
            params_.recovery_limits.start_vel_cmd_act_diff_max,
            params_.recovery_limits.acc_max,
            MPC_DT
        ),
        clamp_prev_cmd(
            last_cmd_.y(),
            chassis_status.y(),
            params_.recovery_limits.start_omega_cmd_act_diff_max,
            params_.recovery_limits.alpha_max,
            MPC_DT
        )
    );

    const CostMapGridView cg(cost_map);
    const GridInfo ci = make_grid_info(cost_map);
    const DirectionMapGridView dg(direction_map);
    const GridInfo di = make_grid_info(direction_map);
    const StateVec x0 = make_initial_state(chassis_pose_map, chassis_status, cmd0, 0.0);

    RecoveryProblem prob(goal_map, params_, cg, ci, dg, di, remaining_energy_, rfr_pwr_limit_);
    initialize_primal_trajectory(recovery_solver_, prob, x0, recovery_warm_);

    fddp::SolverOptions opts;
    opts.max_iters = 25;
    opts.tol_grad = 1e-4;
    opts.tol_cost = 1e-6;
    recovery_solver_.solve(prob, opts);
    recovery_warm_ = true;

    const Eigen::Vector2d cmd(recovery_solver_.us[0](0), recovery_solver_.us[0](1));
    last_cmd_ = cmd;
    return std::tuple {cmd, rollout_prediction(prob, recovery_solver_, x0)};
}

std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> MPCSolver::hold_at_point(
    const Eigen::Vector2d& goal_map,
    const Eigen::Vector3d& chassis_pose_map,
    const Eigen::Vector2d& chassis_status,
    const CostMap& cost_map,
    const DirectionMap& direction_map
) {
    const Eigen::Vector2d cmd0(
        clamp_prev_cmd(
            last_cmd_.x(),
            chassis_status.x(),
            params_.fixed_limits.start_vel_cmd_act_diff_max,
            params_.fixed_limits.acc_max,
            MPC_DT
        ),
        clamp_prev_cmd(
            last_cmd_.y(),
            chassis_status.y(),
            params_.fixed_limits.start_omega_cmd_act_diff_max,
            params_.fixed_limits.alpha_max,
            MPC_DT
        )
    );

    const CostMapGridView cg(cost_map);
    const GridInfo ci = make_grid_info(cost_map);
    const DirectionMapGridView dg(direction_map);
    const GridInfo di = make_grid_info(direction_map);
    const StateVec x0 = make_initial_state(chassis_pose_map, chassis_status, cmd0, 0.0);

    FixedProblem prob(goal_map, params_, cg, ci, dg, di, remaining_energy_, rfr_pwr_limit_);
    initialize_primal_trajectory(fixed_solver_, prob, x0, fixed_warm_);

    fddp::SolverOptions opts;
    opts.max_iters = 25;
    opts.tol_grad = 1e-4;
    opts.tol_cost = 1e-6;
    fixed_solver_.solve(prob, opts);
    fixed_warm_ = true;

    const Eigen::Vector2d cmd(fixed_solver_.us[0](0), fixed_solver_.us[0](1));
    last_cmd_ = cmd;
    return std::tuple {cmd, rollout_prediction(prob, fixed_solver_, x0)};
}

} // namespace path_follower