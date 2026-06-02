#include <path_follower/solver/follow_problem.hpp>
#include <path_follower/solver/mpc_utils.hpp>
#include <path_follower/solver/bilinear_sampling.hpp>
#include <path_follower/solver/lpv_model.hpp>
#include <path_follower/solver/mppi_sampler.hpp>
#include <array>
#include <cmath>

namespace path_follower {

constexpr double COST_EPS = 1e-9;
constexpr double REACHABILITY_EPS = 1e-6;

// ─── 前向声明 ───

double advance_u_progress(double u_cur, const StateVec& x, const SplinePath& spline);
double advance_u_progress_extrapolated(double u_cur, const StateVec& x, const SplinePath& spline);

struct AdvanceUProgressEval {
    double u_next_extrap;
    StateVec du_next_dx;
};

AdvanceUProgressEval
advance_u_progress_extrapolated_with_jacobian(double u_cur, const StateVec& x, const SplinePath& spline);

// ─── 残差实现 ───

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
    const SplinePath& spline,
    const MPCParams& p,
    const CostMapGridView& cg,
    const GridInfo& ci,
    const DirectionMapGridView& dg,
    const GridInfo& di,
    double rfr_pwr_limit,
    const MPCMotionConstraints& motion_lim,
    std::optional<ActiveStepMode> active_step_mode
) {
    const auto& follow = p.follow;
    const auto& tracking_w = follow.tracking_weights;
    const auto& command_w = follow.command_weights;
    const auto& motion_w = follow.motion_constraint_weights;
    const auto& terrain_w = follow.terrain_weights;
    const auto& env_w = follow.environment_weights;

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
    const double uc = SplinePath::clamp_u_extrapolated(uc_raw);
    const double duc_dpathu = clamp_derivative_piecewise(uc_raw, SplinePath::U_EXTRAP_MIN, SplinePath::U_EXTRAP_MAX);

    const auto se = spline.eval(uc);

    const double d1_norm2 = se.d1.squaredNorm();
    const double dtheta_du = (se.d1.x() * se.d2.y() - se.d1.y() * se.d2.x()) / std::max(d1_norm2, 1e-12);

    const double ex = px - se.p.x(), ey_w = py - se.p.y();
    const double ey = -ex * se.sin_r + ey_w * se.cos_r;
    const double dey_dpx = -se.sin_r;
    const double dey_dpy = se.cos_r;
    const double dey_du = se.sin_r * se.d1.x() - se.cos_r * se.d1.y() - dtheta_du * (ex * se.cos_r + ey_w * se.sin_r);
    const double dey_dpathu = dey_du * duc_dpathu;

    const double etheta = wrap_pi(theta - se.thetar);
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

    const double dv_lim = motion_lim.acc_max * MPC_DT;
    const double dw_lim = motion_lim.alpha_max * MPC_DT;

    const double a_lat = std::abs(v_cmd * w_cmd);

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

    const double sign_lat = sign_or_zero(v_cmd * w_cmd);
    const double relu_lat = positive_part(a_lat - motion_lim.a_lat_max);
    out.r(8) = motion_w.lat_acc * relu_lat;
    const double coeff_lat = motion_w.lat_acc * positive_part_derivative(a_lat - motion_lim.a_lat_max);
    out.ju(8, 0) = coeff_lat * sign_lat * w_cmd;
    out.ju(8, 1) = coeff_lat * sign_lat * v_cmd;

    out.r(9) = env_w.obstacle * cs.value / 255.0;
    out.jx(9, ix::X) = env_w.obstacle * cs.dx / 255.0;
    out.jx(9, ix::Y) = env_w.obstacle * cs.dy / 255.0;

    const double sign_cross = sign_or_zero(cross);
    out.r(10) = terrain_w.direction * std::abs(cross);
    out.jx(10, ix::X) = terrain_w.direction * sign_cross * dcross_dx;
    out.jx(10, ix::Y) = terrain_w.direction * sign_cross * dcross_dy;
    out.jx(10, ix::THETA) = terrain_w.direction * sign_cross * dcross_dtheta;

    if (is_active_follow_step_mode(active_step_mode)) {
        const double speed_min = active_step_mode->speed_min;
        const double speed_max = active_step_mode->speed_max;
        const double v_err = v_act < speed_min
            ? (speed_min - v_act)
            : (v_act > speed_max ? (v_act - speed_max) : 0.0);
        const double abs_v_err = std::abs(v_err);
        const double relu_vstep = abs_v_err;

        if (dir_norm_sq > 1e-10) {
            out.r(11) = terrain_w.step_vel_weight * dir_norm * relu_vstep;
            out.jx(11, ix::X) = terrain_w.step_vel_weight * dnorm_dx * relu_vstep;
            out.jx(11, ix::Y) = terrain_w.step_vel_weight * dnorm_dy * relu_vstep;
            out.jx(11, ix::V) = terrain_w.step_vel_weight * dir_norm
                * ((v_act < speed_min) ? -1.0 : (v_act > speed_max) ? 1.0 : 0.0);
        }

        if (active_step_mode->step_entry_u.has_value()) {
            const double entry_u = std::clamp(*active_step_mode->step_entry_u, 0.0, 1.0);
            const double path_u = std::clamp(uc, 0.0, 1.0);
            if (path_u < entry_u) {
                const double d = spline.arc_length(path_u, entry_u, 8);

                const auto se2 = spline.eval(path_u);
                const double ds_du = se2.ds_du * duc_dpathu;

                const double a_guide = std::max(follow.terrain_limits.step_reachability_guide_acc, REACHABILITY_EPS);
                const double r_lo_expr = std::max(0.0, v_act) * std::max(0.0, v_act) + 2.0 * a_guide * d;
                const double r_hi_expr = v_act * v_act - 2.0 * a_guide * d;
                const double r_lo_arg = std::max(REACHABILITY_EPS, r_lo_expr);
                const double r_hi_arg = std::max(REACHABILITY_EPS, r_hi_expr);
                const double r_lo = std::sqrt(r_lo_arg);
                const double r_hi = std::sqrt(r_hi_arg);

                const double relu_lo = positive_part(speed_min - r_lo);
                out.r(12) = std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * relu_lo;
                if (relu_lo > 0.0) {
                    const bool lo_active = r_lo_expr > REACHABILITY_EPS;
                    const double drlo_dv = (lo_active && v_act > 0.0) ? (v_act / r_lo) : 0.0;
                    const double drlo_du = (lo_active) ? (-a_guide * ds_du / r_lo) : 0.0;
                    out.jx(12, ix::V) = -std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * drlo_dv;
                    out.jx(12, ix::PATH_U) = -std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * drlo_du;
                }

                const double relu_hi = positive_part(r_hi - speed_max);
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
    const double s_remaining_approx = (1.0 - uc_clamped_brake) * se.ds_du;
    const double v_allowed = brake_speed_limit(s_remaining_approx, terminal_w);
    const double v_excess = v_act - v_allowed;
    const double relu_brake = positive_part(v_excess);
    out.r(14) = terminal_w.q_v_final * relu_brake;
    if (v_excess > 0.0) {
        out.jx(14, ix::V) = terminal_w.q_v_final;
        const double ds_dpathu = -se.ds_du * duc_dpathu;
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
    const SplinePath& spline,
    const MPCParams& p,
    const CostMapGridView& cg,
    const GridInfo& ci,
    const DirectionMapGridView& dg,
    const GridInfo& di,
    double rfr_pwr_limit,
    const MPCMotionConstraints& motion_lim,
    std::optional<ActiveStepMode> active_step_mode,
    double* cached_cost_value
) {
    const auto& follow = p.follow;
    const auto& tracking_w = follow.tracking_weights;
    const auto& command_w = follow.command_weights;
    const auto& motion_w = follow.motion_constraint_weights;
    const auto& terrain_w = follow.terrain_weights;
    const auto& env_w = follow.environment_weights;

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
    const double uc = SplinePath::clamp_u_extrapolated(uc_raw);

    const auto se = spline.eval(uc);

    const double ex = px - se.p.x();
    const double ey_w = py - se.p.y();
    const double ey = -ex * se.sin_r + ey_w * se.cos_r;
    const double etheta = wrap_pi(theta - se.thetar);

    const Eigen::Vector2d dir = eval_dir_bilinear_value_only(dg, di, px, py);

    double cost_value;
    if (cached_cost_value && *cached_cost_value >= 0.0) {
        cost_value = *cached_cost_value;
        *cached_cost_value = -1.0;
    } else {
        cost_value = eval_cost_bilinear(cg, ci, px, py).value;
    }

    const double dir_norm_sq = dir.squaredNorm();
    const double dir_norm = std::sqrt(dir_norm_sq);
    const double cross = std::cos(theta) * dir.y() - std::sin(theta) * dir.x();
    const double a_lat = std::abs(v_cmd * w_cmd);

    const double dv_lim = motion_lim.acc_max * MPC_DT;
    const double dw_lim = motion_lim.alpha_max * MPC_DT;

    double cost = 0.0;

    cost += 0.5 * (tracking_w.q_y * ey) * (tracking_w.q_y * ey);
    cost += 0.5 * (tracking_w.q_theta * etheta) * (tracking_w.q_theta * etheta);
    cost += 0.5 * (command_w.r_v * v_cmd) * (command_w.r_v * v_cmd);
    cost += 0.5 * (command_w.r_omega * w_cmd) * (command_w.r_omega * w_cmd);
    cost += 0.5 * (command_w.r_dv * dv_cmd) * (command_w.r_dv * dv_cmd);
    cost += 0.5 * (command_w.r_domega * dw_cmd) * (command_w.r_domega * dw_cmd);
    cost += 0.5 * (motion_w.acc_limit * positive_part(std::abs(dv_cmd) - dv_lim)) * (motion_w.acc_limit * positive_part(std::abs(dv_cmd) - dv_lim));
    cost += 0.5 * (motion_w.alpha_limit * positive_part(std::abs(dw_cmd) - dw_lim)) * (motion_w.alpha_limit * positive_part(std::abs(dw_cmd) - dw_lim));
    cost += 0.5 * (motion_w.lat_acc * positive_part(a_lat - motion_lim.a_lat_max)) * (motion_w.lat_acc * positive_part(a_lat - motion_lim.a_lat_max));
    cost += 0.5 * (env_w.obstacle * cost_value / 255.0) * (env_w.obstacle * cost_value / 255.0);
    cost += 0.5 * (terrain_w.direction * std::abs(cross)) * (terrain_w.direction * std::abs(cross));

    if (is_active_follow_step_mode(active_step_mode)) {
        const double speed_min = active_step_mode->speed_min;
        const double speed_max = active_step_mode->speed_max;
        const double v_err = v_act < speed_min
            ? (speed_min - v_act)
            : (v_act > speed_max ? (v_act - speed_max) : 0.0);
        const double relu_vstep = std::abs(v_err);

        if (dir_norm_sq > 1e-10) {
            cost += 0.5 * (terrain_w.step_vel_weight * dir_norm * relu_vstep) * (terrain_w.step_vel_weight * dir_norm * relu_vstep);
        }

        if (active_step_mode->step_entry_u.has_value()) {
            const double entry_u = std::clamp(*active_step_mode->step_entry_u, 0.0, 1.0);
            const double path_u = std::clamp(uc, 0.0, 1.0);
            if (path_u < entry_u) {
                const double d = spline.arc_length(path_u, entry_u, 8);
                const double a_guide = std::max(follow.terrain_limits.step_reachability_guide_acc, REACHABILITY_EPS);
                const double r_lo = std::sqrt(std::max(REACHABILITY_EPS, std::max(0.0, v_act) * std::max(0.0, v_act) + 2.0 * a_guide * d));
                const double r_hi = std::sqrt(std::max(REACHABILITY_EPS, v_act * v_act - 2.0 * a_guide * d));

                cost += 0.5 * (std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * positive_part(speed_min - r_lo)) * (std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * positive_part(speed_min - r_lo));
                cost += 0.5 * (std::sqrt(std::max(terrain_w.step_reachability_hi, 0.0)) * positive_part(r_hi - speed_max)) * (std::sqrt(std::max(terrain_w.step_reachability_hi, 0.0)) * positive_part(r_hi - speed_max));
            }
        }
    }

    if (p.energy.enable) {
        const auto pwr = predict_power_eval_vw(p.power_model, v_act, w_act);
        const double thr = std::max(p.energy.threshold, 1.0);
        const double excess = (pwr.value - rfr_pwr_limit) / thr;
        cost += 0.5 * (p.energy.weight * positive_part(excess)) * (p.energy.weight * positive_part(excess));
    }

    const double uc_clamped = std::clamp(uc, 0.0, 1.0);
    const double s_remaining = spline.arc_length(uc_clamped, 1.0, 8);

    if (uc < 1.0) {
        cost += (p.follow.tracking_weights.q_u / MPC_HORIZON) * s_remaining;
    }

    const auto& terminal_w = follow.terminal_weights;
    const double v_allowed = brake_speed_limit(s_remaining, terminal_w);
    cost += 0.5 * (terminal_w.q_v_final * positive_part(v_act - v_allowed)) * (terminal_w.q_v_final * positive_part(v_act - v_allowed));
    return cost;
}

AdvanceUProgressEval
advance_u_progress_extrapolated_with_jacobian(double u_cur, const StateVec& x, const SplinePath& spline) {
    AdvanceUProgressEval out {};
    out.du_next_dx.setZero();

    const double uc = SplinePath::clamp_u_extrapolated(u_cur);
    const double duc_dpathu = clamp_derivative_piecewise(x(ix::PATH_U), SplinePath::U_EXTRAP_MIN, SplinePath::U_EXTRAP_MAX);

    const auto se = spline.eval(uc);

    const double d1_norm2 = se.d1.squaredNorm();
    const double d1_norm = std::sqrt(d1_norm2 + 0.01);
    const double dsdu = d1_norm + 1e-6;
    const double inv_dsdu = 1.0 / dsdu;

    const double cross12 = se.d1.x() * se.d2.y() - se.d1.y() * se.d2.x();
    const double kappa = cross12 / (dsdu * dsdu * dsdu);
    const double dtheta_du = cross12 / std::max(d1_norm2, 1e-12);

    const double ex = x(ix::X) - se.p.x();
    const double ey_w = x(ix::Y) - se.p.y();
    const double ey = -ex * se.sin_r + ey_w * se.cos_r;

    const double dey_dpx = -se.sin_r;
    const double dey_dpy = se.cos_r;
    const double dey_du = se.sin_r * se.d1.x() - se.cos_r * se.d1.y() - dtheta_du * (ex * se.cos_r + ey_w * se.sin_r);

    const double etheta = wrap_pi(x(ix::THETA) - se.thetar);
    const double cos_e = std::cos(etheta);
    const double sin_e = std::sin(etheta);

    const double num = x(ix::V) * cos_e;
    const double dnum_dv = cos_e;
    const double dnum_dtheta = -x(ix::V) * sin_e;
    const double dnum_du = x(ix::V) * (-sin_e) * (-dtheta_du);

    const double denom_raw = 1.0 - kappa * ey;
    constexpr double DENOM_EPS = 0.05;
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

    const double ddsdu_du = (d1_norm > 1e-12) ? (se.d1.dot(se.d2) / d1_norm) : 0.0;
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

double advance_u_progress_extrapolated(double u_cur, const StateVec& x, const SplinePath& spline) {
    return advance_u_progress_extrapolated_with_jacobian(u_cur, x, spline).u_next_extrap;
}

double advance_u_progress(double u_cur, const StateVec& x, const SplinePath& spline) {
    return SplinePath::clamp_u_extrapolated(advance_u_progress_extrapolated(u_cur, x, spline));
}

template<int Horizon>
FollowProblemT<Horizon>::FollowProblemT(
    const SplinePath& spline,
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
    const CapabilityProfile& blended_profile,
    std::optional<ActiveStepMode> active_step_mode,
    double current_path_u
):
    spline_(spline),
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
    blended_profile_(blended_profile),
    active_step_mode_(active_step_mode),
    current_path_u_(current_path_u) {
    goal_xy_ = spline_.eval(1.0).p;
}

template<int Horizon>
StateVec FollowProblemT<Horizon>::dynamics(int, const StateVec& x, const ControlVec& u) const {
    StateVec xn = mpc_dynamics(x, u, model_);
    xn(ix::PATH_U) = advance_u_progress(x(ix::PATH_U), x, spline_);
    return xn;
}

template<int Horizon>
void FollowProblemT<Horizon>::dynamics_jacobians(int, const StateVec& x, const ControlVec& u, MatXX& dfx, MatXU& dfu) const {
    mpc_dynamics_jacobians(x, u, model_, dfx, dfu);

    const auto adv = advance_u_progress_extrapolated_with_jacobian(x(ix::PATH_U), x, spline_);
    const double dout_din = clamp_derivative_piecewise(adv.u_next_extrap, SplinePath::U_EXTRAP_MIN, SplinePath::U_EXTRAP_MAX);
    dfx.row(ix::PATH_U) = (dout_din * adv.du_next_dx).transpose();
    dfu.row(ix::PATH_U).setZero();
}

template<int Horizon>
ControlVec FollowProblemT<Horizon>::u_lower() const {
    return ControlVec(blended_profile_.command_bounds.vel_min, blended_profile_.command_bounds.omega_min);
}

template<int Horizon>
ControlVec FollowProblemT<Horizon>::u_upper() const {
    return ControlVec(blended_profile_.command_bounds.vel_max, blended_profile_.command_bounds.omega_max);
}

template<int Horizon>
const MPCParams& FollowProblemT<Horizon>::params() const {
    return p_;
}

template<int Horizon>
FollowProblemT<Horizon> FollowProblemT<Horizon>::with_reference_path(const SplinePath& spline) const {
    return FollowProblemT<Horizon>(
        spline,
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
        blended_profile_,
        active_step_mode_,
        current_path_u_
    );
}

template<int Horizon>
const CostMapGridView& FollowProblemT<Horizon>::cost_grid_for_step(int k) const {
    if (step_cost_grids_.size() <= 1) return step_cost_grids_[0];
    int idx = static_cast<int>(static_cast<double>(k) * MPC_DT / prediction_dt_);
    return step_cost_grids_[static_cast<size_t>(std::min(idx, static_cast<int>(step_cost_grids_.size()) - 1))];
}

template<int Horizon>
std::optional<RolloutLethalObstacleInfo> FollowProblemT<Horizon>::detect_lethal_obstacle(int state_index, const StateVec& x, double* out_cost_value) const {
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

    if (out_cost_value) {
        *out_cost_value = sample.value;
    }

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
double FollowProblemT<Horizon>::running_cost_value_only(int k, const StateVec& x, const ControlVec& u, double* cached_cost_value) const {
    return follow_running_cost_value_only_impl(
        x, u, spline_, p_,
        cost_grid_for_step(k), cost_info_, dir_grid_, dir_info_,
        rfr_pwr_limit_, blended_profile_.motion_constraints,
        active_step_mode_, cached_cost_value
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
        x, u, spline_, p_, cg, cost_info_, dir_grid_, dir_info_,
        rfr_pwr_limit_, blended_profile_.motion_constraints,
        active_step_mode_
    );

    lx = lin.jx.transpose() * lin.r;
    lu = lin.ju.transpose() * lin.r;

    {
        const double uc_raw = x(ix::PATH_U);
        const double uc = SplinePath::clamp_u_extrapolated(uc_raw);
        if (uc < 1.0) {
            const double duc_dpathu = clamp_derivative_piecewise(uc_raw, SplinePath::U_EXTRAP_MIN, SplinePath::U_EXTRAP_MAX);
            const double ds_du = spline_.eval(uc).ds_du;
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

} // namespace path_follower
