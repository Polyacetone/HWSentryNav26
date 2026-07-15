#include <nav_executor/path_executor/solver/follow_problem.hpp>
#include <nav_executor/path_executor/solver/mpc_utils.hpp>
#include <nav_executor/path_executor/solver/bilinear_sampling.hpp>
#include <nav_executor/path_executor/solver/lpv_model.hpp>

namespace nav_executor {

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

constexpr int FOLLOW_RESIDUAL_DIM = 18;
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
    const MPCMotionConstraints& motion_lim,
    const std::shared_ptr<const StepConstraintSchedule>& step_constraint_schedule
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
    const double v_act = x(ix::V);
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

    const double dv_lim = motion_lim.acc_max * MPC_DT;
    const double dw_lim = motion_lim.alpha_max * MPC_DT;

    const double a_lat = std::abs(v_cmd * w_cmd);

    // 横向误差管廊：|ey| < y_tube 内不惩罚（允许横向腾挪对齐/助跑台阶），
    // 管廊外按 q_y 二次拉回。残差 = q_y * sign(ey)*max(0,|ey|-W)，管廊外导数为 1。
    const double ey_tube = positive_part(std::abs(ey) - tracking_w.y_tube) * sign_or_zero(ey);
    const double dtube_dey = (std::abs(ey) > tracking_w.y_tube) ? 1.0 : 0.0;
    out.r(0) = tracking_w.q_y * ey_tube;
    out.jx(0, ix::X) = tracking_w.q_y * dtube_dey * dey_dpx;
    out.jx(0, ix::Y) = tracking_w.q_y * dtube_dey * dey_dpy;
    out.jx(0, ix::PATH_U) = tracking_w.q_y * dtube_dey * dey_dpathu;

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

    // ── 台阶约束（残差 10/11/12/13/15/16/17）──
    // 每个预测点按自身 PATH_U 查询不可变约束表，不依赖当前机器人所在段或 FSM 状态。
    // 台阶内部约束随预测 path_u 由 step_window_gate 调度。
    // 门控对 path_u 的梯度冻结：约束跟随轨迹前移，但不诱导优化器移动 path_u 逃逸。
    //   - 10 方向对齐：航向 theta 与规划期常量穿越方向 dir_map 对齐，仅 theta 雅可比。
    //   - 11 速度窗：v_act 保持在 [speed_min, speed_max]，仅 V 雅可比。
    //   两者作用区间 [commit_u, exit_u]（含两侧软过渡）。
    //   - 12/13 可达包络：commit 前整形速度剖面使其在 commit 处可行进窗，仅 V/PATH_U 雅可比。
    const StepTraversalConstraint* const internal_step = step_constraint_schedule
        ? step_constraint_schedule->constraint_at(uc)
        : nullptr;
    if (internal_step) {
        const double speed_min = internal_step->speed_min;
        const double speed_max = internal_step->speed_max;
        const double gate = step_window_gate(uc, *internal_step);

        if (gate > 0.0) {
            // 助跑可行性因子 f：仅在物理边缘上游（uc < step_enter_u）才可能 <1；越过边缘后
            // 已无后退退路，f≡1。作为冻结门控（不并入雅可比），与 step_window_gate 一致，
            // 避免优化器移动 v/path_u 逃避约束。gf = gate * f 调度“助跑期建立约束”的施加。
            double f = 1.0;
            if (uc < internal_step->step_enter_u) {
                const double d_edge = spline.arc_length(
                    std::clamp(uc, 0.0, 1.0), std::clamp(internal_step->step_enter_u, 0.0, 1.0), 8
                );
                const double a_guide = std::max(follow.terrain_limits.step_reachability_guide_acc, REACHABILITY_EPS);
                f = step_feasibility_factor(
                    v_act, d_edge, speed_min, a_guide, follow.terrain_limits.step_feasibility_margin_band
                );
            }
            const double gf = gate * f;

            const Eigen::Vector2d& dir = internal_step->dir_map; // 归一化常量
            const double cos_t = std::cos(theta);
            const double sin_t = std::sin(theta);
            const double cross = cos_t * dir.y() - sin_t * dir.x();
            const double dcross_dtheta = -sin_t * dir.y() - cos_t * dir.x();
            const double sign_cross = sign_or_zero(cross);
            out.r(10) = terrain_w.direction * gf * std::abs(cross);
            out.jx(10, ix::THETA) = terrain_w.direction * gf * sign_cross * dcross_dtheta;

            // 速度窗：under-speed 分支受 f 门控（不可行时释放，让位于后退退路）；
            // over-speed 分支不受 f 影响（任何时候禁止超速进入台阶）。
            if (v_act < speed_min) {
                out.r(11) = terrain_w.step_vel_weight * gf * (speed_min - v_act);
                out.jx(11, ix::V) = -terrain_w.step_vel_weight * gf;
            } else if (v_act > speed_max) {
                out.r(11) = terrain_w.step_vel_weight * gate * (v_act - speed_max);
                out.jx(11, ix::V) = terrain_w.step_vel_weight * gate;
            }

            out.r(15) = terrain_w.step_omega * gf * w_cmd;
            out.ju(15, 1) = terrain_w.step_omega * gf;

            out.r(16) = terrain_w.step_dv * gf * dv_cmd;
            out.ju(16, 0) = terrain_w.step_dv * gf;
            out.jx(16, ix::DV) = -terrain_w.step_dv * gf;

            out.r(17) = terrain_w.step_domega * gf * dw_cmd;
            out.ju(17, 1) = terrain_w.step_domega * gf;
            out.jx(17, ix::DW) = -terrain_w.step_domega * gf;
        }

    }

    const StepTraversalConstraint* const approach_step = step_constraint_schedule
        ? step_constraint_schedule->approach_constraint_at(uc)
        : nullptr;
    if (approach_step) {
        const double speed_min = approach_step->speed_min;
        const double speed_max = approach_step->speed_max;
        // 可达包络参考点为物理边缘 step_enter_u（贯穿 run_up 平地段）：d→0 处退化为入口速度地板，
        // 守住“到真实起跳点必须达速”，同时全程提供“太慢又太近→减小 path_u 后退腾距离”的退路梯度。
        const double enter_u = std::clamp(approach_step->step_enter_u, 0.0, 1.0);
        const double path_u = std::clamp(uc, 0.0, 1.0);
        if (path_u < enter_u) {
            const double d = spline.arc_length(path_u, enter_u, 8);

            const auto se2 = spline.eval(path_u);
            const double ds_du = se2.ds_du * duc_dpathu;

            const double a_guide = std::max(follow.terrain_limits.step_reachability_guide_acc, REACHABILITY_EPS);
            const double r_lo_expr = std::max(0.0, v_act) * std::max(0.0, v_act) + 2.0 * a_guide * d;
            const double r_hi_expr = v_act * v_act - 2.0 * a_guide * d;
            const double r_lo = std::sqrt(std::max(0.0, r_lo_expr));
            const double r_hi = std::sqrt(std::max(0.0, r_hi_expr));

            const double relu_lo = positive_part(speed_min - r_lo);
            out.r(12) = std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * relu_lo;
            if (relu_lo > 0.0) {
                const bool lo_active = r_lo_expr > 0.0;
                const double inv_r_lo = 1.0 / std::max(r_lo, REACHABILITY_EPS);
                const double drlo_dv = (lo_active && v_act > 0.0) ? (v_act * inv_r_lo) : 0.0;
                const double drlo_du = lo_active ? (-a_guide * ds_du * inv_r_lo) : 0.0;
                out.jx(12, ix::V) = -std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * drlo_dv;
                out.jx(12, ix::PATH_U) = -std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * drlo_du;
            }

            const double relu_hi = positive_part(r_hi - speed_max);
            out.r(13) = std::sqrt(std::max(terrain_w.step_reachability_hi, 0.0)) * relu_hi;
            if (relu_hi > 0.0) {
                const bool hi_active = r_hi_expr > 0.0;
                const double inv_r_hi = 1.0 / std::max(r_hi, REACHABILITY_EPS);
                const double drhi_dv = hi_active ? (v_act * inv_r_hi) : 0.0;
                const double drhi_du = hi_active ? (a_guide * ds_du * inv_r_hi) : 0.0;
                out.jx(13, ix::V) = std::sqrt(std::max(terrain_w.step_reachability_hi, 0.0)) * drhi_dv;
                out.jx(13, ix::PATH_U) = std::sqrt(std::max(terrain_w.step_reachability_hi, 0.0)) * drhi_du;
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

    return out;
}

double follow_running_cost_value_only_impl(
    const StateVec& x,
    const ControlVec& u,
    const SplinePath& spline,
    const MPCParams& p,
    const CostMapGridView& cg,
    const GridInfo& ci,
    const MPCMotionConstraints& motion_lim,
    const std::shared_ptr<const StepConstraintSchedule>& step_constraint_schedule,
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

    double cost_value;
    if (cached_cost_value && *cached_cost_value >= 0.0) {
        cost_value = *cached_cost_value;
        *cached_cost_value = -1.0;
    } else {
        cost_value = eval_cost_bilinear(cg, ci, px, py).value;
    }

    const double a_lat = std::abs(v_cmd * w_cmd);

    const double dv_lim = motion_lim.acc_max * MPC_DT;
    const double dw_lim = motion_lim.alpha_max * MPC_DT;

    double cost = 0.0;

    const double ey_tube = positive_part(std::abs(ey) - tracking_w.y_tube) * sign_or_zero(ey);
    cost += 0.5 * (tracking_w.q_y * ey_tube) * (tracking_w.q_y * ey_tube);
    cost += 0.5 * (tracking_w.q_theta * etheta) * (tracking_w.q_theta * etheta);
    cost += 0.5 * (command_w.r_v * v_cmd) * (command_w.r_v * v_cmd);
    cost += 0.5 * (command_w.r_omega * w_cmd) * (command_w.r_omega * w_cmd);
    cost += 0.5 * (command_w.r_dv * dv_cmd) * (command_w.r_dv * dv_cmd);
    cost += 0.5 * (command_w.r_domega * dw_cmd) * (command_w.r_domega * dw_cmd);
    cost += 0.5 * (motion_w.acc_limit * positive_part(std::abs(dv_cmd) - dv_lim)) * (motion_w.acc_limit * positive_part(std::abs(dv_cmd) - dv_lim));
    cost += 0.5 * (motion_w.alpha_limit * positive_part(std::abs(dw_cmd) - dw_lim)) * (motion_w.alpha_limit * positive_part(std::abs(dw_cmd) - dw_lim));
    cost += 0.5 * (motion_w.lat_acc * positive_part(a_lat - motion_lim.a_lat_max)) * (motion_w.lat_acc * positive_part(a_lat - motion_lim.a_lat_max));
    cost += 0.5 * (env_w.obstacle * cost_value / 255.0) * (env_w.obstacle * cost_value / 255.0);

    // ── 台阶约束（与 follow_residual_linearized_impl 逐项对应）──
    const StepTraversalConstraint* const internal_step = step_constraint_schedule
        ? step_constraint_schedule->constraint_at(uc)
        : nullptr;
    if (internal_step) {
        const double speed_min = internal_step->speed_min;
        const double speed_max = internal_step->speed_max;
        const double gate = step_window_gate(uc, *internal_step);

        if (gate > 0.0) {
            // 助跑可行性因子 f（与 linearized 路径逐项对应）。
            double f = 1.0;
            if (uc < internal_step->step_enter_u) {
                const double d_edge = spline.arc_length(
                    std::clamp(uc, 0.0, 1.0), std::clamp(internal_step->step_enter_u, 0.0, 1.0), 8
                );
                const double a_guide = std::max(follow.terrain_limits.step_reachability_guide_acc, REACHABILITY_EPS);
                f = step_feasibility_factor(
                    v_act, d_edge, speed_min, a_guide, follow.terrain_limits.step_feasibility_margin_band
                );
            }
            const double gf = gate * f;

            const Eigen::Vector2d& dir = internal_step->dir_map;
            const double cross = std::cos(theta) * dir.y() - std::sin(theta) * dir.x();
            cost += 0.5 * (terrain_w.direction * gf * std::abs(cross)) * (terrain_w.direction * gf * std::abs(cross));

            // under-speed 受 f 门控；over-speed 不受 f 影响。
            const double vstep_scale = (v_act < speed_min) ? gf : gate;
            const double relu_vstep = v_act < speed_min
                ? (speed_min - v_act)
                : (v_act > speed_max ? (v_act - speed_max) : 0.0);
            cost += 0.5 * (terrain_w.step_vel_weight * vstep_scale * relu_vstep) * (terrain_w.step_vel_weight * vstep_scale * relu_vstep);
            cost += 0.5 * (terrain_w.step_omega * gf * w_cmd) * (terrain_w.step_omega * gf * w_cmd);
            cost += 0.5 * (terrain_w.step_dv * gf * dv_cmd) * (terrain_w.step_dv * gf * dv_cmd);
            cost += 0.5 * (terrain_w.step_domega * gf * dw_cmd) * (terrain_w.step_domega * gf * dw_cmd);
        }

    }

    const StepTraversalConstraint* const approach_step = step_constraint_schedule
        ? step_constraint_schedule->approach_constraint_at(uc)
        : nullptr;
    if (approach_step) {
        const double speed_min = approach_step->speed_min;
        const double speed_max = approach_step->speed_max;
        const double enter_u = std::clamp(approach_step->step_enter_u, 0.0, 1.0);
        const double path_u = std::clamp(uc, 0.0, 1.0);
        if (path_u < enter_u) {
            const double d = spline.arc_length(path_u, enter_u, 8);
            const double a_guide = std::max(follow.terrain_limits.step_reachability_guide_acc, REACHABILITY_EPS);
            const double r_lo = std::sqrt(std::max(0.0, std::max(0.0, v_act) * std::max(0.0, v_act) + 2.0 * a_guide * d));
            const double r_hi = std::sqrt(std::max(0.0, v_act * v_act - 2.0 * a_guide * d));

            cost += 0.5 * (std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * positive_part(speed_min - r_lo)) * (std::sqrt(std::max(terrain_w.step_reachability_lo, 0.0)) * positive_part(speed_min - r_lo));
            cost += 0.5 * (std::sqrt(std::max(terrain_w.step_reachability_hi, 0.0)) * positive_part(r_hi - speed_max)) * (std::sqrt(std::max(terrain_w.step_reachability_hi, 0.0)) * positive_part(r_hi - speed_max));
        }
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
    const CapabilityProfile& blended_profile,
    std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule,
    double current_path_u
):
    spline_(spline),
    p_(params),
    step_cost_grids_(per_step_cost_grids),
    cost_info_(cost_info),
    masked_global_grid_(masked_global_grid),
    prediction_dt_(prediction_dt),
    model_(build_lpv_discrete_model(params.kinematic_model, schedule_rho)),
    blended_profile_(blended_profile),
    step_constraint_schedule_(std::move(step_constraint_schedule)),
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
        blended_profile_,
        step_constraint_schedule_,
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
        cost_grid_for_step(k), cost_info_,
        blended_profile_.motion_constraints,
        step_constraint_schedule_, cached_cost_value
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
        x, u, spline_, p_, cg, cost_info_,
        blended_profile_.motion_constraints,
        step_constraint_schedule_
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
typename FollowProblemT<Horizon>::TerminalEval
FollowProblemT<Horizon>::evaluate_terminal(const StateVec& x) const {
    // 终端 cost-to-go：把路径当作一维值场，但补上横向分量，重建
    // "沿轨剩余弧长 + 横向跨回代价" 的曼哈顿近似，使其不再对横向偏移平坦、
    // 也不再乐观低估远离路径点的到达代价。
    //   V_term = w_prog * s_remaining(u*) + w_lat * |ey|
    // 梯度 = w_prog * (-切向) + w_lat * sign(ey) * (法向)
    // 当投影饱和到路径末端 (u*≈1) 时，进度项退化为到 goal 点的直线距离，
    // 避免 "末端归零 → 失去锚定 → rollout 乱飘"。
    const auto& tw = p_.follow.tracking_weights;
    const double w_prog = tw.q_term_prog;
    const double w_lat = tw.q_term_lateral;

    TerminalEval out;
    if (w_prog <= 0.0 && w_lat <= 0.0) return out;

    const Eigen::Vector2d p_xy(x(ix::X), x(ix::Y));
    const double hint = std::clamp(x(ix::PATH_U), 0.0, 1.0);
    constexpr double projection_window = 3.0;
    const double hint_arc = spline_.arc_length_at_u(hint);
    const auto projection = spline_.project(
        p_xy,
        hint_arc - projection_window,
        hint_arc + projection_window
    );
    const double u_star = projection ? projection->u : hint;
    const double uc = std::clamp(u_star, 0.0, 1.0);
    const auto se = spline_.eval(uc);

    constexpr double DIR_EPS = 1e-9;
    constexpr double END_U = 1.0 - 1e-6;

    if (uc >= END_U) {
        // 末端饱和：进度项接管为到 goal 的直线距离，梯度指向 goal。
        const Eigen::Vector2d to_goal = p_xy - goal_xy_;
        const double d = to_goal.norm();
        out.value += w_prog * d;
        if (d > DIR_EPS) {
            out.grad_xy += w_prog * (to_goal / d);
        }
        // 末端不再叠加横向项（goal 距离已同时含横向信息）。
        return out;
    }

    // ── 进度项：s_remaining，梯度 = -单位切向（包络定理）──
    out.value += w_prog * spline_.arc_length(uc, 1.0, 8);
    const double tnorm = se.d1.norm();
    if (w_prog > 0.0 && tnorm > DIR_EPS) {
        const Eigen::Vector2d t = se.d1 / tnorm;
        out.grad_xy += -w_prog * t;
    }

    // ── 横向项：w_lat * |ey|，梯度 = sign(ey) * 单位法向 ──
    // ey = -ex*sin_r + ey_w*cos_r，其 ∇_xy = (-sin_r, cos_r) = 左法向。
    if (w_lat > 0.0) {
        const double ex = p_xy.x() - se.p.x();
        const double ey_w = p_xy.y() - se.p.y();
        const double ey = -ex * se.sin_r + ey_w * se.cos_r;
        const Eigen::Vector2d n(-se.sin_r, se.cos_r);
        out.value += w_lat * std::abs(ey);
        out.grad_xy += w_lat * sign_or_zero(ey) * n;
    }

    return out;
}

template<int Horizon>
double FollowProblemT<Horizon>::terminal_cost(const StateVec& x) const {
    return evaluate_terminal(x).value;
}

template<int Horizon>
void FollowProblemT<Horizon>::terminal_cost_derivatives(const StateVec& x, StateVec& lfx, MatXX& lfxx) const {
    lfx.setZero();
    lfxx.setZero();
    const auto eval = evaluate_terminal(x);
    lfx(ix::X) = eval.grad_xy.x();
    lfx(ix::Y) = eval.grad_xy.y();
    // 线性/绝对值势无二阶项；backward pass 的 mu 正则负责曲率，lfxx 保持零。
}

template class FollowProblemT<MPC_HORIZON>;

} // namespace nav_executor
