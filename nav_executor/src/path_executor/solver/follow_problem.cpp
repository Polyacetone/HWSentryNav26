#include <nav_executor/path_executor/solver/follow_problem.hpp>

#include <algorithm>
#include <cmath>

#include <nav_executor/path_executor/solver/mpc_utils.hpp>

namespace nav_executor {

namespace {

constexpr double COST_EPS = 1e-9;

// 跟踪残差：横向/沿轨位置误差 + 独立朝向 + 时间缩放后的速度/角速度参考 + 指令/运动约束 + 障碍
// + 台阶（若在约束窗内）。不含：进度项（线性，单独加）、可达包络 / 制动 / cost-to-goal
// （落地版 Q3 已删）。速度跟踪项是折返段反向的驱动力——参考 v_ref(τ) 在回撤段为负。
constexpr int FOLLOW_RESIDUAL_DIM = 18;
using FollowResidualVec = Eigen::Matrix<double, FOLLOW_RESIDUAL_DIM, 1>;

FollowResidualVec follow_residual_impl(
    const StateVec& x,
    const ControlVec& u,
    const MincoTrajectory& traj,
    const MPCParams& p,
    const CostMapGridView& cg,
    const GridInfo& ci,
    const MPCMotionConstraints& motion_lim,
    const std::shared_ptr<const StepConstraintSchedule>& step_schedule
) {
    const auto& follow = p.follow;
    const auto& tracking_w = follow.tracking_weights;
    const auto& command_w = follow.command_weights;
    const auto& motion_w = follow.motion_constraint_weights;
    const auto& terrain_w = follow.terrain_weights;
    const auto& env_w = follow.environment_weights;

    FollowResidualVec r = FollowResidualVec::Zero();

    const double px = x(ix::X), py = x(ix::Y), theta = x(ix::THETA);
    const double v_act = x(ix::V);
    const double v_cmd = u(0), w_cmd = u(1);
    const double dv_cmd = v_cmd - x(ix::DV);
    const double dw_cmd = w_cmd - x(ix::DW);

    const double phase_time = std::clamp(x(ix::PHASE_TIME), 0.0, traj.total_time());
    const double phase_rate = std::clamp(x(ix::PHASE_RATE), 0.0, 1.0);
    const double tau = traj.total_time() > COST_EPS ? phase_time / traj.total_time() : 0.0;
    const TrajSample s = traj.eval_time(phase_time);

    // 横向误差：投影到位置曲线切线帧（几何量，倒车时切线反向仅翻转 ey 符号，|ey| 不变）。
    const double ex = px - s.p.x();
    const double ey_w = py - s.p.y();
    Eigen::Vector2d tangent;
    if (s.ds_dtau > COST_EPS) {
        tangent = s.dp_dtau / s.ds_dtau;
    } else {
        tangent = Eigen::Vector2d(std::cos(s.theta), std::sin(s.theta));
    }
    const Eigen::Vector2d normal(-tangent.y(), tangent.x());
    const Eigen::Vector2d position_error(ex, ey_w);
    const double e_contour = normal.dot(position_error);
    const double e_lag = tangent.dot(position_error);

    // 朝向误差：相对**独立** θ_ref（不再是位置切线 atan2）。
    const double etheta = wrap_pi(theta - s.theta);

    // 带符号纵向速度参考：折返回撤段 v_ref<0，直接驱动反向；原地旋转段 v_ref≈0。
    const double v_ref = phase_rate * traj.longitudinal_velocity(s);
    const double omega_ref = phase_rate * traj.angular_velocity(s);

    const double dv_lim = motion_lim.acc_max * MPC_DT;
    const double dw_lim = motion_lim.alpha_max * MPC_DT;
    const double a_lat = std::abs(v_cmd * w_cmd);

    // 横向管廊：|ey| < y_tube 内不惩罚（允许横向腾挪），管廊外二次拉回。
    const double contour_tube = positive_part(std::abs(e_contour) - tracking_w.y_tube)
        * sign_or_zero(e_contour);
    r(0) = tracking_w.q_y * contour_tube;
    r(1) = tracking_w.q_lag * e_lag;
    r(2) = tracking_w.q_theta * etheta;
    r(3) = tracking_w.q_v * (v_act - v_ref);
    r(4) = tracking_w.q_omega * (x(ix::W) - omega_ref);
    r(5) = command_w.r_v * v_cmd;
    r(6) = command_w.r_omega * w_cmd;
    r(7) = command_w.r_dv * dv_cmd;
    r(8) = command_w.r_domega * dw_cmd;
    r(9) = motion_w.acc_limit * positive_part(std::abs(dv_cmd) - dv_lim);
    r(10) = motion_w.alpha_limit * positive_part(std::abs(dw_cmd) - dw_lim);
    r(11) = motion_w.lat_acc * positive_part(a_lat - motion_lim.a_lat_max);

    const auto cs = eval_cost_bilinear(cg, ci, px, py);
    r(12) = env_w.obstacle * cs.value / 255.0;

    // ── 台阶入阶硬约束（按 τ 索引；无 f 门控，助跑速度剖面已由 MINCO 烘焙）──
    const StepTraversalConstraint* const step = step_schedule
        ? step_schedule->constraint_at(tau)
        : nullptr;
    if (step) {
        const double gate = step_window_gate(tau, *step);
        if (gate > 0.0) {
            const Eigen::Vector2d& dir = step->dir_map; // 归一化常量穿越方向
            const double cross = std::cos(theta) * dir.y() - std::sin(theta) * dir.x();
            r(13) = terrain_w.direction * gate * std::abs(cross);

            const double speed_min = step->speed_min;
            const double speed_max = step->speed_max;
            const double vstep = v_act < speed_min
                ? (speed_min - v_act)
                : (v_act > speed_max ? (v_act - speed_max) : 0.0);
            r(14) = terrain_w.step_vel_weight * gate * vstep;

            r(15) = terrain_w.step_omega * gate * w_cmd;
            r(16) = terrain_w.step_dv * gate * dv_cmd;
            r(17) = terrain_w.step_domega * gate * dw_cmd;
        }
    }

    return r;
}

double remaining_progress(const MincoTrajectory& trajectory, const double tau) {
    return std::max(
        0.0,
        trajectory.total_arc_length() - trajectory.arc_length_at_tau(std::clamp(tau, 0.0, 1.0))
    );
}

} // anonymous namespace

template<int Horizon>
FollowProblemT<Horizon>::FollowProblemT(
    MincoTrajectory trajectory,
    const MPCParams& params,
    const std::vector<CostMapGridView>& per_step_cost_grids,
    const GridInfo& cost_info,
    const CostMapGridView& masked_global_grid,
    double prediction_dt,
    double schedule_rho,
    const CapabilityProfile& blended_profile,
    std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule
):
    trajectory_(std::move(trajectory)),
    p_(params),
    step_cost_grids_(per_step_cost_grids),
    cost_info_(cost_info),
    masked_global_grid_(masked_global_grid),
    prediction_dt_(prediction_dt),
    model_(build_lpv_discrete_model(params.kinematic_model, schedule_rho)),
    blended_profile_(blended_profile),
    step_constraint_schedule_(std::move(step_constraint_schedule)),
    total_arc_(trajectory_.total_arc_length()),
    total_time_(trajectory_.total_time()) {}

template<int Horizon>
TrajectoryPhaseState FollowProblemT<Horizon>::advance_phase(const StateVec& x) const {
    return advance_trajectory_phase(
        trajectory_,
        {.time = x(ix::PHASE_TIME), .rate = x(ix::PHASE_RATE)},
        Eigen::Vector3d(x(ix::X), x(ix::Y), x(ix::THETA)),
        MPC_DT,
        p_.follow.phase
    );
}

template<int Horizon>
StateVec FollowProblemT<Horizon>::dynamics(int, const StateVec& x, const ControlVec& u) const {
    StateVec xn = mpc_dynamics(x, u, model_);
    const TrajectoryPhaseState phase = advance_phase(x);
    xn(ix::PHASE_TIME) = phase.time;
    xn(ix::PHASE_RATE) = phase.rate;
    return xn;
}

template<int Horizon>
void FollowProblemT<Horizon>::dynamics_jacobians(int, const StateVec& x, const ControlVec& u, MatXX& fx, MatXU& fu) const {
    mpc_dynamics_jacobians(x, u, model_, fx, fu);

    // 相位状态只依赖当前预测状态，不直接依赖控制；两行统一有限差分。
    constexpr double EPS_REL = 1e-6;
    constexpr double EPS_ABS = 1e-4;
    for (int i = 0; i < MPC_NX; ++i) {
        const double eps = std::max(EPS_REL * std::abs(x(i)), EPS_ABS);
        StateVec xp = x, xm = x;
        xp(i) += eps;
        xm(i) -= eps;
        const TrajectoryPhaseState pp = advance_phase(xp);
        const TrajectoryPhaseState pm = advance_phase(xm);
        fx(ix::PHASE_TIME, i) = (pp.time - pm.time) / (2.0 * eps);
        fx(ix::PHASE_RATE, i) = (pp.rate - pm.rate) / (2.0 * eps);
    }
    fu.row(ix::PHASE_TIME).setZero();
    fu.row(ix::PHASE_RATE).setZero();
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
const CostMapGridView& FollowProblemT<Horizon>::cost_grid_for_step(int k) const {
    if (step_cost_grids_.size() <= 1) return step_cost_grids_[0];
    int idx = static_cast<int>(static_cast<double>(k) * MPC_DT / prediction_dt_);
    return step_cost_grids_[static_cast<size_t>(std::min(idx, static_cast<int>(step_cost_grids_.size()) - 1))];
}

template<int Horizon>
double FollowProblemT<Horizon>::running_cost(int k, const StateVec& x, const ControlVec& u) const {
    return running_cost_value_only(k, x, u);
}

template<int Horizon>
double FollowProblemT<Horizon>::running_cost_value_only(int k, const StateVec& x, const ControlVec& u, double*) const {
    const auto r = follow_residual_impl(
        x, u, trajectory_, p_, cost_grid_for_step(k), cost_info_,
        blended_profile_.motion_constraints, step_constraint_schedule_
    );
    double cost = residual_cost(r);

    // 进度奖励（线性，非平方；零 Hessian，与旧设计一致）：cost 随 τ→1 递减。
    const double tau = total_time_ > COST_EPS ? x(ix::PHASE_TIME) / total_time_ : 0.0;
    if (tau < 1.0) {
        cost += (p_.follow.tracking_weights.q_u / MPC_HORIZON)
            * remaining_progress(trajectory_, tau);
    }
    return cost;
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
    auto residual_fn = [&](const StateVec& xv, const ControlVec& uv) {
        return follow_residual_impl(
            xv, uv, trajectory_, p_, cg, cost_info_,
            blended_profile_.motion_constraints, step_constraint_schedule_
        );
    };
    gauss_newton_running_derivatives<FOLLOW_RESIDUAL_DIM>(residual_fn, x, u, lx, lu, lxx, lux, luu);

    // 进度线性项梯度：d/dτ[(q_u/H)·(1-τ)·L] = -(q_u/H)·L。零 Hessian。
    const double tau = total_time_ > COST_EPS ? x(ix::PHASE_TIME) / total_time_ : 0.0;
    if (tau < 1.0) {
        const TrajSample reference = trajectory_.eval(std::clamp(tau, 0.0, 1.0));
        const double arc_rate = total_time_ > COST_EPS
            ? reference.ds_dtau / total_time_
            : 0.0;
        lx(ix::PHASE_TIME) -= (p_.follow.tracking_weights.q_u / MPC_HORIZON) * arc_rate;
    }
}

template<int Horizon>
double FollowProblemT<Horizon>::terminal_cost(const StateVec&) const {
    // 终端无 cost-to-goal 势（落地版 Q3 已删）；进度项 + 逐步跟踪已提供到达压力。
    return 0.0;
}

template<int Horizon>
void FollowProblemT<Horizon>::terminal_cost_derivatives(const StateVec&, StateVec& lfx, MatXX& lfxx) const {
    lfx.setZero();
    lfxx.setZero();
}

template<int Horizon>
std::optional<RolloutLethalObstacleInfo> FollowProblemT<Horizon>::detect_lethal_obstacle(int state_index, const StateVec& x, double* out_cost_value) const {
    const auto& safety = p_.follow.rollout_safety;
    if (!safety.enable_lethal_obstacle_check) return std::nullopt;

    const auto sample = eval_cost_bilinear(masked_global_grid_, cost_info_, x(ix::X), x(ix::Y));
    if (out_cost_value) *out_cost_value = sample.value;
    if (sample.value + COST_EPS < safety.lethal_obstacle_threshold) return std::nullopt;

    return RolloutLethalObstacleInfo {
        .state_index = state_index,
        .position_map = Eigen::Vector2d(x(ix::X), x(ix::Y)),
        .sampled_cost = sample.value,
    };
}

template class FollowProblemT<MPC_HORIZON>;

} // namespace nav_executor
