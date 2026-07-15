#include <nav_executor/path_executor/solver/follow_problem.hpp>

#include <algorithm>
#include <cmath>

#include <nav_executor/path_executor/solver/mpc_utils.hpp>

namespace nav_executor {

namespace {

constexpr double COST_EPS = 1e-9;

// τ 允许的外推范围（沿用旧 PATH_U 语义：略微越界以容纳投影/预测漂移）。
constexpr double TAU_EXTRAP_MIN = -0.5;
constexpr double TAU_EXTRAP_MAX = 1.5;

inline double clamp_tau_extrap(double tau) {
    return std::clamp(tau, TAU_EXTRAP_MIN, TAU_EXTRAP_MAX);
}

// 跟踪残差：位置横向管廊 + 独立朝向误差 + 带符号纵向速度跟踪 + 指令/运动约束 + 障碍
// + 台阶（若在约束窗内）。不含：进度项（线性，单独加）、可达包络 / 制动 / cost-to-goal
// （落地版 Q3 已删）。速度跟踪项是折返段反向的驱动力——参考 v_ref(τ) 在回撤段为负。
constexpr int FOLLOW_RESIDUAL_DIM = 16;
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

    const double tau = clamp_tau_extrap(x(ix::PATH_U));
    const TrajSample s = traj.eval(tau);

    // 横向误差：投影到位置曲线切线帧（几何量，倒车时切线反向仅翻转 ey 符号，|ey| 不变）。
    const double ex = px - s.p.x();
    const double ey_w = py - s.p.y();
    const double ey = -ex * s.sin_phi + ey_w * s.cos_phi;

    // 朝向误差：相对**独立** θ_ref（不再是位置切线 atan2）。
    const double etheta = wrap_pi(theta - s.theta);

    // 带符号纵向速度参考：折返回撤段 v_ref<0，直接驱动反向；原地旋转段 v_ref≈0。
    const double v_ref = traj.longitudinal_velocity(s);

    const double dv_lim = motion_lim.acc_max * MPC_DT;
    const double dw_lim = motion_lim.alpha_max * MPC_DT;
    const double a_lat = std::abs(v_cmd * w_cmd);

    // 横向管廊：|ey| < y_tube 内不惩罚（允许横向腾挪），管廊外二次拉回。
    const double ey_tube = positive_part(std::abs(ey) - tracking_w.y_tube) * sign_or_zero(ey);
    r(0) = tracking_w.q_y * ey_tube;
    r(1) = tracking_w.q_theta * etheta;
    r(12) = tracking_w.q_v * (v_act - v_ref);
    r(2) = command_w.r_v * v_cmd;
    r(3) = command_w.r_omega * w_cmd;
    r(4) = command_w.r_dv * dv_cmd;
    r(5) = command_w.r_domega * dw_cmd;
    r(6) = motion_w.acc_limit * positive_part(std::abs(dv_cmd) - dv_lim);
    r(7) = motion_w.alpha_limit * positive_part(std::abs(dw_cmd) - dw_lim);
    r(8) = motion_w.lat_acc * positive_part(a_lat - motion_lim.a_lat_max);

    const auto cs = eval_cost_bilinear(cg, ci, px, py);
    r(9) = env_w.obstacle * cs.value / 255.0;

    // ── 台阶入阶硬约束（按 τ 索引；无 f 门控，助跑速度剖面已由 MINCO 烘焙）──
    const StepTraversalConstraint* const step = step_schedule
        ? step_schedule->constraint_at(tau)
        : nullptr;
    if (step) {
        const double gate = step_window_gate(tau, *step);
        if (gate > 0.0) {
            const Eigen::Vector2d& dir = step->dir_map; // 归一化常量穿越方向
            const double cross = std::cos(theta) * dir.y() - std::sin(theta) * dir.x();
            r(10) = terrain_w.direction * gate * std::abs(cross);

            const double speed_min = step->speed_min;
            const double speed_max = step->speed_max;
            const double vstep = v_act < speed_min
                ? (speed_min - v_act)
                : (v_act > speed_max ? (v_act - speed_max) : 0.0);
            r(11) = terrain_w.step_vel_weight * gate * vstep;

            r(13) = terrain_w.step_omega * gate * w_cmd;
            r(14) = terrain_w.step_dv * gate * dv_cmd;
            r(15) = terrain_w.step_domega * gate * dw_cmd;
        }
    }

    return r;
}

// 沿 τ 的剩余进度（弧长度量），单调递减到 0；驱动 τ→1 的线性进度奖励。
double remaining_progress(double tau, double total_arc) {
    return positive_part(1.0 - std::clamp(tau, 0.0, 1.0)) * total_arc;
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
double FollowProblemT<Horizon>::clock_governor(const TrajSample& s, const StateVec& x) const {
    // 调速因子 s(e) ∈ (0,1]：随「机器人 ↔ 参考」跟踪误差平滑衰减（仅减速，不加速）。
    //   e² = ‖p_robot − p_ref(τ)‖² + λ²·(θ_robot − θ_ref(τ))²
    //   s  = 1 / (1 + (e / e_scale)²)
    // 误差小（跟得上）→ s≈1，时钟按名义速率推进 → θ_ref/p_ref/v_ref 持续前进，
    //   残差梯度始终活着 → 折返顶点、原地旋转不再是鞍点（时钟不依赖机器人此刻是否在动）。
    // 误差大（动态障碍逼停等）→ s→0，时钟自动等待 → 机器人靠障碍残差就地绕行，
    //   误差回落后时钟从暂停处续上 → 全程不重规划（真卡死由 τ 停滞的 no_progress 兜底）。
    return governed_clock_factor(
        s,
        Eigen::Vector3d(x(ix::X), x(ix::Y), x(ix::THETA)),
        p_.follow.governed_clock
    );
}

template<int Horizon>
double FollowProblemT<Horizon>::advance_tau(double tau, const StateVec& x) const {
    // 调速时钟（governed clock，落地版方案 A）：τ 按名义时钟前进，被跟踪误差调速。
    const double tau0 = clamp_tau_extrap(tau);
    const TrajSample s = trajectory_.eval(tau0);
    const double nominal_rate = total_time_ > COST_EPS ? 1.0 / total_time_ : 0.0;
    const double governed = nominal_rate * clock_governor(s, x);
    return clamp_tau_extrap(tau0 + MPC_DT * governed);
}

template<int Horizon>
StateVec FollowProblemT<Horizon>::dynamics(int, const StateVec& x, const ControlVec& u) const {
    StateVec xn = mpc_dynamics(x, u, model_);
    xn(ix::PATH_U) = advance_tau(x(ix::PATH_U), x);
    return xn;
}

template<int Horizon>
void FollowProblemT<Horizon>::dynamics_jacobians(int, const StateVec& x, const ControlVec& u, MatXX& fx, MatXU& fu) const {
    mpc_dynamics_jacobians(x, u, model_, fx, fu);

    // τ 进度行：调速时钟只依赖 x（不依赖 u），故 fu 行为 0，fx 行有限差分。
    constexpr double EPS_REL = 1e-6;
    constexpr double EPS_ABS = 1e-4;
    for (int i = 0; i < MPC_NX; ++i) {
        const double eps = std::max(EPS_REL * std::abs(x(i)), EPS_ABS);
        StateVec xp = x, xm = x;
        xp(i) += eps;
        xm(i) -= eps;
        fx(ix::PATH_U, i) = (advance_tau(xp(ix::PATH_U), xp) - advance_tau(xm(ix::PATH_U), xm)) / (2.0 * eps);
    }
    fu.row(ix::PATH_U).setZero();
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
    const double tau = x(ix::PATH_U);
    if (tau < 1.0) {
        cost += (p_.follow.tracking_weights.q_u / MPC_HORIZON) * remaining_progress(tau, total_arc_);
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
    const double tau = x(ix::PATH_U);
    if (tau < 1.0) {
        lx(ix::PATH_U) -= (p_.follow.tracking_weights.q_u / MPC_HORIZON) * total_arc_;
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
