#include <nav_executor/path_executor/mpc/follow_problem.hpp>

#include <algorithm>
#include <cmath>

#include <nav_executor/path_executor/mpc/mpc_utils.hpp>

namespace nav_executor {

namespace {

constexpr double COST_EPS = 1e-9;
constexpr int FOLLOW_RESIDUAL_DIM = 23;
using FollowResidualVec = Eigen::Matrix<double, FOLLOW_RESIDUAL_DIM, 1>;

CostMap::CostSample sample_cost_or_lethal(
    const CostMap& cost_map, const Eigen::Vector2d& position_map
) {
    return cost_map.sample_map(position_map).value_or(CostMap::CostSample {
        .value = 255.0,
        .gradient = Eigen::Vector2d::Zero(),
    });
}

// 有向路径参考点。发布前的数值验收排除检测到的内部 cusp。
struct ReferenceFrame {
    TrajSample sample;
    double path_progress = 0.0;
    double nominal_path_speed = 0.0;
    Eigen::Vector2d tangent = Eigen::Vector2d::UnitX();
};

struct FrozenStepGate {
    const StepTraversalConstraint* constraint = nullptr;
    double gate = 0.0;
};

ReferenceFrame reference_frame(
    const MincoTrajectory& trajectory,
    const PathSpeedProfile& speed_profile,
    const double requested_path_progress
) {
    ReferenceFrame reference;
    reference.path_progress = std::clamp(
        requested_path_progress, 0.0, trajectory.total_arc_length()
    );
    reference.sample = trajectory.eval_arc_length(reference.path_progress);
    reference.nominal_path_speed =
        speed_profile.eval_arc_length(reference.path_progress).velocity;
    reference.tangent = {
        std::cos(reference.sample.theta), std::sin(reference.sample.theta)
    };
    return reference;
}

FollowResidualVec follow_residual_impl(
    const StateVec& x,
    const ControlVec& u,
    const MincoTrajectory& trajectory,
    const PathSpeedProfile& speed_profile,
    const MPCParams& params,
    const CostMap& cost_map,
    const CommandDynamicsLimits& motion_limits,
    const LPVDiscreteModel& model,
    const std::shared_ptr<const StepConstraintSchedule>& step_schedule,
    const FrozenStepGate* frozen_step_gate = nullptr
) {
    const auto& follow = params.follow;
    const auto& tracking = follow.tracking_weights;
    const auto& progress = follow.progress;
    const auto& command = follow.command_weights;
    const auto& dynamics_weights = follow.command_dynamics_weights;
    const auto& traversal_weights = follow.traversal_target_weights;
    const auto& environment = follow.environment_weights;

    FollowResidualVec residual = FollowResidualVec::Zero();

    const double px = x(ix::X);
    const double py = x(ix::Y);
    const double theta = x(ix::THETA);
    const double v_actual = x(ix::V);
    const double omega_actual = x(ix::W);
    const Eigen::Vector2d next_command = command_after_control(x, u);
    const double v_cmd = next_command.x();
    const double omega_cmd = next_command.y();
    const double path_speed_cmd = u(iu::PATH_SPEED_CMD);
    const double dv_cmd = MPC_DT * u(iu::V_CMD_RATE);
    const double domega_cmd = MPC_DT * u(iu::W_CMD_RATE);

    const ReferenceFrame reference = reference_frame(
        trajectory, speed_profile, x(ix::PATH_PROGRESS)
    );
    const Eigen::Vector2d position_error(
        px - reference.sample.p.x(), py - reference.sample.p.y()
    );
    const Eigen::Vector2d normal(-reference.tangent.y(), reference.tangent.x());
    residual(0) = tracking.contour * normal.dot(position_error);
    residual(1) = tracking.lag * reference.tangent.dot(position_error);
    residual(2) = tracking.heading * wrap_pi(theta - reference.sample.theta);

    // 速度参考使用有向投影 v·cos(θ−θ_path) − ṡ：车身朝向偏离路径时，投影速度自然
    // 下降，虚拟进度无法在车辆走错方向时继续空跑。
    const double projected_velocity = v_actual
        * std::cos(wrap_pi(theta - reference.sample.theta));
    residual(3) = tracking.velocity * (projected_velocity - path_speed_cmd);
    residual(4) = tracking.angular_velocity
        * (omega_actual - reference.sample.kappa * path_speed_cmd);
    residual(5) = progress.speed_tracking_weight
        * (path_speed_cmd - reference.nominal_path_speed);
    residual(6) = progress.speed_smoothness_weight
        * (path_speed_cmd - x(ix::PATH_SPEED));

    residual(7) = command.r_v * v_cmd;
    residual(8) = command.r_omega * omega_cmd;
    residual(9) = command.r_dv * dv_cmd;
    residual(10) = command.r_domega * domega_cmd;
    residual(11) = command.r_jerk_v * (u(iu::V_CMD_RATE) - x(ix::V_CMD_RATE)) / MPC_DT;
    residual(12) = command.r_jerk_omega * (u(iu::W_CMD_RATE) - x(ix::W_CMD_RATE)) / MPC_DT;

    residual(13) = dynamics_weights.lateral_acceleration
        * positive_part(std::abs(v_cmd * omega_cmd) - motion_limits.lateral_acceleration_max);

    residual(14) = environment.obstacle
        * sample_cost_or_lethal(cost_map, {px, py}).value / 255.0;

    const StepTraversalConstraint* const step = frozen_step_gate
        ? frozen_step_gate->constraint
        : (step_schedule ? step_schedule->constraint_at(reference.path_progress) : nullptr);
    if (step) {
        const double gate = frozen_step_gate
            ? frozen_step_gate->gate
            : step_window_gate(reference.path_progress, *step);
        if (gate > 0.0) {
            const Eigen::Vector2d& direction = step->dir_map;
            const double cross = std::cos(theta) * direction.y()
                - std::sin(theta) * direction.x();
            // 最小二乘会自行平方残差；保留 cross 的符号，使中心差分在完全对齐处
            // 仍能得到正确的方向雅可比和 Gauss-Newton 曲率。
            residual(15) = traversal_weights.direction * gate * cross;

            // 约束真实路径方向速度而非车身纵向速度，避免通过偏航降低路径投影速度、
            // 同时维持较高纵向速度来分别满足路径跟踪与地形速度窗。
            const auto& target = step->velocity_window;
            const double velocity_error = projected_velocity < target.min
                ? target.min - projected_velocity
                : (projected_velocity > target.max
                    ? projected_velocity - target.max : 0.0);
            residual(16) = traversal_weights.velocity * gate * velocity_error;
            residual(17) = traversal_weights.angular_velocity_command * gate * omega_cmd;
            residual(18) = traversal_weights.angular_velocity_predicted * gate * omega_actual;
            residual(19) = traversal_weights.velocity_command_smoothness * gate * dv_cmd;
            const StateVec next_state = mpc_dynamics(x, u, model);
            residual(20) = traversal_weights.velocity_predicted_smoothness * gate
                * (next_state(ix::V) - v_actual);
            residual(21) = traversal_weights.angular_velocity_command_smoothness
                * gate * domega_cmd;
            residual(22) = traversal_weights.angular_velocity_predicted_smoothness
                * gate * (next_state(ix::W) - omega_actual);
        }
    }

    return residual;
}

} // anonymous namespace

template<int Horizon>
FollowProblemT<Horizon>::FollowProblemT(
    MincoTrajectory trajectory,
    PathSpeedProfile speed_profile,
    const MPCParams& params,
    const std::vector<const CostMap*>& per_step_cost_maps,
    const CostMap& masked_global_map,
    const double prediction_dt,
    const double schedule_rho,
    const CapabilityProfile& command_capability,
    std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule
):
    trajectory_(std::move(trajectory)),
    speed_profile_(std::move(speed_profile)),
    p_(params),
    step_cost_maps_(per_step_cost_maps),
    masked_global_map_(masked_global_map),
    prediction_dt_(prediction_dt),
    model_(build_lpv_discrete_model(params.kinematic_model, schedule_rho)),
    command_capability_(command_capability),
    step_constraint_schedule_(std::move(step_constraint_schedule)),
    total_length_(trajectory_.total_arc_length()) {}

template<int Horizon>
StateVec FollowProblemT<Horizon>::dynamics(int, const StateVec& x, const ControlVec& u) const {
    StateVec next = mpc_dynamics(x, u, model_);
    // 硬边界 0 ≤ s ≤ L 与 ṡ ≥ 0 由 control_bounds 保证，动力学只做前向积分。
    next(ix::PATH_PROGRESS) = x(ix::PATH_PROGRESS) + MPC_DT * u(iu::PATH_SPEED_CMD);
    next(ix::PATH_SPEED) = u(iu::PATH_SPEED_CMD);
    return next;
}

template<int Horizon>
void FollowProblemT<Horizon>::dynamics_jacobians(
    int,
    const StateVec& x,
    const ControlVec& u,
    MatXX& fx,
    MatXU& fu
) const {
    mpc_dynamics_jacobians(x, u, model_, fx, fu);
    fx.row(ix::PATH_PROGRESS).setZero();
    fx(ix::PATH_PROGRESS, ix::PATH_PROGRESS) = 1.0;
    fu.row(ix::PATH_PROGRESS).setZero();
    fu(ix::PATH_PROGRESS, iu::PATH_SPEED_CMD) = MPC_DT;

    fx.row(ix::PATH_SPEED).setZero();
    fu.row(ix::PATH_SPEED).setZero();
    fu(ix::PATH_SPEED, iu::PATH_SPEED_CMD) = 1.0;
}

template<int Horizon>
MPCControlBounds FollowProblemT<Horizon>::control_bounds(const int k, const StateVec& x) const {
    // 虚拟进度速度允许从零开始，不复用底盘命令的正下限：参考点错位时 MPCC 可以先停车，
    // 而不是被迫以最小速度蠕动。上界还受剩余弧长限制，使 s 恒不超过 L。
    MPCControlBounds bounds = command_rate_control_bounds(
        x, command_capability_, k == 0 ? &p_.follow.start_command : nullptr
    );
    const double envelope_max = command_capability_.command_envelope.velocity.max;
    const double remaining_rate = positive_part(total_length_ - x(ix::PATH_PROGRESS)) / MPC_DT;
    bounds.upper(iu::PATH_SPEED_CMD) = std::min(envelope_max, remaining_rate);
    if (remaining_rate < envelope_max) {
        // 剩余弧长随 s 递减，因此该上界对 s 的雅可比是 −1/Δt。
        bounds.upper_state_jacobian(iu::PATH_SPEED_CMD, ix::PATH_PROGRESS) = -1.0 / MPC_DT;
    }
    return bounds;
}

template<int Horizon>
const CostMap& FollowProblemT<Horizon>::cost_map_for_step(const int k) const {
    // 时域向量约定：下标 0 是当前帧，下标 i 对应 t0 + i·prediction_dt。
    // stage k 位于 t0 + k·MPC_DT，取时间上不晚于它的最近一帧。
    if (step_cost_maps_.size() <= 1) return *step_cost_maps_[0];
    const int index = static_cast<int>(static_cast<double>(k) * MPC_DT / prediction_dt_);
    return *step_cost_maps_[static_cast<size_t>(
        std::min(index, static_cast<int>(step_cost_maps_.size()) - 1)
    )];
}

template<int Horizon>
double FollowProblemT<Horizon>::running_cost(
    const int k, const StateVec& x, const ControlVec& u
) const {
    const FollowResidualVec residual = follow_residual_impl(
        x, u, trajectory_, speed_profile_, p_, cost_map_for_step(k),
        command_capability_.command_dynamics, model_, step_constraint_schedule_
    );
    // 进度奖励：s 的硬上界已保证不会越界，因此线性奖励无需再做剩余弧长截断。
    return residual_cost(residual)
        - p_.follow.progress.progress_reward * u(iu::PATH_SPEED_CMD);
}

template<int Horizon>
void FollowProblemT<Horizon>::running_cost_derivatives(
    const int k,
    const StateVec& x,
    const ControlVec& u,
    StateVec& lx,
    ControlVec& lu,
    MatXX& lxx,
    Eigen::Matrix<double, MPC_NU, MPC_NX>& lux,
    Eigen::Matrix<double, MPC_NU, MPC_NU>& luu
) const {
    const auto& cost_map = cost_map_for_step(k);
    const double path_progress = std::clamp(x(ix::PATH_PROGRESS), 0.0, total_length_);
    FrozenStepGate frozen_step_gate;
    frozen_step_gate.constraint = step_constraint_schedule_
        ? step_constraint_schedule_->constraint_at(path_progress)
        : nullptr;
    if (frozen_step_gate.constraint) {
        frozen_step_gate.gate = step_window_gate(
            path_progress, *frozen_step_gate.constraint
        );
    }
    auto residual_fn = [&](const StateVec& state, const ControlVec& control) {
        return follow_residual_impl(
            state, control, trajectory_, speed_profile_, p_, cost_map,
            command_capability_.command_dynamics, model_, step_constraint_schedule_,
            &frozen_step_gate
        );
    };
    gauss_newton_running_derivatives<FOLLOW_RESIDUAL_DIM>(
        residual_fn, x, u, lx, lu, lxx, lux, luu
    );
    lu(iu::PATH_SPEED_CMD) -= p_.follow.progress.progress_reward;
}

template<int Horizon>
double FollowProblemT<Horizon>::terminal_cost(const StateVec&) const {
    return 0.0;
}

template<int Horizon>
void FollowProblemT<Horizon>::terminal_cost_derivatives(
    const StateVec&,
    StateVec& lfx,
    MatXX& lfxx
) const {
    lfx.setZero();
    lfxx.setZero();
}

template<int Horizon>
std::optional<RolloutLethalObstacleInfo> FollowProblemT<Horizon>::detect_lethal_obstacle(
    const int state_index,
    const StateVec& x,
    double* out_cost_value
) const {
    const auto& safety = p_.follow.rollout_safety;
    if (!safety.enable_lethal_obstacle_check) return std::nullopt;

    const auto sample = sample_cost_or_lethal(
        masked_global_map_, {x(ix::X), x(ix::Y)}
    );
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
