#include <nav_executor/path_executor/solver/follow_problem.hpp>

#include <algorithm>
#include <cmath>

#include <nav_executor/path_executor/solver/mpc_utils.hpp>

namespace nav_executor {

namespace {

constexpr double COST_EPS = 1e-9;
constexpr int FOLLOW_RESIDUAL_DIM = 21;
using FollowResidualVec = Eigen::Matrix<double, FOLLOW_RESIDUAL_DIM, 1>;

struct ReferenceFrame {
    TrajSample sample;
    double phase_time = 0.0;
    double tau = 0.0;
    double nominal_velocity = 0.0;
    double nominal_angular_velocity = 0.0;
    Eigen::Vector2d tangent = Eigen::Vector2d::UnitX();
};

struct FrozenStepGate {
    const StepTraversalConstraint* constraint = nullptr;
    double gate = 0.0;
};

ReferenceFrame reference_frame(
    const MincoTrajectory& trajectory,
    const double requested_phase_time,
    const double tangent_blend_speed_scale
) {
    ReferenceFrame reference;
    const double total_time = trajectory.total_time();
    reference.phase_time = std::clamp(requested_phase_time, 0.0, total_time);
    reference.tau = total_time > COST_EPS ? reference.phase_time / total_time : 0.0;
    reference.sample = trajectory.eval_time(reference.phase_time);
    reference.nominal_velocity = trajectory.longitudinal_velocity(reference.sample);
    reference.nominal_angular_velocity = trajectory.angular_velocity(reference.sample);

    const double translational_speed = total_time > COST_EPS
        ? reference.sample.ds_dtau / total_time
        : 0.0;
    const Eigen::Vector2d heading_axis(
        std::cos(reference.sample.theta), std::sin(reference.sample.theta)
    );
    Eigen::Vector2d path_axis = heading_axis;
    double axis_alignment = 1.0;
    if (reference.sample.ds_dtau > COST_EPS) {
        path_axis = reference.sample.dp_dtau / reference.sample.ds_dtau;
        axis_alignment = path_axis.dot(heading_axis);
        if (axis_alignment < 0.0) path_axis = -path_axis;
    }
    const double scale_squared = tangent_blend_speed_scale * tangent_blend_speed_scale;
    const double speed_blend = translational_speed * translational_speed
        / (translational_speed * translational_speed + scale_squared);
    // 路径轴是模 pi 的无向轴。接近与车身轴垂直时让其权重连续退化为零，避免选取
    // 正/反代表的符号切换污染有限差分；正常非完整轨迹上 alignment≈±1。
    constexpr double ALIGNMENT_BLEND_SCALE = 0.1;
    const double alignment_squared = axis_alignment * axis_alignment;
    const double alignment_blend = alignment_squared
        / (alignment_squared + ALIGNMENT_BLEND_SCALE * ALIGNMENT_BLEND_SCALE);
    const double blend = speed_blend * alignment_blend;
    reference.tangent = ((1.0 - blend) * heading_axis + blend * path_axis).normalized();
    return reference;
}

FollowResidualVec follow_residual_impl(
    const StateVec& x,
    const ControlVec& u,
    const MincoTrajectory& trajectory,
    const MPCParams& params,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info,
    const CommandDynamicsLimits& motion_limits,
    const std::shared_ptr<const StepConstraintSchedule>& step_schedule,
    const FrozenStepGate* frozen_step_gate = nullptr
) {
    const auto& follow = params.follow;
    const auto& tracking = follow.tracking_weights;
    const auto& phase = follow.phase;
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
    const double phase_rate_cmd = u(iu::PHASE_RATE_CMD);
    const double dv_cmd = MPC_DT * u(iu::V_CMD_RATE);
    const double domega_cmd = MPC_DT * u(iu::W_CMD_RATE);

    const ReferenceFrame reference = reference_frame(
        trajectory, x(ix::PHASE_TIME), tracking.tangent_blend_speed_scale
    );
    const Eigen::Vector2d position_error(px - reference.sample.p.x(), py - reference.sample.p.y());
    const Eigen::Vector2d normal(-reference.tangent.y(), reference.tangent.x());
    residual(0) = tracking.contour * normal.dot(position_error);
    residual(1) = tracking.lag * reference.tangent.dot(position_error);

    residual(2) = tracking.heading * wrap_pi(theta - reference.sample.theta);
    residual(3) = tracking.velocity
        * (v_actual - phase_rate_cmd * reference.nominal_velocity);
    residual(4) = tracking.angular_velocity
        * (omega_actual - phase_rate_cmd * reference.nominal_angular_velocity);
    residual(5) = phase.rate_tracking_weight * (phase_rate_cmd - phase.nominal_rate);
    residual(6) = phase.rate_smoothness_weight * (phase_rate_cmd - x(ix::PHASE_RATE));
    residual(7) = phase.overshoot_weight
        * positive_part(x(ix::PHASE_TIME) - trajectory.total_time());

    residual(8) = command.r_v * v_cmd;
    residual(9) = command.r_omega * omega_cmd;
    residual(10) = command.r_dv * dv_cmd;
    residual(11) = command.r_domega * domega_cmd;
    residual(12) = command.r_jerk_v
        * (u(iu::V_CMD_RATE) - x(ix::V_CMD_RATE)) / MPC_DT;
    residual(13) = command.r_jerk_omega
        * (u(iu::W_CMD_RATE) - x(ix::W_CMD_RATE)) / MPC_DT;

    residual(14) = dynamics_weights.lateral_acceleration
        * positive_part(
            std::abs(v_cmd * omega_cmd) - motion_limits.lateral_acceleration_max
        );

    const auto cost_sample = eval_cost_bilinear(cost_grid, cost_info, px, py);
    residual(15) = environment.obstacle * cost_sample.value / 255.0;

    const StepTraversalConstraint* const step = frozen_step_gate
        ? frozen_step_gate->constraint
        : (step_schedule ? step_schedule->constraint_at(reference.tau) : nullptr);
    if (step) {
        const double gate = frozen_step_gate
            ? frozen_step_gate->gate
            : step_window_gate(reference.tau, *step);
        if (gate > 0.0) {
            const Eigen::Vector2d& direction = step->dir_map;
            const double cross = std::cos(theta) * direction.y() - std::sin(theta) * direction.x();
            residual(16) = traversal_weights.direction * gate * std::abs(cross);

            const auto& target = step->velocity_window;
            const double velocity_error = v_actual < target.min
                ? target.min - v_actual
                : (v_actual > target.max ? v_actual - target.max : 0.0);
            residual(17) = traversal_weights.velocity * gate * velocity_error;
            residual(18) = traversal_weights.angular_velocity * gate * omega_cmd;
            residual(19) = traversal_weights.velocity_smoothness * gate * dv_cmd;
            residual(20) = traversal_weights.angular_velocity_smoothness
                * gate * domega_cmd;
        }
    }

    return residual;
}

constexpr int FOLLOW_TERMINAL_RESIDUAL_DIM = 7;
using FollowTerminalResidualVec = Eigen::Matrix<double, FOLLOW_TERMINAL_RESIDUAL_DIM, 1>;

FollowTerminalResidualVec follow_terminal_residual_impl(
    const StateVec& x,
    const MincoTrajectory& trajectory,
    const MPCParams& params
) {
    const auto& weights = params.follow.terminal_weights;
    const ReferenceFrame reference = reference_frame(
        trajectory,
        x(ix::PHASE_TIME),
        params.follow.tracking_weights.tangent_blend_speed_scale
    );

    FollowTerminalResidualVec residual = FollowTerminalResidualVec::Zero();
    residual(0) = weights.position * (x(ix::X) - reference.sample.p.x());
    residual(1) = weights.position * (x(ix::Y) - reference.sample.p.y());
    residual(2) = weights.heading * wrap_pi(x(ix::THETA) - reference.sample.theta);
    residual(3) = weights.velocity
        * (x(ix::V) - x(ix::PHASE_RATE) * reference.nominal_velocity);
    residual(4) = weights.angular_velocity
        * (x(ix::W) - x(ix::PHASE_RATE) * reference.nominal_angular_velocity);
    residual(5) = weights.overshoot
        * positive_part(x(ix::PHASE_TIME) - trajectory.total_time());
    residual(6) = weights.overshoot
        * positive_part(-x(ix::PHASE_TIME));
    return residual;
}

} // anonymous namespace

template<int Horizon>
FollowProblemT<Horizon>::FollowProblemT(
    MincoTrajectory trajectory,
    const MPCParams& params,
    const std::vector<CostMapGridView>& per_step_cost_grids,
    const GridInfo& cost_info,
    const CostMapGridView& masked_global_grid,
    const double prediction_dt,
    const double schedule_rho,
    const CapabilityProfile& effective_capability,
    std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule
):
    trajectory_(std::move(trajectory)),
    p_(params),
    step_cost_grids_(per_step_cost_grids),
    cost_info_(cost_info),
    masked_global_grid_(masked_global_grid),
    prediction_dt_(prediction_dt),
    model_(build_lpv_discrete_model(params.kinematic_model, schedule_rho)),
    effective_capability_(effective_capability),
    step_constraint_schedule_(std::move(step_constraint_schedule)),
    total_time_(trajectory_.total_time()) {}

template<int Horizon>
StateVec FollowProblemT<Horizon>::dynamics(int, const StateVec& x, const ControlVec& u) const {
    StateVec next = mpc_dynamics(x, u, model_);
    next(ix::PHASE_TIME) = x(ix::PHASE_TIME) + MPC_DT * u(iu::PHASE_RATE_CMD);
    next(ix::PHASE_RATE) = u(iu::PHASE_RATE_CMD);
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
    fx.row(ix::PHASE_TIME).setZero();
    fx(ix::PHASE_TIME, ix::PHASE_TIME) = 1.0;
    fu.row(ix::PHASE_TIME).setZero();
    fu(ix::PHASE_TIME, iu::PHASE_RATE_CMD) = MPC_DT;

    fx.row(ix::PHASE_RATE).setZero();
    fu.row(ix::PHASE_RATE).setZero();
    fu(ix::PHASE_RATE, iu::PHASE_RATE_CMD) = 1.0;
}

template<int Horizon>
MPCControlBounds FollowProblemT<Horizon>::control_bounds(const int k, const StateVec& x) const {
    return command_rate_control_bounds(
        x,
        effective_capability_,
        p_.follow.phase.rate_min,
        p_.follow.phase.rate_max,
        k == 0 ? &p_.follow.start_command : nullptr
    );
}

template<int Horizon>
const CostMapGridView& FollowProblemT<Horizon>::cost_grid_for_step(const int k) const {
    if (step_cost_grids_.size() <= 1) return step_cost_grids_[0];
    const int index = static_cast<int>(static_cast<double>(k) * MPC_DT / prediction_dt_);
    return step_cost_grids_[static_cast<size_t>(
        std::min(index, static_cast<int>(step_cost_grids_.size()) - 1)
    )];
}

template<int Horizon>
double FollowProblemT<Horizon>::running_cost(const int k, const StateVec& x, const ControlVec& u) const {
    return running_cost_value_only(k, x, u);
}

template<int Horizon>
double FollowProblemT<Horizon>::running_cost_value_only(
    const int k,
    const StateVec& x,
    const ControlVec& u,
    double*
) const {
    const FollowResidualVec residual = follow_residual_impl(
        x, u, trajectory_, p_, cost_grid_for_step(k), cost_info_,
        effective_capability_.command_dynamics, step_constraint_schedule_
    );
    double cost = residual_cost(residual);
    const double remaining_phase = positive_part(total_time_ - x(ix::PHASE_TIME));
    const double requested_advance = MPC_DT * u(iu::PHASE_RATE_CMD);
    if (remaining_phase > 0.0 && requested_advance > 0.0) {
        const double realized_advance = std::min(requested_advance, remaining_phase);
        cost -= p_.follow.phase.progress_reward * realized_advance / MPC_DT;
    }
    return cost;
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
    const auto& cost_grid = cost_grid_for_step(k);
    const double phase_time = std::clamp(x(ix::PHASE_TIME), 0.0, total_time_);
    const double tau = total_time_ > COST_EPS ? phase_time / total_time_ : 0.0;
    FrozenStepGate frozen_step_gate;
    frozen_step_gate.constraint = step_constraint_schedule_
        ? step_constraint_schedule_->constraint_at(tau)
        : nullptr;
    if (frozen_step_gate.constraint) {
        frozen_step_gate.gate = step_window_gate(tau, *frozen_step_gate.constraint);
    }
    auto residual_fn = [&](const StateVec& state, const ControlVec& control) {
        return follow_residual_impl(
            state, control, trajectory_, p_, cost_grid, cost_info_,
            effective_capability_.command_dynamics, step_constraint_schedule_, &frozen_step_gate
        );
    };
    gauss_newton_running_derivatives<FOLLOW_RESIDUAL_DIM>(
        residual_fn, x, u, lx, lu, lxx, lux, luu
    );
    const double remaining_phase = positive_part(total_time_ - x(ix::PHASE_TIME));
    const double requested_advance = MPC_DT * u(iu::PHASE_RATE_CMD);
    if (remaining_phase > 0.0) {
        if (requested_advance < remaining_phase) {
            lu(iu::PHASE_RATE_CMD) -= p_.follow.phase.progress_reward;
        } else {
            lx(ix::PHASE_TIME) += p_.follow.phase.progress_reward / MPC_DT;
        }
    }
}

template<int Horizon>
double FollowProblemT<Horizon>::terminal_cost(const StateVec& x) const {
    double cost = residual_cost(follow_terminal_residual_impl(x, trajectory_, p_));
    cost += p_.follow.terminal_weights.remaining_phase
        * positive_part(total_time_ - x(ix::PHASE_TIME));
    return cost;
}

template<int Horizon>
void FollowProblemT<Horizon>::terminal_cost_derivatives(
    const StateVec& x,
    StateVec& lfx,
    MatXX& lfxx
) const {
    auto residual_fn = [&](const StateVec& state) {
        return follow_terminal_residual_impl(state, trajectory_, p_);
    };
    gauss_newton_terminal_derivatives<FOLLOW_TERMINAL_RESIDUAL_DIM>(
        residual_fn, x, lfx, lfxx
    );
    if (x(ix::PHASE_TIME) < total_time_) {
        lfx(ix::PHASE_TIME) -= p_.follow.terminal_weights.remaining_phase;
    }
}

template<int Horizon>
std::optional<RolloutLethalObstacleInfo> FollowProblemT<Horizon>::detect_lethal_obstacle(
    const int state_index,
    const StateVec& x,
    double* out_cost_value
) const {
    const auto& safety = p_.follow.rollout_safety;
    if (!safety.enable_lethal_obstacle_check) return std::nullopt;

    const auto sample = eval_cost_bilinear(
        masked_global_grid_, cost_info_, x(ix::X), x(ix::Y)
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
