#include <nav_executor/path_executor/solver/follow_problem.hpp>

#include <algorithm>
#include <cmath>

#include <nav_executor/path_executor/solver/mpc_utils.hpp>

namespace nav_executor {

namespace {

constexpr double COST_EPS = 1e-9;
constexpr int FOLLOW_RESIDUAL_DIM = 22;
using FollowResidualVec = Eigen::Matrix<double, FOLLOW_RESIDUAL_DIM, 1>;

struct ReferenceFrame {
    TrajSample sample;
    double path_progress = 0.0;
    double tau = 0.0;
    double nominal_path_speed = 0.0;
    Eigen::Vector2d tangent = Eigen::Vector2d::UnitX();
};

struct FrozenStepGate {
    const StepTraversalConstraint* constraint = nullptr;
    double gate = 0.0;
};

ReferenceFrame reference_frame(
    const MincoTrajectory& trajectory,
    const double requested_path_progress,
    const double tangent_blend_speed_scale
) {
    ReferenceFrame reference;
    const double total_length = trajectory.total_arc_length();
    reference.path_progress = std::clamp(
        requested_path_progress, 0.0, total_length
    );
    reference.tau = trajectory.tau_at_arc_length(reference.path_progress);
    reference.sample = trajectory.eval(reference.tau);
    reference.nominal_path_speed = trajectory.nominal_path_speed(reference.sample);

    const double translational_speed = reference.nominal_path_speed;
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
        trajectory, x(ix::PATH_PROGRESS), tracking.tangent_blend_speed_scale
    );
    const Eigen::Vector2d position_error(px - reference.sample.p.x(), py - reference.sample.p.y());
    const Eigen::Vector2d normal(-reference.tangent.y(), reference.tangent.x());
    residual(0) = tracking.contour * normal.dot(position_error);
    residual(1) = tracking.lag * reference.tangent.dot(position_error);

    residual(2) = tracking.heading * wrap_pi(theta - reference.sample.theta);
    residual(3) = tracking.velocity
        * (v_actual - path_speed_cmd);
    residual(4) = tracking.angular_velocity
        * (omega_actual
            - trajectory.heading_rate_per_arc_length(reference.sample) * path_speed_cmd);
    residual(5) = progress.speed_tracking_weight
        * (path_speed_cmd - reference.nominal_path_speed);
    residual(6) = progress.speed_smoothness_weight
        * (path_speed_cmd - x(ix::PATH_SPEED));
    residual(7) = progress.overshoot_weight
        * positive_part(x(ix::PATH_PROGRESS) - trajectory.total_arc_length());

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
            residual(18) = traversal_weights.angular_velocity_command
                * gate * omega_cmd;
            residual(19) = traversal_weights.angular_velocity_predicted
                * gate * omega_actual;
            residual(20) = traversal_weights.velocity_command_smoothness
                * gate * dv_cmd;
            const double predicted_velocity_increment =
                mpc_dynamics(x, u, model)(ix::V) - v_actual;
            residual(21) = traversal_weights.velocity_predicted_smoothness
                * gate * predicted_velocity_increment;
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
        x(ix::PATH_PROGRESS),
        params.follow.tracking_weights.tangent_blend_speed_scale
    );

    FollowTerminalResidualVec residual = FollowTerminalResidualVec::Zero();
    residual(0) = weights.position * (x(ix::X) - reference.sample.p.x());
    residual(1) = weights.position * (x(ix::Y) - reference.sample.p.y());
    residual(2) = weights.heading * wrap_pi(x(ix::THETA) - reference.sample.theta);
    residual(3) = weights.velocity
        * (x(ix::V) - x(ix::PATH_SPEED));
    residual(4) = weights.angular_velocity
        * (x(ix::W)
            - trajectory.heading_rate_per_arc_length(reference.sample)
                * x(ix::PATH_SPEED));
    residual(5) = weights.overshoot
        * positive_part(x(ix::PATH_PROGRESS) - trajectory.total_arc_length());
    residual(6) = weights.overshoot
        * positive_part(-x(ix::PATH_PROGRESS));
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
    const CapabilityProfile& command_capability,
    const SignedVelocityBounds& path_speed_bounds,
    std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule
):
    trajectory_(std::move(trajectory)),
    p_(params),
    step_cost_grids_(per_step_cost_grids),
    cost_info_(cost_info),
    masked_global_grid_(masked_global_grid),
    prediction_dt_(prediction_dt),
    model_(build_lpv_discrete_model(params.kinematic_model, schedule_rho)),
    command_capability_(command_capability),
    path_speed_bounds_(path_speed_bounds),
    step_constraint_schedule_(std::move(step_constraint_schedule)),
    total_length_(trajectory_.total_arc_length()) {}

template<int Horizon>
StateVec FollowProblemT<Horizon>::dynamics(int, const StateVec& x, const ControlVec& u) const {
    StateVec next = mpc_dynamics(x, u, model_);
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
    MPCControlBounds bounds = command_rate_control_bounds(
        x,
        command_capability_,
        path_speed_bounds_.min,
        path_speed_bounds_.max,
        k == 0 ? &p_.follow.start_command : nullptr
    );
    return bounds;
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
        command_capability_.command_dynamics, model_, step_constraint_schedule_
    );
    double cost = residual_cost(residual);
    const double remaining_progress = positive_part(
        total_length_ - x(ix::PATH_PROGRESS)
    );
    const double requested_advance = MPC_DT * u(iu::PATH_SPEED_CMD);
    if (remaining_progress > 0.0 && requested_advance > 0.0) {
        const double realized_advance = std::min(requested_advance, remaining_progress);
        cost -= p_.follow.progress.progress_reward * realized_advance / MPC_DT;
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
    const double path_progress = std::clamp(
        x(ix::PATH_PROGRESS), 0.0, total_length_
    );
    const double tau = trajectory_.tau_at_arc_length(path_progress);
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
            command_capability_.command_dynamics, model_, step_constraint_schedule_,
            &frozen_step_gate
        );
    };
    gauss_newton_running_derivatives<FOLLOW_RESIDUAL_DIM>(
        residual_fn, x, u, lx, lu, lxx, lux, luu
    );
    const double remaining_progress = positive_part(
        total_length_ - x(ix::PATH_PROGRESS)
    );
    const double requested_advance = MPC_DT * u(iu::PATH_SPEED_CMD);
    if (remaining_progress > 0.0) {
        if (requested_advance < remaining_progress) {
            lu(iu::PATH_SPEED_CMD) -= p_.follow.progress.progress_reward;
        } else {
            lx(ix::PATH_PROGRESS) += p_.follow.progress.progress_reward / MPC_DT;
        }
    }
}

template<int Horizon>
double FollowProblemT<Horizon>::terminal_cost(const StateVec& x) const {
    double cost = residual_cost(follow_terminal_residual_impl(x, trajectory_, p_));
    cost += p_.follow.terminal_weights.remaining_progress
        * positive_part(total_length_ - x(ix::PATH_PROGRESS));
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
    if (x(ix::PATH_PROGRESS) < total_length_) {
        lfx(ix::PATH_PROGRESS) -= p_.follow.terminal_weights.remaining_progress;
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
