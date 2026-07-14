#include <nav_executor/path_executor/solver/coarse_search_model.hpp>

#include <algorithm>
#include <cmath>

#include <nav_executor/path_executor/solver/mpc_utils.hpp>

namespace nav_executor {
namespace {

Eigen::Vector2d quadratic_position(
    const Eigen::Vector2d& start,
    const Eigen::Vector2d& middle,
    const Eigen::Vector2d& end,
    const double t
) {
    const double w0 = 2.0 * (t - 0.5) * (t - 1.0);
    const double wm = -4.0 * t * (t - 1.0);
    const double w1 = 2.0 * t * (t - 0.5);
    return w0 * start + wm * middle + w1 * end;
}

} // namespace

CoarseSearchModel::DynamicTrace CoarseSearchModel::predict_dynamic_trace(
    const StateVec& state,
    const std::array<ControlVec, MAX_SEARCH_MACRO_STEPS>& controls,
    const int step_count,
    const double nonlinear_velocity,
    const double nonlinear_omega
) const {
    const auto& model = problem_.discrete_model();
    DynamicTrace trace;
    trace.states[0] = DynamicState {
        .xh = state(ix::XH),
        .velocity = state(ix::V),
        .omega = state(ix::W),
    };
    double velocity_command = state(ix::DV);
    double omega_command = state(ix::DW);

    for (int i = 0; i < step_count; ++i) {
        const auto& current = trace.states[static_cast<size_t>(i)];
        const DynamicState next {
            .xh = model.ad00 * current.xh + model.ad01 * current.velocity
                + model.bd0 * velocity_command + model.gd0 * nonlinear_velocity,
            .velocity = model.ad10 * current.xh + model.ad11 * current.velocity
                + model.bd1 * velocity_command + model.gd1 * nonlinear_velocity,
            .omega = model.alpha_w * current.omega + model.beta_w * omega_command
                - model.gamma_w * nonlinear_omega,
        };
        trace.states[static_cast<size_t>(i + 1)] = next;
        velocity_command = controls[static_cast<size_t>(i)](0);
        omega_command = controls[static_cast<size_t>(i)](1);
    }
    return trace;
}

double CoarseSearchModel::path_progress_rate(const StateVec& state) const {
    const auto& path = problem_.reference_path();
    const double path_u = SplinePath::clamp_u_extrapolated(state(ix::PATH_U));
    const auto reference = path.eval(path_u);
    const double ex = state(ix::X) - reference.p.x();
    const double ey_world = state(ix::Y) - reference.p.y();
    const double lateral_error = -ex * reference.sin_r + ey_world * reference.cos_r;
    const double heading_error = wrap_pi(state(ix::THETA) - reference.thetar);
    const double denominator_raw = 1.0 - reference.kappa * lateral_error;
    const double denominator = std::sqrt(denominator_raw * denominator_raw + 0.05 * 0.05);
    const double ds_du = std::sqrt(reference.d1.squaredNorm() + 0.01) + 1e-6;
    return state(ix::V) * std::cos(heading_error) / (denominator * ds_du);
}

CoarseTransition CoarseSearchModel::transition(
    const StateVec& state,
    const double velocity_acceleration,
    const double omega_acceleration,
    const int fine_step,
    const int step_count
) const {
    CoarseTransition result;
    if (step_count <= 0 || step_count > MAX_SEARCH_MACRO_STEPS || !state.allFinite()) return result;

    const auto& profile = problem_.capability_profile();
    const auto& bounds = profile.command_bounds;
    const auto& limits = profile.motion_constraints;
    ControlVec command(state(ix::DV), state(ix::DW));
    double lateral_violation = std::max(std::abs(command(0) * command(1)) - limits.a_lat_max, 0.0);
    for (int i = 0; i < step_count; ++i) {
        command(0) = std::clamp(
            command(0) + velocity_acceleration * MPC_DT,
            bounds.vel_min,
            bounds.vel_max
        );
        command(1) = std::clamp(
            command(1) + omega_acceleration * MPC_DT,
            bounds.omega_min,
            bounds.omega_max
        );
        const double next_lateral_violation = std::max(
            std::abs(command(0) * command(1)) - limits.a_lat_max,
            0.0
        );
        if (next_lateral_violation > lateral_violation + 1e-9) return result;
        lateral_violation = next_lateral_violation;
        result.controls[static_cast<size_t>(i)] = command;
    }
    result.control_count = step_count;

    const auto& model = problem_.discrete_model();
    const auto initial_nonlinearity = evaluate_lpv_nonlinear(state(ix::V), state(ix::W), model);
    const int half_steps = std::max(1, step_count / 2);
    const auto predicted_trace = predict_dynamic_trace(
        state, result.controls, step_count, initial_nonlinearity.nl, initial_nonlinearity.sw
    );
    const auto& predicted_middle = predicted_trace.states[static_cast<size_t>(half_steps)];
    const auto middle_nonlinearity = evaluate_lpv_nonlinear(
        predicted_middle.velocity,
        predicted_middle.omega,
        model
    );
    const auto corrected_trace = predict_dynamic_trace(
        state, result.controls, step_count, middle_nonlinearity.nl, middle_nonlinearity.sw
    );
    const auto& middle = corrected_trace.states[static_cast<size_t>(half_steps)];
    const auto& end = corrected_trace.states[static_cast<size_t>(step_count)];

    const double half_dt = static_cast<double>(half_steps) * MPC_DT;
    const double total_dt = static_cast<double>(step_count) * MPC_DT;
    const double remaining_dt = total_dt - half_dt;
    const double theta_middle = state(ix::THETA) + 0.5 * half_dt * (state(ix::W) + middle.omega);
    const double theta_end = theta_middle + 0.5 * remaining_dt * (middle.omega + end.omega);
    const Eigen::Vector2d position_start(state(ix::X), state(ix::Y));
    const Eigen::Vector2d velocity_start(
        state(ix::V) * std::cos(state(ix::THETA)),
        state(ix::V) * std::sin(state(ix::THETA))
    );
    const Eigen::Vector2d velocity_middle(
        middle.velocity * std::cos(theta_middle),
        middle.velocity * std::sin(theta_middle)
    );
    const Eigen::Vector2d velocity_end(
        end.velocity * std::cos(theta_end),
        end.velocity * std::sin(theta_end)
    );
    const Eigen::Vector2d position_middle = position_start + 0.5 * half_dt * (velocity_start + velocity_middle);
    const Eigen::Vector2d position_end = position_middle + 0.5 * remaining_dt * (velocity_middle + velocity_end);

    StateVec middle_state = state;
    middle_state(ix::X) = position_middle.x();
    middle_state(ix::Y) = position_middle.y();
    middle_state(ix::THETA) = theta_middle;
    middle_state(ix::XH) = middle.xh;
    middle_state(ix::V) = middle.velocity;
    middle_state(ix::W) = middle.omega;
    middle_state(ix::DV) = result.controls[static_cast<size_t>(half_steps - 1)](0);
    middle_state(ix::DW) = result.controls[static_cast<size_t>(half_steps - 1)](1);
    const auto& power_model = problem_.params().power_model;
    double energy = state(ix::ENERGY);
    for (int i = 1; i <= half_steps; ++i) {
        const auto& previous = corrected_trace.states[static_cast<size_t>(i - 1)];
        const auto& current = corrected_trace.states[static_cast<size_t>(i)];
        const double power = predict_power(
            power_model,
            current.velocity,
            current.omega,
            (current.velocity - previous.velocity) / MPC_DT,
            (current.omega - previous.omega) / MPC_DT
        );
        energy += (problem_.charge_power_limit() - power) * MPC_DT;
    }
    middle_state(ix::ENERGY) = energy;
    middle_state(ix::PATH_U) = std::clamp(
        state(ix::PATH_U) + half_dt * path_progress_rate(middle_state),
        SplinePath::U_EXTRAP_MIN,
        SplinePath::U_EXTRAP_MAX
    );

    result.state = state;
    result.state(ix::X) = position_end.x();
    result.state(ix::Y) = position_end.y();
    result.state(ix::THETA) = theta_end;
    result.state(ix::XH) = end.xh;
    result.state(ix::V) = end.velocity;
    result.state(ix::W) = end.omega;
    result.state(ix::DV) = result.controls[static_cast<size_t>(step_count - 1)](0);
    result.state(ix::DW) = result.controls[static_cast<size_t>(step_count - 1)](1);
    for (int i = half_steps + 1; i <= step_count; ++i) {
        const auto& previous = corrected_trace.states[static_cast<size_t>(i - 1)];
        const auto& current = corrected_trace.states[static_cast<size_t>(i)];
        const double power = predict_power(
            power_model,
            current.velocity,
            current.omega,
            (current.velocity - previous.velocity) / MPC_DT,
            (current.omega - previous.omega) / MPC_DT
        );
        energy += (problem_.charge_power_limit() - power) * MPC_DT;
    }
    result.state(ix::ENERGY) = energy;
    result.state(ix::PATH_U) = std::clamp(
        state(ix::PATH_U) + total_dt * path_progress_rate(middle_state),
        SplinePath::U_EXTRAP_MIN,
        SplinePath::U_EXTRAP_MAX
    );
    if (!result.state.allFinite()) return result;

    for (int i = 1; i <= step_count; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(step_count);
        StateVec collision_state = (1.0 - t) * state + t * result.state;
        const Eigen::Vector2d position = quadratic_position(position_start, position_middle, position_end, t);
        collision_state(ix::X) = position.x();
        collision_state(ix::Y) = position.y();
        if (problem_.detect_lethal_obstacle(fine_step + i, collision_state).has_value()) return result;
    }

    const int middle_control_index = std::min(half_steps, step_count - 1);
    result.running_cost = static_cast<double>(step_count) * problem_.running_cost_value_only(
        fine_step + half_steps,
        middle_state,
        result.controls[static_cast<size_t>(middle_control_index)]
    );
    result.valid = std::isfinite(result.running_cost);
    return result;
}

} // namespace nav_executor
