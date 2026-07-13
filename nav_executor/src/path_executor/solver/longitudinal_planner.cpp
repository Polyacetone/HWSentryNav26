#include <nav_executor/path_executor/solver/longitudinal_planner.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace nav_executor {
namespace {

struct DPNode {
    double cost = std::numeric_limits<double>::infinity();
    int previous = -1;
    double acceleration = 0.0;
};

int nearest_cell(const double value, const double lower, const double step, const int count) {
    return std::clamp(static_cast<int>(std::lround((value - lower) / step)), 0, count - 1);
}

} // namespace

TrajectorySeed LongitudinalPlanner::plan(
    const FollowProblem& problem,
    const SplinePath& path,
    const StateVec& x0,
    const std::optional<ActiveStepMode> step_mode,
    const uint64_t sequence
) const {
    TrajectorySeed seed;
    seed.source = SeedSource::LONGITUDINAL;
    seed.origin_seq = sequence;

    const auto& profile = problem.capability_profile();
    const int s_count = std::max(params_.progress_cells, 3);
    const int v_count = std::max(params_.velocity_cells, 3);
    const int a_count = std::max(params_.acceleration_samples, 2);
    const double v_min = profile.command_bounds.vel_min;
    const double v_max = profile.command_bounds.vel_max;
    const double v_step = (v_max - v_min) / static_cast<double>(v_count - 1);
    const double path_length = std::max(path.arc_length(0.0, 1.0, 80), 0.1);
    const double current_path_u = std::clamp(x0(ix::PATH_U), 0.0, 1.0);
    const double current_s = path.arc_length(0.0, current_path_u, 40);
    const double s_lower = std::max(0.0, current_s - params_.reverse_clearance);
    const double s_upper = std::min(path_length, current_s + std::max(v_max, 0.1) * MPC_DT * MPC_HORIZON);
    const double s_step = std::max((s_upper - s_lower) / static_cast<double>(s_count - 1), 1e-4);
    const int cell_count = s_count * v_count;

    std::vector<std::vector<DPNode>> layers(MPC_HORIZON + 1, std::vector<DPNode>(static_cast<size_t>(cell_count)));
    const int initial_s = nearest_cell(current_s, s_lower, s_step, s_count);
    const int initial_v = nearest_cell(x0(ix::V), v_min, v_step, v_count);
    layers[0][static_cast<size_t>(initial_s * v_count + initial_v)].cost = 0.0;

    const double entry_s = step_mode
        ? path.arc_length(0.0, std::clamp(step_mode->commit_u, 0.0, 1.0), 40)
        : std::numeric_limits<double>::infinity();
    const double acc_max = std::max(profile.motion_constraints.acc_max, 1e-3);

    for (int k = 0; k < MPC_HORIZON; ++k) {
        for (int si = 0; si < s_count; ++si) {
            for (int vi = 0; vi < v_count; ++vi) {
                const int index = si * v_count + vi;
                const auto& node = layers[static_cast<size_t>(k)][static_cast<size_t>(index)];
                if (!std::isfinite(node.cost)) continue;
                const double s = s_lower + static_cast<double>(si) * s_step;
                const double v = v_min + static_cast<double>(vi) * v_step;
                for (int ai = 0; ai < a_count; ++ai) {
                    const double acceleration = -acc_max + 2.0 * acc_max * static_cast<double>(ai) / static_cast<double>(a_count - 1);
                    const double next_v_raw = std::clamp(v + acceleration * MPC_DT, v_min, v_max);
                    const double next_s_raw = std::clamp(s + v * MPC_DT, s_lower, s_upper);
                    const int next_si = nearest_cell(next_s_raw, s_lower, s_step, s_count);
                    const int next_vi = nearest_cell(next_v_raw, v_min, v_step, v_count);
                    const int next_index = next_si * v_count + next_vi;
                    const double next_s = s_lower + static_cast<double>(next_si) * s_step;
                    const double next_v = v_min + static_cast<double>(next_vi) * v_step;
                    double transition_cost = -params_.progress_weight * (next_s - s);
                    transition_cost += params_.control_change_weight * acceleration * acceleration * MPC_DT;
                    if (step_mode && s < entry_s && next_s >= entry_s) {
                        const double violation = std::max(0.0, step_mode->speed_min - next_v)
                            + std::max(0.0, next_v - step_mode->speed_max);
                        transition_cost += params_.step_speed_violation_weight * violation * violation;
                    }
                    auto& next = layers[static_cast<size_t>(k + 1)][static_cast<size_t>(next_index)];
                    if (node.cost + transition_cost < next.cost) {
                        next.cost = node.cost + transition_cost;
                        next.previous = index;
                        next.acceleration = acceleration;
                    }
                }
            }
        }
    }

    int best_index = 0;
    for (int index = 1; index < cell_count; ++index) {
        if (layers[MPC_HORIZON][static_cast<size_t>(index)].cost < layers[MPC_HORIZON][static_cast<size_t>(best_index)].cost) {
            best_index = index;
        }
    }
    std::array<double, MPC_HORIZON> velocities {};
    for (int k = MPC_HORIZON; k > 0; --k) {
        const int vi = best_index % v_count;
        velocities[static_cast<size_t>(k - 1)] = v_min + static_cast<double>(vi) * v_step;
        best_index = layers[static_cast<size_t>(k)][static_cast<size_t>(best_index)].previous;
        if (best_index < 0) best_index = initial_s * v_count + initial_v;
    }

    double path_u = x0(ix::PATH_U);
    double heading = x0(ix::THETA);
    for (int k = 0; k < MPC_HORIZON; ++k) {
        const auto reference = path.eval(SplinePath::clamp_u_extrapolated(path_u));
        const double velocity = velocities[static_cast<size_t>(k)];
        const double heading_error = std::atan2(std::sin(reference.thetar - heading), std::cos(reference.thetar - heading));
        const double omega = std::clamp(
            reference.kappa * velocity + 2.0 * heading_error,
            profile.command_bounds.omega_min,
            profile.command_bounds.omega_max
        );
        seed.controls[static_cast<size_t>(k)] = ControlVec(velocity, omega);
        heading += omega * MPC_DT;
        path_u += velocity * MPC_DT / std::max(reference.ds_du, 1e-3);
    }
    return seed;
}

} // namespace nav_executor
