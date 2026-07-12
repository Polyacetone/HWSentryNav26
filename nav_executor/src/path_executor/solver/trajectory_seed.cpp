#include <nav_executor/path_executor/solver/trajectory_seed.hpp>

#include <cmath>

namespace nav_executor {

SeedEvaluation evaluate_seed(
    const FollowProblem& problem,
    const TrajectorySeed& seed,
    const StateVec& x0,
    const bool reject_lethal
) {
    SeedEvaluation result;
    result.states[0] = x0;
    result.cost = 0.0;

    for (int k = 0; k < MPC_HORIZON; ++k) {
        const size_t index = static_cast<size_t>(k);
        const ControlVec control = seed.controls[index].cwiseMax(problem.u_lower()).cwiseMin(problem.u_upper());
        result.cost += problem.running_cost_value_only(k, result.states[index], control);
        result.states[index + 1] = problem.dynamics(k, result.states[index], control);
        if (!std::isfinite(result.cost) || !result.states[index + 1].allFinite()) {
            return result;
        }
        if (reject_lethal && problem.detect_lethal_obstacle(k + 1, result.states[index + 1]).has_value()) {
            return result;
        }
    }

    result.cost += problem.terminal_cost(result.states[MPC_HORIZON]);
    result.valid = std::isfinite(result.cost);
    return result;
}

} // namespace nav_executor
