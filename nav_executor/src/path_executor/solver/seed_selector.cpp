#include <nav_executor/path_executor/solver/seed_selector.hpp>

#include <cmath>

namespace nav_executor {

std::optional<std::pair<TrajectorySeed, double>> SeedSelector::refine(
    const FollowProblem& problem,
    const StateVec& x0,
    const TrajectorySeed& seed
) const {
    const auto evaluation = evaluate_seed(problem, seed, x0);
    if (!evaluation.valid) return std::nullopt;

    fddp::Solver<FollowProblem> solver;
    solver.us = seed.controls;
    solver.xs = evaluation.states;
    fddp::SolverOptions options;
    options.max_iters = params_.refinement_iterations;
    options.tol_grad = SOLVER_TOL_GRAD;
    options.tol_cost = SOLVER_TOL_COST;
    solver.solve(problem, options);

    TrajectorySeed refined = seed;
    refined.controls = solver.us;
    const auto refined_evaluation = evaluate_seed(problem, refined, x0);
    if (!refined_evaluation.valid) return std::nullopt;
    return std::pair {std::move(refined), refined_evaluation.cost};
}

std::optional<TrajectorySeed> SeedSelector::select(
    const FollowProblem& problem,
    const StateVec& x0,
    const TrajectorySeed& incumbent,
    const std::vector<TrajectorySeed>& challengers
) {
    const auto incumbent_result = refine(problem, x0, incumbent);
    if (!incumbent_result) {
        consecutive_improvements_ = 0;
        return std::nullopt;
    }

    std::optional<std::pair<TrajectorySeed, double>> best;
    for (const auto& challenger : challengers) {
        auto refined = refine(problem, x0, challenger);
        if (refined && (!best || refined->second < best->second)) best = std::move(refined);
    }
    const double required_gain = std::max(0.0, params_.improvement_margin) * std::max(std::abs(incumbent_result->second), 1.0);
    if (!best || best->second + required_gain >= incumbent_result->second) {
        consecutive_improvements_ = 0;
        return std::nullopt;
    }
    ++consecutive_improvements_;
    if (consecutive_improvements_ < std::max(params_.hysteresis_count, 1)) return std::nullopt;
    consecutive_improvements_ = 0;
    return std::move(best->first);
}

} // namespace nav_executor
