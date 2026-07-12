#pragma once

#include <optional>
#include <vector>

#include <nav_executor/path_executor/solver/fddp_solver.hpp>
#include <nav_executor/path_executor/solver/trajectory_seed.hpp>

namespace nav_executor {

struct SeedSelectorParams {
    double improvement_margin = 0.05;
    int hysteresis_count = 2;
    int refinement_iterations = 8;
};

class SeedSelector {
public:
    explicit SeedSelector(SeedSelectorParams params): params_(params) {}

    [[nodiscard]] std::optional<TrajectorySeed> select(
        const FollowProblem& problem,
        const StateVec& x0,
        const TrajectorySeed& incumbent,
        const std::vector<TrajectorySeed>& challengers
    );

private:
    [[nodiscard]] std::optional<std::pair<TrajectorySeed, double>> refine(
        const FollowProblem& problem,
        const StateVec& x0,
        const TrajectorySeed& seed
    ) const;

    SeedSelectorParams params_;
    int consecutive_improvements_ = 0;
};

} // namespace nav_executor
