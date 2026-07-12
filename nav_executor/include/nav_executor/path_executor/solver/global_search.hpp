#pragma once

#include <vector>

#include <nav_executor/path_executor/solver/cem_optimizer.hpp>
#include <nav_executor/path_executor/solver/trajectory_seed.hpp>

namespace nav_executor {

struct GlobalSearchResult {
    std::vector<TrajectorySeed> candidates;
    std::vector<std::vector<Eigen::Vector2d>> debug_paths;
};

class GlobalSearch {
public:
    explicit GlobalSearch(GlobalSearchParams params): optimizer_(params) {}

    [[nodiscard]] GlobalSearchResult search(
        const FollowProblem& problem,
        const StateVec& x0,
        const TrajectorySeed& warm_seed,
        const TrajectorySeed& longitudinal_seed,
        int candidate_count
    ) const;

private:
    CEMOptimizer optimizer_;
};

} // namespace nav_executor
