#pragma once

#include <vector>

#include <nav_executor/path_executor/solver/beam_search.hpp>
#include <nav_executor/path_executor/solver/trajectory_seed.hpp>

namespace nav_executor {

struct GlobalSearchResult {
    std::vector<TrajectorySeed> candidates;
    std::vector<std::vector<Eigen::Vector2d>> debug_paths;
};

class GlobalSearch {
public:
    explicit GlobalSearch(GlobalSearchParams params): beam_search_(params.beam) {}

    [[nodiscard]] GlobalSearchResult search(
        const FollowProblem& problem,
        const StateVec& x0,
        const TrajectorySeed& warm_seed,
        const TrajectorySeed& longitudinal_seed,
        int candidate_count
    ) const;

private:
    BeamSearch beam_search_;
};

} // namespace nav_executor
