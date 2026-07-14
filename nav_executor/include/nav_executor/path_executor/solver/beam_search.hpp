#pragma once

#include <vector>

#include <nav_executor/path_executor/solver/coarse_search_model.hpp>
#include <nav_executor/path_executor/solver/trajectory_seed.hpp>

namespace nav_executor {

class BeamSearch {
public:
    explicit BeamSearch(GlobalSearchBeamParams params);

    [[nodiscard]] std::vector<TrajectorySeed> search(
        const FollowProblem& problem,
        const StateVec& x0
    ) const;

private:
    GlobalSearchBeamParams params_;
};

} // namespace nav_executor
