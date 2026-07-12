#include <nav_executor/path_executor/solver/global_search.hpp>

#include <algorithm>

namespace nav_executor {

GlobalSearchResult GlobalSearch::search(
    const FollowProblem& problem,
    const StateVec& x0,
    const TrajectorySeed& warm_seed,
    const TrajectorySeed& longitudinal_seed,
    const int candidate_count
) const {
    GlobalSearchResult result;
    const std::array<const TrajectorySeed*, 2> initial_means {&warm_seed, &longitudinal_seed};
    const int mode_count = std::max(candidate_count, 0);
    result.candidates.reserve(static_cast<size_t>(mode_count));
    for (int mode_index = 0; mode_index < mode_count; ++mode_index) {
        const TrajectorySeed& initial_mean = *initial_means[static_cast<size_t>(mode_index) % initial_means.size()];
        const auto sampled = optimizer_.optimize(problem, x0, initial_mean.controls);
        if (!sampled.valid) continue;
        TrajectorySeed candidate;
        candidate.controls = sampled.us;
        candidate.source = SeedSource::GLOBAL;
        candidate.origin_seq = warm_seed.origin_seq;
        result.candidates.push_back(std::move(candidate));
    }
    std::sort(result.candidates.begin(), result.candidates.end(), [&](const auto& lhs, const auto& rhs) {
        return evaluate_seed(problem, lhs, x0).cost < evaluate_seed(problem, rhs, x0).cost;
    });
    result.debug_paths.reserve(result.candidates.size());
    for (const auto& candidate : result.candidates) {
        const auto evaluation = evaluate_seed(problem, candidate, x0);
        if (!evaluation.valid) continue;
        auto& path = result.debug_paths.emplace_back();
        path.reserve(MPC_HORIZON + 1);
        for (const auto& state : evaluation.states) {
            path.emplace_back(state(ix::X), state(ix::Y));
        }
    }
    return result;
}

} // namespace nav_executor
