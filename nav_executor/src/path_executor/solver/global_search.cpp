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
    const int mode_count = std::max(candidate_count, 0);
    if (mode_count == 0) return result;

    struct EvaluatedCandidate {
        TrajectorySeed seed;
        SeedEvaluation evaluation;
    };
    std::vector<EvaluatedCandidate> evaluated;
    auto coarse_candidates = beam_search_.search(problem, x0);
    evaluated.reserve(coarse_candidates.size() + 1);
    for (auto& coarse : coarse_candidates) {
        coarse.origin_seq = warm_seed.origin_seq;
        auto evaluation = evaluate_seed(problem, coarse, x0);
        if (!evaluation.valid) continue;
        evaluated.push_back(EvaluatedCandidate {
            .seed = std::move(coarse),
            .evaluation = std::move(evaluation),
        });
    }
    auto longitudinal_evaluation = evaluate_seed(problem, longitudinal_seed, x0);
    if (longitudinal_evaluation.valid) {
        evaluated.push_back(EvaluatedCandidate {
            .seed = longitudinal_seed,
            .evaluation = std::move(longitudinal_evaluation),
        });
    }
    std::sort(evaluated.begin(), evaluated.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.evaluation.cost < rhs.evaluation.cost;
    });
    if (static_cast<int>(evaluated.size()) > mode_count) {
        evaluated.resize(static_cast<size_t>(mode_count));
    }

    result.candidates.reserve(evaluated.size());
    result.debug_paths.reserve(evaluated.size());
    for (auto& candidate : evaluated) {
        auto& path = result.debug_paths.emplace_back();
        path.reserve(MPC_HORIZON + 1);
        for (const auto& state : candidate.evaluation.states) {
            path.emplace_back(state(ix::X), state(ix::Y));
        }
        result.candidates.push_back(std::move(candidate.seed));
    }
    return result;
}

} // namespace nav_executor
