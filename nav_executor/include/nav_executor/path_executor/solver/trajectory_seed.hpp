#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include <nav_executor/path_executor/solver/follow_problem.hpp>

namespace nav_executor {

enum class SeedSource : uint8_t {
    WARM_START,
    LONGITUDINAL,
    GLOBAL,
};

struct TrajectorySeed {
    std::array<ControlVec, MPC_HORIZON> controls {};
    SeedSource source = SeedSource::WARM_START;
    uint64_t origin_seq = 0;
};

struct SeedEvaluation {
    std::array<StateVec, MPC_HORIZON + 1> states {};
    double cost = std::numeric_limits<double>::infinity();
    bool valid = false;
};

[[nodiscard]] SeedEvaluation evaluate_seed(
    const FollowProblem& problem,
    const TrajectorySeed& seed,
    const StateVec& x0,
    bool reject_lethal = true
);

} // namespace nav_executor
