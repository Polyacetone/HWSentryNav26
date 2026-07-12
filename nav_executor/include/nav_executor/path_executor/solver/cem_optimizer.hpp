#pragma once

#include <array>

#include <nav_executor/path_executor/solver/mpc_types.hpp>
#include <nav_executor/path_executor/solver/follow_problem.hpp>

namespace nav_executor {

struct CEMOptimizationResult {
    std::array<StateVec, MPC_HORIZON + 1> xs {};
    std::array<ControlVec, MPC_HORIZON> us {};
    double cost = 0.0;
    bool valid = false;

};

class CEMOptimizer {
public:
    explicit CEMOptimizer(GlobalSearchParams params): params_(params) {}

    [[nodiscard]] CEMOptimizationResult optimize(
        const FollowProblem& problem,
        const StateVec& x0,
        const std::array<ControlVec, MPC_HORIZON>& nominal_controls
    ) const;

private:
    GlobalSearchParams params_;
};

} // namespace nav_executor
