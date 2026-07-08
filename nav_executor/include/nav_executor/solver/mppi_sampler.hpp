#pragma once

#include <array>

#include <nav_executor/solver/mpc_types.hpp>
#include <nav_executor/solver/follow_problem.hpp>

namespace nav_executor {

struct MPPIFollowSamplingResult {
    std::array<StateVec, MPC_HORIZON + 1> xs {};
    std::array<ControlVec, MPC_HORIZON> us {};
    double cost = 0.0;
    bool valid = false;

    std::vector<std::vector<Eigen::Vector2d>> rollout_paths;
};

class MPPIFollowSampler {
public:
    explicit MPPIFollowSampler(MPCFollowMPPIParams params): params_(params) {}

    [[nodiscard]] MPPIFollowSamplingResult optimize(
        const FollowProblem& problem,
        const StateVec& x0,
        const std::array<ControlVec, MPC_HORIZON>& nominal_controls
    ) const;

private:
    MPCFollowMPPIParams params_;
};

} // namespace nav_executor
