#pragma once

#include <array>

#include <path_follower/solver/mpc_types.hpp>
#include <path_follower/solver/follow_problem.hpp>

namespace path_follower {

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

} // namespace path_follower
