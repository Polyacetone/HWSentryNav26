#pragma once

#include <array>

#include <path_follower/mpc_solver.hpp>

namespace path_follower {

struct MPPIFollowSamplingResult {
    std::array<StateVec, MPC_HORIZON + 1> xs {};
    std::array<ControlVec, MPC_HORIZON> us {};
    double cost = 0.0;
    bool valid = false;
    std::optional<RolloutLethalObstacleInfo> lethal_obstacle;

    /// 最佳 rollout 路径（map 坐标），仅 debug 模式使用
    std::vector<Eigen::Vector2d> rollout_path;
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

}
