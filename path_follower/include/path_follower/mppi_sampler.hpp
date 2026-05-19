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

    /// 调试用 rollout 轨迹（map 坐标系下的位置路径），仅 debug 模式填充
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

}
