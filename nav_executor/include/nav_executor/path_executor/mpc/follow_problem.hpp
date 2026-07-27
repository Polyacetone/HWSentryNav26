#pragma once

#include <memory>
#include <optional>

#include <nav_executor/common/trajectory/minco_trajectory.hpp>
#include <nav_executor/common/trajectory/path_speed_profile.hpp>
#include <nav_executor/path_executor/mpc/mpc_types.hpp>
#include <nav_executor/path_executor/mpc/lpv_model.hpp>
#include <nav_executor/path_executor/mpc/bilinear_sampling.hpp>
#include <nav_executor/path_executor/mpc/fddp_solver.hpp>

namespace nav_executor {

// 路径跟踪问题。第三控制量是沿有向路径的虚拟进度速度 ṡ，硬约束在 [0, 剩余弧长/Δt]
// 之内，因此 0 ≤ s ≤ L 且 ṡ ≥ 0 恒成立。外部有向进度估计只作为初值输入，
// 预测状态 PATH_PROGRESS 不回写覆盖它。
template<int Horizon>
class FollowProblemT {
public:
    FollowProblemT(
        MincoTrajectory trajectory,
        PathSpeedProfile speed_profile,
        const MPCParams& params,
        const std::vector<CostMapGridView>& per_step_cost_grids,
        const GridInfo& cost_info,
        const CostMapGridView& masked_global_grid,
        double prediction_dt,
        double schedule_rho,
        const CapabilityProfile& command_capability,
        std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule
    );

    StateVec dynamics(int k, const StateVec& x, const ControlVec& u) const;
    void dynamics_jacobians(int k, const StateVec& x, const ControlVec& u, MatXX& fx, MatXU& fu) const;

    double running_cost(int k, const StateVec& x, const ControlVec& u) const;
    void running_cost_derivatives(
        int k,
        const StateVec& x,
        const ControlVec& u,
        StateVec& lx,
        ControlVec& lu,
        MatXX& lxx,
        Eigen::Matrix<double, MPC_NU, MPC_NX>& lux,
        Eigen::Matrix<double, MPC_NU, MPC_NU>& luu
    ) const;

    double terminal_cost(const StateVec& x) const;
    void terminal_cost_derivatives(const StateVec& x, StateVec& lfx, MatXX& lfxx) const;

    MPCControlBounds control_bounds(int k, const StateVec& x) const;

    [[nodiscard]] std::optional<RolloutLethalObstacleInfo> detect_lethal_obstacle(int state_index, const StateVec& x, double* out_cost_value = nullptr) const;

private:
    const CostMapGridView& cost_grid_for_step(int k) const;

    MincoTrajectory trajectory_;
    PathSpeedProfile speed_profile_;
    const MPCParams& p_;
    const std::vector<CostMapGridView>& step_cost_grids_;
    GridInfo cost_info_;
    const CostMapGridView& masked_global_grid_;
    double prediction_dt_;
    LPVDiscreteModel model_ {};
    CapabilityProfile command_capability_;
    std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule_;
    double total_length_ = 0.0;
};

using FollowProblem = FollowProblemT<MPC_HORIZON>;

} // namespace nav_executor

namespace fddp {
template<int Horizon>
struct Dims<nav_executor::FollowProblemT<Horizon>> {
    static constexpr int NX = nav_executor::MPC_NX;
    static constexpr int NU = nav_executor::MPC_NU;
    static constexpr int N = Horizon;
};
} // namespace fddp
