#pragma once

#include <memory>
#include <optional>

#include <nav_executor/common/minco_trajectory.hpp>
#include <nav_executor/common/path_speed_profile.hpp>
#include <nav_executor/path_executor/solver/mpc_types.hpp>
#include <nav_executor/path_executor/solver/lpv_model.hpp>
#include <nav_executor/path_executor/solver/bilinear_sampling.hpp>
#include <nav_executor/path_executor/solver/fddp_solver.hpp>

namespace nav_executor {

// 跟踪问题把物理状态与路径累计弧长一并优化。第三控制量是沿路径正方向的虚拟速度。
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
        const SignedVelocityBounds& path_speed_bounds,
        std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule
    );

    StateVec dynamics(int k, const StateVec& x, const ControlVec& u) const;
    void dynamics_jacobians(int k, const StateVec& x, const ControlVec& u, MatXX& fx, MatXU& fu) const;

    double running_cost(int k, const StateVec& x, const ControlVec& u) const;
    double running_cost_value_only(int k, const StateVec& x, const ControlVec& u, double* cached_cost_value = nullptr) const;
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
    [[nodiscard]] const MPCParams& params() const { return p_; }
    [[nodiscard]] const MincoTrajectory& reference_trajectory() const { return trajectory_; }
    [[nodiscard]] const LPVDiscreteModel& discrete_model() const { return model_; }

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
    SignedVelocityBounds path_speed_bounds_;
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
