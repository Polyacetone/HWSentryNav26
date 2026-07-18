#pragma once

#include <memory>
#include <optional>

#include <nav_executor/common/minco_trajectory.hpp>
#include <nav_executor/path_executor/solver/mpc_types.hpp>
#include <nav_executor/path_executor/solver/lpv_model.hpp>
#include <nav_executor/path_executor/solver/bilinear_sampling.hpp>
#include <nav_executor/path_executor/solver/fddp_solver.hpp>

namespace nav_executor {

// MINCO 全状态轨迹上的 MPCC。PHASE_TIME 是秒制轨迹相位，第三控制量直接决定相位推进率；
// PHASE_RATE 保存上一拍相位率用于平滑。物理状态与虚拟相位在同一个 FDDP 问题内联合优化。
template<int Horizon>
class FollowProblemT {
public:
    FollowProblemT(
        MincoTrajectory trajectory,
        const MPCParams& params,
        const std::vector<CostMapGridView>& per_step_cost_grids,
        const GridInfo& cost_info,
        const CostMapGridView& masked_global_grid,
        double prediction_dt,
        double schedule_rho,
        const CapabilityProfile& effective_capability,
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

    ControlVec u_lower() const;
    ControlVec u_upper() const;

    [[nodiscard]] std::optional<RolloutLethalObstacleInfo> detect_lethal_obstacle(int state_index, const StateVec& x, double* out_cost_value = nullptr) const;
    [[nodiscard]] const MPCParams& params() const { return p_; }
    [[nodiscard]] const CapabilityProfile& effective_capability() const {
        return effective_capability_;
    }
    [[nodiscard]] const MincoTrajectory& reference_trajectory() const { return trajectory_; }
    [[nodiscard]] const LPVDiscreteModel& discrete_model() const { return model_; }

private:
    const CostMapGridView& cost_grid_for_step(int k) const;

    MincoTrajectory trajectory_;
    const MPCParams& p_;
    const std::vector<CostMapGridView>& step_cost_grids_;
    GridInfo cost_info_;
    const CostMapGridView& masked_global_grid_;
    double prediction_dt_;
    LPVDiscreteModel model_ {};
    CapabilityProfile effective_capability_;
    std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule_;
    double total_time_ = 0.0;
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
