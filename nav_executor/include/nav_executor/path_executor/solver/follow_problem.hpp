#pragma once

#include <memory>
#include <optional>

#include <nav_executor/common/minco_trajectory.hpp>
#include <nav_executor/path_executor/solver/mpc_types.hpp>
#include <nav_executor/path_executor/solver/lpv_model.hpp>
#include <nav_executor/path_executor/solver/bilinear_sampling.hpp>
#include <nav_executor/path_executor/solver/fddp_solver.hpp>

namespace nav_executor {

// 参数化轨迹跟踪问题（替换旧 SplinePath 版 FollowProblemT）。
//
// 参考载体是 MincoTrajectory。PHASE_TIME 是秒制虚拟相位，PHASE_RATE 是相位相对真实时间
// 的推进率。物理状态始终按 MPC_DT 递推，参考位置按虚拟相位求值。
//
// 代价导数走有限差分 Gauss-Newton（与 stop/hold 一致，落地版 Q3 决策 B）：残差函数是
// 唯一真值源，不再手推雅可比。τ-进度作为状态转移的一部分放进 dynamics，其雅可比行单独
// 有限差分（物理动力学行仍用解析 mpc_dynamics_jacobians）。
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
        const CapabilityProfile& blended_profile,
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
    [[nodiscard]] const CapabilityProfile& capability_profile() const { return blended_profile_; }
    [[nodiscard]] const MincoTrajectory& reference_trajectory() const { return trajectory_; }
    [[nodiscard]] const LPVDiscreteModel& discrete_model() const { return model_; }

    [[nodiscard]] TrajectoryPhaseState advance_phase(const StateVec& x) const;

private:
    const CostMapGridView& cost_grid_for_step(int k) const;

    MincoTrajectory trajectory_;
    const MPCParams& p_;
    const std::vector<CostMapGridView>& step_cost_grids_;
    GridInfo cost_info_;
    const CostMapGridView& masked_global_grid_;
    double prediction_dt_;
    LPVDiscreteModel model_ {};
    CapabilityProfile blended_profile_;
    std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule_;
    double total_arc_ = 0.0;  // 参考总弧长，用于进度奖励的量纲缩放
    double total_time_ = 0.0; // 参考总时长，用于名义时钟速率 1/T
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
