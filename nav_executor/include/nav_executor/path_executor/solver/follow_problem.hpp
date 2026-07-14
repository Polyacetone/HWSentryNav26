#pragma once

#include <nav_executor/path_executor/solver/mpc_types.hpp>
#include <nav_executor/common/spline_path.hpp>
#include <nav_executor/path_executor/solver/lpv_model.hpp>
#include <nav_executor/path_executor/solver/bilinear_sampling.hpp>
#include <nav_executor/path_executor/solver/fddp_solver.hpp>

namespace nav_executor {

template<int Horizon>
class FollowProblemT {
public:
    FollowProblemT(
        const SplinePath& spline,
        const MPCParams& params,
        const std::vector<CostMapGridView>& per_step_cost_grids,
        const GridInfo& cost_info,
        const CostMapGridView& masked_global_grid,
        double prediction_dt,
        double schedule_rho,
        double remaining_energy,
        double rfr_pwr_limit,
        const CapabilityProfile& blended_profile,
        std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule,
        double current_path_u
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
    [[nodiscard]] const MPCParams& params() const;
    [[nodiscard]] const CapabilityProfile& capability_profile() const { return blended_profile_; }
    [[nodiscard]] const SplinePath& reference_path() const { return spline_; }
    [[nodiscard]] const LPVDiscreteModel& discrete_model() const { return model_; }
    [[nodiscard]] double charge_power_limit() const { return rfr_pwr_limit_; }
    [[nodiscard]] FollowProblemT<Horizon> with_reference_path(const SplinePath& spline) const;

private:
    const CostMapGridView& cost_grid_for_step(int k) const;

    // 终端 cost-to-go 势的值与其对 (x,y) 的梯度（已含权重）。
    struct TerminalEval {
        double value = 0.0;
        Eigen::Vector2d grad_xy = Eigen::Vector2d::Zero();
    };
    TerminalEval evaluate_terminal(const StateVec& x) const;

    Eigen::Vector2d goal_xy_;
    SplinePath spline_;
    const MPCParams& p_;
    const std::vector<CostMapGridView>& step_cost_grids_;
    GridInfo cost_info_;
    const CostMapGridView& masked_global_grid_;
    double prediction_dt_;
    LPVDiscreteModel model_ {};
    double remaining_energy_;
    double rfr_pwr_limit_;
    CapabilityProfile blended_profile_;
    std::shared_ptr<const StepConstraintSchedule> step_constraint_schedule_;
    double current_path_u_;
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
