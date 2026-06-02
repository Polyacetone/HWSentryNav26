#pragma once

#include <path_follower/solver/mpc_types.hpp>
#include <path_follower/path/spline_path.hpp>
#include <path_follower/solver/lpv_model.hpp>
#include <path_follower/solver/bilinear_sampling.hpp>
#include <path_follower/solver/fddp_solver.hpp>

namespace path_follower {

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
        const DirectionMapGridView& dir_grid,
        const GridInfo& dir_info,
        double remaining_energy,
        double rfr_pwr_limit,
        const CapabilityProfile& blended_profile,
        std::optional<ActiveStepMode> active_step_mode,
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
    [[nodiscard]] FollowProblemT<Horizon> with_reference_path(const SplinePath& spline) const;

private:
    const CostMapGridView& cost_grid_for_step(int k) const;

    Eigen::Vector2d goal_xy_;
    SplinePath spline_;
    const MPCParams& p_;
    const std::vector<CostMapGridView>& step_cost_grids_;
    GridInfo cost_info_;
    const CostMapGridView& masked_global_grid_;
    double prediction_dt_;
    LPVDiscreteModel model_ {};
    const DirectionMapGridView& dir_grid_;
    GridInfo dir_info_;
    double remaining_energy_;
    double rfr_pwr_limit_;
    CapabilityProfile blended_profile_;
    std::optional<ActiveStepMode> active_step_mode_;
    double current_path_u_;
};

using FollowProblem = FollowProblemT<MPC_HORIZON>;

} // namespace path_follower

namespace fddp {
template<int Horizon>
struct Dims<path_follower::FollowProblemT<Horizon>> {
    static constexpr int NX = path_follower::MPC_NX;
    static constexpr int NU = path_follower::MPC_NU;
    static constexpr int N = Horizon;
};
} // namespace fddp
