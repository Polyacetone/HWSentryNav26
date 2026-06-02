#pragma once

#include <path_follower/solver/mpc_types.hpp>
#include <path_follower/solver/lpv_model.hpp>
#include <path_follower/solver/bilinear_sampling.hpp>
#include <path_follower/solver/fddp_solver.hpp>

namespace path_follower {

class StopProblem {
public:
    StopProblem(
        const MPCParams& params,
        const CostMapGridView& cost_grid,
        const GridInfo& cost_info,
        double schedule_rho,
        double remaining_energy,
        double rfr_pwr_limit
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

    ControlVec u_lower() const;
    ControlVec u_upper() const;

private:
    const MPCParams& p_;
    const CostMapGridView& cost_grid_;
    GridInfo cost_info_;
    LPVDiscreteModel model_ {};
    double remaining_energy_;
    double rfr_pwr_limit_;
};

} // namespace path_follower

namespace fddp {
template<>
struct Dims<path_follower::StopProblem> {
    static constexpr int NX = path_follower::MPC_NX;
    static constexpr int NU = path_follower::MPC_NU;
    static constexpr int N = path_follower::MPC_HORIZON;
};
} // namespace fddp
