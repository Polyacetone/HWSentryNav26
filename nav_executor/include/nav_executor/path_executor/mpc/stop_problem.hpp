#pragma once

#include <nav_executor/path_executor/mpc/mpc_types.hpp>
#include <nav_executor/path_executor/mpc/lpv_model.hpp>
#include <nav_executor/path_executor/mpc/fddp_solver.hpp>

namespace nav_executor {

class StopProblem {
public:
    StopProblem(
        const MPCParams& params,
        const CostMap& cost_map,
        double schedule_rho
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

private:
    const MPCParams& p_;
    const CostMap& cost_map_;
    LPVDiscreteModel model_ {};
};

} // namespace nav_executor

namespace fddp {
template<>
struct Dims<nav_executor::StopProblem> {
    static constexpr int NX = nav_executor::MPC_NX;
    static constexpr int NU = nav_executor::MPC_NU;
    static constexpr int N = nav_executor::MPC_HORIZON;
};
} // namespace fddp
