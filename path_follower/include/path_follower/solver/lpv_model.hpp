#pragma once

#include <path_follower/solver/mpc_types.hpp>

namespace path_follower {

StateVec mpc_dynamics(const StateVec& x, const ControlVec& u, const LPVDiscreteModel& model);

void mpc_dynamics_jacobians(const StateVec& x, const ControlVec& u, const LPVDiscreteModel& model, MatXX& fx, MatXU& fu);

} // namespace path_follower
