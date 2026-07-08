#pragma once

#include <nav_executor/solver/mpc_types.hpp>

namespace nav_executor {

StateVec mpc_dynamics(const StateVec& x, const ControlVec& u, const LPVDiscreteModel& model);

void mpc_dynamics_jacobians(const StateVec& x, const ControlVec& u, const LPVDiscreteModel& model, MatXX& fx, MatXU& fu);

} // namespace nav_executor
