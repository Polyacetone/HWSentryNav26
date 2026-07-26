#pragma once

#include <nav_executor/path_executor/mpc/mpc_types.hpp>

namespace nav_executor {

CostSample eval_cost_bilinear(const CostMapGridView& grid, const GridInfo& info, double x_map, double y_map);

} // namespace nav_executor
