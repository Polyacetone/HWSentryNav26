#pragma once

#include <nav_executor/path_executor/solver/mpc_types.hpp>

namespace nav_executor {

CostSample eval_cost_bilinear(const CostMapGridView& grid, const GridInfo& info, double x_map, double y_map);
DirSample eval_dir_bilinear(const DirectionMapGridView& grid, const GridInfo& info, double x_map, double y_map);
Eigen::Vector2d eval_dir_bilinear_value_only(const DirectionMapGridView& grid, const GridInfo& info, double x_map, double y_map);

} // namespace nav_executor
