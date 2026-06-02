#pragma once

#include <path_follower/solver/mpc_types.hpp>

namespace path_follower {

CostSample eval_cost_bilinear(const CostMapGridView& grid, const GridInfo& info, double x_map, double y_map);

DirSample eval_dir_bilinear(const DirectionMapGridView& grid, const GridInfo& info, double x_map, double y_map);

Eigen::Vector2d eval_dir_bilinear_value_only(const DirectionMapGridView& grid, const GridInfo& info, double x_map, double y_map);

} // namespace path_follower
