#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>

#include <gtsam/geometry/Point2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <path_planner/nav_map.hpp>

namespace path_planner {
class PathOptimizer {
public:
    using Ptr = std::shared_ptr<PathOptimizer>;

    explicit PathOptimizer(
        const double smoothness_weight,
        const double length_weight,
        const double obstacle_weight,
        const double direction_weight,
        const int max_iterations
    );
    std::vector<Eigen::Vector2d> optimize(
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const std::vector<Eigen::Vector2d>& init_path
    );

private:
    const double smoothness_weight;
    const double length_weight;
    const double obstacle_weight;
    const double direction_weight;
    const int max_iterations;
};

} // namespace path_planner
