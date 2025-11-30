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
class FactorGraphOptimizer {
public:
    using Ptr = std::shared_ptr<FactorGraphOptimizer>;
    using ConstPtr = std::shared_ptr<const FactorGraphOptimizer>;

    explicit FactorGraphOptimizer(
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
    ) const;

private:
    const double smoothness_weight_;
    const double length_weight_;
    const double obstacle_weight_;
    const double direction_weight_;
    const int max_iterations_;
};

} // namespace path_planner
