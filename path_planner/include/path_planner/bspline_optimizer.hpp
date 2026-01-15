#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <path_planner/nav_map.hpp>

namespace path_planner {
class BSplineOptimizer {
public:
    using Ptr = std::shared_ptr<BSplineOptimizer>;
    using ConstPtr = std::shared_ptr<const BSplineOptimizer>;

    explicit BSplineOptimizer(
        const double smoothness_weight,
        const double uniform_speed_weight,
        const double obstacle_weight,
        const double direction_weight,
        const double start_end_weight,
        const double num_samples_per_length,
        const int max_iterations
    );

    // 返回优化后的控制点和采样点
    std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>> optimize(
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const std::vector<Eigen::Vector2d>& init_path,
        const Eigen::Vector2d& start_grid,
        const Eigen::Vector2d& goal_grid
    ) const;

private:
    const double smoothness_weight_;
    const double uniform_speed_weight_;
    const double obstacle_weight_;
    const double direction_weight_;
    const double start_end_weight_;
    const double num_samples_per_length_;
    const int max_iterations_;

    std::vector<Eigen::Vector2d> pad_control_points(const std::vector<Eigen::Vector2d>& path) const;
    double estimate_path_length(const std::vector<Eigen::Vector2d>& path) const;
};
} // namespace path_planner