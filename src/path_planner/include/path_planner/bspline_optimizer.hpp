#pragma once

#include <path_planner/costmap_2d.hpp>

namespace path_planner {

class BSplineOptimizer {
public:
    explicit BSplineOptimizer(
        const double num_samples_per_length = 0.4,
        const double obstable_weight = 1e-0,
        const double length_weight = 1e-3,
        const double smooth_weight = 1e-5
    );
    std::vector<Eigen::Vector2d> optimize(
        const Costmap2D& costmap,
        const std::vector<Eigen::Vector2d>& path
    ) const;

private:
    const double num_samples_per_length_;
    const double obstable_weight_, smooth_weight_, length_weight_;

    std::vector<Eigen::Vector2d> pad_control_points(const std::vector<Eigen::Vector2d>& path) const;
    double estimate_path_length(const std::vector<Eigen::Vector2d>& path) const;
};

} // namespace path_planner