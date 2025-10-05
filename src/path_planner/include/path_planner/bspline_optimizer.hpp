#pragma once

#include <path_planner/costmap_2d.hpp>

namespace path_planner {

class BSplineOptimizer {
public:
    explicit BSplineOptimizer(
        const double num_samples_per_length,
        const double obstable_weight,
        const double length_weight,
        const double smooth_weight
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