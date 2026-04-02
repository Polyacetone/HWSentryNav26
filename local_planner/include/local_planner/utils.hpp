#pragma once

#include <uniform_bspline/uniform_bspline.hpp>

namespace local_planner {
template <typename ValueType>
using SplineT = ubs::UniformBSpline<ValueType, 2, ValueType, Eigen::Matrix<ValueType, 2, 1>, std::vector<Eigen::Matrix<ValueType, 2, 1>>>;
using SplineD = ubs::UniformBSpline<double, 2, double, Eigen::Vector2d, std::vector<Eigen::Vector2d>>;
}

namespace local_planner{
double project_to_spline_u(
    const SplineD& spline,
    const Eigen::Vector2d& pos,
    double u_hint,
    int num_samples,
    double search_window,
    double local_search_lazy_distance
);
}