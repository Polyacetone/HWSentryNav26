#pragma once

#include <uniform_bspline/uniform_bspline.hpp>

namespace path_follower {
template <typename ValueType>
using SplineT = ubs::UniformBSpline<ValueType, 2, ValueType, Eigen::Matrix<ValueType, 2, 1>, std::vector<Eigen::Matrix<ValueType, 2, 1>>>;
using SplineD = ubs::UniformBSpline<double, 2, double, Eigen::Vector2d, std::vector<Eigen::Vector2d>>;
}

namespace path_follower{
inline constexpr double PATH_U_EXTRAP_MIN = -0.35;
inline constexpr double PATH_U_EXTRAP_MAX = 1.35;

double clamp_path_u_extrapolated(
    double u,
    double u_min = PATH_U_EXTRAP_MIN,
    double u_max = PATH_U_EXTRAP_MAX
);

double project_to_spline_u(
    const SplineD& spline,
    const Eigen::Vector2d& pos,
    double u_hint,
    int num_samples,
    double search_window,
    double local_search_lazy_distance
);

double project_to_spline_u_extrapolated(
    const SplineD& spline,
    const Eigen::Vector2d& pos,
    double u_hint,
    int num_samples,
    double search_window,
    double local_search_lazy_distance,
    double u_min = PATH_U_EXTRAP_MIN,
    double u_max = PATH_U_EXTRAP_MAX
);
}