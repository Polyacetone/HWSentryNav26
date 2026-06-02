#pragma once

#include <Eigen/Core>
#include <cmath>
#include <vector>
#include <uniform_bspline/uniform_bspline.hpp>

namespace path_follower {

struct SplineEval {
    Eigen::Vector2d p;
    Eigen::Vector2d d1;
    Eigen::Vector2d d2;
    double thetar = 0.0;
    double sin_r = 0.0;
    double cos_r = 0.0;
    double ds_du = 0.0;
    double kappa = 0.0;
};

class SplinePath {
public:
    static constexpr double U_MIN = 0.0;
    static constexpr double U_MAX = 1.0;
    static constexpr double U_EXTRAP_MIN = -0.5;
    static constexpr double U_EXTRAP_MAX = 1.5;

    explicit SplinePath(const std::vector<Eigen::Vector2d>& cps): spline_(cps) {
        spline_.setExtrapolate(true);
    }
    explicit SplinePath(std::vector<Eigen::Vector2d>&& cps): spline_(std::move(cps)) {
        spline_.setExtrapolate(true);
    }

    [[nodiscard]] SplineEval eval(double u) const;
    [[nodiscard]] Eigen::Vector2d position(double u) const { return spline_.evaluate(u); }
    [[nodiscard]] Eigen::Vector2d tangent(double u) const { return spline_.derivative(u, 1); }
    [[nodiscard]] double arc_length(double u0, double u1, int samples = 8) const;

    [[nodiscard]] double project(
        const Eigen::Vector2d& pos, double u_hint,
        int num_samples, double search_window,
        double local_search_lazy_distance
    ) const;

    [[nodiscard]] double project_extrapolated(
        const Eigen::Vector2d& pos, double u_hint,
        int num_samples, double search_window,
        double local_search_lazy_distance,
        double u_min = U_EXTRAP_MIN,
        double u_max = U_EXTRAP_MAX
    ) const;

    static double clamp_u(double u) { return std::clamp(u, U_MIN, U_MAX); }
    static double clamp_u_extrapolated(double u) { return std::clamp(u, U_EXTRAP_MIN, U_EXTRAP_MAX); }

    [[nodiscard]] bool operator==(const SplinePath& other) const;
    [[nodiscard]] bool operator!=(const SplinePath& other) const { return !(*this == other); }

private:
    using SplineD = ubs::UniformBSpline<double, 2, double, Eigen::Vector2d, std::vector<Eigen::Vector2d>>;
    SplineD spline_;
};

}
