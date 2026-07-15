#pragma once

#include <Eigen/Core>
#include <cmath>
#include <optional>
#include <vector>
#include <uniform_bspline/uniform_bspline.hpp>

namespace nav_executor {

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

struct PathProjection {
    double u = 0.0;
    double arc_length = 0.0;
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    double distance = 0.0;
};

class SplinePath {
public:
    static constexpr double U_MIN = 0.0;
    static constexpr double U_MAX = 1.0;
    static constexpr double U_EXTRAP_MIN = -0.5;
    static constexpr double U_EXTRAP_MAX = 1.5;

    explicit SplinePath(const std::vector<Eigen::Vector2d>& cps);
    explicit SplinePath(std::vector<Eigen::Vector2d>&& cps);

    [[nodiscard]] SplineEval eval(double u) const;
    [[nodiscard]] Eigen::Vector2d position(double u) const { return eval(u).p; }
    [[nodiscard]] Eigen::Vector2d tangent(double u) const { return eval(u).d1; }
    [[nodiscard]] const std::vector<Eigen::Vector2d>& control_points() const { return spline_.getControlPoints(); }
    [[nodiscard]] double arc_length(double u0, double u1, int samples = 8) const;
    [[nodiscard]] double arc_length_at_u(double u) const;
    [[nodiscard]] double u_at_arc_length(double arc_length) const;
    [[nodiscard]] double length() const { return total_length_; }

    [[nodiscard]] std::optional<PathProjection> project(
        const Eigen::Vector2d& pos, double min_arc_length, double max_arc_length
    ) const;

    static double clamp_u(double u) { return std::clamp(u, U_MIN, U_MAX); }
    static double clamp_u_extrapolated(double u) { return std::clamp(u, U_EXTRAP_MIN, U_EXTRAP_MAX); }

    [[nodiscard]] bool operator==(const SplinePath& other) const;
    [[nodiscard]] bool operator!=(const SplinePath& other) const { return !(*this == other); }

private:
    using SplineD = ubs::UniformBSpline<double, 2, double, Eigen::Vector2d, std::vector<Eigen::Vector2d>>;
    void build_arc_length_table();

    SplineD spline_;
    std::vector<double> sample_arc_lengths_;
    double total_length_ = 0.0;
};

}
