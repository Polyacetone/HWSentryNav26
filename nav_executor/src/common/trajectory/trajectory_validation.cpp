#include <nav_executor/common/trajectory/trajectory_validation.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace nav_executor {

namespace {

using ControlPoints = MincoTrajectory::ControlPointBlock;
constexpr int DEGREE = MincoTrajectory::DEGREE;
constexpr int MAX_SUBDIVISION_DEPTH = 14;
constexpr double RELATIVE_DERIVATIVE_EPS = 1e-8;

std::pair<ControlPoints, ControlPoints> subdivide(const ControlPoints& points) {
    ControlPoints work = points;
    ControlPoints left;
    ControlPoints right;
    left.row(0) = work.row(0);
    right.row(DEGREE) = work.row(DEGREE);
    for (int level = 1; level <= DEGREE; ++level) {
        for (int i = 0; i <= DEGREE - level; ++i) {
            work.row(i) = 0.5 * (work.row(i) + work.row(i + 1));
        }
        left.row(level) = work.row(0);
        right.row(DEGREE - level) = work.row(DEGREE - level);
    }
    return {left, right};
}

Eigen::Matrix<double, DEGREE, 2> derivative_control_points(
    const ControlPoints& points
) {
    Eigen::Matrix<double, DEGREE, 2> derivative;
    for (int i = 0; i < DEGREE; ++i) {
        derivative.row(i) = static_cast<double>(DEGREE)
            * (points.row(i + 1) - points.row(i));
    }
    return derivative;
}

Eigen::Vector2d evaluate_derivative(
    Eigen::Matrix<double, DEGREE, 2> points,
    const double u
) {
    for (int level = 1; level < DEGREE; ++level) {
        for (int i = 0; i < DEGREE - level; ++i) {
            points.row(i) = (1.0 - u) * points.row(i) + u * points.row(i + 1);
        }
    }
    return points.row(0).transpose();
}

double derivative_convex_hull_distance_lower_bound(
    const Eigen::Matrix<double, DEGREE, 2>& derivative
) {
    const double x_min = derivative.col(0).minCoeff();
    const double x_max = derivative.col(0).maxCoeff();
    const double y_min = derivative.col(1).minCoeff();
    const double y_max = derivative.col(1).maxCoeff();
    const double dx = x_min > 0.0 ? x_min : (x_max < 0.0 ? -x_max : 0.0);
    const double dy = y_min > 0.0 ? y_min : (y_max < 0.0 ? -y_max : 0.0);
    return std::hypot(dx, dy);
}

bool contains_detected_cusp(
    const ControlPoints& points,
    const double interval_fraction,
    const double derivative_epsilon,
    const int depth
) {
    const auto derivative = derivative_control_points(points);
    if (derivative_convex_hull_distance_lower_bound(derivative) / interval_fraction
        > derivative_epsilon) {
        return false;
    }

    if (depth < MAX_SUBDIVISION_DEPTH) {
        const auto [left, right] = subdivide(points);
        const double half_fraction = 0.5 * interval_fraction;
        return contains_detected_cusp(
            left, half_fraction, derivative_epsilon, depth + 1
        ) || contains_detected_cusp(
            right, half_fraction, derivative_epsilon, depth + 1
        );
    }

    constexpr std::array<double, 5> SAMPLES {0.0, 0.25, 0.5, 0.75, 1.0};
    return std::any_of(SAMPLES.begin(), SAMPLES.end(), [&](const double u) {
        return evaluate_derivative(derivative, u).norm() / interval_fraction
            <= derivative_epsilon;
    });
}

double segment_scale(const ControlPoints& points) {
    const double x_span = points.col(0).maxCoeff() - points.col(0).minCoeff();
    const double y_span = points.col(1).maxCoeff() - points.col(1).minCoeff();
    return std::max(std::hypot(x_span, y_span), 1.0);
}

} // anonymous namespace

std::optional<std::string> validate_trajectory_numerics(
    const MincoTrajectory& trajectory
) {
    if (trajectory.segment_count() < 1) return "trajectory has no segments";
    if (!std::isfinite(trajectory.total_time()) || trajectory.total_time() <= 0.0
        || !std::isfinite(trajectory.total_arc_length())
        || trajectory.total_arc_length() <= 0.0) {
        return "trajectory has an invalid total time or arc length";
    }

    for (int segment = 0; segment < trajectory.segment_count(); ++segment) {
        const ControlPoints points = trajectory.segment_bezier_control_points(segment);
        if (!points.allFinite()) {
            return "trajectory contains non-finite control points";
        }
        const double derivative_epsilon = RELATIVE_DERIVATIVE_EPS
            * segment_scale(points);
        if (contains_detected_cusp(points, 1.0, derivative_epsilon, 0)) {
            return "trajectory contains a zero-derivative cusp";
        }
    }
    return std::nullopt;
}

} // namespace nav_executor
