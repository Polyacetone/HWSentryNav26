#include <nav_executor/common/trajectory/geometry_certificate.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nav_executor {

namespace {

using ControlPoints = MincoTrajectory::ControlPointBlock;
constexpr int DEGREE = MincoTrajectory::DEGREE;
constexpr double INF = std::numeric_limits<double>::infinity();

double cross_2d(const Eigen::Vector2d& lhs, const Eigen::Vector2d& rhs) {
    return lhs.x() * rhs.y() - lhs.y() * rhs.x();
}

double binomial(const int n, const int k) {
    double value = 1.0;
    for (int i = 0; i < k; ++i) {
        value = value * static_cast<double>(n - i) / static_cast<double>(i + 1);
    }
    return value;
}

// Bezier 导数（hodograph）控制点：阶数每次降一。
template<int Order>
Eigen::Matrix<double, Order, 2> hodograph(
    const Eigen::Matrix<double, Order + 1, 2>& points
) {
    Eigen::Matrix<double, Order, 2> derivative;
    for (int i = 0; i < Order; ++i) {
        derivative.row(i) = static_cast<double>(Order) * (points.row(i + 1) - points.row(i));
    }
    return derivative;
}

// 标量区间界：[min, max]。
struct ScalarRange {
    double min = 0.0;
    double max = 0.0;

    [[nodiscard]] double magnitude_max() const {
        return std::max(std::abs(min), std::abs(max));
    }
};

// 两条 Bernstein 向量曲线在双线性标量形式（点积或叉积）下的乘积仍是 Bernstein 标量
// 多项式，其控制点可精确计算。由此得到的凸包界随区间细分二次收敛，比
// |det(a,b)| ≤ |a||b| 之类的分解界紧得多。
template<typename LhsT, typename RhsT, typename FormT>
ScalarRange bilinear_range(const LhsT& lhs, const RhsT& rhs, FormT&& form) {
    const int m = static_cast<int>(lhs.rows()) - 1;
    const int n = static_cast<int>(rhs.rows()) - 1;
    ScalarRange range {INF, -INF};
    for (int k = 0; k <= m + n; ++k) {
        double control = 0.0;
        for (int i = std::max(0, k - n); i <= std::min(m, k); ++i) {
            control += binomial(m, i) * binomial(n, k - i)
                * form(
                    Eigen::Vector2d(lhs.row(i).transpose()),
                    Eigen::Vector2d(rhs.row(k - i).transpose())
                );
        }
        control /= binomial(m + n, k);
        range.min = std::min(range.min, control);
        range.max = std::max(range.max, control);
    }
    return range;
}

// 参数子区间上的保守几何界。曲率与曲率变化率与参数化无关，因此只有有向导数
// 需要按物理时长换算。
struct IntervalBounds {
    double directed_speed_min = 0.0;
    double curvature_max = INF;
    double curvature_rate_max = INF;
};

IntervalBounds interval_bounds(
    const ControlPoints& points,
    const Eigen::Vector2d& seed_tangent,
    const double parameter_span // 该子区间对应的物理时长 (s)
) {
    const Eigen::Matrix<double, DEGREE, 2> first = hodograph<DEGREE>(points);
    const Eigen::Matrix<double, DEGREE - 1, 2> second = hodograph<DEGREE - 1>(first);
    const Eigen::Matrix<double, DEGREE - 2, 2> third = hodograph<DEGREE - 2>(second);

    const auto dot = [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        return a.dot(b);
    };

    IntervalBounds bounds;
    bounds.directed_speed_min = INF;
    for (int i = 0; i < first.rows(); ++i) {
        bounds.directed_speed_min = std::min(
            bounds.directed_speed_min,
            Eigen::Vector2d(first.row(i).transpose()).dot(seed_tangent)
        );
    }
    bounds.directed_speed_min /= parameter_span;
    if (bounds.directed_speed_min <= 0.0) return bounds;

    // |p'|² 的下界。有向导数为正保证曲线正则，因此该下界必然为正。
    const double speed_squared_min = bilinear_range(first, first, dot).min;
    if (speed_squared_min <= 0.0) return bounds;

    const double turn_max = bilinear_range(first, second, cross_2d).magnitude_max();
    const double twist_max = bilinear_range(first, third, cross_2d).magnitude_max();
    const double tangential_max = bilinear_range(first, second, dot).magnitude_max();

    // |κ| = |det(p',p'')| / (|p'|²)^{3/2}
    // |dκ/ds| ≤ |det(p',p''')|/(|p'|²)² + 3·|det(p',p'')|·|p'·p''|/(|p'|²)³
    const double speed_squared = speed_squared_min;
    bounds.curvature_max = turn_max / (speed_squared * std::sqrt(speed_squared));
    bounds.curvature_rate_max = twist_max / (speed_squared * speed_squared)
        + 3.0 * turn_max * tangential_max
            / (speed_squared * speed_squared * speed_squared);
    return bounds;
}

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

// 自适应细分：区间界落在包络内即接受；否则二分。达到最大深度仍无法证明则判定违规。
std::string certify_interval(
    const ControlPoints& points,
    const Eigen::Vector2d& seed_tangent,
    const TrajectoryLimits& limits,
    const int max_depth,
    const int segment,
    const double parameter_span,
    const double u_begin,
    const double u_end,
    const int depth
) {
    const IntervalBounds bounds = interval_bounds(points, seed_tangent, parameter_span);
    const double curvature_limit = limits.curvature_max();
    const double curvature_rate_limit = limits.curvature_rate_max();
    if (bounds.directed_speed_min >= limits.directed_speed_min
        && bounds.curvature_max <= curvature_limit
        && bounds.curvature_rate_max <= curvature_rate_limit) {
        return {};
    }

    if (depth < max_depth) {
        const auto [left, right] = subdivide(points);
        const double midpoint = 0.5 * (u_begin + u_end);
        const double half_span = 0.5 * parameter_span;
        std::string failure = certify_interval(
            left, seed_tangent, limits, max_depth, segment,
            half_span, u_begin, midpoint, depth + 1
        );
        if (!failure.empty()) return failure;
        return certify_interval(
            right, seed_tangent, limits, max_depth, segment,
            half_span, midpoint, u_end, depth + 1
        );
    }

    const std::string where = "segment " + std::to_string(segment)
        + " u=[" + std::to_string(u_begin) + "," + std::to_string(u_end) + "]";
    if (bounds.directed_speed_min < limits.directed_speed_min) {
        return "directed derivative lower bound " + std::to_string(bounds.directed_speed_min)
            + " m/s below the required " + std::to_string(limits.directed_speed_min)
            + " m/s on " + where;
    }
    if (bounds.curvature_max > curvature_limit) {
        return "curvature bound " + std::to_string(bounds.curvature_max)
            + " exceeds the " + std::to_string(curvature_limit) + " 1/m limit on " + where;
    }
    return "curvature rate bound " + std::to_string(bounds.curvature_rate_max)
        + " exceeds the " + std::to_string(curvature_rate_limit) + " 1/m² limit on " + where;
}

// 有向 SE(2) 可分辨性：弧长远离但位置与航向都接近的两点无法由车身位姿区分。
std::string find_unobservable_branch_pair(
    const MincoTrajectory& trajectory,
    const GeometryCertificateParams& params
) {
    const double total_length = trajectory.total_arc_length();
    const int count = std::max(
        2,
        static_cast<int>(std::ceil(total_length / params.observability_sample_spacing)) + 1
    );
    std::vector<double> arc_lengths(static_cast<size_t>(count));
    std::vector<Eigen::Vector2d> positions(static_cast<size_t>(count));
    std::vector<double> headings(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const double arc_length = total_length * static_cast<double>(i)
            / static_cast<double>(count - 1);
        const TrajSample sample = trajectory.eval_arc_length(arc_length);
        arc_lengths[static_cast<size_t>(i)] = arc_length;
        positions[static_cast<size_t>(i)] = sample.p;
        headings[static_cast<size_t>(i)] = sample.theta;
    }

    const int separation_stride = std::max(
        1,
        static_cast<int>(std::ceil(
            params.observability_arc_separation / params.observability_sample_spacing
        ))
    );
    for (int i = 0; i + separation_stride < count; ++i) {
        for (int j = i + separation_stride; j < count; ++j) {
            const double distance = (
                positions[static_cast<size_t>(j)] - positions[static_cast<size_t>(i)]
            ).norm();
            if (distance > params.observability_position_distance) continue;
            const double difference = headings[static_cast<size_t>(j)]
                - headings[static_cast<size_t>(i)];
            const double heading_difference = std::abs(
                std::atan2(std::sin(difference), std::cos(difference))
            );
            if (heading_difference > params.observability_heading_angle) continue;
            return "directionally indistinguishable branches at s="
                + std::to_string(arc_lengths[static_cast<size_t>(i)]) + " and s="
                + std::to_string(arc_lengths[static_cast<size_t>(j)])
                + " (distance=" + std::to_string(distance) + " m, heading difference="
                + std::to_string(heading_difference) + " rad)";
        }
    }
    return {};
}

} // anonymous namespace

GeometryCertificate certify_trajectory_geometry(
    const MincoTrajectory& trajectory,
    const std::vector<Eigen::Vector2d>& seed_tangents,
    const TrajectoryLimits& limits,
    const GeometryCertificateParams& params
) {
    GeometryCertificate certificate;
    const int segments = trajectory.segment_count();
    if (segments < 1) {
        certificate.rejection = "trajectory has no segments";
        return certificate;
    }
    if (static_cast<int>(seed_tangents.size()) != segments + 1) {
        certificate.rejection = "seed tangent count does not match the segment boundaries";
        return certificate;
    }
    if (!std::isfinite(trajectory.total_time()) || trajectory.total_time() <= 0.0
        || !std::isfinite(trajectory.total_arc_length())
        || trajectory.total_arc_length() <= 0.0) {
        certificate.rejection = "trajectory has an invalid total time or arc length";
        return certificate;
    }

    for (int segment = 0; segment < segments; ++segment) {
        const ControlPoints points = trajectory.segment_bezier_control_points(segment);
        if (!points.allFinite()) {
            certificate.rejection = "segment " + std::to_string(segment)
                + " contains non-finite control points";
            return certificate;
        }
        // 段内种子方向取两端切向的角平分线，容许种子在段内的有限转向。
        const Eigen::Vector2d seed_sum = seed_tangents[static_cast<size_t>(segment)]
            + seed_tangents[static_cast<size_t>(segment + 1)];
        if (seed_sum.norm() <= 1e-9) {
            certificate.rejection = "seed direction reverses across segment "
                + std::to_string(segment);
            return certificate;
        }
        const std::string failure = certify_interval(
            points,
            seed_sum.normalized(),
            limits,
            params.max_subdivision_depth,
            segment,
            trajectory.segment_duration(segment),
            0.0,
            1.0,
            0
        );
        if (!failure.empty()) {
            certificate.rejection = std::move(failure);
            return certificate;
        }
    }

    std::string unobservable = find_unobservable_branch_pair(trajectory, params);
    if (!unobservable.empty()) {
        certificate.rejection = std::move(unobservable);
        return certificate;
    }

    certificate.valid = true;
    return certificate;
}

} // namespace nav_executor
