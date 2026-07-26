#include <nav_executor/common/trajectory/trajectory_topology.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace nav_executor {

namespace {

struct TrajectoryEdge {
    Eigen::Vector2d begin;
    Eigen::Vector2d end;
    int segment = 0;
    double progress_begin = 0.0;
    double progress_end = 0.0;
    double geometric_error_bound = 0.0;
};

double cross_2d(const Eigen::Vector2d& lhs, const Eigen::Vector2d& rhs) {
    return lhs.x() * rhs.y() - lhs.y() * rhs.x();
}

bool point_on_segment(
    const Eigen::Vector2d& point,
    const Eigen::Vector2d& begin,
    const Eigen::Vector2d& end,
    const double epsilon
) {
    const Eigen::Vector2d direction = end - begin;
    return std::abs(cross_2d(direction, point - begin))
            <= epsilon * std::max(direction.norm(), epsilon)
        && point.x() >= std::min(begin.x(), end.x()) - epsilon
        && point.x() <= std::max(begin.x(), end.x()) + epsilon
        && point.y() >= std::min(begin.y(), end.y()) - epsilon
        && point.y() <= std::max(begin.y(), end.y()) + epsilon;
}

double point_segment_distance(
    const Eigen::Vector2d& point,
    const Eigen::Vector2d& begin,
    const Eigen::Vector2d& end
) {
    const Eigen::Vector2d direction = end - begin;
    const double length_squared = direction.squaredNorm();
    if (length_squared <= 1e-20) return (point - begin).norm();
    const double fraction = std::clamp(
        (point - begin).dot(direction) / length_squared, 0.0, 1.0
    );
    return (point - (begin + fraction * direction)).norm();
}

bool edges_may_intersect(const TrajectoryEdge& lhs, const TrajectoryEdge& rhs) {
    constexpr double EPSILON = 1e-7;
    const Eigen::Vector2d lhs_direction = lhs.end - lhs.begin;
    const Eigen::Vector2d rhs_direction = rhs.end - rhs.begin;
    const double c1 = cross_2d(lhs_direction, rhs.begin - lhs.begin);
    const double c2 = cross_2d(lhs_direction, rhs.end - lhs.begin);
    const double c3 = cross_2d(rhs_direction, lhs.begin - rhs.begin);
    const double c4 = cross_2d(rhs_direction, lhs.end - rhs.begin);
    const double lhs_cross_tolerance = EPSILON
        * std::max(lhs_direction.norm(), EPSILON);
    const double rhs_cross_tolerance = EPSILON
        * std::max(rhs_direction.norm(), EPSILON);
    if (((c1 > lhs_cross_tolerance && c2 < -lhs_cross_tolerance)
            || (c1 < -lhs_cross_tolerance && c2 > lhs_cross_tolerance))
        && ((c3 > rhs_cross_tolerance && c4 < -rhs_cross_tolerance)
            || (c3 < -rhs_cross_tolerance && c4 > rhs_cross_tolerance))) {
        return true;
    }
    const bool chords_intersect = (std::abs(c1) <= lhs_cross_tolerance
            && point_on_segment(rhs.begin, lhs.begin, lhs.end, EPSILON))
        || (std::abs(c2) <= lhs_cross_tolerance
            && point_on_segment(rhs.end, lhs.begin, lhs.end, EPSILON))
        || (std::abs(c3) <= rhs_cross_tolerance
            && point_on_segment(lhs.begin, rhs.begin, rhs.end, EPSILON))
        || (std::abs(c4) <= rhs_cross_tolerance
            && point_on_segment(lhs.end, rhs.begin, rhs.end, EPSILON));
    if (chords_intersect) return true;

    const double chord_distance = std::min({
        point_segment_distance(lhs.begin, rhs.begin, rhs.end),
        point_segment_distance(lhs.end, rhs.begin, rhs.end),
        point_segment_distance(rhs.begin, lhs.begin, lhs.end),
        point_segment_distance(rhs.end, lhs.begin, lhs.end),
    });
    return chord_distance <= lhs.geometric_error_bound
        + rhs.geometric_error_bound + EPSILON;
}

std::vector<TrajectoryEdge> flatten_trajectory(
    const MincoTrajectory& trajectory,
    const TrajectoryTopologyParams& params
) {
    std::vector<TrajectoryEdge> edges;
    edges.reserve(static_cast<size_t>(trajectory.segment_count() * 16));
    for (int segment = 0; segment < trajectory.segment_count(); ++segment) {
        const double tau_begin = trajectory.segment_boundary_tau(segment);
        const double tau_end = trajectory.segment_boundary_tau(segment + 1);
        const auto append_flattened = [&, segment, tau_begin, tau_end](
            auto&& self,
            const MincoTrajectory::ControlPointBlock& control_points,
            const double u_begin,
            const double u_end,
            const int depth
        ) -> void {
            const Eigen::Vector2d begin = control_points.row(0).transpose();
            const Eigen::Vector2d end = control_points.row(MincoTrajectory::DEGREE).transpose();
            const Eigen::Vector2d chord = end - begin;
            const double chord_length = chord.norm();
            double flatness = 0.0;
            bool control_polygon_is_monotone = true;
            double previous_projection = 0.0;
            for (int i = 1; i < MincoTrajectory::DEGREE; ++i) {
                const Eigen::Vector2d point = control_points.row(i).transpose();
                const double distance = chord_length > 1e-12
                    ? std::abs(cross_2d(chord, point - begin)) / chord_length
                    : (point - begin).norm();
                flatness = std::max(flatness, distance);
                if (chord_length > 1e-12) {
                    const double projection = (point - begin).dot(chord)
                        / chord.squaredNorm();
                    if (projection < previous_projection - 1e-12
                        || projection > 1.0 + 1e-12) {
                        control_polygon_is_monotone = false;
                    }
                    previous_projection = projection;
                }
            }
            constexpr int MAX_SUBDIVISION_DEPTH = 16;
            if ((flatness <= params.flatness_tolerance
                    && control_polygon_is_monotone
                    && chord_length <= params.max_edge_length)
                || depth >= MAX_SUBDIVISION_DEPTH) {
                if (chord.squaredNorm() <= 1e-16) return;
                const double begin_tau = std::lerp(tau_begin, tau_end, u_begin);
                const double end_tau = std::lerp(tau_begin, tau_end, u_end);
                edges.push_back({
                    .begin = begin,
                    .end = end,
                    .segment = segment,
                    .progress_begin = trajectory.arc_length_at_tau(begin_tau),
                    .progress_end = trajectory.arc_length_at_tau(end_tau),
                    .geometric_error_bound = flatness,
                });
                return;
            }

            MincoTrajectory::ControlPointBlock work = control_points;
            MincoTrajectory::ControlPointBlock left;
            MincoTrajectory::ControlPointBlock right;
            left.row(0) = work.row(0);
            right.row(MincoTrajectory::DEGREE) = work.row(MincoTrajectory::DEGREE);
            for (int level = 1; level <= MincoTrajectory::DEGREE; ++level) {
                for (int i = 0; i <= MincoTrajectory::DEGREE - level; ++i) {
                    work.row(i) = 0.5 * (work.row(i) + work.row(i + 1));
                }
                left.row(level) = work.row(0);
                right.row(MincoTrajectory::DEGREE - level)
                    = work.row(MincoTrajectory::DEGREE - level);
            }
            const double midpoint = 0.5 * (u_begin + u_end);
            self(self, left, u_begin, midpoint, depth + 1);
            self(self, right, midpoint, u_end, depth + 1);
        };
        append_flattened(
            append_flattened,
            trajectory.segment_bezier_control_points(segment),
            0.0,
            1.0,
            0
        );
    }
    return edges;
}

} // anonymous namespace

std::optional<TrajectorySelfIntersection> find_possible_self_intersection(
    const MincoTrajectory& trajectory,
    const TrajectoryTopologyParams& params
) {
    const std::vector<TrajectoryEdge> edges = flatten_trajectory(trajectory, params);
    for (size_t i = 0; i < edges.size(); ++i) {
        for (size_t j = i + 1; j < edges.size(); ++j) {
            if (j == i + 1) continue;
            const TrajectoryEdge& lhs = edges[i];
            const TrajectoryEdge& rhs = edges[j];
            if (!edges_may_intersect(lhs, rhs)) continue;
            return TrajectorySelfIntersection {
                .first_segment = lhs.segment,
                .second_segment = rhs.segment,
            };
        }
    }
    return std::nullopt;
}

} // namespace nav_executor
