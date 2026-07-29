#include <nav_executor/path_executor/monitoring/recovery_helpers.hpp>
#include <nav_executor/path_executor/state/state_machine.hpp>

namespace nav_executor::recovery_helpers {

std::optional<FieldSample> sample_fields(
    const CostMap& cost_map,
    const DirectionMap& dir_map,
    const Eigen::Vector2d& pos
) {
    const auto cost = cost_map.sample_map(pos);
    const auto direction = dir_map.sample_map(pos);
    if (!cost || !direction) return std::nullopt;

    FieldSample s;
    s.cost = cost->value;
    s.step_norm = direction->value.norm();
    return s;
}

bool is_safe_goal(const RecoveryParams& p, const FieldSample& s) {
    return (s.cost < p.safe_cost_threshold) && (s.step_norm < p.safe_step_norm_threshold);
}

std::optional<PathScore> score_candidate_by_path_integral(
    const RecoveryParams& p,
    const CostMap& cost_map,
    const DirectionMap& dir_map,
    const Eigen::Vector2d& origin,
    const Eigen::Vector2d& goal,
    const double radius,
    const DirectionMap* base_dir_map
) {
    double acc = 0.0;
    std::optional<FieldSample> end_s;
    const int n = std::max(1, static_cast<int>(radius / p.path_integral_resolution));
    const Eigen::Vector2d path_dir = (goal - origin);
    const double path_dir_norm = path_dir.norm();

    for (int i = 0; i <= n; i++) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        const Eigen::Vector2d pos = origin + (goal - origin) * t;
        const auto s = sample_fields(cost_map, dir_map, pos);
        if (!s) return std::nullopt;

        const double cost01 = std::clamp(s->cost / 255.0, 0.0, 1.0);
        acc += p.path_integral_cost_weight * cost01
            + p.path_integral_step_weight * s->step_norm;

        if (base_dir_map && path_dir_norm > 1e-6) {
            if (const auto direction = base_dir_map->sample_map(pos)) {
                const Eigen::Vector2d& step_dir = direction->value;
                if (step_dir.norm() >= p.step_ascent_penalty_norm_threshold) {
                    const double dot = step_dir.normalized().dot(path_dir / path_dir_norm);
                    if (dot >= p.step_ascent_penalty_dot_threshold) {
                        acc += p.step_ascent_penalty_weight * dot;
                    }
                }
            }
        }

        if (i == n) end_s = s;
    }

    if (!end_s || !is_safe_goal(p, *end_s)) {
        return std::nullopt;
    }

    PathScore out;
    out.score = acc;
    out.end_sample = *end_s;
    out.end_safe = is_safe_goal(p, *end_s);
    return out;
}

std::optional<Eigen::Vector2d> find_goal(
    const RecoveryParams& p,
    const CostMap& cost_map,
    const DirectionMap& dir_map,
    const Eigen::Vector3d& chassis_pose,
    const DirectionMap* base_dir_map
) {
    const Eigen::Vector2d origin = chassis_pose.head<2>();

    const double r_min = std::min(p.radius_min, p.radius_max);
    const double r_max = std::max(p.radius_min, p.radius_max);
    const int r_n = std::max(1, p.radius_samples);
    const int a_n = std::max(1, p.angle_samples);

    std::optional<Eigen::Vector2d> best_pt;
    std::optional<PathScore> best_sc;

    for (int ri = 0; ri < r_n; ri++) {
        const double rt = (r_n == 1) ? 0.0 : (static_cast<double>(ri) / static_cast<double>(r_n - 1));
        const double r = r_min + (r_max - r_min) * rt;

        for (int ai = 0; ai < a_n; ai++) {
            const double a = 2.0 * std::numbers::pi * static_cast<double>(ai) / static_cast<double>(a_n);
            const Eigen::Vector2d pt = origin + Eigen::Vector2d(std::cos(a), std::sin(a)) * r;
            const auto field = sample_fields(cost_map, dir_map, pt);
            if (!field) continue;
            if (field->cost >= p.recovery_cost_threshold) continue;
            const auto sc = score_candidate_by_path_integral(p, cost_map, dir_map, origin, pt, r, base_dir_map);
            if (!sc) continue;
            if (!best_sc || sc->score < best_sc->score) {
                best_sc = *sc;
                best_pt = pt;
            }
        }
    }

    return best_pt;
}

Eigen::Vector2d rotate_vector(const Eigen::Vector2d& v, const double angle) {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return Eigen::Vector2d(c * v.x() - s * v.y(), s * v.x() + c * v.y());
}

std::optional<double> max_cost_along_segment(
    const CostMap& cost_map,
    const Eigen::Vector2d& a_map,
    const Eigen::Vector2d& b_map,
    const int samples
) {
    const int n = std::max(1, samples);
    double max_cost = 0.0;
    for (int i = 0; i <= n; i++) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        const Eigen::Vector2d pos = a_map + (b_map - a_map) * t;
        const auto cost = cost_map.sample_map(pos);
        if (!cost) return std::nullopt;
        max_cost = std::max(max_cost, cost->value);
    }
    return max_cost;
}

} // namespace nav_executor::recovery_helpers
