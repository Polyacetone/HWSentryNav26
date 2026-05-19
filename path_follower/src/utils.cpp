#include <path_follower/utils.hpp>
#include <path_follower/state_machine.hpp>

namespace path_follower {
double clamp_path_u_extrapolated(const double u, const double u_min, const double u_max) {
    return std::clamp(u, u_min, u_max);
}

double project_to_spline_u(
    const SplineD& spline,
    const Eigen::Vector2d& pos,
    double u_hint,
    int num_samples,
    double search_window,
    double local_search_lazy_distance
) {
    const auto search = [&](double a, double b, int n) {
        double best_u = a;
        double best_d2 = std::numeric_limits<double>::infinity();
        for (int i = 0; i <= n; i++) {
            const double u = a + (b - a) * (double(i) / double(n));
            const Eigen::Vector2d p = spline.evaluate(u);
            const double d2 = (p - pos).squaredNorm();
            if (d2 < best_d2) {
                best_d2 = d2;
                best_u = u;
            }
        }
        return best_u;
    };

    // 先在局部范围找
    double u_best = search(
        std::clamp(u_hint - search_window, 0.0, 1.0),
        std::clamp(u_hint + search_window, 0.0, 1.0),
        num_samples
    );

    if ((spline.evaluate(u_best) - pos).norm() <= local_search_lazy_distance) {
        return std::clamp(u_best, 0.0, 1.0);
    }

    // 局部找不到再全局找
    u_best = search(0.0, 1.0, num_samples);
    u_best = search(
        std::clamp(u_best - search_window, 0.0, 1.0),
        std::clamp(u_best + search_window, 0.0, 1.0),
        num_samples
    );

    return std::clamp(u_best, 0.0, 1.0);
}

double project_to_spline_u_extrapolated(
    const SplineD& spline,
    const Eigen::Vector2d& pos,
    const double u_hint,
    const int num_samples,
    const double search_window,
    const double local_search_lazy_distance,
    const double u_min,
    const double u_max
) {
    double u_best = project_to_spline_u(
        spline,
        pos,
        std::clamp(u_hint, 0.0, 1.0),
        num_samples,
        search_window,
        local_search_lazy_distance
    );

    double best_d2 = (spline.evaluate(u_best) - pos).squaredNorm();

    const auto check_endpoint_extension = [&](const double u_edge) {
        const Eigen::Vector2d p_edge = spline.evaluate(u_edge);
        const Eigen::Vector2d d1_edge = spline.derivative(u_edge, 1);
        const double tangent_norm2 = d1_edge.squaredNorm();
        if (tangent_norm2 <= 1e-9) return;

        double u_ext = u_edge;
        if (u_edge <= 0.0) {
            u_ext = (pos - p_edge).dot(d1_edge) / tangent_norm2;
            if (u_ext >= 0.0) return;
        } else {
            u_ext = 1.0 + (pos - p_edge).dot(d1_edge) / tangent_norm2;
            if (u_ext <= 1.0) return;
        }

        const Eigen::Vector2d p_ext = p_edge + d1_edge * (u_ext - u_edge);
        const double d2 = (p_ext - pos).squaredNorm();
        if (d2 < best_d2) {
            best_d2 = d2;
            u_best = u_ext;
        }
    };

    check_endpoint_extension(0.0);
    check_endpoint_extension(1.0);

    return clamp_path_u_extrapolated(u_best, u_min, u_max);
}

void eval_quadratic_bspline2(
    const std::vector<Eigen::Vector2d>& cps,
    const double u_in,
    Eigen::Vector2d* p,
    Eigen::Vector2d* d1,
    Eigen::Vector2d* d2
) {
    const int n = static_cast<int>(cps.size());
    if (n < 3) {
        if (p) *p = Eigen::Vector2d::Zero();
        if (d1) *d1 = Eigen::Vector2d::Zero();
        if (d2) *d2 = Eigen::Vector2d::Zero();
        return;
    }

    const double scale = static_cast<double>(n - 2);
    const double u = std::clamp(u_in, 0.0, 1.0);
    const double bx = u * scale;
    const int xi = std::clamp(static_cast<int>(std::floor(bx)), 0, n - 3);
    const double t = bx - static_cast<double>(xi);
    const double omt = 1.0 - t;
    const auto& p0 = cps[static_cast<size_t>(xi)];
    const auto& p1 = cps[static_cast<size_t>(xi + 1)];
    const auto& p2 = cps[static_cast<size_t>(xi + 2)];

    if (p) *p = 0.5 * omt * omt * p0 + 0.5 * (-2.0 * t * t + 2.0 * t + 1.0) * p1 + 0.5 * t * t * p2;
    if (d1) *d1 = (-omt * p0 + (-2.0 * t + 1.0) * p1 + t * p2) * scale;
    if (d2) *d2 = (p0 - 2.0 * p1 + p2) * (scale * scale);
}

void eval_quadratic_bspline2_extrapolated(
    const std::vector<Eigen::Vector2d>& cps,
    const double u_in,
    Eigen::Vector2d* p,
    Eigen::Vector2d* d1,
    Eigen::Vector2d* d2
) {
    if (u_in >= 0.0 && u_in <= 1.0) {
        eval_quadratic_bspline2(cps, u_in, p, d1, d2);
        return;
    }

    Eigen::Vector2d p_edge = Eigen::Vector2d::Zero();
    Eigen::Vector2d d1_edge = Eigen::Vector2d::Zero();
    if (u_in < 0.0) {
        eval_quadratic_bspline2(cps, 0.0, &p_edge, &d1_edge, nullptr);
        if (p) *p = p_edge + d1_edge * u_in;
    } else {
        eval_quadratic_bspline2(cps, 1.0, &p_edge, &d1_edge, nullptr);
        if (p) *p = p_edge + d1_edge * (u_in - 1.0);
    }

    if (d1) *d1 = d1_edge;
    if (d2) *d2 = Eigen::Vector2d::Zero();
}

double quadratic_bspline_curvature(const Eigen::Vector2d& d1, const Eigen::Vector2d& d2) {
    const double dsdu = std::sqrt(d1.squaredNorm() + 0.01) + 1e-6;
    return (d1.x() * d2.y() - d1.y() * d2.x()) / (dsdu * dsdu * dsdu);
}
}

namespace path_follower::recovery_helpers {

std::optional<FieldSample> sample_fields(
    const CostMap& cost_map,
    const DirectionMap& dir_map,
    const Eigen::Vector2d& pos
) {
    const Eigen::Vector2d gc = cost_map.map_coord_to_grid(pos);
    if (!cost_map.is_valid_coord(gc)) return std::nullopt;
    const Eigen::Vector2d gd = dir_map.map_coord_to_grid(pos);
    if (!dir_map.is_valid_coord(gd)) return std::nullopt;

    FieldSample s;
    s.cost = cost_map.interpolate(gc);
    s.step_norm = dir_map.interpolate(gd).norm();
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
            const Eigen::Vector2d gd = base_dir_map->map_coord_to_grid(pos);
            if (base_dir_map->is_valid_coord(gd)) {
                const Eigen::Vector2d step_dir = base_dir_map->interpolate(gd);
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
        const Eigen::Vector2d g = cost_map.map_coord_to_grid(pos);
        if (!cost_map.is_valid_coord(g)) return std::nullopt;
        max_cost = std::max(max_cost, cost_map.interpolate(g));
    }
    return max_cost;
}

} // namespace path_follower::recovery_helpers
