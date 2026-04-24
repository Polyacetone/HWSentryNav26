#include <path_follower/utils.hpp>

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
}