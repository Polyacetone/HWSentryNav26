#include <mpc_controller/utils.hpp>

namespace mpc_controller {
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
}