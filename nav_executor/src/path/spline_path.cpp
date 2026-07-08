#include <nav_executor/path/spline_path.hpp>

#include <algorithm>
#include <cmath>
#include <Eigen/Core>

namespace nav_executor {

SplineEval SplinePath::eval(const double u) const {
    SplineEval out;
    if (spline_.getControlPoints().size() < 3) {
        out.p = Eigen::Vector2d::Zero();
        out.d1 = Eigen::Vector2d::Zero();
        out.d2 = Eigen::Vector2d::Zero();
        return out;
    }

    if (u >= 0.0 && u <= 1.0) {
        out.p = spline_.evaluate(u);
        out.d1 = spline_.derivative(u, 1);
        out.d2 = spline_.derivative(u, 2);
    } else {
        Eigen::Vector2d p_edge, d1_edge;
        if (u < 0.0) {
            p_edge = spline_.evaluate(0.0);
            d1_edge = spline_.derivative(0.0, 1);
            out.p = p_edge + d1_edge * u;
            out.d1 = d1_edge;
        } else {
            p_edge = spline_.evaluate(1.0);
            d1_edge = spline_.derivative(1.0, 1);
            out.p = p_edge + d1_edge * (u - 1.0);
            out.d1 = d1_edge;
        }
        out.d2 = Eigen::Vector2d::Zero();
    }

    out.thetar = std::atan2(out.d1.y(), out.d1.x());
    out.sin_r = std::sin(out.thetar);
    out.cos_r = std::cos(out.thetar);
    out.ds_du = out.d1.norm();

    const double dsdu_sq = out.ds_du * out.ds_du;
    const double cross = out.d1.x() * out.d2.y() - out.d1.y() * out.d2.x();
    out.kappa = cross / (dsdu_sq * out.ds_du + 1e-12);

    return out;
}

double SplinePath::arc_length(const double u0, const double u1, int samples) const {
    const int n = std::max(1, samples);
    const double ua = std::clamp(u0, 0.0, 1.0);
    const double ub = std::clamp(u1, 0.0, 1.0);
    if (std::abs(ub - ua) <= 1e-9) {
        return 0.0;
    }

    const double h = (ub - ua) / static_cast<double>(n);
    double acc = 0.0;
    for (int i = 0; i <= n; ++i) {
        const Eigen::Vector2d d1 = spline_.derivative(ua + h * static_cast<double>(i), 1);
        const double w = (i == 0 || i == n) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
        acc += w * d1.norm();
    }
    return std::abs(h) * acc / 3.0;
}

double SplinePath::project(
    const Eigen::Vector2d& pos, const double u_hint,
    const int num_samples, const double search_window,
    const double local_search_lazy_distance
) const {
    const auto search = [&](const double a, const double b, const int n) {
        double best_u = a;
        double best_d2 = std::numeric_limits<double>::infinity();
        for (int i = 0; i <= n; i++) {
            const double u = a + (b - a) * (static_cast<double>(i) / static_cast<double>(n));
            const Eigen::Vector2d p = spline_.evaluate(u);
            const double d2 = (p - pos).squaredNorm();
            if (d2 < best_d2) {
                best_d2 = d2;
                best_u = u;
            }
        }
        return best_u;
    };

    double u_best = search(
        std::clamp(u_hint - search_window, 0.0, 1.0),
        std::clamp(u_hint + search_window, 0.0, 1.0),
        num_samples
    );

    if ((spline_.evaluate(u_best) - pos).norm() <= local_search_lazy_distance) {
        return std::clamp(u_best, 0.0, 1.0);
    }

    u_best = search(0.0, 1.0, num_samples);
    u_best = search(
        std::clamp(u_best - search_window, 0.0, 1.0),
        std::clamp(u_best + search_window, 0.0, 1.0),
        num_samples
    );

    return std::clamp(u_best, 0.0, 1.0);
}

double SplinePath::project_extrapolated(
    const Eigen::Vector2d& pos, const double u_hint,
    const int num_samples, const double search_window,
    const double local_search_lazy_distance,
    const double u_min, const double u_max
) const {
    double u_best = project(
        pos,
        std::clamp(u_hint, 0.0, 1.0),
        num_samples,
        search_window,
        local_search_lazy_distance
    );

    double best_d2 = (spline_.evaluate(u_best) - pos).squaredNorm();

    const auto check_endpoint_extension = [&](const double u_edge) {
        const Eigen::Vector2d p_edge = spline_.evaluate(u_edge);
        const Eigen::Vector2d d1_edge = spline_.derivative(u_edge, 1);
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

    return std::clamp(u_best, u_min, u_max);
}

bool SplinePath::operator==(const SplinePath& other) const {
    const auto& a = spline_.getControlPoints();
    const auto& b = other.spline_.getControlPoints();
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

}
