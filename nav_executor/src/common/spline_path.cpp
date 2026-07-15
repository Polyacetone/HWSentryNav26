#include <nav_executor/common/spline_path.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <Eigen/Core>

namespace nav_executor {

namespace {

constexpr int ARC_LENGTH_ESTIMATE_SAMPLES = 200;
constexpr double ARC_LENGTH_SAMPLE_SPACING = 0.05;
constexpr int ARC_LENGTH_MAX_SAMPLES = 20000;
constexpr int PROJECTION_REFINEMENT_ITERATIONS = 16;

} // anonymous namespace

SplinePath::SplinePath(const std::vector<Eigen::Vector2d>& cps) : spline_(cps) {
    spline_.setExtrapolate(true);
    build_arc_length_table();
}

SplinePath::SplinePath(std::vector<Eigen::Vector2d>&& cps) : spline_(std::move(cps)) {
    spline_.setExtrapolate(true);
    build_arc_length_table();
}

void SplinePath::build_arc_length_table() {
    double estimated_length = 0.0;
    Eigen::Vector2d previous = spline_.evaluate(0.0);
    for (int i = 1; i <= ARC_LENGTH_ESTIMATE_SAMPLES; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(ARC_LENGTH_ESTIMATE_SAMPLES);
        const Eigen::Vector2d current = spline_.evaluate(u);
        estimated_length += (current - previous).norm();
        previous = current;
    }

    const int sample_count = std::clamp(
        static_cast<int>(std::ceil(estimated_length / ARC_LENGTH_SAMPLE_SPACING)),
        ARC_LENGTH_ESTIMATE_SAMPLES,
        ARC_LENGTH_MAX_SAMPLES
    );
    sample_arc_lengths_.assign(static_cast<size_t>(sample_count + 1), 0.0);

    previous = spline_.evaluate(0.0);
    for (int i = 1; i <= sample_count; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(sample_count);
        const Eigen::Vector2d current = spline_.evaluate(u);
        sample_arc_lengths_[static_cast<size_t>(i)] = sample_arc_lengths_[static_cast<size_t>(i - 1)]
            + (current - previous).norm();
        previous = current;
    }
    total_length_ = sample_arc_lengths_.back();
}

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

double SplinePath::arc_length(const double u0, const double u1, const int) const {
    return std::abs(arc_length_at_u(u1) - arc_length_at_u(u0));
}

double SplinePath::arc_length_at_u(const double u) const {
    if (sample_arc_lengths_.empty()) return 0.0;
    const double clamped_u = std::clamp(u, 0.0, 1.0);
    const double scaled = clamped_u * static_cast<double>(sample_arc_lengths_.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(scaled));
    const size_t upper = std::min(lower + 1, sample_arc_lengths_.size() - 1);
    const double t = scaled - static_cast<double>(lower);
    return std::lerp(sample_arc_lengths_[lower], sample_arc_lengths_[upper], t);
}

double SplinePath::u_at_arc_length(const double arc_length) const {
    if (sample_arc_lengths_.empty() || total_length_ <= 1e-12) return 0.0;
    const double clamped = std::clamp(arc_length, 0.0, total_length_);
    const auto upper_it = std::lower_bound(sample_arc_lengths_.begin(), sample_arc_lengths_.end(), clamped);
    if (upper_it == sample_arc_lengths_.begin()) return 0.0;
    if (upper_it == sample_arc_lengths_.end()) return 1.0;

    const size_t upper = static_cast<size_t>(std::distance(sample_arc_lengths_.begin(), upper_it));
    const size_t lower = upper - 1;
    const double span = sample_arc_lengths_[upper] - sample_arc_lengths_[lower];
    const double t = span > 1e-12 ? (clamped - sample_arc_lengths_[lower]) / span : 0.0;
    return (static_cast<double>(lower) + t) / static_cast<double>(sample_arc_lengths_.size() - 1);
}

std::optional<PathProjection> SplinePath::project(
    const Eigen::Vector2d& pos, double min_arc_length, double max_arc_length
) const {
    if (sample_arc_lengths_.size() < 2) return std::nullopt;
    min_arc_length = std::clamp(min_arc_length, 0.0, total_length_);
    max_arc_length = std::clamp(max_arc_length, 0.0, total_length_);
    if (min_arc_length > max_arc_length) std::swap(min_arc_length, max_arc_length);

    const double min_u = u_at_arc_length(min_arc_length);
    const double max_u = u_at_arc_length(max_arc_length);
    const int table_segments = static_cast<int>(sample_arc_lengths_.size() - 1);
    const int first = std::clamp(static_cast<int>(std::floor(min_u * table_segments)), 0, table_segments - 1);
    const int last = std::clamp(static_cast<int>(std::ceil(max_u * table_segments)), first + 1, table_segments);

    double best_u = min_u;
    double best_d2 = (spline_.evaluate(min_u) - pos).squaredNorm();
    for (int i = first; i < last; ++i) {
        const double ua = std::max(min_u, static_cast<double>(i) / table_segments);
        const double ub = std::min(max_u, static_cast<double>(i + 1) / table_segments);
        if (ub < ua) continue;
        const Eigen::Vector2d a = spline_.evaluate(ua);
        const Eigen::Vector2d b = spline_.evaluate(ub);
        const Eigen::Vector2d ab = b - a;
        const double t = ab.squaredNorm() > 1e-12
            ? std::clamp((pos - a).dot(ab) / ab.squaredNorm(), 0.0, 1.0)
            : 0.0;
        const double candidate_u = std::lerp(ua, ub, t);
        const double d2 = (spline_.evaluate(candidate_u) - pos).squaredNorm();
        if (d2 < best_d2) {
            best_d2 = d2;
            best_u = candidate_u;
        }
    }

    double left = std::max(min_u, best_u - 1.0 / table_segments);
    double right = std::min(max_u, best_u + 1.0 / table_segments);
    for (int i = 0; i < PROJECTION_REFINEMENT_ITERATIONS; ++i) {
        const double m1 = left + (right - left) / 3.0;
        const double m2 = right - (right - left) / 3.0;
        if ((spline_.evaluate(m1) - pos).squaredNorm() <= (spline_.evaluate(m2) - pos).squaredNorm()) {
            right = m2;
        } else {
            left = m1;
        }
    }
    best_u = 0.5 * (left + right);
    const Eigen::Vector2d projected = spline_.evaluate(best_u);
    return PathProjection {
        .u = best_u,
        .arc_length = arc_length_at_u(best_u),
        .position = projected,
        .distance = (projected - pos).norm(),
    };
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
