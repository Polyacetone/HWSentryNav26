#include <nav_executor/common/trajectory/minco_trajectory.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace nav_executor {

namespace {

constexpr double MIN_SEGMENT_DURATION = 1e-6;
} // anonymous namespace

MincoTrajectory::MincoTrajectory(
    std::vector<double> durations,
    std::vector<CoefBlock> coeffs,
    const GeometryLimits geometry_limits
)
    : durations_(std::move(durations)),
      coeffs_(std::move(coeffs)),
      geometry_limits_(geometry_limits) {
    cumulative_times_.assign(durations_.size() + 1, 0.0);
    std::partial_sum(durations_.begin(), durations_.end(), cumulative_times_.begin() + 1);
    total_time_ = cumulative_times_.empty() ? 0.0 : cumulative_times_.back();
    compute_arc_length();
}

void MincoTrajectory::compute_arc_length() {
    constexpr int SUBDIVISIONS_PER_SEGMENT = 32;
    constexpr std::array<double, 5> GAUSS_NODES {
        -0.9061798459386640,
        -0.5384693101056831,
        0.0,
        0.5384693101056831,
        0.9061798459386640,
    };
    constexpr std::array<double, 5> GAUSS_WEIGHTS {
        0.2369268850561891,
        0.4786286704993665,
        0.5688888888888889,
        0.4786286704993665,
        0.2369268850561891,
    };
    total_arc_length_ = 0.0;
    arc_taus_.clear();
    arc_samples_.clear();
    if (durations_.empty() || total_time_ <= MIN_SEGMENT_DURATION) return;
    arc_taus_.reserve(durations_.size() * SUBDIVISIONS_PER_SEGMENT + 1);
    arc_samples_.reserve(durations_.size() * SUBDIVISIONS_PER_SEGMENT + 1);
    arc_taus_.push_back(0.0);
    arc_samples_.push_back(0.0);

    for (int segment = 0; segment < segment_count(); ++segment) {
        const double duration = durations_[static_cast<size_t>(segment)];
        for (int subdivision = 0; subdivision < SUBDIVISIONS_PER_SEGMENT; ++subdivision) {
            const double local_begin = duration * static_cast<double>(subdivision)
                / static_cast<double>(SUBDIVISIONS_PER_SEGMENT);
            const double local_end = duration * static_cast<double>(subdivision + 1)
                / static_cast<double>(SUBDIVISIONS_PER_SEGMENT);
            const double midpoint = 0.5 * (local_begin + local_end);
            const double half_span = 0.5 * (local_end - local_begin);
            double interval_length = 0.0;
            for (size_t i = 0; i < GAUSS_NODES.size(); ++i) {
                const double local_time = midpoint + half_span * GAUSS_NODES[i];
                const TrajSample sample = sample_at(segment, local_time);
                interval_length += GAUSS_WEIGHTS[i]
                    * sample.dp_dtau.norm() / total_time_;
            }
            total_arc_length_ += half_span * interval_length;
            const double absolute_time = cumulative_times_[static_cast<size_t>(segment)]
                + local_end;
            arc_taus_.push_back(absolute_time / total_time_);
            arc_samples_.push_back(total_arc_length_);
        }
    }
}

double MincoTrajectory::arc_length_at_tau(const double tau) const {
    if (arc_taus_.size() < 2 || arc_samples_.size() != arc_taus_.size()) return 0.0;
    const double clamped = std::clamp(tau, 0.0, 1.0);
    const auto upper_it = std::lower_bound(arc_taus_.begin(), arc_taus_.end(), clamped);
    if (upper_it == arc_taus_.begin()) return 0.0;
    if (upper_it == arc_taus_.end()) return total_arc_length_;
    const size_t upper = static_cast<size_t>(std::distance(arc_taus_.begin(), upper_it));
    const size_t lower = upper - 1;
    const double tau_span = arc_taus_[upper] - arc_taus_[lower];
    const double frac = tau_span > 1e-12
        ? (clamped - arc_taus_[lower]) / tau_span
        : 0.0;
    return std::lerp(arc_samples_[lower], arc_samples_[upper], frac);
}

double MincoTrajectory::segment_boundary_tau(const int boundary_index) const {
    if (durations_.empty() || total_time_ <= MIN_SEGMENT_DURATION) return 0.0;
    const int clamped = std::clamp(boundary_index, 0, segment_count());
    return cumulative_times_[static_cast<size_t>(clamped)] / total_time_;
}

double MincoTrajectory::segment_boundary_arc_length(const int boundary_index) const {
    return arc_length_at_tau(segment_boundary_tau(boundary_index));
}

double MincoTrajectory::segment_duration(const int segment_index) const {
    if (durations_.empty()) return 0.0;
    return durations_[static_cast<size_t>(
        std::clamp(segment_index, 0, segment_count() - 1)
    )];
}

double MincoTrajectory::tau_at_arc_length(const double arc_length) const {
    if (arc_samples_.size() < 2 || total_arc_length_ <= 1e-12) return 0.0;
    const double clamped = std::clamp(arc_length, 0.0, total_arc_length_);
    const auto upper_it = std::lower_bound(arc_samples_.begin(), arc_samples_.end(), clamped);
    if (upper_it == arc_samples_.begin()) return 0.0;
    if (upper_it == arc_samples_.end()) return 1.0;
    const size_t upper = static_cast<size_t>(std::distance(arc_samples_.begin(), upper_it));
    const size_t lower = upper - 1;
    const double span = arc_samples_[upper] - arc_samples_[lower];
    const double frac = span > 1e-12 ? (clamped - arc_samples_[lower]) / span : 0.0;
    return std::lerp(arc_taus_[lower], arc_taus_[upper], frac);
}

MincoTrajectory::Locator MincoTrajectory::locate_time(const double t) const {
    Locator loc;
    if (durations_.empty()) return loc;

    const double clamped = std::clamp(t, 0.0, total_time_);
    const auto upper = std::upper_bound(cumulative_times_.begin(), cumulative_times_.end(), clamped);
    int segment = static_cast<int>(std::distance(cumulative_times_.begin(), upper)) - 1;
    segment = std::clamp(segment, 0, segment_count() - 1);
    loc.segment = segment;
    loc.local_t = clamped - cumulative_times_[static_cast<size_t>(segment)];
    return loc;
}

TrajSample MincoTrajectory::sample_at(const int segment, const double local_t) const {
    // 段内 2D 平坦多项式求值，导数对**真实时间**（局部时间即真实时间）。τ 换算在此处完成。
    const CoefBlock& coef = coeffs_[static_cast<size_t>(segment)];

    Eigen::Vector2d pos = Eigen::Vector2d::Zero();
    Eigen::Vector2d vel_t = Eigen::Vector2d::Zero();
    Eigen::Vector2d acc_t = Eigen::Vector2d::Zero();
    Eigen::Vector2d jerk_t = Eigen::Vector2d::Zero();

    double t_pow = 1.0;                 // t^k
    for (int k = 0; k < NCOEF; ++k) {
        for (int dim = 0; dim < DIM; ++dim) pos(dim) += coef(k, dim) * t_pow;
        t_pow *= local_t;
    }
    t_pow = 1.0; // 一阶导：Σ_{k>=1} k c_k t^{k-1}
    for (int k = 1; k < NCOEF; ++k) {
        for (int dim = 0; dim < DIM; ++dim) vel_t(dim) += static_cast<double>(k) * coef(k, dim) * t_pow;
        t_pow *= local_t;
    }
    t_pow = 1.0; // 二阶导：Σ_{k>=2} k(k-1) c_k t^{k-2}
    for (int k = 2; k < NCOEF; ++k) {
        for (int dim = 0; dim < DIM; ++dim) acc_t(dim) += static_cast<double>(k * (k - 1)) * coef(k, dim) * t_pow;
        t_pow *= local_t;
    }
    t_pow = 1.0; // 三阶导：Σ_{k>=3} k(k-1)(k-2) c_k t^{k-3}
    for (int k = 3; k < NCOEF; ++k) {
        for (int dim = 0; dim < DIM; ++dim) {
            jerk_t(dim) += static_cast<double>(k * (k - 1) * (k - 2)) * coef(k, dim) * t_pow;
        }
        t_pow *= local_t;
    }

    // 真实时间导数 → τ 导数：τ = t/total_time ⇒ d/dτ = total_time · d/dt。
    const double time_scale = total_time_;
    TrajSample out;
    out.p = pos;
    out.dp_dtau = vel_t * time_scale;
    out.ddp_dtau = acc_t * time_scale * time_scale;
    out.ds_dtau = out.dp_dtau.norm();
    if (!std::isfinite(out.ds_dtau) || out.ds_dtau == 0.0) return out;

    const Eigen::Vector2d third = jerk_t * time_scale * time_scale * time_scale;
    const auto cross = [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        return a.x() * b.y() - a.y() * b.x();
    };
    // 发布前的数值验收排除检测到的内部 cusp。
    out.theta = std::atan2(out.dp_dtau.y(), out.dp_dtau.x());
    // 正则尺度定义在单段归一化参数 u=t/T_i 下，换到全局 τ 后与导数同比缩放。
    // 因而曲率与曲率变化率在任意全局时标缩放下保持不变。
    const double duration = durations_[static_cast<size_t>(segment)];
    const double tangent_regularization = geometry_limits_.tangent_regularization
        * total_time_ / duration;
    const double speed_squared = out.dp_dtau.squaredNorm()
        + tangent_regularization * tangent_regularization;
    const double speed = std::sqrt(speed_squared);
    const double speed_cubed = speed_squared * speed;
    const double turn = cross(out.dp_dtau, out.ddp_dtau);
    out.kappa = turn / speed_cubed;
    // dκ/ds = (1/|p'|)·d/dτ[ det(p',p'')/|p'|³ ]，其中 d/dτ det(p',p'') = det(p',p''')。
    out.kappa_rate = cross(out.dp_dtau, third) / (speed_squared * speed_squared)
        - 3.0 * turn * out.dp_dtau.dot(out.ddp_dtau)
            / (speed_squared * speed_squared * speed_squared);
    return out;
}

TrajSample MincoTrajectory::eval_time(const double t) const {
    if (durations_.empty()) return TrajSample {};

    if (t >= 0.0 && t <= total_time_) {
        const Locator loc = locate_time(t);
        return sample_at(loc.segment, loc.local_t);
    }

    // 超界外推为端点切线上的直线：位置线性延伸，航向保持，二阶量与曲率归零。
    const bool before = t < 0.0;
    const double edge_t = before ? 0.0 : total_time_;
    const Locator loc = locate_time(edge_t);
    const TrajSample edge = sample_at(loc.segment, loc.local_t);

    const double dtau = (t - edge_t) / std::max(total_time_, MIN_SEGMENT_DURATION);
    TrajSample out = edge;
    out.p = edge.p + edge.dp_dtau * dtau;
    out.ddp_dtau = Eigen::Vector2d::Zero();
    out.kappa = 0.0;
    out.kappa_rate = 0.0;
    return out;
}

TrajSample MincoTrajectory::eval(const double tau) const {
    return eval_time(tau * total_time_);
}

TrajSample MincoTrajectory::eval_arc_length(const double arc_length) const {
    return eval(tau_at_arc_length(arc_length));
}

MincoTrajectory::ControlPointBlock MincoTrajectory::segment_bezier_control_points(
    const int segment_index
) const {
    ControlPointBlock control_points = ControlPointBlock::Zero();
    if (coeffs_.empty()) return control_points;
    const int segment = std::clamp(segment_index, 0, segment_count() - 1);
    const CoefBlock& coefficients = coeffs_[static_cast<size_t>(segment)];
    const double duration = durations_[static_cast<size_t>(segment)];
    constexpr int BINOMIAL[NCOEF][NCOEF] {
        {1, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 0},
        {1, 2, 1, 0, 0, 0},
        {1, 3, 3, 1, 0, 0},
        {1, 4, 6, 4, 1, 0},
        {1, 5, 10, 10, 5, 1},
    };
    std::array<double, NCOEF> duration_powers {};
    duration_powers[0] = 1.0;
    for (int i = 1; i < NCOEF; ++i) {
        duration_powers[static_cast<size_t>(i)]
            = duration_powers[static_cast<size_t>(i - 1)] * duration;
    }
    for (int i = 0; i < NCOEF; ++i) {
        for (int k = 0; k <= i; ++k) {
            const double factor = static_cast<double>(BINOMIAL[i][k])
                / static_cast<double>(BINOMIAL[DEGREE][k]);
            control_points.row(i) += factor
                * duration_powers[static_cast<size_t>(k)] * coefficients.row(k);
        }
    }
    return control_points;
}

bool MincoTrajectory::operator==(const MincoTrajectory& other) const {
    if (durations_.size() != other.durations_.size()) return false;
    for (size_t i = 0; i < durations_.size(); ++i) {
        if (durations_[i] != other.durations_[i]) return false;
    }
    for (size_t i = 0; i < coeffs_.size(); ++i) {
        if (coeffs_[i] != other.coeffs_[i]) return false;
    }
    return true;
}

} // namespace nav_executor
