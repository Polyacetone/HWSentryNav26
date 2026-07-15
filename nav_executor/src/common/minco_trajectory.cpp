#include <nav_executor/common/minco_trajectory.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

#include <Eigen/Dense>

namespace nav_executor {

namespace {

constexpr double MIN_SEGMENT_DURATION = 1e-6;
constexpr double TANGENT_EPS = 1e-9;

} // anonymous namespace

MincoTrajectory::MincoTrajectory(std::vector<double> durations, std::vector<CoefBlock> coeffs)
    : durations_(std::move(durations)), coeffs_(std::move(coeffs)) {
    cumulative_times_.assign(durations_.size() + 1, 0.0);
    std::partial_sum(durations_.begin(), durations_.end(), cumulative_times_.begin() + 1);
    total_time_ = cumulative_times_.empty() ? 0.0 : cumulative_times_.back();
    compute_arc_length();
}

void MincoTrajectory::compute_arc_length() {
    constexpr int ARC_SAMPLES = 400;
    total_arc_length_ = 0.0;
    arc_samples_.assign(ARC_SAMPLES + 1, 0.0);
    if (durations_.empty()) return;
    Eigen::Vector2d prev = eval(0.0).p;
    for (int i = 1; i <= ARC_SAMPLES; ++i) {
        const double tau = static_cast<double>(i) / static_cast<double>(ARC_SAMPLES);
        const Eigen::Vector2d cur = eval(tau).p;
        total_arc_length_ += (cur - prev).norm();
        arc_samples_[static_cast<size_t>(i)] = total_arc_length_;
        prev = cur;
    }
}

double MincoTrajectory::arc_length_at_tau(const double tau) const {
    if (arc_samples_.size() < 2) return 0.0;
    const double clamped = std::clamp(tau, 0.0, 1.0);
    const double scaled = clamped * static_cast<double>(arc_samples_.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(scaled));
    const size_t upper = std::min(lower + 1, arc_samples_.size() - 1);
    const double frac = scaled - static_cast<double>(lower);
    return std::lerp(arc_samples_[lower], arc_samples_[upper], frac);
}

double MincoTrajectory::segment_boundary_tau(const int boundary_index) const {
    if (durations_.empty() || total_time_ <= MIN_SEGMENT_DURATION) return 0.0;
    const int clamped = std::clamp(boundary_index, 0, segment_count());
    return cumulative_times_[static_cast<size_t>(clamped)] / total_time_;
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
    return (static_cast<double>(lower) + frac) / static_cast<double>(arc_samples_.size() - 1);
}

double MincoTrajectory::arc_length_between(const double tau0, const double tau1) const {
    return std::abs(arc_length_at_tau(tau1) - arc_length_at_tau(tau0));
}

MincoTrajectory MincoTrajectory::from_boundary_states(
    const std::vector<BoundaryState>& states,
    const std::vector<double>& durations
) {
    const size_t segment_count = durations.size();
    if (states.size() != segment_count + 1 || segment_count == 0) {
        return MincoTrajectory {};
    }

    std::vector<CoefBlock> coeffs(segment_count);
    for (size_t seg = 0; seg < segment_count; ++seg) {
        const double t_dur = std::max(durations[seg], MIN_SEGMENT_DURATION);
        const BoundaryState& s0 = states[seg];
        const BoundaryState& s1 = states[seg + 1];

        // 每段每维求唯一五次多项式 p(t)，t∈[0,T]，匹配两端 pos/vel/acc（min-jerk）。
        // c0=p0, c1=v0, c2=a0/2 直接给定；c3/c4/c5 解 3×3。
        Eigen::Matrix3d a_mat;
        a_mat << std::pow(t_dur, 3), std::pow(t_dur, 4), std::pow(t_dur, 5),
                 3.0 * std::pow(t_dur, 2), 4.0 * std::pow(t_dur, 3), 5.0 * std::pow(t_dur, 4),
                 6.0 * t_dur, 12.0 * std::pow(t_dur, 2), 20.0 * std::pow(t_dur, 3);
        const Eigen::Matrix3d a_inv = a_mat.inverse();

        CoefBlock& coef = coeffs[seg];
        for (int dim = 0; dim < DIM; ++dim) {
            const double p0 = s0.pos(dim);
            const double v0 = s0.vel(dim);
            const double acc0 = s0.acc(dim);
            const double p1 = s1.pos(dim);
            const double v1 = s1.vel(dim);
            const double acc1 = s1.acc(dim);

            const Eigen::Vector3d rhs(
                p1 - p0 - v0 * t_dur - 0.5 * acc0 * t_dur * t_dur,
                v1 - v0 - acc0 * t_dur,
                acc1 - acc0
            );
            const Eigen::Vector3d higher = a_inv * rhs;

            coef(0, dim) = p0;
            coef(1, dim) = v0;
            coef(2, dim) = 0.5 * acc0;
            coef(3, dim) = higher(0);
            coef(4, dim) = higher(1);
            coef(5, dim) = higher(2);
        }
    }

    return MincoTrajectory(durations, std::move(coeffs));
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
    // 段内多项式求值，导数对**真实时间**（局部时间即真实时间）。τ 换算在 eval 层完成。
    const CoefBlock& coef = coeffs_[static_cast<size_t>(segment)];

    Eigen::Matrix<double, DIM, 1> pos = Eigen::Matrix<double, DIM, 1>::Zero();
    Eigen::Matrix<double, DIM, 1> vel_t = Eigen::Matrix<double, DIM, 1>::Zero();
    Eigen::Matrix<double, DIM, 1> acc_t = Eigen::Matrix<double, DIM, 1>::Zero();

    double t_pow = 1.0;                 // t^k
    for (int k = 0; k < NCOEF; ++k) {
        for (int dim = 0; dim < DIM; ++dim) {
            pos(dim) += coef(k, dim) * t_pow;
        }
        t_pow *= local_t;
    }
    // 一阶导：Σ_{k>=1} k c_k t^{k-1}
    t_pow = 1.0;
    for (int k = 1; k < NCOEF; ++k) {
        for (int dim = 0; dim < DIM; ++dim) {
            vel_t(dim) += static_cast<double>(k) * coef(k, dim) * t_pow;
        }
        t_pow *= local_t;
    }
    // 二阶导：Σ_{k>=2} k(k-1) c_k t^{k-2}
    t_pow = 1.0;
    for (int k = 2; k < NCOEF; ++k) {
        for (int dim = 0; dim < DIM; ++dim) {
            acc_t(dim) += static_cast<double>(k * (k - 1)) * coef(k, dim) * t_pow;
        }
        t_pow *= local_t;
    }

    // 真实时间导数 → τ 导数：τ = t/total_time ⇒ d/dτ = total_time · d/dt。
    const double s = total_time_;
    TrajSample out;
    out.p = pos.head<2>();
    out.theta = pos(2);
    out.dp_dtau = vel_t.head<2>() * s;
    out.ddp_dtau = acc_t.head<2>() * s * s;
    out.dtheta_dtau = vel_t(2) * s;

    // 位置曲线切线帧（几何量，由 τ 导数或时间导数得到的方向一致）。
    out.ds_dtau = out.dp_dtau.norm();
    out.phi = std::atan2(out.dp_dtau.y(), out.dp_dtau.x());
    out.sin_phi = std::sin(out.phi);
    out.cos_phi = std::cos(out.phi);

    const double cross = out.dp_dtau.x() * out.ddp_dtau.y() - out.dp_dtau.y() * out.ddp_dtau.x();
    const double ds = out.ds_dtau;
    out.kappa = ds > TANGENT_EPS ? cross / (ds * ds * ds) : 0.0;
    return out;
}

TrajSample MincoTrajectory::eval_time(const double t) const {
    if (durations_.empty()) return TrajSample {};

    if (t >= 0.0 && t <= total_time_) {
        const Locator loc = locate_time(t);
        return sample_at(loc.segment, loc.local_t);
    }

    // 超界线性外推：保持端点朝向/切线，位置沿端点 τ 导数线性延伸，二阶量归零。
    const bool before = t < 0.0;
    const double edge_t = before ? 0.0 : total_time_;
    const Locator loc = locate_time(edge_t);
    TrajSample edge = sample_at(loc.segment, loc.local_t);

    const double dtau = (t - edge_t) / std::max(total_time_, MIN_SEGMENT_DURATION);
    TrajSample out = edge;
    out.p = edge.p + edge.dp_dtau * dtau;
    out.theta = edge.theta + edge.dtheta_dtau * dtau;
    out.ddp_dtau = Eigen::Vector2d::Zero();
    out.kappa = 0.0;
    return out;
}

TrajSample MincoTrajectory::eval(const double tau) const {
    return eval_time(tau * total_time_);
}

double MincoTrajectory::project(const Eigen::Vector2d& pos, double tau_lo, double tau_hi) const {
    if (durations_.empty()) return 0.0;
    tau_lo = std::clamp(tau_lo, 0.0, 1.0);
    tau_hi = std::clamp(tau_hi, 0.0, 1.0);
    if (tau_lo > tau_hi) std::swap(tau_lo, tau_hi);

    // 粗采样定位最近区段，再三分细化（与 SplinePath::project 同构）。
    constexpr int COARSE_SAMPLES = 40;
    constexpr int REFINE_ITERS = 20;
    double best_tau = tau_lo;
    double best_d2 = (eval(tau_lo).p - pos).squaredNorm();
    for (int i = 1; i <= COARSE_SAMPLES; ++i) {
        const double tau = tau_lo + (tau_hi - tau_lo) * static_cast<double>(i) / COARSE_SAMPLES;
        const double d2 = (eval(tau).p - pos).squaredNorm();
        if (d2 < best_d2) {
            best_d2 = d2;
            best_tau = tau;
        }
    }

    const double span = (tau_hi - tau_lo) / COARSE_SAMPLES;
    double left = std::max(tau_lo, best_tau - span);
    double right = std::min(tau_hi, best_tau + span);
    for (int i = 0; i < REFINE_ITERS; ++i) {
        const double m1 = left + (right - left) / 3.0;
        const double m2 = right - (right - left) / 3.0;
        if ((eval(m1).p - pos).squaredNorm() <= (eval(m2).p - pos).squaredNorm()) {
            right = m2;
        } else {
            left = m1;
        }
    }
    return 0.5 * (left + right);
}

double MincoTrajectory::longitudinal_velocity(const TrajSample& s) const {
    if (total_time_ <= MIN_SEGMENT_DURATION) return 0.0;
    const Eigen::Vector2d dp_dt = s.dp_dtau / total_time_;
    return dp_dt.x() * std::cos(s.theta) + dp_dt.y() * std::sin(s.theta);
}

double MincoTrajectory::angular_velocity(const TrajSample& s) const {
    if (total_time_ <= MIN_SEGMENT_DURATION) return 0.0;
    return s.dtheta_dtau / total_time_;
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

double governed_clock_factor(
    const TrajSample& reference,
    const Eigen::Vector3d& chassis_pose_map,
    const GovernedClockParams& params
) {
    const double ex = chassis_pose_map.x() - reference.p.x();
    const double ey = chassis_pose_map.y() - reference.p.y();
    const double heading_error = std::atan2(
        std::sin(chassis_pose_map.z() - reference.theta),
        std::cos(chassis_pose_map.z() - reference.theta)
    );
    const double weighted_heading_error = params.heading_weight * heading_error;
    const double error_squared = ex * ex + ey * ey
        + weighted_heading_error * weighted_heading_error;
    const double scale_squared = params.error_scale * params.error_scale;
    return scale_squared / (scale_squared + error_squared);
}

} // namespace nav_executor
