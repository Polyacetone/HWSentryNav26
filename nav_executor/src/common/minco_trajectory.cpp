#include <nav_executor/common/minco_trajectory.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

#include <Eigen/Dense>

namespace nav_executor {

namespace {

constexpr double MIN_SEGMENT_DURATION = 1e-6;
constexpr double TANGENT_EPS = 1e-9;
// ω=(ẋÿ−ẏẍ)/‖ṗ‖² 分母的速度正则尺度 (m/s)：抑制 v→0 处 0/0 发散。取较小值，
// 使正常行驶速度（≳0.3 m/s）下的 ω 误差可忽略。
constexpr double OMEGA_SPEED_REG = 0.05;

} // anonymous namespace

MincoTrajectory::MincoTrajectory(
    std::vector<double> durations,
    std::vector<CoefBlock> coeffs,
    std::vector<double> gears
)
    : durations_(std::move(durations)),
      coeffs_(std::move(coeffs)),
      gears_(std::move(gears)) {
    if (gears_.size() != durations_.size()) {
        gears_.assign(durations_.size(), 1.0); // 缺省全前进
    }
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
    // 段内 2D 平坦多项式求值，导数对**真实时间**（局部时间即真实时间）。τ 换算在 eval 层完成。
    const CoefBlock& coef = coeffs_[static_cast<size_t>(segment)];
    const double gear = gears_[static_cast<size_t>(segment)];
    const double T = durations_[static_cast<size_t>(segment)];

    Eigen::Vector2d pos = Eigen::Vector2d::Zero();
    Eigen::Vector2d vel_t = Eigen::Vector2d::Zero();
    Eigen::Vector2d acc_t = Eigen::Vector2d::Zero();

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

    // 真实时间导数 → τ 导数：τ = t/total_time ⇒ d/dτ = total_time · d/dt。
    const double s = total_time_;
    TrajSample out;
    out.p = pos;
    out.dp_dtau = vel_t * s;
    out.ddp_dtau = acc_t * s * s;

    // 位置曲线切线帧（几何量，与 gear 无关）。
    out.ds_dtau = out.dp_dtau.norm();
    out.phi = std::atan2(out.dp_dtau.y(), out.dp_dtau.x());
    out.sin_phi = std::sin(out.phi);
    out.cos_phi = std::cos(out.phi);
    const double cross = out.dp_dtau.x() * out.ddp_dtau.y() - out.dp_dtau.y() * out.ddp_dtau.x();
    const double ds = out.ds_dtau;
    out.kappa = ds > TANGENT_EPS ? cross / (ds * ds * ds) : 0.0;

    // ── 平坦朝向：θ_body = atan2(gear · 运动方向)。──
    const double speed_t = vel_t.norm();
    Eigen::Vector2d motion_dir = vel_t;
    if (speed_t <= TANGENT_EPS) {
        // v=0（尖点 / 端点）：运动方向取自段**内侧**一小步的速度——该处 speed>0，其方向
        // 即本段行进方向（gear 已编码前进 / 倒车），无需翻正。端点 a、j 可能同时退化，
        // 用有限步比 l'Hôpital 更稳。
        const double h = std::max(T * 1e-3, 1e-4);
        const double t_probe = local_t < 0.5 * T ? local_t + h : local_t - h;
        Eigen::Vector2d v_probe = Eigen::Vector2d::Zero();
        double tp = 1.0;
        for (int k = 1; k < NCOEF; ++k) {
            for (int dim = 0; dim < DIM; ++dim) v_probe(dim) += static_cast<double>(k) * coef(k, dim) * tp;
            tp *= t_probe;
        }
        motion_dir = v_probe;
        if (motion_dir.norm() <= TANGENT_EPS) motion_dir = acc_t; // 极端退化兜底
    }
    out.theta = std::atan2(gear * motion_dir.y(), gear * motion_dir.x());

    // ── 角速度 ω = θ̇ = (ẋÿ − ẏẍ)/‖ṗ‖²（与 gear 无关）。分母用物理速度尺度正则化，
    //    使尖点 / 端点（v→0）处 ω→0 而非发散；高速时误差可忽略（O(reg²/s²)）。──
    const double omega_t = (vel_t.x() * acc_t.y() - vel_t.y() * acc_t.x())
        / (speed_t * speed_t + OMEGA_SPEED_REG * OMEGA_SPEED_REG);
    out.dtheta_dtau = omega_t * s;
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
    for (size_t i = 0; i < gears_.size(); ++i) {
        if (gears_[i] != other.gears_[i]) return false;
    }
    return true;
}

} // namespace nav_executor
