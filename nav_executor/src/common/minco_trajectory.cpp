#include <nav_executor/common/minco_trajectory.hpp>

#include <algorithm>
#include <array>
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

double MincoTrajectory::arc_length_between(const double tau0, const double tau1) const {
    return std::abs(arc_length_at_tau(tau1) - arc_length_at_tau(tau0));
}

int MincoTrajectory::segment_index_at_arc_length(const double arc_length) const {
    if (durations_.empty()) return 0;
    const double tau = tau_at_arc_length(arc_length);
    return locate_time(tau * total_time_).segment;
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
    out.gear = gear;

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

TrajSample MincoTrajectory::eval_arc_length(const double arc_length) const {
    return eval(tau_at_arc_length(arc_length));
}

double MincoTrajectory::longitudinal_velocity(const TrajSample& s) const {
    return s.gear * nominal_path_speed(s);
}

double MincoTrajectory::angular_velocity(const TrajSample& s) const {
    if (total_time_ <= MIN_SEGMENT_DURATION) return 0.0;
    return s.dtheta_dtau / total_time_;
}

double MincoTrajectory::nominal_path_speed(const TrajSample& s) const {
    if (total_time_ <= MIN_SEGMENT_DURATION) return 0.0;
    return s.ds_dtau / total_time_;
}

double MincoTrajectory::heading_rate_per_arc_length(const TrajSample& s) const {
    const double path_speed = nominal_path_speed(s);
    return path_speed > TANGENT_EPS ? angular_velocity(s) / path_speed : 0.0;
}

double MincoTrajectory::segment_gear(const int segment_index) const {
    if (gears_.empty()) return 1.0;
    const int clamped = std::clamp(segment_index, 0, segment_count() - 1);
    return gears_[static_cast<size_t>(clamped)];
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
    for (size_t i = 0; i < gears_.size(); ++i) {
        if (gears_[i] != other.gears_[i]) return false;
    }
    return true;
}

} // namespace nav_executor
