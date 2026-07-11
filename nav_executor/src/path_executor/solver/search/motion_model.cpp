#include <nav_executor/path_executor/solver/search/motion_model.hpp>

#include <algorithm>
#include <cmath>

namespace nav_executor::search {

namespace {
constexpr double TAU_MIN = 0.05;   // τ_v 下限 (s)，避免退化为瞬时响应
constexpr double TAU_MAX = 2.0;    // τ_v 上限 (s)
} // namespace

MotionModel::MotionModel(
    double dt,
    double tau_v,
    const CapabilityProfile& profile,
    const std::vector<double>& v_primitive_fracs,
    const std::vector<double>& omega_primitive_fracs
) : dt_(dt), profile_(profile) {
    const double tau = std::clamp(tau_v, TAU_MIN, TAU_MAX);
    alpha_v_ = 1.0 - std::exp(-dt_ / tau);

    const auto& cb = profile_.command_bounds;
    const auto& mc = profile_.motion_constraints;

    // 基元展开：v_cmd 由比例映射到 [vel_min, vel_max]（正比例→前进上限，负比例→后退下限），
    // ω 由比例映射到 [omega_min, omega_max]。剪除超侧向加速度约束的组合。
    primitives_.reserve(v_primitive_fracs.size() * omega_primitive_fracs.size());
    for (const double vf : v_primitive_fracs) {
        const double v_cmd = vf >= 0.0 ? vf * cb.vel_max : -vf * cb.vel_min;
        const double v_clamped = std::clamp(v_cmd, cb.vel_min, cb.vel_max);
        for (const double wf : omega_primitive_fracs) {
            const double omega = wf >= 0.0 ? wf * cb.omega_max : -wf * cb.omega_min;
            const double w_clamped = std::clamp(omega, cb.omega_min, cb.omega_max);
            if (std::abs(v_clamped * w_clamped) > mc.a_lat_max + 1e-6) continue;
            primitives_.push_back(MotionPrimitive {.v_cmd = v_clamped, .omega = w_clamped});
        }
    }
}

SearchState MotionModel::step(const SearchState& s, const MotionPrimitive& prim) const {
    const double v_next = s.v + (prim.v_cmd - s.v) * alpha_v_;

    const double theta_next = s.theta + prim.omega * dt_;
    const double v_mid = 0.5 * (s.v + v_next);
    const double theta_mid = 0.5 * (s.theta + theta_next);

    SearchState out;
    out.x = s.x + v_mid * std::cos(theta_mid) * dt_;
    out.y = s.y + v_mid * std::sin(theta_mid) * dt_;
    out.theta = theta_next;
    out.v = v_next;
    return out;
}

double MotionModel::derive_tau_v(const LPVDiscreteModel& m) {
    // 离散极点 ad11 = exp(-MPC_DT/τ) ⇒ τ = -MPC_DT / ln(ad11)。
    const double pole = std::clamp(m.ad11, 1e-3, 0.999);
    const double tau = -MPC_DT / std::log(pole);
    return std::clamp(tau, TAU_MIN, TAU_MAX);
}

} // namespace nav_executor::search
