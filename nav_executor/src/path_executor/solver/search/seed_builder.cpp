#include <nav_executor/path_executor/solver/search/seed_builder.hpp>

#include <algorithm>
#include <cmath>

#include <nav_executor/path_executor/solver/mpc_utils.hpp>

namespace nav_executor::search {

namespace {

/// 在 dt 间隔的搜索状态序列上，按连续时间 t 做线性插值。
SearchState sample_at(const std::vector<SearchState>& states, double search_dt, double t) {
    if (states.empty()) return {};
    const double fidx = t / search_dt;
    const size_t i0 = static_cast<size_t>(std::floor(fidx));
    if (i0 + 1 >= states.size()) return states.back();
    const double frac = fidx - static_cast<double>(i0);
    const SearchState& a = states[i0];
    const SearchState& b = states[i0 + 1];
    SearchState out;
    out.x = a.x + (b.x - a.x) * frac;
    out.y = a.y + (b.y - a.y) * frac;
    // 航向按最短弧插值，避免 ±π 跨越跳变。
    out.theta = a.theta + wrap_pi(b.theta - a.theta) * frac;
    out.v = a.v + (b.v - a.v) * frac;
    return out;
}

/// 对应基元（v_cmd, ω）：t 落在第几段搜索区间。
MotionPrimitive control_at(const std::vector<MotionPrimitive>& controls, double search_dt, double t) {
    if (controls.empty()) return {};
    const size_t idx = std::min(controls.size() - 1, static_cast<size_t>(std::floor(t / search_dt)));
    return controls[idx];
}

} // namespace

FddpSeed build_fddp_seed(
    const SearchResult& result,
    double search_dt,
    const StateVec& x0,
    const SplinePath& spline
) {
    FddpSeed seed;
    if (!result.valid || result.states.size() < 2) return seed;

    const double search_horizon = static_cast<double>(result.states.size() - 1) * search_dt;

    double u_hint = x0(ix::PATH_U);

    // 逐 MPC 步插值填充 xs；us 由基元展开。
    for (int k = 0; k <= MPC_HORIZON; ++k) {
        const double t = std::min(static_cast<double>(k) * MPC_DT, search_horizon);
        const SearchState s = sample_at(result.states, search_dt, t);

        StateVec& x = seed.xs[static_cast<size_t>(k)];
        x = x0; // 继承 XH / ENERGY 等搜索不建模的分量
        x(ix::X) = s.x;
        x(ix::Y) = s.y;
        x(ix::THETA) = s.theta;
        x(ix::V) = s.v;

        // W 由相邻 θ 差分近似；PATH_U 逐点重投影。
        u_hint = spline.project_extrapolated(Eigen::Vector2d(s.x, s.y), u_hint, 20, 0.2, 0.1);
        x(ix::PATH_U) = u_hint;

        if (k < MPC_HORIZON) {
            const MotionPrimitive prim = control_at(result.controls, search_dt, t);
            seed.us[static_cast<size_t>(k)] = ControlVec(prim.v_cmd, prim.omega);
            x(ix::DV) = prim.v_cmd;
            x(ix::DW) = prim.omega;
        }
    }

    // W（角速度）由航向差分回填。
    for (int k = 0; k < MPC_HORIZON; ++k) {
        const double dtheta = wrap_pi(seed.xs[static_cast<size_t>(k) + 1](ix::THETA) - seed.xs[static_cast<size_t>(k)](ix::THETA));
        seed.xs[static_cast<size_t>(k)](ix::W) = dtheta / MPC_DT;
    }
    seed.xs[MPC_HORIZON](ix::W) = seed.xs[MPC_HORIZON - 1](ix::W);

    seed.valid = true;
    return seed;
}

} // namespace nav_executor::search
