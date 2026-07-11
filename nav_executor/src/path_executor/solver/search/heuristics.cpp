#include <nav_executor/path_executor/solver/search/heuristics.hpp>

#include <algorithm>
#include <cmath>

#include <nav_executor/path_executor/solver/mpc_utils.hpp>

namespace nav_executor::search {

namespace {
constexpr double REACH_EPS = 1e-6;
// 搜索期样条投影参数：与 search_cost / SplineAlignHeuristic 保持一致。
constexpr int PROJ_SAMPLES = 20;
constexpr double PROJ_WINDOW = 0.2;
constexpr double PROJ_LAZY = 0.1;

/// admissible 到达目标的最小步数代价：每步至多推进 v_max·dt，每步至少花 w_time。
double anchor_cost(const SearchState& s, const SearchEnvironment& env) {
    const double dist = (Eigen::Vector2d(s.x, s.y) - env.goal_xy).norm();
    const double max_step_advance = std::max(env.max_step_advance, 1e-3);
    return (dist / max_step_advance) * env.params.w_time;
}
} // namespace

double EuclideanHeuristic::h(const SearchState& s, const SearchEnvironment& env) const {
    return anchor_cost(s, env);
}

double SplineAlignHeuristic::h(const SearchState& s, const SearchEnvironment& env) const {
    const double anchor = anchor_cost(s, env);

    // 离样条的横向偏离：投影后取横向误差量级。
    const double u = env.spline.project_extrapolated(
        Eigen::Vector2d(s.x, s.y), env.start_u, 20, 0.2, 0.1
    );
    const auto se = env.spline.eval(u);
    const double ex = s.x - se.p.x();
    const double ey_w = s.y - se.p.y();
    const double ey = -ex * se.sin_r + ey_w * se.cos_r;

    return anchor + env.params.spline_bias * std::abs(ey);
}

double StepReachabilityHeuristic::h(const SearchState& s, const SearchEnvironment& env) const {
    const double anchor = anchor_cost(s, env);

    // 无 active step 入口 → 退化为纯 anchor（与 H0 同底）。
    if (!env.active_step_mode.has_value() || !env.active_step_mode->step_entry_u.has_value()) {
        return anchor;
    }

    const double entry_u = std::clamp(*env.active_step_mode->step_entry_u, 0.0, 1.0);
    const double u = env.spline.project_extrapolated(
        Eigen::Vector2d(s.x, s.y), env.start_u, PROJ_SAMPLES, PROJ_WINDOW, PROJ_LAZY
    );
    const double cur_u = std::clamp(u, 0.0, 1.0);
    if (cur_u >= entry_u) {
        return anchor; // 已过入口，无可达性欠缺。
    }

    // 复用 FDDP 台阶速度势函数：以引导加速度推算到入口时的可达速度区间 [r_lo, r_hi]。
    const double d = env.spline.arc_length(cur_u, entry_u, 8);
    const double a_guide = std::max(env.step_guide_acc, REACH_EPS);
    const double v = s.v;
    const double r_lo = std::sqrt(std::max(0.0, std::max(0.0, v) * std::max(0.0, v) + 2.0 * a_guide * d));
    const double r_hi = std::sqrt(std::max(0.0, v * v - 2.0 * a_guide * d));

    const double shortfall = positive_part(env.active_step_mode->speed_min - r_lo)
        + positive_part(r_hi - env.active_step_mode->speed_max);

    return anchor + env.params.w_step_reach_heur * shortfall;
}

std::vector<std::unique_ptr<Heuristic>> build_default_heuristics() {
    std::vector<std::unique_ptr<Heuristic>> hs;
    hs.push_back(std::make_unique<EuclideanHeuristic>());       // [0] anchor (H0)
    hs.push_back(std::make_unique<SplineAlignHeuristic>());      // [1] inadmissible (H1)
    hs.push_back(std::make_unique<StepReachabilityHeuristic>()); // [2] inadmissible (H2)
    return hs;
}

} // namespace nav_executor::search
