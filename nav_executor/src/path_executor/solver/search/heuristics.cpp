#include <nav_executor/path_executor/solver/search/heuristics.hpp>

#include <algorithm>
#include <cmath>

#include <nav_executor/path_executor/solver/mpc_utils.hpp>

namespace nav_executor::search {

namespace {
constexpr double REACH_EPS = 1e-6;

/// 剩余时域代价的 admissible 下界：时间基线 + 进度项紧下界。
/// 进度项按"每步以 max_step_advance 最快推进"折算：剩余弧长 S 逐步线性递减，
/// 累加各到达状态的 w_progress·s_remaining 下界。remaining_steps <= horizon，循环开销可忽略。
double progress_lower_bound(const SearchEnvironment& env, double u, int remaining_steps) {
    const int r = std::max(0, remaining_steps);
    double bound = static_cast<double>(r) * env.params.w_time;

    const double adv = std::max(env.max_step_advance, 1e-3);
    double s = env.s_remaining(u);
    for (int i = 0; i < r && s > 0.0; ++i) {
        s = std::max(0.0, s - adv);
        bound += env.params.w_progress * s;
    }
    return bound;
}
} // namespace

double ProgressHeuristic::h(const SearchState&, const SearchEnvironment& env, double u, int remaining_steps) const {
    // 与具体位姿无关：给定样条投影 u 与剩余步数即确定进度下界。
    return progress_lower_bound(env, u, remaining_steps);
}

double SplineAlignHeuristic::h(const SearchState& s, const SearchEnvironment& env, double u, int remaining_steps) const {
    const double anchor = progress_lower_bound(env, u, remaining_steps);

    // 离样条的横向偏离量级（用缓存投影 u，避免重复投影）。
    const auto se = env.spline.eval(std::clamp(u, 0.0, 1.0));
    const double ex = s.x - se.p.x();
    const double ey_w = s.y - se.p.y();
    const double ey = -ex * se.sin_r + ey_w * se.cos_r;

    return anchor + env.params.spline_bias * std::abs(ey);
}

double StepReachabilityHeuristic::h(const SearchState& s, const SearchEnvironment& env, double u, int remaining_steps) const {
    // 进度解耦：只保留时间基线，剔除进度项，使"后退换跑道"分支不被进度距离压到堆底。
    const double base = static_cast<double>(std::max(0, remaining_steps)) * env.params.w_time;

    if (!env.active_step_mode.has_value() || !env.active_step_mode->step_entry_u.has_value()) {
        return base;
    }

    const double entry_u = std::clamp(*env.active_step_mode->step_entry_u, 0.0, 1.0);
    const double cur_u = std::clamp(u, 0.0, 1.0);
    if (cur_u >= entry_u) {
        return base; // 已过入口，无可达性欠缺。
    }

    // 复用 FDDP 台阶速度势函数：以引导加速度推算到入口时的可达速度区间 [r_lo, r_hi]。
    const double d = env.spline.arc_length(cur_u, entry_u, 8);
    const double a_guide = std::max(env.step_guide_acc, REACH_EPS);
    const double v = s.v;
    const double r_lo = std::sqrt(std::max(0.0, std::max(0.0, v) * std::max(0.0, v) + 2.0 * a_guide * d));
    const double r_hi = std::sqrt(std::max(0.0, v * v - 2.0 * a_guide * d));

    const double shortfall = positive_part(env.active_step_mode->speed_min - r_lo)
        + positive_part(r_hi - env.active_step_mode->speed_max);

    return base + env.params.w_step_reach_heur * shortfall;
}

std::vector<std::unique_ptr<Heuristic>> build_default_heuristics() {
    std::vector<std::unique_ptr<Heuristic>> hs;
    hs.push_back(std::make_unique<ProgressHeuristic>());        // [0] anchor (H0)
    hs.push_back(std::make_unique<SplineAlignHeuristic>());      // [1] inadmissible (H1)
    hs.push_back(std::make_unique<StepReachabilityHeuristic>()); // [2] inadmissible (H2)
    return hs;
}

} // namespace nav_executor::search
