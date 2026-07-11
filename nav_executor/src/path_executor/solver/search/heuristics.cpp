#include <nav_executor/path_executor/solver/search/heuristics.hpp>

#include <cmath>

namespace nav_executor::search {

namespace {
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

std::vector<std::unique_ptr<Heuristic>> build_default_heuristics() {
    std::vector<std::unique_ptr<Heuristic>> hs;
    hs.push_back(std::make_unique<EuclideanHeuristic>());   // [0] anchor
    hs.push_back(std::make_unique<SplineAlignHeuristic>());  // [1] inadmissible
    return hs;
}

} // namespace nav_executor::search
