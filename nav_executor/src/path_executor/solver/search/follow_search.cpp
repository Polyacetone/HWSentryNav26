#include <nav_executor/path_executor/solver/search/follow_search.hpp>

#include <algorithm>

namespace nav_executor::search {

FollowSearchSeeder::FollowSearchSeeder(const FollowSearchParams& params)
    : params_(params),
      solver_(build_default_heuristics(), params.w_anchor, params.w_inadmissible) {}

void FollowSearchSeeder::ensure_tau_v(const LPVDiscreteModel& nominal_model) {
    if (tau_v_ready_) return;
    tau_v_ = params_.tau_v > 0.0 ? params_.tau_v : MotionModel::derive_tau_v(nominal_model);
    tau_v_ready_ = true;
}

SeedingResult FollowSearchSeeder::run(
    const StateVec& x0,
    const SplinePath& spline,
    double start_u,
    const CostMapGridView& cost_grid,
    const GridInfo& cost_info,
    const DirectionMapGridView& dir_grid,
    const GridInfo& dir_info,
    const CapabilityProfile& profile,
    std::optional<ActiveStepMode> active_step_mode,
    double step_guide_acc
) {
    SeedingResult out;
    if (!tau_v_ready_) return out;

    const MotionModel model(
        params_.dt, tau_v_, profile, params_.v_primitive_fracs, params_.omega_primitive_fracs
    );

    // 前瞻目标：沿样条从 start_u 推进 lookahead_distance 弧长处的点。
    const double total_len = spline.arc_length(0.0, 1.0, 32);
    const double target_len = std::min(
        spline.arc_length(0.0, start_u, 32) + params_.lookahead_distance, total_len
    );
    // 二分求 target_len 对应的 u。
    double lo = start_u, hi = 1.0;
    for (int i = 0; i < 24; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (spline.arc_length(0.0, mid, 32) < target_len) lo = mid;
        else hi = mid;
    }
    const double goal_u = 0.5 * (lo + hi);
    const Eigen::Vector2d goal_xy = spline.position(goal_u);

    SearchEnvironment env {
        .cost_grid = cost_grid,
        .cost_info = cost_info,
        .dir_grid = dir_grid,
        .dir_info = dir_info,
        .spline = spline,
        .params = params_,
        .active_step_mode = active_step_mode,
        .step_guide_acc = step_guide_acc,
        .max_step_advance = std::max(model.v_max() * params_.dt, 1e-3),
        .start_u = start_u,
        .goal_xy = goal_xy,
        .goal_u = goal_u,
        .max_depth = params_.horizon_steps,
    };

    const SearchState start {
        .x = x0(ix::X), .y = x0(ix::Y), .theta = x0(ix::THETA), .v = x0(ix::V)
    };

    const SearchResult sr = solver_.search(
        start, env, model, params_.budget_ms, params_.max_expansions
    );
    if (!sr.valid) return out;

    out.seed = build_fddp_seed(sr, params_.dt, x0, spline);
    out.search_path.reserve(sr.states.size());
    for (const auto& s : sr.states) {
        out.search_path.emplace_back(s.x, s.y);
    }
    return out;
}

} // namespace nav_executor::search
