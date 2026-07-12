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
    double step_guide_acc,
    double brake_decel,
    double brake_v_target,
    const DirectionMap* base_dir,
    const TerrainTraversalConstraints* terrain
) {
    SeedingResult out;
    if (!tau_v_ready_) return out;

    const MotionModel model(
        params_.dt, tau_v_, profile, params_.v_primitive_fracs, params_.omega_primitive_fracs
    );

    // 目标为定时域最小代价（展开满 horizon_steps 步）；无外部前瞻目标点，
    // 沿样条推进由 edge_cost 的进度项自发牵引，超越终点由制动项抑制。
    SearchEnvironment env {
        .cost_grid = cost_grid,
        .cost_info = cost_info,
        .dir_grid = dir_grid,
        .dir_info = dir_info,
        .spline = spline,
        .params = params_,
        .active_step_mode = active_step_mode,
        .step_guide_acc = step_guide_acc,
        .brake_decel = brake_decel,
        .brake_v_target = brake_v_target,
        .max_step_advance = std::max(model.v_max() * params_.dt, 1e-3),
        .base_dir = base_dir,
        .terrain = terrain,
        .start_u = start_u,
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
