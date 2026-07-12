#pragma once

/// @file follow_search.hpp
/// @brief FDDP 前的 MHA* 播种器门面：绑定环境、跑搜索、产出 FDDP 种子。
///        由 MPCSolver 持有，在 solve_follow 中作为 warm-shift 之外的第二播种源。

#include <memory>
#include <optional>

#include <nav_executor/common/spline_path.hpp>
#include <nav_executor/path_executor/solver/mpc_types.hpp>
#include <nav_executor/path_executor/solver/search/heuristics.hpp>
#include <nav_executor/path_executor/solver/search/mha_star.hpp>
#include <nav_executor/path_executor/solver/search/seed_builder.hpp>

namespace nav_executor::search {

/// 搜索播种结果：FDDP 种子 + 供调试可视化的 map 坐标粗路径。
struct SeedingResult {
    FddpSeed seed;
    std::vector<Eigen::Vector2d> search_path;
};

class FollowSearchSeeder {
public:
    explicit FollowSearchSeeder(const FollowSearchParams& params);

    /// 惰性初始化速度时间常数（首次由 LPV 标称模型推导；参数已配置则直接用）。
    void ensure_tau_v(const LPVDiscreteModel& nominal_model);

    [[nodiscard]] bool enabled() const { return params_.enable; }

    /// 跑一次搜索并构建 FDDP 种子。失败（不可行/超预算无解）时 seed.valid=false。
    /// base_dir / terrain 供台阶方向硬约束（可为 nullptr，此时跳过该约束）；
    /// 使用未掩码的 base 方向场，避免走廊掩码擦除方向矢量导致约束失效。
    [[nodiscard]] SeedingResult run(
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
    );

private:
    FollowSearchParams params_;
    MHAStar solver_;
    double tau_v_ = -1.0;
    bool tau_v_ready_ = false;
};

} // namespace nav_executor::search
