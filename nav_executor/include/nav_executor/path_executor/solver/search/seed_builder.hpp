#pragma once

/// @file seed_builder.hpp
/// @brief 将 MHA* 搜索结果（dt 间隔的 x,y,θ,v）插值展开为 FDDP warm-start
///        （MPC_DT × MPC_HORIZON 的 xs/us）。FDDP 为 feasibility-driven，
///        种子无需与 dynamics 自洽，gap 会被逐步消解。

#include <array>
#include <optional>

#include <nav_executor/common/spline_path.hpp>
#include <nav_executor/path_executor/solver/mpc_types.hpp>
#include <nav_executor/path_executor/solver/search/search_types.hpp>

namespace nav_executor::search {

struct FddpSeed {
    std::array<StateVec, MPC_HORIZON + 1> xs {};
    std::array<ControlVec, MPC_HORIZON> us {};
    bool valid = false;
};

/// 由搜索结果构建 FDDP 种子。
/// @param result   MHA* 输出（dt 间隔状态 + 基元序列）
/// @param search_dt 搜索步长
/// @param x0       FDDP 初始状态（提供 XH/ENERGY 等搜索不建模的分量）
/// @param spline   全局参考样条（用于逐点重投影 PATH_U）
[[nodiscard]] FddpSeed build_fddp_seed(
    const SearchResult& result,
    double search_dt,
    const StateVec& x0,
    const SplinePath& spline
);

} // namespace nav_executor::search
