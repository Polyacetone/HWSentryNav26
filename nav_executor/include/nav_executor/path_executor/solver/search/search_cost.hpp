#pragma once

/// @file search_cost.hpp
/// @brief 搜索环境绑定 + 边代价 + 可行性检查（Q4=B）。
///        边代价镜像 FDDP running cost 主项（避障 / Frenet 横向 / 台阶对齐 / 台阶可达），
///        剔除平滑与命令正则项，使搜索 basin 与 FDDP basin 定义一致。

#include <optional>

#include <nav_executor/common/spline_path.hpp>
#include <nav_executor/path_executor/solver/mpc_types.hpp>
#include <nav_executor/path_executor/solver/search/search_types.hpp>
#include <nav_executor/path_planner/nav_map.hpp>

namespace nav_executor::search {

/// 搜索期只读环境：栅格视图、样条参考、台阶模式、参数。
/// 全部为引用/值语义的轻量绑定，生命周期由调用方（solve_follow 栈帧）保证。
///
/// 目标形式：定时域最小代价（goal = 展开到 max_depth）。不再设外部前瞻目标点，
/// 由 edge_cost 的进度项（沿样条剩余弧长）自发牵引"沿路径推进"，
/// 由制动项抑制超越终点，由 anchor 启发式提供剩余时域的 admissible 下界。
struct SearchEnvironment {
    const CostMapGridView& cost_grid;   // 避障用（当前帧融合代价）
    const GridInfo& cost_info;
    const DirectionMapGridView& dir_grid; // 台阶方向场（掩码后，用于 cost 项，与 FDDP 一致）
    const GridInfo& dir_info;
    const SplinePath& spline;           // 全局参考样条
    const FollowSearchParams& params;
    std::optional<ActiveStepMode> active_step_mode;
    double step_guide_acc = 1.2;        // 台阶入口可达势函数引导加速度（取自 FDDP terrain_limits）
    double brake_decel = 2.0;           // 终端制动减速度 a_brake（取自 FDDP terminal_weights）
    double brake_v_target = 0.1;        // 终端目标速度（取自 FDDP terminal_weights）
    double max_step_advance = 0.1;      // 单步最大推进距离 v_max·dt (m)，供 admissible 启发式折算

    // 台阶方向硬约束：单/双向可通行 + 夹角判据。使用未掩码的 base 方向场，
    // 否则掩码会沿走廊擦除方向矢量、恰在最需要处失效。为空指针时跳过（如单测）。
    const DirectionMap* base_dir = nullptr;
    const TerrainTraversalConstraints* terrain = nullptr;

    double start_u = 0.0;               // 起点在样条上的投影 u
    int max_depth = 15;                 // 搜索时域步数上限（= 目标深度）

    /// 栅格坐标离散键。
    [[nodiscard]] StateKey make_key(const SearchState& s) const;

    /// 该状态是否可通行（在界内 + 代价低于阈值）。
    [[nodiscard]] bool is_feasible(const SearchState& s) const;

    /// 从 from 到 to 的移动是否被台阶方向硬约束允许（逆向行驶按位移方向判定）。
    [[nodiscard]] bool is_move_allowed(const SearchState& from, const SearchState& to) const;

    /// 样条参数 u 处到终点的剩余弧长（进度项 / 制动项 / anchor 下界共用）。
    [[nodiscard]] double s_remaining(double u) const;

    /// 到达 next 的边代价（>= 0）。u 为 next 在样条上的缓存投影，避免重复投影。
    [[nodiscard]] double edge_cost(const SearchState& next, double u) const;
};

} // namespace nav_executor::search
