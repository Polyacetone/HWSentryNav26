#pragma once

/// @file search_cost.hpp
/// @brief 搜索环境绑定 + 边代价 + 可行性检查（Q4=B）。
///        边代价镜像 FDDP running cost 主项（避障 / Frenet 横向 / 台阶对齐 / 台阶可达），
///        剔除平滑与命令正则项，使搜索 basin 与 FDDP basin 定义一致。

#include <optional>

#include <nav_executor/common/spline_path.hpp>
#include <nav_executor/path_executor/solver/mpc_types.hpp>
#include <nav_executor/path_executor/solver/search/search_types.hpp>

namespace nav_executor::search {

/// 搜索期只读环境：栅格视图、样条参考、台阶模式、参数。
/// 全部为引用/值语义的轻量绑定，生命周期由调用方（solve_follow 栈帧）保证。
struct SearchEnvironment {
    const CostMapGridView& cost_grid;   // 避障用（当前帧融合代价）
    const GridInfo& cost_info;
    const DirectionMapGridView& dir_grid; // 台阶方向场
    const GridInfo& dir_info;
    const SplinePath& spline;           // 全局参考样条
    const FollowSearchParams& params;
    std::optional<ActiveStepMode> active_step_mode;
    double step_guide_acc = 1.2;        // 台阶入口可达势函数引导加速度（取自 FDDP terrain_limits）
    double max_step_advance = 0.1;      // 单步最大推进距离 v_max·dt (m)，供 admissible 启发式折算

    double start_u = 0.0;               // 起点在样条上的投影 u
    Eigen::Vector2d goal_xy = Eigen::Vector2d::Zero(); // 前瞻目标点 (map)
    double goal_u = 1.0;                // 前瞻目标点对应的 u
    int max_depth = 15;                 // 搜索时域步数上限（超过则不再扩展）

    /// 栅格坐标离散键。
    [[nodiscard]] StateKey make_key(const SearchState& s) const;

    /// 该状态是否可通行（在界内 + 代价低于阈值）。
    [[nodiscard]] bool is_feasible(const SearchState& s) const;

    /// 是否已到达前瞻目标。
    [[nodiscard]] bool is_goal(const SearchState& s) const;

    /// 到达 next 的边代价（>= 0）。当前各项均为到达状态的函数，故不依赖 parent。
    [[nodiscard]] double edge_cost(const SearchState& next) const;
};

} // namespace nav_executor::search
