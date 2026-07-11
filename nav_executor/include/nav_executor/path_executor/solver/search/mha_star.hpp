#pragma once

/// @file mha_star.hpp
/// @brief Anytime Shared Multi-Heuristic A*（SMHA*）通用骨架（Q10）。
///        共享一份 g 值 + 一个 admissible anchor 队列 + N 个 inadmissible 队列。
///        anytime：预算耗尽或到达目标即返回 best-so-far。
///        原型-1 挂 H0+H1，后续 push 更多启发式即可，零重构。

#include <memory>
#include <vector>

#include <nav_executor/path_executor/solver/search/heuristics.hpp>
#include <nav_executor/path_executor/solver/search/motion_model.hpp>
#include <nav_executor/path_executor/solver/search/search_cost.hpp>
#include <nav_executor/path_executor/solver/search/search_types.hpp>

namespace nav_executor::search {

class MHAStar {
public:
    MHAStar(std::vector<std::unique_ptr<Heuristic>> heuristics, double w_anchor, double w_inadmissible);

    /// 从 start 搜索到 env 的前瞻目标。budget_ms 到或 max_expansions 到即返回当前最优。
    [[nodiscard]] SearchResult search(
        const SearchState& start,
        const SearchEnvironment& env,
        const MotionModel& model,
        double budget_ms,
        int max_expansions
    ) const;

private:
    std::vector<std::unique_ptr<Heuristic>> heuristics_;
    double w_anchor_;        // w1
    double w_inadmissible_;  // w2
};

} // namespace nav_executor::search
