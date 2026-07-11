#pragma once

/// @file heuristics.hpp
/// @brief MHA* 启发式集合（Q8）。原型-1 挂 H0(admissible anchor) + H1(样条对齐)。
///        H2(台阶可达引导) / H3(时空避让) 后续以同一接口 push 进队列列表，零重构。

#include <memory>
#include <vector>

#include <nav_executor/path_executor/solver/search/search_cost.hpp>
#include <nav_executor/path_executor/solver/search/search_types.hpp>

namespace nav_executor::search {

/// 启发式接口：估计从 state 到目标的剩余代价。
class Heuristic {
public:
    virtual ~Heuristic() = default;
    [[nodiscard]] virtual double h(const SearchState& state, const SearchEnvironment& env) const = 0;
    [[nodiscard]] virtual bool admissible() const = 0;
};

/// H0：admissible anchor —— 到前瞻目标的欧氏距离（保证有界次优 + 完备）。
class EuclideanHeuristic final : public Heuristic {
public:
    [[nodiscard]] double h(const SearchState& state, const SearchEnvironment& env) const override;
    [[nodiscard]] bool admissible() const override { return true; }
};

/// H1：inadmissible —— anchor + 偏向贴近全局样条（离样条越远惩罚越大），
///     把搜索朝"沿样条推进"的同伦引导。
class SplineAlignHeuristic final : public Heuristic {
public:
    [[nodiscard]] double h(const SearchState& state, const SearchEnvironment& env) const override;
    [[nodiscard]] bool admissible() const override { return false; }
};

/// H2：inadmissible —— anchor + 台阶入口可达性欠缺量（复用 FDDP r_lo/r_hi 速度势函数）。
///     事前引导搜索优先扩展"能达成可行台阶入口速度"的节点（含提前调速/后退腾挪），
///     提速并增强"先退后进"策略的可靠性。
class StepReachabilityHeuristic final : public Heuristic {
public:
    [[nodiscard]] double h(const SearchState& state, const SearchEnvironment& env) const override;
    [[nodiscard]] bool admissible() const override { return false; }
};

/// 构建启发式集合：[0]=anchor(H0)，其后为 inadmissible([1]=H1 样条，[2]=H2 台阶可达)。
[[nodiscard]] std::vector<std::unique_ptr<Heuristic>> build_default_heuristics();

} // namespace nav_executor::search
