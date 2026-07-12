#pragma once

/// @file heuristics.hpp
/// @brief MHA* 启发式集合。定时域最小代价形式下：
///        H0(admissible anchor) = 剩余时域进度下界；H1 = 叠加样条对齐；H2 = 进度解耦的台阶可达引导。
///        新启发式以同一接口 push 进队列列表，零重构。

#include <memory>
#include <vector>

#include <nav_executor/path_executor/solver/search/search_cost.hpp>
#include <nav_executor/path_executor/solver/search/search_types.hpp>

namespace nav_executor::search {

/// 启发式接口：估计从 state（样条投影为 u，距时域末端还剩 remaining_steps 步）到目标的剩余代价。
/// remaining_steps 让 anchor 能公平比较开表中不同深度的节点（浅节点 g 低但离满时域更远）。
class Heuristic {
public:
    virtual ~Heuristic() = default;
    [[nodiscard]] virtual double h(const SearchState& state, const SearchEnvironment& env, double u, int remaining_steps) const = 0;
    [[nodiscard]] virtual bool admissible() const = 0;
};

/// H0：admissible anchor —— 剩余时域的代价下界。
///     时间基线 remaining_steps·w_time + 进度项的紧下界（假设每步以 max_step_advance 最快推进，
///     剩余弧长线性递减时累加的进度代价）。两者皆为真实 edge_cost 之和的下界，故 admissible。
class ProgressHeuristic final : public Heuristic {
public:
    [[nodiscard]] double h(const SearchState& state, const SearchEnvironment& env, double u, int remaining_steps) const override;
    [[nodiscard]] bool admissible() const override { return true; }
};

/// H1：inadmissible —— anchor + 偏向贴近全局样条（离样条越远惩罚越大），
///     把搜索朝"沿样条推进"的同伦引导。
class SplineAlignHeuristic final : public Heuristic {
public:
    [[nodiscard]] double h(const SearchState& state, const SearchEnvironment& env, double u, int remaining_steps) const override;
    [[nodiscard]] bool admissible() const override { return false; }
};

/// H2：inadmissible —— 时间基线 + 台阶入口可达性欠缺量（复用 FDDP r_lo/r_hi 速度势函数），
///     刻意剔除进度项：后退虽降低进度，却增大入口前跑道→减小可达性欠缺，
///     故该队列会提前浮现"先退后进"的腾挪分支，避免其在预算内被进度锚队列饿死。
class StepReachabilityHeuristic final : public Heuristic {
public:
    [[nodiscard]] double h(const SearchState& state, const SearchEnvironment& env, double u, int remaining_steps) const override;
    [[nodiscard]] bool admissible() const override { return false; }
};

/// 构建启发式集合：[0]=anchor(H0 进度)，其后为 inadmissible([1]=H1 样条，[2]=H2 台阶可达)。
[[nodiscard]] std::vector<std::unique_ptr<Heuristic>> build_default_heuristics();

} // namespace nav_executor::search
