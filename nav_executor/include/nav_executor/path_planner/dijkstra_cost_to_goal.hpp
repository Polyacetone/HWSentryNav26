#pragma once

#include <limits>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/path_planner/nav_map.hpp>

namespace nav_executor {

// ── 2D Dijkstra cost-to-goal 场 ──
//
// 在 cost_map（含 blocked 层融合）上从 goal 反向扩散，产出稠密 cost-to-goal 数组。
// 用途 A：kinodynamic A* 弹节点时 O(1) 查启发 h（质点最短路下界）。
// 用途 B：开阔段几何 seed（沿场梯度下降回溯，替代旧几何 A*）。
//
// 代价 = 欧氏步长（8 邻域，对角 √2）+ 障碍代价（cost_map 值 · 权重）。
// 质点最短路：不含朝向/非完整，开阔区精确，窄口/台阶口偏差可接受（MINCO 精修）。
class DijkstraCostToGoal {
public:
    static constexpr double UNREACHABLE = std::numeric_limits<double>::infinity();

    struct Params {
        double obstacle_weight = 0.02; // 障碍代价系数（cost_map∈[0,255] 与步长同量级缩放）
        int feasible_threshold = 200;  // cost_map 值 ≥ 此视为不可通行（不扩散）
    };

    DijkstraCostToGoal() = default;

    // 从 goal_grid 反向扩散铺满可达区域。cost_map 决定分辨率与几何。
    void build(const CostMap& cost_map, const Eigen::Vector2i& goal_grid, const Params& params);

    [[nodiscard]] bool ready() const { return width_ > 0; }

    // O(1) 查 map 坐标处的 cost-to-goal（双线性插值；越界/不可达返回 UNREACHABLE）。
    [[nodiscard]] double at_map(const Eigen::Vector2d& map_pt) const;
    // O(1) 查栅格处 cost-to-goal（最近格）。
    [[nodiscard]] double at_grid(const Eigen::Vector2i& grid) const;

    [[nodiscard]] int width() const { return width_; }
    [[nodiscard]] int height() const { return height_; }

private:
    [[nodiscard]] size_t index(int x, int y) const {
        return static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
    }

    std::vector<double> cost_;   // cost-to-goal，UNREACHABLE 表示不可达
    int width_ = 0;
    int height_ = 0;
    double resolution_ = 1.0;
    double origin_x_ = 0.0;
    double origin_y_ = 0.0;
};

} // namespace nav_executor
