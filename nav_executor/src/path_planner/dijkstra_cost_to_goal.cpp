#include <nav_executor/path_planner/dijkstra_cost_to_goal.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <queue>

namespace nav_executor {

namespace {
constexpr double SQRT2 = 1.4142135623730951;

struct DijkstraNode {
    double cost;
    int x, y;
    bool operator>(const DijkstraNode& o) const { return cost > o.cost; }
};
} // anonymous namespace

void DijkstraCostToGoal::build(const CostMap& cost_map, const Eigen::Vector2i& goal_grid, const Params& params) {
    width_ = cost_map.width;
    height_ = cost_map.height;
    resolution_ = cost_map.resolution;
    origin_x_ = cost_map.origin_x;
    origin_y_ = cost_map.origin_y;

    cost_.assign(static_cast<size_t>(width_) * static_cast<size_t>(height_), UNREACHABLE);

    if (!cost_map.is_valid_coord(goal_grid)) {
        return;
    }

    // 8 邻域步长（对角 √2），格代价系数将障碍值并入。
    static constexpr std::array<int, 8> DX = {1, -1, 0, 0, 1, 1, -1, -1};
    static constexpr std::array<int, 8> DY = {0, 0, 1, -1, 1, -1, 1, -1};
    static const std::array<double, 8> STEP = {1.0, 1.0, 1.0, 1.0, SQRT2, SQRT2, SQRT2, SQRT2};

    std::priority_queue<DijkstraNode, std::vector<DijkstraNode>, std::greater<>> open;
    const size_t g_idx = index(goal_grid.x(), goal_grid.y());
    cost_[g_idx] = 0.0;
    open.push({0.0, goal_grid.x(), goal_grid.y()});

    while (!open.empty()) {
        const DijkstraNode cur = open.top();
        open.pop();
        const size_t cur_idx = index(cur.x, cur.y);
        if (cur.cost > cost_[cur_idx]) continue; // 过期条目

        for (int i = 0; i < 8; ++i) {
            const int nx = cur.x + DX[static_cast<size_t>(i)];
            const int ny = cur.y + DY[static_cast<size_t>(i)];
            if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) continue;

            const Eigen::Vector2i ncoord(nx, ny);
            const double occ = static_cast<double>(cost_map.at(ncoord));
            if (occ >= static_cast<double>(params.feasible_threshold)) continue; // 不可通行

            const double step_cost = resolution_ * STEP[static_cast<size_t>(i)]
                * (1.0 + params.obstacle_weight * occ);
            const double new_cost = cur.cost + step_cost;
            const size_t n_idx = index(nx, ny);
            if (new_cost < cost_[n_idx]) {
                cost_[n_idx] = new_cost;
                open.push({new_cost, nx, ny});
            }
        }
    }
}

double DijkstraCostToGoal::at_grid(const Eigen::Vector2i& grid) const {
    if (grid.x() < 0 || grid.x() >= width_ || grid.y() < 0 || grid.y() >= height_) {
        return UNREACHABLE;
    }
    return cost_[index(grid.x(), grid.y())];
}

double DijkstraCostToGoal::at_map(const Eigen::Vector2d& map_pt) const {
    if (width_ <= 1 || height_ <= 1) return UNREACHABLE;

    const double gx = (map_pt.x() - origin_x_) / resolution_;
    const double gy = (map_pt.y() - origin_y_) / resolution_;
    if (gx < 0.0 || gy < 0.0 || gx >= static_cast<double>(width_ - 1) || gy >= static_cast<double>(height_ - 1)) {
        return UNREACHABLE;
    }

    const int x0 = static_cast<int>(std::floor(gx));
    const int y0 = static_cast<int>(std::floor(gy));
    const double tx = gx - static_cast<double>(x0);
    const double ty = gy - static_cast<double>(y0);

    const double c00 = cost_[index(x0, y0)];
    const double c10 = cost_[index(x0 + 1, y0)];
    const double c01 = cost_[index(x0, y0 + 1)];
    const double c11 = cost_[index(x0 + 1, y0 + 1)];
    // 任一角不可达 → 该点视为不可达（不跨障碍插值）。
    if (std::isinf(c00) || std::isinf(c10) || std::isinf(c01) || std::isinf(c11)) {
        return UNREACHABLE;
    }

    const double c0 = std::lerp(c00, c10, tx);
    const double c1 = std::lerp(c01, c11, tx);
    return std::lerp(c0, c1, ty);
}

} // namespace nav_executor
