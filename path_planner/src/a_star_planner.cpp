#include <queue>
#include <cmath>
#include <algorithm>
#include <path_planner/a_star_planner.hpp>
#include <path_planner/nav_map.hpp>

namespace path_planner {
constexpr int MIN_PATH_SIZE = 3;

struct Node {
    using Ptr = std::shared_ptr<Node>;
    Eigen::Vector2i coord;
    double g, h;
    Node* parent;
    double f() const { return g + h; }
};
}

namespace path_planner {
AStarPlanner::AStarPlanner(
    const double step_alignment_weight,
    const double obstacle_weight,
    const double step_proximity_weight,
    const double step_mode_dot_threshold,
    const int downsampled_waypoint_max_interval,
    const int feasible_threshold
):
    step_alignment_weight_(step_alignment_weight),
    obstacle_weight_(obstacle_weight),
    step_proximity_weight_(step_proximity_weight),
    step_mode_dot_threshold_(step_mode_dot_threshold),
    downsampled_waypoint_max_interval_(downsampled_waypoint_max_interval),
    feasible_threshold_(feasible_threshold) {}

double AStarPlanner::heuristic(const Eigen::Vector2i& s, const Eigen::Vector2i& t) const {
    return std::abs(s.x() - t.x()) + std::abs(s.y() - t.y());
}

bool AStarPlanner::is_valid(const CostMap& costmap, const Eigen::Vector2i& coord) const {
    return costmap.is_valid_coord(coord) && costmap.at(coord) <= feasible_threshold_;
}

std::expected<std::vector<Eigen::Vector2i>, std::string> AStarPlanner::search_path(
    const CostMap& costmap,
    const DirectionMap& direction_map,
    const Eigen::Vector2i& start_grid,
    const Eigen::Vector2i& goal_grid
) const {
    // 保证路径长度大于等于MIN_PATH_SIZE
    if (std::max(std::abs(start_grid.x() - goal_grid.x()), std::abs(start_grid.y() - goal_grid.y())) < MIN_PATH_SIZE) {
        return std::unexpected("Start and goal too close");
    }
    if (!costmap.is_valid_coord(start_grid)) {
        return std::unexpected("Invalid start point");
    }
    if (!costmap.is_valid_coord(goal_grid)) {
        return std::unexpected("Invalid goal point");
    }
    const auto cmp = [](const Node::Ptr& n1, const Node::Ptr& n2) { return n1->f() > n2->f(); };
    std::priority_queue<Node::Ptr, std::vector<Node::Ptr>, decltype(cmp)> open_fwd(cmp), open_bwd(cmp);
    const size_t map_size = costmap.data.size();
    std::vector<Node::Ptr> all_fwd(map_size, nullptr), all_bwd(map_size, nullptr);
    std::vector<double> closed_fwd(map_size, 0.0), closed_bwd(map_size, 0.0);

    const auto get_key = [&](const Eigen::Vector2i& pt) {
        return static_cast<size_t>(pt.y()) * static_cast<size_t>(costmap.width) + static_cast<size_t>(pt.x());
    };
    const size_t s_key = get_key(start_grid), g_key = get_key(goal_grid);
    const Node::Ptr start_node = std::make_shared<Node>(start_grid, 0, heuristic(start_grid, goal_grid), nullptr);
    const Node::Ptr goal_node  = std::make_shared<Node>(goal_grid, 0, heuristic(goal_grid, start_grid), nullptr);
    open_fwd.push(start_node);
    open_bwd.push(goal_node);
    all_fwd[s_key] = start_node;
    all_bwd[g_key] = goal_node;

    // 计算给定方向栅格、方向地图的单步台阶对齐代价与台阶接近代价
    const auto step_costs = [&](const Eigen::Vector2i& coord, const Eigen::Vector2d& move_dir) {
        const Eigen::Vector2d step_dir = direction_map.at(coord);
        double alignment = 0.0;
        double proximity = 0.0;
        if (step_dir != Eigen::Vector2d::Zero()) {
            alignment = (1.0 - std::abs(move_dir.dot(step_dir))) * step_alignment_weight_;
        }
        proximity = step_dir.norm() * step_proximity_weight_;
        return std::pair{alignment, proximity};
    };

    Node* meet_fwd = nullptr;
    Node* meet_bwd = nullptr;
    double best_cost = std::numeric_limits<double>::max();

    while (!open_fwd.empty() && !open_bwd.empty()) {
        // 提前终止条件：两个方向堆顶的 f 值均超过 best_cost/2 时停止
        const double top_fwd = open_fwd.top()->f();
        const double top_bwd = open_bwd.top()->f();
        if (best_cost < std::numeric_limits<double>::max() && top_fwd > best_cost * 0.5 && top_bwd > best_cost * 0.5) {
            break;
        }

        // 前向搜索
        if (!open_fwd.empty()) {
            const auto current = open_fwd.top(); open_fwd.pop();
            const size_t c_key = get_key(current->coord);
            if (closed_fwd[c_key] != 0) continue;
            closed_fwd[c_key] = current->g;
            if (closed_bwd[c_key] > 0) {
                const double total_cost = current->g + closed_bwd[c_key];
                if (total_cost < best_cost) {
                    best_cost = total_cost;
                    meet_fwd = current.get();
                    meet_bwd = all_bwd[c_key].get();
                }
            }
            for (const auto& dir : directions_) {
                const Eigen::Vector2i next = current->coord + dir;
                const size_t n_key = get_key(next);
                if (!is_valid(costmap, next) || closed_fwd[n_key] != 0) continue;

                const Eigen::Vector2d move_dir = dir.cast<double>().normalized();
                if (direction_map.is_direction_prohibited(next, move_dir, step_mode_dot_threshold_)) continue;

                const double obstacle_cost = costmap.at(next) * obstacle_weight_;
                const auto [step_alignment, step_proximity] = step_costs(next, move_dir);

                const double cost = current->g + dir.norm() + obstacle_cost + step_alignment + step_proximity;
                if (all_fwd[n_key] && all_fwd[n_key]->g <= cost) continue;
                const Node::Ptr neighbor = std::make_shared<Node>(next, cost, heuristic(next, goal_grid), current.get());
                open_fwd.push(neighbor);
                all_fwd[n_key] = neighbor;
            }
        }

        // 反向搜索
        if (!open_bwd.empty()) {
            const auto current = open_bwd.top(); open_bwd.pop();
            const size_t c_key = get_key(current->coord);
            if (closed_bwd[c_key] != 0) continue;
            closed_bwd[c_key] = current->g;
            if (closed_fwd[c_key] > 0) {
                double total_cost = current->g + closed_fwd[c_key];
                if (total_cost < best_cost) {
                    best_cost = total_cost;
                    meet_fwd = all_fwd[c_key].get();
                    meet_bwd = current.get();
                }
            }
            for (const auto& dir : directions_) {
                const Eigen::Vector2i next = current->coord + dir;
                const size_t n_key = get_key(next);
                if (!is_valid(costmap, next) || closed_bwd[n_key] != 0) continue;

                // 反向搜索: 路径真实方向为 next → current
                const Eigen::Vector2d travel_dir = -dir.cast<double>().normalized();
                if (direction_map.is_direction_prohibited(current->coord, travel_dir, step_mode_dot_threshold_)) continue;
                // 注: step_costs 内部使用 |dot|, travel_dir 与 -travel_dir 结果相同;
                // 进入当前栅格的方向为 travel_dir (next→current), 与正向搜索语义一致
                const double obstacle_cost = costmap.at(next) * obstacle_weight_;
                const auto [step_alignment, step_proximity] = step_costs(next, -travel_dir);

                const double cost = current->g + dir.norm() + obstacle_cost + step_alignment + step_proximity;
                if (all_bwd[n_key] && all_bwd[n_key]->g <= cost) continue;
                const Node::Ptr neighbor = std::make_shared<Node>(next, cost, heuristic(next, start_grid), current.get());
                open_bwd.push(neighbor);
                all_bwd[n_key] = neighbor;
            }
        }

        if (meet_fwd && meet_bwd) break;
    }
    if (!meet_fwd || !meet_bwd) {
        return std::unexpected("No feasible path found");
    }

    // 根据相遇节点反向构建路径
    std::vector<Eigen::Vector2i> raw_path;
    while (meet_fwd) {
        raw_path.push_back(meet_fwd->coord);
        meet_fwd = meet_fwd->parent;
    }
    std::reverse(raw_path.begin(), raw_path.end());
    if (meet_bwd) {
        meet_bwd = meet_bwd->parent;
        while (meet_bwd) {
            raw_path.push_back(meet_bwd->coord);
            meet_bwd = meet_bwd->parent;
        }
    }

    // 注意: raw_path 是 grid 坐标, 后续 path_planner_node 中
    // 乘 resolution_ 转为世界坐标时 grid 坐标下溢会导致负的世界坐标。
    // 若在此处添加 grid 边界保护, 请同时验证 world 转换无下溢。
    // 路径降采样
    const int resolution = std::min<int>(
        downsampled_waypoint_max_interval_,
        std::max<int>(static_cast<int>((raw_path.size() - 1) / (MIN_PATH_SIZE - 1)), 1) // 使路径点不少于MIN_PATH_SIZE个
    );
    std::vector<Eigen::Vector2i> downsampled_path;
    for (size_t i = 0; i < raw_path.size() - static_cast<size_t>(resolution); i += static_cast<size_t>(resolution)) {
        downsampled_path.emplace_back(raw_path[i]);
    }
    downsampled_path.emplace_back(raw_path[raw_path.size() - 1]);

    return downsampled_path;
}
}