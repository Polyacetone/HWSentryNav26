#include <queue>
#include <cmath>
#include <algorithm>
#include <rclcpp/logging.hpp>
#include <rclcpp/clock.hpp>
#include <path_planner/a_star_planner.hpp>
#include <path_planner/nav_map.hpp>

namespace path_planner {
constexpr int MIN_PATH_SIZE = 3;

struct Node {
    using Ptr = std::shared_ptr<Node>;
    Eigen::Vector2i coord;
    double g, h;
    Node::Ptr parent;
    double f() const { return g + h; }
};
}

namespace path_planner {
AStarPlanner::AStarPlanner(
    const double direction_weight,
    const double obstacle_weight,
    const int downsampled_waypoint_max_interval,
    const int feasible_threshold
):
    direction_weight_(direction_weight),
    obstacle_weight_(obstacle_weight),
    downsampled_waypoint_max_interval_(downsampled_waypoint_max_interval),
    feasible_threshold_(feasible_threshold) {}

double AStarPlanner::heuristic(const Eigen::Vector2i& s, const Eigen::Vector2i& t) const {
    return std::abs(s.x() - t.x()) + std::abs(s.y() - t.y());
}

bool AStarPlanner::is_valid(const CostMap& costmap, const Eigen::Vector2i& coord) const {
    return costmap.is_valid_coord(coord) && costmap.at(coord) <= feasible_threshold_;
}

bool AStarPlanner::is_line_safe(
    const CostMap& costmap,
    const Eigen::Vector2i& s,
    const Eigen::Vector2i& t
) const {
    int x1 = s.x(), x2 = t.x(), y1 = s.y(), y2 = t.y();
    const int dx = std::abs(x2 - x1), dy = std::abs(y2 - y1);
    const int sx = x1 < x2 ? 1 : -1, sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;
    // Bresenham网格直线遍历
    for (int _ = 0; _ < 100000; _++) {
        if (costmap.at({x1, y1}) > feasible_threshold_) return false;
        if (x1 == x2 && y1 == y2) return true;
        const int err2 = err * 2;
        if (err2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (err2 < dx) {
            err += dx;
            y1 += sy;
        }
    }
    RCLCPP_ERROR(rclcpp::get_logger("a_star_planner"), "Unexpected situation occurred in Bresenham iteration!");
    return false;
}

std::vector<Eigen::Vector2i> AStarPlanner::search_path(
    const CostMap& costmap,
    const DirectionMap& direction_map,
    const Eigen::Vector2i& start_grid,
    const Eigen::Vector2i& goal_grid
) const {
    // 保证路径长度大于等于MIN_PATH_SIZE
    if (std::max(std::abs(start_grid.x() - goal_grid.x()), std::abs(start_grid.y() - goal_grid.y())) < MIN_PATH_SIZE) {
        RCLCPP_ERROR(rclcpp::get_logger("a_star_planner"), "Start and goal too close!");
        return {};
    }
    if (!costmap.is_valid_coord(start_grid)) {
        RCLCPP_ERROR(rclcpp::get_logger("a_star_planner"), "Invalid start pose!");
        return {};
    }
    if (!costmap.is_valid_coord(goal_grid)) {
        RCLCPP_ERROR(rclcpp::get_logger("a_star_planner"), "Invalid goal pose!");
        return {};
    }
    const auto cmp = [](const Node::Ptr& n1, const Node::Ptr& n2) { return n1->f() > n2->f(); };
    std::priority_queue<Node::Ptr, std::vector<Node::Ptr>, decltype(cmp)> open_fwd(cmp), open_bwd(cmp);
    const int map_size = costmap.data.size();
    std::vector<Node::Ptr> all_fwd(map_size, nullptr), all_bwd(map_size, nullptr);
    std::vector<double> closed_fwd(map_size, 0.0), closed_bwd(map_size, 0.0);

    const auto get_key = [&](const Eigen::Vector2i& pt) { return pt.y() * costmap.width + pt.x(); };
    const int s_key = get_key(start_grid), g_key = get_key(goal_grid);
    const Node::Ptr start_node = std::make_shared<Node>(start_grid, 0, heuristic(start_grid, goal_grid), nullptr);
    const Node::Ptr goal_node  = std::make_shared<Node>(goal_grid, 0, heuristic(goal_grid, start_grid), nullptr);
    open_fwd.push(start_node);
    open_bwd.push(goal_node);
    all_fwd[s_key] = start_node;
    all_bwd[g_key] = goal_node;

    Node::Ptr meet_fwd = nullptr, meet_bwd = nullptr;
    double best_cost = std::numeric_limits<double>::max();

    while (!open_fwd.empty() && !open_bwd.empty()) {
        // 前向搜索
        if (!open_fwd.empty()) {
            const auto current = open_fwd.top(); open_fwd.pop();
            const int c_key = get_key(current->coord);
            if (closed_fwd[c_key]) continue;
            closed_fwd[c_key] = current->g;
            if (closed_bwd[c_key] > 0) {
                const double total_cost = current->g + closed_bwd[c_key];
                if (total_cost < best_cost) {
                    best_cost = total_cost;
                    meet_fwd = current;
                    meet_bwd = all_bwd[c_key];
                }
            }
            for (const auto& dir : directions_) {
                const Eigen::Vector2i next = current->coord + dir;
                const int n_key = get_key(next);
                if (!is_valid(costmap, next) || closed_fwd[n_key]) continue;

                double obstacle_cost = costmap.at(next) * obstacle_weight_;
                double step_cost = 0;
                Eigen::Vector2d step_dir = direction_map.at(next);
                if (step_dir != Eigen::Vector2d::Zero()) {
                    Eigen::Vector2d move_dir = dir.cast<double>().normalized();
                    step_cost = (1 - std::abs(move_dir.dot(step_dir))) * direction_weight_;
                }

                const double cost = current->g + dir.norm() + obstacle_cost + step_cost;
                if (all_fwd[n_key] && all_fwd[n_key]->g <= cost) continue;
                const Node::Ptr neighbor = std::make_shared<Node>(next, cost, heuristic(next, goal_grid), current);
                open_fwd.push(neighbor);
                all_fwd[n_key] = neighbor;
            }
        }

        // 反向搜索
        if (!open_bwd.empty()) {
            const auto current = open_bwd.top(); open_bwd.pop();
            const int c_key = get_key(current->coord);
            if (closed_bwd[c_key]) continue;
            closed_bwd[c_key] = current->g;
            if (closed_fwd[c_key] > 0) {
                double total_cost = current->g + closed_fwd[c_key];
                if (total_cost < best_cost) {
                    best_cost = total_cost;
                    meet_fwd = all_fwd[c_key];
                    meet_bwd = current;
                }
            }
            for (const auto& dir : directions_) {
                const Eigen::Vector2i next = current->coord + dir;
                const int n_key = get_key(next);
                if (!is_valid(costmap, next) || closed_bwd[n_key]) continue;
                
                double obstacle_cost = costmap.at(next) * obstacle_weight_;
                double step_cost = 0;
                Eigen::Vector2d step_dir = direction_map.at(next);
                if (step_dir != Eigen::Vector2d::Zero()) {
                    Eigen::Vector2d move_dir = dir.cast<double>().normalized();
                    step_cost = (1 - std::abs(move_dir.dot(step_dir))) * direction_weight_;
                }

                const double cost = current->g + dir.norm() + obstacle_cost + step_cost;
                if (all_bwd[n_key] && all_bwd[n_key]->g <= cost) continue;
                const Node::Ptr neighbor = std::make_shared<Node>(next, cost, heuristic(next, start_grid), current);
                open_bwd.push(neighbor);
                all_bwd[n_key] = neighbor;
            }
        }

        if (meet_fwd && meet_bwd) break;
    }
    if (!meet_bwd || !meet_fwd) {
        RCLCPP_WARN(rclcpp::get_logger("a_star_planner"), "No path found!");
        return {};
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

    // 路径降采样
    const int resolution = std::min<int>(
        downsampled_waypoint_max_interval_,
        std::max<int>((raw_path.size() - 1) / (MIN_PATH_SIZE - 1), 1) // 使路径点不少于MIN_PATH_SIZE个
    );
    std::vector<Eigen::Vector2i> downsampled_path;
    for (int i = 0; i < raw_path.size() - resolution; i += resolution) {
        downsampled_path.emplace_back(raw_path[i]);
    }
    downsampled_path.emplace_back(raw_path[raw_path.size() - 1]);

    return downsampled_path;
}
}