#include <nav_executor/path_executor/solver/search/mha_star.hpp>

#include <algorithm>
#include <chrono>
#include <limits>
#include <queue>
#include <unordered_map>

namespace nav_executor::search {

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

// 搜索期样条投影参数：粗搜不需要 FDDP 那么密的采样。
constexpr int PROJ_SAMPLES = 20;
constexpr double PROJ_WINDOW = 0.2;
constexpr double PROJ_LAZY = 0.1;

/// 搜索节点：闭表/开表共享的持久记录。
struct Node {
    SearchState state;
    double g = INF;
    double u = 0.0;             // 在样条上的缓存投影（供 edge_cost / 启发式复用，避免重复投影）
    int parent = -1;            // 父节点在 nodes_ 池中的索引
    int depth = 0;              // 从起点起的搜索步数
    MotionPrimitive incoming {}; // 到达本节点所用基元
    bool closed_anchor = false;  // 是否已在 anchor 队列展开（SMHA* 闭表语义）
};

/// 开表条目：按 key 值排序，携带节点池索引与创建时的 g（用于惰性删除）。
struct OpenEntry {
    double key;
    int node_index;
    double g_at_insert;
};

struct OpenCmp {
    bool operator()(const OpenEntry& a, const OpenEntry& b) const {
        return a.key > b.key; // 小顶堆
    }
};

using OpenQueue = std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenCmp>;

} // namespace

MHAStar::MHAStar(std::vector<std::unique_ptr<Heuristic>> heuristics, double w_anchor, double w_inadmissible)
    : heuristics_(std::move(heuristics)), w_anchor_(w_anchor), w_inadmissible_(w_inadmissible) {}

SearchResult MHAStar::search(
    const SearchState& start,
    const SearchEnvironment& env,
    const MotionModel& model,
    double budget_ms,
    int max_expansions
) const {
    SearchResult result;
    if (heuristics_.empty()) return result;

    const size_t num_queues = heuristics_.size();     // [0]=anchor，其余 inadmissible
    const size_t num_inadmissible = num_queues - 1;

    std::vector<Node> nodes;
    nodes.reserve(static_cast<size_t>(std::max(max_expansions, 64)) * 4);
    std::unordered_map<StateKey, int, StateKeyHash> visited;

    OpenQueue anchor_open;
    std::vector<OpenQueue> inadmissible_open(num_inadmissible);

    const auto key_for = [&](const Node& n, size_t heuristic_idx) {
        const double w = (heuristic_idx == 0) ? w_anchor_ : (w_anchor_ * w_inadmissible_);
        const int remaining = env.max_depth - n.depth;
        return n.g + w * heuristics_[heuristic_idx]->h(n.state, env, n.u, remaining);
    };

    // ── 插入起点 ──
    nodes.push_back(Node {.state = start, .g = 0.0, .u = env.start_u, .parent = -1});
    visited[env.make_key(start)] = 0;
    anchor_open.push({key_for(nodes[0], 0), 0, 0.0});
    for (size_t i = 0; i < num_inadmissible; ++i) {
        inadmissible_open[i].push({key_for(nodes[0], i + 1), 0, 0.0});
    }

    const auto t_start = std::chrono::steady_clock::now();
    const auto budget = std::chrono::duration<double, std::milli>(budget_ms);

    // incumbent：已找到的最优完整轨迹（展开满 max_depth 的节点）。
    int best_goal = -1;
    double best_goal_g = INF;
    int expansions = 0;

    // 从某个开表弹出下一个未过期、未闭合的节点索引，返回 -1 表示队列空。
    const auto pop_valid = [&](OpenQueue& q) -> int {
        while (!q.empty()) {
            const OpenEntry e = q.top();
            q.pop();
            const Node& n = nodes[static_cast<size_t>(e.node_index)];
            if (e.g_at_insert > n.g + 1e-9) continue; // 惰性删除：已被更优路径取代
            if (n.closed_anchor) continue;
            return e.node_index;
        }
        return -1;
    };

    const auto expand = [&](int node_idx) {
        nodes[static_cast<size_t>(node_idx)].closed_anchor = true;
        ++expansions;

        const Node parent = nodes[static_cast<size_t>(node_idx)];
        if (parent.depth >= env.max_depth) return; // 已是完整轨迹末端，不再扩展
        const int child_depth = parent.depth + 1;

        for (const auto& prim : model.primitives()) {
            const SearchState next = model.step(parent.state, prim);
            if (!env.is_feasible(next)) continue;
            if (!env.is_move_allowed(parent.state, next)) continue; // 台阶方向硬约束

            const double u_next = env.spline.project_extrapolated(
                Eigen::Vector2d(next.x, next.y), parent.u, PROJ_SAMPLES, PROJ_WINDOW, PROJ_LAZY
            );
            const double new_g = parent.g + env.edge_cost(next, u_next);
            const StateKey nk = env.make_key(next);

            auto it = visited.find(nk);
            int next_idx;
            if (it == visited.end()) {
                next_idx = static_cast<int>(nodes.size());
                nodes.push_back(Node {
                    .state = next, .g = new_g, .u = u_next,
                    .parent = node_idx, .depth = child_depth, .incoming = prim
                });
                visited.emplace(nk, next_idx);
            } else {
                next_idx = it->second;
                Node& existing = nodes[static_cast<size_t>(next_idx)];
                if (new_g + 1e-9 >= existing.g || existing.closed_anchor) continue;
                existing.g = new_g;
                existing.u = u_next;
                existing.parent = node_idx;
                existing.depth = child_depth;
                existing.incoming = prim;
            }

            Node& nn = nodes[static_cast<size_t>(next_idx)];
            anchor_open.push({key_for(nn, 0), next_idx, new_g});
            for (size_t i = 0; i < num_inadmissible; ++i) {
                inadmissible_open[i].push({key_for(nn, i + 1), next_idx, new_g});
            }

            // 完整轨迹（展开满时域）即目标候选，更新 incumbent。
            if (child_depth >= env.max_depth && new_g < best_goal_g) {
                best_goal_g = new_g;
                best_goal = next_idx;
            }
        }
    };

    // 取某开表的当前有效最小 key（惰性清理队顶过期/闭合项）。返回 INF 表示队列空。
    const auto clean_min_key = [&](OpenQueue& q) -> double {
        while (!q.empty()) {
            const OpenEntry e = q.top();
            const Node& n = nodes[static_cast<size_t>(e.node_index)];
            if (e.g_at_insert > n.g + 1e-9 || n.closed_anchor) {
                q.pop();
                continue;
            }
            return e.key;
        }
        return INF;
    };

    // ── SMHA* 主循环 + 分支定界早停 ──
    // anchor 提供 admissible 下界；当 incumbent 代价 <= 某队列最小 key 时，该队列无法再改进解，
    // anchor 满足该条件即证得（有界次优内）最优，提前返回，无需耗尽预算。
    while (true) {
        if (expansions >= max_expansions) break;                              // anytime 兜底
        if (std::chrono::steady_clock::now() - t_start > budget) break;       // anytime 兜底

        const double anchor_min_key = clean_min_key(anchor_open);
        if (anchor_min_key == INF) break;                 // 全部队列耗尽
        if (best_goal_g <= anchor_min_key) break;         // incumbent 已达最优下界，早停

        bool expanded = false;
        for (size_t i = 0; i < num_inadmissible; ++i) {
            const double min_key_i = clean_min_key(inadmissible_open[i]);
            if (min_key_i == INF) continue;
            if (min_key_i > w_inadmissible_ * anchor_min_key) continue; // 未满足 inadmissible 准入
            if (best_goal_g <= min_key_i) continue;                     // 该队列无法改进 incumbent

            const int idx = pop_valid(inadmissible_open[i]);
            if (idx >= 0) {
                expand(idx);
                expanded = true;
                break;
            }
        }

        if (!expanded) {
            // 无 inadmissible 队列可推进，展开 anchor（保证进展与完备性）。
            const int idx = pop_valid(anchor_open);
            if (idx < 0) break;
            expand(idx);
        }
    }

    // ── 回溯最优解（若无完整解，取 anchor key 最小的已展开节点作 best-effort 种子）──
    int trace = best_goal;
    if (trace < 0) {
        double best_key = INF;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (!nodes[static_cast<size_t>(i)].closed_anchor) continue;
            const double k = key_for(nodes[static_cast<size_t>(i)], 0);
            if (k < best_key) {
                best_key = k;
                trace = i;
            }
        }
    }
    if (trace < 0) return result;

    std::vector<int> chain;
    for (int i = trace; i >= 0; i = nodes[static_cast<size_t>(i)].parent) {
        chain.push_back(i);
    }
    std::reverse(chain.begin(), chain.end());

    result.states.reserve(chain.size());
    result.controls.reserve(chain.size() > 0 ? chain.size() - 1 : 0);
    for (size_t i = 0; i < chain.size(); ++i) {
        const Node& n = nodes[static_cast<size_t>(chain[i])];
        result.states.push_back(n.state);
        if (i > 0) result.controls.push_back(n.incoming);
    }
    result.cost = nodes[static_cast<size_t>(trace)].g;
    result.expansions = expansions;
    result.valid = result.states.size() >= 2;
    return result;
}

} // namespace nav_executor::search
