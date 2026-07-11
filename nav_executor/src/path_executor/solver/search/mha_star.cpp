#include <nav_executor/path_executor/solver/search/mha_star.hpp>

#include <algorithm>
#include <chrono>
#include <limits>
#include <queue>
#include <unordered_map>

namespace nav_executor::search {

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

/// 搜索节点：闭表/开表共享的持久记录。
struct Node {
    SearchState state;
    double g = INF;
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
        return n.g + w * heuristics_[heuristic_idx]->h(n.state, env);
    };

    // ── 插入起点 ──
    nodes.push_back(Node {.state = start, .g = 0.0, .parent = -1});
    visited[env.make_key(start)] = 0;
    anchor_open.push({key_for(nodes[0], 0), 0, 0.0});
    for (size_t i = 0; i < num_inadmissible; ++i) {
        inadmissible_open[i].push({key_for(nodes[0], i + 1), 0, 0.0});
    }

    const auto t_start = std::chrono::steady_clock::now();
    const auto budget = std::chrono::duration<double, std::milli>(budget_ms);

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

        const SearchState parent = nodes[static_cast<size_t>(node_idx)].state;
        const double parent_g = nodes[static_cast<size_t>(node_idx)].g;
        const int parent_depth = nodes[static_cast<size_t>(node_idx)].depth;
        if (parent_depth >= env.max_depth) return; // 达到时域上限，不再扩展

        for (const auto& prim : model.primitives()) {
            const SearchState next = model.step(parent, prim);
            if (!env.is_feasible(next)) continue;

            const double new_g = parent_g + env.edge_cost(next);
            const StateKey nk = env.make_key(next);

            auto it = visited.find(nk);
            int next_idx;
            if (it == visited.end()) {
                next_idx = static_cast<int>(nodes.size());
                nodes.push_back(Node {.state = next, .g = new_g, .parent = node_idx, .depth = parent_depth + 1, .incoming = prim});
                visited.emplace(nk, next_idx);
            } else {
                next_idx = it->second;
                Node& existing = nodes[static_cast<size_t>(next_idx)];
                if (new_g + 1e-9 >= existing.g || existing.closed_anchor) continue;
                existing.g = new_g;
                existing.parent = node_idx;
                existing.depth = parent_depth + 1;
                existing.incoming = prim;
            }

            Node& nn = nodes[static_cast<size_t>(next_idx)];
            anchor_open.push({key_for(nn, 0), next_idx, new_g});
            for (size_t i = 0; i < num_inadmissible; ++i) {
                inadmissible_open[i].push({key_for(nn, i + 1), next_idx, new_g});
            }

            if (env.is_goal(next) && new_g < best_goal_g) {
                best_goal_g = new_g;
                best_goal = next_idx;
            }
        }
    };

    // ── SMHA* 主循环：轮询 inadmissible 队列，anchor 提供有界次优保证 ──
    while (true) {
        if (best_goal >= 0) break; // 首个到达目标的解即返回（anytime + 有界次优）
        if (expansions >= max_expansions) break;
        if (std::chrono::steady_clock::now() - t_start > budget) break;

        // anchor 队列最小 key，作为 inadmissible 展开的准入界。
        int anchor_top = -1;
        while (!anchor_open.empty()) {
            const OpenEntry e = anchor_open.top();
            const Node& n = nodes[static_cast<size_t>(e.node_index)];
            if (e.g_at_insert > n.g + 1e-9 || n.closed_anchor) {
                anchor_open.pop();
                continue;
            }
            anchor_top = e.node_index;
            break;
        }
        if (anchor_top < 0) break; // 全部队列耗尽

        const double anchor_min_key = key_for(nodes[static_cast<size_t>(anchor_top)], 0);

        bool expanded_inadmissible = false;
        for (size_t i = 0; i < num_inadmissible; ++i) {
            if (inadmissible_open[i].empty()) continue;
            // 惰性清理队顶
            const OpenEntry top = inadmissible_open[i].top();
            const Node& tn = nodes[static_cast<size_t>(top.node_index)];
            if (top.g_at_insert > tn.g + 1e-9 || tn.closed_anchor) {
                inadmissible_open[i].pop();
                expanded_inadmissible = true; // 本轮做了工作，继续
                break;
            }
            // 准入判据：inadmissible 队顶 key <= w2 * anchor 最小 key
            if (top.key <= w_inadmissible_ * anchor_min_key) {
                const int idx = pop_valid(inadmissible_open[i]);
                if (idx >= 0) {
                    expand(idx);
                    expanded_inadmissible = true;
                    break;
                }
            }
        }

        if (!expanded_inadmissible) {
            // 无 inadmissible 队列满足准入，展开 anchor 队列（保证进展与完备性）。
            const int idx = pop_valid(anchor_open);
            if (idx < 0) break;
            expand(idx);
        }
    }

    // ── 回溯最优解（若无目标解，取 anchor 队列中离目标启发值最小的已展开节点作 best-effort）──
    int trace = best_goal;
    if (trace < 0) {
        // best-effort：选 g + anchor_h 最小的已扩展节点，给 FDDP 一个尽量靠前的种子。
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
