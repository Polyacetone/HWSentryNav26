#include <nav_executor/path_planner/kinodynamic_astar.hpp>

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>

namespace nav_executor {

namespace {

constexpr double EPS = 1e-9;

// 去重键：把连续状态量化到网格。用 64 位打包 (ix, iy, itheta, iv)。
struct StateKey {
    uint64_t packed = 0;
    bool operator==(const StateKey& o) const { return packed == o.packed; }
};

struct StateKeyHash {
    size_t operator()(const StateKey& k) const { return std::hash<uint64_t>{}(k.packed); }
};

int64_t quantize_component(double value, double resolution) {
    return static_cast<int64_t>(std::llround(value / std::max(resolution, EPS)));
}

StateKey make_key(const KinodynamicAstar::State& s, const KinodynamicAstar::Params& p) {
    // θ 归一到 [0,2π) 再量化，避免 ±π 跨界重复。
    double th = std::fmod(s.theta, 2.0 * M_PI);
    if (th < 0) th += 2.0 * M_PI;
    const int64_t ix = quantize_component(s.x, p.dedup_xy);
    const int64_t iy = quantize_component(s.y, p.dedup_xy);
    const int64_t theta_bins = std::max<int64_t>(1, std::llround(2.0 * M_PI / std::max(p.dedup_theta, EPS)));
    const int64_t it = quantize_component(th, p.dedup_theta) % theta_bins;
    const int64_t iv = quantize_component(s.v, p.dedup_v);
    // 打包（各 16 位，位置范围 ±32767·分辨率 足够场地尺度）。
    StateKey k;
    k.packed = (static_cast<uint64_t>(ix) & 0xFFFFULL) << 48
        | (static_cast<uint64_t>(iy) & 0xFFFFULL) << 32
        | (static_cast<uint64_t>(it) & 0xFFFFULL) << 16
        | (static_cast<uint64_t>(iv) & 0xFFFFULL);
    return k;
}

struct SearchNode {
    KinodynamicAstar::State state;
    double g = 0.0;
    double f = 0.0;
    int parent = -1;
    double applied_accel = 0.0;
    double applied_omega = 0.0;
};

struct OpenEntry {
    double f;
    int node_index;
    bool operator>(const OpenEntry& o) const { return f > o.f; }
};

} // anonymous namespace

KinodynamicAstar::Result KinodynamicAstar::search(
    const State& start,
    const DijkstraCostToGoal& dijkstra,
    const TransitionFeasibleFn& transition_feasible,
    const GoalReachedFn& goal_reached
) const {
    Result result;

    if (!dijkstra.ready()) {
        result.error = "Dijkstra field not built";
        return result;
    }

    std::vector<SearchNode> nodes;
    nodes.reserve(4096);
    std::unordered_map<StateKey, double, StateKeyHash> best_g; // 已知最优 g（去重）
    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<>> open;

    const auto heuristic = [&](const State& s) -> double {
        const double h = dijkstra.at_map(Eigen::Vector2d(s.x, s.y));
        return std::isinf(h) ? DijkstraCostToGoal::UNREACHABLE : params_.heuristic_weight * h;
    };

    // 起点
    SearchNode start_node;
    start_node.state = start;
    start_node.g = 0.0;
    start_node.f = heuristic(start);
    nodes.push_back(start_node);
    best_g[make_key(start, params_)] = 0.0;
    open.push({start_node.f, 0});

    // 原语采样表
    std::vector<double> accels;
    accels.reserve(static_cast<size_t>(std::max(params_.accel_samples, 1)));
    if (params_.accel_samples <= 1) {
        accels.push_back(0.0);
    } else {
        for (int i = 0; i < params_.accel_samples; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(params_.accel_samples - 1);
            accels.push_back(-params_.accel_max + 2.0 * params_.accel_max * t);
        }
    }
    std::vector<double> omegas;
    omegas.reserve(static_cast<size_t>(std::max(params_.omega_samples, 1)));
    if (params_.omega_samples <= 1) {
        omegas.push_back(0.0);
    } else {
        for (int i = 0; i < params_.omega_samples; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(params_.omega_samples - 1);
            omegas.push_back(-params_.omega_max + 2.0 * params_.omega_max * t);
        }
    }

    const double dt = params_.primitive_duration;
    const int substeps = std::max(params_.collision_substeps, 1);
    const double sub_dt = dt / static_cast<double>(substeps);

    int goal_node_index = -1;

    while (!open.empty()) {
        if (result.expansions >= params_.max_expansions) {
            result.error = "expansion limit reached";
            break;
        }
        const OpenEntry entry = open.top();
        open.pop();
        const int cur_index = entry.node_index;
        const SearchNode cur = nodes[static_cast<size_t>(cur_index)];

        // 过期条目（已有更优 g）跳过。
        const StateKey cur_key = make_key(cur.state, params_);
        const auto it = best_g.find(cur_key);
        if (it != best_g.end() && cur.g > it->second + EPS) continue;

        ++result.expansions;

        if (goal_reached(cur.state)) {
            goal_node_index = cur_index;
            result.success = true;
            break;
        }

        // 展开原语
        for (const double a : accels) {
            for (const double omega : omegas) {
                // a_lat 剪枝：全程 |v·ω| ≤ a_lat_max（用起点 v 近似判定，逐细步再查）。
                State s = cur.state;
                bool feasible_primitive = true;

                for (int k = 0; k < substeps; ++k) {
                    const State previous = s;
                    // 前向积分（半隐式：先更新 v，再用中点朝向推进位置）。
                    const double v_next = std::clamp(s.v + a * sub_dt, params_.vel_min, params_.vel_max);
                    const double v_mid = 0.5 * (s.v + v_next);
                    // a_lat 剪枝
                    if (std::abs(v_mid * omega) > params_.a_lat_max + EPS) {
                        feasible_primitive = false;
                        break;
                    }
                    const double theta_next = s.theta + omega * sub_dt;
                    const double theta_mid = s.theta + 0.5 * omega * sub_dt;
                    s.x += v_mid * std::cos(theta_mid) * sub_dt;
                    s.y += v_mid * std::sin(theta_mid) * sub_dt;
                    s.theta = theta_next;
                    s.v = v_next;

                    if (!transition_feasible(previous, s)) {
                        feasible_primitive = false;
                        break;
                    }
                }
                if (!feasible_primitive) continue;

                // h 不可达 → 剪枝（如落入障碍隔离区）。
                const double h = heuristic(s);
                if (std::isinf(h)) continue;

                // 边代价：时长 + 倒车惩罚。
                double edge_cost = params_.time_weight * dt;
                if (s.v < 0.0) edge_cost += params_.reverse_weight * std::abs(s.v) * dt;
                const double new_g = cur.g + edge_cost;

                const StateKey nkey = make_key(s, params_);
                const auto nit = best_g.find(nkey);
                if (nit != best_g.end() && new_g >= nit->second - EPS) continue;

                best_g[nkey] = new_g;
                SearchNode nn;
                nn.state = s;
                nn.g = new_g;
                nn.f = new_g + h;
                nn.parent = cur_index;
                nn.applied_accel = a;
                nn.applied_omega = omega;
                const int nidx = static_cast<int>(nodes.size());
                nodes.push_back(nn);
                open.push({nn.f, nidx});
            }
        }
    }

    if (goal_node_index < 0) {
        if (result.error.empty()) result.error = "no feasible kinodynamic path";
        result.success = false;
        return result;
    }

    // 回溯节点，再按搜索时相同积分细分重放每条原语。细分状态会保留真实台阶入口，
    // 供后续 MINCO 固定入口位姿，避免入口落在两个粗节点之间。
    std::vector<int> reversed_indices;
    for (int idx = goal_node_index; idx >= 0; idx = nodes[static_cast<size_t>(idx)].parent) {
        reversed_indices.push_back(idx);
    }
    std::reverse(reversed_indices.begin(), reversed_indices.end());
    result.states.clear();
    result.states.reserve(1 + (reversed_indices.size() - 1) * static_cast<size_t>(substeps));
    result.states.push_back(start);
    State replay = start;
    for (size_t i = 1; i < reversed_indices.size(); ++i) {
        const SearchNode& node = nodes[static_cast<size_t>(reversed_indices[i])];
        for (int k = 0; k < substeps; ++k) {
            const double v_next = std::clamp(
                replay.v + node.applied_accel * sub_dt, params_.vel_min, params_.vel_max
            );
            const double v_mid = 0.5 * (replay.v + v_next);
            const double theta_mid = replay.theta + 0.5 * node.applied_omega * sub_dt;
            replay.x += v_mid * std::cos(theta_mid) * sub_dt;
            replay.y += v_mid * std::sin(theta_mid) * sub_dt;
            replay.theta += node.applied_omega * sub_dt;
            replay.v = v_next;
            result.states.push_back(replay);
        }
    }
    return result;
}

} // namespace nav_executor
