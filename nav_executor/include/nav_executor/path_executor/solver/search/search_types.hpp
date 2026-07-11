#pragma once

/// @file search_types.hpp
/// @brief Anytime MHA* 局部搜索的基础类型：运动学状态、离散键、运动基元、搜索结果。

#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace nav_executor::search {

/// 搜索状态：降阶运动学独轮车 (x, y, θ, v)。
/// 隐状态 XH / 角速度 W 不在搜索中建模，交由 FDDP 的 feasibility gap 吸收。
struct SearchState {
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;
    double v = 0.0;
};

/// 闭表离散键：(col, row, θ_bin, v_bin)。
struct StateKey {
    int col = 0;
    int row = 0;
    int theta_bin = 0;
    int v_bin = 0;

    bool operator==(const StateKey&) const = default;
};

struct StateKeyHash {
    size_t operator()(const StateKey& k) const noexcept {
        // 混合四个整数分量（θ_bin / v_bin 范围小，用位移打散）。
        uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(k.col));
        h = h * 0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(static_cast<uint32_t>(k.row));
        h = h * 0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(static_cast<uint32_t>(k.theta_bin));
        h = h * 0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(static_cast<uint32_t>(k.v_bin));
        return static_cast<size_t>(h ^ (h >> 32));
    }
};

/// 运动基元：一组 (v_cmd, ω) 常值控制，作用 dt 时长。
struct MotionPrimitive {
    double v_cmd = 0.0;
    double omega = 0.0;
};

/// 搜索输出：dt 间隔的状态序列（含起点）。
struct SearchResult {
    std::vector<SearchState> states;   // states[0] == 起点
    std::vector<MotionPrimitive> controls; // controls[k] 从 states[k] 到 states[k+1]
    double cost = 0.0;
    bool valid = false;
    int expansions = 0;
};

} // namespace nav_executor::search
