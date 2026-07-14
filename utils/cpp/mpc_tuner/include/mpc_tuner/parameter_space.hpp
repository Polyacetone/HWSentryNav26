#pragma once

#include <array>
#include <string_view>

#include <nav_executor/path_executor/solver/mpc_types.hpp>

namespace mpc_tuner {

// 单个可调参数在归一化搜索空间 [0,1] 与物理量之间的映射区间。
struct SearchRange {
    std::string_view name;
    double lower = 0.0;
    double upper = 0.0;
    bool logarithmic = true;
};

// 可调参数的单一真相源：描述该参数在 CEM 搜索空间中的默认区间，以及它与 MPCParams
// 之间的读写方式。新增/删除一个可调参数只需增删此表中的一条记录，config_loader /
// parameter_mapping / optimizer 全部由此派生，不再各自维护平行的参数列表。
struct ParameterDescriptor {
    std::string_view name;    // CSV 列名与 tuner.yaml search_space 覆盖键
    double default_lower;     // 默认搜索下界
    double default_upper;     // 默认搜索上界
    bool logarithmic;         // true 表示在对数尺度上采样

    // 从基线 MPCParams 读取该参数的初值（决定 CEM 起始均值）。
    // 对“共享乘子”类参数，其自身初值恒为 1.0（乘子的单位元）。
    double (*read_baseline)(const nav_executor::MPCParams&);

    // 把一个物理量写入 MPCParams。对共享乘子，则以该值缩放其作用的整组权重。
    // 约定：apply 始终作用在基线的新副本上，因此乘子直接就地相乘即可，跨候选幂等。
    void (*apply)(nav_executor::MPCParams&, double);
};

inline constexpr auto PARAMETER_DESCRIPTORS = std::to_array<ParameterDescriptor>({
    // ── 跟踪权重 ──
    {"q_y", 0.08, 1.20, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.tracking_weights.q_y; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.tracking_weights.q_y = v; }},
    {"q_theta", 0.04, 0.80, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.tracking_weights.q_theta; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.tracking_weights.q_theta = v; }},
    {"q_u", 0.25, 4.00, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.tracking_weights.q_u; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.tracking_weights.q_u = v; }},
    {"y_tube", 0.10, 0.50, false,
     +[](const nav_executor::MPCParams& p) { return p.follow.tracking_weights.y_tube; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.tracking_weights.y_tube = v; }},
    {"q_term_prog", 0.05, 1.50, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.tracking_weights.q_term_prog; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.tracking_weights.q_term_prog = v; }},
    {"q_term_lateral", 0.05, 1.50, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.tracking_weights.q_term_lateral; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.tracking_weights.q_term_lateral = v; }},

    // ── 命令权重 ──
    {"r_v", 0.002, 0.10, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.command_weights.r_v; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.command_weights.r_v = v; }},
    {"r_omega", 0.002, 0.10, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.command_weights.r_omega; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.command_weights.r_omega = v; }},
    {"r_dv", 0.03, 1.00, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.command_weights.r_dv; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.command_weights.r_dv = v; }},
    {"r_domega", 0.02, 0.80, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.command_weights.r_domega; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.command_weights.r_domega = v; }},

    // ── 环境避障 ──
    {"environment_obstacle", 1.0, 25.0, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.environment_weights.obstacle; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.environment_weights.obstacle = v; }},

    // ── 台阶权重（穿越质量与助跑可达性）──
    {"step_direction", 0.5, 12.0, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.terrain_weights.direction; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.terrain_weights.direction = v; }},
    {"step_velocity_window", 0.4, 10.0, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.terrain_weights.step_vel_weight; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.terrain_weights.step_vel_weight = v; }},
    {"step_reachability_lo", 0.15, 4.0, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.terrain_weights.step_reachability_lo; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.terrain_weights.step_reachability_lo = v; }},
    {"step_reachability_hi", 0.15, 4.0, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.terrain_weights.step_reachability_hi; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.terrain_weights.step_reachability_hi = v; }},

    // ── 终端减速 ──
    {"q_v_final", 0.04, 1.0, true,
     +[](const nav_executor::MPCParams& p) { return p.follow.terminal_weights.q_v_final; },
     +[](nav_executor::MPCParams& p, double v) { p.follow.terminal_weights.q_v_final = v; }},

    // ── 共享乘子（同时缩放一整组语义相近的惩罚，避免维度爆炸）──
    // motion_penalty_scale：缩放指令可行性惩罚（加速度/角加速度/侧向）。
    {"motion_penalty_scale", 0.3, 3.0, true,
     +[](const nav_executor::MPCParams&) { return 1.0; },
     +[](nav_executor::MPCParams& p, double v) {
         p.follow.motion_constraint_weights.acc_limit *= v;
         p.follow.motion_constraint_weights.alpha_limit *= v;
         p.follow.motion_constraint_weights.lat_acc *= v;
     }},
    // step_smoothness_scale：缩放台阶内部平滑惩罚（角速度/线加速/角加速）。
    {"step_smoothness_scale", 0.3, 3.0, true,
     +[](const nav_executor::MPCParams&) { return 1.0; },
     +[](nav_executor::MPCParams& p, double v) {
         p.follow.terrain_weights.step_omega *= v;
         p.follow.terrain_weights.step_dv *= v;
         p.follow.terrain_weights.step_domega *= v;
     }},
});

inline constexpr size_t PARAMETER_COUNT = PARAMETER_DESCRIPTORS.size();

} // namespace mpc_tuner
