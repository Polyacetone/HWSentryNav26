#pragma once

namespace nav_executor {

// KinoA* 与 MINCO 共用的几何塑形动力学包络。
//
// 这些量只用于证明/推动空间路径存在合理的运动学见证，不构成最终参考速度。
// 路径发布前，SpeedProfileOptimizer 会在固定几何上根据 MPC capability 重新求解唯一执行时标。
struct ShapingDynamicsLimits {
    double velocity_max;                   // 见证线速度上限 (m/s)
    double tangential_acceleration_max;    // 标量速度变化率 |d speed/dt| 上限 (m/s²)
    double angular_velocity_max;           // |omega| 上限 (rad/s)
    double angular_acceleration_max;       // |domega/dt| 上限 (rad/s²)
    double lateral_acceleration_max;       // |kappa|v² 上限 (m/s²)
};

} // namespace nav_executor
