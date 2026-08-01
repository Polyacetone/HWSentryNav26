#pragma once

namespace nav_executor {

// State-lattice 搜索使用的动力学包络，只负责构造带方向和时长的可达搜索见证。
// MINCO 只读取见证的几何边界；路径发布前，SpeedProfileOptimizer 会在固定几何上
// 根据 MPC capability 求解唯一执行时标。
struct ShapingDynamicsLimits {
    double velocity_max;                   // 见证线速度上限 (m/s)
    double tangential_acceleration_max;    // 标量速度变化率 |d speed/dt| 上限 (m/s²)
    double angular_velocity_max;           // |omega| 上限 (rad/s)
    double angular_acceleration_max;       // |domega/dt| 上限 (rad/s²)
    double lateral_acceleration_max;       // |kappa|v² 上限 (m/s²)
};

} // namespace nav_executor
