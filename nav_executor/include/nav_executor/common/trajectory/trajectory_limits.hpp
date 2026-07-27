#pragma once

#include <algorithm>

namespace nav_executor {

// MINCO 形状优化和速度剖面共用的规划限值来源。
struct TrajectoryLimits {
    double velocity_max;              // 路径速度幅值上限 (m/s)
    double acceleration_max;          // 切向加速度上限 (m/s²)
    double angular_velocity_max;      // 角速度上限 (rad/s)
    double angular_acceleration_max;  // 角加速度上限 (rad/s²)
    double lateral_acceleration_max;  // 侧向加速度上限 (m/s²)

    // 经实验降额的最低可稳定跟踪速度；用于前端种子和 MINCO 软约束。
    double min_trackable_speed;

    // MINCO 参数化下速度沿种子方向分量的软约束下限 (m/s)。
    double directed_speed_min;

    // MINCO 曲率软约束阈值：κ_max = min(ω_max/v_track, a_lat_max/v_track²)
    [[nodiscard]] double curvature_max() const {
        const double speed = min_trackable_speed;
        return std::min(
            angular_velocity_max / speed,
            lateral_acceleration_max / (speed * speed)
        );
    }

    // MINCO 曲率变化率软约束阈值，在 v_track 处由角加速度包络换算。
    [[nodiscard]] double curvature_rate_max() const {
        const double speed = min_trackable_speed;
        return angular_acceleration_max / (speed * speed);
    }
};

} // namespace nav_executor
