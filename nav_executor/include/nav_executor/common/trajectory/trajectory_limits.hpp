#pragma once

#include <algorithm>

namespace nav_executor {

// 全局规划包络：几何证书、MINCO 形状优化和速度剖面共用的唯一限值来源。
// 曲率上限由底盘能力在最低可稳定跟踪速度处推导，因此“几何可行”与
// “该几何在可跟踪速度下动力学可行”是同一个判据。
struct TrajectoryLimits {
    double velocity_max;              // 路径速度幅值上限 (m/s)
    double acceleration_max;          // 切向加速度上限 (m/s²)
    double angular_velocity_max;      // 角速度上限 (rad/s)
    double angular_acceleration_max;  // 角加速度上限 (rad/s²)
    double lateral_acceleration_max;  // 侧向加速度上限 (m/s²)

    // 经实验降额的最低可稳定跟踪速度；曲率包络在该速度处取等号。
    double min_trackable_speed;

    // 有向正则性下限：MINCO 参数化下速度沿种子方向分量的最小值 (m/s)。
    // 这是严格不等式 p'·t̂ > 0 的数值裕度，排除内部零速点与逆向段。
    double directed_speed_min;

    // κ_max = min(ω_max/v_track, a_lat_max/v_track²)
    [[nodiscard]] double curvature_max() const {
        const double speed = min_trackable_speed;
        return std::min(
            angular_velocity_max / speed,
            lateral_acceleration_max / (speed * speed)
        );
    }

    // dκ/ds 上限：在 v_track 处以角加速度包络换算，抑制“极短距离内建立角速度”。
    [[nodiscard]] double curvature_rate_max() const {
        const double speed = min_trackable_speed;
        return angular_acceleration_max / (speed * speed);
    }
};

} // namespace nav_executor
