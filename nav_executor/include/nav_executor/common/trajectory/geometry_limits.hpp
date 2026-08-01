#pragma once

namespace nav_executor {

// 固定空间曲线上的几何能力边界。所有量都与轨迹时标无关。
struct GeometryLimits {
    double curvature_max = 4.0;             // 最小转弯半径的倒数 (1/m)
    double curvature_rate_max = 8.0;        // 最大弧长曲率变化率 (1/m²)
    double tangent_regularization = 1e-6;   // 局部归一化段参数下的切向正则尺度 (m)
};

} // namespace nav_executor
