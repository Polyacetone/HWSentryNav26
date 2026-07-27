#pragma once

#include <optional>
#include <string>

#include <nav_executor/common/trajectory/minco_trajectory.hpp>

namespace nav_executor {

// 只验证会使路径数学定义失效的数值错误。曲率、曲率变化率和路径拓扑
// 属于优化质量或速度剖面问题，不在这里否决确定性的规划结果。
[[nodiscard]] std::optional<std::string> validate_trajectory_numerics(
    const MincoTrajectory& trajectory
);

} // namespace nav_executor
