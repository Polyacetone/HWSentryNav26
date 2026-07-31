#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/chassis_defs.hpp>
#include <nav_executor/common/environment/nav_map.hpp>

namespace nav_executor {

// 路径台阶约束。所有边界均为累计弧长，不携带底盘模式或 FSM 语义。
struct StepTraversalConstraint {
    TraversalVelocityWindow velocity_window;
    double commit_arc_length = 0.0;
    double step_enter_arc_length = 0.0;
    double exit_arc_length = 0.0;
    double gate_start_arc_length = 0.0;
    double gate_end_arc_length = 0.0;
    Eigen::Vector2d dir_map = Eigen::Vector2d::Zero();

    bool operator==(const StepTraversalConstraint&) const = default;
};

// 台阶软约束沿累计弧长使用同一梯形窗；规划期速度剖面与运行期 MPC 共同调用。
inline double step_window_gate(
    const double path_progress,
    const StepTraversalConstraint& constraint
) {
    if (path_progress >= constraint.commit_arc_length
        && path_progress <= constraint.exit_arc_length) return 1.0;
    if (path_progress <= constraint.gate_start_arc_length
        || path_progress >= constraint.gate_end_arc_length) return 0.0;
    const auto smoothstep = [](double value) {
        value = std::clamp(value, 0.0, 1.0);
        return value * value * (3.0 - 2.0 * value);
    };
    if (path_progress < constraint.commit_arc_length) {
        const double width = constraint.commit_arc_length
            - constraint.gate_start_arc_length;
        return width > 1e-9
            ? smoothstep(
                (path_progress - constraint.gate_start_arc_length) / width
            ) : 1.0;
    }
    const double width = constraint.gate_end_arc_length - constraint.exit_arc_length;
    return width > 1e-9
        ? smoothstep((constraint.gate_end_arc_length - path_progress) / width) : 1.0;
}

class StepConstraintSchedule {
public:
    explicit StepConstraintSchedule(std::vector<StepTraversalConstraint> constraints)
        : constraints_(std::move(constraints)) {}

    [[nodiscard]] const StepTraversalConstraint* constraint_at(
        const double path_progress
    ) const {
        for (const auto& constraint : constraints_) {
            if (path_progress < constraint.gate_start_arc_length) break;
            if (path_progress <= constraint.gate_end_arc_length) return &constraint;
        }
        return nullptr;
    }

private:
    std::vector<StepTraversalConstraint> constraints_;
};

struct StepChassisCommand {
    uint8_t mode = chassis_mode::NORMAL;
    CapabilityLevel capability = CapabilityLevel::LOW;

    bool operator==(const StepChassisCommand&) const = default;
};

// 规划期生成、运行期只读的完整台阶生命周期。
struct StepPlanSegment {
    double prepare_arc_length = 0.0; // 开始渐变 capability 的位置。
    double active_arc_length = 0.0; // STEPPING-ARMED 起点。开始给底盘发送台阶模式的位置。
    double commit_arc_length = 0.0; // STEPPING-COMMITTED 起点。底盘执行器实际开始做地形跨越动作的预期位置。因此是 runup 约束开始处。
    double step_enter_arc_length = 0.0; // 物理台阶边缘入口。
    double step_exit_arc_length = 0.0; // 物理台阶边缘出口。
    double release_arc_length = 0.0; // STEPPING 父状态退出位置。
    Eigen::Vector2d step_enter_pos_map = Eigen::Vector2d::Zero();
    Eigen::Vector2d step_exit_pos_map = Eigen::Vector2d::Zero();
    Eigen::Vector2d dir_map = Eigen::Vector2d::Zero();
    StepDirection direction = StepDirection::UP;
    StepChassisCommand chassis_command;
    StepTraversalConstraint traversal_constraint;
    uint8_t terrain_label = 0;
    bool requires_high_performance = false;
};

} // namespace nav_executor
