#pragma once

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
    double prepare_arc_length = 0.0;
    double active_arc_length = 0.0;
    double commit_arc_length = 0.0;
    double step_enter_arc_length = 0.0;
    double step_exit_arc_length = 0.0;
    double release_arc_length = 0.0;
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
