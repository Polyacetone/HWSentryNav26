#pragma once

#include <vector>

#include <rclcpp/logger.hpp>

#include <nav_executor/common/trajectory/annotated_path.hpp>
#include <nav_executor/common/trajectory/minco_trajectory.hpp>
#include <nav_executor/common/environment/nav_map.hpp>

namespace nav_executor {

struct TraversalAnnotationParams {
    double sample_spacing;
};

struct StepExecutionTimingParams {
    double profile_prepare_distance;
    double chassis_activation_distance;
    double fsm_release_distance;
};

struct TraversalConstraintGateParams {
    double gate_transition_distance; // 约束窗软门控两侧过渡带宽度 (m)
};

// 方向地形标注器：规划期扫描原始地形本体，产出执行层使用的不可变 StepPlanSegment 列表。
// 无状态纯函数集合，可在规划 worker 线程安全调用。
namespace traversal_annotator {

std::vector<StepPlanSegment> build_step_plan(
    const TraversalAnnotationParams& annotation,
    const StepExecutionTimingParams& execution_timing,
    const TraversalConstraintGateParams& constraint_gate,
    const MincoTrajectory& path,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    rclcpp::Logger logger
);

} // namespace traversal_annotator

} // namespace nav_executor
