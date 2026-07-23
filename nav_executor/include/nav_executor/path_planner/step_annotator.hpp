#pragma once

#include <vector>

#include <rclcpp/logger.hpp>

#include <nav_executor/common/annotated_path.hpp>
#include <nav_executor/common/minco_trajectory.hpp>
#include <nav_executor/path_planner/nav_map.hpp>

namespace nav_executor {

struct StepDetectionParams {
    double detect_dot_threshold;
    double path_sample_resolution;
    double profile_prepare_distance;
    double chassis_activation_distance;
    double fsm_release_distance;
    double gate_transition_distance; // 约束窗软门控两侧过渡带宽度 (m)
};

// 台阶几何标注器：规划期基于原始方向地形本体对样条做一次扫描，产出不可变的 StepPlanSegment 列表装入 AnnotatedPath.step_segments。
// 无状态纯函数集合，可在规划 worker 线程安全调用。
namespace step_annotator {

std::vector<StepPlanSegment> build_step_plan(
    const StepDetectionParams& params,
    const MincoTrajectory& path,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    rclcpp::Logger logger
);

} // namespace step_annotator

} // namespace nav_executor
