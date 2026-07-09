#pragma once

#include <vector>

#include <rclcpp/logger.hpp>

#include <nav_executor/path/annotated_path.hpp>
#include <nav_executor/path/spline_path.hpp>
#include <nav_executor/planner/nav_map.hpp>

namespace nav_executor {

struct StepDetectionParams {
    double detect_norm_threshold;
    double detect_dot_threshold;
    double path_sample_resolution;
    double prepare_distance;
    double active_distance;
    double release_distance;
};

// 台阶几何标注器（build_step_plan 从 StepController 上移到 planner）。
//
// 在规划期基于方向场对样条做一次台阶扫描，产出不可变的 StepPlanSegment 列表，
// 装入 AnnotatedPath.step_segments。运行时不再重建。
//
// 这是一个无状态的纯函数集合，可在规划 worker 线程安全调用。
namespace step_annotator {

std::vector<StepPlanSegment> build_step_plan(
    const StepDetectionParams& params,
    const SplinePath& path,
    const DirectionMap& direction_map,
    rclcpp::Logger logger
);

} // namespace step_annotator

} // namespace nav_executor
