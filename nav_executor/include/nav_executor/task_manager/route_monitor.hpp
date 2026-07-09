#pragma once

#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/common/annotated_path.hpp>
#include <nav_executor/path_planner/nav_map.hpp>

namespace nav_executor {

struct FollowProjectionGuardParams {
    double dist_max;
    double cost_max;
    int cost_samples;
};

struct StepBlockReplanParams {
    bool enable;
    double lookahead_distance;
    double sample_resolution;
    double step_norm_threshold;
    double obstacle_cost_threshold;
    double predicted_obstacle_ratio_threshold;
};

// RouteMonitor 输出的重规划原因。自身仅产出前三种；
// EXECUTOR_REPLAN_EVENT 由 node 在 step 3 处理，此处仅用于诊断记录。
enum class ReplanReason : uint8_t {
    NONE = 0,
    PROJECTION_GUARD = 1,
    STEP_BLOCKED = 2,
    MPC_LETHAL = 3,
    EXECUTOR_REPLAN_EVENT = 4,
};

const char* replan_reason_str(ReplanReason reason);

// RouteMonitor 输入，由 nav_executor_node 每周期组装后传入。
struct RouteMonitorInput {
    const AnnotatedPath* active_path = nullptr;
    double current_u = 0.0;

    Eigen::Vector2d chassis_pos_map = Eigen::Vector2d::Zero();

    const CostMap* masked_global_cost_map = nullptr;
    const CostMap* current_dynamic_cost_map = nullptr;
    std::vector<const CostMap*> per_step_dynamic_cost_maps;

    const DirectionMap* masked_direction_map = nullptr;

    FollowProjectionGuardParams proj_guard{};
    StepBlockReplanParams step_block{};

    // 上一周期 PathExecutor 输出的 one-shot MPC_LETHAL 事实。
    bool mpc_lethal = false;
};

struct RouteMonitorReport {
    double current_u = 0.0;
    bool needs_replan = false;
    ReplanReason reason = ReplanReason::NONE;
};

// 顶层 FOLLOW 阶段 path monitoring。无状态纯函数。只回答："当前 active_path 还值不值得继续执行？"
// 运行条件：仅在 motion_state==FOLLOW && active_path!=nullptr 时调用。
// 触发 needs_replan=true 的来源仅有：
//   PROJECTION_GUARD / STEP_BLOCKED / MPC_LETHAL / EXECUTOR_REPLAN_EVENT
// 以下属于底层运动异常处理，不属于本模块：
//   follow_no_progress / stepping_no_progress / stuck / hazard / recovery
// 输入由调用者每周期组装，不接受额外外部状态。
RouteMonitorReport run_route_monitor(const RouteMonitorInput& input, rclcpp::Logger logger);

} // namespace nav_executor
