#pragma once

#include <memory>
#include <vector>

#include <nav_executor/common/environment/nav_map.hpp>

namespace nav_executor {

struct ObstacleLayers {
    // map_server 提供的全局静态障碍，已按机器人半径膨胀。
    CostMap::ConstPtr global_static;
    // map_server 提供的当前动态障碍原始帧，包含地形和平地上的动态障碍。
    CostMap::ConstPtr dynamic_current;
    // 当前帧之后的动态障碍预测原始帧，按 prediction_dt 等间隔排列。
    std::vector<CostMap::ConstPtr> dynamic_predictions;
    // 未经路径擦除的连续地形影响层：方向场模长映射到 [0, 255]。
    CostMap::ConstPtr base_terrain_cost;
    // 全局方向场；原始方向模长大于 0.95 的格子定义为地形物理本体。
    DirectionMap::ConstPtr base_direction;
};

// 规划器只处理会改变地形通道拓扑的障碍：全局静态障碍，以及预测窗口内
// 经连续地形影响层加权的动态障碍。平地动态障碍和路径生成后的台阶掩码不进入此视图。
struct PlannerObstacleView {
    // 仅含全局静态障碍，用于起终点严格检查和普通目标 nudge。
    CostMap::ConstPtr global_static;
    // 全局静态障碍 + 地形加权动态障碍的预测时域逐格最大值，用于规划搜索与验收。
    CostMap::ConstPtr hard_cost;
    // 所有动态帧与 base_terrain_cost 的归一化乘积，供 RouteMonitor 复用。
    std::vector<CostMap::ConstPtr> terrain_dynamic_timeline;
};

// 跟随侧保留动态障碍的逐帧时间结构。动态障碍通常只进入 MPC 软成本；唯一例外是
// RouteMonitor 按预计到达时间发现路径前方的地形通行区被占用时触发重新规划。
struct FollowerObstacleView {
    // 全局静态障碍 + 当前路径的台阶掩码；供 lethal、hazard 和投影保护等硬判定使用。
    CostMap::ConstPtr hard_route_cost;
    // hard_route_cost + 当前动态障碍；供当前时刻的 MPC 软避障使用。
    CostMap::ConstPtr soft_current_cost;
    // 每张预测动态帧分别与 hard_route_cost 融合，供未来 MPC stage 使用。
    std::vector<CostMap::ConstPtr> soft_prediction_costs;
    std::vector<const CostMap*> soft_prediction_cost_ptrs;

    // 每帧动态代价与 base_terrain_cost 的归一化乘积：[当前帧, 未来预测帧...]。
    // RouteMonitor 按预计到达时间读取，与规划器使用相同的地形相关障碍语义。
    std::vector<CostMap::ConstPtr> terrain_dynamic_timeline;

    DirectionMap::ConstPtr base_direction;
    // 当前路径擦除已选台阶通道后的方向场；无活动路径时等于 base_direction。
    DirectionMap::ConstPtr route_direction;
    double prediction_dt = 0.0;
    int occupied_threshold = 0;
};

[[nodiscard]] PlannerObstacleView build_planner_obstacle_view(
    const ObstacleLayers& layers,
    double prediction_dt,
    double prediction_horizon_seconds
);

[[nodiscard]] FollowerObstacleView build_follower_obstacle_view(
    const ObstacleLayers& layers,
    const std::vector<CostMap::ConstPtr>& terrain_dynamic_timeline,
    const CostMap::ConstPtr& step_cost_layer,
    const DirectionMap::ConstPtr& route_direction,
    double prediction_dt,
    int occupied_threshold
);

} // namespace nav_executor
