#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/common/trajectory/annotated_path.hpp>
#include <nav_executor/common/trajectory/trajectory_validation.hpp>
#include <nav_executor/path_planner/search/guide_field.hpp>
#include <nav_executor/path_planner/search/reference_path.hpp>
#include <nav_executor/path_planner/search/spatial_grid_astar.hpp>
#include <nav_executor/path_planner/search/kino_a_star.hpp>
#include <nav_executor/path_planner/trajectory/minco_optimizer.hpp>
#include <nav_executor/common/environment/nav_map.hpp>
#include <nav_executor/common/environment/obstacle_semantics.hpp>
#include <nav_executor/path_planner/trajectory/traversal_annotator.hpp>
#include <nav_executor/path_planner/search/route_terrain_mask.hpp>
#include <nav_executor/path_planner/trajectory/speed_profile_optimizer.hpp>

namespace nav_executor {

// ── 任务目标值对象 ──
struct Goal {
    uint64_t id = 0;
    Eigen::Vector2d position_map = Eigen::Vector2d::Zero();
    bool fixed = false;
};

// ── PlanRequest：控制线程提交给规划 worker 的完整任务包 ──
struct PlanRequest {
    Goal goal;

    // 同一 goal 下每次路径失效或重新规划意图都会推进 generation。
    // worker 结果必须与 TaskManager 当前 generation 一致，避免旧地图/旧失效周期的
    // 晚到结果重新获得执行权。
    uint64_t plan_generation = 0;

    // 当前位姿快照（map 系）
    Eigen::Vector2d current_pos_map = Eigen::Vector2d::Zero();
    double current_yaw = 0.0;
    double current_velocity = 0.0;

    // 地图快照
    PlannerObstacleView obstacles;
    DirectionMap::ConstPtr direction_map; // 方向场（含地形标签）
    TerrainTraversalConstraints terrain_constraints;
    PerformanceState performance;
};

// ── PlanResult ──
struct PlanResult {
    enum class Kind : uint8_t {
        PATH = 0, // 得到新可执行路径
        COMPLETE_NO_PLAN_NEEDED = 1, // 已足够近且 fixed=false
        USE_AS_FIXED_GOAL = 2, // 已足够近且 fixed=true，直接进入保持
        FAILED = 3, // 本次规划失败
    };

    Kind kind = Kind::FAILED;
    uint64_t goal_id = 0;         // 对应请求的 goal id（接纳验证用）
    uint64_t plan_generation = 0; // 对应请求的规划代次（同一 goal 内拒绝过期结果）
    AnnotatedPath::ConstPtr path; // 仅 kind==PATH 有效
    Eigen::Vector2d goal_pos = Eigen::Vector2d::Zero(); // fixed 短路时的保持点

    // 仅 kind==FAILED 时填充，由 TaskManager 在确认结果仍有效后统一记录。
    std::string failure_reason;

    // 可执行但存在降级项；仅在结果被 TaskManager 接纳后记录。
    std::vector<std::string> warnings;

    std::vector<Eigen::Vector2d> debug_spatial_path;
    std::vector<Eigen::Vector2d> debug_smoothed_spatial_path;
    std::vector<Eigen::Vector2d> debug_kino_path;
};

struct PlannerConfig {
    struct DirectionalTerrainParams {
        double min_alignment_cosine;
    } directional_terrain;

    struct MincoSeedParams {
        double max_point_spacing;
        double max_heading_change;
    } minco_seed;

    // 环境验收只检查路径与地图的关系。
    struct TrajectoryValidationParams {
        int samples_per_segment;
        double alignment_warning_angle; // 方向地形偏差告警角（弧度），只记录软诊断
    };

    // 可行性判定
    int occupied_cost_threshold;
    double endpoint_direction_norm_max;

    // nudge
    double endpoint_nudge_max_distance;

    // 近距离短路（robot-to-goal 完成阈值，不是 goal-to-goal 去重阈值）
    double endpoint_goal_reached_distance;

    MotionPrimitiveLibrary::Params motion_primitives;
    SpatialGridAstar::Params spatial_a_star;
    ReferencePathBuilder::Params reference_path;
    GuideFieldBuilder::Params guide_field;
    KinoAStar::Params kino_a_star;
    MincoOptimizer::Params minco;

    TraversalAnnotationParams traversal_annotation;
    StepExecutionTimingParams step_execution_timing;
    TraversalConstraintGateParams traversal_constraint_gate;
    TrajectoryValidationParams trajectory_validation;
    SpeedProfileOptimizer::Params speed_profile;

    bool enable_diagnostics;
};

// 独立线程规划 worker：控制线程 submit()（latest-wins）提交请求，worker 被 cv 唤醒后读取快照做规划，产出不可变 AnnotatedPath。
class PathPlanner {
public:
    PathPlanner(
        const PlannerConfig& config,
        std::shared_ptr<RouteTerrainMask> route_terrain_mask,
        rclcpp::Logger logger
    );
    ~PathPlanner();

    PathPlanner(const PathPlanner&) = delete;
    PathPlanner& operator=(const PathPlanner&) = delete;

    void start();
    void stop();

    // latest-wins：覆盖最新待处理请求槽位并唤醒 worker。
    void submit(const PlanRequest& request);

    // 非阻塞轮询结果。返回 nullopt 表示暂无新结果。
    [[nodiscard]] std::optional<PlanResult> try_take_result();

    // worker 是否正在执行规划（用于统一调度的空闲判定）。
    [[nodiscard]] bool busy() const;

private:
    void worker_loop();
    PlanResult plan(const PlanRequest& request) const;

    // ── 规划期几何工具 ──
    [[nodiscard]] bool is_map_point_feasible(const CostMap& cost_map, const DirectionMap& direction_map, const Eigen::Vector2d& map_pt) const;
    [[nodiscard]] std::optional<Eigen::Vector2d> nudge_point_to_free(const CostMap& cost_map, const DirectionMap& direction_map, const Eigen::Vector2d& map_pt, double max_nudge_distance) const;

    PlannerConfig config_;
    MotionPrimitiveLibrary primitive_library_;
    std::shared_ptr<RouteTerrainMask> route_terrain_mask_;
    rclcpp::Logger logger_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::optional<PlanRequest> pending_request_;
    std::optional<PlanResult> result_;
    bool busy_ = false;
    bool running_ = false;
    std::thread worker_;
};

} // namespace nav_executor
