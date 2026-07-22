#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/common/annotated_path.hpp>
#include <nav_executor/path_planner/dijkstra_cost_to_goal.hpp>
#include <nav_executor/path_planner/kinodynamic_astar.hpp>
#include <nav_executor/path_planner/minco_optimizer.hpp>
#include <nav_executor/path_planner/nav_map.hpp>
#include <nav_executor/path_planner/step_annotator.hpp>
#include <nav_executor/path_planner/step_routing_mask.hpp>

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

    // 当前位姿快照（map 系）
    Eigen::Vector2d current_pos_map = Eigen::Vector2d::Zero();
    double current_yaw = 0.0;
    double current_velocity = 0.0;

    // 地图快照
    CostMap::ConstPtr global_cost_map;   // 全局先验（用于严格可行性检查）
    CostMap::ConstPtr merged_cost_map;   // global + 时域融合动态障碍物（A* / 样条用）
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
    AnnotatedPath::ConstPtr path; // 仅 kind==PATH 有效
    Eigen::Vector2d goal_pos = Eigen::Vector2d::Zero(); // fixed 短路时的保持点

    std::vector<Eigen::Vector2d> debug_rough_path;
};

struct PlannerConfig {
    struct TrajectoryValidationParams {
        int samples_per_segment;
        double velocity_tolerance;
        double omega_tolerance;
        double acceleration_tolerance;
        double lateral_acceleration_tolerance;
        double traversal_velocity_target_tolerance;
        double traversal_angle_tolerance; // 台阶方向角允许偏差（弧度），只打印日志不进行拒绝
        double self_intersection_flatness_tolerance;
        double self_intersection_max_edge_length;
        double cusp_retrace_alignment_threshold;
    };

    // 可行性判定
    int occupied_threshold;
    double on_step_threshold;

    // 起点预测
    bool start_prediction_enable;
    double start_prediction_max_accel;
    double start_prediction_planning_delay;
    double start_prediction_min_speed;
    double start_prediction_collision_check_step;

    // nudge
    double nudge_max_distance;

    // 近距离短路（robot-to-goal 完成阈值，不是 goal-to-goal 去重阈值）
    double goal_reached_distance;

    // 短路：起终点距离 < 该值时跳过 kinodynamic，直连折线种子
    double skip_distance;

    // MINCO 种子构造：kinodynamic 状态序列的重采样间隔（米），决定 MINCO 段数
    double seed_resample_distance;

    DijkstraCostToGoal::Params dijkstra;
    KinodynamicAstar::Params kinodynamic;
    MincoOptimizer::Params minco;

    StepDetectionParams step_detection;
    TrajectoryValidationParams trajectory_validation;

    bool enable_debug;
};

// 独立线程规划 worker：控制线程 submit()（latest-wins）提交请求，worker 被 cv 唤醒后读取快照做规划，产出不可变 AnnotatedPath。
class PathPlanner {
public:
    PathPlanner(
        const PlannerConfig& config,
        std::shared_ptr<StepRoutingMask> step_routing_mask,
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
    [[nodiscard]] bool is_map_point_feasible(const CostMap& cost_map, const DirectionMap& direction_map, const TerrainTraversalConstraints& terrain_constraints, const Eigen::Vector2d& map_pt) const;
    [[nodiscard]] Eigen::Vector2d predict_start_map(const PlanRequest& req) const;
    [[nodiscard]] Eigen::Vector2d adjust_reachable_start_on_segment(const PlanRequest& req, const Eigen::Vector2d& from_map, const Eigen::Vector2d& to_map) const;
    [[nodiscard]] std::optional<Eigen::Vector2d> nudge_point_to_free(const PlanRequest& req, const Eigen::Vector2d& map_pt, double max_nudge_distance) const;

    PlannerConfig config_;
    std::shared_ptr<StepRoutingMask> step_routing_mask_;
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
