#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <interfaces/msg/nav_goal.hpp>
#include <interfaces/msg/spin_cmd.hpp>
#include <interfaces/msg/cost_maps.hpp>
#include <interfaces/msg/chassis_cmd.hpp>
#include <interfaces/msg/chassis_status.hpp>
#include <interfaces/msg/comp_stage.hpp>
#include <interfaces/msg/idle_chassis_mode_override.hpp>
#include <interfaces/msg/nav_executor_diag.hpp>

#include <nav_executor/common/environment/obstacle_semantics.hpp>
#include <nav_executor/common/tracking/route_tracker.hpp>
#include <nav_executor/task_manager/task_manager.hpp>
#include <nav_executor/path_executor/path_executor.hpp>
#include <nav_executor/path_executor/state/step_controller.hpp>
#include <nav_executor/task_manager/route_monitor.hpp>
#include <nav_executor/path_planner/path_planner.hpp>
#include <nav_executor/common/environment/nav_map.hpp>
#include <nav_executor/path_planner/search/route_terrain_mask.hpp>
#include <nav_executor/path_executor/mpc/mpc_solver.hpp>

namespace nav_executor {

class NavExecutorNode : public rclcpp::Node {
public:
    explicit NavExecutorNode(const rclcpp::NodeOptions& options);
    ~NavExecutorNode() override;

private:
    // ─── 初始化 / 参数加载（实现见 nav_executor_init.cpp）───
    void load_terrain_config();
    MPCParams load_mpc_params();
    FsmParams load_fsm_params();
    PathExecutorParams load_executor_params();
    PlannerConfig load_planner_config(
        const CapabilityProfile& normal_profile,
        const std::array<CapabilityProfile, 3>& step_profiles
    );
    TaskManagerParams load_task_params();
    ProfileBlendParams load_blend_params();
    CapabilityProfile load_capability_profile(const std::string& prefix);

    // ─── ROS 回调 ───
    void chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg);
    void local_cost_maps_callback(const interfaces::msg::CostMaps::SharedPtr msg);
    void spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg);
    void control_tick();

    // ─── 工具 ───
    bool get_chassis_pose(Eigen::Vector3d& chassis_pose) const;
    void try_init_route_terrain_mask();
    void refresh_planner_obstacles();
    nav_msgs::msg::Path path_to_nav_msg(const std::vector<Eigen::Vector2d>& points) const;
    nav_msgs::msg::Path trajectory_to_nav_msg(const MincoTrajectory& trajectory) const;
    visualization_msgs::msg::Marker trajectory_to_marker(const AnnotatedPath& path) const;
    void record_input_rejection(uint8_t reason);
    void publish_diagnostics(
        uint8_t cycle_result,
        const rclcpp::Time& stamp,
        const TaskDiagnostics& diag,
        const std::optional<RouteEstimate>& route,
        const ExecutorOutput* executor_output,
        const AnnotatedPath::ConstPtr& active_path
    );

    // ─── ROS 通信 ───
    rclcpp::Subscription<interfaces::msg::NavGoal>::SharedPtr goal_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_sub_;
    rclcpp::Subscription<interfaces::msg::CostMaps>::SharedPtr local_cost_maps_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr global_direction_map_sub_;
    rclcpp::Subscription<interfaces::msg::SpinCmd>::SharedPtr spin_cmd_sub_;
    rclcpp::Subscription<interfaces::msg::ChassisStatus>::SharedPtr chassis_status_sub_;
    rclcpp::Subscription<interfaces::msg::CompStage>::SharedPtr comp_stage_sub_;
    rclcpp::Subscription<interfaces::msg::IdleChassisModeOverride>::SharedPtr idle_chassis_mode_override_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_path_pub_;
    rclcpp::Publisher<interfaces::msg::ChassisCmd>::SharedPtr chassis_cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr debug_final_cost_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_spatial_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_smoothed_spatial_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_kino_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_mpc_path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr debug_minco_trajectory_pub_;
    rclcpp::Publisher<interfaces::msg::NavExecutorDiag>::SharedPtr debug_diag_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // ─── 参数 ───
    bool enable_debug_{};
    double remaining_energy_filter_alpha_{};
    double prediction_dt_{};
    double dynamic_prediction_horizon_seconds_{};
    int obstacle_occupied_threshold_{};
    double path_publish_sample_resolution_{};
    double debug_velocity_color_min_{};
    double debug_velocity_color_max_{};
    FollowProjectionGuardParams proj_guard_params_{};
    RouteTrackerParams route_tracker_params_{};
    StepBlockReplanParams step_block_params_{};
    PerformanceReplanParams performance_replan_params_{};

    // ─── 核心组件 ───
    std::unique_ptr<PathPlanner> planner_;
    std::unique_ptr<PathExecutor> executor_;
    std::unique_ptr<TaskManager> task_;
    std::unique_ptr<RouteTracker> route_tracker_;

    // 只读配置（跨线程共享）
    TraversalConfiguration traversal_configuration_;
    CapabilityProfile bidirectional_profile_;
    std::shared_ptr<RouteTerrainMask> route_terrain_mask_;
    bool route_terrain_mask_ready_ = false;

    // ─── 缓存数据 ───
    CostMap::ConstPtr global_cost_map_;
    CostMap::ConstPtr current_cost_map_;              // cost_maps[0]，当前帧动态
    PlannerObstacleView planner_obstacles_;
    std::vector<CostMap::ConstPtr> prediction_maps_;  // cost_maps[1..N]
    DirectionMap::ConstPtr global_direction_map_;

    // ROS 回调缓存槽位（只写不做状态转移）
    std::optional<Goal> pending_goal_;

    ChassisMotionState chassis_state_{};
    uint64_t chassis_state_sequence_ = 0;
    // 底盘状态流 liveness：最后一次有效 ChassisStatus 的接收时刻。
    // 控制 tick 不要求每周期都有新序列号（20Hz 状态与 20Hz 控制同频时会因相位抖动
    // 导致控制率减半），只要求状态流没有断流；断流则停止发令（fail-safe）。
    std::chrono::steady_clock::time_point last_chassis_status_time_{};
    std::chrono::steady_clock::duration chassis_status_timeout_{std::chrono::milliseconds(100)};
    bool chassis_state_valid_ = false;
    uint8_t chassis_leg_mode_ = 4;
    uint8_t comp_stage_ = 4;
    double rfr_pwr_limit_ = 90.0;
    double remaining_energy_supercap_filtered_ = 1400.0;
    double remaining_energy_buffercap_filtered_ = 0.0;
    uint8_t idle_chassis_mode_override_ = 0;

    uint64_t control_cycle_ = 0;
    uint8_t last_input_rejection_reason_ = 0;
    uint64_t input_rejection_count_ = 0;

    enum class SpinState { STOP, SPIN_SLOW, SPIN_FAST } spin_state_ = SpinState::STOP;
    bool spin_high_priority_ = false;

    MotionFeedback previous_motion_feedback_;
};

} // namespace nav_executor
