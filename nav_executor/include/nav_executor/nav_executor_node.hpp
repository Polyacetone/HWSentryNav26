#pragma once

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
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/float64.hpp>
#include <interfaces/msg/nav_goal.hpp>
#include <interfaces/msg/spin_cmd.hpp>
#include <interfaces/msg/cost_maps.hpp>
#include <interfaces/msg/chassis_cmd.hpp>
#include <interfaces/msg/chassis_status.hpp>
#include <interfaces/msg/comp_stage.hpp>
#include <interfaces/msg/nav_executor_state.hpp>

#include <nav_executor/common/world_context.hpp>
#include <nav_executor/task_manager/task_manager.hpp>
#include <nav_executor/path_executor/path_executor.hpp>
#include <nav_executor/path_executor/step_controller.hpp>
#include <nav_executor/task_manager/route_monitor.hpp>
#include <nav_executor/path_planner/path_planner.hpp>
#include <nav_executor/path_planner/a_star_planner.hpp>
#include <nav_executor/path_planner/bspline_optimizer.hpp>
#include <nav_executor/path_planner/nav_map.hpp>
#include <nav_executor/path_planner/step_routing_mask.hpp>
#include <nav_executor/path_executor/solver/mpc_solver.hpp>

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
    PlannerConfig load_planner_config();
    TaskManagerParams load_task_params();
    ProfileBlendParams load_blend_params();
    BSplineOptimizer::Params load_optimizer_params();
    CapabilityProfile load_capability_profile(const std::string& prefix);

    // ─── ROS 回调 ───
    void chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg);
    void local_cost_maps_callback(const interfaces::msg::CostMaps::SharedPtr msg);
    void spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg);
    void control_tick();

    // ─── 工具 ───
    bool get_chassis_pose(Eigen::Vector3d& chassis_pose) const;
    void try_init_step_mask();
    nav_msgs::msg::Path path_to_nav_msg(const std::vector<Eigen::Vector2d>& points) const;
    void publish_mppi_rollouts(const std::vector<std::vector<Eigen::Vector2d>>& rollouts);
    void publish_diagnostics(const TaskDiagnostics& diag, MotionState motion_state);

    // ─── ROS 通信 ───
    rclcpp::Subscription<interfaces::msg::NavGoal>::SharedPtr goal_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_sub_;
    rclcpp::Subscription<interfaces::msg::CostMaps>::SharedPtr local_cost_maps_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr global_direction_map_sub_;
    rclcpp::Subscription<interfaces::msg::SpinCmd>::SharedPtr spin_cmd_sub_;
    rclcpp::Subscription<interfaces::msg::ChassisStatus>::SharedPtr chassis_status_sub_;
    rclcpp::Subscription<interfaces::msg::CompStage>::SharedPtr comp_stage_sub_;
    rclcpp::Publisher<interfaces::msg::ChassisCmd>::SharedPtr chassis_cmd_pub_;
    rclcpp::Publisher<interfaces::msg::NavExecutorState>::SharedPtr state_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_predicted_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_optimized_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_rough_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_warmup_path_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr debug_v_pred_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr debug_w_pred_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr debug_final_cost_map_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_mppi_rollouts_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // ─── 参数 ───
    bool enable_debug_ = false;
    double remaining_energy_filter_alpha_ = 1.0;
    double prediction_dt_ = 0.2;
    double prediction_horizon_seconds_ = 2.0;
    double prediction_weight_decay_ = 1.0;
    FollowProjectionGuardParams proj_guard_params_{};
    StepBlockReplanParams step_block_params_{};

    // ─── 核心组件 ───
    std::unique_ptr<PathPlanner> planner_;
    std::unique_ptr<PathExecutor> executor_;
    std::unique_ptr<TaskManager> task_;

    // 只读配置（跨线程共享）
    TerrainProfiles terrain_profiles_;
    TerrainRuleTable terrain_rules_{};
    std::shared_ptr<StepRoutingMask> step_routing_mask_;
    bool step_mask_ready_ = false;

    // ─── 缓存数据 ───
    CostMap::ConstPtr global_cost_map_;
    CostMap::ConstPtr current_cost_map_;              // cost_maps[0]，当前帧动态
    CostMap::ConstPtr merged_prediction_cost_map_;    // planner 用：global + 时域融合动态
    std::vector<CostMap::ConstPtr> prediction_maps_;  // cost_maps[1..N]
    DirectionMap::ConstPtr global_direction_map_;

    // ROS 回调缓存槽位（只写不做状态转移）
    std::optional<Goal> pending_goal_;

    ChassisMotionState chassis_state_{};
    uint8_t chassis_leg_mode_ = 4;
    uint8_t comp_stage_ = 4;
    double rfr_pwr_limit_ = 90.0;
    double remaining_energy_filtered_ = 1400.0;

    enum class SpinState { STOP, SPIN_SLOW, SPIN_FAST } spin_state_ = SpinState::STOP;
    bool spin_high_priority_ = false;

    MotionFeedback previous_motion_feedback_;
};

} // namespace nav_executor
