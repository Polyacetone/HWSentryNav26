#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <interfaces/msg/chassis_status.hpp>
#include <interfaces/msg/chassis_cmd.hpp>
#include <interfaces/msg/spin_cmd.hpp>
#include <interfaces/msg/comp_stage.hpp>
#include <interfaces/msg/global_path.hpp>
#include <interfaces/msg/follower_state.hpp>
#include <interfaces/msg/local_plan.hpp>

#include <common_utils/convert.hpp>
#include <local_planner/nav_map.hpp>
#include <local_planner/utils.hpp>
#include <local_planner/step_routing_mask.hpp>
#include <local_planner/global_path_manager.hpp>
#include <local_planner/state_machine.hpp>
#include <local_planner/mppi_planner.hpp>

namespace local_planner {

class LocalPlannerNode : public rclcpp::Node {
public:
    explicit LocalPlannerNode(const rclcpp::NodeOptions& options);

private:
    // ─── ROS 回调 ───
    void global_path_callback(const interfaces::msg::GlobalPath::SharedPtr msg);
    void chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg);
    void chassis_cmd_callback(const interfaces::msg::ChassisCmd::SharedPtr msg);
    void spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg);
    void control_timer_callback();

    // ─── 工具 ───
    bool get_chassis_pose(Eigen::Vector3d& pose) const;
    void initialize_step_layers();
    void update_step_layers(const std::vector<Eigen::Vector2d>& local_trajectory);
    void refresh_step_layers_from_global_path();
    void update_merged_cost_maps();
    void update_step_flags(bool raw_up, bool raw_down);
    bool check_stuck();
    bool check_in_hazard(const Eigen::Vector2d& robot_pos) const;
    bool is_recovery_safe(const Eigen::Vector2d& robot_pos) const;
    std::optional<Eigen::Vector2d> find_recovery_goal(const Eigen::Vector2d& robot_pos, double robot_theta) const;
    std::vector<Eigen::Vector2d> sample_active_path_for_step_mask() const;
    nav_msgs::msg::Path poses_to_path_msg(const std::vector<Eigen::Vector2d>& pts) const;
    visualization_msgs::msg::MarkerArray rollouts_to_marker_array(const std::vector<std::vector<Eigen::Vector2d>>& rollouts);
    nav_msgs::msg::OccupancyGrid cost_map_to_occupancy_grid(const CostMap& cm) const;
    void clear_debug_visualization();

    // ─── ROS 通信 ───
    rclcpp::Subscription<interfaces::msg::GlobalPath>::SharedPtr global_path_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr local_cost_map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr global_direction_map_sub_;
    rclcpp::Subscription<interfaces::msg::SpinCmd>::SharedPtr spin_cmd_sub_;
    rclcpp::Subscription<interfaces::msg::ChassisCmd>::SharedPtr chassis_cmd_sub_;
    rclcpp::Subscription<interfaces::msg::ChassisStatus>::SharedPtr chassis_status_sub_;
    rclcpp::Subscription<interfaces::msg::CompStage>::SharedPtr comp_stage_sub_;

    rclcpp::Publisher<interfaces::msg::LocalPlan>::SharedPtr local_plan_pub_;
    rclcpp::Publisher<interfaces::msg::FollowerState>::SharedPtr planner_state_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_best_path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_rollouts_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr debug_final_cost_map_pub_;

    rclcpp::TimerBase::SharedPtr control_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // ─── 核心组件 ───
    GlobalPathManager path_manager_;
    std::unique_ptr<LocalPlannerStateMachine> fsm_;
    std::unique_ptr<StepRoutingMask> step_routing_mask_;
    std::unique_ptr<MPPIPlanner> mppi_planner_;

    // ─── 台阶掩码缓存 ───
    CostMap::ConstPtr step_cost_layer_;
    DirectionMap::ConstPtr masked_direction_map_;

    // ─── 地图缓存 ───
    CostMap::ConstPtr global_cost_map_, local_cost_map_;
    CostMap::ConstPtr final_cost_map_;
    CostMap::ConstPtr masked_global_cost_map_; // global + step，不含 local（用于 hazard 判定）
    DirectionMap::ConstPtr global_direction_map_;

    // ─── 底盘状态 ───
    Eigen::Vector2d chassis_status_ = Eigen::Vector2d::Zero();
    uint8_t chassis_leg_mode_ = 4;
    uint8_t comp_stage_ = 4;

    // ─── 小陀螺请求 ───
    enum class SpinState { STOP, SLOW, FAST } spin_state_ = SpinState::STOP;
    bool spin_high_priority_ = false;

    // ─── 路径投影状态 ───
    double last_reference_u_ = 0.0;  // MPPI horizon 末端 u，用于下次 warm start
    double last_robot_u_ = 0.0;      // 机器人实际在路径上的 u，用于到达判定

    // ─── 台阶前瞻检测 ───
    struct StepAheadParams {
        double detect_norm_threshold;
        double detect_dot_threshold;
        int on_count_threshold;
        int off_count_threshold;
    } step_ahead_params_{};
    int step_mask_num_samples_ = 100;
    int step_up_on_ = 0, step_up_off_ = 0;
    int step_down_on_ = 0, step_down_off_ = 0;
    bool step_up_flag_ = false, step_down_flag_ = false;

    // ─── stuck 检测 ───
    StuckParams stuck_params_{};
    bool stuck_active_ = false;
    std::chrono::steady_clock::time_point stuck_start_time_;
    Eigen::Vector2d stuck_start_pos_ = Eigen::Vector2d::Zero();
    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero(); // 来自 mpc_controller 的实际命令速度

    // ─── 下层底盘实际状态（用于 MPPI 初始化）───
    Eigen::Vector2d actual_chassis_vel_ = Eigen::Vector2d::Zero(); // [v_act, ω_act]

    // ─── HAZARD_RECOVERY ───
    double recovery_search_radius_min_ = 0.5;
    double recovery_search_radius_max_ = 3.0;
    double recovery_safe_threshold_ = 80.0; // 代价值低于此认为安全
    double hazard_cost_threshold_ = 200.0;  // 代价值高于此认为处于障碍物中（触发 hazard）
    double recovery_safe_hold_time_ = 0.3;
    struct RecoveryGoalScoreParams {
        double cost_weight = 1.0;
        double distance_weight = 30.0;
        double forward_reward_weight = 50.0;
    } recovery_goal_score_params_{};
    bool recovery_safe_active_ = false;
    std::chrono::steady_clock::time_point recovery_safe_since_;
    std::optional<Eigen::Vector2d> recovery_goal_;

    // ─── 到达判定参数 ───
    double stop_threshold_dist_ = 0.5;
    double stop_threshold_u_ = 0.99;

    double control_dt_ = 0.05;
    bool enable_debug_ = false;
    int debug_max_rollouts_ = 24;
    int last_debug_rollout_marker_count_ = 0;
};

// ═══════════════════════ 构造函数 ════════════════════════════

LocalPlannerNode::LocalPlannerNode(const rclcpp::NodeOptions& options) : Node("local_planner", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    enable_debug_ = declare_parameter<bool>("debug.enable");
    if (enable_debug_) {
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        debug_best_path_pub_ = create_publisher<nav_msgs::msg::Path>(
            declare_parameter<std::string>("debug.best_path_pub_topic"), 1
        );
        debug_rollouts_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            declare_parameter<std::string>("debug.rollouts_pub_topic"), 1
        );
        debug_final_cost_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
            declare_parameter<std::string>("debug.final_cost_map_pub_topic"), 1
        );
        debug_max_rollouts_ = static_cast<int>(declare_parameter<int>("debug.max_rollouts"));
    }

    // ─── FSM 参数 ───
    LocalPlannerFsmParams fsm_params;
    fsm_params.transition = {
        .follow_to_spin_vel_max = declare_parameter<double>("state_machine.follow_to_spin_vel_max"),
        .spin_to_follow_omega_max = declare_parameter<double>("state_machine.spin_to_follow_omega_max"),
        .to_idle_vel_max = declare_parameter<double>("state_machine.to_idle_vel_max"),
        .to_idle_omega_max = declare_parameter<double>("state_machine.to_idle_omega_max"),
        .stopping_timeout = declare_parameter<double>("state_machine.stopping_timeout"),
    };

    stuck_params_ = {
        .enable = declare_parameter<bool>("recovery.stuck.enable"),
        .cmd_vel_threshold = declare_parameter<double>("recovery.stuck.cmd_vel_threshold"),
        .timeout = declare_parameter<double>("recovery.stuck.timeout"),
        .max_displacement = declare_parameter<double>("recovery.stuck.max_displacement"),
        .reverse_speed = declare_parameter<double>("recovery.stuck.reverse_speed"),
        .reverse_duration = declare_parameter<double>("recovery.stuck.reverse_duration"),
    };
    fsm_params.stuck = stuck_params_;

    fsm_ = std::make_unique<LocalPlannerStateMachine>(fsm_params, get_logger());

    // ─── 台阶掩码参数 ───
    StepRoutingMaskParams step_mask_params;
    step_mask_params.path_align_dot_threshold = declare_parameter<double>("step_mask.path_align_dot_threshold");
    step_mask_params.full_effect_radius = declare_parameter<double>("step_mask.full_effect_radius");
    step_mask_params.cutoff_radius = declare_parameter<double>("step_mask.cutoff_radius");
    step_mask_num_samples_ = static_cast<int>(declare_parameter<int>("step_mask.length_num_samples"));
    step_mask_params.length_num_samples = step_mask_num_samples_;
    step_routing_mask_ = std::make_unique<StepRoutingMask>(step_mask_params);

    // ─── 台阶前瞻参数 ───
    step_ahead_params_.detect_norm_threshold = declare_parameter<double>("step_ahead_flag.detect_norm_threshold");
    step_ahead_params_.detect_dot_threshold = declare_parameter<double>("step_ahead_flag.detect_dot_threshold");
    step_ahead_params_.on_count_threshold = static_cast<int>(declare_parameter<int>("step_ahead_flag.on_count_threshold"));
    step_ahead_params_.off_count_threshold = static_cast<int>(declare_parameter<int>("step_ahead_flag.off_count_threshold"));

    // ─── MPPI 参数 ───
    {
        MPPIParams mp;
        mp.num_rollouts       = static_cast<int>(declare_parameter<int>("mppi.num_rollouts"));
        mp.horizon            = static_cast<int>(declare_parameter<int>("mppi.horizon"));
        mp.dt                 = declare_parameter<double>("mppi.dt");
        mp.Av                 = declare_parameter<double>("mppi.Av");
        mp.A22                = declare_parameter<double>("mppi.A22");
        mp.A24                = declare_parameter<double>("mppi.A24");
        mp.v_cmd_max          = declare_parameter<double>("mppi.v_cmd_max");
        mp.v_cmd_min          = declare_parameter<double>("mppi.v_cmd_min");
        mp.omega_cmd_max      = declare_parameter<double>("mppi.omega_cmd_max");
        mp.omega_cmd_min      = declare_parameter<double>("mppi.omega_cmd_min");
        mp.noise_sigma_v      = declare_parameter<double>("mppi.noise_sigma_v");
        mp.noise_sigma_omega  = declare_parameter<double>("mppi.noise_sigma_omega");
        mp.temperature        = declare_parameter<double>("mppi.temperature");
        mp.w_obstacle         = declare_parameter<double>("mppi.w_obstacle");
        mp.w_path_follow      = declare_parameter<double>("mppi.w_path_follow");
        mp.w_heading          = declare_parameter<double>("mppi.w_heading");
        mp.w_progress         = declare_parameter<double>("mppi.w_progress");
        mp.w_direction        = declare_parameter<double>("mppi.w_direction");
        mp.w_control_v        = declare_parameter<double>("mppi.w_control_v");
        mp.w_control_omega    = declare_parameter<double>("mppi.w_control_omega");
        mp.w_terminal_goal    = declare_parameter<double>("mppi.w_terminal_goal");
        mp.w_control_dv       = declare_parameter<double>("mppi.w_control_dv");
        mp.w_control_domega   = declare_parameter<double>("mppi.w_control_domega");
        mp.step_norm_threshold = declare_parameter<double>("mppi.step_norm_threshold");
        mp.proj_num_samples   = static_cast<int>(declare_parameter<int>("mppi.proj_num_samples"));
        mp.proj_search_window = declare_parameter<double>("mppi.proj_search_window");
        mp.proj_lazy_dist     = declare_parameter<double>("mppi.proj_lazy_dist");
        mp.obstacle_threshold = declare_parameter<double>("mppi.obstacle_threshold");
        mp.collision_cost     = declare_parameter<double>("mppi.collision_cost");
        mp.smooth_alpha       = declare_parameter<double>("mppi.smooth_alpha");
        mp.num_iterations     = static_cast<int>(declare_parameter<int>("mppi.num_iterations"));
        mp.num_threads        = static_cast<int>(declare_parameter<int>("mppi.num_threads"));
        mppi_planner_ = std::make_unique<MPPIPlanner>(mp);
    }

    // ─── 到达判定参数 ───
    stop_threshold_dist_ = declare_parameter<double>("misc.stop_threshold_dist");
    stop_threshold_u_ = declare_parameter<double>("misc.stop_threshold_u");
    control_dt_ = declare_parameter<double>("misc.control_dt");

    // ─── HAZARD_RECOVERY 参数 ───
    recovery_search_radius_min_ = declare_parameter<double>("recovery.hazard.search_radius_min");
    recovery_search_radius_max_ = declare_parameter<double>("recovery.hazard.search_radius_max");
    recovery_safe_threshold_ = declare_parameter<double>("recovery.hazard.safe_threshold");
    hazard_cost_threshold_ = declare_parameter<double>("recovery.hazard.hazard_cost_threshold");
    recovery_safe_hold_time_ = declare_parameter<double>("recovery.hazard.safe_hold_time");
    recovery_goal_score_params_.cost_weight = declare_parameter<double>("recovery.hazard.goal_score.cost_weight");
    recovery_goal_score_params_.distance_weight = declare_parameter<double>("recovery.hazard.goal_score.distance_weight");
    recovery_goal_score_params_.forward_reward_weight = declare_parameter<double>("recovery.hazard.goal_score.forward_reward_weight");

    // ─── 订阅 ───
    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("global_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            global_cost_map_ = std::make_shared<CostMap>(*msg);
            RCLCPP_INFO(get_logger(), "Received global cost map: size=(%d,%d)", global_cost_map_->width, global_cost_map_->height);
            initialize_step_layers();
            global_cost_map_sub_.reset();
        }
    );

    global_direction_map_sub_ = create_subscription<sensor_msgs::msg::Image>(
        declare_parameter<std::string>("global_direction_map_sub_topic"), 1,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            if (!global_cost_map_) {
                RCLCPP_WARN(get_logger(), "Received direction map but global cost map is not ready!");
                return;
            }
            cv::Mat img = cv_bridge::toCvShare(msg, "8UC2")->image;
            global_direction_map_ = std::make_shared<DirectionMap>(img, global_cost_map_->resolution, global_cost_map_->origin_x, global_cost_map_->origin_y);
            RCLCPP_INFO(get_logger(), "Received global direction map");
            initialize_step_layers();
            global_direction_map_sub_.reset();
        }
    );

    local_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("local_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            if (!global_cost_map_) return;
            local_cost_map_ = std::make_shared<CostMap>(*msg);
            update_merged_cost_maps();
        }
    );

    global_path_sub_ = create_subscription<interfaces::msg::GlobalPath>(
        declare_parameter<std::string>("global_path_sub_topic"), 1,
        [this](const interfaces::msg::GlobalPath::SharedPtr msg) { global_path_callback(msg); }
    );

    chassis_status_sub_ = create_subscription<interfaces::msg::ChassisStatus>(
        declare_parameter<std::string>("chassis_status_sub_topic"), 1,
        [this](const interfaces::msg::ChassisStatus::SharedPtr msg) { chassis_status_callback(msg); }
    );

    comp_stage_sub_ = create_subscription<interfaces::msg::CompStage>(
        declare_parameter<std::string>("comp_stage_sub_topic"), 1,
        [this](const interfaces::msg::CompStage::SharedPtr msg) { comp_stage_ = msg->game_progress; }
    );

    spin_cmd_sub_ = create_subscription<interfaces::msg::SpinCmd>(
        declare_parameter<std::string>("spin_cmd_sub_topic"), 1,
        [this](const interfaces::msg::SpinCmd::SharedPtr msg) { spin_cmd_callback(msg); }
    );

    chassis_cmd_sub_ = create_subscription<interfaces::msg::ChassisCmd>(
        declare_parameter<std::string>("chassis_cmd_sub_topic"), 1,
        [this](const interfaces::msg::ChassisCmd::SharedPtr msg) { chassis_cmd_callback(msg); }
    );

    // ─── 发布 ───
    local_plan_pub_ = create_publisher<interfaces::msg::LocalPlan>(
        declare_parameter<std::string>("local_plan_pub_topic"), 1
    );
    planner_state_pub_ = create_publisher<interfaces::msg::FollowerState>(
        declare_parameter<std::string>("planner_state_pub_topic"), 1
    );

    control_timer_ = create_wall_timer(
        std::chrono::duration<double>(control_dt_),
        [this]() { control_timer_callback(); }
    );
}

// ═══════════════════ 回调 ════════════════════════════════════

void LocalPlannerNode::global_path_callback(const interfaces::msg::GlobalPath::SharedPtr msg) {
    std::vector<Eigen::Vector2d> cpts;
    cpts.reserve(msg->x.size());
    for (size_t i = 0; i < msg->x.size(); ++i) {
        cpts.emplace_back(static_cast<double>(msg->x[i]), static_cast<double>(msg->y[i]));
    }
    path_manager_.set_path(cpts, msg->fixed);
    refresh_step_layers_from_global_path();
}

void LocalPlannerNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    chassis_status_.x() = msg->velocity;
    chassis_status_.y() = msg->omega;
    chassis_leg_mode_ = msg->leg_mode;
    actual_chassis_vel_.x() = msg->velocity;
    actual_chassis_vel_.y() = msg->omega;
}

void LocalPlannerNode::chassis_cmd_callback(const interfaces::msg::ChassisCmd::SharedPtr msg) {
    last_cmd_.x() = static_cast<double>(msg->velocity);
    last_cmd_.y() = static_cast<double>(msg->omega);
}

void LocalPlannerNode::spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg) {
    switch (msg->spin_mode) {
        case 0: spin_state_ = SpinState::STOP; break;
        case 1: spin_state_ = SpinState::SLOW; break;
        case 2: spin_state_ = SpinState::FAST; break;
        default: RCLCPP_ERROR(get_logger(), "Invalid spin_mode: %d", msg->spin_mode); return;
    }
    spin_high_priority_ = msg->high_priority;
}

// ═══════════════════ 台阶掩码层 ═════════════════════════════

void LocalPlannerNode::initialize_step_layers() {
    if (!global_cost_map_ || !global_direction_map_ || !step_routing_mask_) return;
    if (!step_routing_mask_->ready()) {
        try {
            step_routing_mask_->initialize(*global_cost_map_, global_direction_map_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "StepRoutingMask init failed: %s", e.what());
            return;
        }
    }
    refresh_step_layers_from_global_path();
}

void LocalPlannerNode::update_step_layers(const std::vector<Eigen::Vector2d>& local_trajectory) {
    if (!step_routing_mask_ || !step_routing_mask_->ready()) return;
    step_routing_mask_->update(local_trajectory);
    step_cost_layer_ = step_routing_mask_->step_cost_layer();
    masked_direction_map_ = step_routing_mask_->masked_direction_map();
    update_merged_cost_maps();
}

void LocalPlannerNode::refresh_step_layers_from_global_path() {
    if (!step_routing_mask_ || !step_routing_mask_->ready()) return;
    update_step_layers(sample_active_path_for_step_mask());
}

void LocalPlannerNode::update_merged_cost_maps() {
    if (!global_cost_map_) return;
    try {
        if (step_cost_layer_) {
            auto masked_global = std::make_shared<CostMap>(global_cost_map_->merge(*step_cost_layer_));
            masked_global_cost_map_ = masked_global; // global + step，不含 local
            if (local_cost_map_) {
                final_cost_map_ = std::make_shared<CostMap>(masked_global->merge(*local_cost_map_));
            } else {
                final_cost_map_ = masked_global;
            }
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Cost map merge failed: %s", e.what());
    }
}

void LocalPlannerNode::update_step_flags(bool raw_up, bool raw_down) {
    if (raw_up)  { step_up_on_++;  step_up_off_ = 0;  if (step_up_on_  >= step_ahead_params_.on_count_threshold)  step_up_flag_ = true; }
    else         { step_up_off_++; step_up_on_ = 0;    if (step_up_off_ >= step_ahead_params_.off_count_threshold) step_up_flag_ = false; }
    if (raw_down) { step_down_on_++;  step_down_off_ = 0;  if (step_down_on_  >= step_ahead_params_.on_count_threshold)  step_down_flag_ = true; }
    else          { step_down_off_++; step_down_on_ = 0;    if (step_down_off_ >= step_ahead_params_.off_count_threshold) step_down_flag_ = false; }
}

// ═══════════════════ stuck 检测 ══════════════════════════════

bool LocalPlannerNode::check_stuck() {
    if (!stuck_params_.enable) return false;
    if (std::abs(last_cmd_.x()) < stuck_params_.cmd_vel_threshold) {
        stuck_active_ = false;
        return false;
    }

    Eigen::Vector3d pose;
    if (!get_chassis_pose(pose)) return false;
    const Eigen::Vector2d pos = pose.head<2>();

    if (!stuck_active_) {
        stuck_active_ = true;
        stuck_start_time_ = std::chrono::steady_clock::now();
        stuck_start_pos_ = pos;
        return false;
    }

    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - stuck_start_time_).count();
    const double disp = (pos - stuck_start_pos_).norm();
    if (disp > stuck_params_.max_displacement) {
        stuck_start_time_ = std::chrono::steady_clock::now();
        stuck_start_pos_ = pos;
        return false;
    }
    return dt >= stuck_params_.timeout;
}

// ═══════════════════ 主定时回调 ══════════════════════════════

void LocalPlannerNode::control_timer_callback() {
    if (!global_cost_map_ || !final_cost_map_ || !masked_direction_map_) return;

    Eigen::Vector3d chassis_pose;
    if (!get_chassis_pose(chassis_pose)) return;

    const auto now_tp = std::chrono::steady_clock::now();
    const bool has_path = path_manager_.has_path();
    const bool has_new_path = path_manager_.path_updated();

    // 到达判定（使用机器人实际 u 投影，而非 MPPI forward propagation u）
    bool reach_goal = false;
    if (has_path) {
        const auto& gp = *path_manager_.global_path();
        const double dist_to_end = (chassis_pose.head<2>() - gp.evaluate(1.0)).norm();
        reach_goal = (dist_to_end < stop_threshold_dist_) || (last_robot_u_ > stop_threshold_u_);
    }

    // 检测当前是否处于障碍物中
    const bool in_hazard = check_in_hazard(chassis_pose.head<2>());

    // ─── 组装 FSM 输入 ───
    LocalPlannerFsmInput fsm_in;
    fsm_in.has_path = has_path;
    fsm_in.has_new_path = has_new_path;
    fsm_in.fixed_goal_flag = path_manager_.fixed_goal();
    fsm_in.reach_goal = reach_goal;
    fsm_in.spin_requested = (spin_state_ != SpinState::STOP);
    fsm_in.spin_high_priority = spin_high_priority_;
    fsm_in.is_stuck = check_stuck();
    fsm_in.is_in_hazard = in_hazard;
    fsm_in.velocity = last_cmd_.x();
    fsm_in.omega = last_cmd_.y();
    fsm_in.stamp = now_tp;

    // HAZARD_RECOVERY 输入（已处于 recovery 时用于退出判定）
    if (fsm_->state() == PlannerState::HAZARD_RECOVERY) {
        const bool raw_recovery_safe = is_recovery_safe(chassis_pose.head<2>());
        if (raw_recovery_safe) {
            if (!recovery_safe_active_) {
                recovery_safe_active_ = true;
                recovery_safe_since_ = now_tp;
            }
        } else {
            recovery_safe_active_ = false;
        }
        fsm_in.is_recovery_safe = recovery_safe_active_
            && std::chrono::duration<double>(now_tp - recovery_safe_since_).count() >= recovery_safe_hold_time_;
    }

    // 记录旧状态用于检测状态切换
    const PlannerState prev_state = fsm_->state();

    // ─── FSM 更新 ───
    const auto fsm_out = fsm_->update(fsm_in);
    const PlannerState state = fsm_out.state;

    // 状态发生切换时重置 MPPI warm start
    if (state != prev_state) {
        mppi_planner_->reset();
        if (state == PlannerState::HAZARD_RECOVERY || prev_state == PlannerState::HAZARD_RECOVERY) {
            recovery_safe_active_ = false;
        }
    }

    if (state == PlannerState::HAZARD_RECOVERY) {
        recovery_goal_ = find_recovery_goal(chassis_pose.head<2>(), chassis_pose.z());
    } else {
        recovery_goal_.reset();
    }

    // 处理路径消费
    if (fsm_out.consume_global_path) {
        const bool keep_fixed = (state == PlannerState::HOLD_FIXED);
        path_manager_.consume(keep_fixed);
        last_reference_u_ = 0.0;
        last_robot_u_ = 0.0;
        stuck_active_ = false;
        // 重置台阶检测防抖
        step_up_on_ = step_up_off_ = step_down_on_ = step_down_off_ = 0;
        step_up_flag_ = step_down_flag_ = false;
        recovery_safe_active_ = false;
        // 用空轨迹重置台阶层
        update_step_layers({});
    }

    // path_updated 标记在本周期结束后重置
    path_manager_.reset_update_flag();

    // ─── 构建 LocalPlan 消息 ───
    interfaces::msg::LocalPlan plan_msg;
    plan_msg.consume_global_path = fsm_out.consume_global_path;
    plan_msg.step_up_ahead = false;
    plan_msg.step_down_ahead = false;
    plan_msg.slow_spin = false;
    plan_msg.fast_spin = false;

    switch (state) {
    case PlannerState::IDLE: {
        plan_msg.mode = interfaces::msg::LocalPlan::MODE_INVALID;
        last_cmd_ = Eigen::Vector2d::Zero();
        break;
    }

    case PlannerState::TRACK: {
        if (!path_manager_.has_path()) {
            plan_msg.mode = interfaces::msg::LocalPlan::MODE_STOP;
            break;
        }

        // v2: MPPI 局部轨迹采样
        MPPIInput mppi_in;
        mppi_in.robot_pose         = chassis_pose;
        mppi_in.robot_vel          = actual_chassis_vel_;
        mppi_in.global_path        = &*path_manager_.global_path();
        mppi_in.global_path_u_hint = last_robot_u_;
        mppi_in.final_cost_map     = final_cost_map_.get();
        mppi_in.direction_map      = masked_direction_map_.get();
        mppi_in.collect_debug_rollouts = enable_debug_ && static_cast<bool>(debug_rollouts_pub_);
        mppi_in.max_debug_rollouts = debug_max_rollouts_;
        const auto mppi_start = std::chrono::steady_clock::now();
        const auto mppi_out = mppi_planner_->plan(mppi_in);
        const double plan_total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - mppi_start
        ).count();
        last_robot_u_ = mppi_out.robot_u;
        last_reference_u_ = mppi_out.updated_u_hint;

        // 台阶前瞻（带防抖）
        update_step_flags(mppi_out.step_up_ahead, mppi_out.step_down_ahead);
        plan_msg.step_up_ahead  = step_up_flag_;
        plan_msg.step_down_ahead = step_down_flag_;

        plan_msg.mode = interfaces::msg::LocalPlan::MODE_TRACK;
        plan_msg.x.reserve(mppi_out.best_trajectory.size());
        plan_msg.y.reserve(mppi_out.best_trajectory.size());
        plan_msg.t.reserve(mppi_out.best_timestamps.size());
        for (const auto& pt : mppi_out.best_trajectory) {
            plan_msg.x.push_back(static_cast<float>(pt.x()));
            plan_msg.y.push_back(static_cast<float>(pt.y()));
        }
        plan_msg.t.assign(mppi_out.best_timestamps.begin(), mppi_out.best_timestamps.end());

        if (enable_debug_) {
            if (debug_best_path_pub_) {
                debug_best_path_pub_->publish(poses_to_path_msg(mppi_out.best_trajectory));
            }
            if (debug_rollouts_pub_) {
                debug_rollouts_pub_->publish(rollouts_to_marker_array(mppi_out.debug_rollouts));
            }
            RCLCPP_DEBUG(
                get_logger(),
                "MPPI debug: plan_total=%.2fms internal=%.2fms traj_pts=%lu rollouts=%lu cost[min/mean/max]=[%.3f, %.3f, %.3f] step_flags=[%d,%d]",
                plan_total_ms,
                mppi_out.debug_plan_time_ms,
                static_cast<unsigned long>(mppi_out.best_trajectory.size()),
                static_cast<unsigned long>(mppi_out.debug_rollouts.size()),
                mppi_out.debug_cost_min,
                mppi_out.debug_cost_mean,
                mppi_out.debug_cost_max,
                mppi_out.step_up_ahead ? 1 : 0,
                mppi_out.step_down_ahead ? 1 : 0
            );
        }
        break;
    }

    case PlannerState::SPIN: {
        plan_msg.mode = interfaces::msg::LocalPlan::MODE_SPIN;
        plan_msg.slow_spin = (spin_state_ == SpinState::SLOW);
        plan_msg.fast_spin = (spin_state_ == SpinState::FAST);
        last_cmd_ = Eigen::Vector2d::Zero();
        break;
    }

    case PlannerState::STOP_TRANSITION: {
        plan_msg.mode = interfaces::msg::LocalPlan::MODE_STOP;
        // last_cmd 会由 mpc_controller 的 stop() 减小，这里不更新
        break;
    }

    case PlannerState::HOLD_FIXED: {
        plan_msg.mode = interfaces::msg::LocalPlan::MODE_HOLD_FIXED;
        plan_msg.fixed_x = static_cast<float>(path_manager_.fixed_goal_pos().x());
        plan_msg.fixed_y = static_cast<float>(path_manager_.fixed_goal_pos().y());
        // last_cmd_ 由 chassis_cmd_callback 从 mpc_controller 反馈更新
        break;
    }

    case PlannerState::REVERSE: {
        plan_msg.mode = interfaces::msg::LocalPlan::MODE_REVERSE;
        last_cmd_.x() = -stuck_params_.reverse_speed;
        last_cmd_.y() = 0.0;
        break;
    }

    case PlannerState::HAZARD_RECOVERY: {
        if (recovery_goal_.has_value()) {
            // 发布简单直线路径走向 recovery goal 并附带时间戳
            plan_msg.mode = interfaces::msg::LocalPlan::MODE_TRACK;
            const auto& goal = recovery_goal_.value();
            constexpr int recovery_path_points = 10;
            const double total_dist = (goal - chassis_pose.head<2>()).norm();
            const double recovery_speed = 0.5; // m/s
            const double total_time = std::max(total_dist / recovery_speed, 0.5);
            plan_msg.x.reserve(recovery_path_points);
            plan_msg.y.reserve(recovery_path_points);
            plan_msg.t.reserve(recovery_path_points);
            for (int i = 0; i < recovery_path_points; ++i) {
                const double alpha = static_cast<double>(i) / static_cast<double>(recovery_path_points - 1);
                plan_msg.x.push_back(static_cast<float>(chassis_pose.x() * (1.0 - alpha) + goal.x() * alpha));
                plan_msg.y.push_back(static_cast<float>(chassis_pose.y() * (1.0 - alpha) + goal.y() * alpha));
                plan_msg.t.push_back(static_cast<float>(alpha * total_time));
            }
        } else {
            plan_msg.mode = interfaces::msg::LocalPlan::MODE_STOP;
        }
        break;
    }
    } // switch

    if (enable_debug_ && state != PlannerState::TRACK) {
        clear_debug_visualization();
    }

    // ─── 发布 debug final cost map ───
    if (enable_debug_ && debug_final_cost_map_pub_ && final_cost_map_) {
        debug_final_cost_map_pub_->publish(cost_map_to_occupancy_grid(*final_cost_map_));
    }

    // ─── 发布 ───
    local_plan_pub_->publish(plan_msg);

    interfaces::msg::FollowerState state_msg;
    state_msg.state = static_cast<uint8_t>(state);
    planner_state_pub_->publish(state_msg);
}

// ═══════════════════ 工具函数 ════════════════════════════════

bool LocalPlannerNode::check_in_hazard(const Eigen::Vector2d& robot_pos) const {
    if (!final_cost_map_) return false;
    const Eigen::Vector2d grid = final_cost_map_->map_coord_to_grid(robot_pos);
    const Eigen::Vector2i gi(static_cast<int>(std::floor(grid.x())),
                             static_cast<int>(std::floor(grid.y())));
    if (!final_cost_map_->is_valid_coord(gi)) return true; // 越界视为 hazard
    return static_cast<double>(final_cost_map_->at(gi)) >= hazard_cost_threshold_;
}

bool LocalPlannerNode::is_recovery_safe(const Eigen::Vector2d& robot_pos) const {
    if (!masked_global_cost_map_) return false;

    const Eigen::Vector2d grid = masked_global_cost_map_->map_coord_to_grid(robot_pos);
    const Eigen::Vector2i gi(static_cast<int>(std::floor(grid.x())),
                             static_cast<int>(std::floor(grid.y())));
    if (!masked_global_cost_map_->is_valid_coord(gi)) return false;
    if (static_cast<double>(masked_global_cost_map_->at(gi)) >= recovery_safe_threshold_) {
        return false;
    }

    if (masked_direction_map_ && masked_direction_map_->is_valid_coord(gi)) {
        const Eigen::Vector2d dir = masked_direction_map_->interpolate(grid);
        if (dir.norm() >= step_ahead_params_.detect_norm_threshold) {
            return false;
        }
    }

    return true;
}

std::optional<Eigen::Vector2d> LocalPlannerNode::find_recovery_goal(const Eigen::Vector2d& robot_pos, double robot_theta) const {
    if (!masked_global_cost_map_) return std::nullopt;
    const auto& cm = *masked_global_cost_map_;
    const double res = cm.resolution;

    // 在环形区域内搜索加权最优的安全点（考虑代价、距离和朝向）
    const int r_min_cells = static_cast<int>(std::ceil(recovery_search_radius_min_ / res));
    const int r_max_cells = static_cast<int>(std::ceil(recovery_search_radius_max_ / res));

    const Eigen::Vector2d robot_grid = cm.map_coord_to_grid(robot_pos);
    const int cx = static_cast<int>(std::round(robot_grid.x()));
    const int cy = static_cast<int>(std::round(robot_grid.y()));

    const Eigen::Vector2d heading(std::cos(robot_theta), std::sin(robot_theta));

    double best_score = std::numeric_limits<double>::max();
    Eigen::Vector2i best_cell(-1, -1);

    for (int dy = -r_max_cells; dy <= r_max_cells; ++dy) {
        for (int dx = -r_max_cells; dx <= r_max_cells; ++dx) {
            const int dist_sq = dx * dx + dy * dy;
            if (dist_sq < r_min_cells * r_min_cells || dist_sq > r_max_cells * r_max_cells) continue;
            const Eigen::Vector2i cell(cx + dx, cy + dy);
            if (!cm.is_valid_coord(cell)) continue;
            const double cost = static_cast<double>(cm.at(cell));
            if (cost >= recovery_safe_threshold_) continue; // 不安全的点直接跳过

            // 综合评分：代价 + 距离惩罚 - 朝向奖励
            const double dist = std::sqrt(static_cast<double>(dist_sq)) * res;
            const Eigen::Vector2d to_cell(dx * res, dy * res);
            const double to_cell_norm = to_cell.norm();
            // 朝向一致性：dot ∈ [-1, 1]，越大越好
            const double heading_dot = (to_cell_norm > 1e-6) ? heading.dot(to_cell / to_cell_norm) : 0.0;
            const double score = recovery_goal_score_params_.cost_weight * cost
                + recovery_goal_score_params_.distance_weight * dist
                - recovery_goal_score_params_.forward_reward_weight * std::max(heading_dot, 0.0);
            if (score < best_score) {
                best_score = score;
                best_cell = cell;
            }
        }
    }

    if (best_cell.x() < 0) return std::nullopt;
    return cm.grid_coord_to_map(best_cell.cast<double>());
}

std::vector<Eigen::Vector2d> LocalPlannerNode::sample_active_path_for_step_mask() const {
    if (!path_manager_.has_path()) {
        return {};
    }

    const int num_samples = std::max(2, step_mask_num_samples_);
    std::vector<Eigen::Vector2d> samples;
    samples.reserve(static_cast<size_t>(num_samples));

    const auto& path = *path_manager_.global_path();
    for (int i = 0; i < num_samples; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(num_samples - 1);
        samples.push_back(path.evaluate(u));
    }
    return samples;
}

nav_msgs::msg::OccupancyGrid LocalPlannerNode::cost_map_to_occupancy_grid(const CostMap& cm) const {
    nav_msgs::msg::OccupancyGrid grid;
    grid.header.stamp = now();
    grid.header.frame_id = "map";
    grid.info.width = static_cast<uint32_t>(cm.width);
    grid.info.height = static_cast<uint32_t>(cm.height);
    grid.info.resolution = static_cast<float>(cm.resolution);
    grid.info.origin.position.x = cm.origin_x;
    grid.info.origin.position.y = cm.origin_y;
    grid.data.resize(cm.data.size());
    for (size_t i = 0; i < cm.data.size(); ++i) {
        grid.data[i] = static_cast<int8_t>(std::min(static_cast<int>(cm.data[i] * 100 / 255), 100));
    }
    return grid;
}

nav_msgs::msg::Path LocalPlannerNode::poses_to_path_msg(const std::vector<Eigen::Vector2d>& pts) const {
    nav_msgs::msg::Path msg;
    msg.header.stamp = now();
    msg.header.frame_id = "map";
    msg.poses.reserve(pts.size());
    for (const auto& p : pts) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = msg.header;
        ps.pose.position.x = p.x();
        ps.pose.position.y = p.y();
        msg.poses.push_back(ps);
    }
    return msg;
}

visualization_msgs::msg::MarkerArray LocalPlannerNode::rollouts_to_marker_array(
    const std::vector<std::vector<Eigen::Vector2d>>& rollouts
) {
    visualization_msgs::msg::MarkerArray markers;
    const auto stamp = now();

    for (size_t rollout_idx = 0; rollout_idx < rollouts.size(); ++rollout_idx) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = stamp;
        marker.ns = "mppi_rollouts";
        marker.id = static_cast<int>(rollout_idx);
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.015;
        marker.color.r = 0.10f;
        marker.color.g = 0.80f;
        marker.color.b = 0.95f;
        marker.color.a = 0.28f;
        marker.pose.orientation.w = 1.0;
        marker.points.reserve(rollouts[rollout_idx].size());
        for (const auto& pt : rollouts[rollout_idx]) {
            geometry_msgs::msg::Point p;
            p.x = pt.x();
            p.y = pt.y();
            marker.points.push_back(p);
        }
        markers.markers.push_back(std::move(marker));
    }

    for (int marker_id = static_cast<int>(rollouts.size()); marker_id < last_debug_rollout_marker_count_; ++marker_id) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = stamp;
        marker.ns = "mppi_rollouts";
        marker.id = marker_id;
        marker.action = visualization_msgs::msg::Marker::DELETE;
        markers.markers.push_back(std::move(marker));
    }

    last_debug_rollout_marker_count_ = static_cast<int>(rollouts.size());
    return markers;
}

void LocalPlannerNode::clear_debug_visualization() {
    if (debug_best_path_pub_) {
        debug_best_path_pub_->publish(poses_to_path_msg({}));
    }
    if (debug_rollouts_pub_ && last_debug_rollout_marker_count_ > 0) {
        debug_rollouts_pub_->publish(rollouts_to_marker_array({}));
    }
}

bool LocalPlannerNode::get_chassis_pose(Eigen::Vector3d& chassis_pose) const {
    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = tf_buffer_->lookupTransform("map", "chassis_link", tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(get_logger(), "TF lookup failed: %s", ex.what());
        return false;
    }
    chassis_pose.head<2>() = utils::convert_to<Eigen::Vector3d>(tf.transform.translation).head<2>();
    const Eigen::Quaterniond q = utils::convert_to<Eigen::Quaterniond>(tf.transform.rotation);
    const Eigen::Vector2d x_axis = (q * Eigen::Vector3d::UnitX()).head<2>();
    if (x_axis.norm() < 1e-6) return false;
    chassis_pose.z() = std::atan2(x_axis.y(), x_axis.x());
    return true;
}

} // namespace local_planner

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(local_planner::LocalPlannerNode)