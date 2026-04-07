#include <chrono>
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64.hpp>
#include <interfaces/msg/chassis_status.hpp>
#include <interfaces/msg/spin_cmd.hpp>
#include <interfaces/msg/predicted_cost_maps.hpp>
#include <interfaces/msg/chassis_cmd.hpp>
#include <interfaces/msg/comp_stage.hpp>
#include <interfaces/msg/global_path.hpp>
#include <interfaces/msg/follower_state.hpp>

#include <uniform_bspline/uniform_bspline.hpp>
#include <common_utils/convert.hpp>
#include <path_follower/nav_map.hpp>
#include <path_follower/main_controller.hpp>
#include <path_follower/mpc_solver.hpp>
#include <path_follower/step_routing_mask.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

class PathFollowerNode : public rclcpp::Node {
public:
    explicit PathFollowerNode(const rclcpp::NodeOptions& options);

private:
    // ─── ROS 回调 ───
    void control_points_callback(const interfaces::msg::GlobalPath::SharedPtr msg);
    void chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg);
    void local_cost_map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void predicted_cost_maps_callback(const interfaces::msg::PredictedCostMaps::SharedPtr msg);
    void spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg);
    void control_timer_callback();

    // ─── 工具函数 ───
    bool get_chassis_pose(Eigen::Vector3d& chassis_pose) const;
    std::optional<double> get_map_to_imu_world_yaw() const;
    std::optional<double> get_chassis_theta_imu_world() const;
    void publish_chassis_cmd(const ControlOutput& output);
    nav_msgs::msg::Path path_to_nav_msg(const std::vector<Eigen::Vector2d>& points) const;
    bool should_use_prediction_maps() const;

    // ─── 台阶掩码层更新 ───
    void update_step_layers();
    void update_masked_cost_maps();

    // ─── ROS 通信 ───
    rclcpp::Subscription<interfaces::msg::GlobalPath>::SharedPtr control_points_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr local_cost_map_sub_;
    rclcpp::Subscription<interfaces::msg::PredictedCostMaps>::SharedPtr predicted_cost_maps_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr global_direction_map_sub_;
    rclcpp::Subscription<interfaces::msg::SpinCmd>::SharedPtr spin_cmd_sub_;
    rclcpp::Subscription<interfaces::msg::ChassisStatus>::SharedPtr chassis_status_sub_;
    rclcpp::Subscription<interfaces::msg::CompStage>::SharedPtr comp_stage_sub_;
    rclcpp::Publisher<interfaces::msg::ChassisCmd>::SharedPtr chassis_cmd_pub_;
    rclcpp::Publisher<interfaces::msg::FollowerState>::SharedPtr follower_state_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_predicted_path_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr debug_v_pred_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr debug_w_pred_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr debug_final_cost_map_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // ─── 参数 ───
    bool enable_debug_;
    bool use_predicted_cost_maps_ = true;
    double remaining_energy_filter_alpha_ = 1.0;  // 一阶惯性滤波系数
    double prediction_dt_ = 0.2;                  // 预测步长 (s)

    // ─── 核心组件 ───
    std::unique_ptr<MainController> nav_controller_;

    // ─── 台阶障碍物/方向场掩码模块 ───
    std::unique_ptr<StepRoutingMask> step_routing_mask_;
    CostMap::ConstPtr step_cost_layer_;
    DirectionMap::ConstPtr masked_direction_map_;

    // ─── 缓存数据 ───
    CostMap::ConstPtr global_cost_map_, current_local_cost_map_;
    std::vector<CostMap::ConstPtr> prediction_maps_;           // 逐步预测代价地图（maps[0] 为当前帧）
    std::vector<CostMap::ConstPtr> per_step_final_cost_maps_;  // 逐步融合后的最终代价地图
    CostMap::ConstPtr masked_global_cost_map_, final_cost_map_;
    DirectionMap::ConstPtr global_direction_map_;
    std::optional<SplineD> global_path_;
    bool path_updated_ = false;
    bool fixed_goal_ = false;
    Eigen::Vector2d fixed_goal_pos_ = Eigen::Vector2d::Zero();
    Eigen::Vector2d chassis_status_ = Eigen::Vector2d::Zero();
    uint8_t chassis_leg_mode_ = 4;
    uint8_t comp_stage_ = 4;
    double rfr_pwr_limit_ = 90.0;
    double remaining_energy_filtered_ = 1400.0;
    int64_t current_local_cost_map_stamp_ns_ = 0;
    int64_t prediction_maps_stamp_ns_ = 0;
    enum class SpinState { STOP, SPIN_SLOW, SPIN_FAST } spin_state_ = SpinState::STOP;
    bool spin_high_priority_ = false;
};

// ═══════════════════════ 构造函数 ════════════════════════════

PathFollowerNode::PathFollowerNode(const rclcpp::NodeOptions& options) : Node("path_follower", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    enable_debug_ = declare_parameter<bool>("debug.enable");
    if (enable_debug_) {
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        RCLCPP_DEBUG(get_logger(), "Debug mode enabled");
        debug_predicted_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("debug.predicted_path_pub_topic"), 1);
        debug_v_pred_pub_ = create_publisher<std_msgs::msg::Float64>(declare_parameter<std::string>("debug.v_pred_pub_topic"), 1);
        debug_w_pred_pub_ = create_publisher<std_msgs::msg::Float64>(declare_parameter<std::string>("debug.w_pred_pub_topic"), 1);
        debug_final_cost_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(declare_parameter<std::string>("debug.final_cost_map_pub_topic"), 1);
    }

    // ─── MPC 参数加载 ───
    MPCParams mpc_params = {
        .follow_limits = {
            .vel_max = declare_parameter<double>("mpc.follow_path.limits.vel_max"),
            .vel_min = declare_parameter<double>("mpc.follow_path.limits.vel_min"),
            .omega_max = declare_parameter<double>("mpc.follow_path.limits.omega_max"),
            .omega_min = declare_parameter<double>("mpc.follow_path.limits.omega_min"),
            .start_vel_cmd_act_diff_max = declare_parameter<double>("mpc.follow_path.limits.start_vel_cmd_act_diff_max"),
            .start_omega_cmd_act_diff_max = declare_parameter<double>("mpc.follow_path.limits.start_omega_cmd_act_diff_max"),
            .acc_max = declare_parameter<double>("mpc.follow_path.limits.acc_max"),
            .alpha_max = declare_parameter<double>("mpc.follow_path.limits.alpha_max"),
            .vel_step_up = declare_parameter<double>("mpc.follow_path.limits.vel_step_up"),
            .vel_step_down = declare_parameter<double>("mpc.follow_path.limits.vel_step_down"),
            .a_lat_max = declare_parameter<double>("mpc.follow_path.limits.a_lat_max"),
            .slow_down_deceleration = declare_parameter<double>("mpc.follow_path.limits.slow_down_deceleration"),
            .slow_down_target_vel = declare_parameter<double>("mpc.follow_path.limits.slow_down_target_vel"),
            .slow_down_num_samples = static_cast<int>(declare_parameter<int>("mpc.follow_path.limits.slow_down_num_samples"))
        },
        .follow_weights = {
            .q_y = declare_parameter<double>("mpc.follow_path.weights.q_y"),
            .q_theta = declare_parameter<double>("mpc.follow_path.weights.q_theta"),
            .q_u = declare_parameter<double>("mpc.follow_path.weights.q_u"),
            .q_v_final = declare_parameter<double>("mpc.follow_path.weights.q_v_final"),
            .r_v = declare_parameter<double>("mpc.follow_path.weights.r_v"),
            .r_omega = declare_parameter<double>("mpc.follow_path.weights.r_omega"),
            .r_dv = declare_parameter<double>("mpc.follow_path.weights.r_dv"),
            .r_domega = declare_parameter<double>("mpc.follow_path.weights.r_domega"),
            .acc_limit = declare_parameter<double>("mpc.follow_path.weights.acc_limit"),
            .alpha_limit = declare_parameter<double>("mpc.follow_path.weights.alpha_limit"),
            .lat_acc = declare_parameter<double>("mpc.follow_path.weights.lat_acc"),
            .vel_on_step = declare_parameter<double>("mpc.follow_path.weights.vel_on_step"),
            .obstacle = declare_parameter<double>("mpc.follow_path.weights.obstacle"),
            .direction = declare_parameter<double>("mpc.follow_path.weights.direction")
        },
        .follow_projection = {
            .proj_num_samples = static_cast<int>(declare_parameter<int>("mpc.follow_path.projection.num_samples")),
            .proj_search_window = declare_parameter<double>("mpc.follow_path.projection.search_window"),
            .local_search_lazy_distance = declare_parameter<double>("mpc.follow_path.projection.local_search_lazy_distance")
        },
        .stop_limits = {
            .vel_max = declare_parameter<double>("mpc.stop.limits.vel_max"),
            .omega_max = declare_parameter<double>("mpc.stop.limits.omega_max"),
            .omega_min = declare_parameter<double>("mpc.stop.limits.omega_min"),
            .start_vel_cmd_act_diff_max = declare_parameter<double>("mpc.stop.limits.start_vel_cmd_act_diff_max"),
            .start_omega_cmd_act_diff_max = declare_parameter<double>("mpc.stop.limits.start_omega_cmd_act_diff_max"),
            .acc_max = declare_parameter<double>("mpc.stop.limits.acc_max"),
            .alpha_max = declare_parameter<double>("mpc.stop.limits.alpha_max"),
            .vel_step_up = declare_parameter<double>("mpc.stop.limits.vel_step_up"),
            .vel_step_down = declare_parameter<double>("mpc.stop.limits.vel_step_down"),
            .a_lat_max = declare_parameter<double>("mpc.stop.limits.a_lat_max")
        },
        .stop_weights = {
            .q_v = declare_parameter<double>("mpc.stop.weights.q_v"),
            .q_omega = declare_parameter<double>("mpc.stop.weights.q_omega"),
            .r_dv = declare_parameter<double>("mpc.stop.weights.r_dv"),
            .r_domega = declare_parameter<double>("mpc.stop.weights.r_domega"),
            .acc_limit = declare_parameter<double>("mpc.stop.weights.acc_limit"),
            .alpha_limit = declare_parameter<double>("mpc.stop.weights.alpha_limit"),
            .lat_acc = declare_parameter<double>("mpc.stop.weights.lat_acc"),
            .vel_on_step = declare_parameter<double>("mpc.stop.weights.vel_on_step"),
            .obstacle = declare_parameter<double>("mpc.stop.weights.obstacle"),
            .obstacle_terminal = declare_parameter<double>("mpc.stop.weights.obstacle_terminal"),
            .direction = declare_parameter<double>("mpc.stop.weights.direction"),
            .step_terminal = declare_parameter<double>("mpc.stop.weights.step_terminal")
        },
        .recovery_limits = {
            .vel_max = declare_parameter<double>("mpc.recovery.limits.vel_max"),
            .vel_min = declare_parameter<double>("mpc.recovery.limits.vel_min"),
            .omega_max = declare_parameter<double>("mpc.recovery.limits.omega_max"),
            .omega_min = declare_parameter<double>("mpc.recovery.limits.omega_min"),
            .start_vel_cmd_act_diff_max = declare_parameter<double>("mpc.recovery.limits.start_vel_cmd_act_diff_max"),
            .start_omega_cmd_act_diff_max = declare_parameter<double>("mpc.recovery.limits.start_omega_cmd_act_diff_max"),
            .acc_max = declare_parameter<double>("mpc.recovery.limits.acc_max"),
            .alpha_max = declare_parameter<double>("mpc.recovery.limits.alpha_max"),
            .a_lat_max = declare_parameter<double>("mpc.recovery.limits.a_lat_max")
        },
        .recovery_weights = {
            .q_goal_xy = declare_parameter<double>("mpc.recovery.weights.q_goal_xy"),
            .q_goal_theta = declare_parameter<double>("mpc.recovery.weights.q_goal_theta"),
            .r_v = declare_parameter<double>("mpc.recovery.weights.r_v"),
            .r_omega = declare_parameter<double>("mpc.recovery.weights.r_omega"),
            .r_dv = declare_parameter<double>("mpc.recovery.weights.r_dv"),
            .r_domega = declare_parameter<double>("mpc.recovery.weights.r_domega"),
            .acc_limit = declare_parameter<double>("mpc.recovery.weights.acc_limit"),
            .alpha_limit = declare_parameter<double>("mpc.recovery.weights.alpha_limit"),
            .lat_acc = declare_parameter<double>("mpc.recovery.weights.lat_acc"),
            .obstacle = declare_parameter<double>("mpc.recovery.weights.obstacle"),
            .step = declare_parameter<double>("mpc.recovery.weights.step"),
            .q_goal_xy_terminal = declare_parameter<double>("mpc.recovery.weights.q_goal_xy_terminal"),
            .obstacle_terminal = declare_parameter<double>("mpc.recovery.weights.obstacle_terminal"),
            .step_terminal = declare_parameter<double>("mpc.recovery.weights.step_terminal")
        },
        .fixed_limits = {
            .vel_max = declare_parameter<double>("mpc.fixed.limits.vel_max"),
            .vel_min = declare_parameter<double>("mpc.fixed.limits.vel_min"),
            .omega_max = declare_parameter<double>("mpc.fixed.limits.omega_max"),
            .omega_min = declare_parameter<double>("mpc.fixed.limits.omega_min"),
            .start_vel_cmd_act_diff_max = declare_parameter<double>("mpc.fixed.limits.start_vel_cmd_act_diff_max"),
            .start_omega_cmd_act_diff_max = declare_parameter<double>("mpc.fixed.limits.start_omega_cmd_act_diff_max"),
            .acc_max = declare_parameter<double>("mpc.fixed.limits.acc_max"),
            .alpha_max = declare_parameter<double>("mpc.fixed.limits.alpha_max"),
            .a_lat_max = declare_parameter<double>("mpc.fixed.limits.a_lat_max")
        },
        .fixed_weights = {
            .q_goal_xy = declare_parameter<double>("mpc.fixed.weights.q_goal_xy"),
            .q_goal_theta = declare_parameter<double>("mpc.fixed.weights.q_goal_theta"),
            .goal_deadzone = declare_parameter<double>("mpc.fixed.weights.goal_deadzone"),
            .r_v = declare_parameter<double>("mpc.fixed.weights.r_v"),
            .r_omega = declare_parameter<double>("mpc.fixed.weights.r_omega"),
            .r_dv = declare_parameter<double>("mpc.fixed.weights.r_dv"),
            .r_domega = declare_parameter<double>("mpc.fixed.weights.r_domega"),
            .acc_limit = declare_parameter<double>("mpc.fixed.weights.acc_limit"),
            .alpha_limit = declare_parameter<double>("mpc.fixed.weights.alpha_limit"),
            .lat_acc = declare_parameter<double>("mpc.fixed.weights.lat_acc"),
            .obstacle = declare_parameter<double>("mpc.fixed.weights.obstacle"),
            .step = declare_parameter<double>("mpc.fixed.weights.step"),
            .q_goal_xy_terminal = declare_parameter<double>("mpc.fixed.weights.q_goal_xy_terminal"),
            .obstacle_terminal = declare_parameter<double>("mpc.fixed.weights.obstacle_terminal"),
            .step_terminal = declare_parameter<double>("mpc.fixed.weights.step_terminal")
        },
        .energy = {
            .enable = declare_parameter<bool>("mpc.energy.enable"),
            .threshold = declare_parameter<double>("mpc.energy.threshold"),
            .weight = declare_parameter<double>("mpc.energy.weight"),
            .softplus_beta = declare_parameter<double>("mpc.energy.softplus_beta")
        },
        .mh_params = {
            .enable = declare_parameter<bool>("mpc.multi_hypothesis.enable"),
            .lateral_offset = declare_parameter<double>("mpc.multi_hypothesis.lateral_offset"),
            .target_ey_penalty = declare_parameter<double>("mpc.multi_hypothesis.target_ey_penalty")
        }
    };

    auto mpc_controller = std::make_shared<MPCSolver>(mpc_params);

    // ─── FSM 参数 ───
    FsmParams fsm_params;
    fsm_params.transition = {
        .follow_to_spin_vel_max = declare_parameter<double>("state_machine.follow_to_spin_vel_max"),
        .spin_to_follow_omega_max = declare_parameter<double>("state_machine.spin_to_follow_omega_max"),
        .to_idle_vel_max = declare_parameter<double>("state_machine.to_idle_vel_max"),
        .to_idle_omega_max = declare_parameter<double>("state_machine.to_idle_omega_max"),
        .stopping_timeout = declare_parameter<double>("state_machine.stopping_timeout")
    };
    fsm_params.recovery = {
        .enable = declare_parameter<bool>("recovery.enable"),
        .hazard_cost_threshold = declare_parameter<double>("recovery.hazard.cost_threshold"),
        .hazard_step_norm_threshold = declare_parameter<double>("recovery.hazard.step_norm_threshold"),
        .safe_cost_threshold = declare_parameter<double>("recovery.safe.cost_threshold"),
        .safe_step_norm_threshold = declare_parameter<double>("recovery.safe.step_norm_threshold"),
        .radius_min = declare_parameter<double>("recovery.search.radius_min"),
        .radius_max = declare_parameter<double>("recovery.search.radius_max"),
        .radius_samples = static_cast<int>(declare_parameter<int>("recovery.search.radius_samples")),
        .angle_samples = static_cast<int>(declare_parameter<int>("recovery.search.angle_samples")),
        .path_integral_resolution = declare_parameter<double>("recovery.search.path_integral_resolution"),
        .safe_hold_time = declare_parameter<double>("recovery.exit.safe_hold_time"),
        .goal_timeout = declare_parameter<double>("recovery.search.goal_timeout")
    };
    fsm_params.stuck = {
        .enable = declare_parameter<bool>("recovery.stuck.enable"),
        .cmd_vel_threshold = declare_parameter<double>("recovery.stuck.cmd_vel_threshold"),
        .timeout = declare_parameter<double>("recovery.stuck.timeout"),
        .max_displacement = declare_parameter<double>("recovery.stuck.max_displacement"),
        .reverse_speed = declare_parameter<double>("recovery.stuck.reverse_speed"),
        .reverse_duration = declare_parameter<double>("recovery.stuck.reverse_duration")
    };

    // ─── MainController 参数 ───
    NavigationParams nav_params;
    nav_params.stop_threshold_dist = declare_parameter<double>("misc.stop_threshold_dist");
    nav_params.stop_threshold_u = declare_parameter<double>("misc.stop_threshold_u");

    nav_params.follow_proj_dist_max = declare_parameter<double>("follow_proj_guard.proj_dist_max");
    nav_params.follow_proj_cost_max = declare_parameter<double>("follow_proj_guard.proj_cost_max");
    nav_params.follow_proj_cost_samples = static_cast<int>(declare_parameter<int>("follow_proj_guard.proj_cost_samples"));
    nav_params.step_detect_norm_threshold = declare_parameter<double>("step_ahead_flag.detect_norm_threshold");
    nav_params.step_detect_dot_threshold = declare_parameter<double>("step_ahead_flag.detect_dot_threshold");
    nav_params.step_on_count_threshold = static_cast<int>(declare_parameter<int>("step_ahead_flag.on_count_threshold"));
    nav_params.step_off_count_threshold = static_cast<int>(declare_parameter<int>("step_ahead_flag.off_count_threshold"));

    nav_params.step_up_failsafe_enable = declare_parameter<bool>("step_up_failsafe.enable");
    nav_params.step_up_failsafe_similar_attempts = static_cast<int>(declare_parameter<int>("step_up_failsafe.similar_attempts"));
    nav_params.step_up_failsafe_similar_dist = declare_parameter<double>("step_up_failsafe.similar_dist");

    // ─── 创建 MainController ───
    nav_controller_ = std::make_unique<MainController>(nav_params, fsm_params, mpc_controller, get_logger());

    // ─── 读取台阶路径掩码相关参数 ───
    StepRoutingMaskParams step_params;
    step_params.path_align_dot_threshold = declare_parameter<double>("step_mask.path_align_dot_threshold");
    step_params.full_effect_radius = declare_parameter<double>("step_mask.full_effect_radius");
    step_params.cutoff_radius = declare_parameter<double>("step_mask.cutoff_radius");
    step_params.length_num_samples = static_cast<int>(declare_parameter<int>("step_mask.length_num_samples"));
    step_routing_mask_ = std::make_unique<StepRoutingMask>(step_params);

    // ─── 功率限制滤波参数 ───
    remaining_energy_filter_alpha_ = declare_parameter<double>("misc.remaining_energy_filter_alpha");
    use_predicted_cost_maps_ = declare_parameter<bool>("prediction.use_predicted_cost_maps");

    // ─── 订阅 / 发布 ───
    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("global_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            global_cost_map_ = std::make_shared<CostMap>(*msg);
            RCLCPP_INFO(
                get_logger(), "Received global cost map: size=(%d,%d), resolution=%.2f",
                global_cost_map_->width, global_cost_map_->height, global_cost_map_->resolution
            );
            // 如果方向场已到达，则初始化台阶掩码模块并生成输出层
            update_step_layers();
            global_cost_map_sub_.reset();
        }
    );

    global_direction_map_sub_ = create_subscription<sensor_msgs::msg::Image>(
        declare_parameter<std::string>("global_direction_map_sub_topic"), 1,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            if (!global_cost_map_) {
                RCLCPP_WARN(get_logger(), "Received direction map but global cost map is not ready yet!");
                return;
            }
            cv::Mat img = cv_bridge::toCvShare(msg, "8UC2")->image;
            global_direction_map_ = std::make_shared<DirectionMap>(img, global_cost_map_->resolution, global_cost_map_->origin_x, global_cost_map_->origin_y);
            if (global_direction_map_->width != global_cost_map_->width || global_direction_map_->height != global_cost_map_->height) {
                RCLCPP_FATAL(
                    get_logger(), "Direction map size (%d,%d) does not match cost map (%d,%d)!",
                    global_direction_map_->width, global_direction_map_->height,
                    global_cost_map_->width, global_cost_map_->height
                );
                throw std::runtime_error("Direction map size does not match cost map size");
            }
            RCLCPP_INFO(get_logger(), "Received global direction map");
            // 初始化台阶掩码模块并生成输出层
            update_step_layers();
            global_direction_map_sub_.reset();
        }
    );

    local_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("local_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) { local_cost_map_callback(msg); }
    );

    predicted_cost_maps_sub_ = create_subscription<interfaces::msg::PredictedCostMaps>(
        declare_parameter<std::string>("predicted_cost_maps_sub_topic"), 1,
        [this](const interfaces::msg::PredictedCostMaps::SharedPtr msg) { predicted_cost_maps_callback(msg); }
    );

    control_points_sub_ = create_subscription<interfaces::msg::GlobalPath>(
        declare_parameter<std::string>("control_points_sub_topic"), 1,
        [this](const interfaces::msg::GlobalPath::SharedPtr msg) { control_points_callback(msg); }
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

    chassis_cmd_pub_ = create_publisher<interfaces::msg::ChassisCmd>(declare_parameter<std::string>("chassis_cmd_pub_topic"), 1);
    follower_state_pub_ = create_publisher<interfaces::msg::FollowerState>(declare_parameter<std::string>("follower_state_pub_topic"), 1);
    control_timer_ = create_wall_timer(std::chrono::duration<double>(MPC_DT), [this]() { control_timer_callback(); });
}

// ═══════════════════════ ROS 回调 ════════════════════════════

void PathFollowerNode::control_points_callback(const interfaces::msg::GlobalPath::SharedPtr msg) {
    if (msg->x.size() < 3) {
        if (!msg->x.empty()) {
            RCLCPP_WARN(get_logger(), "Received insufficient control points (%zu), need at least 3!", msg->x.size());
        }
        global_path_ = std::nullopt;
        // 空路径取消 fixed 目标（除非当前已在 FIXED 模式中且目标是 fixed 的空路径，保持 fixed_goal_）
        if (!msg->fixed) {
            fixed_goal_ = false;
        }
        update_step_layers();
        return;
    }
    if (msg->x.size() != msg->y.size()) {
        RCLCPP_ERROR(get_logger(), "GlobalPath x/y size mismatch (%zu vs %zu)!", msg->x.size(), msg->y.size());
        return;
    }
    std::vector<Eigen::Vector2d> cpts;
    cpts.reserve(msg->x.size());
    for (size_t i = 0; i < msg->x.size(); ++i) {
        cpts.emplace_back(static_cast<double>(msg->x[i]), static_cast<double>(msg->y[i]));
    }
    global_path_ = SplineD(cpts);
    global_path_->setExtrapolate(true);
    path_updated_ = true;

    // 更新 fixed 目标信息
    fixed_goal_ = msg->fixed;
    if (fixed_goal_) {
        // fixed 目标位置为路径末端
        fixed_goal_pos_ = global_path_->evaluate(1.0);
        RCLCPP_INFO(get_logger(), "Received fixed goal path, target: (%.2f, %.2f)", fixed_goal_pos_.x(), fixed_goal_pos_.y());
    }

    // 新路径到来，基于方向场擦除对应台阶
    update_step_layers();
}

void PathFollowerNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    chassis_status_.x() = msg->velocity;
    chassis_status_.y() = msg->omega;
    chassis_leg_mode_ = msg->leg_mode;
    rfr_pwr_limit_ = static_cast<double>(msg->rfr_pwr_limit);
    remaining_energy_filtered_ = remaining_energy_filter_alpha_ * static_cast<double>(msg->remaining_energy_supercap) + (1.0 - remaining_energy_filter_alpha_) * remaining_energy_filtered_;
}

void PathFollowerNode::spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg) {
    switch (msg->spin_mode) {
        case 0: spin_state_ = SpinState::STOP; break;
        case 1: spin_state_ = SpinState::SPIN_SLOW; break;
        case 2: spin_state_ = SpinState::SPIN_FAST; break;
        default: RCLCPP_ERROR(get_logger(), "Received invalid spin state: %d", msg->spin_mode); return;
    }
    spin_high_priority_ = msg->high_priority;
}

void PathFollowerNode::local_cost_map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    if (!global_cost_map_) return;
    current_local_cost_map_ = std::make_shared<CostMap>(*msg);
    current_local_cost_map_stamp_ns_ = rclcpp::Time(msg->header.stamp).nanoseconds();
    update_masked_cost_maps();
}

void PathFollowerNode::predicted_cost_maps_callback(const interfaces::msg::PredictedCostMaps::SharedPtr msg) {
    if (!global_cost_map_ || msg->maps.empty()) return;
    const int w = global_cost_map_->width;
    const int h = global_cost_map_->height;
    const auto total = static_cast<size_t>(w * h);

    prediction_dt_ = msg->prediction_dt;
    prediction_maps_stamp_ns_ = rclcpp::Time(msg->maps.front().header.stamp).nanoseconds();
    prediction_maps_.clear();
    for (const auto& map : msg->maps) {
        if (map.data.size() != total) continue;
        std::vector<uint8_t> data(total);
        for (size_t j = 0; j < total; j++) {
            data[j] = static_cast<uint8_t>(map.data[j]);
        }
        prediction_maps_.push_back(std::make_shared<CostMap>(
            w, h, global_cost_map_->resolution,
            global_cost_map_->origin_x, global_cost_map_->origin_y,
            data
        ));
    }
    update_masked_cost_maps();
}

// ═══════════════════ 台阶掩码层更新 ════════════════════════════════

bool PathFollowerNode::should_use_prediction_maps() const {
    if (!use_predicted_cost_maps_ || prediction_maps_.empty()) {
        return false;
    }
    if (!current_local_cost_map_ || current_local_cost_map_stamp_ns_ == 0 || prediction_maps_stamp_ns_ == 0) {
        return true;
    }
    return prediction_maps_stamp_ns_ == current_local_cost_map_stamp_ns_;
}

void PathFollowerNode::update_step_layers() {
    if (!global_cost_map_ || !global_direction_map_ || !step_routing_mask_) return;

    if (!step_routing_mask_->ready()) {
        try {
            step_routing_mask_->initialize(*global_cost_map_, global_direction_map_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize StepRoutingMask: %s", e.what());
            return;
        }
    }

    step_routing_mask_->update(global_path_);
    step_cost_layer_ = step_routing_mask_->step_cost_layer();
    masked_direction_map_ = step_routing_mask_->masked_direction_map();
    update_masked_cost_maps();
}

void PathFollowerNode::update_masked_cost_maps() {
    if (!global_cost_map_) return;

    try {
        if (step_cost_layer_) {
            masked_global_cost_map_ = std::make_shared<CostMap>(global_cost_map_->merge(*step_cost_layer_));
        } else {
            masked_global_cost_map_ = global_cost_map_;
        }

        per_step_final_cost_maps_.clear();
        if (should_use_prediction_maps()) {
            for (const auto& pred_map : prediction_maps_) {
                per_step_final_cost_maps_.push_back(
                    std::make_shared<CostMap>(masked_global_cost_map_->merge(*pred_map))
                );
            }
            if (!per_step_final_cost_maps_.empty()) {
                final_cost_map_ = per_step_final_cost_maps_[0];
                return;
            }
        }

        if (current_local_cost_map_) {
            final_cost_map_ = std::make_shared<CostMap>(masked_global_cost_map_->merge(*current_local_cost_map_));
        } else {
            final_cost_map_ = masked_global_cost_map_;
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to merge cost maps: %s", e.what());
    }
}

// ═══════════════════ 控制主循环 ══════════════════════════════

void PathFollowerNode::control_timer_callback() {
    if (!global_cost_map_ || !final_cost_map_ || !masked_direction_map_) return;

    if (enable_debug_) {
        nav_msgs::msg::OccupancyGrid grid_msg;
        grid_msg.header.stamp = now();
        grid_msg.header.frame_id = "map";
        grid_msg.info.width = static_cast<uint32_t>(final_cost_map_->width);
        grid_msg.info.height = static_cast<uint32_t>(final_cost_map_->height);
        grid_msg.info.resolution = static_cast<float>(final_cost_map_->resolution);
        grid_msg.info.origin.position.x = final_cost_map_->origin_x;
        grid_msg.info.origin.position.y = final_cost_map_->origin_y;
        grid_msg.data.resize(final_cost_map_->data.size());
        for (size_t idx = 0; idx < final_cost_map_->data.size(); idx++) {
            grid_msg.data[idx] = static_cast<int8_t>(final_cost_map_->data[idx]);
        }
        debug_final_cost_map_pub_->publish(grid_msg);
    }

    Eigen::Vector3d chassis_pose_map;
    if (!get_chassis_pose(chassis_pose_map)) return;

    // 构建逐步代价地图指针数组
    std::vector<const CostMap*> per_step_ptrs;
    for (const auto& m : per_step_final_cost_maps_) {
        per_step_ptrs.push_back(m.get());
    }

    // 组装控制输入
    const ControlInput input = {
        .global_path = global_path_,
        .path_updated = path_updated_,
        .fixed_goal = fixed_goal_,
        .fixed_goal_pos = fixed_goal_pos_,
        .chassis_pose_map = chassis_pose_map,
        .chassis_status = chassis_status_,
        .remaining_energy = remaining_energy_filtered_,
        .rfr_pwr_limit = rfr_pwr_limit_,
        .chassis_leg_mode = chassis_leg_mode_,
        .comp_stage = comp_stage_,
        .spin_requested = (spin_state_ != SpinState::STOP),
        .spin_high_priority = spin_high_priority_,
        .spin_slow = (spin_state_ == SpinState::SPIN_SLOW),
        .spin_fast = (spin_state_ == SpinState::SPIN_FAST),
        .final_cost_map = final_cost_map_.get(),
        .masked_global_cost_map = masked_global_cost_map_.get(),
        .masked_direction_map = masked_direction_map_.get(),
        .per_step_cost_maps = std::move(per_step_ptrs),
        .prediction_dt = prediction_dt_,
        .stamp = std::chrono::steady_clock::now()
    };

    // 调用控制逻辑层
    const ControlOutput output = nav_controller_->update(input);
    path_updated_ = false;

    interfaces::msg::FollowerState state_msg;
    state_msg.state = static_cast<uint8_t>(output.fsm_state);
    follower_state_pub_->publish(state_msg);

    // 处理路径消费请求：清空 spline 路径，并据此重建台阶擦除地图。
    // 若当前已进入 FIXED，只消费路径，不取消 fixed 目标标记。
    if (output.consume_global_path) {
        global_path_ = std::nullopt;
        // 非 fixed 模式才清除 fixed 标志
        if (output.fsm_state != FsmState::FIXED) {
            fixed_goal_ = false;
        }
        update_step_layers();
    }

    // 发布指令
    if (output.valid) {
        publish_chassis_cmd(output);
        if (enable_debug_) {
            if (output.predicted_path_map) {
                debug_predicted_path_pub_->publish(path_to_nav_msg(*output.predicted_path_map));
            }
            if (output.predicted_v && output.predicted_w) {
                std_msgs::msg::Float64 v_msg, w_msg;
                v_msg.data = (*output.predicted_v)[0];
                w_msg.data = (*output.predicted_w)[0];
                debug_v_pred_pub_->publish(v_msg);
                debug_w_pred_pub_->publish(w_msg);
            }
        }
    }
}

// ═══════════════════ 工具函数 ════════════════════════════════

void PathFollowerNode::publish_chassis_cmd(const ControlOutput& output) {
    interfaces::msg::ChassisCmd msg;
    msg.velocity = static_cast<float>(output.velocity);
    msg.omega = static_cast<float>(output.omega);
    msg.step_up_ahead = output.step_up_ahead;
    msg.step_down_ahead = output.step_down_ahead;
    msg.slow_spin = output.slow_spin;
    msg.fast_spin = output.fast_spin;
    chassis_cmd_pub_->publish(msg);
}

bool PathFollowerNode::get_chassis_pose(Eigen::Vector3d& chassis_pose) const {
    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = tf_buffer_->lookupTransform("map", "chassis_link", tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(get_logger(), "Could not transform chassis_link to map: %s", ex.what());
        return false;
    }
    chassis_pose.head<2>() = utils::convert_to<Eigen::Vector3d>(tf.transform.translation).head<2>();
    const Eigen::Quaterniond q = utils::convert_to<Eigen::Quaterniond>(tf.transform.rotation);
    const Eigen::Vector2d x_axis = (q * Eigen::Vector3d::UnitX()).head<2>();
    if (x_axis.norm() < 1e-6) {
        RCLCPP_ERROR(get_logger(), "Invalid chassis_link orientation");
        return false;
    }
    chassis_pose.z() = atan2(x_axis.y(), x_axis.x());
    return true;
}

nav_msgs::msg::Path PathFollowerNode::path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const {
    nav_msgs::msg::Path msg;
    msg.header.stamp = now();
    msg.header.frame_id = "map";
    for (const auto& p : path) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = msg.header;
        ps.pose.position.x = p.x();
        ps.pose.position.y = p.y();
        ps.pose.position.z = 0.0;
        msg.poses.push_back(ps);
    }
    return msg;
}

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(path_follower::PathFollowerNode)