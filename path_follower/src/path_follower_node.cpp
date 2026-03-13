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
#include <interfaces/msg/chassis_cmd.hpp>
#include <interfaces/msg/comp_stage.hpp>
#include <interfaces/msg/global_path.hpp>

#include <uniform_bspline/uniform_bspline.hpp>
#include <common_utils/convert.hpp>
#include <path_follower/nav_map.hpp>
#include <path_follower/main_controller.hpp>
#include <path_follower/mpc_solver.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

class PathFollowerNode : public rclcpp::Node {
public:
    explicit PathFollowerNode(const rclcpp::NodeOptions& options);

private:
    struct StepEraseKernelCell { int dx; int dy; uint8_t delta; };

    // ─── ROS 回调 ───
    void control_points_callback(const interfaces::msg::GlobalPath::SharedPtr msg);
    void chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg);
    void local_cost_map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg);
    void control_timer_callback();

    // ─── 工具函数 ───
    bool get_chassis_pose(Eigen::Vector3d& chassis_pose) const;
    std::optional<double> get_map_to_imu_world_yaw() const;
    std::optional<double> get_chassis_theta_imu_world() const;
    void publish_chassis_cmd(const ControlOutput& output);
    nav_msgs::msg::Path path_to_nav_msg(const std::vector<Eigen::Vector2d>& points) const;

    // ─── 台阶擦除辅助 ───
    void build_step_maps();
    void clear_steps_along_global_path();
    void rebuild_eroded_step_map();
    void update_merged_cost_map();
    double approximate_path_length(const SplineD& spline) const;
    void erase_kernel_at(const Eigen::Vector2i& grid_coord, std::vector<uint8_t>& max_erase_delta) const;

    // ─── ROS 通信 ───
    rclcpp::Subscription<interfaces::msg::GlobalPath>::SharedPtr control_points_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr local_cost_map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr global_direction_map_sub_;
    rclcpp::Subscription<interfaces::msg::SpinCmd>::SharedPtr spin_cmd_sub_;
    rclcpp::Subscription<interfaces::msg::ChassisStatus>::SharedPtr chassis_status_sub_;
    rclcpp::Subscription<interfaces::msg::CompStage>::SharedPtr comp_stage_sub_;
    rclcpp::Publisher<interfaces::msg::ChassisCmd>::SharedPtr chassis_cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_predicted_path_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr debug_v_pred_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr debug_w_pred_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr debug_merged_cost_map_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // ─── 参数 ───
    bool enable_debug_;
    double remaining_energy_filter_alpha_ = 1.0;  // 一阶惯性滤波系数

    // ─── 核心组件 ───
    std::unique_ptr<MainController> nav_controller_;

    // ─── 台阶障碍物相关 ───
    CostMap::Ptr base_step_map_;            // 原始方向场模长生成的台阶地图（不可变）
    CostMap::Ptr eroded_step_map_;          // 根据全局路径擦除后的台阶地图（每次重建）
    std::vector<uint8_t> eroded_step_data_; // 用于构造 eroded_step_map_ 的可变数据
    double step_erase_dot_threshold_;       // 点积阈值, 高于则认为路径经过台阶
    double step_full_erase_radius_;         // 完全擦除半径 (米)
    double step_cutoff_radius_;             // 擦除截止半径 (米)
    int path_length_num_samples_;           // 路径长度估计采样数
    std::vector<StepEraseKernelCell> step_erase_kernel_;

    // ─── 缓存数据 ───
    CostMap::ConstPtr global_cost_map_, local_cost_map_, merged_cost_map_;
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
        debug_merged_cost_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(declare_parameter<std::string>("debug.merged_cost_map_pub_topic"), 1);
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
            .max_correspondence_distance = declare_parameter<double>("mpc.follow_path.projection.max_correspondence_distance")
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
            .keep_steps = static_cast<int>(declare_parameter<int>("mpc.multi_hypothesis.keep_steps")),
            .lateral_offset = declare_parameter<double>("mpc.multi_hypothesis.lateral_offset")
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
        .circ_radius = declare_parameter<double>("recovery.search.circ_radius"),
        .circ_angle_samples = static_cast<int>(declare_parameter<int>("recovery.search.circ_angle_samples")),
        .circ_radius_samples = static_cast<int>(declare_parameter<int>("recovery.search.circ_radius_samples")),
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
    nav_params.step_detect_norm_threshold = declare_parameter<double>("step_ahead_flag.detect_norm_threshold");
    nav_params.step_detect_dot_threshold = declare_parameter<double>("step_ahead_flag.detect_dot_threshold");
    nav_params.step_on_count_threshold = static_cast<int>(declare_parameter<int>("step_ahead_flag.on_count_threshold"));
    nav_params.step_off_count_threshold = static_cast<int>(declare_parameter<int>("step_ahead_flag.off_count_threshold"));

    // ─── 创建 MainController ───
    nav_controller_ = std::make_unique<MainController>(nav_params, fsm_params, mpc_controller, get_logger());

    // ─── 读取台阶擦除相关参数 ───
    step_erase_dot_threshold_ = declare_parameter<double>("step_erase.dot_threshold");
    step_full_erase_radius_ = declare_parameter<double>("step_erase.full_erase_radius");
    step_cutoff_radius_ = declare_parameter<double>("step_erase.cutoff_radius");
    path_length_num_samples_ = static_cast<int>(declare_parameter<int>("step_erase.length_num_samples"));

    // ─── 功率限制滤波参数 ───
    remaining_energy_filter_alpha_ = declare_parameter<double>("misc.remaining_energy_filter_alpha");

    // ─── 订阅 / 发布 ───
    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("global_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            global_cost_map_ = std::make_shared<CostMap>(*msg);
            RCLCPP_INFO(
                get_logger(), "Received global cost map: size=(%d,%d), resolution=%.2f",
                global_cost_map_->width, global_cost_map_->height, global_cost_map_->resolution
            );
            // 如果方向场已经到达，则可以构建台阶图
            if (global_direction_map_) {
                build_step_maps();
                clear_steps_along_global_path();
            }
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
            // 生成台阶障碍物基础图，并根据当前路径擦除
            build_step_maps();
            clear_steps_along_global_path();
            global_direction_map_sub_.reset();
        }
    );

    local_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("local_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) { local_cost_map_callback(msg); }
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
        clear_steps_along_global_path();
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
    clear_steps_along_global_path();
}

void PathFollowerNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    chassis_status_.x() = msg->velocity;
    chassis_status_.y() = msg->omega;
    chassis_leg_mode_ = msg->leg_mode;
    rfr_pwr_limit_ = static_cast<double>(msg->rfr_pwr_limit);
    remaining_energy_filtered_ = remaining_energy_filter_alpha_ * static_cast<double>(msg->remaining_energy) + (1.0 - remaining_energy_filter_alpha_) * remaining_energy_filtered_;
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
    local_cost_map_ = std::make_shared<CostMap>(*msg);
    update_merged_cost_map();
}

// ═══════════════════ 台阶擦除辅助实现 ════════════════════════════════

void PathFollowerNode::rebuild_eroded_step_map() {
    if (!base_step_map_) return;
    eroded_step_map_ = std::make_shared<CostMap>(
        base_step_map_->width,
        base_step_map_->height,
        base_step_map_->resolution,
        base_step_map_->origin_x,
        base_step_map_->origin_y,
        eroded_step_data_
    );
}

void PathFollowerNode::update_merged_cost_map() {
    if (!global_cost_map_) return;

    try {
        const CostMap merged = [&]() -> CostMap {
            if (local_cost_map_ && eroded_step_map_) {
                return global_cost_map_->merge(*local_cost_map_).merge(*eroded_step_map_);
            }
            if (local_cost_map_) {
                return global_cost_map_->merge(*local_cost_map_);
            }
            if (eroded_step_map_) {
                return global_cost_map_->merge(*eroded_step_map_);
            }
            return CostMap(*global_cost_map_);
        }();
        merged_cost_map_ = std::make_shared<CostMap>(std::move(merged));
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to merge cost maps: %s", e.what());
    }
}

// ═══════════════════ 控制主循环 ══════════════════════════════

void PathFollowerNode::control_timer_callback() {
    if (!merged_cost_map_ || !global_direction_map_) return;

    if (enable_debug_) {
        nav_msgs::msg::OccupancyGrid grid_msg;
        grid_msg.header.stamp = now();
        grid_msg.header.frame_id = "map";
        grid_msg.info.width = static_cast<uint32_t>(merged_cost_map_->width);
        grid_msg.info.height = static_cast<uint32_t>(merged_cost_map_->height);
        grid_msg.info.resolution = static_cast<float>(merged_cost_map_->resolution);
        grid_msg.info.origin.position.x = merged_cost_map_->origin_x;
        grid_msg.info.origin.position.y = merged_cost_map_->origin_y;
        grid_msg.data.resize(merged_cost_map_->data.size());
        for (size_t idx = 0; idx < merged_cost_map_->data.size(); idx++) {
            grid_msg.data[idx] = static_cast<int8_t>(merged_cost_map_->data[idx]);
        }
        debug_merged_cost_map_pub_->publish(grid_msg);
    }

    Eigen::Vector3d chassis_pose_map;
    if (!get_chassis_pose(chassis_pose_map)) return;

    // 组装控制输入
    ControlInput input;
    input.global_path = global_path_;
    input.path_updated = path_updated_;
    input.fixed_goal = fixed_goal_;
    input.fixed_goal_pos = fixed_goal_pos_;
    input.chassis_pose_map = chassis_pose_map;
    input.chassis_status = chassis_status_;
    input.chassis_leg_mode = chassis_leg_mode_;
    input.comp_stage = comp_stage_;
    input.spin_requested = (spin_state_ != SpinState::STOP);
    input.spin_high_priority = spin_high_priority_;
    input.spin_slow = (spin_state_ == SpinState::SPIN_SLOW);
    input.spin_fast = (spin_state_ == SpinState::SPIN_FAST);
    input.merged_cost_map = merged_cost_map_.get();
    input.global_direction_map = global_direction_map_.get();
    input.stamp = now();
    input.remaining_energy = remaining_energy_filtered_;
    input.rfr_pwr_limit = rfr_pwr_limit_;

    // 调用控制逻辑层
    const ControlOutput output = nav_controller_->update(input);
    path_updated_ = false;

    // 处理路径消费请求：清空 spline 路径，并据此重建台阶擦除地图。
    // 若当前已进入 FIXED，只消费路径，不取消 fixed 目标标记。
    if (output.consume_global_path) {
        global_path_ = std::nullopt;
        // 非 fixed 模式才清除 fixed 标志
        if (output.fsm_state != FsmState::FIXED) {
            fixed_goal_ = false;
        }
        clear_steps_along_global_path();
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

void PathFollowerNode::build_step_maps() {
    if (!global_direction_map_ || !global_cost_map_) return;

    // 生成基于方向场模的台阶障碍物图
    std::vector<uint8_t> data(static_cast<size_t>(global_direction_map_->width) * static_cast<size_t>(global_direction_map_->height));
    for (size_t idx = 0; idx < global_direction_map_->data.size(); idx++) {
        const double mag = global_direction_map_->data[idx].norm();
        data[idx] = static_cast<uint8_t>(std::clamp(mag * 255.0, 0.0, 255.0));
    }
    base_step_map_ = std::make_shared<CostMap>(
        global_direction_map_->width,
        global_direction_map_->height,
        global_direction_map_->resolution,
        global_direction_map_->origin_x,
        global_direction_map_->origin_y,
        data
    );

    // 初始化可变数据并构建 eroded_step_map_
    eroded_step_data_ = base_step_map_->data;
    rebuild_eroded_step_map();

    // 计算擦除半径对应的栅格数
    const int full_erase_radius = static_cast<int>(std::ceil(step_full_erase_radius_ / global_direction_map_->resolution));
    const int cutoff_radius = static_cast<int>(std::ceil(step_cutoff_radius_ / global_direction_map_->resolution));
    step_erase_kernel_.clear();
    if (cutoff_radius <= 0 || full_erase_radius <= 0) return;

    step_erase_kernel_.reserve(static_cast<size_t>((2 * cutoff_radius + 1) * (2 * cutoff_radius + 1)));
    for (int dy = -cutoff_radius; dy <= cutoff_radius; dy++) {
        for (int dx = -cutoff_radius; dx <= cutoff_radius; dx++) {
            const double dist = std::hypot(static_cast<double>(dx), static_cast<double>(dy));
            if (dist > static_cast<double>(cutoff_radius)) continue;
            const double factor = dist <= static_cast<double>(full_erase_radius) ? 1.0 : (1.0 - (dist - static_cast<double>(full_erase_radius)) / (static_cast<double>(cutoff_radius) - static_cast<double>(full_erase_radius)));
            const auto delta = static_cast<uint8_t>(std::round(255.0 * factor));
            step_erase_kernel_.push_back({dx, dy, delta});
        }
    }
}

void PathFollowerNode::clear_steps_along_global_path() {
    if (!base_step_map_) return;
    eroded_step_data_ = base_step_map_->data;

    if (global_path_ && !step_erase_kernel_.empty()) {
        const double length = approximate_path_length(*global_path_);
        if (length > 0.0) {
            const double sample_spacing = base_step_map_->resolution * 0.5;
            const int samples = std::max(1, static_cast<int>(std::ceil(length / sample_spacing)));
            std::vector<uint8_t> max_erase_delta(base_step_map_->data.size(), 0);
            const auto& spline = *global_path_;
            const auto& direction_map = *global_direction_map_;
            const auto& step_map = *base_step_map_;

            for (int i = 0; i <= samples; i++) {
                const double u = static_cast<double>(i) / static_cast<double>(samples);
                const Eigen::Vector2d pos = spline.evaluate(u);
                Eigen::Vector2d tangent = spline.derivative(u, 1);
                const double tangent_norm = tangent.norm();
                if (tangent_norm < 1e-6) continue;
                tangent /= tangent_norm;
                const Eigen::Vector2d gc_dir = direction_map.map_coord_to_grid(pos);
                if (!direction_map.is_valid_coord(gc_dir)) continue;

                const Eigen::Vector2d dir = direction_map.interpolate(gc_dir);
                const double dot = std::abs(tangent.dot(dir));
                if (dot > step_erase_dot_threshold_) {
                    const Eigen::Vector2d gc_erase = step_map.map_coord_to_grid(pos);
                    const Eigen::Vector2i erase_center{
                        static_cast<int>(std::round(gc_erase.x())),
                        static_cast<int>(std::round(gc_erase.y()))
                    };
                    if (!step_map.is_valid_coord(erase_center)) continue;
                    erase_kernel_at(erase_center, max_erase_delta);
                }
            }

            for (size_t idx = 0; idx < eroded_step_data_.size(); idx++) {
                const uint8_t delta = max_erase_delta[idx];
                if (delta == 0) continue;
                const uint8_t base = base_step_map_->data[idx];
                eroded_step_data_[idx] = base > delta ? static_cast<uint8_t>(base - delta) : 0;
            }
        }
    }

    rebuild_eroded_step_map();
    update_merged_cost_map();
}

double PathFollowerNode::approximate_path_length(const SplineD& spline) const {
    const int samples = std::max(1, path_length_num_samples_);
    double len = 0.0;
    const double du = 1.0 / static_cast<double>(samples);
    for (int i = 0; i < samples; i++) {
        const double u = (static_cast<double>(i) + 0.5) * du;
        len += spline.derivative(u, 1).norm() * du;
    }
    return len;
}

void PathFollowerNode::erase_kernel_at(const Eigen::Vector2i& grid_coord, std::vector<uint8_t>& max_erase_delta) const {
    if (!base_step_map_ || step_erase_kernel_.empty()) return;
    const int width = base_step_map_->width;
    const int height = base_step_map_->height;
    const int cx = grid_coord.x();
    const int cy = grid_coord.y();
    for (const auto& cell : step_erase_kernel_) {
        const int x = cx + cell.dx;
        const int y = cy + cell.dy;
        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
        max_erase_delta[idx] = std::max(max_erase_delta[idx], cell.delta);
    }
}

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(path_follower::PathFollowerNode)