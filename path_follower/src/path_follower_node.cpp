#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <interfaces/msg/chassis_status.hpp>
#include <interfaces/msg/spin_cmd.hpp>
#include <interfaces/msg/chassis_cmd.hpp>

#include <uniform_bspline/uniform_bspline.hpp>
#include <common_utils/convert.hpp>
#include <path_follower/nav_map.hpp>
#include <path_follower/navigation_controller.hpp>
#include <path_follower/mpc_controller.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

class PathFollowerNode : public rclcpp::Node {
public:
    explicit PathFollowerNode(const rclcpp::NodeOptions& options);

private:
    // ─── ROS 回调 ───
    void control_points_callback(const nav_msgs::msg::Path::SharedPtr msg);
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

    // ─── ROS 通信 ───
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr control_points_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr local_cost_map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr global_direction_map_sub_;
    rclcpp::Subscription<interfaces::msg::SpinCmd>::SharedPtr spin_cmd_sub_;
    rclcpp::Subscription<interfaces::msg::ChassisStatus>::SharedPtr chassis_status_sub_;
    rclcpp::Publisher<interfaces::msg::ChassisCmd>::SharedPtr chassis_cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_predicted_path_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // ─── 参数 ───
    bool enable_debug_;

    // ─── 核心组件 ───
    std::unique_ptr<NavigationController> nav_controller_;

    // ─── 缓存数据 ───
    CostMap::ConstPtr global_cost_map_, local_cost_map_, merged_cost_map_;
    DirectionMap::ConstPtr global_direction_map_;
    std::optional<SplineD> global_path_;
    Eigen::Vector2d chassis_status_ = Eigen::Vector2d::Zero();
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
    }

    // ─── MPC 参数加载 ───
    const int horizon = static_cast<int>(declare_parameter<int>("mpc.general.horizon"));
    const double mpc_dt = declare_parameter<double>("mpc.general.dt");
    const int max_iterations = static_cast<int>(declare_parameter<int>("mpc.general.max_iterations"));

    const std::string prediction_model_str = declare_parameter<std::string>("mpc.general.model");
    PredictionModel prediction_model;
    if (prediction_model_str == "NONE") {
        prediction_model = PredictionModel::NONE;
    } else if (prediction_model_str == "LAG") {
        prediction_model = PredictionModel::LAG;
    } else if (prediction_model_str == "LQR") {
        prediction_model = PredictionModel::LQR;
    } else {
        RCLCPP_FATAL(get_logger(), "Invalid mpc.general.model: %s", prediction_model_str.c_str());
        throw std::runtime_error("Invalid mpc.general.model");
    }

    LagModel lag_model;
    lag_model.tau_v = declare_parameter<double>("mpc.general.lag.tau_v");
    lag_model.k_v = declare_parameter<double>("mpc.general.lag.k_v");
    lag_model.tau_omega = declare_parameter<double>("mpc.general.lag.tau_omega");

    LQRModel lqr_model;
    const double lqr_dt = declare_parameter<double>("mpc.general.lqr.dt");
    const int substeps = static_cast<int>(declare_parameter<int>("mpc.general.lqr.substeps"));
    if (substeps <= 0) {
        RCLCPP_FATAL(get_logger(), "mpc.general.lqr.substeps must be > 0, got %d", substeps);
        throw std::runtime_error("Invalid mpc.general.lqr.substeps");
    }
    const double dt_sub = mpc_dt / substeps;
    const int lqr_steps_per_substep = std::max<int>(1, static_cast<int>(std::lround(dt_sub / lqr_dt)));
    const double dt_sub_from_lqr = lqr_steps_per_substep * lqr_dt;
    if (std::abs(dt_sub_from_lqr - dt_sub) > 1e-6) {
        RCLCPP_WARN(
            get_logger(), "MPC dt/substeps=%.6f is not an integer multiple of lqr dt=%.6f (nearest %d steps => %.6f)",
            dt_sub, lqr_dt, lqr_steps_per_substep, dt_sub_from_lqr
        );
    }

    const auto load_matrix = [this](const std::string& name, int rows, int cols) {
        const auto data = declare_parameter<std::vector<double>>(name);
        if (data.size() != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
            RCLCPP_FATAL(get_logger(), "Parameter %s size %zu != %d", name.c_str(), data.size(), rows * cols);
            throw std::runtime_error("Invalid matrix size");
        }
        Eigen::MatrixXd mat(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                mat(r, c) = data[static_cast<size_t>(r * cols + c)];
            }
        }
        return mat;
    };

    lqr_model.substeps = substeps;
    lqr_model.dt_sub = dt_sub;
    const Eigen::Matrix<double, 10, 10> A = load_matrix("mpc.general.lqr.A", 10, 10);
    const Eigen::Matrix<double, 10, 4> B = load_matrix("mpc.general.lqr.B", 10, 4);
    const Eigen::Matrix<double, 4, 10> K = load_matrix("mpc.general.lqr.K", 4, 10);
    const Eigen::Matrix<double, 10, 10> Acl = A - B * K;
    const Eigen::Matrix<double, 10, 10> BK = B * K;
    Eigen::Matrix<double, 20, 20> M = Eigen::Matrix<double, 20, 20>::Zero();
    M.block<10, 10>(0, 0) = Acl;
    M.block<10, 10>(0, 10) = BK;
    const Eigen::Matrix<double, 20, 20> Mexp = (M * dt_sub).exp();
    lqr_model.A_cl = Mexp.block<10, 10>(0, 0);
    lqr_model.B_ref = Mexp.block<10, 10>(0, 10);

    MPCParams mpc_params = {
        .horizon = horizon,
        .dt = mpc_dt,
        .max_iterations = max_iterations,
        .prediction_model = prediction_model,
        .lag = lag_model,
        .lqr = lqr_model,
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
            .slow_down_num_samples = static_cast<int>(declare_parameter<int>("mpc.follow_path.limits.slow_down_num_samples")),
            .step_norm_threshold = declare_parameter<double>("mpc.follow_path.limits.step_norm_threshold"),
            .step_norm_transition = declare_parameter<double>("mpc.follow_path.limits.step_norm_transition")
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
            .direction = declare_parameter<double>("mpc.follow_path.weights.direction"),
            .step = declare_parameter<double>("mpc.follow_path.weights.step")
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
        }
    };

    auto mpc_controller = std::make_shared<MPCController>(mpc_params);

    // ─── FSM 参数 ───
    FsmParams fsm_params;
    fsm_params.transition = {
        .follow_to_spin_vel_max = declare_parameter<double>("control_fsm.follow_to_spin_vel_max"),
        .spin_to_follow_omega_max = declare_parameter<double>("control_fsm.spin_to_follow_omega_max"),
        .to_idle_vel_max = declare_parameter<double>("control_fsm.to_idle_vel_max"),
        .to_idle_omega_max = declare_parameter<double>("control_fsm.to_idle_omega_max"),
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
        .goal_reached_dist = declare_parameter<double>("recovery.exit.goal_reached_dist"),
        .safe_hold_time = declare_parameter<double>("recovery.exit.safe_hold_time"),
        .goal_timeout = declare_parameter<double>("recovery.search.goal_timeout"),
    };
    fsm_params.stuck = {
        .enable = declare_parameter<bool>("recovery.stuck.enable"),
        .cmd_vel_threshold = declare_parameter<double>("recovery.stuck.cmd_vel_threshold"),
        .timeout = declare_parameter<double>("recovery.stuck.timeout"),
        .max_displacement = declare_parameter<double>("recovery.stuck.max_displacement"),
        .reverse_speed = declare_parameter<double>("recovery.stuck.reverse_speed"),
        .reverse_duration = declare_parameter<double>("recovery.stuck.reverse_duration"),
    };

    // ─── NavigationController 参数 ───
    NavigationParams nav_params;
    nav_params.stop_threshold_dist = declare_parameter<double>("stop_threshold_dist");
    nav_params.stop_threshold_u = declare_parameter<double>("stop_threshold_u");
    nav_params.step_check_back = declare_parameter<double>("step_ahead_flag.window_back");
    nav_params.step_check_front = declare_parameter<double>("step_ahead_flag.window_front");
    nav_params.step_check_sample_step = declare_parameter<double>("step_ahead_flag.sample_step");

    // ─── 创建 NavigationController ───
    nav_controller_ = std::make_unique<NavigationController>(nav_params, fsm_params, mpc_controller, get_logger());

    // ─── 订阅 / 发布 ───
    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("global_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            global_cost_map_ = std::make_shared<CostMap>(*msg);
            RCLCPP_INFO(
                get_logger(), "Received global cost map: size=(%d,%d), resolution=%.2f",
                global_cost_map_->width, global_cost_map_->height, global_cost_map_->resolution
            );
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
            global_direction_map_sub_.reset();
        }
    );

    local_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("local_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) { local_cost_map_callback(msg); }
    );

    control_points_sub_ = create_subscription<nav_msgs::msg::Path>(
        declare_parameter<std::string>("control_points_sub_topic"), 1,
        [this](const nav_msgs::msg::Path::SharedPtr msg) { control_points_callback(msg); }
    );

    chassis_status_sub_ = create_subscription<interfaces::msg::ChassisStatus>(
        declare_parameter<std::string>("chassis_status_sub_topic"), 1,
        [this](const interfaces::msg::ChassisStatus::SharedPtr msg) { chassis_status_callback(msg); }
    );

    spin_cmd_sub_ = create_subscription<interfaces::msg::SpinCmd>(
        declare_parameter<std::string>("spin_cmd_sub_topic"), 1,
        [this](const interfaces::msg::SpinCmd::SharedPtr msg) { spin_cmd_callback(msg); }
    );

    chassis_cmd_pub_ = create_publisher<interfaces::msg::ChassisCmd>(declare_parameter<std::string>("chassis_cmd_pub_topic"), 1);
    control_timer_ = create_wall_timer(std::chrono::duration<double>(mpc_params.dt), [this]() { control_timer_callback(); });
}

// ═══════════════════════ ROS 回调 ════════════════════════════

void PathFollowerNode::control_points_callback(const nav_msgs::msg::Path::SharedPtr msg) {
    if (msg->poses.size() < 3) {
        if (!msg->poses.empty()) {
            RCLCPP_WARN(get_logger(), "Received insufficient control points (%zu), need at least 3!", msg->poses.size());
        }
        global_path_ = std::nullopt;
        return;
    }
    std::vector<Eigen::Vector2d> cpts;
    cpts.reserve(msg->poses.size());
    for (const auto& ps : msg->poses) {
        cpts.emplace_back(ps.pose.position.x, ps.pose.position.y);
    }
    global_path_ = SplineD(cpts);
    global_path_->setExtrapolate(true);
}

void PathFollowerNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    chassis_status_.x() = msg->velocity;
    chassis_status_.y() = msg->omega;
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
    try {
        merged_cost_map_ = std::make_shared<CostMap>(global_cost_map_->merge(*local_cost_map_));
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to merge cost maps: %s", e.what());
    }
}

// ═══════════════════ 控制主循环 ══════════════════════════════

void PathFollowerNode::control_timer_callback() {
    if (!merged_cost_map_ || !global_direction_map_) return;

    Eigen::Vector3d chassis_pose_map;
    if (!get_chassis_pose(chassis_pose_map)) return;

    // 组装控制输入
    ControlInput input;
    input.global_path = global_path_;
    input.chassis_pose_map = chassis_pose_map;
    input.chassis_status = chassis_status_;
    input.spin_requested = (spin_state_ != SpinState::STOP);
    input.spin_high_priority = spin_high_priority_;
    input.spin_slow = (spin_state_ == SpinState::SPIN_SLOW);
    input.spin_fast = (spin_state_ == SpinState::SPIN_FAST);
    input.merged_cost_map = merged_cost_map_.get();
    input.global_direction_map = global_direction_map_.get();
    input.map_to_imu_world_yaw = get_map_to_imu_world_yaw();
    input.chassis_theta_imu_world = get_chassis_theta_imu_world();
    input.stamp = now();

    // 调用控制逻辑层
    const ControlOutput output = nav_controller_->update(input);

    // 处理路径清除请求
    if (output.path_cleared) {
        RCLCPP_INFO(get_logger(), "Path cleared by controller");
        global_path_ = std::nullopt;
    }

    // 发布指令
    if (output.valid) {
        publish_chassis_cmd(output);
        if (enable_debug_ && output.predicted_path_map) {
            debug_predicted_path_pub_->publish(path_to_nav_msg(*output.predicted_path_map));
        }
    }
}

// ═══════════════════ 工具函数 ════════════════════════════════

void PathFollowerNode::publish_chassis_cmd(const ControlOutput& output) {
    interfaces::msg::ChassisCmd msg;
    msg.velocity = static_cast<float>(output.velocity);
    msg.theta = static_cast<float>(output.theta_imu_world);
    msg.omega = static_cast<float>(output.omega);
    msg.step_up_ahead = output.step_up_ahead;
    msg.step_down_ahead = output.step_down_ahead;
    msg.slow_spin = output.slow_spin;
    msg.fast_spin = output.fast_spin;
    chassis_cmd_pub_->publish(msg);
}

std::optional<double> PathFollowerNode::get_chassis_theta_imu_world() const {
    Eigen::Vector3d chassis_pose_map;
    if (!get_chassis_pose(chassis_pose_map)) return std::nullopt;
    const auto yaw = get_map_to_imu_world_yaw();
    if (!yaw) return std::nullopt;
    return std::atan2(std::sin(chassis_pose_map.z() - *yaw), std::cos(chassis_pose_map.z() - *yaw));
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

std::optional<double> PathFollowerNode::get_map_to_imu_world_yaw() const {
    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = tf_buffer_->lookupTransform("imu_world", "map", tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(get_logger(), "Could not transform map to imu_world: %s", ex.what());
        return std::nullopt;
    }
    const Eigen::Quaterniond q = utils::convert_to<Eigen::Quaterniond>(tf.transform.rotation);
    const Eigen::Vector2d x_axis = (q * Eigen::Vector3d::UnitX()).head<2>();
    if (x_axis.norm() < 1e-6) {
        RCLCPP_ERROR(get_logger(), "Invalid map to imu_world orientation");
        return std::nullopt;
    }
    if (x_axis.norm() < 0.8) {
        RCLCPP_WARN(get_logger(), "map to imu_world x-axis norm too small: %.3f", x_axis.norm());
    }
    return std::atan2(x_axis.y(), x_axis.x());
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