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
#include <path_follower/control_fsm.hpp>
#include <path_follower/mpc_controller.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {
class PathFollowerNode: public rclcpp::Node {
public:
    explicit PathFollowerNode(const rclcpp::NodeOptions& options);

private:
    void control_points_callback(const nav_msgs::msg::Path::SharedPtr msg);
    void chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg);
    void local_cost_map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg);
    void control_timer_callback();
    void handle_stop_state();
    void handle_follow_state();
    bool get_current_pose(Eigen::Vector3d& current_pose) const;
    std::optional<double> get_map_to_imu_world_yaw() const;
    std::optional<double> get_current_phi_imu_world() const;
    void publish_chassis_cmd(double velocity, double phi, double palstance, bool step_up_ahead, bool step_down_ahead, bool slow_spin, bool fast_spin) const;
    nav_msgs::msg::Path path_to_nav_msg(const std::vector<Eigen::Vector2d>& points) const;
    std::tuple<bool, bool> detect_steps_on_spline(const Eigen::Vector3d& current_pose_map, const double u0);

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

    double stop_threshold_;
    bool enable_debug_;
    double step_check_back_;
    double step_check_front_;
    double step_check_sample_step_;
    MPCParams params_;
    std::unique_ptr<MPCController> mpc_controller_;

    CostMap::ConstPtr global_cost_map_, local_cost_map_, merged_cost_map_;
    DirectionMap::ConstPtr global_direction_map_;
    enum class SpinState { STOP, SPIN_SLOW, SPIN_FAST } spin_state_ = SpinState::STOP;
    bool spin_high_priority_ = false;

    std::unique_ptr<ControlFsm> control_fsm_;
    ControlFsm::State last_fsm_state_ = ControlFsm::State::IDLE; // 用于打印日志

    Eigen::Vector2d current_status_ = Eigen::Vector2d::Zero();

    // 最近一次发布的 phi 指令（imu_world系，[-pi, pi]），用于在 TF 短暂不可用时保持角度不跳 0
    double last_phi_cmd_ = 0.0;
    bool has_last_phi_cmd_ = false;

    std::optional<SplineD> global_path_;
    double last_reference_u_ = 0.0;
};

PathFollowerNode::PathFollowerNode(const rclcpp::NodeOptions& options): Node("path_follower", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    stop_threshold_ = declare_parameter<double>("stop_threshold");
    enable_debug_ = declare_parameter<bool>("debug.enable");
    if (enable_debug_) {
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        RCLCPP_DEBUG(get_logger(), "Debug mode enabled");
        debug_predicted_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("debug.predicted_path_pub_topic"), 1);
    }

    const auto load_matrix = [this](const std::string& name, int rows, int cols) {
        const auto data = declare_parameter<std::vector<double>>(name);
        if (data.size() != rows * cols) {
            RCLCPP_FATAL(get_logger(), "Parameter %s size %zu != %d", name.c_str(), data.size(), rows * cols);
            throw std::runtime_error("Invalid matrix size");
        }
        Eigen::MatrixXd mat(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                mat(r, c) = data[r * cols + c];
            }
        }
        return mat;
    };

    const int horizon = (int)declare_parameter<int>("mpc.general.horizon");
    const double mpc_dt = declare_parameter<double>("mpc.general.dt");
    const int max_iterations = (int)declare_parameter<int>("mpc.general.max_iterations");

    const double lqr_dt = declare_parameter<double>("mpc.lqr.dt");
    const int substeps = (int)declare_parameter<int>("mpc.lqr.substeps");
    if (substeps <= 0) {
        RCLCPP_FATAL(get_logger(), "mpc.lqr.substeps must be > 0, got %d", substeps);
        throw std::runtime_error("Invalid mpc.lqr.substeps");
    }
    const double dt_sub = mpc_dt / substeps;

    const int lqr_steps_per_substep = (lqr_dt > 0.0) ? std::max(1, (int)std::llround(dt_sub / lqr_dt)) : 1;
    const double dt_sub_from_lqr = lqr_steps_per_substep * lqr_dt;
    if (lqr_dt <= 0.0) {
        RCLCPP_WARN(get_logger(), "mpc.lqr.dt <= 0, skip LQR-step bookkeeping");
    } else if (std::abs(dt_sub_from_lqr - dt_sub) > 1e-6) {
        RCLCPP_WARN(
            get_logger(),
            "MPC dt/substeps=%.6f is not an integer multiple of lqr dt=%.6f (nearest %d steps => %.6f). ",
            dt_sub, lqr_dt, lqr_steps_per_substep, dt_sub_from_lqr
        );
    }

    LQRModel lqr_model;
    lqr_model.substeps = substeps;
    lqr_model.dt_sub = dt_sub;
    const Eigen::Matrix<double, 10, 10> A = load_matrix("mpc.lqr.A", 10, 10);
    const Eigen::Matrix<double, 10, 4> B = load_matrix("mpc.lqr.B", 10, 4);
    const Eigen::Matrix<double, 4, 10> K = load_matrix("mpc.lqr.K", 4, 10);
    const Eigen::Matrix<double, 10, 10> Acl = A - B * K;
    const Eigen::Matrix<double, 10, 10> BK = B * K;
    Eigen::Matrix<double, 20, 20> M = Eigen::Matrix<double, 20, 20>::Zero();
    M.block<10, 10>(0, 0) = Acl;
    M.block<10, 10>(0, 10) = BK;
    const Eigen::Matrix<double, 20, 20> Mexp = (M * dt_sub).exp();
    lqr_model.A_cl = Mexp.block<10, 10>(0, 0);
    lqr_model.B_ref = Mexp.block<10, 10>(0, 10);

    params_ = {
        .horizon = horizon,
        .dt = mpc_dt,
        .max_iterations = max_iterations,
        .lqr = lqr_model,
        .follow_limits = {
            .vel_max = declare_parameter<double>("mpc.follow_path.limits.vel_max"),
            .vel_min = declare_parameter<double>("mpc.follow_path.limits.vel_min"),
            .omega_max = declare_parameter<double>("mpc.follow_path.limits.omega_max"),
            .omega_min = declare_parameter<double>("mpc.follow_path.limits.omega_min"),
            .acc_max = declare_parameter<double>("mpc.follow_path.limits.acc_max"),
            .alpha_max = declare_parameter<double>("mpc.follow_path.limits.alpha_max"),
            .phys_acc_max = declare_parameter<double>("mpc.follow_path.limits.phys_acc_max"),
            .phys_alpha_max = declare_parameter<double>("mpc.follow_path.limits.phys_alpha_max"),
            .vel_on_step = declare_parameter<double>("mpc.follow_path.limits.vel_on_step"),
            .a_lat_max = declare_parameter<double>("mpc.follow_path.limits.a_lat_max"),
            .slow_down_distance = declare_parameter<double>("mpc.follow_path.limits.slow_down_distance"),
            .slow_down_num_samples = (int)declare_parameter<int>("mpc.follow_path.limits.slow_down_num_samples"),
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
            .acc_limit_weight = declare_parameter<double>("mpc.follow_path.weights.acc_limit"),
            .alpha_limit_weight = declare_parameter<double>("mpc.follow_path.weights.alpha_limit"),
            .phys_acc_limit_weight = declare_parameter<double>("mpc.follow_path.weights.phys_acc_limit"),
            .phys_alpha_limit_weight = declare_parameter<double>("mpc.follow_path.weights.phys_alpha_limit"),
            .lat_acc_weight = declare_parameter<double>("mpc.follow_path.weights.lat_acc"),
            .vel_on_step_weight = declare_parameter<double>("mpc.follow_path.weights.vel_on_step"),
            .obstacle_weight = declare_parameter<double>("mpc.follow_path.weights.obstacle"),
            .direction_weight = declare_parameter<double>("mpc.follow_path.weights.direction"),
            .step_weight = declare_parameter<double>("mpc.follow_path.weights.step")
        },
        .follow_projection = {
            .proj_num_samples = (int)declare_parameter<int>("mpc.follow_path.projection.num_samples"),
            .proj_search_window = declare_parameter<double>("mpc.follow_path.projection.search_window"),
            .max_correspondence_distance = declare_parameter<double>("mpc.follow_path.projection.max_correspondence_distance")
        },
        .stop_limits = {
            .vel_max = declare_parameter<double>("mpc.stop.limits.vel_max"),
            .omega_max = declare_parameter<double>("mpc.stop.limits.omega_max"),
            .omega_min = declare_parameter<double>("mpc.stop.limits.omega_min"),
            .acc_max = declare_parameter<double>("mpc.stop.limits.acc_max"),
            .alpha_max = declare_parameter<double>("mpc.stop.limits.alpha_max"),
            .phys_acc_max = declare_parameter<double>("mpc.stop.limits.phys_acc_max"),
            .phys_alpha_max = declare_parameter<double>("mpc.stop.limits.phys_alpha_max"),
            .vel_on_step = declare_parameter<double>("mpc.stop.limits.vel_on_step"),
            .a_lat_max = declare_parameter<double>("mpc.stop.limits.a_lat_max")
        },
        .stop_weights = {
            .q_v = declare_parameter<double>("mpc.stop.weights.q_v"),
            .q_omega = declare_parameter<double>("mpc.stop.weights.q_omega"),
            .acc_limit_weight = declare_parameter<double>("mpc.stop.weights.acc_limit"),
            .alpha_limit_weight = declare_parameter<double>("mpc.stop.weights.alpha_limit"),
            .phys_acc_limit_weight = declare_parameter<double>("mpc.stop.weights.phys_acc_limit"),
            .phys_alpha_limit_weight = declare_parameter<double>("mpc.stop.weights.phys_alpha_limit"),
            .lat_acc_weight = declare_parameter<double>("mpc.stop.weights.lat_acc"),
            .vel_on_step_weight = declare_parameter<double>("mpc.stop.weights.vel_on_step"),
            .obstacle_weight = declare_parameter<double>("mpc.stop.weights.obstacle"),
            .obstacle_terminal_weight = declare_parameter<double>("mpc.stop.weights.obstacle_terminal"),
            .direction_weight = declare_parameter<double>("mpc.stop.weights.direction"),
            .step_terminal_weight = declare_parameter<double>("mpc.stop.weights.step_terminal")
        }
    };

    mpc_controller_ = std::make_unique<MPCController>(params_);

    control_fsm_ = std::make_unique<ControlFsm>(ControlFsm::Params{
        .follow_to_spin_vel_max = declare_parameter<double>("control_fsm.follow_to_spin_vel_max"),
        .spin_to_follow_omega_max = declare_parameter<double>("control_fsm.spin_to_follow_omega_max"),
        .to_idle_vel_max = declare_parameter<double>("control_fsm.to_idle_vel_max"),
        .to_idle_omega_max = declare_parameter<double>("control_fsm.to_idle_omega_max")
    });
    last_fsm_state_ = control_fsm_->state();

    step_check_back_ = declare_parameter<double>("step_ahead_flag.window_back");
    step_check_front_ = declare_parameter<double>("step_ahead_flag.window_front");
    step_check_sample_step_ = declare_parameter<double>("step_ahead_flag.sample_step");

    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("global_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            global_cost_map_ = std::make_shared<CostMap>(*msg);
            RCLCPP_INFO(get_logger(), "Received global cost map: size=(%d,%d), resolution=%.2f", global_cost_map_->width, global_cost_map_->height, global_cost_map_->resolution);
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
                RCLCPP_FATAL(get_logger(), "Direction map size (%d,%d) does not match cost map (%d,%d)!", global_direction_map_->width, global_direction_map_->height, global_cost_map_->width, global_cost_map_->height);
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
    control_timer_ = create_wall_timer(std::chrono::duration<double>(params_.dt), [this]() { control_timer_callback(); });
}

void PathFollowerNode::control_points_callback(const nav_msgs::msg::Path::SharedPtr msg) {
    if (msg->poses.size() < 3) {
        if (!msg->poses.empty()) {
            RCLCPP_WARN(get_logger(), "Received insufficient control points (%zu), need at least 3!", msg->poses.size());
        }
        global_path_ = std::nullopt;
        last_reference_u_ = 0.0;
        return;
    }
    std::vector<Eigen::Vector2d> cpts;
    cpts.reserve(msg->poses.size());
    for (const auto& ps: msg->poses) {
        cpts.emplace_back(ps.pose.position.x, ps.pose.position.y);
    }
    global_path_ = SplineD(cpts);
    global_path_->setExtrapolate(true);
    last_reference_u_ = 0.0;
}

void PathFollowerNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    current_status_.x() = msg->velocity;
    current_status_.y() = msg->palstance;
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

void PathFollowerNode::control_timer_callback() {
    // 地图未就绪时也尽量保持 phi 为当前角度，避免 Idle 下 phi 回到 0
    if (!merged_cost_map_ || !global_direction_map_) {
        double hold_phi = has_last_phi_cmd_ ? last_phi_cmd_ : 0.0;
        if (const auto phi_now = get_current_phi_imu_world()) {
            hold_phi = *phi_now;
        }
        publish_chassis_cmd(0.0, hold_phi, 0.0, false, false, false, false);
        return;
    }

    ControlFsm::Inputs fsm_inputs;
    fsm_inputs.has_path = global_path_.has_value();
    fsm_inputs.spin_requested = (spin_state_ != SpinState::STOP);
    fsm_inputs.spin_high_priority = spin_high_priority_;
    fsm_inputs.velocity = current_status_.x();
    fsm_inputs.palstance = current_status_.y();
    control_fsm_->update(fsm_inputs);

    const auto fsm_state = control_fsm_->state();
    if (fsm_state != last_fsm_state_) {
        const auto dest = control_fsm_->destination();
        const char* state_str = (fsm_state == ControlFsm::State::IDLE) ? "IDLE" :
            (fsm_state == ControlFsm::State::FOLLOW) ? "FOLLOW" :
            (fsm_state == ControlFsm::State::SPIN) ? "SPIN" : "STOPPING";
        const char* dest_str = (dest == ControlFsm::Destination::IDLE) ? "IDLE" :
            (dest == ControlFsm::Destination::FOLLOW) ? "FOLLOW" : "SPIN";
        if (fsm_state == ControlFsm::State::STOPPING) {
            RCLCPP_INFO(get_logger(), "Control FSM -> %s (dest=%s)", state_str, dest_str);
        } else {
            RCLCPP_INFO(get_logger(), "Control FSM -> %s", state_str);
        }
        last_fsm_state_ = fsm_state;
    }

    switch (fsm_state) {
        case ControlFsm::State::IDLE: { // 闲置模式：无路径且无需小陀螺
            // Idle也持续发送当前角度（imu_world系），避免电控误以为目标角度为0
            double hold_phi = has_last_phi_cmd_ ? last_phi_cmd_ : 0.0;
            if (const auto phi_now = get_current_phi_imu_world()) {
                hold_phi = *phi_now;
            }
            publish_chassis_cmd(0.0, hold_phi, 0.0, false, false, false, false);
            break;
        }
        case ControlFsm::State::FOLLOW: { // 路径跟随模式
            handle_follow_state();
            break;
        }
        case ControlFsm::State::SPIN: { // 小陀螺模式：速度/角速度由电控负责（通过slow_spin/fast_spin标志）
            const bool slow_spin = (spin_state_ == SpinState::SPIN_SLOW);
            const bool fast_spin = (spin_state_ == SpinState::SPIN_FAST);
            publish_chassis_cmd(0.0, 0.0, 0.0, false, false, slow_spin, fast_spin);
            break;
        }
        case ControlFsm::State::STOPPING: { // 过渡停止模式：用于Follow<->Spin以及路径丢失等场景，保证指令平滑
            handle_stop_state();
            break;
        }
    }
}

void PathFollowerNode::handle_stop_state() {
    Eigen::Vector3d current_pose;
    if (!get_current_pose(current_pose)) return;

    auto start_time = std::chrono::high_resolution_clock::now();

    const RobotStatus current_status = {
        .v = current_status_.x(),
        .omega = current_status_.y()
    };

    const auto result = mpc_controller_->stop(
        current_pose,
        current_status,
        *merged_cost_map_,
        *global_direction_map_
    );
    if (!result) {
        RCLCPP_ERROR(get_logger(), "MPCController(Stop) solve failed: %s", result.error().c_str());
        return;
    }

    const double solve_ms = (std::chrono::high_resolution_clock::now() - start_time).count() / 1e6;
    if (solve_ms > params_.dt * 500.0) {
        RCLCPP_WARN(get_logger(), "MPCController(Stop) solve time %.2f ms > %.2f ms", solve_ms, params_.dt * 500.0);
    } else {
        RCLCPP_DEBUG(get_logger(), "MPCController(Stop) solve time: %.2f ms", solve_ms);
    }

    // 将 map 系下的期望角度转换到 imu_world 系
    double phi_imu_world = 0.0;
    const auto map_to_imu_world_yaw = get_map_to_imu_world_yaw();
    if (map_to_imu_world_yaw) {
        phi_imu_world = result->theta_map - *map_to_imu_world_yaw;
        // 角度归一化到 [-pi, pi]
        phi_imu_world = std::atan2(std::sin(phi_imu_world), std::cos(phi_imu_world));
    }

    publish_chassis_cmd(result->cmd.x(), phi_imu_world, result->cmd.y(), false, false, false, false);
    if (enable_debug_) {
        debug_predicted_path_pub_->publish(path_to_nav_msg(result->predicted_path));
    }
}

void PathFollowerNode::handle_follow_state() {
    Eigen::Vector3d current_pose;
    if (!get_current_pose(current_pose)) return;

    const double u0 = project_to_spline_u(
        *global_path_,
        current_pose.head<2>(),
        last_reference_u_,
        params_.follow_projection.proj_num_samples,
        params_.follow_projection.proj_search_window,
        params_.follow_projection.max_correspondence_distance
    );
    last_reference_u_ = u0;

    auto start_time = std::chrono::high_resolution_clock::now();

    const RobotStatus current_status = {
        .v = current_status_.x(),
        .omega = current_status_.y()
    };

    const auto result = mpc_controller_->follow_path(
        *global_path_,
        current_pose,
        current_status,
        *merged_cost_map_,
        *global_direction_map_
    );
    if (!result) {
        RCLCPP_ERROR(get_logger(), "MPCController(Follow) solve failed: %s", result.error().c_str());
        return;
    }

    const double solve_ms = (std::chrono::high_resolution_clock::now() - start_time).count() / 1e6;
    if (solve_ms > params_.dt * 500.0) {
        RCLCPP_WARN(get_logger(), "MPCController(Follow) solve time %.2f ms > %.2f ms", solve_ms, params_.dt * 500.0);
    } else {
        RCLCPP_DEBUG(get_logger(), "MPCController(Follow) solve time: %.2f ms", solve_ms);
    }

    // 如果已经到达目标点，则清除路径，准备进入闲置状态
    if (u0 >= stop_threshold_) {
        RCLCPP_INFO(get_logger(), "Reached goal, currently at (%.2f, %.2f)", current_pose.x(), current_pose.y());
        global_path_ = std::nullopt;
        last_reference_u_ = 0.0;
    }

    // 将 map 系下的期望角度转换到 imu_world 系
    double phi_imu_world = 0.0;
    const auto map_to_imu_world_yaw = get_map_to_imu_world_yaw();
    if (map_to_imu_world_yaw) {
        phi_imu_world = result->theta_map - *map_to_imu_world_yaw;
        // 角度归一化到 [-pi, pi]
        phi_imu_world = std::atan2(std::sin(phi_imu_world), std::cos(phi_imu_world));
    }

    const auto [step_up_ahead, step_down_ahead] = detect_steps_on_spline(current_pose, u0);
    publish_chassis_cmd(result->cmd.x(), phi_imu_world, result->cmd.y(), step_up_ahead, step_down_ahead, false, false);
    if (enable_debug_) {
        debug_predicted_path_pub_->publish(path_to_nav_msg(result->predicted_path));
    }
}

std::tuple<bool, bool> PathFollowerNode::detect_steps_on_spline(const Eigen::Vector3d& current_pose_map, const double u0) {
    constexpr double dir_norm_threshold = 0.8;
    constexpr double dot_threshold = 0.8;

    if (!global_direction_map_) return {false, false};
    if (!global_path_) return {false, false};
    if (step_check_front_ <= 0.0 && step_check_back_ <= 0.0) return {false, false};
    if (step_check_sample_step_ <= 1e-6) return {false, false};

    const Eigen::Vector2d heading(std::cos(current_pose_map.z()), std::sin(current_pose_map.z()));
    bool step_up = false;
    bool step_down = false;

    const auto sample_at_u = [&](double u) {
        const Eigen::Vector2d p_map = global_path_->evaluate(u);
        const Eigen::Vector2d g = global_direction_map_->map_coord_to_grid(p_map);
        const Eigen::Vector2d dir = global_direction_map_->interpolate(g);
        const double n = dir.norm();
        if (n < dir_norm_threshold) return;
        const double dot = dir.normalized().dot(heading);
        if (dot > dot_threshold) step_up = true;
        if (dot < -dot_threshold) step_down = true;
    };

    sample_at_u(u0);

    // 向前：按弧长近似等间距采样
    double u_fwd = u0;
    double dist_fwd = 0.0;
    const double target_fwd = std::max(0.0, step_check_front_);
    while (dist_fwd + 1e-9 < target_fwd && u_fwd < 1.0 - 1e-9 && !(step_up && step_down)) {
        const Eigen::Vector2d d1 = global_path_->derivative(u_fwd, 1);
        const double dsdu = std::max(1e-6, d1.norm());
        const double du = step_check_sample_step_ / dsdu;
        u_fwd = std::min(1.0, u_fwd + du);
        dist_fwd += step_check_sample_step_;
        sample_at_u(u_fwd);
    }

    // 向后：按弧长近似等间距采样
    double u_bwd = u0;
    double dist_bwd = 0.0;
    const double target_bwd = std::max(0.0, step_check_back_);
    while (dist_bwd + 1e-9 < target_bwd && u_bwd > 1e-9 && !(step_up && step_down)) {
        const Eigen::Vector2d d1 = global_path_->derivative(u_bwd, 1);
        const double dsdu = std::max(1e-6, d1.norm());
        const double du = step_check_sample_step_ / dsdu;
        u_bwd = std::max(0.0, u_bwd - du);
        dist_bwd += step_check_sample_step_;
        sample_at_u(u_bwd);
    }

    return {step_up, step_down};
}

void PathFollowerNode::publish_chassis_cmd(double velocity, double phi, double palstance, bool step_up_ahead, bool step_down_ahead, bool slow_spin, bool fast_spin) const {
    interfaces::msg::ChassisCmd msg;
    msg.velocity = velocity;
    msg.phi = phi;
    msg.palstance = palstance;
    msg.step_up_ahead = step_up_ahead;
    msg.step_down_ahead = step_down_ahead;
    msg.slow_spin = slow_spin;
    msg.fast_spin = fast_spin;
    chassis_cmd_pub_->publish(msg);

    // 仅在非小陀螺模式下更新 last_phi（小陀螺期望全 0，不应覆盖缓存）
    if (!slow_spin && !fast_spin) {
        // 归一化到 [-pi, pi]
        const double phi_norm = std::atan2(std::sin(phi), std::cos(phi));
        const_cast<PathFollowerNode*>(this)->last_phi_cmd_ = phi_norm;
        const_cast<PathFollowerNode*>(this)->has_last_phi_cmd_ = true;
    }
}

std::optional<double> PathFollowerNode::get_current_phi_imu_world() const {
    Eigen::Vector3d current_pose_map;
    if (!get_current_pose(current_pose_map)) {
        return std::nullopt;
    }

    const auto map_to_imu_world_yaw = get_map_to_imu_world_yaw();
    if (!map_to_imu_world_yaw) {
        return std::nullopt;
    }

    double phi_imu_world = current_pose_map.z() - *map_to_imu_world_yaw;
    phi_imu_world = std::atan2(std::sin(phi_imu_world), std::cos(phi_imu_world));
    return phi_imu_world;
}

bool PathFollowerNode::get_current_pose(Eigen::Vector3d& current_pose) const {
    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = tf_buffer_->lookupTransform("map", "chassis_link", tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(get_logger(), "Could not transform chassis_link to map: %s", ex.what());
        return false;
    }

    current_pose.head<2>() = utils::convert_to<Eigen::Vector3d>(tf.transform.translation).head<2>();
    const Eigen::Quaterniond q = utils::convert_to<Eigen::Quaterniond>(tf.transform.rotation);
    const Eigen::Vector2d x_axis = (q * Eigen::Vector3d::UnitX()).head<2>();
    if (x_axis.norm() < 1e-6) {
        RCLCPP_ERROR(get_logger(), "Invalid chassis_link orientation");
        return false;
    }
    current_pose.z() = atan2(x_axis.y(), x_axis.x());
    return true;
}

std::optional<double> PathFollowerNode::get_map_to_imu_world_yaw() const {
    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = tf_buffer_->lookupTransform("imu_world", "map", tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(get_logger(), "Could not transform map to imu_world: %s", ex.what());
        return std::nullopt;
    }

    // 提取 map 到 imu_world 的 yaw 角
    const Eigen::Quaterniond q = utils::convert_to<Eigen::Quaterniond>(tf.transform.rotation);
    const Eigen::Vector2d x_axis = (q * Eigen::Vector3d::UnitX()).head<2>();
    if (x_axis.norm() < 1e-6) {
        RCLCPP_WARN(get_logger(), "Invalid map to imu_world orientation");
        return std::nullopt;
    }
    return std::atan2(x_axis.y(), x_axis.x());
}

nav_msgs::msg::Path PathFollowerNode::path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const {
    nav_msgs::msg::Path msg;
    msg.header.stamp = now();
    msg.header.frame_id = "map";
    for (const auto& p: path) {
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