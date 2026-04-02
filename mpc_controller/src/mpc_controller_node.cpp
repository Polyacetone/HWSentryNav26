#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>

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
#include <std_msgs/msg/float64.hpp>

#include <interfaces/msg/chassis_cmd.hpp>
#include <interfaces/msg/chassis_status.hpp>
#include <interfaces/msg/comp_stage.hpp>
#include <interfaces/msg/local_plan.hpp>

#include <common_utils/convert.hpp>
#include <mpc_controller/mpc_solver.hpp>
#include <mpc_controller/nav_map.hpp>
#include <mpc_controller/utils.hpp>

namespace mpc_controller {

namespace {

inline bool is_chassis_dead(uint8_t leg_mode, uint8_t comp_stage) {
    return (leg_mode != 2u && leg_mode != 4u) || comp_stage != 4u;
}

} // anonymous namespace

class MpcControllerNode : public rclcpp::Node {
public:
    explicit MpcControllerNode(const rclcpp::NodeOptions& options);

private:
    void local_plan_callback(const interfaces::msg::LocalPlan::SharedPtr msg);
    void chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg);
    void control_timer_callback();
    bool get_chassis_pose(Eigen::Vector3d& pose) const;
    nav_msgs::msg::Path poses_to_path_msg(const std::vector<Eigen::Vector2d>& pts) const;

    // ─── ROS 通信 ───
    rclcpp::Subscription<interfaces::msg::LocalPlan>::SharedPtr local_plan_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr local_cost_map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr direction_map_sub_;
    rclcpp::Subscription<interfaces::msg::ChassisStatus>::SharedPtr chassis_status_sub_;
    rclcpp::Subscription<interfaces::msg::CompStage>::SharedPtr comp_stage_sub_;
    rclcpp::Publisher<interfaces::msg::ChassisCmd>::SharedPtr chassis_cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_predicted_path_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr debug_v_pred_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr debug_w_pred_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // ─── 核心组件 ───
    std::shared_ptr<MPCSolver> mpc_solver_;

    // ─── 缓存数据 ───
    CostMap::ConstPtr global_cost_map_, local_cost_map_, final_cost_map_;
    DirectionMap::ConstPtr direction_map_;
    CostMap::ConstPtr step_cost_layer_;

    // 最新 LocalPlan
    interfaces::msg::LocalPlan::SharedPtr last_plan_;

    // ─── 状态 ───
    Eigen::Vector2d chassis_status_ = Eigen::Vector2d::Zero();
    uint8_t chassis_leg_mode_ = 4;
    uint8_t comp_stage_ = 4;
    double rfr_pwr_limit_ = 90.0;
    double remaining_energy_filtered_ = 1400.0;
    double remaining_energy_filter_alpha_ = 0.5;

    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero();
    bool enable_debug_ = false;
};

// ═══════════════════════ 构造函数 ════════════════════════════

MpcControllerNode::MpcControllerNode(const rclcpp::NodeOptions& options) : Node("mpc_controller", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    enable_debug_ = declare_parameter<bool>("debug.enable");
    if (enable_debug_) {
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        debug_predicted_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("debug.predicted_path_pub_topic"), 1);
        debug_v_pred_pub_ = create_publisher<std_msgs::msg::Float64>(declare_parameter<std::string>("debug.v_pred_pub_topic"), 1);
        debug_w_pred_pub_ = create_publisher<std_msgs::msg::Float64>(declare_parameter<std::string>("debug.w_pred_pub_topic"), 1);
    }

    // ─── MPC 参数 ───
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
            .slow_down_num_samples = static_cast<int>(declare_parameter<int>("mpc.follow_path.limits.slow_down_num_samples")),
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
        },
        .follow_projection = {
            .proj_num_samples = static_cast<int>(declare_parameter<int>("mpc.follow_path.projection.num_samples")),
            .proj_search_window = declare_parameter<double>("mpc.follow_path.projection.search_window"),
            .local_search_lazy_distance = declare_parameter<double>("mpc.follow_path.projection.local_search_lazy_distance"),
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
            .a_lat_max = declare_parameter<double>("mpc.stop.limits.a_lat_max"),
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
            .step_terminal = declare_parameter<double>("mpc.stop.weights.step_terminal"),
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
            .a_lat_max = declare_parameter<double>("mpc.recovery.limits.a_lat_max"),
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
            .step_terminal = declare_parameter<double>("mpc.recovery.weights.step_terminal"),
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
            .a_lat_max = declare_parameter<double>("mpc.fixed.limits.a_lat_max"),
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
            .step_terminal = declare_parameter<double>("mpc.fixed.weights.step_terminal"),
        },
        .energy = {
            .enable = declare_parameter<bool>("mpc.energy.enable"),
            .threshold = declare_parameter<double>("mpc.energy.threshold"),
            .weight = declare_parameter<double>("mpc.energy.weight"),
            .softplus_beta = declare_parameter<double>("mpc.energy.softplus_beta"),
        },
        .mh_params = {
            .enable = declare_parameter<bool>("mpc.multi_hypothesis.enable"),
            .keep_steps = static_cast<int>(declare_parameter<int>("mpc.multi_hypothesis.keep_steps")),
            .lateral_offset = declare_parameter<double>("mpc.multi_hypothesis.lateral_offset"),
        },
    };
    mpc_solver_ = std::make_shared<MPCSolver>(mpc_params);

    remaining_energy_filter_alpha_ = declare_parameter<double>("misc.remaining_energy_filter_alpha");

    // ─── 订阅 ───
    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("global_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            global_cost_map_ = std::make_shared<CostMap>(*msg);
            RCLCPP_INFO(get_logger(), "Received global cost map");
            global_cost_map_sub_.reset();
        }
    );

    // mpc_controller 仍需要 local_cost_map 做 MPC 避障（merge 到 final）
    local_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("local_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            if (!global_cost_map_) return;
            local_cost_map_ = std::make_shared<CostMap>(*msg);
            if (global_cost_map_) {
                try {
                    final_cost_map_ = std::make_shared<CostMap>(global_cost_map_->merge(*local_cost_map_));
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(get_logger(), "Cost map merge error: %s", e.what());
                }
            }
        }
    );

    direction_map_sub_ = create_subscription<sensor_msgs::msg::Image>(
        declare_parameter<std::string>("direction_map_sub_topic"), 1,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            if (!global_cost_map_) return;
            cv::Mat img = cv_bridge::toCvShare(msg, "8UC2")->image;
            direction_map_ = std::make_shared<DirectionMap>(img, global_cost_map_->resolution, global_cost_map_->origin_x, global_cost_map_->origin_y);
            RCLCPP_INFO(get_logger(), "Received direction map");
            direction_map_sub_.reset();
        }
    );

    local_plan_sub_ = create_subscription<interfaces::msg::LocalPlan>(
        declare_parameter<std::string>("local_plan_sub_topic"), 1,
        [this](const interfaces::msg::LocalPlan::SharedPtr msg) { local_plan_callback(msg); }
    );

    chassis_status_sub_ = create_subscription<interfaces::msg::ChassisStatus>(
        declare_parameter<std::string>("chassis_status_sub_topic"), 1,
        [this](const interfaces::msg::ChassisStatus::SharedPtr msg) { chassis_status_callback(msg); }
    );

    comp_stage_sub_ = create_subscription<interfaces::msg::CompStage>(
        declare_parameter<std::string>("comp_stage_sub_topic"), 1,
        [this](const interfaces::msg::CompStage::SharedPtr msg) { comp_stage_ = msg->game_progress; }
    );

    chassis_cmd_pub_ = create_publisher<interfaces::msg::ChassisCmd>(
        declare_parameter<std::string>("chassis_cmd_pub_topic"), 1
    );

    control_timer_ = create_wall_timer(
        std::chrono::duration<double>(MPC_DT),
        [this]() { control_timer_callback(); }
    );
}

// ═══════════════════════ 回调 ════════════════════════════════

void MpcControllerNode::local_plan_callback(const interfaces::msg::LocalPlan::SharedPtr msg) {
    last_plan_ = msg;
}

void MpcControllerNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    chassis_status_.x() = msg->velocity;
    chassis_status_.y() = msg->omega;
    chassis_leg_mode_ = msg->leg_mode;
    rfr_pwr_limit_ = static_cast<double>(msg->rfr_pwr_limit);
    remaining_energy_filtered_ = remaining_energy_filter_alpha_ * static_cast<double>(msg->remaining_energy_supercap) +
                                 (1.0 - remaining_energy_filter_alpha_) * remaining_energy_filtered_;
}

// ═══════════════════════ 主控制循环 ══════════════════════════

void MpcControllerNode::control_timer_callback() {
    // Dead 底盘全局拦截（§3.1）
    if (is_chassis_dead(chassis_leg_mode_, comp_stage_)) {
        interfaces::msg::ChassisCmd cmd;
        cmd.velocity = 0.0f;
        cmd.omega = 0.0f;
        chassis_cmd_pub_->publish(cmd);
        last_cmd_ = Eigen::Vector2d::Zero();
        mpc_solver_->set_last_cmd(last_cmd_);
        mpc_solver_->reset_warm_start();
        return;
    }

    if (!last_plan_ || !global_cost_map_ || !direction_map_) return;

    Eigen::Vector3d chassis_pose;
    if (!get_chassis_pose(chassis_pose)) return;

    // 更新观测器和能量状态
    mpc_solver_->set_last_cmd(last_cmd_);
    mpc_solver_->update_observer(chassis_status_.x(), chassis_status_.y());
    mpc_solver_->set_energy_state(remaining_energy_filtered_, rfr_pwr_limit_);

    // 确定使用的 cost_map（优先用 final = global+local）
    const CostMap* cost_map_ptr = final_cost_map_ ? final_cost_map_.get() : global_cost_map_.get();
    const DirectionMap* dir_map_ptr = direction_map_.get();

    const auto& plan = *last_plan_;
    interfaces::msg::ChassisCmd cmd_msg;
    cmd_msg.step_up_ahead = plan.step_up_ahead;
    cmd_msg.step_down_ahead = plan.step_down_ahead;
    cmd_msg.slow_spin = plan.slow_spin;
    cmd_msg.fast_spin = plan.fast_spin;

    switch (plan.mode) {

    // ─── INVALID: 输出零速 ───
    case interfaces::msg::LocalPlan::MODE_INVALID: {
        cmd_msg.velocity = 0.0f;
        cmd_msg.omega = 0.0f;
        last_cmd_ = Eigen::Vector2d::Zero();
        mpc_solver_->reset_warm_start();
        break;
    }

    // ─── TRACK: MPC 跟踪局部路径 ───
    case interfaces::msg::LocalPlan::MODE_TRACK: {
        if (plan.x.size() < 3) {
            cmd_msg.velocity = 0.0f;
            cmd_msg.omega = 0.0f;
            break;
        }
        // 将 LocalPlan 点序列构建为 B-spline
        std::vector<Eigen::Vector2d> cpts;
        cpts.reserve(plan.x.size());
        for (size_t i = 0; i < plan.x.size(); i++) {
            cpts.emplace_back(static_cast<double>(plan.x[i]), static_cast<double>(plan.y[i]));
        }
        SplineD local_spline(cpts);
        local_spline.setExtrapolate(true);

        const auto result = mpc_solver_->follow_path(
            local_spline, chassis_pose, chassis_status_,
            *cost_map_ptr, *dir_map_ptr
        );
        if (!result) {
            RCLCPP_ERROR(get_logger(), "MPC(Track) failed: %s", result.error().c_str());
            cmd_msg.velocity = 0.0f;
            cmd_msg.omega = 0.0f;
            break;
        }
        const auto& [cmd_vec, prediction] = *result;
        cmd_msg.velocity = static_cast<float>(cmd_vec.x());
        cmd_msg.omega = static_cast<float>(cmd_vec.y());
        last_cmd_ = cmd_vec;

        if (enable_debug_ && debug_predicted_path_pub_) {
            debug_predicted_path_pub_->publish(poses_to_path_msg(prediction.path_map));
            std_msgs::msg::Float64 v_msg, w_msg;
            v_msg.data = prediction.v_pred.empty() ? 0.0 : prediction.v_pred[0];
            w_msg.data = prediction.w_pred.empty() ? 0.0 : prediction.w_pred[0];
            debug_v_pred_pub_->publish(v_msg);
            debug_w_pred_pub_->publish(w_msg);
        }
        break;
    }

    // ─── STOP: MPC 平滑减速 ───
    case interfaces::msg::LocalPlan::MODE_STOP: {
        const auto result = mpc_solver_->stop(
            chassis_pose, chassis_status_,
            *cost_map_ptr, *dir_map_ptr
        );
        if (!result) {
            RCLCPP_ERROR(get_logger(), "MPC(Stop) failed: %s", result.error().c_str());
            cmd_msg.velocity = 0.0f;
            cmd_msg.omega = 0.0f;
            break;
        }
        const auto& [cmd_vec, prediction] = *result;
        cmd_msg.velocity = static_cast<float>(cmd_vec.x());
        cmd_msg.omega = static_cast<float>(cmd_vec.y());
        last_cmd_ = cmd_vec;
        break;
    }

    // ─── HOLD_FIXED: MPC 位置保持 ───
    case interfaces::msg::LocalPlan::MODE_HOLD_FIXED: {
        const Eigen::Vector2d goal(static_cast<double>(plan.fixed_x), static_cast<double>(plan.fixed_y));
        const auto result = mpc_solver_->hold_at_point(
            goal, chassis_pose, chassis_status_,
            *cost_map_ptr, *dir_map_ptr
        );
        if (!result) {
            RCLCPP_ERROR(get_logger(), "MPC(Fixed) failed: %s", result.error().c_str());
            cmd_msg.velocity = 0.0f;
            cmd_msg.omega = 0.0f;
            break;
        }
        const auto& [cmd_vec, prediction] = *result;
        cmd_msg.velocity = static_cast<float>(cmd_vec.x());
        cmd_msg.omega = static_cast<float>(cmd_vec.y());
        last_cmd_ = cmd_vec;
        break;
    }

    // ─── SPIN: 透传 spin 标志，零速 ───
    case interfaces::msg::LocalPlan::MODE_SPIN: {
        cmd_msg.velocity = 0.0f;
        cmd_msg.omega = 0.0f;
        last_cmd_ = Eigen::Vector2d::Zero();
        mpc_solver_->reset_warm_start();
        break;
    }

    // ─── REVERSE: 直接倒退 ───
    case interfaces::msg::LocalPlan::MODE_REVERSE: {
        cmd_msg.velocity = -0.5f; // 使用固定倒车速度（由 local_planner 决策触发）
        cmd_msg.omega = 0.0f;
        last_cmd_ = Eigen::Vector2d(-0.5, 0.0);
        mpc_solver_->reset_warm_start();
        break;
    }

    default: {
        RCLCPP_WARN(get_logger(), "Unknown LocalPlan mode: %d", plan.mode);
        cmd_msg.velocity = 0.0f;
        cmd_msg.omega = 0.0f;
        break;
    }
    } // switch

    chassis_cmd_pub_->publish(cmd_msg);
}

// ═══════════════════ 工具函数 ════════════════════════════════

bool MpcControllerNode::get_chassis_pose(Eigen::Vector3d& chassis_pose) const {
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

nav_msgs::msg::Path MpcControllerNode::poses_to_path_msg(const std::vector<Eigen::Vector2d>& pts) const {
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

} // namespace mpc_controller

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mpc_controller::MpcControllerNode)
