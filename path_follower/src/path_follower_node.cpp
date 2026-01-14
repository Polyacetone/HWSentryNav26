#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <nav_msgs/msg/path.hpp>
#include <interfaces/msg/chassis_status.hpp>
#include <interfaces/msg/chassis_cmd.hpp>

#include <common_utils/convert.hpp>
#include <path_follower/mpc_controller.hpp>

namespace path_follower {
class PathFollowerNode: public rclcpp::Node {
public:
    explicit PathFollowerNode(const rclcpp::NodeOptions& options);

private:
    void control_points_callback(const nav_msgs::msg::Path::SharedPtr msg);
    void chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg);
    void timer_callback();
    
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr control_points_sub_;
    rclcpp::Subscription<interfaces::msg::ChassisStatus>::SharedPtr chassis_status_sub_;
    rclcpp::Publisher<interfaces::msg::ChassisCmd>::SharedPtr chassis_cmd_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    std::vector<Eigen::Vector2d> control_points_; // B样条控制点(x, y)
    Eigen::Vector2d current_status_; // 从串口接收的当前(v, omega)
    Eigen::Vector2d last_cmd_status_; // MPC给出的上一个控制量(v, omega)
    
    MPCParams params_;
    std::unique_ptr<MPCController> mpc_controller_;
    double stop_threshold_;
};

PathFollowerNode::PathFollowerNode(const rclcpp::NodeOptions& options): Node("path_follower", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    params_ = {
        .horizon = (int)declare_parameter<int>("mpc.horizon"),
        .dt = declare_parameter<double>("mpc.dt"),
        .vel_max = declare_parameter<double>("mpc.thresholds.vel_max"),
        .vel_min = declare_parameter<double>("mpc.thresholds.vel_min"),
        .omega_max = declare_parameter<double>("mpc.thresholds.omega_max"),
        .omega_min = declare_parameter<double>("mpc.thresholds.omega_min"),
        .acc_max = declare_parameter<double>("mpc.thresholds.acc_max"),
        .alpha_max = declare_parameter<double>("mpc.thresholds.alpha_max"),
        .q_y = declare_parameter<double>("mpc.weights.q_y"),
        .q_theta = declare_parameter<double>("mpc.weights.q_theta"),
        .q_u = declare_parameter<double>("mpc.weights.q_u"),
        .q_v_final = declare_parameter<double>("mpc.weights.q_v_final"),
        .r_v = declare_parameter<double>("mpc.weights.r_v"),
        .r_omega = declare_parameter<double>("mpc.weights.r_omega"),
        .r_dv = declare_parameter<double>("mpc.weights.r_dv"),
        .r_domega = declare_parameter<double>("mpc.weights.r_domega"),
        .acc_limit_weight = declare_parameter<double>("mpc.weights.acc_limit"),
        .alpha_limit_weight = declare_parameter<double>("mpc.weights.alpha_limit"),
        .proj_num_samples = (int)declare_parameter<int>("mpc.projection.num_samples"),
        .proj_search_window = declare_parameter<double>("mpc.projection.search_window"),
        .max_correspondence_distance = declare_parameter<double>("mpc.projection.max_correspondence_distance")
    };
    mpc_controller_ = std::make_unique<MPCController>(params_);
    stop_threshold_ = declare_parameter<double>("stop_threshold");
    current_status_ = Eigen::Vector2d::Zero();
    last_cmd_status_ = Eigen::Vector2d::Zero();

    control_points_sub_ = create_subscription<nav_msgs::msg::Path>(
        declare_parameter<std::string>("control_points_sub_topic"), 1,
        [this](const nav_msgs::msg::Path::SharedPtr msg) { control_points_callback(msg); }
    );
    chassis_status_sub_ = create_subscription<interfaces::msg::ChassisStatus>(
        declare_parameter<std::string>("chassis_status_sub_topic"), 1,
        [this](const interfaces::msg::ChassisStatus::SharedPtr msg) { chassis_status_callback(msg); }
    );
    chassis_cmd_pub_ = create_publisher<interfaces::msg::ChassisCmd>(declare_parameter<std::string>("chassis_cmd_pub_topic"), 1);
    timer_ = create_wall_timer(std::chrono::duration<double>(params_.dt), [this]() { timer_callback(); });
}

void PathFollowerNode::control_points_callback(const nav_msgs::msg::Path::SharedPtr msg) {
    if (msg->poses.size() < 3) {
        if (!msg->poses.empty()) {
            RCLCPP_WARN(get_logger(), "Received insufficient control points (%zu), need at least 3!", msg->poses.size());
        }
        control_points_.clear();
        return;
    }
    control_points_.clear();
    control_points_.reserve(msg->poses.size());
    for (const auto& ps : msg->poses) {
        control_points_.emplace_back(ps.pose.position.x, ps.pose.position.y);
    }
}

void PathFollowerNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    current_status_.x() = msg->velocity;
    current_status_.y() = msg->palstance;
}

void PathFollowerNode::timer_callback() {
    if (control_points_.empty()) return;

    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = tf_buffer_->lookupTransform("map", "chassis_link", tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(get_logger(), "Could not transform chassis_link to map: %s", ex.what());
        return;
    }

    Eigen::Vector3d current_pose;
    current_pose.head<2>() = utils::convert_to<Eigen::Vector3d>(tf.transform.translation).head<2>();

    Eigen::Quaterniond q = utils::convert_to<Eigen::Quaterniond>(tf.transform.rotation);
    Eigen::Vector2d x_axis = (q * Eigen::Vector3d::UnitX()).head<2>();
    if (x_axis.norm() < 1e-6) {
        RCLCPP_WARN(get_logger(), "Invalid chassis_link orientation");
        return;
    }
    current_pose.z() = atan2(x_axis.y(), x_axis.x());

    if ((mpc_controller_->get_end_position(control_points_) - current_pose.head<2>()).norm() < stop_threshold_) {
        interfaces::msg::ChassisCmd msg;
        msg.velocity = 0.0;
        msg.palstance = 0.0;
        msg.step_ahead = false;
        chassis_cmd_pub_->publish(msg);
        last_cmd_status_ = Eigen::Vector2d::Zero();
        return;
    }

    Eigen::Vector2d cmd_status = mpc_controller_->solve(control_points_, current_pose, last_cmd_status_);
    last_cmd_status_ = cmd_status;

    interfaces::msg::ChassisCmd msg;
    msg.velocity = cmd_status.x();
    msg.palstance = cmd_status.y();
    msg.step_ahead = false;
    chassis_cmd_pub_->publish(msg);
}
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(path_follower::PathFollowerNode)