#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#include <common_utils/convert.hpp>
#include <common_utils/ema_filter.hpp>

namespace path_follower {
class PathFollowerNode: public rclcpp::Node {
public:
    explicit PathFollowerNode(const rclcpp::NodeOptions& options);

private:
    void timer_callback();
    int find_nearest_point_on_path(const Eigen::Vector2d& point) const;
    Eigen::Vector2d find_lookahead_point_on_path(const int current_index, const double lookahead_distance) const;
    void publish_velocity(const Eigen::Vector2d& velocity) const;

    int control_freq_;
    double max_velocity_, stop_distance_, nearest_point_threshold_, lookahead_distance_basic_;

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    nav_msgs::msg::Path path_;
    std::unique_ptr<utils::EMAFilter<Eigen::Vector2d>> velocity_;
};

PathFollowerNode::PathFollowerNode(const rclcpp::NodeOptions& options): Node("path_follower_node", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    control_freq_ = declare_parameter<int>("control_freq");
    max_velocity_ = declare_parameter<double>("max_velocity");
    stop_distance_ = declare_parameter<double>("stop_distance");
    nearest_point_threshold_ = declare_parameter<double>("nearest_point_threshold");
    lookahead_distance_basic_ = declare_parameter<double>("lookahead_distance_basic");
    double ema_filter_ratio = declare_parameter<double>("ema_filter_ratio");
    velocity_ = std::make_unique<utils::EMAFilter<Eigen::Vector2d>>(ema_filter_ratio);

    std::string path_sub_topic = declare_parameter<std::string>("path_sub_topic");
    path_sub_ = create_subscription<nav_msgs::msg::Path>(
        path_sub_topic, 1,
        [this](const nav_msgs::msg::Path::SharedPtr msg) { path_ = *msg; }
    );
    std::string cmd_vel_pub_topic = declare_parameter<std::string>("cmd_vel_pub_topic");
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(cmd_vel_pub_topic, 1);
    timer_ = create_wall_timer(
        std::chrono::milliseconds(1000 / control_freq_),
        [this]() { timer_callback(); }
    );
}

void PathFollowerNode::timer_callback() {
    if (path_.poses.size() < 3) {
        publish_velocity({0, 0});
        return;
    }

    Eigen::Vector2d base_to_map;
    try {
        base_to_map = utils::convert_to<Eigen::Vector3d>(tf_buffer_->lookupTransform(
            "map", "base", tf2::TimePointZero
        ).transform.translation)(Eigen::seq(0, 1));
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(), "Failed to lookup base to map: %s", ex.what());
        publish_velocity({0, 0});
        return;
    }

    const Eigen::Vector2d last_point = utils::convert_to<Eigen::Vector3d>(
        path_.poses[path_.poses.size() - 1].pose.position
    )(Eigen::seq(0, 1));
    if ((last_point - base_to_map).norm() < stop_distance_) {
        publish_velocity({0, 0});
        return;
    }

    const int nearest_point_idx = find_nearest_point_on_path(base_to_map);
    if (nearest_point_idx == -1) {
        RCLCPP_WARN(get_logger(), "Base too far from path!");
        publish_velocity({0, 0});
        return;
    }

    const double lookahead_distance = 1.0 / control_freq_ * velocity_->value().norm() + lookahead_distance_basic_;
    const Eigen::Vector2d lookahead_point = find_lookahead_point_on_path(nearest_point_idx, lookahead_distance);
    const Eigen::Vector2d position_diff = lookahead_point - base_to_map;
    const Eigen::Vector2d velocity = position_diff.normalized() * (position_diff.norm() / lookahead_distance * max_velocity_);
    publish_velocity(velocity);
}

int PathFollowerNode::find_nearest_point_on_path(const Eigen::Vector2d& point) const {
    double min_distance = std::numeric_limits<double>::max();
    int point_index = -1;
    for (int i = 0; i < path_.poses.size(); i++) {
        const Eigen::Vector2d path_point(path_.poses[i].pose.position.x, path_.poses[i].pose.position.y);
        const double distance = (point - path_point).norm();
        if (distance < nearest_point_threshold_ && distance < min_distance) {
            min_distance = distance;
            point_index = i;
        }
    }
    return point_index;
}

Eigen::Vector2d PathFollowerNode::find_lookahead_point_on_path(const int current_index, const double lookahead_distance) const {
    double accumulated_distance = 0;
    int i;
    for (i = current_index + 1; i < path_.poses.size(); i++) {
        const Eigen::Vector2d prev_point(path_.poses[i - 1].pose.position.x, path_.poses[i - 1].pose.position.y);
        const Eigen::Vector2d curr_point(path_.poses[i].pose.position.x, path_.poses[i].pose.position.y);
        accumulated_distance += (curr_point - prev_point).norm();
        if (accumulated_distance > lookahead_distance) break;
    }
    return {path_.poses[i].pose.position.x, path_.poses[i].pose.position.y};
}

void PathFollowerNode::publish_velocity(const Eigen::Vector2d& velocity) const {
    velocity_->update(velocity);
    geometry_msgs::msg::TwistStamped msg;
    msg.header.frame_id = "map";
    msg.header.stamp = now();
    msg.twist.linear.x = velocity_->value().x();
    msg.twist.linear.y = velocity_->value().y();
    cmd_vel_pub_->publish(msg);
}
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(path_follower::PathFollowerNode)