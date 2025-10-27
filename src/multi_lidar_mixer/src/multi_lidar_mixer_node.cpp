#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

#include <common_utils/convert.hpp>

namespace multi_lidar_mixer {
struct LidarSub {
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr cloud_sub;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
};

class MultiLidarMixerNode: public rclcpp::Node {
public:
    explicit MultiLidarMixerNode(const rclcpp::NodeOptions& options);

private:
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_cloud_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<LidarSub> lidar_sub_queue_;

    std::vector<Eigen::Vector3d> last_angular_vel_base_;
    std::vector<uint64_t> last_imu_time_;

    std::vector<livox_ros_driver2::msg::CustomPoint> livox_points_;
    uint64_t timebase_;
};

MultiLidarMixerNode::MultiLidarMixerNode(const rclcpp::NodeOptions& options): Node("multi_lidar_mixer", options) {
    livox_points_.reserve(500000);
    const std::string imu_pub_topic = declare_parameter<std::string>("imu_pub_topic");
    const std::string livox_cloud_pub_topic = declare_parameter<std::string>("livox_cloud_pub_topic");
    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>(imu_pub_topic, 1);
    livox_cloud_pub_ = create_publisher<livox_ros_driver2::msg::CustomMsg>(livox_cloud_pub_topic, 1);

    const std::vector<std::string> lidar_names = declare_parameter<std::vector<std::string>>("lidar_names");
    last_angular_vel_base_.resize(lidar_names.size());
    last_imu_time_.resize(lidar_names.size());
    for (int i = 0; i < lidar_names.size(); i++) {
        const auto& lidar_name = lidar_names[i];
        const std::string cloud_topic = declare_parameter<std::string>(lidar_name + ".cloud_topic");
        const std::string imu_topic = declare_parameter<std::string>(lidar_name + ".imu_topic");
        const Eigen::Vector3d translation(
            declare_parameter<double>(lidar_name + ".transform_to_base.translation.x"),
            declare_parameter<double>(lidar_name + ".transform_to_base.translation.y"),
            declare_parameter<double>(lidar_name + ".transform_to_base.translation.z")
        );
        const Eigen::Quaterniond rotation(
            declare_parameter<double>(lidar_name + ".transform_to_base.rotation.w"),
            declare_parameter<double>(lidar_name + ".transform_to_base.rotation.x"),
            declare_parameter<double>(lidar_name + ".transform_to_base.rotation.y"),
            declare_parameter<double>(lidar_name + ".transform_to_base.rotation.z")
        );

        const auto cloud_callback = [=, this](const livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
            if (livox_points_.empty()) timebase_ = msg->timebase;
            if (msg->timebase < timebase_) {
                const uint32_t timebase_diff = timebase_ - msg->timebase;
                timebase_ = msg->timebase;
                for (auto& point: livox_points_) {
                    point.offset_time += timebase_diff;
                }
            }
            const uint32_t timebase_diff = msg->timebase - timebase_;
            for (const auto& point: msg->points) {
                const Eigen::Vector3d original(point.x, point.y, point.z);
                const Eigen::Vector3d transformed = rotation * original + translation;
                livox_ros_driver2::msg::CustomPoint point_base = point;
                point_base.x = transformed.x();
                point_base.y = transformed.y();
                point_base.z = transformed.z();
                point_base.offset_time = point.offset_time + timebase_diff;
                livox_points_.emplace_back(point_base);
            }
        };
        const auto imu_callback = [=, this](const sensor_msgs::msg::Imu::SharedPtr msg) {
            const Eigen::Vector3d angular_vel = utils::convert_to<Eigen::Vector3d>(msg->angular_velocity);
            const Eigen::Vector3d linear_acc = utils::convert_to<Eigen::Vector3d>(msg->linear_acceleration);
            const uint64_t timestamp = msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec;
            const Eigen::Vector3d angular_vel_base = rotation * angular_vel;
            const Eigen::Vector3d angular_acc_base = (angular_vel_base - last_angular_vel_base_[i]) / (timestamp - last_imu_time_[i]);
            const Eigen::Vector3d linear_acc_base = rotation * linear_acc -
                angular_acc_base.cross(translation) -
                angular_vel_base.cross(angular_vel_base.cross(translation));
            sensor_msgs::msg::Imu imu_base;
            imu_base.header.frame_id = "base";
            imu_base.header.stamp = msg->header.stamp;
            imu_base.linear_acceleration = utils::convert_to<geometry_msgs::msg::Vector3>(linear_acc_base);
            imu_base.angular_velocity = utils::convert_to<geometry_msgs::msg::Vector3>(angular_vel_base);
            imu_pub_->publish(imu_base);
            last_angular_vel_base_[i] = angular_vel_base;
            last_imu_time_[i] = timestamp;
        };
        lidar_sub_queue_.emplace_back(
            create_subscription<livox_ros_driver2::msg::CustomMsg>(cloud_topic, 1, cloud_callback),
            create_subscription<sensor_msgs::msg::Imu>(imu_topic, 1, imu_callback)
        );
    }

    const int cloud_pub_freq = declare_parameter<int>("cloud_pub_freq");
    const auto timer_callback = [this]() {
        livox_ros_driver2::msg::CustomMsg msg;
        msg.header.frame_id = "base";
        msg.header.stamp = rclcpp::Time(timebase_);
        msg.timebase = timebase_;
        msg.point_num = livox_points_.size();
        msg.points = livox_points_;
        livox_cloud_pub_->publish(msg);
        livox_points_.clear();
    };
    timer_ = create_wall_timer(std::chrono::milliseconds(1000 / cloud_pub_freq), timer_callback);
}
} // namespace multi_lidar_mixer

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(multi_lidar_mixer::MultiLidarMixerNode)