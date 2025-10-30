/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#include "small_point_lio_node.hpp"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <common_utils/convert.hpp>

namespace small_point_lio {
SmallPointLioNode::SmallPointLioNode(const rclcpp::NodeOptions &options): Node("small_point_lio", options) {
    std::string lidar_topic = declare_parameter<std::string>("lidar_topic");
    std::string imu_topic = declare_parameter<std::string>("imu_topic");
    bool save_pcd = declare_parameter<bool>("save_pcd");
    small_point_lio = std::make_unique<small_point_lio::SmallPointLio>(*this);
    odometry_publisher = create_publisher<nav_msgs::msg::Odometry>("odometry", 1);
    pointcloud_publisher = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered", 1);
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf_buffer = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    map_save_trigger = create_service<std_srvs::srv::Trigger>(
        "map_save",
        [this, save_pcd](const std_srvs::srv::Trigger::Request::SharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res) {
            if (!save_pcd) {
                RCLCPP_ERROR(rclcpp::get_logger("small_point_lio"), "pcd save is disabled");
                return;
            }
            voxelgrid_sampling::VoxelgridSampling downsampler;
            std::vector<Eigen::Vector3f> downsampled;
            downsampler.voxelgrid_sampling_omp(pointcloud_to_save, downsampled, 0.02);
            pcl::PointCloud<pcl::PointXYZ> pcl_pointcloud;
            pcl_pointcloud.reserve(downsampled.size());
            for (const auto &point: downsampled) {
                pcl::PointXYZ new_point(point.x(), point.y(), point.z());
                pcl_pointcloud.push_back(new_point);
            }
            pcl::PCDWriter writer;
            writer.writeBinary(ROOT_DIR + "/pcd/scan.pcd", pcl_pointcloud);
            res->success = true;
            res->message = "pcd saved with " + std::to_string(pcl_pointcloud.size()) + " points";
            RCLCPP_INFO(rclcpp::get_logger("small_point_lio"), "save pcd success");
        }
    );
    small_point_lio->set_odometry_callback([this](const common::Odometry &odometry) {
        last_odometry = odometry;

        nav_msgs::msg::Odometry odometry_msg;
        odometry_msg.header.stamp.sec = std::floor(odometry.timestamp);
        odometry_msg.header.stamp.nanosec = static_cast<uint32_t>((odometry.timestamp - odometry_msg.header.stamp.sec) * 1e9);
        odometry_msg.header.frame_id = "odom";
        odometry_msg.child_frame_id = "base";
        odometry_msg.pose.pose.position = utils::convert_to<geometry_msgs::msg::Point>(odometry.position);
        odometry_msg.pose.pose.orientation = utils::convert_to<geometry_msgs::msg::Quaternion>(odometry.orientation);
        odometry_msg.twist.twist.linear = utils::convert_to<geometry_msgs::msg::Vector3>(odometry.velocity);
        odometry_msg.twist.twist.angular = utils::convert_to<geometry_msgs::msg::Vector3>(odometry.angular_velocity);

        geometry_msgs::msg::TransformStamped transform_stamped;
        transform_stamped.header.stamp = odometry_msg.header.stamp;
        transform_stamped.header.frame_id = "odom";
        transform_stamped.child_frame_id = "base";
        transform_stamped.transform.translation = utils::convert_to<geometry_msgs::msg::Vector3>(odometry.position);
        transform_stamped.transform.rotation = utils::convert_to<geometry_msgs::msg::Quaternion>(odometry.orientation);

        tf_broadcaster->sendTransform(transform_stamped);
        odometry_publisher->publish(odometry_msg);
    });
    small_point_lio->set_pointcloud_callback([this, save_pcd](const std::vector<Eigen::Vector3f> &pointcloud) {
        pcl::PointCloud<pcl::PointXYZ> pcl_pointcloud;
        pcl_pointcloud.reserve(pointcloud.size());
        for (const auto &point: pointcloud) {
            pcl::PointXYZ new_point(point.x(), point.y(), point.z());
            pcl_pointcloud.push_back(new_point);
        }
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(pcl_pointcloud, msg);
        msg.header.stamp.sec = std::floor(last_odometry.timestamp);
        msg.header.stamp.nanosec = static_cast<uint32_t>((last_odometry.timestamp - msg.header.stamp.sec) * 1e9);
        msg.header.frame_id = "odom";
        pointcloud_publisher->publish(msg);
        if (save_pcd) {
            pointcloud_to_save.insert(pointcloud_to_save.end(), pointcloud.begin(), pointcloud.end());
        }
    });
    pointcloud_subsciber = create_subscription<livox_ros_driver2::msg::CustomMsg>(
        lidar_topic,
        rclcpp::QoS(1),
        [this](const livox_ros_driver2::msg::CustomMsg &msg) {
            pointcloud.clear();
            pointcloud.reserve(msg.points.size());
            common::Point new_point;
            for (const auto &pt: msg.points) {
                if ((pt.tag & 0b00110000) || (pt.tag & 0b00001100) || (pt.tag & 0b00000011)) continue;
                new_point.position << pt.x, pt.y, pt.z;
                new_point.timestamp = static_cast<double>(msg.timebase + pt.offset_time) * 1e-9;
                pointcloud.push_back(new_point);
            }
            small_point_lio->on_point_cloud_callback(pointcloud);
            small_point_lio->handle_once();
        }
    );
    imu_subsciber = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic,
        rclcpp::QoS(1),
        [this](const sensor_msgs::msg::Imu &msg) {
            common::ImuMsg imu_msg;
            imu_msg.angular_velocity = utils::convert_to<Eigen::Vector3d>(msg.angular_velocity);
            imu_msg.linear_acceleration = utils::convert_to<Eigen::Vector3d>(msg.linear_acceleration);
            imu_msg.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;
            small_point_lio->on_imu_callback(imu_msg);
            small_point_lio->handle_once();
        }
    );
}
}// namespace small_point_lio

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(small_point_lio::SmallPointLioNode)