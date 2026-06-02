/**
 * This file is part of Mid-360 driver.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#include "mid360_driver/mid360_driver_node.hpp"

namespace mid360_driver {

    void LidarPublisher::ensure_initialized(rclcpp::Node &node, const std::string &lidar_topic, const std::string &imu_topic) {
        if (!is_initialized) {
            pointcloud_publisher = node.create_publisher<sensor_msgs::msg::PointCloud2>(lidar_topic, 1000);
            imu_publisher = node.create_publisher<sensor_msgs::msg::Imu>(imu_topic, 1000);
            is_initialized = true;
        }
    }

    void LidarPublisher::ensure_initialized(rclcpp::Node &node, const std::string &lidar_topic, const std::string &imu_topic, const asio::ip::address &lidar_ip) {
        if (!is_initialized) {
            auto lidar_ip_bytes = lidar_ip.to_v4().to_bytes();
            std::string lidar_ip_str;
            lidar_ip_str.push_back('_');
            lidar_ip_str.append(std::to_string(static_cast<int>(lidar_ip_bytes[0])));
            lidar_ip_str.push_back('_');
            lidar_ip_str.append(std::to_string(static_cast<int>(lidar_ip_bytes[1])));
            lidar_ip_str.push_back('_');
            lidar_ip_str.append(std::to_string(static_cast<int>(lidar_ip_bytes[2])));
            lidar_ip_str.push_back('_');
            lidar_ip_str.append(std::to_string(static_cast<int>(lidar_ip_bytes[3])));
            pointcloud_publisher = node.create_publisher<sensor_msgs::msg::PointCloud2>(lidar_topic + lidar_ip_str, 1000);
            imu_publisher = node.create_publisher<sensor_msgs::msg::Imu>(imu_topic + lidar_ip_str, 1000);
            is_initialized = true;
        }
    }

    void LidarPublisher::on_receive_pointcloud(const std::vector<Point> &points) {
        if (points_wait_to_publish.size() + points.size() > MAX_PENDING_POINTS) {
            points_wait_to_publish.clear();
        }
        points_wait_to_publish.reserve(points_wait_to_publish.size() + points.size());
        std::copy(points.begin(), points.end(), std::back_inserter(points_wait_to_publish));
    }

    void LidarPublisher::on_receive_imu(const ImuMsg &imu_msg) {
        if (imu_wait_to_publish.size() >= MAX_PENDING_IMU) {
            imu_wait_to_publish.clear();
        }
        imu_wait_to_publish.push_back(imu_msg);
    }

    void LidarPublisher::prepare_pointcloud_to_publish() {
        std::swap(points_wait_to_publish, points_to_publish);
        points_wait_to_publish.clear();
    }

    void LidarPublisher::prepare_imu_to_publish() {
        std::swap(imu_wait_to_publish, imu_to_publish);
        imu_wait_to_publish.clear();
    }

    void LidarPublisher::publish_pointcloud(const std::string &frame_id) const {
        if (points_to_publish.empty()) {
            return;
        }
        double timestamp = std::numeric_limits<double>::max();
        for (const auto &point: points_to_publish) {
            if (point.timestamp < timestamp) {
                timestamp = point.timestamp;
            }
        }
        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp.sec = static_cast<int32_t>(std::floor(timestamp));
        msg.header.stamp.nanosec = static_cast<uint32_t>((timestamp - msg.header.stamp.sec) * 1e9);
        msg.header.frame_id = frame_id;
        msg.width = static_cast<uint32_t>(points_to_publish.size());
        msg.height = 1;
        msg.fields.reserve(4);
        sensor_msgs::msg::PointField field;
        field.name = "x";
        field.offset = 0;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field.count = 1;
        msg.fields.push_back(field);
        field.name = "y";
        field.offset = 4;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field.count = 1;
        msg.fields.push_back(field);
        field.name = "z";
        field.offset = 8;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field.count = 1;
        msg.fields.push_back(field);
        field.name = "intensity";
        field.offset = 12;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field.count = 1;
        msg.fields.push_back(field);
        field.name = "timestamp";
        field.offset = 16;
        field.datatype = sensor_msgs::msg::PointField::FLOAT64;
        field.count = 1;
        msg.fields.push_back(field);
        msg.is_bigendian = false;
        msg.point_step = 24;
        msg.row_step = msg.width * msg.point_step;
        msg.data.resize(msg.row_step * msg.height);
        auto* pointer = reinterpret_cast<float*>(msg.data.data());
        for (const auto &point: points_to_publish) {
            *pointer = point.x;
            ++pointer;
            *pointer = point.y;
            ++pointer;
            *pointer = point.z;
            ++pointer;
            *pointer = point.intensity;
            ++pointer;
            *reinterpret_cast<double *>(pointer) = point.timestamp;
            pointer += 2;
        }
        msg.is_dense = true;
        pointcloud_publisher->publish(msg);
    }

    void LidarPublisher::publish_imu(const std::string &frame_id) const {
        for (const auto &imu: imu_to_publish) {
            sensor_msgs::msg::Imu msg;
            msg.header.stamp.sec = static_cast<int32_t>(std::floor(imu.timestamp));
            msg.header.stamp.nanosec = static_cast<uint32_t>((imu.timestamp - msg.header.stamp.sec) * 1e9);
            msg.header.frame_id = frame_id;
            msg.angular_velocity.x = imu.angular_velocity_x;
            msg.angular_velocity.y = imu.angular_velocity_y;
            msg.angular_velocity.z = imu.angular_velocity_z;
            msg.linear_acceleration.x = imu.linear_acceleration_x;
            msg.linear_acceleration.y = imu.linear_acceleration_y;
            msg.linear_acceleration.z = imu.linear_acceleration_z;
            imu_publisher->publish(msg);
        }
    }

    Mid360DriverNode::Mid360DriverNode(const rclcpp::NodeOptions &options) : Node("mid360_driver", options) {
        std::string lidar_topic = declare_parameter<std::string>("lidar_topic");
        std::string lidar_frame = declare_parameter<std::string>("lidar_frame");
        std::string imu_topic = declare_parameter<std::string>("imu_topic");
        std::string imu_frame = declare_parameter<std::string>("imu_frame");
        std::string host_ip = declare_parameter<std::string>("host_ip");
        double lidar_publish_time_interval = declare_parameter<double>("lidar_publish_time_interval");
        bool is_topic_name_with_lidar_ip = declare_parameter<bool>("is_topic_name_with_lidar_ip");
        if (!is_topic_name_with_lidar_ip) {
            lidar_publisher.ensure_initialized(*this, lidar_topic, imu_topic);
        }
        mid360_driver = std::make_unique<mid360_driver::Mid360Driver>(
                io_context,
                asio::ip::make_address(host_ip),
                [this, is_topic_name_with_lidar_ip](const asio::ip::address &lidar_ip, const std::vector<Point> &points) {
                    std::lock_guard lock(multi_lidar_mutex_);
                    if (is_topic_name_with_lidar_ip) {
                        auto iter = multi_lidar_publishers.try_emplace(lidar_ip).first;
                        iter->second.on_receive_pointcloud(points);
                    } else {
                        lidar_publisher.on_receive_pointcloud(points);
                    }
                },
                [this, is_topic_name_with_lidar_ip](const asio::ip::address &lidar_ip, const ImuMsg &imu_msg) {
                    std::lock_guard lock(multi_lidar_mutex_);
                    if (is_topic_name_with_lidar_ip) {
                        auto iter = multi_lidar_publishers.try_emplace(lidar_ip).first;
                        iter->second.on_receive_imu(imu_msg);
                    } else {
                        lidar_publisher.on_receive_imu(imu_msg);
                    }
                });
        if (is_topic_name_with_lidar_ip) {
            publish_pointcloud_timer = rclcpp::create_timer(this, get_clock(), std::chrono::milliseconds(100), [this, lidar_topic, imu_topic, lidar_frame]() {
                std::vector<std::pair<asio::ip::address, LidarPublisher*>> snapshot;
                {
                    std::lock_guard lock(multi_lidar_mutex_);
                    snapshot.reserve(multi_lidar_publishers.size());
                    for (auto &[lidar_ip, publisher] : multi_lidar_publishers) {
                        publisher.prepare_pointcloud_to_publish();
                        snapshot.emplace_back(lidar_ip, &publisher);
                    }
                }
                for (auto &[lidar_ip, publisher] : snapshot) {
                    publisher->ensure_initialized(*this, lidar_topic, imu_topic, lidar_ip);
                    publisher->publish_pointcloud(lidar_frame);
                }
            });
            publish_imu_timer = rclcpp::create_timer(this, get_clock(), std::chrono::milliseconds(1), [this, lidar_topic, imu_topic, imu_frame]() {
                std::vector<std::pair<asio::ip::address, LidarPublisher*>> snapshot;
                {
                    std::lock_guard lock(multi_lidar_mutex_);
                    snapshot.reserve(multi_lidar_publishers.size());
                    for (auto &[lidar_ip, publisher] : multi_lidar_publishers) {
                        publisher.prepare_imu_to_publish();
                        snapshot.emplace_back(lidar_ip, &publisher);
                    }
                }
                for (auto &[lidar_ip, publisher] : snapshot) {
                    publisher->ensure_initialized(*this, lidar_topic, imu_topic, lidar_ip);
                    publisher->publish_imu(imu_frame);
                }
            });
        } else {
            publish_pointcloud_timer = rclcpp::create_timer(this, get_clock(), std::chrono::duration<double, std::ratio<1, 1>>(lidar_publish_time_interval), [this, lidar_frame]() {
                {
                    std::lock_guard lock(multi_lidar_mutex_);
                    lidar_publisher.prepare_pointcloud_to_publish();
                }
                lidar_publisher.publish_pointcloud(lidar_frame);
            });
            publish_imu_timer = rclcpp::create_timer(this, get_clock(), std::chrono::milliseconds(1), [this, imu_frame]() {
                {
                    std::lock_guard lock(multi_lidar_mutex_);
                    lidar_publisher.prepare_imu_to_publish();
                }
                lidar_publisher.publish_imu(imu_frame);
            });
        }
        io_thread = std::thread([this]() {
            io_context.run();
        });
    }

    Mid360DriverNode::~Mid360DriverNode() {
        if (mid360_driver) {
            mid360_driver->stop();
        }
        io_context.stop();
        io_thread.join();
    }

}// namespace mid360_driver

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(mid360_driver::Mid360DriverNode)
