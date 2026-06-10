/**
 * This file is part of Mid-360 driver.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#pragma once

#define ASIO_NO_DEPRECATED
#include <asio.hpp>
#include <atomic>
#include <functional>
#include <unordered_map>
#include <vector>

namespace mid360_driver {

    struct DriverRobustnessConfig {
        bool validate_crc;
        double max_packet_time_jump;
        double max_packet_time_span;
        double max_point_range;
        double max_imu_acc;
        double max_imu_gyro;
        double min_drop_log_interval;
    };

    struct Point {
        double timestamp;
        float x, y, z;
        float intensity;
    };

    struct ImuMsg {
        double timestamp;
        float angular_velocity_x;
        float angular_velocity_y;
        float angular_velocity_z;
        float linear_acceleration_x;
        float linear_acceleration_y;
        float linear_acceleration_z;
    };

    struct IpAddressHasher {
        std::size_t operator()(const asio::ip::address &addr) const noexcept;
    };

    class Mid360Driver {
    private:
        std::atomic<bool> is_running = true;
        asio::ip::address host_ip;
        asio::ip::udp::socket receive_pointcloud_socket;
        asio::ip::udp::socket receive_imu_socket;
        DriverRobustnessConfig robustness_config;
        std::unordered_map<asio::ip::address, double, IpAddressHasher> delta_time_map;
        std::unordered_map<asio::ip::address, double, IpAddressHasher> last_lidar_timestamp_map;
        std::unordered_map<asio::ip::address, double, IpAddressHasher> last_imu_timestamp_map;
        std::function<void(const asio::ip::address &lidar_ip, const std::vector<Point> &points)> on_receive_pointcloud;
        std::function<void(const asio::ip::address &lidar_ip, const ImuMsg &imu_msg)> on_receive_imu;

    public:
        Mid360Driver(asio::io_context &io_context,
                     const asio::ip::address &host_ip,
                     DriverRobustnessConfig robustness_config,
                     std::function<void(const asio::ip::address &lidar_ip, const std::vector<Point> &points)> on_receive_pointcloud,
                     std::function<void(const asio::ip::address &lidar_ip, const ImuMsg &imu_msg)> on_receive_imu);

        ~Mid360Driver();

        void stop();

        asio::awaitable<void> receive_pointcloud();

        asio::awaitable<void> receive_imu();
    };

}// namespace mid360_driver
