/**
 * This file is part of Mid-360 driver.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#include "mid360_driver/mid360_driver.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <mutex>
#include <numbers>
#include <rclcpp/rclcpp.hpp>
#include <string>

namespace mid360_driver {

    constexpr std::size_t MAX_PACKET_SIZE = 1400;

    enum DataType : std::uint8_t {
        LIVOX_LIDAR_IMU_DATA = 0,
        LIVOX_LIDAR_CARTESIAN_COORDINATE_HIGH_DATA = 0x01,
        LIVOX_LIDAR_CARTESIAN_COORDINATE_LOW_DATA = 0x02,
        LIVOX_LIDAR_SPHERICAL_COORDINATE_DATA = 0x03
    };

    enum TimestampType : std::uint8_t {
        TIMESTAMP_TYPE_NO_SYNC = 0,
        TIMESTAMP_TYPE_GPTP_OR_PTP = 1,
        TIMESTAMP_TYPE_GPS = 2
    };

    struct DataHeader {
        uint8_t version;
        uint16_t length;
        uint16_t time_interval;
        uint16_t dot_num;
        uint16_t udp_cnt;
        uint8_t frame_cnt;
        DataType data_type;
        TimestampType time_type;
        uint8_t reserved[12];
        uint32_t crc32;
        uint64_t timestamp;
    } __attribute__((packed));

    struct Imu {
        float angular_velocity_x;
        float angular_velocity_y;
        float angular_velocity_z;
        float linear_acceleration_x;
        float linear_acceleration_y;
        float linear_acceleration_z;
    } __attribute__((packed));

    struct CartesianHighPoint {
        int32_t x;
        int32_t y;
        int32_t z;
        uint8_t reflectivity;
        uint8_t tag;
    } __attribute__((packed));

    struct CartesianLowPoint {
        int16_t x;
        int16_t y;
        int16_t z;
        uint8_t reflectivity;
        uint8_t tag;
    } __attribute__((packed));

    struct SphericalPoint {
        uint32_t depth;
        uint16_t theta;
        uint16_t phi;
        uint8_t reflectivity;
        uint8_t tag;
    } __attribute__((packed));

    constexpr std::uint32_t make_crc_entry(std::uint32_t value) noexcept {
        for (int i = 0; i < 8; i++) {
            value = (value & 1U) ? (0xEDB88320U ^ (value >> 1U)) : (value >> 1U);
        }
        return value;
    }

    constexpr std::array<std::uint32_t, 256> make_crc_table() noexcept {
        std::array<std::uint32_t, 256> table {};
        for (std::size_t i = 0; i < table.size(); i++) {
            table[i] = make_crc_entry(static_cast<std::uint32_t>(i));
        }
        return table;
    }

    std::uint32_t crc32(const std::uint8_t *data, const std::size_t size) noexcept {
        static constexpr auto table = make_crc_table();
        std::uint32_t crc = 0xFFFFFFFFU;
        for (std::size_t i = 0; i < size; i++) {
            crc = table[(crc ^ data[i]) & 0xFFU] ^ (crc >> 8U);
        }
        return crc ^ 0xFFFFFFFFU;
    }

    void log_packet_drop(const char *reason, const double min_interval) {
        static std::mutex mutex;
        static auto last_log_time = std::chrono::steady_clock::time_point{};
        static std::size_t total_drops = 0;
        std::lock_guard lock(mutex);
        const auto now = std::chrono::steady_clock::now();
        ++total_drops;
        if (last_log_time == std::chrono::steady_clock::time_point{} ||
            now - last_log_time >= std::chrono::duration<double>(min_interval)) {
            last_log_time = now;
            RCLCPP_WARN(
                rclcpp::get_logger("mid360_driver"),
                "dropped packet: %s (total drops: %zu)",
                reason,
                total_drops
            );
        }
    }

    std::size_t point_size(const DataType data_type) noexcept {
        switch (data_type) {
            case LIVOX_LIDAR_CARTESIAN_COORDINATE_HIGH_DATA:
                return sizeof(CartesianHighPoint);
            case LIVOX_LIDAR_CARTESIAN_COORDINATE_LOW_DATA:
                return sizeof(CartesianLowPoint);
            case LIVOX_LIDAR_SPHERICAL_COORDINATE_DATA:
                return sizeof(SphericalPoint);
            default:
                return 0;
        }
    }

    bool is_timestamp_plausible(
        std::unordered_map<asio::ip::address, double, IpAddressHasher> &last_timestamp_map,
        const asio::ip::address &address,
        const double timestamp,
        const double max_time_jump
    ) {
        if (!std::isfinite(timestamp) || timestamp <= 0.0) {
            return false;
        }

        auto [iter, inserted] = last_timestamp_map.try_emplace(address, timestamp);
        if (inserted) {
            return true;
        }

        const double diff = timestamp - iter->second;
        if (diff < -1e-3 || diff > max_time_jump) {
            return false;
        }
        iter->second = timestamp;
        return true;
    }

    bool is_point_valid(uint8_t tag) noexcept {
        constexpr uint8_t kDualReturnMask = 0b00110000;
        constexpr uint8_t kTagTypeMask = 0b00001100;
        constexpr uint8_t kConfidenceMask = 0b00000011;
        return (tag & (kDualReturnMask | kTagTypeMask | kConfidenceMask)) == 0;
    }

    void combine_4_bytes(std::size_t &seed, const unsigned char *bytes) {
        const std::size_t bytes_hash =
                (static_cast<std::size_t>(bytes[0]) << 24) |
                (static_cast<std::size_t>(bytes[1]) << 16) |
                (static_cast<std::size_t>(bytes[2]) << 8) |
                (static_cast<std::size_t>(bytes[3]));
        seed ^= bytes_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    std::size_t IpAddressHasher::operator()(const asio::ip::address &addr) const noexcept {
        if (addr.is_v4()) {
            return std::hash<unsigned int>()(addr.to_v4().to_uint());
        } else {
            const asio::ip::address_v6::bytes_type bytes = addr.to_v6().to_bytes();
            std::size_t result = static_cast<std::size_t>(addr.to_v6().scope_id());
            combine_4_bytes(result, &bytes[0]);
            combine_4_bytes(result, &bytes[4]);
            combine_4_bytes(result, &bytes[8]);
            combine_4_bytes(result, &bytes[12]);
            return result;
        }
    }

    Mid360Driver::Mid360Driver(asio::io_context &io_context,
                               const asio::ip::address &host_ip,
                               DriverRobustnessConfig robustness_config,
                               std::function<void(const asio::ip::address &lidar_ip, const std::vector<Point> &points)> on_receive_pointcloud,
                               std::function<void(const asio::ip::address &lidar_ip, const ImuMsg &imu_msg)> on_receive_imu)
        : host_ip(host_ip),
          receive_pointcloud_socket(io_context),
          receive_imu_socket(io_context),
          robustness_config(robustness_config),
          on_receive_pointcloud(std::move(on_receive_pointcloud)),
          on_receive_imu(std::move(on_receive_imu)) {
        receive_pointcloud_socket.open(asio::ip::udp::v4());
        receive_pointcloud_socket.bind(asio::ip::udp::endpoint(host_ip, 56301));
        receive_imu_socket.open(asio::ip::udp::v4());
        receive_imu_socket.bind(asio::ip::udp::endpoint(host_ip, 56401));
        co_spawn(io_context, receive_pointcloud(), asio::detached);
        co_spawn(io_context, receive_imu(), asio::detached);
    }

    Mid360Driver::~Mid360Driver() {
        stop();
    }

    void Mid360Driver::stop() {
        is_running.store(false, std::memory_order_relaxed);
        asio::error_code error_code;
        receive_pointcloud_socket.close(error_code);
        receive_imu_socket.close(error_code);
    }

    asio::awaitable<void> Mid360Driver::receive_pointcloud() {
        uint8_t buffer[MAX_PACKET_SIZE];
        asio::ip::udp::endpoint sender_endpoint;
        std::vector<Point> points;
        while (is_running.load(std::memory_order_relaxed)) {
            asio::error_code error_code;
            const std::size_t received_size = co_await receive_pointcloud_socket.async_receive_from(
                    asio::buffer(buffer, MAX_PACKET_SIZE),
                    sender_endpoint,
                    asio::redirect_error(asio::use_awaitable, error_code));
            if (error_code || sender_endpoint.port() != 56300) [[unlikely]] {
                continue;
            }
            if (received_size < sizeof(DataHeader)) [[unlikely]] {
                log_packet_drop("lidar: packet smaller than header", robustness_config.min_drop_log_interval);
                continue;
            }

            const auto &header = *reinterpret_cast<const DataHeader *>(buffer);
            const std::size_t raw_point_size = point_size(header.data_type);
            if (raw_point_size == 0 || header.dot_num == 0) [[unlikely]] {
                log_packet_drop("lidar: invalid data_type or dot_num", robustness_config.min_drop_log_interval);
                continue;
            }
            const std::size_t payload_size = static_cast<std::size_t>(header.dot_num) * raw_point_size;
            const std::size_t expected_size = sizeof(DataHeader) + payload_size;
            if (expected_size > received_size) [[unlikely]] {
                log_packet_drop("lidar: packet smaller than declared payload", robustness_config.min_drop_log_interval);
                continue;
            }
            if (header.time_type != TIMESTAMP_TYPE_NO_SYNC && header.time_type != TIMESTAMP_TYPE_GPTP_OR_PTP && header.time_type != TIMESTAMP_TYPE_GPS) [[unlikely]] {
                log_packet_drop("lidar: invalid timestamp type", robustness_config.min_drop_log_interval);
                continue;
            }
            const double packet_time_span = static_cast<double>(header.time_interval) * 1e-10;
            if (!std::isfinite(packet_time_span) || packet_time_span < 0.0 || packet_time_span > robustness_config.max_packet_time_span) [[unlikely]] {
                log_packet_drop("lidar: invalid packet time span", robustness_config.min_drop_log_interval);
                continue;
            }
            if (robustness_config.validate_crc && header.crc32 != 0) {
                const auto computed_crc = crc32(buffer + offsetof(DataHeader, timestamp), sizeof(header.timestamp) + payload_size);
                if (computed_crc != header.crc32) [[unlikely]] {
                    log_packet_drop("lidar: CRC mismatch", robustness_config.min_drop_log_interval);
                    continue;
                }
            }

            double header_timestamp = static_cast<double>(header.timestamp) * 1e-9;
            if (header.time_type == TIMESTAMP_TYPE_NO_SYNC) {
                auto [iter, inserted] = delta_time_map.try_emplace(sender_endpoint.address());
                if (inserted) {
                    auto now = static_cast<double>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) * 1e-9;
                    iter->second = now - header_timestamp;
                    header_timestamp = now;
                } else {
                    header_timestamp += iter->second;
                }
            }
            if (!is_timestamp_plausible(last_lidar_timestamp_map, sender_endpoint.address(), header_timestamp, robustness_config.max_packet_time_jump)) [[unlikely]] {
                log_packet_drop("lidar: implausible timestamp", robustness_config.min_drop_log_interval);
                continue;
            }
            auto interpolate_timestamp = [&](std::size_t i) {
                return header_timestamp + packet_time_span * static_cast<double>(i) / static_cast<double>(header.dot_num);
            };
            points.clear();
            points.reserve(header.dot_num);
            if (header.data_type == LIVOX_LIDAR_CARTESIAN_COORDINATE_HIGH_DATA) {
                const auto *raw_points = reinterpret_cast<const CartesianHighPoint *>(buffer + sizeof(DataHeader));
                for (std::size_t i = 0; i < header.dot_num; ++i) {
                    const auto &raw_point = raw_points[i];
                    if (!is_point_valid(raw_point.tag)) {
                        continue;
                    }
                    Point point;
                    point.timestamp = interpolate_timestamp(i);
                    point.x = static_cast<float>(raw_point.x * 0.001);
                    point.y = static_cast<float>(raw_point.y * 0.001);
                    point.z = static_cast<float>(raw_point.z * 0.001);
                    point.intensity = raw_point.reflectivity;
                    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) && point.x * point.x + point.y * point.y + point.z * point.z <= robustness_config.max_point_range * robustness_config.max_point_range) {
                        points.push_back(point);
                    }
                }
            } else if (header.data_type == LIVOX_LIDAR_CARTESIAN_COORDINATE_LOW_DATA) {
                const auto *raw_points = reinterpret_cast<const CartesianLowPoint *>(buffer + sizeof(DataHeader));
                for (std::size_t i = 0; i < header.dot_num; ++i) {
                    const auto &raw_point = raw_points[i];
                    if (!is_point_valid(raw_point.tag)) {
                        continue;
                    }
                    Point point;
                    point.timestamp = interpolate_timestamp(i);
                    point.x = static_cast<float>(raw_point.x * 0.001);
                    point.y = static_cast<float>(raw_point.y * 0.001);
                    point.z = static_cast<float>(raw_point.z * 0.001);
                    point.intensity = raw_point.reflectivity;
                    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) && point.x * point.x + point.y * point.y + point.z * point.z <= robustness_config.max_point_range * robustness_config.max_point_range) {
                        points.push_back(point);
                    }
                }
            } else if (header.data_type == LIVOX_LIDAR_SPHERICAL_COORDINATE_DATA) {
                const auto *raw_points = reinterpret_cast<const SphericalPoint *>(buffer + sizeof(DataHeader));
                for (std::size_t i = 0; i < header.dot_num; ++i) {
                    const auto &raw_point = raw_points[i];
                    if (!is_point_valid(raw_point.tag)) {
                        continue;
                    }
                    Point point;
                    point.timestamp = interpolate_timestamp(i);
                    double radius = raw_point.depth / 1000.0;
                    double theta = raw_point.theta / 100.0 / 180.0 * std::numbers::pi;
                    double phi = raw_point.phi / 100.0 / 180.0 * std::numbers::pi;
                    point.x = static_cast<float>(radius * sin(theta) * cos(phi));
                    point.y = static_cast<float>(radius * sin(theta) * sin(phi));
                    point.z = static_cast<float>(radius * cos(theta));
                    point.intensity = raw_point.reflectivity;
                    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) && point.x * point.x + point.y * point.y + point.z * point.z <= robustness_config.max_point_range * robustness_config.max_point_range) {
                        points.push_back(point);
                    }
                }
            }
            on_receive_pointcloud(sender_endpoint.address(), points);
        }
    }

    asio::awaitable<void> Mid360Driver::receive_imu() {
        uint8_t buffer[MAX_PACKET_SIZE];
        asio::ip::udp::endpoint sender_endpoint;
        while (is_running.load(std::memory_order_relaxed)) {
            asio::error_code error_code;
            const std::size_t received_size = co_await receive_imu_socket.async_receive_from(
                    asio::buffer(buffer, MAX_PACKET_SIZE),
                    sender_endpoint,
                    asio::redirect_error(asio::use_awaitable, error_code));
            if (error_code || sender_endpoint.port() != 56400) [[unlikely]] {
                continue;
            }
            if (received_size < sizeof(DataHeader) + sizeof(Imu)) [[unlikely]] {
                log_packet_drop("imu: packet smaller than IMU payload", robustness_config.min_drop_log_interval);
                continue;
            }
            const auto &header = *reinterpret_cast<const DataHeader *>(buffer);
            if (header.data_type != LIVOX_LIDAR_IMU_DATA) [[unlikely]] {
                log_packet_drop("imu: invalid data_type", robustness_config.min_drop_log_interval);
                continue;
            }
            if (header.time_type != TIMESTAMP_TYPE_NO_SYNC && header.time_type != TIMESTAMP_TYPE_GPTP_OR_PTP && header.time_type != TIMESTAMP_TYPE_GPS) [[unlikely]] {
                log_packet_drop("imu: invalid timestamp type", robustness_config.min_drop_log_interval);
                continue;
            }
            if (robustness_config.validate_crc && header.crc32 != 0) {
                const auto computed_crc = crc32(buffer + offsetof(DataHeader, timestamp), sizeof(header.timestamp) + sizeof(Imu));
                if (computed_crc != header.crc32) [[unlikely]] {
                    log_packet_drop("imu: CRC mismatch", robustness_config.min_drop_log_interval);
                    continue;
                }
            }
            double header_timestamp = static_cast<double>(header.timestamp) * 1e-9;
            if (header.time_type == TIMESTAMP_TYPE_NO_SYNC) {
                auto [iter, inserted] = delta_time_map.try_emplace(sender_endpoint.address());
                if (inserted) {
                    auto now = static_cast<double>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) * 1e-9;
                    iter->second = now - header_timestamp;
                    header_timestamp = now;
                } else {
                    header_timestamp += iter->second;
                }
            }
            if (!is_timestamp_plausible(last_imu_timestamp_map, sender_endpoint.address(), header_timestamp, robustness_config.max_packet_time_jump)) [[unlikely]] {
                log_packet_drop("imu: implausible timestamp", robustness_config.min_drop_log_interval);
                continue;
            }
            const auto &raw_imu = *reinterpret_cast<const Imu *>(buffer + sizeof(DataHeader));
            ImuMsg imu_msg;
            imu_msg.timestamp = header_timestamp;
            imu_msg.angular_velocity_x = raw_imu.angular_velocity_x;
            imu_msg.angular_velocity_y = raw_imu.angular_velocity_y;
            imu_msg.angular_velocity_z = raw_imu.angular_velocity_z;
            imu_msg.linear_acceleration_x = raw_imu.linear_acceleration_x;
            imu_msg.linear_acceleration_y = raw_imu.linear_acceleration_y;
            imu_msg.linear_acceleration_z = raw_imu.linear_acceleration_z;
            const double acc_norm = std::hypot(imu_msg.linear_acceleration_x, imu_msg.linear_acceleration_y, imu_msg.linear_acceleration_z);
            const double gyro_norm = std::hypot(imu_msg.angular_velocity_x, imu_msg.angular_velocity_y, imu_msg.angular_velocity_z);
            if (!std::isfinite(acc_norm) || !std::isfinite(gyro_norm) || acc_norm > robustness_config.max_imu_acc || gyro_norm > robustness_config.max_imu_gyro) [[unlikely]] {
                log_packet_drop("imu: invalid IMU measurement", robustness_config.min_drop_log_interval);
                continue;
            }
            on_receive_imu(sender_endpoint.address(), imu_msg);
        }
    }

} // namespace mid360_driver
