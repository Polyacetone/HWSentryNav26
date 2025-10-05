#pragma once

#include <tf2/convert.hpp>
#include <tf2/utils.hpp>
#include <tf2/time.hpp>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <common_utils/convert_utils.hpp>

namespace utils {

static constexpr bool try_lookup_tf(
    const std::shared_ptr<tf2_ros::Buffer> tf_buffer,
    const std::string& target,
    const std::string& source,
    const std::optional<rclcpp::Time>& opt_time,
    tf2::Transform& tf_out,
    const std::function<void(const std::string& err_info)> err_info_handler
) {
    constexpr int MAX_ATTEMPTS = 50;
    std::string err_info;
    for (int i = 0; i < MAX_ATTEMPTS; i++) {
        try {
            geometry_msgs::msg::Transform transform;
            if (opt_time.has_value()) {
                transform = tf_buffer->lookupTransform(target, source, *opt_time).transform;
            } else {
                transform = tf_buffer->lookupTransform(target, source, tf2::TimePointZero).transform;
            }
            tf_out = utils::convert_to<tf2::Transform>(transform);
            return true;
        } catch (const std::exception& ex) {
            err_info = ex.what();
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    err_info_handler(err_info);
    return false;
}

} // namespace utils