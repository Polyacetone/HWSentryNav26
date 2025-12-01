#pragma once

#include <format>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>

namespace small_glim::logger {

constexpr std::string LOGGER_NAME = "small_glim";

template<typename... Args>
inline void debug(const char* name, const std::format_string<Args...>& fmt, Args&&... args) {
    RCLCPP_DEBUG(
        rclcpp::get_logger(LOGGER_NAME),
        "<%s>: %s", name, std::format(fmt, std::forward<Args>(args)...).c_str()
    );
}

template<typename... Args>
inline void info(const char* name, const std::format_string<Args...>& fmt, Args&&... args) {
    RCLCPP_INFO(
        rclcpp::get_logger(LOGGER_NAME),
        "<%s>: %s", name, std::format(fmt, std::forward<Args>(args)...).c_str()
    );
}

template<typename... Args>
inline void warn(const char* name, const std::format_string<Args...>& fmt, Args&&... args) {
    RCLCPP_WARN(
        rclcpp::get_logger(LOGGER_NAME),
        "<%s>: %s", name, std::format(fmt, std::forward<Args>(args)...).c_str()
    );
}

template<typename... Args>
inline void error(const char* name, const std::format_string<Args...>& fmt, Args&&... args) {
    RCLCPP_ERROR(
        rclcpp::get_logger(LOGGER_NAME),
        "<%s>: %s", name, std::format(fmt, std::forward<Args>(args)...).c_str()
    );
}

template<typename... Args>
inline void fatal(const char* name, const std::format_string<Args...>& fmt, Args&&... args) {
    RCLCPP_FATAL(
        rclcpp::get_logger(LOGGER_NAME),
        "<%s>: %s", name, std::format(fmt, std::forward<Args>(args)...).c_str()
    );
}

}