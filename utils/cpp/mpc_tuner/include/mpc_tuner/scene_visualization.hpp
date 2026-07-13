#pragma once

#include <optional>
#include <span>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <mpc_tuner/types.hpp>

namespace mpc_tuner {

[[nodiscard]] nav_msgs::msg::OccupancyGrid make_cost_map_message(
    const MapSnapshot& map, const rclcpp::Time& stamp
);
[[nodiscard]] sensor_msgs::msg::Image make_direction_map_message(
    const MapSnapshot& map, const rclcpp::Time& stamp
);
[[nodiscard]] nav_msgs::msg::Path make_route_path(
    const StoredRoute& route, const rclcpp::Time& stamp
);
[[nodiscard]] visualization_msgs::msg::MarkerArray make_route_markers(
    std::span<const StoredRoute> routes,
    std::optional<size_t> selected_index,
    const StoredRoute* candidate,
    const rclcpp::Time& stamp
);

} // namespace mpc_tuner
