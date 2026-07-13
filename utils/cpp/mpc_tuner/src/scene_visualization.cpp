#include <mpc_tuner/scene_visualization.hpp>

#include <algorithm>
#include <cmath>

#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>

namespace mpc_tuner {
namespace {

std_msgs::msg::ColorRGBA color(const float r, const float g, const float b, const float a = 1.0f) {
    std_msgs::msg::ColorRGBA result;
    result.r = r;
    result.g = g;
    result.b = b;
    result.a = a;
    return result;
}

void append_route_markers(
    visualization_msgs::msg::MarkerArray& array,
    const StoredRoute& route,
    int& next_id,
    const bool selected,
    const bool candidate,
    const rclcpp::Time& stamp
) {
    const auto route_color = candidate ? color(1.0f, 0.75f, 0.05f)
        : selected ? color(0.1f, 0.85f, 0.35f) : color(0.55f, 0.58f, 0.62f, 0.55f);

    visualization_msgs::msg::Marker line;
    line.header.frame_id = "map";
    line.header.stamp = stamp;
    line.ns = "routes";
    line.id = next_id++;
    line.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line.action = visualization_msgs::msg::Marker::ADD;
    line.scale.x = selected ? 0.07 : 0.04;
    line.color = route_color;
    const nav_executor::SplinePath spline(route.spline_control_points);
    for (int i = 0; i <= 200; ++i) {
        const Eigen::Vector2d point = spline.position(static_cast<double>(i) / 200.0);
        geometry_msgs::msg::Point ros_point;
        ros_point.x = point.x();
        ros_point.y = point.y();
        line.points.push_back(ros_point);
    }
    array.markers.push_back(std::move(line));

    visualization_msgs::msg::Marker start;
    start.header.frame_id = "map";
    start.header.stamp = stamp;
    start.ns = "route_starts";
    start.id = next_id++;
    start.type = visualization_msgs::msg::Marker::ARROW;
    start.action = visualization_msgs::msg::Marker::ADD;
    start.pose.position.x = route.spec.start_pose.x();
    start.pose.position.y = route.spec.start_pose.y();
    start.pose.orientation.z = std::sin(route.spec.start_pose.z() * 0.5);
    start.pose.orientation.w = std::cos(route.spec.start_pose.z() * 0.5);
    start.scale.x = 0.45;
    start.scale.y = 0.12;
    start.scale.z = 0.12;
    start.color = route_color;
    array.markers.push_back(std::move(start));

    visualization_msgs::msg::Marker goal;
    goal.header.frame_id = "map";
    goal.header.stamp = stamp;
    goal.ns = "route_goals";
    goal.id = next_id++;
    goal.type = visualization_msgs::msg::Marker::SPHERE;
    goal.action = visualization_msgs::msg::Marker::ADD;
    goal.pose.position.x = route.spec.goal.x();
    goal.pose.position.y = route.spec.goal.y();
    goal.scale.x = goal.scale.y = goal.scale.z = 0.25;
    goal.color = route_color;
    array.markers.push_back(std::move(goal));

    visualization_msgs::msg::Marker label;
    label.header.frame_id = "map";
    label.header.stamp = stamp;
    label.ns = "route_labels";
    label.id = next_id++;
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    label.pose.position.x = route.spec.goal.x();
    label.pose.position.y = route.spec.goal.y();
    label.pose.position.z = 0.35;
    label.scale.z = 0.22;
    label.color = route_color;
    label.text = candidate ? route.spec.name + " (candidate)" : route.spec.name;
    array.markers.push_back(std::move(label));
}

} // namespace

nav_msgs::msg::OccupancyGrid make_cost_map_message(
    const MapSnapshot& map, const rclcpp::Time& stamp
) {
    nav_msgs::msg::OccupancyGrid message;
    message.header.frame_id = "map";
    message.header.stamp = stamp;
    message.info.width = static_cast<uint32_t>(map.width);
    message.info.height = static_cast<uint32_t>(map.height);
    message.info.resolution = static_cast<float>(map.resolution);
    message.info.origin.position.x = map.origin_x;
    message.info.origin.position.y = map.origin_y;
    message.info.origin.orientation.w = 1.0;
    message.data.assign(map.global_cost_data.begin(), map.global_cost_data.end());
    return message;
}

sensor_msgs::msg::Image make_direction_map_message(
    const MapSnapshot& map, const rclcpp::Time& stamp
) {
    sensor_msgs::msg::Image message;
    message.header.frame_id = "map";
    message.header.stamp = stamp;
    message.width = static_cast<uint32_t>(map.width);
    message.height = static_cast<uint32_t>(map.height);
    message.encoding = "8UC3";
    message.is_bigendian = false;
    message.step = static_cast<sensor_msgs::msg::Image::_step_type>(map.width * 3);
    message.data = map.direction_image_data;
    return message;
}

nav_msgs::msg::Path make_route_path(const StoredRoute& route, const rclcpp::Time& stamp) {
    nav_msgs::msg::Path message;
    message.header.frame_id = "map";
    message.header.stamp = stamp;
    const nav_executor::SplinePath spline(route.spline_control_points);
    for (int i = 0; i <= 200; ++i) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header = message.header;
        const auto evaluation = spline.eval(static_cast<double>(i) / 200.0);
        pose.pose.position.x = evaluation.p.x();
        pose.pose.position.y = evaluation.p.y();
        pose.pose.orientation.z = std::sin(evaluation.thetar * 0.5);
        pose.pose.orientation.w = std::cos(evaluation.thetar * 0.5);
        message.poses.push_back(std::move(pose));
    }
    return message;
}

visualization_msgs::msg::MarkerArray make_route_markers(
    const std::span<const StoredRoute> routes,
    const std::optional<size_t> selected_index,
    const StoredRoute* candidate,
    const rclcpp::Time& stamp
) {
    visualization_msgs::msg::MarkerArray array;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    array.markers.push_back(clear);
    int next_id = 1;
    for (size_t i = 0; i < routes.size(); ++i) {
        append_route_markers(array, routes[i], next_id, selected_index == i, false, stamp);
    }
    if (candidate) append_route_markers(array, *candidate, next_id, true, true, stamp);
    return array;
}

} // namespace mpc_tuner
