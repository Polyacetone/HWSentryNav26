#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_ros/buffer.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <common_utils/convert.hpp>

namespace map_server {
class MapServerNode: public rclcpp::Node {
public:
    explicit MapServerNode(const rclcpp::NodeOptions& options);

private:
    double map_resolution_;
    int map_size_x_, map_size_y_;

    cv::Mat global_direction_map_, global_cost_map_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr direction_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr cost_map_pub_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    rclcpp::TimerBase::SharedPtr timer_;

    void timer_callback();
    void pub_direction_map(const cv::Mat& direction_map, const rclcpp::Time& stamp) const;
    void pub_cost_map(const cv::Mat& cost_map, const rclcpp::Time& stamp) const;
};

MapServerNode::MapServerNode(const rclcpp::NodeOptions& options): Node("map_server_node", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    map_resolution_ = declare_parameter<double>("map_resolution");
    map_size_x_ = declare_parameter<int>("map_size_x");
    map_size_y_ = declare_parameter<int>("map_size_y");
    std::string direction_map_pub_topic = declare_parameter<std::string>("direction_map_pub_topic");
    direction_map_pub_ = create_publisher<sensor_msgs::msg::Image>(direction_map_pub_topic, 1);
    std::string cost_map_pub_topic = declare_parameter<std::string>("cost_map_pub_topic");
    cost_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(cost_map_pub_topic, 1);
    std::string global_map_filename = declare_parameter<std::string>("global_map_filename");
    std::string global_map_path = ament_index_cpp::get_package_share_directory("map_server") + "/maps/" + global_map_filename;
    cv::Mat global_map = cv::imread(global_map_path, cv::IMREAD_COLOR);
    if (global_map.empty()) {
        RCLCPP_FATAL(get_logger(), "Failed to load global navmap from %s", global_map_path.c_str());
        throw std::runtime_error("Failed to load global navmap");
    }
    std::vector<cv::Mat> channels;
    cv::split(global_map, channels);
    cv::merge(std::array{channels[0], channels[1]}, global_direction_map_); // 前两个通道表示台阶方向
    global_cost_map_ = channels[2]; // 第三个通道表示代价地图
    timer_ = create_wall_timer(
        std::chrono::seconds(1),
        [this] { timer_callback(); }
    );
}

void MapServerNode::timer_callback() {
    pub_direction_map(global_direction_map_, now());
    pub_cost_map(global_cost_map_, now());
}

void MapServerNode::pub_direction_map(const cv::Mat& direction_map, const rclcpp::Time& stamp) const {
    sensor_msgs::msg::Image::SharedPtr direction_map_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "8UC2", direction_map).toImageMsg();
    direction_map_msg->header.stamp = stamp;
    direction_map_msg->header.frame_id = "map";
    direction_map_pub_->publish(*direction_map_msg);
}

void MapServerNode::pub_cost_map(const cv::Mat& cost_map, const rclcpp::Time& stamp) const {
    nav_msgs::msg::OccupancyGrid occupancy_grid;
    occupancy_grid.header.frame_id = "map";
    occupancy_grid.header.stamp = stamp;
    occupancy_grid.info.resolution = map_resolution_;
    occupancy_grid.info.height = map_size_y_;
    occupancy_grid.info.width = map_size_x_;
    occupancy_grid.data.resize(map_size_x_ * map_size_y_);
    std::copy(cost_map.data, cost_map.data + map_size_x_ * map_size_y_, occupancy_grid.data.data());
    cost_map_pub_->publish(occupancy_grid);
}
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(map_server::MapServerNode)