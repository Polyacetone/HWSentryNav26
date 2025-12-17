#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_ros/buffer.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>

#include <path_planner/nav_map.hpp>
#include <path_planner/a_star_planner.hpp>
#include <path_planner/bspline_optimizer.hpp>
#include <common_utils/convert.hpp>

namespace path_planner {
class PathPlannerNode: public rclcpp::Node {
public:
    explicit PathPlannerNode(const rclcpp::NodeOptions& options);

private:
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr cost_map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr direction_map_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr goal_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    CostMap::ConstPtr cost_map_;
    DirectionMap::ConstPtr direction_map_;
    geometry_msgs::msg::PointStamped::SharedPtr goal_;
    AStarPlanner::ConstPtr path_planner_;
    BSplineOptimizer::ConstPtr path_optimizer_;

    nav_msgs::msg::Path path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const;
    std::vector<Eigen::Vector2d> vv2i_to_vv2d(const std::vector<Eigen::Vector2i>& path) const;
    void timer_callback();
};

PathPlannerNode::PathPlannerNode(const rclcpp::NodeOptions& options): Node("path_planner", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    path_planner_ = std::make_shared<AStarPlanner>(
        declare_parameter<double>("path_planner.direction_weight"),
        declare_parameter<double>("path_planner.obstacle_weight"),
        declare_parameter<int>("path_planner.downsampled_waypoint_max_interval"),
        declare_parameter<int>("path_planner.occupied_threshold")
    );
    path_optimizer_ = std::make_shared<BSplineOptimizer>(
        declare_parameter<double>("path_optimizer.smoothness_weight"),
        declare_parameter<double>("path_optimizer.length_weight"),
        declare_parameter<double>("path_optimizer.obstacle_weight"),
        declare_parameter<double>("path_optimizer.direction_weight"),
        declare_parameter<double>("path_optimizer.num_samples_per_length"),
        declare_parameter<int>("path_optimizer.max_iterations")
    );
    std::string cost_map_sub_topic = declare_parameter<std::string>("cost_map_sub_topic");
    cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        cost_map_sub_topic, rclcpp::QoS(1),
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            cost_map_ = std::make_shared<CostMap>(*msg);
        }
    );
    std::string direction_map_sub_topic = declare_parameter<std::string>("direction_map_sub_topic");
    direction_map_sub_ = create_subscription<sensor_msgs::msg::Image>(
        direction_map_sub_topic, rclcpp::QoS(1),
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            cv::Mat img = cv_bridge::toCvShare(msg, "8UC2")->image;
            if (!cost_map_) {
                RCLCPP_WARN(get_logger(), "Received direction map but cost map is not ready yet!");
                return;
            }
            direction_map_ = std::make_shared<DirectionMap>(img, cost_map_->resolution, cost_map_->origin_x, cost_map_->origin_y);
        }
    );
    std::string goal_sub_topic = declare_parameter<std::string>("goal_sub_topic");
    goal_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
        goal_sub_topic, rclcpp::QoS(1),
        [this](const geometry_msgs::msg::PointStamped::SharedPtr msg) { goal_ = msg; }
    );
    std::string path_pub_topic = declare_parameter<std::string>("path_pub_topic");
    path_pub_ = create_publisher<nav_msgs::msg::Path>(path_pub_topic, 1);
    int plan_freq = declare_parameter<int>("plan_freq");
    timer_ = create_wall_timer(std::chrono::milliseconds(1000 / plan_freq), [this]() { timer_callback(); });
}

void PathPlannerNode::timer_callback() {
    if (!cost_map_ || !goal_ || !direction_map_) return;

    tf2::Transform base_to_map;
    try {
        base_to_map = utils::convert_to<tf2::Transform>(
            tf_buffer_->lookupTransform("map", "base", tf2::TimePointZero).transform
        );
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(), "Failed to lookup base to map: %s", ex.what());
        return;
    }

    const Eigen::Vector2i start_grid(cost_map_->map_coord_to_grid({base_to_map.getOrigin().x(), base_to_map.getOrigin().y()}).cast<int>());
    const Eigen::Vector2i goal_grid(cost_map_->map_coord_to_grid({goal_->point.x, goal_->point.y}).cast<int>());

    auto start = std::chrono::high_resolution_clock::now();
    // A*搜索得到初始路径，输入输出均为格点坐标系
    auto rough = path_planner_->search_path(*cost_map_, *direction_map_, start_grid, goal_grid);
    auto end = std::chrono::high_resolution_clock::now();
    RCLCPP_INFO(get_logger(), "Plan: %.2fms", (end - start).count() / 1e6);

    start = std::chrono::high_resolution_clock::now();
    auto rough_f64 = vv2i_to_vv2d(rough);
    // 优化路径，输入输出均为格点坐标系
    auto optimized = path_optimizer_->optimize(*cost_map_, *direction_map_, rough_f64);
    end = std::chrono::high_resolution_clock::now();
    RCLCPP_INFO(get_logger(), "Optimize: %.2fms", (end - start).count() / 1e6);

    // 最后再转换成map坐标系
    std::vector<Eigen::Vector2d> optimized_map;
    for (auto& pt: optimized) {
        optimized_map.push_back(cost_map_->grid_coord_to_map(pt));
    }
    path_pub_->publish(path_to_nav_msg(optimized_map));
}

std::vector<Eigen::Vector2d> PathPlannerNode::vv2i_to_vv2d(const std::vector<Eigen::Vector2i>& path) const {
    std::vector<Eigen::Vector2d> f64path;
    for (const auto& pt: path) {
        f64path.emplace_back(pt.cast<double>());
    }
    return f64path;
}

nav_msgs::msg::Path PathPlannerNode::path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const {
    nav_msgs::msg::Path msg;
    msg.header.stamp = now();
    msg.header.frame_id = "map";
    std::for_each(path.begin(), path.end(), [&msg](const auto& point) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.pose.position.x = point.x();
        pose.pose.position.y = point.y();
        msg.poses.emplace_back(pose);
    });
    return msg;
}
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(path_planner::PathPlannerNode)