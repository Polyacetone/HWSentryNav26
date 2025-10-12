#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_ros/buffer.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>

#include <path_planner/costmap_2d.hpp>
#include <path_planner/a_star_planner.hpp>
#include <path_planner/bspline_optimizer.hpp>
#include <common_utils/tf_utils.hpp>

namespace path_planner {
class PathPlannerNode: public rclcpp::Node {
public:
    explicit PathPlannerNode(const rclcpp::NodeOptions& options);

private:
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    nav_msgs::msg::OccupancyGrid::SharedPtr map_;
    geometry_msgs::msg::PoseStamped::SharedPtr goal_;
    std::shared_ptr<AStarPlanner> a_star_planner_;
    std::shared_ptr<BSplineOptimizer> bspline_optimizer_;

    nav_msgs::msg::Path path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const;
    std::vector<Eigen::Vector2d> vv2i_to_vv2d(const std::vector<Eigen::Vector2i>& path) const;
    void timer_callback();
};

PathPlannerNode::PathPlannerNode(const rclcpp::NodeOptions& options): Node("path_planner_node") {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    int downsampled_waypoint_max_interval = declare_parameter<int>("a_star_planner.downsampled_waypoint_max_interval");
    int occupied_threshold = declare_parameter<int>("a_star_planner.occupied_threshold");
    a_star_planner_ = std::make_shared<AStarPlanner>(downsampled_waypoint_max_interval, occupied_threshold);
    double num_samples_per_length = declare_parameter<double>("bspline_optimizer.num_samples_per_length");
    double obstable_weight = declare_parameter<double>("bspline_optimizer.obstable_weight");
    double length_weight = declare_parameter<double>("bspline_optimizer.length_weight");
    double smooth_weight = declare_parameter<double>("bspline_optimizer.smooth_weight");
    bspline_optimizer_ = std::make_shared<BSplineOptimizer>(num_samples_per_length, obstable_weight, length_weight, smooth_weight);
    std::string costmap_sub_topic = declare_parameter<std::string>("costmap_sub_topic");
    costmap_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        costmap_sub_topic, rclcpp::QoS(1),
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) { map_ = msg; }
    );
    std::string goal_sub_topic = declare_parameter<std::string>("goal_sub_topic");
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        goal_sub_topic, rclcpp::QoS(1),
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { goal_ = msg; }
    );
    std::string path_pub_topic = declare_parameter<std::string>("path_pub_topic");
    path_pub_ = create_publisher<nav_msgs::msg::Path>(path_pub_topic, 1);
    int plan_freq = declare_parameter<int>("plan_freq");
    timer_ = create_wall_timer(std::chrono::milliseconds(1000 / plan_freq), [this]() { timer_callback(); });
}

void PathPlannerNode::timer_callback() {
    if (!map_ || !goal_) return;

    tf2::Transform base_to_map;
    try {
        base_to_map = utils::convert_to<tf2::Transform>(
            tf_buffer_->lookupTransform("map", "base", tf2::TimePointZero).transform
        );
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(), "Failed to lookup base to map: %s", ex.what());
        return;
    }

    Costmap2D costmap(*map_);
    const Eigen::Vector2i start_grid(costmap.map_coord_to_grid({base_to_map.getOrigin().x(), base_to_map.getOrigin().y()}).cast<int>());
    const Eigen::Vector2i goal_grid(costmap.map_coord_to_grid({goal_->pose.position.x, goal_->pose.position.y}).cast<int>());

    auto start = std::chrono::high_resolution_clock::now();
    // 在costmap的格点坐标系下搜索，返回的路径也是基于格点坐标系的
    auto rough = a_star_planner_->search_path(costmap, start_grid, goal_grid);
    auto end = std::chrono::high_resolution_clock::now();
    RCLCPP_INFO(get_logger(), "Astar plan: %.2fms", (end - start).count() / 1e6);

    start = std::chrono::high_resolution_clock::now();
    // 在格点坐标系下进行路径优化
    auto optimized = bspline_optimizer_->optimize(costmap, vv2i_to_vv2d(rough));
    end = std::chrono::high_resolution_clock::now();
    RCLCPP_INFO(get_logger(), "Bspline opt: %.2fms", (end - start).count() / 1e6);

    // 最后再转换成map坐标系
    for (auto& pt: optimized) pt = costmap.grid_coord_to_map(pt);
    path_pub_->publish(path_to_nav_msg(optimized));
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