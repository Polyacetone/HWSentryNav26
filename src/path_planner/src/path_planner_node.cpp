#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>

#include <path_planner/costmap_2d.hpp>
#include <path_planner/a_star_planner.hpp>
#include <path_planner/bspline_optimizer.hpp>

namespace path_planner {
class PathPlannerNode: public rclcpp::Node {
public:
    explicit PathPlannerNode(const rclcpp::NodeOptions& options);

private:
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr start_sub_, goal_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    nav_msgs::msg::OccupancyGrid::SharedPtr map_;
    geometry_msgs::msg::PoseStamped::SharedPtr goal_, start_;

    nav_msgs::msg::Path path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const;
    std::vector<Eigen::Vector2d> vv2i_to_vv2d(const std::vector<Eigen::Vector2i>& path) const;
    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) { map_ = msg; }
    void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) { goal_ = msg; }
    void start_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) { start_ = msg; }
    void timer_callback();
};

PathPlannerNode::PathPlannerNode(const rclcpp::NodeOptions& options): Node("path_planner_node") {
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        "global_costmap", rclcpp::QoS(1),
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) { map_callback(msg); }
    );
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "goal", rclcpp::QoS(1),
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { goal_callback(msg); }
    );
    start_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "start", rclcpp::QoS(1),
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { start_callback(msg); }
    );
    path_pub_ = create_publisher<nav_msgs::msg::Path>("path", 1);
    timer_ = create_wall_timer(std::chrono::milliseconds(1000), [this]() { timer_callback(); });
}

void PathPlannerNode::timer_callback() {
    if (!map_ || !start_ || !goal_) return;
    Costmap2D costmap(*map_);

    auto start = std::chrono::high_resolution_clock::now();
    AStarPlanner a_star;
    auto path = a_star.search_path(costmap, *start_, *goal_);
    auto end = std::chrono::high_resolution_clock::now();
    RCLCPP_INFO(get_logger(), "Astar plan: %.2fms", (end - start).count() / 1e6);

    start = std::chrono::high_resolution_clock::now();
    BSplineOptimizer optimizer;
    auto optimized = optimizer.optimize(costmap, vv2i_to_vv2d(path));
    end = std::chrono::high_resolution_clock::now();
    RCLCPP_INFO(get_logger(), "Bspline opt: %.2fms", (end - start).count() / 1e6);
    
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