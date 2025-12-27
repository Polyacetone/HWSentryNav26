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
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr global_direction_map_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr local_cost_map_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr goal_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_rough_path_pub_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    CostMap::ConstPtr global_cost_map_;
    DirectionMap::ConstPtr global_direction_map_;
    Eigen::Vector2d start_grid_, goal_grid_;
    std::vector<Eigen::Vector2d> rough_path_, optimized_path_;
    AStarPlanner::ConstPtr path_planner_;
    BSplineOptimizer::ConstPtr path_optimizer_;
    double feasible_threshold_;
    bool enable_debug_;

    nav_msgs::msg::Path path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const;
    void goal_callback(const geometry_msgs::msg::PointStamped::SharedPtr msg);
    void local_cost_map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    bool check_optimized_path_feasibility(const CostMap& final_cost_map) const;
};

PathPlannerNode::PathPlannerNode(const rclcpp::NodeOptions& options): Node("path_planner", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    feasible_threshold_ = declare_parameter<int>("feasible_threshold");
    enable_debug_ = declare_parameter<bool>("debug_mode.enable");
    if (enable_debug_) {
        std::string rough_path_pub_topic = declare_parameter<std::string>("debug_mode.rough_path_pub_topic");
        debug_rough_path_pub_ = create_publisher<nav_msgs::msg::Path>(rough_path_pub_topic, 1);
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        RCLCPP_DEBUG(get_logger(), "Debug mode enabled");
    }
    path_planner_ = std::make_shared<AStarPlanner>(
        declare_parameter<double>("path_planner.direction_weight"),
        declare_parameter<double>("path_planner.obstacle_weight"),
        declare_parameter<int>("path_planner.downsampled_waypoint_max_interval"),
        feasible_threshold_
    );
    path_optimizer_ = std::make_shared<BSplineOptimizer>(
        declare_parameter<double>("path_optimizer.smoothness_weight"),
        declare_parameter<double>("path_optimizer.length_weight"),
        declare_parameter<double>("path_optimizer.obstacle_weight"),
        declare_parameter<double>("path_optimizer.direction_weight"),
        declare_parameter<double>("path_optimizer.start_end_weight"),
        declare_parameter<double>("path_optimizer.num_samples_per_length"),
        declare_parameter<int>("path_optimizer.max_iterations")
    );
    std::string global_cost_map_sub_topic = declare_parameter<std::string>("global_cost_map_sub_topic");
    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        global_cost_map_sub_topic, 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            global_cost_map_ = std::make_shared<CostMap>(*msg);
            RCLCPP_INFO(get_logger(), "Received global cost map: size=(%d, %d), resolution=%.2f", global_cost_map_->width, global_cost_map_->height, global_cost_map_->resolution);
            global_cost_map_sub_.reset();  // 全局代价地图只需要接收一次
        }
    );
    std::string global_direction_map_sub_topic = declare_parameter<std::string>("global_direction_map_sub_topic");
    global_direction_map_sub_ = create_subscription<sensor_msgs::msg::Image>(
        global_direction_map_sub_topic, 1,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            cv::Mat img = cv_bridge::toCvShare(msg, "8UC2")->image;
            if (!global_cost_map_) {
                RCLCPP_WARN(get_logger(), "Received direction map but cost map is not ready yet!");
                return;
            }
            global_direction_map_ = std::make_shared<DirectionMap>(img, global_cost_map_->resolution, global_cost_map_->origin_x, global_cost_map_->origin_y);
            if (global_direction_map_->width != global_cost_map_->width || global_direction_map_->height != global_cost_map_->height) {
                RCLCPP_FATAL(get_logger(), "Direction map size (%d, %d) does not match cost map size (%d, %d)!", global_direction_map_->width, global_direction_map_->height, global_cost_map_->width, global_cost_map_->height);
                throw std::runtime_error("Direction map size does not match cost map size");
            }
            RCLCPP_INFO(get_logger(), "Received global direction map");
            global_direction_map_sub_.reset();  // 全局方向地图只需要接收一次
        }
    );
    std::string goal_sub_topic = declare_parameter<std::string>("goal_sub_topic");
    goal_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
        goal_sub_topic, 1,
        [this](const geometry_msgs::msg::PointStamped::SharedPtr msg) { goal_callback(msg); }
    );
    std::string local_cost_map_sub_topic = declare_parameter<std::string>("local_cost_map_sub_topic");
    local_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        local_cost_map_sub_topic, 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) { local_cost_map_callback(msg); }
    );
    std::string path_pub_topic = declare_parameter<std::string>("path_pub_topic");
    path_pub_ = create_publisher<nav_msgs::msg::Path>(path_pub_topic, 1);
}

void PathPlannerNode::goal_callback(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
    if (!global_cost_map_ || !global_direction_map_) {
        RCLCPP_WARN(get_logger(), "Cost map or direction map not ready yet!");
        return;
    }

    tf2::Transform chassis_to_map;
    try {
        chassis_to_map = utils::convert_to<tf2::Transform>(
            tf_buffer_->lookupTransform("map", "chassis_link", tf2::TimePointZero).transform
        );
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(), "Failed to lookup chassis_link to map: %s", ex.what());
        return;
    }

    RCLCPP_INFO(get_logger(), "Received new goal: (%.2f, %.2f) -> (%.2f, %.2f)", chassis_to_map.getOrigin().x(), chassis_to_map.getOrigin().y(), msg->point.x, msg->point.y);
    start_grid_ = global_cost_map_->map_coord_to_grid({chassis_to_map.getOrigin().x(), chassis_to_map.getOrigin().y()});
    goal_grid_ = global_cost_map_->map_coord_to_grid({msg->point.x, msg->point.y});
    
    // A*搜索得到初始路径，输入输出均为格点坐标系
    auto rough = path_planner_->search_path(
        *global_cost_map_,
        *global_direction_map_,
        start_grid_.cast<int>(),
        goal_grid_.cast<int>()
    );
    rough_path_.clear();
    std::for_each(rough.begin(), rough.end(), [this](const Eigen::Vector2i& pt) {
        rough_path_.push_back(pt.cast<double>());
    });
    optimized_path_.clear();

    RCLCPP_DEBUG(get_logger(), "New rough path size: %zu", rough_path_.size());
    // 发布调试用的原始路径
    if (enable_debug_) {
        std::vector<Eigen::Vector2d> rough_path_map;
        for (auto& pt: rough_path_) {
            rough_path_map.push_back(global_cost_map_->grid_coord_to_map(pt));
        }
        debug_rough_path_pub_->publish(path_to_nav_msg(rough_path_map));
    }
}

void PathPlannerNode::local_cost_map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    if (rough_path_.empty()) { // 还没有收到目标点且全局地图，或者A*没有找到路径时发布空路径
        path_pub_->publish(path_to_nav_msg({}));
        return;
    }

    CostMap local_cost_map(*msg);
    CostMap final_cost_map = global_cost_map_->merge(local_cost_map);
    if (check_optimized_path_feasibility(local_cost_map)) { // 注意这里使用的是local_cost_map
        // 原来的优化路径仍然可行，直接发布
        path_pub_->publish(path_to_nav_msg(optimized_path_));
        return;
    }

    // 优化路径，输入输出均为格点坐标系
    auto optimized = path_optimizer_->optimize(
        final_cost_map,
        *global_direction_map_,
        rough_path_,
        start_grid_,
        goal_grid_
    );

    // 最后转换成map坐标系发布
    optimized_path_.clear();
    for (auto& pt: optimized) {
        optimized_path_.push_back(global_cost_map_->grid_coord_to_map(pt));
    }
    path_pub_->publish(path_to_nav_msg(optimized_path_));
    RCLCPP_DEBUG(get_logger(), "New optimized path size: %zu", optimized_path_.size());
}

bool PathPlannerNode::check_optimized_path_feasibility(const CostMap& final_cost_map) const {
    if (optimized_path_.empty()) {
        RCLCPP_DEBUG(get_logger(), "Optimized path not feasible: empty path");
        return false;
    }
    for (const auto& pt: optimized_path_) {
        double cost = final_cost_map.at(final_cost_map.map_coord_to_grid(pt).cast<int>());
        if (cost >= feasible_threshold_) {
            RCLCPP_DEBUG(get_logger(), "Optimized path not feasible: point (%.2f, %.2f) has cost %.2f", pt.x(), pt.y(), cost);
            return false;
        }
    }
    return true;
}

nav_msgs::msg::Path PathPlannerNode::path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const {
    nav_msgs::msg::Path msg;
    msg.header.stamp = now();
    msg.header.frame_id = "map";
    for (int i = 0; i < path.size(); i++) {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = msg.header;
        pose_stamped.pose.position.x = path[i].x();
        pose_stamped.pose.position.y = path[i].y();
        pose_stamped.pose.position.z = 0.0;
        Eigen::Vector2d dir;
        if (i == 0) dir = path[i + 1] - path[i];
        else if (i == path.size() - 1) dir = path[i] - path[i - 1];
        else dir = path[i + 1] - path[i - 1];
        dir.normalize();
        double yaw = std::atan2(dir.y(), dir.x());
        pose_stamped.pose.orientation = utils::convert_to<geometry_msgs::msg::Quaternion>(
            tf2::Quaternion(tf2::Vector3(0, 0, 1), yaw)
        );
        msg.poses.push_back(pose_stamped);
    }
    return msg;
}
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(path_planner::PathPlannerNode)