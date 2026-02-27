#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <nav_msgs/msg/odometry.hpp>

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
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr global_direction_map_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr goal_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr control_points_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_optimized_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_rough_path_pub_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    CostMap::ConstPtr global_cost_map_;
    DirectionMap::ConstPtr global_direction_map_;
    nav_msgs::msg::Odometry::SharedPtr last_odom_msg_;
    AStarPlanner::ConstPtr path_planner_;
    BSplineOptimizer::ConstPtr path_optimizer_;
    double lazy_distance_;
    bool enable_debug_;
    int occupied_threshold_;
    double on_step_threshold_;
    bool start_prediction_enable_;
    double start_prediction_max_accel_;
    double start_prediction_planning_delay_;
    double start_prediction_min_speed_;
    double start_prediction_collision_check_step_;

    nav_msgs::msg::Path path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const;
    void goal_callback(const geometry_msgs::msg::PointStamped::SharedPtr msg);
    void plan_new_path();
    void update_optimized_path();
    bool is_map_point_feasible(const Eigen::Vector2d& map_pt) const;
    Eigen::Vector2d adjust_reachable_start_on_segment(const Eigen::Vector2d& from_map, const Eigen::Vector2d& to_map) const;
    Eigen::Vector2d predict_start_map(const Eigen::Vector2d& current_map) const;
    void publish_path(
        const std::vector<Eigen::Vector2d>& control_points,
        const std::vector<Eigen::Vector2d>& rough_path,
        const std::vector<Eigen::Vector2d>& optimized_path
    );
};

PathPlannerNode::PathPlannerNode(const rclcpp::NodeOptions& options): Node("path_planner", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    lazy_distance_ = declare_parameter<double>("lazy_distance");
    enable_debug_ = declare_parameter<bool>("debug.enable");
    occupied_threshold_ = (int)declare_parameter<int>("occupied_threshold");
    on_step_threshold_ = declare_parameter<double>("on_step_threshold");
    start_prediction_enable_ = declare_parameter<bool>("start_prediction.enable");
    start_prediction_max_accel_ = declare_parameter<double>("start_prediction.max_accel");
    start_prediction_planning_delay_ = declare_parameter<double>("start_prediction.planning_delay");
    start_prediction_min_speed_ = declare_parameter<double>("start_prediction.min_speed");
    start_prediction_collision_check_step_ = declare_parameter<double>("start_prediction.collision_check_step");

    if (enable_debug_) {
        std::string rough_path_pub_topic = declare_parameter<std::string>("debug.rough_path_pub_topic");
        debug_rough_path_pub_ = create_publisher<nav_msgs::msg::Path>(rough_path_pub_topic, 1);
        std::string optimized_path_pub_topic = declare_parameter<std::string>("debug.optimized_path_pub_topic");
        debug_optimized_path_pub_ = create_publisher<nav_msgs::msg::Path>(optimized_path_pub_topic, 1);
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        RCLCPP_DEBUG(get_logger(), "Debug mode enabled");
    }
    path_planner_ = std::make_shared<AStarPlanner>(
        declare_parameter<double>("path_planner.direction_weight"),
        declare_parameter<double>("path_planner.obstacle_weight"),
        declare_parameter<double>("path_planner.step_weight"),
        declare_parameter<int>("path_planner.downsampled_waypoint_max_interval"),
        declare_parameter<int>("path_planner.feasible_threshold")
    );
    path_optimizer_ = std::make_shared<BSplineOptimizer>(
        declare_parameter<double>("path_optimizer.smoothness_weight"),
        declare_parameter<double>("path_optimizer.uniform_speed_weight"),
        declare_parameter<double>("path_optimizer.obstacle_weight"),
        declare_parameter<double>("path_optimizer.direction_weight"),
        declare_parameter<double>("path_optimizer.step_weight"),
        declare_parameter<double>("path_optimizer.step_norm_threshold"),
        declare_parameter<double>("path_optimizer.step_norm_transition"),
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
    std::string odom_sub_topic = declare_parameter<std::string>("odom_sub_topic");
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_sub_topic, 1,
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) { last_odom_msg_ = msg; }
    );
    std::string control_points_pub_topic = declare_parameter<std::string>("control_points_pub_topic");
    control_points_pub_ = create_publisher<nav_msgs::msg::Path>(control_points_pub_topic, 1);
}

bool PathPlannerNode::is_map_point_feasible(const Eigen::Vector2d& map_pt) const {
    if (!global_cost_map_ || !global_direction_map_) return false;
    const Eigen::Vector2d grid = global_cost_map_->map_coord_to_grid(map_pt);
    if (!global_cost_map_->is_valid_coord(grid) || !global_direction_map_->is_valid_coord(grid)) return false;
    if (global_cost_map_->interpolate(grid) > occupied_threshold_) return false;
    if (global_direction_map_->interpolate(grid).norm() > on_step_threshold_) return false;
    return true;
}

Eigen::Vector2d PathPlannerNode::adjust_reachable_start_on_segment(const Eigen::Vector2d& from_map, const Eigen::Vector2d& to_map) const {
    const Eigen::Vector2d delta = to_map - from_map;
    const double length = delta.norm();
    if (length <= 1e-6) return from_map;
    const double step = std::max(1e-3, start_prediction_collision_check_step_);
    const int n = std::max(1, static_cast<int>(std::ceil(length / step)));
    const Eigen::Vector2d dir = delta / length;

    Eigen::Vector2d last_feasible = from_map;
    for (int i = 0; i <= n; i++) {
        const double d = length * (static_cast<double>(i) / static_cast<double>(n));
        const Eigen::Vector2d pt = from_map + dir * d;
        if (!is_map_point_feasible(pt)) {
            break;
        }
        last_feasible = pt;
    }
    return last_feasible;
}

Eigen::Vector2d PathPlannerNode::predict_start_map(const Eigen::Vector2d& current_map) const {
    if (!start_prediction_enable_) return current_map;
    const auto odom = last_odom_msg_;
    if (!odom) return current_map;

    Eigen::Vector2d v_map(0.0, 0.0);
    const Eigen::Vector2d v_raw(odom->twist.twist.linear.x, odom->twist.twist.linear.y);
    try {
        const auto imu_link_to_map = tf_buffer_->lookupTransform("map", "imu_link", tf2::TimePointZero);
        const tf2::Matrix3x3 R(utils::convert_to<tf2::Transform>(imu_link_to_map.transform).getRotation());
        const tf2::Vector3 v = R * tf2::Vector3(v_raw.x(), v_raw.y(), 0.0);
        v_map = {v.x(), v.y()};
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(), "Failed to transform velocity to map frame: %s", ex.what());
        return current_map;
    }

    const double speed = v_map.norm();
    if (speed < std::max(0.0, start_prediction_min_speed_)) return current_map;

    const double brake_distance = speed * speed / (2.0 * start_prediction_max_accel_);
    const double delay_distance = speed * std::max(0.0, start_prediction_planning_delay_);
    const double total_distance = brake_distance + delay_distance;

    const Eigen::Vector2d dir = v_map / speed;
    const Eigen::Vector2d predicted = current_map + dir * total_distance;
    return adjust_reachable_start_on_segment(current_map, predicted);
}

void PathPlannerNode::goal_callback(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
    if (!global_cost_map_ || !global_direction_map_) { // 确保地图已经准备好
        RCLCPP_ERROR(get_logger(), "Map not ready yet!");
        publish_path({}, {}, {});
        return;
    }

    tf2::Transform chassis_to_map;
    try {
        chassis_to_map = utils::convert_to<tf2::Transform>(
            tf_buffer_->lookupTransform("map", "chassis_link", tf2::TimePointZero).transform
        );
    } catch (const std::exception& ex) {
        RCLCPP_ERROR(get_logger(), "Failed to lookup chassis_link to map: %s", ex.what());
        publish_path({}, {}, {});
        return;
    }

    const Eigen::Vector2d goal_map(msg->point.x, msg->point.y);
    const Eigen::Vector2d current_map(chassis_to_map.getOrigin().x(), chassis_to_map.getOrigin().y());

    if ((goal_map - current_map).norm() < lazy_distance_) {
        RCLCPP_INFO(get_logger(), "New goal is within lazy distance (%.2f m)", lazy_distance_);
        publish_path({}, {}, {});
        return;
    }

    const Eigen::Vector2d start_map = predict_start_map(current_map);
    const Eigen::Vector2d start_grid = global_cost_map_->map_coord_to_grid(start_map);
    const Eigen::Vector2d goal_grid = global_cost_map_->map_coord_to_grid(goal_map);

    if (!global_cost_map_->is_valid_coord(start_grid)) {
        RCLCPP_WARN(get_logger(), "Start (%.2f, %.2f) is out of bound!", start_map.x(), start_map.y());
        publish_path({}, {}, {});
        return;
    }

    if (!is_map_point_feasible(start_map)) {
        RCLCPP_ERROR(get_logger(), "Start (%.2f, %.2f) is not feasible!", start_map.x(), start_map.y());
        publish_path({}, {}, {});
        return;
    }

    if (!global_cost_map_->is_valid_coord(goal_grid)) {
        RCLCPP_WARN(get_logger(), "Goal (%.2f, %.2f) is out of bound!", goal_map.x(), goal_map.y());
        publish_path({}, {}, {});
        return;
    }

    if (!is_map_point_feasible(goal_map)) {
        RCLCPP_ERROR(get_logger(), "Goal (%.2f, %.2f) is not feasible!", goal_map.x(), goal_map.y());
        publish_path({}, {}, {});
        return;
    }

    RCLCPP_INFO(
        get_logger(), "New valid goal: Src(%.2f, %.2f) -> Dst(%.2f, %.2f)",
        start_map.x(), start_map.y(), goal_map.x(), goal_map.y()
    );

    auto start_time = std::chrono::high_resolution_clock::now();
    
    // A*搜索得到初始路径，输入输出均为格点坐标系
    const auto plan_result = path_planner_->search_path(
        *global_cost_map_,
        *global_direction_map_,
        start_grid.cast<int>(),
        goal_grid.cast<int>()
    );
    if (!plan_result) {
        RCLCPP_ERROR(get_logger(), "Path planning failed: %s", plan_result.error().c_str());
        publish_path({}, {}, {});
        return;
    }
    std::vector<Eigen::Vector2d> rough_path;
    std::for_each(plan_result->begin(), plan_result->end(), [&](const Eigen::Vector2i& pt) {
        rough_path.push_back(pt.cast<double>());
    });

    RCLCPP_DEBUG(
        get_logger(), "A* planning took %.2f ms, path length: %d",
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count(),
        static_cast<int>(rough_path.size())
    );
    start_time = std::chrono::high_resolution_clock::now();

    // 优化路径，输入输出均为格点坐标系
    std::vector<Eigen::Vector2d> control_points, optimized_path;
    const auto optimize_result = path_optimizer_->optimize(
        *global_cost_map_,
        *global_direction_map_,
        rough_path,
        start_grid,
        goal_grid
    );
    if (!optimize_result) {
        RCLCPP_ERROR(get_logger(), "Path optimization failed: %s", optimize_result.error().c_str());
        publish_path({}, {}, {});
        return;
    }
    std::tie(control_points, optimized_path) = *optimize_result;

    RCLCPP_DEBUG(
        get_logger(), "Path optimization took %.2f ms, control points: %d, optimized path length: %d",
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count(),
        static_cast<int>(control_points.size()),
        static_cast<int>(optimized_path.size())
    );

    publish_path(control_points, rough_path, optimized_path);
}

void PathPlannerNode::publish_path(
    const std::vector<Eigen::Vector2d>& control_points,
    const std::vector<Eigen::Vector2d>& rough_path,
    const std::vector<Eigen::Vector2d>& optimized_path
) {
    // 发布路径转换到地图坐标系
    const auto to_map_coord = [this](const std::vector<Eigen::Vector2d>& points) {
        std::vector<Eigen::Vector2d> map_points;
        for (const auto& pt: points) {
            map_points.push_back(global_cost_map_->grid_coord_to_map(pt));
        }
        return map_points;
    };
    control_points_pub_->publish(path_to_nav_msg(to_map_coord(control_points)));
    if (enable_debug_) {
        debug_rough_path_pub_->publish(path_to_nav_msg(to_map_coord(rough_path)));
        debug_optimized_path_pub_->publish(path_to_nav_msg(to_map_coord(optimized_path)));
    }
}

nav_msgs::msg::Path PathPlannerNode::path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const {
    nav_msgs::msg::Path msg;
    msg.header.stamp = now();
    msg.header.frame_id = "map";
    for (const auto& p: path) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = msg.header;
        ps.pose.position.x = p.x();
        ps.pose.position.y = p.y();
        ps.pose.position.z = 0.0;
        msg.poses.push_back(ps);
    }
    return msg;
}
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(path_planner::PathPlannerNode)