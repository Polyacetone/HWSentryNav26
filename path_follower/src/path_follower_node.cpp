#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <interfaces/msg/chassis_status.hpp>
#include <interfaces/msg/chassis_cmd.hpp>

#include <common_utils/convert.hpp>
#include <path_follower/nav_map.hpp>
#include <path_follower/teb_controller.hpp>

namespace path_follower {
class PathFollowerNode: public rclcpp::Node {
public:
    explicit PathFollowerNode(const rclcpp::NodeOptions& options);

private:
    void control_points_callback(const nav_msgs::msg::Path::SharedPtr msg);
    void chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg);
    void local_cost_map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void control_timer_callback();
    void publish_chassis_cmd(double velocity, double palstance, bool step_ahead);

    bool get_current_pose(Eigen::Vector3d& current_pose) const;
    nav_msgs::msg::Path path_to_nav_msg(const std::vector<Eigen::Vector2d>& points) const;
    
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr control_points_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr local_cost_map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr global_direction_map_sub_;

    rclcpp::Subscription<interfaces::msg::ChassisStatus>::SharedPtr chassis_status_sub_;
    rclcpp::Publisher<interfaces::msg::ChassisCmd>::SharedPtr chassis_cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_predicted_path_pub_;

    rclcpp::TimerBase::SharedPtr control_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    
    double stop_threshold_;
    bool enable_debug_;
    TebParams params_;
    std::unique_ptr<TebController> teb_controller_;

    CostMap::ConstPtr global_cost_map_, local_cost_map_, merged_cost_map_;
    DirectionMap::ConstPtr global_direction_map_;
    Eigen::Vector2d current_state_ = Eigen::Vector2d::Zero(); // 从串口接收的当前(v, omega)
};

PathFollowerNode::PathFollowerNode(const rclcpp::NodeOptions& options): Node("path_follower", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    stop_threshold_ = declare_parameter<double>("stop_threshold");
    enable_debug_ = declare_parameter<bool>("debug.enable");
    if (enable_debug_) {
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        RCLCPP_DEBUG(get_logger(), "Debug mode enabled");
        debug_predicted_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("debug.predicted_path_pub_topic"), 1);
    }

    params_ = {
        .horizon = (int)declare_parameter<int>("teb.horizon"),
        .dt = declare_parameter<double>("teb.dt"),
        .max_iterations = (int)declare_parameter<int>("teb.max_iterations"),
        .num_threads = (int)declare_parameter<int>("teb.num_threads"),
        .vel_max = declare_parameter<double>("teb.thresholds.vel_max"),
        .vel_min = declare_parameter<double>("teb.thresholds.vel_min"),
        .omega_max = declare_parameter<double>("teb.thresholds.omega_max"),
        .omega_min = declare_parameter<double>("teb.thresholds.omega_min"),
        .acc_max = declare_parameter<double>("teb.thresholds.acc_max"),
        .alpha_max = declare_parameter<double>("teb.thresholds.alpha_max"),
        .q_y = declare_parameter<double>("teb.weights.q_y"),
        .q_theta = declare_parameter<double>("teb.weights.q_theta"),
        .q_u = declare_parameter<double>("teb.weights.q_u"),
        .q_v_final = declare_parameter<double>("teb.weights.q_v_final"),
        .r_v = declare_parameter<double>("teb.weights.r_v"),
        .r_omega = declare_parameter<double>("teb.weights.r_omega"),
        .r_dv = declare_parameter<double>("teb.weights.r_dv"),
        .r_domega = declare_parameter<double>("teb.weights.r_domega"),
        .acc_limit_weight = declare_parameter<double>("teb.weights.acc_limit"),
        .alpha_limit_weight = declare_parameter<double>("teb.weights.alpha_limit"),
        .obstacle_weight = declare_parameter<double>("teb.weights.obstacle"),
        .direction_weight = declare_parameter<double>("teb.weights.direction"),
        .proj_num_samples = (int)declare_parameter<int>("teb.projection.num_samples"),
        .proj_search_window = declare_parameter<double>("teb.projection.search_window"),
        .max_correspondence_distance = declare_parameter<double>("teb.projection.max_correspondence_distance")
    };
    teb_controller_ = std::make_unique<TebController>(params_);

    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("global_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            global_cost_map_ = std::make_shared<CostMap>(*msg);
            RCLCPP_INFO(get_logger(), "Received global cost map: size=(%d,%d), resolution=%.3f", global_cost_map_->width, global_cost_map_->height, global_cost_map_->resolution);
            global_cost_map_sub_.reset();
        }
    );
    global_direction_map_sub_ = create_subscription<sensor_msgs::msg::Image>(
        declare_parameter<std::string>("global_direction_map_sub_topic"), 1,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            if (!global_cost_map_) {
                RCLCPP_WARN(get_logger(), "Received direction map but global cost map is not ready yet!");
                return;
            }
            cv::Mat img = cv_bridge::toCvShare(msg, "8UC2")->image;
            global_direction_map_ = std::make_shared<DirectionMap>(img, global_cost_map_->resolution, global_cost_map_->origin_x, global_cost_map_->origin_y);
            if (global_direction_map_->width != global_cost_map_->width || global_direction_map_->height != global_cost_map_->height) {
                RCLCPP_FATAL(get_logger(), "Direction map size (%d,%d) does not match cost map (%d,%d)!", global_direction_map_->width, global_direction_map_->height, global_cost_map_->width, global_cost_map_->height);
                throw std::runtime_error("Direction map size does not match cost map size");
            }
            RCLCPP_INFO(get_logger(), "Received global direction map");
            global_direction_map_sub_.reset();
        }
    );
    local_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("local_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) { local_cost_map_callback(msg); }
    );
    control_points_sub_ = create_subscription<nav_msgs::msg::Path>(
        declare_parameter<std::string>("control_points_sub_topic"), 1,
        [this](const nav_msgs::msg::Path::SharedPtr msg) { control_points_callback(msg); }
    );
    chassis_status_sub_ = create_subscription<interfaces::msg::ChassisStatus>(
        declare_parameter<std::string>("chassis_status_sub_topic"), 1,
        [this](const interfaces::msg::ChassisStatus::SharedPtr msg) { chassis_status_callback(msg); }
    );
    chassis_cmd_pub_ = create_publisher<interfaces::msg::ChassisCmd>(declare_parameter<std::string>("chassis_cmd_pub_topic"), 1);
    control_timer_ = create_wall_timer(std::chrono::duration<double>(params_.dt), [this]() { control_timer_callback(); });
}

void PathFollowerNode::control_points_callback(const nav_msgs::msg::Path::SharedPtr msg) {
    if (msg->poses.size() < 3) {
        if (!msg->poses.empty()) {
            RCLCPP_WARN(get_logger(), "Received insufficient control points (%zu), need at least 3!", msg->poses.size());
        }
        teb_controller_->set_reference_path({});
        return;
    }
    std::vector<Eigen::Vector2d> path;
    path.reserve(msg->poses.size());
    for (const auto& ps: msg->poses) {
        path.emplace_back(ps.pose.position.x, ps.pose.position.y);
    }
    teb_controller_->set_reference_path(path);
}

void PathFollowerNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    current_state_.x() = msg->velocity;
    current_state_.y() = msg->palstance;
}

void PathFollowerNode::local_cost_map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    if (!global_cost_map_) return;
    local_cost_map_ = std::make_shared<CostMap>(*msg);
    try {
        merged_cost_map_ = std::make_shared<CostMap>(global_cost_map_->merge(*local_cost_map_));
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to merge cost maps: %s", e.what());
    }
}

void PathFollowerNode::control_timer_callback() {
    if (!teb_controller_->has_reference_path()) {
        publish_chassis_cmd(0, 0, false);
        return;
    }

    Eigen::Vector3d current_pose;
    if (!get_current_pose(current_pose)) return;

    if ((teb_controller_->get_destination() - current_pose.head<2>()).norm() < stop_threshold_) {
        RCLCPP_INFO(get_logger(), "Reached goal, currently at (%.2f, %.2f)", current_pose.x(), current_pose.y());
        publish_chassis_cmd(0, 0, false);
        teb_controller_->set_reference_path({});
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    const auto result = teb_controller_->solve(
        current_pose,
        current_state_,
        merged_cost_map_.get(),
        global_direction_map_.get()
    );
    if (!result.ok) {
        publish_chassis_cmd(0, 0, false);
        return;
    }

    RCLCPP_DEBUG(get_logger(), "TebController solve time: %.2f ms",
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count()
    );

    publish_chassis_cmd(result.cmd_v_omega.x(), result.cmd_v_omega.y(), false);
    if (enable_debug_) {
        debug_predicted_path_pub_->publish(path_to_nav_msg(result.predicted_path_map));
    }
}

void PathFollowerNode::publish_chassis_cmd(double velocity, double palstance, bool step_ahead) {
    interfaces::msg::ChassisCmd msg;
    msg.velocity = velocity;
    msg.palstance = palstance;
    msg.step_ahead = step_ahead;
    chassis_cmd_pub_->publish(msg);
}

bool PathFollowerNode::get_current_pose(Eigen::Vector3d& current_pose) const {
    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = tf_buffer_->lookupTransform("map", "chassis_link", tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(get_logger(), "Could not transform chassis_link to map: %s", ex.what());
        return false;
    }

    current_pose.head<2>() = utils::convert_to<Eigen::Vector3d>(tf.transform.translation).head<2>();
    const Eigen::Quaterniond q = utils::convert_to<Eigen::Quaterniond>(tf.transform.rotation);
    const Eigen::Vector2d x_axis = (q * Eigen::Vector3d::UnitX()).head<2>();
    if (x_axis.norm() < 1e-6) {
        RCLCPP_ERROR(get_logger(), "Invalid chassis_link orientation");
        return false;
    }
    current_pose.z() = atan2(x_axis.y(), x_axis.x());
    return true;
}

nav_msgs::msg::Path PathFollowerNode::path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const {
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
RCLCPP_COMPONENTS_REGISTER_NODE(path_follower::PathFollowerNode)