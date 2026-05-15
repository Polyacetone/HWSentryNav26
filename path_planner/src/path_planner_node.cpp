#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <nav_msgs/msg/odometry.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/empty.hpp>
#include <interfaces/msg/nav_goal.hpp>
#include <interfaces/msg/global_path.hpp>
#include <interfaces/msg/chassis_status.hpp>
#include <interfaces/msg/cost_maps.hpp>

#include <path_planner/nav_map.hpp>
#include <path_planner/a_star_planner.hpp>
#include <path_planner/bspline_optimizer.hpp>
#include <common_utils/convert.hpp>

namespace path_planner {

namespace {

std::vector<Eigen::Vector2d> make_direct_init_path(
    const Eigen::Vector2d& start_grid,
    const Eigen::Vector2d& goal_grid,
    int width,
    int height
) {
    const Eigen::Vector2d delta = goal_grid - start_grid;
    const double dist = delta.norm();

    if (dist <= 1e-6) {
        Eigen::Vector2d axis = Eigen::Vector2d::UnitX();
        if (width <= 1 && height > 1) axis = Eigen::Vector2d::UnitY();

        Eigen::Vector2d mid = start_grid + 0.5 * axis;
        mid.x() = std::clamp(mid.x(), 0.0, static_cast<double>(std::max(0, width - 1)));
        mid.y() = std::clamp(mid.y(), 0.0, static_cast<double>(std::max(0, height - 1)));
        return {start_grid, mid, goal_grid};
    }

    const int point_count = std::max(3, static_cast<int>(std::ceil(dist)) + 1);
    std::vector<Eigen::Vector2d> path;
    path.reserve(static_cast<size_t>(point_count));
    for (int i = 0; i < point_count; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(point_count - 1);
        path.push_back((1.0 - t) * start_grid + t * goal_grid);
    }
    return path;
}

} // namespace

class PathPlannerNode: public rclcpp::Node {
public:
    explicit PathPlannerNode(const rclcpp::NodeOptions& options);

private:
    enum class ReplanReason { GOAL_UPDATE, EXTERNAL_TRIGGER };
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_sub_;
    rclcpp::Subscription<interfaces::msg::CostMaps>::SharedPtr local_cost_maps_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr global_direction_map_sub_;
    rclcpp::Subscription<interfaces::msg::NavGoal>::SharedPtr goal_sub_;
    rclcpp::Subscription<interfaces::msg::ChassisStatus>::SharedPtr chassis_status_sub_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr replan_trigger_sub_;
    rclcpp::Publisher<interfaces::msg::GlobalPath>::SharedPtr control_points_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr optimized_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_rough_path_pub_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    CostMap::ConstPtr global_cost_map_;
    CostMap::ConstPtr local_cost_map_;
    CostMap::ConstPtr merged_cost_map_;
    DirectionMap::ConstPtr global_direction_map_;
    interfaces::msg::ChassisStatus::SharedPtr chassis_status_;
    AStarPlanner::ConstPtr path_planner_;
    BSplineOptimizer::ConstPtr path_optimizer_;
    double skip_distance_;
    bool enable_debug_;
    int occupied_threshold_;
    double on_step_threshold_;
    double step_mode_dot_threshold_;
    bool start_prediction_enable_;
    double start_prediction_max_accel_;
    double start_prediction_planning_delay_;
    double start_prediction_min_speed_;
    double start_prediction_collision_check_step_;

    std::optional<Eigen::Vector2d> last_goal_map_;
    bool last_goal_fixed_ = false;
    nav_msgs::msg::Path path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const;
    void goal_callback(const interfaces::msg::NavGoal::SharedPtr msg);
    void local_cost_maps_callback(const interfaces::msg::CostMaps::SharedPtr msg);

    void update_merged_cost_map();
    bool is_map_point_feasible(const CostMap& cost_map, const DirectionMap& direction_map, const Eigen::Vector2d& map_pt) const;
    void plan_and_publish_to_goal(const Eigen::Vector2d& goal_map, bool fixed, const ReplanReason reason);
    Eigen::Vector2d adjust_reachable_start_on_segment(const Eigen::Vector2d& from_map, const Eigen::Vector2d& to_map) const;
    Eigen::Vector2d predict_start_map(const Eigen::Vector2d& current_map) const;
    void publish_path(
        const std::vector<Eigen::Vector2d>& control_points,
        const std::vector<Eigen::Vector2d>& rough_path,
        const std::vector<Eigen::Vector2d>& optimized_path,
        bool fixed = false
    );
};

PathPlannerNode::PathPlannerNode(const rclcpp::NodeOptions& options): Node("path_planner", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    enable_debug_ = declare_parameter<bool>("debug.enable");
    occupied_threshold_ = (int)declare_parameter<int>("occupied_threshold");
    on_step_threshold_ = declare_parameter<double>("on_step_threshold");
    step_mode_dot_threshold_ = declare_parameter<double>("step_mode_dot_threshold");
    start_prediction_enable_ = declare_parameter<bool>("start_prediction.enable");
    start_prediction_max_accel_ = declare_parameter<double>("start_prediction.max_accel");
    start_prediction_planning_delay_ = declare_parameter<double>("start_prediction.planning_delay");
    start_prediction_min_speed_ = declare_parameter<double>("start_prediction.min_speed");
    start_prediction_collision_check_step_ = declare_parameter<double>("start_prediction.collision_check_step");

    if (enable_debug_) {
        std::string rough_path_pub_topic = declare_parameter<std::string>("debug.rough_path_pub_topic");
        debug_rough_path_pub_ = create_publisher<nav_msgs::msg::Path>(rough_path_pub_topic, 1);
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        RCLCPP_DEBUG(get_logger(), "Debug mode enabled");
    }

    skip_distance_ = declare_parameter<double>("path_planner.skip_distance");
    path_planner_ = std::make_shared<AStarPlanner>(
        declare_parameter<double>("path_planner.direction_weight"),
        declare_parameter<double>("path_planner.obstacle_weight"),
        declare_parameter<double>("path_planner.step_weight"),
        step_mode_dot_threshold_,
        declare_parameter<int>("path_planner.downsampled_waypoint_max_interval"),
        declare_parameter<int>("path_planner.feasible_threshold")
    );
    const BSplineOptimizer::Params path_optimizer_params{
        .step_norm_threshold = declare_parameter<double>("path_optimizer.step_norm_threshold"),
        .step_norm_transition = declare_parameter<double>("path_optimizer.step_norm_transition"),
        .step_mode_dot_threshold = step_mode_dot_threshold_,
        .step_detection_samples_per_meter = declare_parameter<double>("path_optimizer.step_detection_samples_per_meter"),
        .warmup = {
            .obstacle_weight = declare_parameter<double>("path_optimizer.warmup.obstacle_weight"),
            .direction_weight = declare_parameter<double>("path_optimizer.warmup.direction_weight"),
            .step_weight = declare_parameter<double>("path_optimizer.warmup.step_weight"),
            .prohibited_direction_weight = declare_parameter<double>("path_optimizer.warmup.prohibited_direction_weight"),
            .start_end_weight = declare_parameter<double>("path_optimizer.warmup.start_end_weight"),
            .smoothness_weight = declare_parameter<double>("path_optimizer.warmup.smoothness_weight"),
            .samples_per_meter = declare_parameter<double>("path_optimizer.warmup.samples_per_meter"),
            .max_iterations = static_cast<int>(declare_parameter<int>("path_optimizer.warmup.max_iterations")),
            .max_curvature = declare_parameter<double>("path_optimizer.warmup.max_curvature"),
            .length_penalty = {
                .weight = declare_parameter<double>("path_optimizer.warmup.length_penalty.weight")
            },
            .curvature = {
                .base_weight = declare_parameter<double>("path_optimizer.warmup.curvature.base_weight"),
                .base_beta = declare_parameter<double>("path_optimizer.warmup.curvature.base_beta"),
                .limit_weight = declare_parameter<double>("path_optimizer.warmup.curvature.limit_weight"),
                .limit_beta = declare_parameter<double>("path_optimizer.warmup.curvature.limit_beta"),
                .min_speed_epsilon = declare_parameter<double>("path_optimizer.warmup.curvature.min_speed_epsilon"),
                .speed_gate_threshold = declare_parameter<double>("path_optimizer.warmup.curvature.speed_gate_threshold")
            }
        },
        .main = {
            .obstacle_weight = declare_parameter<double>("path_optimizer.main.obstacle_weight"),
            .direction_weight = declare_parameter<double>("path_optimizer.main.direction_weight"),
            .step_weight = declare_parameter<double>("path_optimizer.main.step_weight"),
            .prohibited_direction_weight = declare_parameter<double>("path_optimizer.main.prohibited_direction_weight"),
            .start_end_weight = declare_parameter<double>("path_optimizer.main.start_end_weight"),
            .smoothness_weight = declare_parameter<double>("path_optimizer.main.smoothness_weight"),
            .samples_per_meter = declare_parameter<double>("path_optimizer.main.samples_per_meter"),
            .max_iterations = static_cast<int>(declare_parameter<int>("path_optimizer.main.max_iterations")),
            .max_refinement_iterations = static_cast<int>(declare_parameter<int>("path_optimizer.main.max_refinement_iterations")),
            .near_max_curvature = declare_parameter<double>("path_optimizer.main.near_max_curvature"),
            .far_max_curvature = declare_parameter<double>("path_optimizer.main.far_max_curvature"),
            .step_extension_distance = declare_parameter<double>("path_optimizer.main.step_extension_distance"),
            .step_transition_distance = declare_parameter<double>("path_optimizer.main.step_transition_distance"),
            .interval_iou_threshold = declare_parameter<double>("path_optimizer.main.interval_iou_threshold"),
            .length_penalty = {
                .weight = declare_parameter<double>("path_optimizer.main.length_penalty.weight")
            },
            .curvature = {
                .base_weight = declare_parameter<double>("path_optimizer.main.curvature.base_weight"),
                .base_beta = declare_parameter<double>("path_optimizer.main.curvature.base_beta"),
                .limit_weight = declare_parameter<double>("path_optimizer.main.curvature.limit_weight"),
                .limit_beta = declare_parameter<double>("path_optimizer.main.curvature.limit_beta"),
                .min_speed_epsilon = declare_parameter<double>("path_optimizer.main.curvature.min_speed_epsilon"),
                .speed_gate_threshold = declare_parameter<double>("path_optimizer.main.curvature.speed_gate_threshold")
            }
        }
    };
    path_optimizer_ = std::make_shared<BSplineOptimizer>(path_optimizer_params);

    const std::string global_cost_map_sub_topic = declare_parameter<std::string>("global_cost_map_sub_topic");
    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        global_cost_map_sub_topic, 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            global_cost_map_ = std::make_shared<CostMap>(*msg);
            RCLCPP_INFO(
                get_logger(), "Received global cost map: size=(%d, %d), resolution=%.2f",
                global_cost_map_->width, global_cost_map_->height, global_cost_map_->resolution
            );
            update_merged_cost_map();
            global_cost_map_sub_.reset();
        }
    );

    local_cost_maps_sub_ = create_subscription<interfaces::msg::CostMaps>(
        declare_parameter<std::string>("local_cost_maps_sub_topic"), 1,
        [this](const interfaces::msg::CostMaps::SharedPtr msg) { local_cost_maps_callback(msg); }
    );

    const std::string global_direction_map_sub_topic = declare_parameter<std::string>("global_direction_map_sub_topic");
    global_direction_map_sub_ = create_subscription<sensor_msgs::msg::Image>(
        global_direction_map_sub_topic, 1,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            if (!global_cost_map_) {
                RCLCPP_WARN(get_logger(), "Received global direction map before cost map, waiting for cost map first");
                return;
            }

            const auto cv_ptr = cv_bridge::toCvShare(msg, "8UC3");
            global_direction_map_ = std::make_shared<DirectionMap>(
                cv_ptr->image,
                global_cost_map_->resolution,
                global_cost_map_->origin_x,
                global_cost_map_->origin_y
            );

            if (global_direction_map_->width != global_cost_map_->width || global_direction_map_->height != global_cost_map_->height) {
                RCLCPP_FATAL(
                    get_logger(),
                    "Direction map size (%d, %d) does not match cost map size (%d, %d)!",
                    global_direction_map_->width, global_direction_map_->height,
                    global_cost_map_->width, global_cost_map_->height
                );
                throw std::runtime_error("Direction map size does not match cost map size");
            }

            RCLCPP_INFO(get_logger(), "Received global direction map");
            global_direction_map_sub_.reset();
        }
    );

    chassis_status_sub_ = create_subscription<interfaces::msg::ChassisStatus>(
        declare_parameter<std::string>("chassis_status_sub_topic"), 1,
        [this](const interfaces::msg::ChassisStatus::SharedPtr msg) { chassis_status_ = msg; }
    );

    replan_trigger_sub_ = create_subscription<std_msgs::msg::Empty>(
        declare_parameter<std::string>("replan_trigger_sub_topic"), 1,
        [this](const std_msgs::msg::Empty::SharedPtr) {
            if (!last_goal_map_) {
                RCLCPP_WARN(get_logger(), "Received external replan trigger, but no goal is available yet");
                return;
            }
            RCLCPP_INFO(get_logger(), "Received external replan trigger, replanning from current pose");
            plan_and_publish_to_goal(*last_goal_map_, last_goal_fixed_, ReplanReason::EXTERNAL_TRIGGER);
        }
    );

    goal_sub_ = create_subscription<interfaces::msg::NavGoal>(
        declare_parameter<std::string>("goal_sub_topic"), 1,
        [this](const interfaces::msg::NavGoal::SharedPtr msg) { goal_callback(msg); }
    );

    control_points_pub_ = create_publisher<interfaces::msg::GlobalPath>(
        declare_parameter<std::string>("control_points_pub_topic"), 1
    );

    optimized_path_pub_ = create_publisher<nav_msgs::msg::Path>(
        declare_parameter<std::string>("optimized_path_pub_topic"), 1
    );

}

bool PathPlannerNode::is_map_point_feasible(const CostMap& cost_map, const DirectionMap& direction_map, const Eigen::Vector2d& map_pt) const {
    const Eigen::Vector2d cost_grid = cost_map.map_coord_to_grid(map_pt);
    const Eigen::Vector2d dir_grid = direction_map.map_coord_to_grid(map_pt);
    if (!cost_map.is_valid_coord(cost_grid) || !direction_map.is_valid_coord(dir_grid)) return false;

    return cost_map.interpolate(cost_grid) < occupied_threshold_ &&
        direction_map.interpolate(dir_grid).norm() < on_step_threshold_ &&
        !direction_map.is_fully_prohibited(dir_grid);
}

void PathPlannerNode::local_cost_maps_callback(const interfaces::msg::CostMaps::SharedPtr msg) {
    if (!global_cost_map_ || msg->maps.empty()) return;
    local_cost_map_ = std::make_shared<CostMap>(msg->maps[0]);
    update_merged_cost_map();
}

void PathPlannerNode::update_merged_cost_map() {
    if (!global_cost_map_) return;

    try {
        CostMap merged = [&]() {
            if (local_cost_map_) return global_cost_map_->merge(*local_cost_map_);
            return CostMap(*global_cost_map_);
        }();
        const auto ptr = std::make_shared<CostMap>(std::move(merged));
        merged_cost_map_ = ptr;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to merge cost maps: %s", e.what());
        merged_cost_map_ = global_cost_map_;
    }
}

Eigen::Vector2d PathPlannerNode::adjust_reachable_start_on_segment(const Eigen::Vector2d& from_map, const Eigen::Vector2d& to_map) const {
    const Eigen::Vector2d delta = to_map - from_map;
    const double length = delta.norm();
    if (length <= 1e-6) return from_map;

    const double step = std::max(1e-3, start_prediction_collision_check_step_);
    const int n = std::max(1, static_cast<int>(std::ceil(length / step)));
    const Eigen::Vector2d dir = delta / length;

    Eigen::Vector2d last_feasible = from_map;
    for (int i = 0; i <= n; ++i) {
        const double d = length * (static_cast<double>(i) / static_cast<double>(n));
        const Eigen::Vector2d pt = from_map + dir * d;
        if (!is_map_point_feasible(*merged_cost_map_, *global_direction_map_, pt)) break;
        last_feasible = pt;
    }
    return last_feasible;
}

Eigen::Vector2d PathPlannerNode::predict_start_map(const Eigen::Vector2d& current_map) const {
    if (!start_prediction_enable_) return current_map;
    if (!chassis_status_) return current_map;

    const double speed = chassis_status_->velocity;
    if (speed < std::max(0.0, start_prediction_min_speed_)) return current_map;

    Eigen::Vector2d v_map;
    try {
        const auto chassis_to_map = tf_buffer_->lookupTransform("map", "chassis_link", tf2::TimePointZero);
        const tf2::Matrix3x3 R(utils::convert_to<tf2::Transform>(chassis_to_map.transform).getRotation());
        const tf2::Vector3 v = R * tf2::Vector3(speed, 0.0, 0.0);
        v_map = Eigen::Vector2d(v.x(), v.y());
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(), "Failed to transform chassis_link to map: %s", ex.what());
        return current_map;
    }

    const double brake_distance = speed * speed / (2.0 * start_prediction_max_accel_);
    const double delay_distance = speed * std::max(0.0, start_prediction_planning_delay_);
    const double total_distance = brake_distance + delay_distance;

    const Eigen::Vector2d dir = v_map / speed;
    const Eigen::Vector2d predicted = current_map + dir * total_distance;
    return adjust_reachable_start_on_segment(current_map, predicted);
}

void PathPlannerNode::goal_callback(const interfaces::msg::NavGoal::SharedPtr msg) {
    last_goal_map_ = Eigen::Vector2d(msg->x, msg->y);
    last_goal_fixed_ = msg->fixed;
    plan_and_publish_to_goal(Eigen::Vector2d(msg->x, msg->y), msg->fixed, ReplanReason::GOAL_UPDATE);
}

void PathPlannerNode::plan_and_publish_to_goal(const Eigen::Vector2d& goal_map, bool fixed, const ReplanReason reason) {
    const char* reason_label = reason == ReplanReason::GOAL_UPDATE
        ? "goal update"
        : "external trigger";
    const bool use_merged_cost_map = reason == ReplanReason::EXTERNAL_TRIGGER;
    const auto publish_empty_path = [&](const std::string& message, const bool is_error) {
        if (is_error) {
            RCLCPP_ERROR(get_logger(), "%s (%s)", message.c_str(), reason_label);
        } else {
            RCLCPP_WARN(get_logger(), "%s (%s)", message.c_str(), reason_label);
        }
        publish_path({}, {}, {}, false);
    };

    if (!merged_cost_map_ || !global_direction_map_) {
        publish_empty_path("Map not ready yet!", false);
        return;
    }

    const CostMap& planning_cost_map = use_merged_cost_map ? *merged_cost_map_ : *global_cost_map_;

    tf2::Transform chassis_to_map;
    try {
        chassis_to_map = utils::convert_to<tf2::Transform>(
            tf_buffer_->lookupTransform("map", "chassis_link", tf2::TimePointZero).transform
        );
    } catch (const std::exception& ex) {
        publish_empty_path(std::string("Failed to lookup chassis_link to map: ") + ex.what(), false);
        return;
    }

    const Eigen::Vector2d current_map(chassis_to_map.getOrigin().x(), chassis_to_map.getOrigin().y());
    const Eigen::Vector2d start_map = predict_start_map(current_map);

    const Eigen::Vector2d start_grid = global_cost_map_->map_coord_to_grid(start_map);
    const Eigen::Vector2d goal_grid = global_cost_map_->map_coord_to_grid(goal_map);

    if (!global_cost_map_->is_valid_coord(start_grid)) {
        publish_empty_path(
            "Start (" + std::to_string(start_map.x()) + ", " + std::to_string(start_map.y()) + ") is out of bound!",
            true
        );
        return;
    }

    if (!is_map_point_feasible(*global_cost_map_, *global_direction_map_, start_map)) {
        publish_empty_path(
            "Start (" + std::to_string(start_map.x()) + ", " + std::to_string(start_map.y()) + ") is not feasible!",
            true
        );
        return;
    }

    if (!global_cost_map_->is_valid_coord(goal_grid)) {
        publish_empty_path(
            "Goal (" + std::to_string(goal_map.x()) + ", " + std::to_string(goal_map.y()) + ") is out of bound!",
            true
        );
        return;
    }

    if (!is_map_point_feasible(*global_cost_map_, *global_direction_map_, goal_map)) {
        publish_empty_path(
            "Goal (" + std::to_string(goal_map.x()) + ", " + std::to_string(goal_map.y()) + ") is not feasible!",
            true
        );
        return;
    }

    const double dist = (goal_map - start_map).norm();

    RCLCPP_INFO(
        get_logger(), "Planning (%s): Src(%.2f, %.2f) -> Dst(%.2f, %.2f) %s",
        reason_label,
        start_map.x(), start_map.y(), goal_map.x(), goal_map.y(),
        fixed ? "[FIXED]" : ""
    );

    const auto start_time = std::chrono::steady_clock::now();
    std::vector<Eigen::Vector2d> rough_path;

    const auto optimize_and_publish = [&](const std::vector<Eigen::Vector2d>& init_path) {
        std::vector<Eigen::Vector2d> control_points, optimized_path;
        const auto optimize_result = path_optimizer_->optimize(
            planning_cost_map,
            *global_direction_map_,
            init_path,
            start_grid,
            goal_grid
        );
        if (!optimize_result) {
            RCLCPP_ERROR(get_logger(), "Path optimization failed: %s", optimize_result.error().c_str());
            publish_path({}, {}, {}, false);
            return;
        }

        std::tie(control_points, optimized_path) = *optimize_result;
        RCLCPP_DEBUG(
            get_logger(), "Path optimization took %.2f ms, control points: %d, optimized path length: %d",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count(),
            static_cast<int>(control_points.size()),
            static_cast<int>(optimized_path.size())
        );
        publish_path(control_points, init_path, optimized_path, fixed);
    };

    if (dist < skip_distance_) {
        if (fixed) {
            rough_path = make_direct_init_path(start_grid, goal_grid, merged_cost_map_->width, merged_cost_map_->height);
            RCLCPP_INFO(get_logger(), "Goal is within lazy distance (%.2f m) but in fixed mode, skipping A* and using direct spline init with %zu points", skip_distance_, rough_path.size());
            optimize_and_publish(rough_path);
        } else {
            RCLCPP_INFO(get_logger(), "Goal is within lazy distance (%.2f m)", skip_distance_);
            publish_path({}, {}, {}, fixed);
        }
        return;
    }

    const auto plan_result = path_planner_->search_path(
        planning_cost_map,
        *global_direction_map_,
        start_grid.cast<int>(),
        goal_grid.cast<int>()
    );
    if (!plan_result) {
        RCLCPP_ERROR(get_logger(), "Path planning failed: %s", plan_result.error().c_str());
        publish_path({}, {}, {}, false);
        return;
    }

    rough_path.reserve(plan_result->size());
    for (const auto& pt : *plan_result) {
        rough_path.push_back(pt.cast<double>());
    }

    RCLCPP_DEBUG(
        get_logger(), "A* planning took %.2f ms, path length: %d",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count(),
        static_cast<int>(rough_path.size())
    );

    optimize_and_publish(rough_path);
}

void PathPlannerNode::publish_path(
    const std::vector<Eigen::Vector2d>& control_points,
    const std::vector<Eigen::Vector2d>& rough_path,
    const std::vector<Eigen::Vector2d>& optimized_path,
    bool fixed
) {
    // 发布路径转换到地图坐标系
    const auto to_map_coord = [&](const std::vector<Eigen::Vector2d>& points) {
        std::vector<Eigen::Vector2d> map_points;
        for (const auto& pt: points) {
            map_points.push_back(merged_cost_map_->grid_coord_to_map(pt));
        }
        return map_points;
    };
    const auto map_cps = to_map_coord(control_points);
    interfaces::msg::GlobalPath gp_msg;
    gp_msg.x.reserve(map_cps.size());
    gp_msg.y.reserve(map_cps.size());
    for (const auto& pt : map_cps) {
        gp_msg.x.push_back(static_cast<float>(pt.x()));
        gp_msg.y.push_back(static_cast<float>(pt.y()));
    }
    gp_msg.fixed = fixed;
    control_points_pub_->publish(gp_msg);
    const auto optimized_path_map = to_map_coord(optimized_path);
    optimized_path_pub_->publish(path_to_nav_msg(optimized_path_map));
    if (enable_debug_) {
        debug_rough_path_pub_->publish(path_to_nav_msg(to_map_coord(rough_path)));
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
