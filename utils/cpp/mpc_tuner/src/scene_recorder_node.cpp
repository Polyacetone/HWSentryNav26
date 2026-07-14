#include <mpc_tuner/config_loader.hpp>
#include <mpc_tuner/console_input.hpp>
#include <mpc_tuner/scenario_compiler.hpp>
#include <mpc_tuner/scene_bundle.hpp>
#include <mpc_tuner/scene_map.hpp>
#include <mpc_tuner/scene_visualization.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <numbers>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace mpc_tuner {
namespace {

std::string timestamp_string() {
    const std::time_t time = std::time(nullptr);
    std::tm local {};
    localtime_r(&time, &local);
    std::ostringstream stream;
    stream << std::put_time(&local, "%Y%m%d_%H%M%S");
    return stream.str();
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion& quaternion) {
    return std::atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    );
}

bool is_map_frame(const std::string& frame) {
    return frame.empty() || frame == "map" || frame == "/map";
}

} // namespace

class SceneRecorderNode final : public rclcpp::Node {
public:
    SceneRecorderNode(): Node("mpc_scene_recorder") {
        split_ = declare_parameter<std::string>("split", "train");
        if (split_ != "train" && split_ != "validation") {
            throw std::runtime_error("split must be 'train' or 'validation'");
        }
        output_directory_ = std::filesystem::absolute(
            declare_parameter<std::string>("output_directory", ".")
        );
        const auto seed_values = declare_parameter<std::vector<int64_t>>(
            "seeds", std::vector<int64_t> {1}
        );
        if (seed_values.empty()) throw std::runtime_error("seeds must not be empty");
        for (const int64_t seed : seed_values) seeds_.push_back(static_cast<uint64_t>(seed));

        const std::filesystem::path nav_config_directory = declare_parameter<std::string>(
            "nav_config_directory",
            ament_index_cpp::get_package_share_directory("nav_executor") + "/config"
        );
        const std::filesystem::path map_directory = declare_parameter<std::string>(
            "map_directory",
            ament_index_cpp::get_package_share_directory("map_server") + "/maps"
        );
        const std::string map_filename = declare_parameter<std::string>("map_filename");
        if (map_filename.empty()) throw std::runtime_error("map_filename must not be empty");
        const map_server::map_utils::MapInflationParams inflation {
            .robot_radius_m = declare_parameter<double>("map_inflation.robot_radius_m"),
            .cutoff_radius_m = declare_parameter<double>("map_inflation.cutoff_radius_m"),
            .decay_alpha = declare_parameter<double>("map_inflation.decay_alpha"),
        };

        runtime_config_ = load_runtime_config(nav_config_directory);
        auto loaded_map = load_scene_map(map_directory / map_filename, inflation);
        map_snapshot_ = std::move(loaded_map.snapshot);
        planner_ = std::make_unique<ScenarioPlanner>(
            runtime_config_, std::move(loaded_map.cost_map), std::move(loaded_map.direction_map),
            get_logger().get_child("planner")
        );

        const auto start_topic = declare_parameter<std::string>("start_topic", "/initialpose");
        const auto goal_topic = declare_parameter<std::string>("goal_topic", "/goal_pose");
        start_subscription_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            start_topic, 1,
            [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr message) {
                receive_start(*message);
            }
        );
        goal_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            goal_topic, 1,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr message) {
                receive_goal(*message);
            }
        );

        const auto durable_qos = rclcpp::QoS(1).reliable().transient_local();
        cost_map_publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
            "~/global_cost_map", durable_qos
        );
        direction_map_publisher_ = create_publisher<sensor_msgs::msg::Image>(
            "~/direction_map", durable_qos
        );
        path_publisher_ = create_publisher<nav_msgs::msg::Path>("~/candidate_path", durable_qos);
        marker_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "~/routes", durable_qos
        );
        command_timer_ = create_wall_timer(
            std::chrono::milliseconds(100), [this] { process_commands(); }
        );
        const auto stamp = now();
        cost_map_publisher_->publish(make_cost_map_message(map_snapshot_, stamp));
        direction_map_publisher_->publish(make_direction_map_message(map_snapshot_, stamp));

        RCLCPP_INFO(
            get_logger(), "Loaded %s: %dx%d at %.3f m/cell",
            map_filename.c_str(), map_snapshot_.width, map_snapshot_.height,
            map_snapshot_.resolution
        );
        RCLCPP_INFO(
            get_logger(), "Recording '%s' scenes to %s",
            split_.c_str(), output_directory_.c_str()
        );
        RCLCPP_INFO(
            get_logger(),
            "Publish start poses on %s and goals on %s. RViz topics: map=%s, direction=%s, path=%s, routes=%s",
            start_topic.c_str(), goal_topic.c_str(), cost_map_publisher_->get_topic_name(),
            direction_map_publisher_->get_topic_name(), path_publisher_->get_topic_name(),
            marker_publisher_->get_topic_name()
        );
        if (!console_.available()) {
            RCLCPP_WARN(get_logger(), "No controlling terminal; recorder commands are unavailable");
        }
        print_help();
    }

private:
    void receive_start(const geometry_msgs::msg::PoseWithCovarianceStamped& message) {
        if (!is_map_frame(message.header.frame_id)) {
            RCLCPP_WARN(get_logger(), "Start pose must be in the map frame");
            return;
        }
        draft_start_ = Eigen::Vector3d(
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw_from_quaternion(message.pose.pose.orientation)
        );
        candidate_.reset();
        publish_visualization();
        RCLCPP_INFO(
            get_logger(), "Start set to (%.2f, %.2f, %.1f deg); publish a goal pose",
            draft_start_->x(), draft_start_->y(),
            draft_start_->z() * 180.0 / std::numbers::pi
        );
    }

    void receive_goal(const geometry_msgs::msg::PoseStamped& message) {
        if (!draft_start_) {
            RCLCPP_WARN(get_logger(), "Publish a start pose before publishing the goal");
            return;
        }
        if (!is_map_frame(message.header.frame_id)) {
            RCLCPP_WARN(get_logger(), "Goal must be in the map frame");
            return;
        }

        ScenarioSpec spec;
        char name[32];
        std::snprintf(name, sizeof(name), "route_%03zu", routes_.size() + 1);
        spec.name = name;
        spec.split = split_;
        spec.start_pose = *draft_start_;
        spec.goal = {message.pose.position.x, message.pose.position.y};
        spec.seeds = seeds_;

        RCLCPP_INFO(get_logger(), "Planning candidate %s...", spec.name.c_str());
        try {
            const auto compiled = planner_->plan(spec);
            candidate_ = StoredRoute {
                .spec = spec,
                .spline_control_points = compiled.path->spline.control_points(),
                .step_segments = compiled.path->step_segments,
                .control_cost_data = compiled.control_cost_map->data,
            };
            publish_visualization();
            RCLCPP_INFO(get_logger(), "Candidate ready; use 'a' to accept or 'r' to reject");
        } catch (const std::exception& error) {
            candidate_.reset();
            publish_visualization();
            RCLCPP_ERROR(get_logger(), "%s", error.what());
        }
    }

    void accept_candidate() {
        if (!candidate_) {
            RCLCPP_WARN(get_logger(), "There is no candidate route to accept");
            return;
        }
        routes_.push_back(std::move(*candidate_));
        candidate_.reset();
        draft_start_.reset();
        publish_visualization();
        RCLCPP_INFO(
            get_logger(), "Accepted %s (%zu routes); publish the next start pose",
            routes_.back().spec.name.c_str(), routes_.size()
        );
    }

    void reject_candidate() {
        if (!candidate_ && !draft_start_) {
            RCLCPP_WARN(get_logger(), "There is no draft route to reject");
            return;
        }
        candidate_.reset();
        draft_start_.reset();
        publish_visualization();
        RCLCPP_INFO(get_logger(), "Draft discarded; publish a new start pose");
    }

    void save_and_exit() {
        if (routes_.empty()) {
            RCLCPP_WARN(get_logger(), "Refusing to save an empty bundle");
            return;
        }
        const std::string stamp = timestamp_string();
        std::filesystem::create_directories(output_directory_);
        const auto path = output_directory_ / (split_ + "_" + stamp + ".msgpack");
        try {
            write_scene_bundle({
                .split = split_,
                .created_at = stamp,
                .map = map_snapshot_,
                .routes = routes_,
            }, path);
            RCLCPP_INFO(get_logger(), "Saved %zu routes to %s", routes_.size(), path.c_str());
            rclcpp::shutdown();
        } catch (const std::exception& error) {
            RCLCPP_ERROR(get_logger(), "%s", error.what());
        }
    }

    void publish_visualization() {
        marker_publisher_->publish(make_route_markers(
            routes_, std::nullopt, candidate_ ? &*candidate_ : nullptr, now()
        ));
        if (candidate_) {
            path_publisher_->publish(make_route_path(*candidate_, now()));
        } else {
            nav_msgs::msg::Path empty;
            empty.header.frame_id = "map";
            empty.header.stamp = now();
            path_publisher_->publish(empty);
        }
    }

    void process_commands() {
        for (const char command : console_.take_commands()) {
            switch (command) {
                case 'a': accept_candidate(); break;
                case 'r': reject_candidate(); break;
                case 's': save_and_exit(); return;
                case 'q':
                    RCLCPP_INFO(get_logger(), "Exiting without saving");
                    rclcpp::shutdown();
                    return;
                case 'h': print_help(); break;
                default:
                    RCLCPP_WARN(get_logger(), "Unknown command '%c'; use 'h' for help", command);
                    break;
            }
        }
    }

    void print_help() const {
        RCLCPP_INFO(
            get_logger(),
            "Commands (key then Enter): a=accept candidate, r=reject draft, "
            "s=save and exit, q=discard and exit, h=help"
        );
    }

    std::string split_;
    std::filesystem::path output_directory_;
    std::vector<uint64_t> seeds_;
    RuntimeConfig runtime_config_;
    MapSnapshot map_snapshot_;
    std::unique_ptr<ScenarioPlanner> planner_;

    std::optional<Eigen::Vector3d> draft_start_;
    std::optional<StoredRoute> candidate_;
    std::vector<StoredRoute> routes_;

    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr start_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_subscription_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr cost_map_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr direction_map_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    rclcpp::TimerBase::SharedPtr command_timer_;
    ConsoleInput console_;
};

} // namespace mpc_tuner

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<mpc_tuner::SceneRecorderNode>());
        if (rclcpp::ok()) rclcpp::shutdown();
        return 0;
    } catch (const std::exception& error) {
        RCLCPP_FATAL(rclcpp::get_logger("mpc_scene_recorder"), "%s", error.what());
        if (rclcpp::ok()) rclcpp::shutdown();
        return 1;
    }
}
