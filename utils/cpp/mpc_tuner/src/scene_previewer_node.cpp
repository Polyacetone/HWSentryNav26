#include <mpc_tuner/console_input.hpp>
#include <mpc_tuner/scene_bundle.hpp>
#include <mpc_tuner/scene_visualization.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace mpc_tuner {
namespace {

struct SceneFile {
    std::filesystem::path path;
    SceneBundle bundle;
};

std::vector<SceneFile> load_scene_files(const std::filesystem::path& directory) {
    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error("Scene directory does not exist: " + directory.string());
    }

    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".msgpack") {
            paths.push_back(entry.path());
        }
    }
    std::ranges::sort(paths);
    if (paths.empty()) {
        throw std::runtime_error("No .msgpack scene bundles found in " + directory.string());
    }

    std::vector<SceneFile> files;
    files.reserve(paths.size());
    for (const auto& path : paths) {
        files.push_back({.path = path, .bundle = load_scene_bundle(path)});
    }
    return files;
}

} // namespace

class ScenePreviewerNode final : public rclcpp::Node {
public:
    ScenePreviewerNode(): Node("mpc_scene_previewer") {
        scene_directory_ = std::filesystem::absolute(declare_parameter<std::string>(
            "scene_directory",
            ament_index_cpp::get_package_share_directory("mpc_tuner") + "/scenes"
        ));
        files_ = load_scene_files(scene_directory_);

        const auto durable_qos = rclcpp::QoS(1).reliable().transient_local();
        cost_map_publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
            "~/global_cost_map", durable_qos
        );
        direction_map_publisher_ = create_publisher<sensor_msgs::msg::Image>(
            "~/direction_map", durable_qos
        );
        path_publisher_ = create_publisher<nav_msgs::msg::Path>("~/selected_path", durable_qos);
        marker_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "~/routes", durable_qos
        );
        command_timer_ = create_wall_timer(
            std::chrono::milliseconds(100), [this] { process_commands(); }
        );

        size_t route_count = 0;
        for (const auto& file : files_) route_count += file.bundle.routes.size();
        RCLCPP_INFO(
            get_logger(), "Loaded %zu bundles with %zu routes from %s",
            files_.size(), route_count, scene_directory_.c_str()
        );
        RCLCPP_INFO(
            get_logger(), "View in RViz: map=%s, direction=%s, path=%s, markers=%s",
            cost_map_publisher_->get_topic_name(), direction_map_publisher_->get_topic_name(),
            path_publisher_->get_topic_name(), marker_publisher_->get_topic_name()
        );
        if (!console_.available()) {
            RCLCPP_WARN(get_logger(), "No controlling terminal; preview commands are unavailable");
        }
        print_help();
        publish_selection();
    }

private:
    void select_route(const int delta) {
        const int route_count = static_cast<int>(current_file().bundle.routes.size());
        route_index_ = static_cast<size_t>(
            (static_cast<int>(route_index_) + delta + route_count) % route_count
        );
        publish_selection();
    }

    void select_file(const int delta) {
        const int file_count = static_cast<int>(files_.size());
        file_index_ = static_cast<size_t>(
            (static_cast<int>(file_index_) + delta + file_count) % file_count
        );
        route_index_ = 0;
        publish_selection();
    }

    void publish_selection() {
        const auto& file = current_file();
        const auto& route = file.bundle.routes.at(route_index_);
        const auto stamp = now();
        cost_map_publisher_->publish(make_cost_map_message(file.bundle.map, stamp));
        direction_map_publisher_->publish(make_direction_map_message(file.bundle.map, stamp));
        path_publisher_->publish(make_route_path(route, stamp));
        marker_publisher_->publish(make_route_markers(
            file.bundle.routes, route_index_, nullptr, stamp
        ));
        RCLCPP_INFO(
            get_logger(), "[%zu/%zu] %s | route [%zu/%zu] %s | split=%s",
            file_index_ + 1, files_.size(), file.path.filename().c_str(),
            route_index_ + 1, file.bundle.routes.size(), route.spec.name.c_str(),
            file.bundle.split.c_str()
        );
    }

    void process_commands() {
        for (const char command : console_.take_commands()) {
            switch (command) {
                case 'n': select_route(1); break;
                case 'p': select_route(-1); break;
                case 'f': select_file(1); break;
                case 'b': select_file(-1); break;
                case 'q':
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
            "Commands (key then Enter): n/p=next/previous route, f/b=next/previous bundle, "
            "q=exit, h=help"
        );
    }

    [[nodiscard]] const SceneFile& current_file() const { return files_.at(file_index_); }

    std::filesystem::path scene_directory_;
    std::vector<SceneFile> files_;
    size_t file_index_ = 0;
    size_t route_index_ = 0;

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
        rclcpp::spin(std::make_shared<mpc_tuner::ScenePreviewerNode>());
        if (rclcpp::ok()) rclcpp::shutdown();
        return 0;
    } catch (const std::exception& error) {
        RCLCPP_FATAL(rclcpp::get_logger("mpc_scene_previewer"), "%s", error.what());
        if (rclcpp::ok()) rclcpp::shutdown();
        return 1;
    }
}
