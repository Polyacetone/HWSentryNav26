#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <gtsam_points/optimizers/linearization_hook.hpp>

#include <small_glim/common/config.hpp>
#include <small_glim/common/logger.hpp>
#include <small_glim/preprocess/cloud_preprocessor.hpp>
#include <small_glim/preprocess/time_keeper.hpp>
#include <small_glim/odometry/async_odometry_estimation.hpp>
#include <small_glim/mapping/async_sub_mapping.hpp>
#include <small_glim/mapping/async_global_mapping.hpp>

namespace small_glim {

class SmallGlimNode: public rclcpp::Node {
public:
    explicit SmallGlimNode(const rclcpp::NodeOptions& options);
    ~SmallGlimNode() override;

    bool needs_wait();
    void timer_callback();

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
    size_t points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

    void wait(bool auto_quit = false);
    void save(const std::string& path);

private:
    std::unique_ptr<CloudPreprocessor> preprocessor;
    std::unique_ptr<TimeKeeper> time_keeper;
    std::shared_ptr<AsyncOdometryEstimation> odometry_estimation;
    std::unique_ptr<AsyncSubMapping> sub_mapping;
    std::unique_ptr<AsyncGlobalMapping> global_mapping;

    bool keep_raw_points;
    double imu_time_offset;
    double points_time_offset;
    double acc_scale;
    bool dump_on_unload;

    std::string intensity_field, ring_field;

    // ROS-related
    rclcpp::TimerBase::SharedPtr timer;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
};

}

namespace small_glim {

SmallGlimNode::SmallGlimNode(const rclcpp::NodeOptions& options): Node("small_glim", options) {
    auto config = std::make_shared<Config>(this);

    bool debug = config->param<bool>("ros.debug");
    if (debug) {
        logger::info("main", "enable debug printing");
        get_logger().set_level(rclcpp::Logger::Level::Debug);
    }

    dump_on_unload = config->param<bool>("ros.dump_on_unload");
    if (dump_on_unload) {
        logger::info("main", "dump_on_unload=true");
    }

    keep_raw_points = config->param<bool>("ros.keep_raw_points");
    imu_time_offset = config->param<double>("ros.imu_time_offset");
    points_time_offset = config->param<double>("ros.points_time_offset");
    acc_scale = config->param<double>("ros.acc_scale");

    intensity_field = config->param<std::string>("sensors.intensity_field");
    ring_field = config->param<std::string>("sensors.ring_field");

    // Preprocessing
    time_keeper = std::make_unique<TimeKeeper>(config);
    preprocessor = std::make_unique<CloudPreprocessor>(config);

    // Odometry estimation
    auto odom = std::make_shared<OdometryEstimationCPU>(config);
    odometry_estimation = std::make_shared<AsyncOdometryEstimation>(odom);

    // Sub mapping
    if (config->param<bool>("ros.enable_local_mapping")) {
        auto sub = std::make_shared<SubMapping>(config);
        sub_mapping = std::make_unique<AsyncSubMapping>(sub);
    }

    // Global mapping
    if (config->param<bool>("ros.enable_global_mapping")) {
        auto global = std::make_shared<GlobalMapping>(config);
        global_mapping = std::make_unique<AsyncGlobalMapping>(global);
    }

    // ROS-related
    const std::string imu_topic = config->param<std::string>("ros.imu_topic");
    const std::string points_topic = config->param<std::string>("ros.points_topic");

    // Subscribers
    imu_sub = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic,
        rclcpp::QoS(1),
        [this](const sensor_msgs::msg::Imu::SharedPtr msg) { imu_callback(msg); }
    );
    points_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
        points_topic,
        rclcpp::QoS(1),
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { points_callback(msg); }
    );

    // Start timer
    timer = create_wall_timer(std::chrono::milliseconds(1), [this]() { timer_callback(); });
}

SmallGlimNode::~SmallGlimNode() {
    logger::debug("main", "quit");
    if (dump_on_unload) {
        std::string dump_path = "/tmp/dump";
        wait(true);
        save(dump_path);
    }
}

void SmallGlimNode::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    const double imu_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9 + imu_time_offset;
    const Eigen::Vector3d linear_acc = acc_scale * Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    const Eigen::Vector3d angular_vel(
        msg->angular_velocity.x,
        msg->angular_velocity.y,
        msg->angular_velocity.z
    );

    if (!time_keeper->validate_imu_stamp(imu_stamp)) {
        logger::warn("main", "skip an invalid IMU data (stamp={})", imu_stamp);
        return;
    }

    odometry_estimation->insert_imu(imu_stamp, linear_acc, angular_vel);
    if (sub_mapping) {
        sub_mapping->insert_imu(imu_stamp, linear_acc, angular_vel);
    }
    if (global_mapping) {
        global_mapping->insert_imu(imu_stamp, linear_acc, angular_vel);
    }
}

size_t SmallGlimNode::points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    auto raw_points = std::make_shared<RawPoints>(*msg, intensity_field, ring_field);
    if (raw_points == nullptr) {
        logger::warn("main", "failed to extract points from message");
        return 0;
    }

    raw_points->stamp += points_time_offset;
    time_keeper->process(raw_points);
    auto preprocessed = preprocessor->preprocess(raw_points);

    if (keep_raw_points) {
        // note: Raw points are used only in extension modules for visualization purposes.
        //       If you need to reduce the memory footprint, you can safely comment out the following line.
        preprocessed->raw_points = raw_points;
    }

    odometry_estimation->insert_frame(preprocessed);

    const size_t workload = odometry_estimation->workload();
    logger::debug("main", "workload={}", workload);

    return workload;
}

void SmallGlimNode::timer_callback() {
    std::vector<EstimationFrame::ConstPtr> estimation_frames;
    std::vector<EstimationFrame::ConstPtr> marginalized_frames;
    odometry_estimation->get_results(estimation_frames, marginalized_frames);

    if (sub_mapping) {
        for (const auto& frame: marginalized_frames) {
            sub_mapping->insert_frame(frame);
        }

        auto submaps = sub_mapping->get_results();
        if (global_mapping) {
            for (const auto& submap: submaps) {
                global_mapping->insert_submap(submap);
            }
        }
    }
}

void SmallGlimNode::wait(bool auto_quit) {
    logger::info("main", "waiting for odometry estimation");
    odometry_estimation->join();

    if (sub_mapping) {
        std::vector<EstimationFrame::ConstPtr> estimation_results;
        std::vector<EstimationFrame::ConstPtr> marginalized_frames;
        odometry_estimation->get_results(estimation_results, marginalized_frames);
        for (const auto& marginalized_frame: marginalized_frames) {
            sub_mapping->insert_frame(marginalized_frame);
        }

        logger::info("main", "waiting for local mapping");
        sub_mapping->join();

        const auto submaps = sub_mapping->get_results();
        if (global_mapping) {
            for (const auto& submap: submaps) {
                global_mapping->insert_submap(submap);
            }
            logger::info("main", "waiting for global mapping");
            global_mapping->join();
        }
    }
}

void SmallGlimNode::save(const std::string& path) {
    if (global_mapping) {
        global_mapping->save(path);
    }
}

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(small_glim::SmallGlimNode)