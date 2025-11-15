#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <gtsam_points/optimizers/linearization_hook.hpp>
#include <common_utils/convert.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>

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

    void timer_callback();
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
    size_t points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

    void send_odometry(const EstimationFrame::ConstPtr frame);
    void wait(bool auto_quit = false);
    void save(const std::string& path);

private:
    std::unique_ptr<tf2_ros::Buffer> tf_buffer;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

    std::unique_ptr<CloudPreprocessor> preprocessor;
    std::unique_ptr<TimeKeeper> time_keeper;
    std::unique_ptr<AsyncOdometryEstimation> odometry_estimation;
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
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub;
};

}

namespace small_glim {

SmallGlimNode::SmallGlimNode(const rclcpp::NodeOptions& options): Node("small_glim_node", options) {
    tf_buffer = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener = std::make_unique<tf2_ros::TransformListener>(*tf_buffer);
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    auto config = std::make_shared<Config>(this);

    bool debug = config->param<bool>("node.debug");
    if (debug) {
        logger::info("node", "enable debug printing");
        get_logger().set_level(rclcpp::Logger::Level::Debug);
    }

    dump_on_unload = config->param<bool>("node.dump_on_unload");
    if (dump_on_unload) {
        logger::info("node", "dump_on_unload=true");
    }

    keep_raw_points = config->param<bool>("node.keep_raw_points");
    imu_time_offset = config->param<double>("node.imu_time_offset");
    points_time_offset = config->param<double>("node.points_time_offset");
    acc_scale = config->param<double>("node.acc_scale");

    intensity_field = config->param<std::string>("sensors.intensity_field");
    ring_field = config->param<std::string>("sensors.ring_field");

    // Preprocessing
    time_keeper = std::make_unique<TimeKeeper>(config);
    preprocessor = std::make_unique<CloudPreprocessor>(config);

    // Odometry estimation
    auto odom = std::make_shared<OdometryEstimationCPU>(config);
    odometry_estimation = std::make_unique<AsyncOdometryEstimation>(odom);

    // Sub mapping
    if (config->param<bool>("node.enable_local_mapping")) {
        auto sub = std::make_shared<SubMapping>(config);
        sub_mapping = std::make_unique<AsyncSubMapping>(sub);
    }

    // Global mapping
    if (config->param<bool>("node.enable_global_mapping")) {
        auto global = std::make_shared<GlobalMapping>(config);
        global_mapping = std::make_unique<AsyncGlobalMapping>(global);
    }

    // ROS-related
    const std::string imu_topic = config->param<std::string>("node.imu_topic");
    const std::string points_topic = config->param<std::string>("node.points_topic");
    const std::string odometry_topic = config->param<std::string>("node.odometry_topic");

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
    odometry_pub = create_publisher<nav_msgs::msg::Odometry>(odometry_topic, rclcpp::QoS(1));

    // Start timer
    timer = create_wall_timer(std::chrono::milliseconds(1), [this]() { timer_callback(); });
}

SmallGlimNode::~SmallGlimNode() {
    logger::debug("node", "quit");
    if (dump_on_unload) {
        std::string dump_path = "/tmp/dump";
        wait(true);
        save(dump_path);
    }
}

void SmallGlimNode::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    const double imu_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9 + imu_time_offset;
    const Eigen::Vector3d linear_acc = acc_scale * utils::convert_to<Eigen::Vector3d>(msg->linear_acceleration);
    const Eigen::Vector3d angular_vel = utils::convert_to<Eigen::Vector3d>(msg->angular_velocity);

    if (!time_keeper->validate_imu_stamp(imu_stamp)) {
        logger::warn("node", "skip an invalid IMU data (stamp={})", imu_stamp);
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
    const auto raw_points = std::make_shared<RawPoints>(*msg, intensity_field, ring_field);
    if (raw_points == nullptr) {
        logger::warn("node", "failed to extract points from message");
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
    logger::debug("node", "workload={}", workload);

    return workload;
}

void SmallGlimNode::timer_callback() {
    std::vector<EstimationFrame::ConstPtr> estimation_frames;
    std::vector<EstimationFrame::ConstPtr> marginalized_frames;
    odometry_estimation->get_results(estimation_frames, marginalized_frames);
    if (!estimation_frames.empty()) {
        send_odometry(estimation_frames.back());
    }

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

void SmallGlimNode::send_odometry(const EstimationFrame::ConstPtr frame) {
    if (!frame) return;
    const rclcpp::Time stamp(frame->stamp * 1e9);
    const Eigen::Isometry3d T_odom_imu = frame->T_world_imu;
    const Eigen::Isometry3d T_lidar_imu = frame->T_lidar_imu;
    const Eigen::Isometry3d T_odom_lidar = T_odom_imu * T_lidar_imu.inverse();

    geometry_msgs::msg::TransformStamped tf;
    tf.header.frame_id = "odom";
    tf.child_frame_id = "lidar";
    tf.header.stamp = stamp;
    utils::convert(T_odom_lidar, tf.transform);
    tf_broadcaster->sendTransform(tf);

    const Eigen::Vector3d v_odom_imu = frame->v_world_imu;
    const Eigen::Vector3d v_odom_lidar = T_odom_imu.linear() * v_odom_imu;
    nav_msgs::msg::Odometry odom;
    odom.header.frame_id = "odom";
    odom.child_frame_id = "lidar";
    odom.header.stamp = stamp;
    utils::convert(T_odom_lidar, odom.pose.pose);
    utils::convert(v_odom_lidar, odom.twist.twist.linear);
    odometry_pub->publish(odom);
}

void SmallGlimNode::wait(bool auto_quit) {
    logger::info("node", "waiting for odometry estimation");
    odometry_estimation->join();

    if (sub_mapping) {
        std::vector<EstimationFrame::ConstPtr> estimation_results;
        std::vector<EstimationFrame::ConstPtr> marginalized_frames;
        odometry_estimation->get_results(estimation_results, marginalized_frames);
        for (const auto& marginalized_frame: marginalized_frames) {
            sub_mapping->insert_frame(marginalized_frame);
        }

        logger::info("node", "waiting for local mapping");
        sub_mapping->join();

        const auto submaps = sub_mapping->get_results();
        if (global_mapping) {
            for (const auto& submap: submaps) {
                global_mapping->insert_submap(submap);
            }
            logger::info("node", "waiting for global mapping");
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