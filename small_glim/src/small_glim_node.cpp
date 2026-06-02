#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <gtsam_points/optimizers/linearization_hook.hpp>
#include <common_utils/convert.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <interfaces/msg/comp_stage.hpp>

#include <small_glim/common/config.hpp>
#include <small_glim/common/logger.hpp>
#include <small_glim/preprocess/cloud_preprocessor.hpp>
#include <small_glim/preprocess/time_keeper.hpp>
#include <small_glim/odometry/async_odometry_estimation.hpp>
#include <small_glim/mapping/async_mapping.hpp>

namespace small_glim {

class SmallGlimNode: public rclcpp::Node {
public:
    explicit SmallGlimNode(const rclcpp::NodeOptions& options);
    ~SmallGlimNode() override;

    void timer_callback();
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
    size_t lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
    void comp_stage_callback(const interfaces::msg::CompStage::SharedPtr msg);

    void pub_odometry(const EstimationFrame::ConstPtr frame);
    void pub_cloud(const EstimationFrame::ConstPtr frame, const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher);
    void wait();

private:
    std::unique_ptr<tf2_ros::Buffer> tf_buffer;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster;

    std::shared_ptr<Config> config;
    std::unique_ptr<CloudPreprocessor> preprocessor;
    std::unique_ptr<TimeKeeper> time_keeper;
    std::unique_ptr<AsyncOdometryEstimation> odometry_estimation;
    std::unique_ptr<AsyncMapping> mapping;

    double imu_time_offset;
    double lidar_time_offset;
    double acc_scale;
    bool enable_mapping;
    bool use_mapping_trigger;
    bool enable_tf_publish;

    std::string intensity_field, ring_field;
    std::string imu_frame_id, lidar_frame_id, odometry_frame_id;

    uint8_t prev_game_progress = 0;

    rclcpp::TimerBase::SharedPtr timer;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub;
    rclcpp::Subscription<interfaces::msg::CompStage>::SharedPtr comp_stage_sub;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr registered_cloud_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ivox_cloud_pub;
};

}

namespace small_glim {

SmallGlimNode::SmallGlimNode(const rclcpp::NodeOptions& options): Node("small_glim", options) {
    tf_buffer = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener = std::make_unique<tf2_ros::TransformListener>(*tf_buffer);
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    tf_static_broadcaster = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);

    config = std::make_shared<Config>(this);
    bool debug = config->param<bool>("node.debug");
    if (debug) {
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        logger::debug("node", "enable debug printing");
    }

    enable_tf_publish = config->param<bool>("node.enable_tf_publish");
    enable_mapping = config->param<bool>("node.enable_mapping");
    use_mapping_trigger = config->param<bool>("node.use_mapping_trigger");

    imu_time_offset = config->param<double>("node.imu_time_offset");
    lidar_time_offset = config->param<double>("node.lidar_time_offset");
    acc_scale = config->param<double>("node.acc_scale");

    intensity_field = config->param<std::string>("sensors.intensity_field");
    ring_field = config->param<std::string>("sensors.ring_field");
    imu_frame_id = config->param<std::string>("node.imu_frame_id");
    lidar_frame_id = config->param<std::string>("node.lidar_frame_id");
    odometry_frame_id = config->param<std::string>("node.odometry_frame_id");

    // Preprocessing
    time_keeper = std::make_unique<TimeKeeper>(config);
    preprocessor = std::make_unique<CloudPreprocessor>(config);

    // Odometry estimation
    odometry_estimation = std::make_unique<AsyncOdometryEstimation>(config);

    // Mapping
    if (enable_mapping && !use_mapping_trigger) {
        mapping = std::make_unique<AsyncMapping>(config);
    }

    // ROS-related
    const std::string imu_sub_topic = config->param<std::string>("node.imu_sub_topic");
    const std::string lidar_sub_topic = config->param<std::string>("node.lidar_sub_topic");
    const std::string odometry_pub_topic = config->param<std::string>("node.odometry_pub_topic");
    const std::string registered_cloud_pub_topic = config->param<std::string>("node.registered_cloud_pub_topic");
    const std::string ivox_cloud_pub_topic = config->param<std::string>("node.ivox_cloud_pub_topic");
    const std::string comp_stage_sub_topic = config->param<std::string>("node.comp_stage_sub_topic");

    // Subscribers
    imu_sub = create_subscription<sensor_msgs::msg::Imu>(
        imu_sub_topic,
        rclcpp::QoS(3),
        [this](const sensor_msgs::msg::Imu::SharedPtr msg) { imu_callback(msg); }
    );
    lidar_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_sub_topic,
        rclcpp::QoS(1),
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { lidar_callback(msg); }
    );
    comp_stage_sub = create_subscription<interfaces::msg::CompStage>(
        comp_stage_sub_topic,
        rclcpp::QoS(1),
        [this](const interfaces::msg::CompStage::SharedPtr msg) { comp_stage_callback(msg); }
    );
    odometry_pub = create_publisher<nav_msgs::msg::Odometry>(odometry_pub_topic, rclcpp::QoS(1));
    registered_cloud_pub = create_publisher<sensor_msgs::msg::PointCloud2>(registered_cloud_pub_topic, rclcpp::QoS(1));
    ivox_cloud_pub = create_publisher<sensor_msgs::msg::PointCloud2>(ivox_cloud_pub_topic, rclcpp::QoS(1));

    // Start timer
    timer = create_wall_timer(std::chrono::milliseconds(1), [this]() { timer_callback(); });
}

SmallGlimNode::~SmallGlimNode() {
    logger::info("node", "waiting for odometry estimation");
    odometry_estimation->join();
    if (mapping) {
        std::vector<EstimationFrame::ConstPtr> estimation_results;
        std::vector<EstimationFrame::ConstPtr> target_ivox_frames;
        std::vector<EstimationFrame::ConstPtr> marginalized_frames;
        odometry_estimation->get_results(estimation_results, target_ivox_frames, marginalized_frames);
        for (const auto& marginalized_frame: marginalized_frames) {
            mapping->insert_frame(marginalized_frame);
        }
        logger::info("node", "waiting for mapping");
        mapping->request_finish();
        mapping->join();
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
}

size_t SmallGlimNode::lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    const auto raw_points = std::make_shared<RawPoints>(*msg, intensity_field, ring_field);
    if (raw_points == nullptr) {
        logger::warn("node", "failed to extract points from message");
        return 0;
    }
    raw_points->stamp += lidar_time_offset;
    time_keeper->process(raw_points);
    auto preprocessed = preprocessor->preprocess(raw_points);
    odometry_estimation->insert_frame(preprocessed);
    const size_t workload = odometry_estimation->workload();
    logger::debug("node", "workload={}", workload);
    return workload;
}

void SmallGlimNode::timer_callback() {
    std::vector<EstimationFrame::ConstPtr> estimation_frames;
    std::vector<EstimationFrame::ConstPtr> target_ivox_frames;
    std::vector<EstimationFrame::ConstPtr> marginalized_frames;
    odometry_estimation->get_results(estimation_frames, target_ivox_frames, marginalized_frames);
    if (!estimation_frames.empty()) {
        pub_odometry(estimation_frames.back());
        pub_cloud(estimation_frames.back(), registered_cloud_pub);
    }
    if (!target_ivox_frames.empty()) {
        pub_cloud(target_ivox_frames.back(), ivox_cloud_pub);
    }
    if (mapping) {
        for (const auto& marginalized_frame: marginalized_frames) {
            mapping->insert_frame(marginalized_frame);
        }
    }
}

void SmallGlimNode::comp_stage_callback(const interfaces::msg::CompStage::SharedPtr msg) {
    const uint8_t game_progress = msg->game_progress;
    if (enable_mapping && use_mapping_trigger) {
        if (prev_game_progress != 4 && game_progress == 4) {
            logger::info("node", "receive start mapping signal");
            mapping = std::make_unique<AsyncMapping>(config);
        }
        if (prev_game_progress == 4 && game_progress != 4) {
            logger::info("node", "receive stop mapping signal");
            mapping.reset();
            sync();  // fsync map files to disk (unistd.h ::sync)
        }
    }
    prev_game_progress = game_progress;
}

void SmallGlimNode::pub_odometry(const EstimationFrame::ConstPtr frame) {
    const rclcpp::Time stamp(static_cast<int64_t>(frame->stamp * 1e9));
    const Eigen::Isometry3d T_odom_imu = frame->T_world_imu;
    const Eigen::Vector3d v_odom_imu = frame->v_world_imu;
    const Eigen::Isometry3d T_lidar_imu = frame->T_lidar_imu;

    if (enable_tf_publish) {
        geometry_msgs::msg::TransformStamped tf_lidar_to_imu;
        tf_lidar_to_imu.header.frame_id = imu_frame_id;
        tf_lidar_to_imu.child_frame_id = lidar_frame_id;
        tf_lidar_to_imu.header.stamp = stamp;
        utils::convert(T_lidar_imu.inverse(), tf_lidar_to_imu.transform);
        tf_static_broadcaster->sendTransform(tf_lidar_to_imu);
        geometry_msgs::msg::TransformStamped tf_imu_to_odom;
        tf_imu_to_odom.header.frame_id = odometry_frame_id;
        tf_imu_to_odom.child_frame_id = imu_frame_id;
        tf_imu_to_odom.header.stamp = stamp;
        utils::convert(T_odom_imu, tf_imu_to_odom.transform);
        tf_broadcaster->sendTransform(tf_imu_to_odom);
    }

    nav_msgs::msg::Odometry odom;
    odom.header.frame_id = odometry_frame_id;
    odom.child_frame_id = imu_frame_id;
    odom.header.stamp = stamp;
    utils::convert(T_odom_imu, odom.pose.pose);
    utils::convert(v_odom_imu, odom.twist.twist.linear);
    odometry_pub->publish(odom);
}

void SmallGlimNode::pub_cloud(
    const EstimationFrame::ConstPtr frame,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher
) {
    const size_t num_points = frame->frame->num_points;
    const auto& points = frame->frame->points;
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.frame_id = odometry_frame_id;
    msg.header.stamp = rclcpp::Time(static_cast<int64_t>(frame->stamp * 1e9));
    msg.height = 1;
    msg.width = static_cast<uint32_t>(num_points);
    msg.is_dense = true;
    msg.point_step = 12;
    msg.row_step = static_cast<uint32_t>(12 * num_points);
    sensor_msgs::msg::PointField field_x;
    field_x.name = "x";
    field_x.offset = 0;
    field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_x.count = 1;
    sensor_msgs::msg::PointField field_y;
    field_y.name = "y";
    field_y.offset = 4;
    field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_y.count = 1;
    sensor_msgs::msg::PointField field_z;
    field_z.name = "z";
    field_z.offset = 8;
    field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_z.count = 1;
    msg.fields = {field_x, field_y, field_z};
    msg.data.resize(msg.row_step * msg.height);
    for (size_t i = 0; i < num_points; i++) {
        Eigen::Vector3f pt = points[i](Eigen::seq(0, 2)).cast<float>();
        if (frame->frame_type != FrameType::WORLD) pt = frame->T_world_frame().cast<float>() * pt;
        std::memcpy(msg.data.data() + i * 12, pt.data(), sizeof(pt));
    }
    publisher->publish(msg);
}

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(small_glim::SmallGlimNode)