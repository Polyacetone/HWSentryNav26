#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <tf2_ros/static_transform_broadcaster.hpp>

#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>

#include <common_utils/convert.hpp>
#include <common_utils/ema_filter.hpp>
#include <interfaces/msg/robot_status.hpp>

namespace odom_localizer {
using namespace small_gicp;

class OdomLocalizerNode: public rclcpp::Node {
public:
    explicit OdomLocalizerNode(const rclcpp::NodeOptions& options);

private:
    int num_threads_, cov_num_neighbors_map_, cov_num_neighbors_lidar_, gicp_max_iterations_, min_points_lidar_;
    double downsample_resolution_map_, downsample_resolution_lidar_, gicp_max_correspondence_distance_;
    double odom_to_map_no_filter_distance_, odom_to_map_no_filter_angle_;
    double normalized_error_threshold_, overlap_threshold_;
    double robot_color_wait_timeout_;
    bool enable_debug_, enable_gicp_registration_, perform_gicp_only_once_;

    PointCloud::Ptr map_cloud_ = std::make_shared<PointCloud>();
    KdTree<PointCloud>::Ptr map_kd_tree_;
    PointCloud::Ptr source_cloud_;
    std::unique_ptr<utils::EMAFilter<Eigen::Isometry3d>> odom_to_map_;
    std::optional<Eigen::Isometry3d> last_imu_world_to_map_;
    bool gicp_performed_ = false, initial_transform_initialized_ = false;
    std::chrono::steady_clock::time_point initialize_time_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Subscription<interfaces::msg::RobotStatus>::SharedPtr robot_status_sub_;
    rclcpp::TimerBase::SharedPtr publish_timer_;

    void publish_timer_callback();
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void robot_status_callback(const interfaces::msg::RobotStatus::SharedPtr msg);
    bool perform_gicp_registration();
    void publish() const;
    void set_initial_transform(const std::vector<double>& initial_transform_vec);
    void update(const Eigen::Isometry3d& transform) const;
    std::optional<Eigen::Isometry3d> lookup_tf(const std::string& parent_frame, const std::string& child_frame) const;
    small_gicp::PointCloud::Ptr convert_pointcloud2_to_small_gicp(sensor_msgs::msg::PointCloud2::SharedPtr msg) const;
    small_gicp::PointCloud::Ptr convert_pcl_to_small_gicp(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud) const;
};

OdomLocalizerNode::OdomLocalizerNode(const rclcpp::NodeOptions& options): Node("odom_localizer", options) {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);
    num_threads_ = declare_parameter<int>("num_threads");
    enable_debug_ = declare_parameter<bool>("enable_debug");
    robot_color_wait_timeout_ = declare_parameter<double>("robot_color_wait_timeout");
    enable_gicp_registration_ = declare_parameter<bool>("enable_gicp_registration");
    perform_gicp_only_once_ = declare_parameter<bool>("perform_gicp_only_once");
    cov_num_neighbors_map_ = declare_parameter<int>("cov_num_neighbors_map");
    cov_num_neighbors_lidar_ = declare_parameter<int>("cov_num_neighbors_lidar");
    min_points_lidar_ = declare_parameter<int>("min_points_lidar");
    gicp_max_iterations_ = declare_parameter<int>("gicp_max_iterations");
    gicp_max_correspondence_distance_ = declare_parameter<double>("gicp_max_correspondence_distance");
    downsample_resolution_map_ = declare_parameter<double>("downsample_resolution_map");
    downsample_resolution_lidar_ = declare_parameter<double>("downsample_resolution_lidar");
    normalized_error_threshold_ = declare_parameter<double>("normalized_error_threshold");
    overlap_threshold_ = declare_parameter<double>("overlap_threshold");
    std::string map_cloud_filename = declare_parameter<std::string>("map_cloud_filename");
    std::string map_cloud_path = ament_index_cpp::get_package_share_directory("odom_localizer") + "/maps/" + map_cloud_filename;
    std::string source_cloud_sub_topic = declare_parameter<std::string>("source_cloud_sub_topic");
    double odom_to_map_filter_ratio = declare_parameter<double>("odom_to_map_filter_ratio");
    odom_to_map_ = std::make_unique<utils::EMAFilter<Eigen::Isometry3d>>(odom_to_map_filter_ratio);
    odom_to_map_no_filter_distance_ = declare_parameter<double>("odom_to_map_no_filter_distance");
    odom_to_map_no_filter_angle_ = declare_parameter<double>("odom_to_map_no_filter_angle");

    if (enable_debug_) {
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        RCLCPP_DEBUG(get_logger(), "Debug mode enabled");
    }

    robot_status_sub_ = create_subscription<interfaces::msg::RobotStatus>(
        declare_parameter<std::string>("robot_status_sub_topic"), 1,
        [this](interfaces::msg::RobotStatus::SharedPtr msg) { robot_status_callback(msg); }
    );

    if (enable_gicp_registration_) {
        auto map_cloud_pcl = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        if (pcl::io::loadPCDFile(map_cloud_path, *map_cloud_pcl) == -1) {
            throw std::runtime_error("Failed to load map PCD: " + map_cloud_path);
        }
        map_cloud_ = convert_pcl_to_small_gicp(map_cloud_pcl);
        RCLCPP_INFO(get_logger(), "Loaded map point cloud with %zu points", map_cloud_->size());

        if (downsample_resolution_map_ > 0) {
            map_cloud_ = voxelgrid_sampling_omp(*map_cloud_, downsample_resolution_map_, num_threads_);
            RCLCPP_INFO(get_logger(), "Downsampled map point cloud to %zu points", map_cloud_->size());
        }
        map_kd_tree_ = std::make_shared<KdTree<PointCloud>>(map_cloud_, KdTreeBuilderOMP(num_threads_));
        estimate_covariances_omp(*map_cloud_, *map_kd_tree_, cov_num_neighbors_map_, num_threads_);

        cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            source_cloud_sub_topic,
            rclcpp::QoS(1),
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { cloud_callback(msg); }
        );
    }

    double publish_rate = declare_parameter<double>("publish_rate");
    publish_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / publish_rate),
        [this]() { publish_timer_callback(); }
    );

    initialize_time_ = std::chrono::steady_clock::now();
}

void OdomLocalizerNode::cloud_callback(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    source_cloud_ = convert_pointcloud2_to_small_gicp(msg);
}

void OdomLocalizerNode::publish_timer_callback() {
    // 等待初始变换初始化
    if (!initial_transform_initialized_) {
        const auto now_time = std::chrono::steady_clock::now();
        // 超时未收到机器人颜色，使用默认初始变换
        if (std::chrono::duration<double>(now_time - initialize_time_).count() > robot_color_wait_timeout_) {
            RCLCPP_ERROR(get_logger(), "Robot color not initialized within timeout %.2f seconds", robot_color_wait_timeout_);
            initial_transform_initialized_ = true;
            set_initial_transform(declare_parameter<std::vector<double>>("initial_transform_default"));
        }
        return;
    }

    if (!enable_gicp_registration_ || (gicp_performed_ && perform_gicp_only_once_)) {
        // 如果没有启用GICP配准，或者已经完成过一次配准且只进行一次配准，则直接发布当前的odom_to_map
        publish();
        return;
    }

    if (!source_cloud_) {
        return;
    }

    if (source_cloud_->points.size() < min_points_lidar_) {
        RCLCPP_WARN(get_logger(), "Received point cloud has too few points (%zu), skipping frame", source_cloud_->points.size());
        return;
    }

    if (perform_gicp_registration()) {
        gicp_performed_ = true;
    }
    publish();

    last_imu_world_to_map_ = lookup_tf("map", "imu_world");
}

void OdomLocalizerNode::robot_status_callback(const interfaces::msg::RobotStatus::SharedPtr msg) {
    initial_transform_initialized_ = true;
    if (msg->robot_color) { // true为红色
        set_initial_transform(declare_parameter<std::vector<double>>("initial_transform_red"));
    } else { // false为蓝色
        set_initial_transform(declare_parameter<std::vector<double>>("initial_transform_blue"));
    }
}

bool OdomLocalizerNode::perform_gicp_registration() {
    KdTree<PointCloud>::Ptr source_kd_tree;
    if (downsample_resolution_lidar_ > 0) {
        source_cloud_ = voxelgrid_sampling_omp(*source_cloud_, downsample_resolution_lidar_, num_threads_);
    }
    source_kd_tree = std::make_shared<KdTree<PointCloud>>(source_cloud_, KdTreeBuilderOMP(num_threads_));
    estimate_covariances_omp(*source_cloud_, *source_kd_tree, cov_num_neighbors_lidar_, num_threads_);

    Registration<GICPFactor, ParallelReductionOMP> reg;
    reg.reduction.num_threads = num_threads_;
    reg.optimizer.max_iterations = gicp_max_iterations_;
    reg.rejector.max_dist_sq = gicp_max_correspondence_distance_ * gicp_max_correspondence_distance_;

    double min_normalized_error = std::numeric_limits<double>::max();
    double max_overlap = 0.0;
    const auto align = [&](const Eigen::Isometry3d& init) {
        const auto result = reg.align(*map_cloud_, *source_cloud_, *map_kd_tree_, init);
        const double normalized_error = result.error / result.num_inliers;
        const double overlap = static_cast<double>(result.num_inliers) / source_cloud_->size();
        min_normalized_error = std::min(min_normalized_error, normalized_error);
        max_overlap = std::max(max_overlap, overlap);
        RCLCPP_DEBUG(get_logger(), "Normalized error: %.4f, Overlap: %.4f", normalized_error, overlap);
        if (normalized_error < normalized_error_threshold_ && overlap > overlap_threshold_) {
            update(result.T_target_source);
            return true;
        }
        return false;
    };

    // 尝试使用之前的odom_to_map作为初始值进行对齐（基于里程计不漂移的假设）
    if (align(odom_to_map_->value())) {
        RCLCPP_DEBUG(get_logger(), "GICP succeed using initial: previous odom to map");
        return true;
    }

    // 尝试使用之前的imu_world_to_map计算得到的odom_to_map进行对齐（基于车体位置（imu_world）在map下是连续的假设）
    const auto imu_world_to_odom = lookup_tf("odom", "imu_world");
    if (last_imu_world_to_map_ && imu_world_to_odom) {
        const Eigen::Isometry3d imu_world_to_map = *last_imu_world_to_map_;
        const Eigen::Isometry3d odom_to_map = imu_world_to_map * imu_world_to_odom->inverse();
        if (align(odom_to_map)) {
            RCLCPP_DEBUG(get_logger(), "GICP succeed using initial: imu_world to map");
            return true;
        }
    }

    RCLCPP_INFO(get_logger(), "All GICP attempts failed for this frame! Minimum normalized error: %.4f, Maximum overlap: %.4f", min_normalized_error, max_overlap);
    return false;
}

void OdomLocalizerNode::update(const Eigen::Isometry3d& transform) const {
    if (!perform_gicp_only_once_ &&
        Eigen::Vector3d(transform.translation() - odom_to_map_->value().translation()).norm() < odom_to_map_no_filter_distance_ &&
        Eigen::AngleAxisd(transform.rotation().inverse() * odom_to_map_->value().rotation()).angle() < odom_to_map_no_filter_angle_) {
        odom_to_map_->update(transform);
        RCLCPP_DEBUG(get_logger(), "Updated odom->map with filtering, current T=(%.3f, %.3f, %.3f)", 
            odom_to_map_->value().translation().x(),
            odom_to_map_->value().translation().y(),
            odom_to_map_->value().translation().z()
        );
    } else {
        odom_to_map_->initialize(transform);
        RCLCPP_DEBUG(get_logger(), "Updated odom->map forcedly without filtering, current T=(%.3f, %.3f, %.3f)", 
            odom_to_map_->value().translation().x(),
            odom_to_map_->value().translation().y(),
            odom_to_map_->value().translation().z()
        );
    }
}

void OdomLocalizerNode::set_initial_transform(const std::vector<double>& initial_transform_vec) {
    if (initial_transform_vec.size() != 7) {
        RCLCPP_FATAL(get_logger(), "Initial transform parameter must have 7 elements");
        std::exit(EXIT_FAILURE);
    }
    Eigen::Isometry3d initial = Eigen::Isometry3d::Identity();
    initial.translate(Eigen::Vector3d(initial_transform_vec[0], initial_transform_vec[1], initial_transform_vec[2]));
    initial.rotate(Eigen::Quaterniond(initial_transform_vec[6], initial_transform_vec[3], initial_transform_vec[4], initial_transform_vec[5]));
    odom_to_map_->initialize(initial);
}

void OdomLocalizerNode::publish() const {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = now();
    tf.header.frame_id = "map";
    tf.child_frame_id = "odom";
    tf.transform = utils::convert_to<geometry_msgs::msg::Transform>(odom_to_map_->value());
    static_tf_broadcaster_->sendTransform(tf);
}

std::optional<Eigen::Isometry3d> OdomLocalizerNode::lookup_tf(const std::string& parent_frame, const std::string& child_frame) const {
    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = tf_buffer_->lookupTransform(parent_frame, child_frame, tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(get_logger(), "Could not transform %s to %s: %s", child_frame.c_str(), parent_frame.c_str(), ex.what());
        return std::nullopt;
    }
    return utils::convert_to<Eigen::Isometry3d>(tf.transform);
}

small_gicp::PointCloud::Ptr OdomLocalizerNode::convert_pointcloud2_to_small_gicp(sensor_msgs::msg::PointCloud2::SharedPtr msg) const {
    auto cloud = std::make_shared<small_gicp::PointCloud>();

    int offset_x = -1, offset_y = -1, offset_z = -1;
    for (const auto& field : msg->fields) {
        if (field.name == "x") offset_x = field.offset;
        else if (field.name == "y") offset_y = field.offset;
        else if (field.name == "z") offset_z = field.offset;
    }
    if (offset_x < 0 || offset_y < 0 || offset_z < 0) {
        RCLCPP_WARN(get_logger(), "PointCloud2 missing x/y/z fields");
        return cloud;
    }

    const size_t point_step = msg->point_step;
    const size_t num_points = msg->width * msg->height;
    const uint8_t* data_ptr = msg->data.data();
    for (size_t i = 0; i < num_points; i++) {
        cloud->points.emplace_back(
            *reinterpret_cast<const float*>(data_ptr + offset_x),
            *reinterpret_cast<const float*>(data_ptr + offset_y),
            *reinterpret_cast<const float*>(data_ptr + offset_z),
            1.0
        );
        data_ptr += point_step;
    }
    return cloud;
}

small_gicp::PointCloud::Ptr OdomLocalizerNode::convert_pcl_to_small_gicp(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud) const {
    auto small_gicp_cloud = std::make_shared<small_gicp::PointCloud>();
    small_gicp_cloud->points.resize(pcl_cloud->size());
    for (size_t i = 0; i < pcl_cloud->size(); i++) {
        small_gicp_cloud->points[i] = Eigen::Vector4d(
            pcl_cloud->points[i].x,
            pcl_cloud->points[i].y,
            pcl_cloud->points[i].z,
            1.0
        );
    }
    return small_gicp_cloud;
}
} // namespace odom_localizer

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(odom_localizer::OdomLocalizerNode)