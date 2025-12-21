#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

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

namespace odom_localizer {
using namespace small_gicp;

class OdomLocalizerNode: public rclcpp::Node {
public:
    explicit OdomLocalizerNode(const rclcpp::NodeOptions& options);

private:
    int num_threads_, cov_num_neighbors_map_, cov_num_neighbors_lidar_, gicp_max_iterations_;
    double downsample_resolution_map_, downsample_resolution_lidar_, gicp_max_correspondence_distance_;
    double fitness_score_max_dist_, fitness_score_threshold_;
    double odom_to_map_no_filter_distance_, odom_to_map_no_filter_angle_;
    bool enable_debug_;

    PointCloud::Ptr map_cloud_ = std::make_shared<PointCloud>();
    KdTree<PointCloud>::Ptr map_kd_tree_;
    std::unique_ptr<utils::EMAFilter<Eigen::Isometry3d>> odom_to_map_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void update_and_publish(const Eigen::Isometry3d& transform, const rclcpp::Time& stamp) const;
    double calc_fitness_score(const PointCloud::Ptr source_down, const Eigen::Isometry3d& transform) const;
    small_gicp::PointCloud::Ptr convert_pointcloud2_to_small_gicp(sensor_msgs::msg::PointCloud2::SharedPtr msg) const;
    small_gicp::PointCloud::Ptr convert_pcl_to_small_gicp(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud) const;
};

OdomLocalizerNode::OdomLocalizerNode(const rclcpp::NodeOptions& options): Node("odom_localizer", options) {
    num_threads_ = declare_parameter<int>("num_threads");
    enable_debug_ = declare_parameter<bool>("enable_debug");
    cov_num_neighbors_map_ = declare_parameter<int>("cov_num_neighbors_map");
    cov_num_neighbors_lidar_ = declare_parameter<int>("cov_num_neighbors_lidar");
    gicp_max_iterations_ = declare_parameter<int>("gicp_max_iterations");
    gicp_max_correspondence_distance_ = declare_parameter<double>("gicp_max_correspondence_distance");
    downsample_resolution_map_ = declare_parameter<double>("downsample_resolution_map");
    downsample_resolution_lidar_ = declare_parameter<double>("downsample_resolution_lidar");
    fitness_score_max_dist_ = declare_parameter<double>("fitness_score_max_dist");
    fitness_score_threshold_ = declare_parameter<double>("fitness_score_threshold");
    std::string map_cloud_filename = declare_parameter<std::string>("map_cloud_filename");
    std::string map_cloud_path = ament_index_cpp::get_package_share_directory("odom_localizer") + "/maps/" + map_cloud_filename;
    std::string source_cloud_sub_topic = declare_parameter<std::string>("source_cloud_sub_topic");
    double odom_to_map_filter_ratio = declare_parameter<double>("odom_to_map_filter_ratio");
    odom_to_map_ = std::make_unique<utils::EMAFilter<Eigen::Isometry3d>>(odom_to_map_filter_ratio);
    odom_to_map_no_filter_distance_ = declare_parameter<double>("odom_to_map_no_filter_distance");
    odom_to_map_no_filter_angle_ = declare_parameter<double>("odom_to_map_no_filter_angle");
    std::vector<double> initial_transform = declare_parameter<std::vector<double>>("initial_transform");
    if (initial_transform.size() != 7) throw std::invalid_argument("initial_transform requires 7 arguments");
    Eigen::Isometry3d initial = Eigen::Isometry3d::Identity();
    initial.translate(Eigen::Vector3d(initial_transform[0], initial_transform[1], initial_transform[2]));
    initial.rotate(Eigen::Quaterniond(initial_transform[6], initial_transform[3], initial_transform[4], initial_transform[5]));
    odom_to_map_->initialize(initial);

    if (enable_debug_) {
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        RCLCPP_DEBUG(get_logger(), "Debug mode enabled");
    }

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

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        source_cloud_sub_topic,
        rclcpp::QoS(1),
        [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { cloud_callback(msg); }
    );
}

void OdomLocalizerNode::cloud_callback(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto source = convert_pointcloud2_to_small_gicp(msg);
    if (source->points.size() < 100) {
        RCLCPP_INFO(get_logger(), "Received point cloud has too few points (%zu), skipping frame", source->points.size());
        return;
    }

    KdTree<PointCloud>::Ptr source_kd_tree;
    if (downsample_resolution_lidar_ > 0) {
        source = voxelgrid_sampling_omp(*source, downsample_resolution_lidar_, num_threads_);
    }
    source_kd_tree = std::make_shared<KdTree<PointCloud>>(source, KdTreeBuilderOMP(num_threads_));
    estimate_covariances_omp(*source, *source_kd_tree, cov_num_neighbors_lidar_, num_threads_);

    Registration<GICPFactor, ParallelReductionOMP> reg;
    reg.reduction.num_threads = num_threads_;
    reg.optimizer.max_iterations = gicp_max_iterations_;
    reg.rejector.max_dist_sq = gicp_max_correspondence_distance_ * gicp_max_correspondence_distance_;

    float min_fitness_score = std::numeric_limits<float>::max();
    const auto align = [&](const Eigen::Isometry3d& init) {
        const auto result = reg.align(
            *map_cloud_, *source, *map_kd_tree_, init
        );
        if (result.converged) {
            const float fitness_score = calc_fitness_score(source, result.T_target_source);
            min_fitness_score = std::min(min_fitness_score, fitness_score);
            RCLCPP_DEBUG(get_logger(), "Fitness score: %.4f", fitness_score);
            if (fitness_score < fitness_score_threshold_) {
                update_and_publish(result.T_target_source, msg->header.stamp);
                return true;
            }
        }
        return false;
    };

    // 尝试使用之前的odom_to_map作为初始值进行对齐
    if (align(odom_to_map_->value())) {
        RCLCPP_DEBUG(get_logger(), "GICP succeed using initial: previous odom to map");
        return;
    }

    if (min_fitness_score == std::numeric_limits<float>::max()) {
        RCLCPP_WARN(get_logger(), "All GICP attempts failed for this frame! No attempt converged!");
    } else {
        RCLCPP_WARN(get_logger(), "All GICP attempts failed for this frame! Minimum fitness score: %.4f", min_fitness_score);
    }
}

void OdomLocalizerNode::update_and_publish(const Eigen::Isometry3d& transform, const rclcpp::Time& stamp) const {
    if (Eigen::Vector3d(transform.translation() - odom_to_map_->value().translation()).norm() < odom_to_map_no_filter_distance_ &&
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
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = "map";
    tf.child_frame_id = "odom";
    tf.transform = utils::convert_to<geometry_msgs::msg::Transform>(odom_to_map_->value());
    tf_broadcaster_->sendTransform(tf);
}

double OdomLocalizerNode::calc_fitness_score(
    const PointCloud::Ptr source,
    const Eigen::Isometry3d& transform
) const {
    double sum_dist = 0;
    int num_points = 0;
    #pragma omp parallel for num_threads(num_threads_) schedule(guided) \
    reduction(+:num_points) reduction(+:sum_dist)
    for (int i = 0; i < source->size(); i++) {
        Eigen::Vector4d query = transform * source->points[i];
        size_t index; double dist;
        map_kd_tree_->nearest_neighbor_search(query, &index, &dist);
        if (dist <= fitness_score_max_dist_) {
            sum_dist += dist;
            num_points++;
        }
    }
    if (num_points > 0) return sum_dist / num_points;
    else return std::numeric_limits<double>::max();
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