#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>

#include <common_utils/convert.hpp>

namespace odom_localizer {
using namespace small_gicp;
using PointCov = pcl::PointCovariance;
using PointCovCloud = pcl::PointCloud<PointCov>;

class OdomLocalizerNode: public rclcpp::Node {
public:
    explicit OdomLocalizerNode(const rclcpp::NodeOptions& options);

private:
    unsigned num_threads_, cov_num_neighbors_, gicp_max_iterations_;
    double downsample_resolution_map_, downsample_resolution_lidar_, gicp_max_correspondence_distance_;
    double fitness_score_max_dist_, fitness_score_threshold_;

    PointCovCloud::Ptr map_down_;
    KdTree<PointCovCloud>::Ptr map_kd_tree_;
    Eigen::Isometry3d last_successful_odom_to_map_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void publish_tf(const Eigen::Isometry3d& transform, const rclcpp::Time& stamp) const;
    double calc_fitness_score(const PointCovCloud::Ptr source_down, const Eigen::Isometry3d& transform) const;
};

OdomLocalizerNode::OdomLocalizerNode(const rclcpp::NodeOptions& options): Node("odom_localizer", options) {
    num_threads_ = declare_parameter<int>("num_threads");
    cov_num_neighbors_ = declare_parameter<int>("cov_num_neighbors");
    gicp_max_iterations_ = declare_parameter<int>("gicp_max_iterations");
    gicp_max_correspondence_distance_ = declare_parameter<double>("gicp_max_correspondence_distance");
    downsample_resolution_map_ = declare_parameter<double>("downsample_resolution_map");
    downsample_resolution_lidar_ = declare_parameter<double>("downsample_resolution_lidar");
    fitness_score_max_dist_ = declare_parameter<double>("fitness_score_max_dist");
    fitness_score_threshold_ = declare_parameter<double>("fitness_score_threshold");
    std::string pcd_path = declare_parameter<std::string>("pcd_path");
    std::string cloud_sub_topic = declare_parameter<std::string>("cloud_sub_topic");
    Eigen::Vector3d initial_pos;
    initial_pos.x() = declare_parameter<double>("initial_pos_x");
    initial_pos.y() = declare_parameter<double>("initial_pos_y");
    initial_pos.z() = declare_parameter<double>("initial_pos_z");
    last_successful_odom_to_map_.setIdentity();
    last_successful_odom_to_map_.translate(initial_pos);

    auto map_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (pcl::io::loadPCDFile(pcd_path, *map_cloud) == -1) {
        RCLCPP_FATAL(get_logger(), "Failed to load map PCD: %s", pcd_path.c_str());
        throw std::runtime_error("Failed to load map PCD");
    }
    RCLCPP_INFO(get_logger(), "Loaded map point cloud with %zu points", map_cloud->size());
    map_down_ = voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZ>, PointCovCloud>(
        *map_cloud, downsample_resolution_map_, num_threads_
    );
    map_kd_tree_ = std::make_shared<KdTree<PointCovCloud>>(map_down_, KdTreeBuilderOMP(num_threads_));
    estimate_covariances_omp(*map_down_, *map_kd_tree_, cov_num_neighbors_, num_threads_);
    RCLCPP_INFO(get_logger(), "Downsampled map point cloud to %zu points", map_down_->size());

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_sub_topic,
        rclcpp::QoS(1),
        [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { cloud_callback(msg); }
    );
}

void OdomLocalizerNode::cloud_callback(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);

    PointCovCloud::Ptr source_down;
    KdTree<PointCovCloud>::Ptr source_kd_tree;
    source_down = voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZ>, PointCovCloud>(*cloud, downsample_resolution_lidar_, num_threads_);
    source_kd_tree = std::make_shared<KdTree<PointCovCloud>>(source_down, KdTreeBuilderOMP(num_threads_));
    estimate_covariances_omp(*source_down, *source_kd_tree, cov_num_neighbors_, num_threads_);

    Registration<GICPFactor, ParallelReductionOMP> reg;
    reg.reduction.num_threads = num_threads_;
    reg.optimizer.max_iterations = gicp_max_iterations_;
    reg.rejector.max_dist_sq = gicp_max_correspondence_distance_ * gicp_max_correspondence_distance_;

    const auto result = reg.align(*map_down_, *source_down, *map_kd_tree_, last_successful_odom_to_map_);
    if (result.converged) {
        const float fitness_score = calc_fitness_score(source_down, result.T_target_source);
        if (fitness_score < fitness_score_threshold_) {
            publish_tf(result.T_target_source, msg->header.stamp);
            last_successful_odom_to_map_ = result.T_target_source;
            return;
        }
    }
    RCLCPP_WARN(get_logger(), "GICP failed to converge!");
}

void OdomLocalizerNode::publish_tf(const Eigen::Isometry3d& transform, const rclcpp::Time& stamp) const {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = "map";
    t.child_frame_id = "odom";
    t.transform = utils::convert_to<geometry_msgs::msg::Transform>(transform);
    tf_broadcaster_->sendTransform(t);
}

double OdomLocalizerNode::calc_fitness_score(
    const PointCovCloud::Ptr source_down,
    const Eigen::Isometry3d& transform
) const {
    double sum_dist = 0;
    unsigned num_points = 0;
    #pragma omp parallel for num_threads(num_threads_) schedule(guided) \
    reduction(+:num_points) reduction(+:sum_dist)
    for (int i = 0; i < source_down->size(); i++) {
        const PointCov& point_cov = source_down->points[i];
        Eigen::Vector4d query = transform * point_cov.getVector4fMap().cast<double>();
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
} // namespace odom_localizer

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(odom_localizer::OdomLocalizerNode)