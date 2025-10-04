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

namespace localizer {
using namespace small_gicp;
using PointCov = pcl::PointCovariance;
using PointCovCloud = pcl::PointCloud<PointCov>;

class LocalizerNode: public rclcpp::Node {
public:
    explicit LocalizerNode(const rclcpp::NodeOptions& options);

private:
    unsigned num_threads, cov_num_neighbors, gicp_max_iterations, search_max_spread_num;
    float leaf_size_map, leaf_size_lidar, gicp_max_correspondence_distance;
    float fitness_score_max_dist, fitness_score_threshold;
    float search_spread_interval;

    PointCovCloud::Ptr map_down_;
    KdTree<PointCovCloud>::Ptr map_kd_tree_;
    std::vector<Eigen::Vector2f> search_relative_pos_;
    float map_major_x_min, map_major_x_max, map_major_y_min, map_major_y_max;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const;
    void publish_tf(const Eigen::Isometry3d& transform, const rclcpp::Time& stamp) const;
    float calc_fitness_score(const PointCovCloud::Ptr source_down, const Eigen::Isometry3d& transform) const;
};

LocalizerNode::LocalizerNode(const rclcpp::NodeOptions& options): Node("localizer", options) {
    num_threads = declare_parameter<int>("num_threads");
    cov_num_neighbors = declare_parameter<int>("cov_num_neighbors");
    gicp_max_iterations = declare_parameter<int>("gicp_max_iterations");
    search_max_spread_num = declare_parameter<int>("search_max_spread_num");
    gicp_max_correspondence_distance = declare_parameter<float>("gicp_max_correspondence_distance");
    leaf_size_map = declare_parameter<float>("leaf_size_map");
    leaf_size_lidar = declare_parameter<float>("leaf_size_lidar");
    fitness_score_max_dist = declare_parameter<float>("fitness_score_max_dist");
    fitness_score_threshold = declare_parameter<float>("fitness_score_threshold");
    search_spread_interval = declare_parameter<float>("search_spread_interval");
    float major_part_proportion = declare_parameter<float>("major_part_proportion");
    std::string pcd_path = declare_parameter<std::string>("pcd_path");
    std::string cloud_sub_topic = declare_parameter<std::string>("cloud_sub_topic");

    auto map_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (pcl::io::loadPCDFile(pcd_path, *map_cloud) == -1) {
        RCLCPP_FATAL(get_logger(), "Couldn't read map PCD file: %s", pcd_path.c_str());
        throw std::runtime_error("Failed to load map PCD");
    }
    RCLCPP_INFO(get_logger(), "Loaded map point cloud with %zu points", map_cloud->size());
    map_down_ = voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZ>, PointCovCloud>(*map_cloud, leaf_size_map, num_threads);
    map_kd_tree_ = std::make_shared<KdTree<PointCovCloud>>(map_down_, KdTreeBuilderOMP(num_threads));
    estimate_covariances_omp(*map_down_, *map_kd_tree_, cov_num_neighbors, num_threads);
    RCLCPP_INFO(get_logger(), "Downsampled map point cloud to %zu points", map_down_->size());

    std::vector<float> map_coord_x, map_coord_y;
    for (const auto& pt: map_down_->points) {
        map_coord_x.emplace_back(pt.x);
        map_coord_y.emplace_back(pt.y);
    }
    constexpr auto comp = [](float x, float y) { return x < y; };
    quick_sort_omp(map_coord_x.begin(), map_coord_x.end(), comp, num_threads);
    quick_sort_omp(map_coord_y.begin(), map_coord_y.end(), comp, num_threads);
    const size_t size = map_down_->size();
    map_major_x_min = map_coord_x[size * (0.5 - major_part_proportion / 2)];
    map_major_x_max = map_coord_x[size * (0.5 + major_part_proportion / 2)];
    map_major_y_min = map_coord_y[size * (0.5 - major_part_proportion / 2)];
    map_major_y_max = map_coord_y[size * (0.5 + major_part_proportion / 2)];
    RCLCPP_INFO(
        get_logger(),
        "Downsampled map major part distribution: x: [%.3f, %.3f], y: [%.3f, %.3f]",
        map_major_x_min, map_major_x_max,
        map_major_y_min, map_major_y_max
    );

    search_relative_pos_.emplace_back(0, 0);
    for (int spread = 1; spread <= search_max_spread_num; spread++) {
        for (int i = -spread; i <= spread; i++) {
            int x = i * search_spread_interval;
            int y = (spread - std::abs(i)) * search_spread_interval;
            search_relative_pos_.emplace_back(x, y);
            if (y != 0) search_relative_pos_.emplace_back(x, -y);
        }
    }

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_sub_topic,
        rclcpp::QoS(1),
        [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { cloud_callback(msg); }
    );
}

void LocalizerNode::cloud_callback(sensor_msgs::msg::PointCloud2::SharedPtr msg) const {
    auto start = std::chrono::high_resolution_clock::now();

    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);

    PointCovCloud::Ptr source_down;
    KdTree<PointCovCloud>::Ptr source_kd_tree;
    source_down = voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZ>, PointCovCloud>(*cloud, leaf_size_lidar, num_threads);
    source_kd_tree = std::make_shared<KdTree<PointCovCloud>>(source_down, KdTreeBuilderOMP(num_threads));
    estimate_covariances_omp(*source_down, *source_kd_tree, cov_num_neighbors, num_threads);

    Registration<GICPFactor, ParallelReductionOMP> reg;
    reg.reduction.num_threads = num_threads;
    reg.optimizer.max_iterations = gicp_max_iterations;
    reg.rejector.max_dist_sq = gicp_max_correspondence_distance * gicp_max_correspondence_distance;

    Eigen::Isometry3d init = Eigen::Isometry3d::Identity();
    for (const auto& relative_pos: search_relative_pos_) {
        Eigen::Isometry3d search_pose = init;
        search_pose.translation()(0) += relative_pos.x();
        search_pose.translation()(1) += relative_pos.y();
        if (search_pose.translation()(0) < map_major_x_min) continue;
        if (search_pose.translation()(0) > map_major_x_max) continue;
        if (search_pose.translation()(1) < map_major_y_min) continue;
        if (search_pose.translation()(1) > map_major_y_max) continue;
        std::cout << "Trying: " << search_pose.translation().x() << ", " << search_pose.translation().y() << std::endl;
        auto result = reg.align(*map_down_, *source_down, *map_kd_tree_, search_pose);
        float fitness_score = calc_fitness_score(source_down, result.T_target_source);
        if (result.converged && fitness_score < fitness_score_threshold) {
            publish_tf(result.T_target_source, msg->header.stamp);
            std::cout << "Time: " << (std::chrono::high_resolution_clock::now() - start).count() / 1e6 << "ms" << std::endl;
            return;
        }
    }

    std::cout << "Time: " << (std::chrono::high_resolution_clock::now() - start).count() / 1e6 << "ms" << std::endl;
    RCLCPP_WARN(get_logger(), "Localization failed.");
}

void LocalizerNode::publish_tf(const Eigen::Isometry3d& transform, const rclcpp::Time& stamp) const {
    Eigen::Vector3d translation(transform.translation());
    Eigen::Quaterniond rotation(transform.rotation());
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = "map";
    t.child_frame_id = "lidar";
    t.transform.translation.x = translation.x();
    t.transform.translation.y = translation.y();
    t.transform.translation.z = translation.z();
    t.transform.rotation.x = rotation.x();
    t.transform.rotation.y = rotation.y();
    t.transform.rotation.z = rotation.z();
    t.transform.rotation.w = rotation.w();
    tf_broadcaster_->sendTransform(t);
}

float LocalizerNode::calc_fitness_score(
    const PointCovCloud::Ptr source_down,
    const Eigen::Isometry3d& transform
) const {
    double sum_dist = 0;
    unsigned num_points = 0;
    const unsigned size = source_down->size();
    #pragma omp parallel for num_threads(num_threads) schedule(guided) \
    reduction(+:num_points) reduction(+:sum_dist)
    for (int i = 0; i < size; i++) {
        const PointCov& point_cov = source_down->points[i];
        Eigen::Vector4d query = transform * point_cov.getVector4fMap().cast<double>();
        size_t index; double dist;
        map_kd_tree_->nearest_neighbor_search(query, &index, &dist);
        if (dist <= fitness_score_max_dist) {
            sum_dist += dist;
            num_points++;
        }
    }
    if (num_points > 0) return sum_dist / num_points;
    else return std::numeric_limits<float>::max();
}
} // namespace localizer

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(localizer::LocalizerNode)