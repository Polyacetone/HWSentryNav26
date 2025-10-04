#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

namespace terrain_analysis {
using Point = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<Point>;
using PointCov = pcl::PointCovariance;
using PointCovCloud = pcl::PointCloud<PointCov>;
using PointNor = pcl::PointNormal;
using PointNorCloud = pcl::PointCloud<PointNor>;

class TerrainAnalysisNode: public rclcpp::Node {
public:
    explicit TerrainAnalysisNode(const rclcpp::NodeOptions& options);

private:
    unsigned num_threads_, cloud_queue_size_, num_neighbors_;
    unsigned map_x_size_, map_y_size_, gaussian_blur_size_;
    float voxel_leaf_size_, max_radius_, max_relative_z_, min_relative_z_;

    std::deque<PointCloud::Ptr> cloud_queue_;
    nav_msgs::msg::OccupancyGrid global_costmap_;
    cv::Mat close_kernel_, dilate_kernel_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_costmap_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    void timer_callback();
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    cv::Mat cost_analysis_plane(const PointCloud::Ptr cloud, const Eigen::Vector3f& car_to_odom);
    nav_msgs::msg::OccupancyGrid to_occupancy_grid_msg(const cv::Mat& costmap);
};

TerrainAnalysisNode::TerrainAnalysisNode(const rclcpp::NodeOptions& options): Node("terrain_analysis_node", options) {
    num_threads_ = declare_parameter<int>("num_threads");
    cloud_queue_size_ = declare_parameter<int>("cloud_queue_size");
    num_neighbors_ = declare_parameter<int>("num_neighbors");
    map_x_size_ = declare_parameter<int>("map_x_size");
    map_y_size_ = declare_parameter<int>("map_y_size");
    gaussian_blur_size_ = declare_parameter<int>("gaussian_blur_size");
    voxel_leaf_size_ = declare_parameter<float>("voxel_leaf_size");
    max_radius_ = declare_parameter<float>("max_radius");
    max_relative_z_ = declare_parameter<float>("max_relative_z");
    min_relative_z_ = declare_parameter<float>("min_relative_z");

    int close_kernel_size = declare_parameter<int>("close_kernel_size");
    close_kernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, {close_kernel_size, close_kernel_size});
    int dilate_kernel_size = declare_parameter<int>("dilate_kernel_size");
    dilate_kernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, {dilate_kernel_size, dilate_kernel_size});

    std::string cloud_sub_topic = declare_parameter<std::string>("cloud_sub_topic");
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_sub_topic, rclcpp::QoS(1),
        [&](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { cloud_callback(msg); }
    );

    std::string global_costmap_pub_topic = declare_parameter<std::string>("global_costmap_pub_topic");
    global_costmap_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
        global_costmap_pub_topic,
        rclcpp::QoS(1)
    );

    PointCloud::Ptr global_cloud = std::make_shared<PointCloud>();
    pcl::io::loadPCDFile("1897translated.pcd", *global_cloud);
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat costmap = cost_analysis_plane(global_cloud, {0, 0, 0});
    global_costmap_ = to_occupancy_grid_msg(costmap);
    auto end = std::chrono::high_resolution_clock::now();
    RCLCPP_INFO(get_logger(), "Global costmap analysis: %.2fms", (end - start).count() / 1e6);
    timer_ = create_timer(std::chrono::seconds(1), [&]() { timer_callback(); });
}

void TerrainAnalysisNode::cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto cloud = std::make_shared<PointCloud>();
    pcl::fromROSMsg(*msg, *cloud);
    cloud_queue_.push_front(cloud);
    if (cloud_queue_.size() < cloud_queue_size_) return;
    auto cloud_sum = std::make_shared<PointCloud>();
    std::for_each(cloud_queue_.begin(), cloud_queue_.end(), [&](auto& c) { *cloud_sum += *c; });
    cloud_queue_.pop_back();

    cv::Mat costmap = cost_analysis_plane(cloud_sum, {0, 0, 0});
    nav_msgs::msg::OccupancyGrid occupancy_grid = to_occupancy_grid_msg(costmap);
    // ...
}

void TerrainAnalysisNode::timer_callback() {
    global_costmap_pub_->publish(global_costmap_);
}

cv::Mat TerrainAnalysisNode::cost_analysis_plane(const PointCloud::Ptr cloud, const Eigen::Vector3f& car_to_odom) {
    using namespace small_gicp;
    auto cloud_down = voxelgrid_sampling_omp<PointCloud, PointNorCloud>(*cloud, voxel_leaf_size_, num_threads_);
    estimate_local_features<NormalSetter<PointNorCloud>>(*cloud_down, num_neighbors_);
    cv::Mat costmap = cv::Mat::zeros(cv::Size(map_y_size_, map_x_size_), CV_8U);
    const double inv_leaf_size = 1.0 / voxel_leaf_size_;
    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 64)
    for (int i = 0; i < cloud_down->size(); i++) {
        const auto point = cloud_down->points[i];
        if (std::hypot(point.x - car_to_odom.x(), point.y - car_to_odom.y()) > max_radius_) continue;
        if (point.z > car_to_odom.z() + max_relative_z_ || point.z < car_to_odom.z() + min_relative_z_) continue;
        int x = point.x * inv_leaf_size;
        int y = point.y * inv_leaf_size;
        if (x >= map_x_size_ || x < 0) continue;
        if (y >= map_y_size_ || y < 0) continue;
        costmap.at<int8_t>(x, y) = std::max(
            costmap.at<int8_t>(x, y),
            std::clamp<int8_t>((1 - point.normal_z) * 100, 0, 100)
        );
    }
    cv::morphologyEx(costmap, costmap, cv::MORPH_CLOSE, close_kernel_);
    cv::dilate(costmap, costmap, dilate_kernel_);
    cv::GaussianBlur(costmap, costmap, cv::Size(gaussian_blur_size_, gaussian_blur_size_), 0);
    return costmap;
}

nav_msgs::msg::OccupancyGrid TerrainAnalysisNode::to_occupancy_grid_msg(const cv::Mat& costmap) {
    nav_msgs::msg::OccupancyGrid occupancy_grid;
    occupancy_grid.header.frame_id = "map";
    occupancy_grid.header.stamp = now();
    occupancy_grid.info.resolution = voxel_leaf_size_;
    occupancy_grid.info.height = map_y_size_;
    occupancy_grid.info.width = map_x_size_;
    occupancy_grid.data.resize(map_x_size_ * map_y_size_);
    const cv::Mat transposed = costmap.t();
    std::copy(transposed.data, transposed.data + map_x_size_ * map_y_size_, occupancy_grid.data.data());
    return occupancy_grid;
}
} // namespace terrain_analysis

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(terrain_analysis::TerrainAnalysisNode)