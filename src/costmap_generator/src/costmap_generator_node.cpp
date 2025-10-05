#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <common_utils/tf_utils.hpp>

namespace costmap_generator {
using Point = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<Point>;
using PointCov = pcl::PointCovariance;
using PointCovCloud = pcl::PointCloud<PointCov>;
using PointNor = pcl::PointNormal;
using PointNorCloud = pcl::PointCloud<PointNor>;

class CostmapGeneratorNode: public rclcpp::Node {
public:
    explicit CostmapGeneratorNode(const rclcpp::NodeOptions& options);

private:
    unsigned num_threads_, num_neighbors_, cloud_queue_size_;
    unsigned map_x_size_, map_y_size_;
    double map_resolution_, max_radius_, max_relative_z_, min_relative_z_, gaussian_blur_sigma_;

    std::deque<PointCloud::Ptr> cloud_queue_;
    cv::Mat global_costmap_, dilate_kernel_;
    cv::Size gaussian_blur_size_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_pub_;
    
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    cv::Mat cost_analysis(
        const PointCloud::Ptr cloud,
        const Eigen::Vector3f& base_to_odom,
        const double max_relative_z,
        const double min_relative_z,
        const double max_radius = std::numeric_limits<double>::max()
    ) const;
    void postprocess_costmap(cv::Mat& costmap) const;
    nav_msgs::msg::OccupancyGrid to_occupancy_grid_msg(
        const cv::Mat& costmap,
        const rclcpp::Time& stamp
    ) const;
};

CostmapGeneratorNode::CostmapGeneratorNode(const rclcpp::NodeOptions& options): Node("costmap_generator_node", options) {
    const std::string pub_topic = declare_parameter<std::string>("pub_topic");
    num_threads_ = declare_parameter<int>("num_threads");
    num_neighbors_ = declare_parameter<int>("num_neighbors");
    map_resolution_ = declare_parameter<double>("map_resolution");
    map_x_size_ = declare_parameter<int>("map_x_size");
    map_y_size_ = declare_parameter<int>("map_y_size");
    const int dilate_kernel_size = declare_parameter<int>("dilate_kernel_size");
    dilate_kernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, {dilate_kernel_size, dilate_kernel_size});
    const int gaussian_blur_size = declare_parameter<int>("gaussian_blur_size");
    gaussian_blur_size_ = cv::Size(gaussian_blur_size, gaussian_blur_size);
    gaussian_blur_sigma_ = declare_parameter<double>("gaussian_blur_sigma");
    const std::string cloud_sub_topic = declare_parameter<std::string>("local_costmap.cloud_sub_topic");
    cloud_queue_size_ = declare_parameter<int>("local_costmap.cloud_queue_size");
    max_radius_ = declare_parameter<double>("local_costmap.max_radius");
    max_relative_z_ = declare_parameter<double>("local_costmap.max_relative_z");
    min_relative_z_ = declare_parameter<double>("local_costmap.min_relative_z");
    const std::string filepath = declare_parameter<std::string>("global_costmap.filepath");
    
    if (filepath.ends_with("pcd")) {
        PointCloud::Ptr global_cloud = std::make_shared<PointCloud>();
        pcl::io::loadPCDFile(filepath, *global_cloud);
        const double max_z = declare_parameter<double>("global_costmap.file_type_pcd.max_z");
        const double min_z = declare_parameter<double>("global_costmap.file_type_pcd.min_z");
        global_costmap_ = cost_analysis(global_cloud, {0, 0, 0}, max_z, min_z);
    } else {
        global_costmap_ = cv::Mat::zeros(cv::Size(map_x_size_, map_y_size_), CV_8U);
        cv::Mat img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
        cv::flip(img, img, 0);
        cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
        #pragma omp parallel for num_threads(num_threads_) schedule(guided, 16)
        for (int row = 0; row < map_y_size_; row++) {
            for (int col = 0; col < map_x_size_; col++) {
                if (col >= img.cols || row >= img.rows) global_costmap_.at<int8_t>(row, col) = 100;
                else global_costmap_.at<int8_t>(row, col) = std::clamp<uint8_t>(
                    img.at<uint8_t>(row, col),
                    0, 100
                );
            }
        }
        postprocess_costmap(global_costmap_);
    }

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_sub_topic, rclcpp::QoS(1),
        [&](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { cloud_callback(msg); }
    );
    costmap_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
        pub_topic,
        rclcpp::QoS(1)
    );

    while (rclcpp::ok()) {
        costmap_pub_->publish(to_occupancy_grid_msg(global_costmap_, now()));
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void CostmapGeneratorNode::cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto cloud = std::make_shared<PointCloud>();
    pcl::fromROSMsg(*msg, *cloud);
    cloud_queue_.push_front(cloud);
    if (cloud_queue_.size() < cloud_queue_size_) return;
    auto cloud_sum = std::make_shared<PointCloud>();
    std::for_each(cloud_queue_.begin(), cloud_queue_.end(), [&](auto& c) { *cloud_sum += *c; });
    cloud_queue_.pop_back();

    // ...
}

cv::Mat CostmapGeneratorNode::cost_analysis(
    const PointCloud::Ptr cloud,
    const Eigen::Vector3f& base_to_odom,
    const double max_relative_z,
    const double min_relative_z,
    const double max_radius
) const {
    using namespace small_gicp;
    auto cloud_down = voxelgrid_sampling_omp<PointCloud, PointNorCloud>(*cloud, map_resolution_, num_threads_);
    estimate_local_features<NormalSetter<PointNorCloud>>(*cloud_down, num_neighbors_);
    cv::Mat costmap = cv::Mat::zeros(cv::Size(map_x_size_, map_y_size_), CV_8U);
    const double inv_leaf_size = 1.0 / map_resolution_;
    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 64)
    for (int i = 0; i < cloud_down->size(); i++) {
        const auto point = cloud_down->points[i];
        if (std::hypot(point.x - base_to_odom.x(), point.y - base_to_odom.y()) > max_radius) continue;
        if (point.z > base_to_odom.z() + max_relative_z || point.z < base_to_odom.z() + min_relative_z) continue;
        int x = point.x * inv_leaf_size;
        int y = point.y * inv_leaf_size;
        if (x >= map_x_size_ || x < 0) continue;
        if (y >= map_y_size_ || y < 0) continue;
        costmap.at<uint8_t>(y, x) = std::max(
            costmap.at<uint8_t>(y, x),
            std::clamp<uint8_t>((1 - point.normal_z) * 100, 0, 100)
        );
    }
    postprocess_costmap(costmap);
    return costmap;
}

void CostmapGeneratorNode::postprocess_costmap(cv::Mat& costmap) const {
    cv::dilate(costmap, costmap, dilate_kernel_);
    cv::GaussianBlur(costmap, costmap, gaussian_blur_size_, gaussian_blur_sigma_);
}

nav_msgs::msg::OccupancyGrid CostmapGeneratorNode::to_occupancy_grid_msg(
    const cv::Mat& costmap,
    const rclcpp::Time& stamp
) const {
    nav_msgs::msg::OccupancyGrid occupancy_grid;
    occupancy_grid.header.frame_id = "map";
    occupancy_grid.header.stamp = stamp;
    occupancy_grid.info.resolution = map_resolution_;
    occupancy_grid.info.height = map_y_size_;
    occupancy_grid.info.width = map_x_size_;
    occupancy_grid.data.resize(map_x_size_ * map_y_size_);
    std::copy(costmap.data, costmap.data + map_x_size_ * map_y_size_, occupancy_grid.data.data());
    return occupancy_grid;
}
} // namespace costmap_generator

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(costmap_generator::CostmapGeneratorNode)