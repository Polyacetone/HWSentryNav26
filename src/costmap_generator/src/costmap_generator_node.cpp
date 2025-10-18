#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_ros/buffer.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <common_utils/convert.hpp>

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
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void transform_cloud(PointCloud& cloud, const std::string& cloud_frame, const std::string& target_frame) const;
    void select_cloud(PointCloud& cloud, const double max_z, const double min_z) const;
    void select_cloud(
        PointCloud& cloud, const double max_relative_z, const double min_relative_z, const double radius,
        const std::string& center_frame, const std::string& cloud_frame
    ) const;
    cv::Mat cost_analysis_normal(const PointCloud& cloud) const;
    nav_msgs::msg::OccupancyGrid to_occupancy_grid_msg(
        const cv::Mat& costmap,
        const rclcpp::Time& stamp
    ) const;
};

CostmapGeneratorNode::CostmapGeneratorNode(const rclcpp::NodeOptions& options): Node("costmap_generator_node", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
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
    const std::string file_path = declare_parameter<std::string>("global_costmap.file_path");

    if (file_path.ends_with("pcd")) {
        PointCloud global_cloud;
        pcl::io::loadPCDFile(file_path, global_cloud);
        const double max_z = declare_parameter<double>("global_costmap.file_type_pcd.max_z");
        const double min_z = declare_parameter<double>("global_costmap.file_type_pcd.min_z");
        select_cloud(global_cloud, max_z, min_z);
        global_costmap_ = cost_analysis_normal(global_cloud);
    } else {
        global_costmap_ = cv::Mat::zeros(cv::Size(map_x_size_, map_y_size_), CV_8U);
        cv::Mat img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
        cv::flip(img, img, 0);
        cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
        #pragma omp parallel for num_threads(num_threads_) schedule(guided, 16)
        for (int row = 0; row < map_y_size_; row++) {
            for (int col = 0; col < map_x_size_; col++) {
                if (col >= img.cols || row >= img.rows) global_costmap_.at<uint8_t>(row, col) = 100;
                else global_costmap_.at<uint8_t>(row, col) = std::clamp<uint8_t>(
                    img.at<uint8_t>(row, col),
                    0, 100
                );
            }
        }
    }

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_sub_topic, rclcpp::QoS(1),
        [&](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { cloud_callback(msg); }
    );
    costmap_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
        pub_topic,
        rclcpp::QoS(1)
    );
}

void CostmapGeneratorNode::cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    const auto cloud = std::make_shared<PointCloud>();
    pcl::fromROSMsg(*msg, *cloud);
    transform_cloud(*cloud, "odom", "map");
    select_cloud(*cloud, max_relative_z_, min_relative_z_, max_radius_, "base", "map");
    cloud_queue_.push_front(cloud);
    if (cloud_queue_.size() < cloud_queue_size_) return;
    PointCloud cloud_sum;
    std::for_each(cloud_queue_.begin(), cloud_queue_.end(), [&](const auto& c) { cloud_sum += *c; });
    cloud_queue_.pop_back();

    cv::Mat costmap = cost_analysis_normal(cloud_sum);
    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 16)
    for (int row = 0; row < map_y_size_; row++) {
        for (int col = 0; col < map_x_size_; col++) {
            costmap.at<uint8_t>(row, col) = std::max(
                costmap.at<uint8_t>(row, col),
                global_costmap_.at<uint8_t>(row, col)
            );
        }
    }
    cv::dilate(costmap, costmap, dilate_kernel_);
    cv::GaussianBlur(costmap, costmap, gaussian_blur_size_, gaussian_blur_sigma_);
    costmap_pub_->publish(to_occupancy_grid_msg(costmap, msg->header.stamp));
}

void CostmapGeneratorNode::transform_cloud(
    PointCloud& cloud,
    const std::string& cloud_frame,
    const std::string& target_frame
) const {
    tf2::Transform cloud_to_target;
    try {
        cloud_to_target = utils::convert_to<tf2::Transform>(
            tf_buffer_->lookupTransform(target_frame, cloud_frame, tf2::TimePointZero).transform
        );
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(), "Failed to lookup %s to %s: %s", cloud_frame.c_str(), target_frame.c_str(), ex.what());
        cloud = PointCloud();
        return;
    }
    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 64)
    for (int i = 0; i < cloud.size(); i++) {
        auto& point = cloud.points[i];
        const tf2::Vector3 transformed = cloud_to_target * tf2::Vector3(point.x, point.y, point.z);
        point.x = transformed.x(), point.y = transformed.y(), point.z = transformed.z();
    }
}

void CostmapGeneratorNode::select_cloud(
    PointCloud& cloud,
    const double max_z,
    const double min_z
) const {
    PointCloud cropped;
    for (int i = 0; i < cloud.size(); i++) {
        const auto& point = cloud.points[i];
        if (point.z > max_z || point.z < min_z) continue;
        cropped.points.emplace_back(point);
    }
    cloud = cropped;
}

void CostmapGeneratorNode::select_cloud(
    PointCloud& cloud, const double max_relative_z, const double min_relative_z, const double radius,
    const std::string& center_frame, const std::string& cloud_frame
) const {
    tf2::Transform center_to_cloud;
    try {
        center_to_cloud = utils::convert_to<tf2::Transform>(
            tf_buffer_->lookupTransform(cloud_frame, center_frame, tf2::TimePointZero).transform
        );
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(), "Failed to lookup %s to %s: %s", center_frame.c_str(), cloud_frame.c_str(), ex.what());
        cloud = PointCloud();
        return;
    }
    const Eigen::Vector3d center = utils::convert_to<Eigen::Vector3d>(center_to_cloud.getOrigin());
    PointCloud cropped;
    for (int i = 0; i < cloud.size(); i++) {
        const auto& point = cloud.points[i];
        const Eigen::Vector3d point_to_center = utils::convert_to<Eigen::Vector3d>(point) - center;
        if (point_to_center.z() > max_relative_z || point_to_center.z() < min_relative_z) continue;
        if (point_to_center(Eigen::seq(0, 1)).norm() > radius) continue;
        cropped.points.emplace_back(point);
    }
    cloud = cropped;
}

cv::Mat CostmapGeneratorNode::cost_analysis_normal(const PointCloud& cloud) const {
    using namespace small_gicp;
    auto cloud_down = voxelgrid_sampling_omp<PointCloud, PointNorCloud>(cloud, map_resolution_, num_threads_);
    estimate_local_features<NormalSetter<PointNorCloud>>(*cloud_down, num_neighbors_);
    cv::Mat costmap = cv::Mat::zeros(cv::Size(map_x_size_, map_y_size_), CV_8U);
    const double inv_leaf_size = 1.0 / map_resolution_;
    for (int i = 0; i < cloud_down->size(); i++) {
        const auto& point = cloud_down->points[i];
        const int x = point.x * inv_leaf_size;
        const int y = point.y * inv_leaf_size;
        if (x >= map_x_size_ || x < 0) continue;
        if (y >= map_y_size_ || y < 0) continue;
        costmap.at<uint8_t>(y, x) = std::max(
            costmap.at<uint8_t>(y, x),
            std::clamp<uint8_t>((1 - point.normal_z) * 100, 0, 100)
        );
    }
    return costmap;
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