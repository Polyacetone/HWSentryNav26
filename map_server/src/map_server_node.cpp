#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_ros/buffer.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <common_utils/convert.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling_omp.hpp>

namespace map_server {
class MapServerNode: public rclcpp::Node {
public:
    explicit MapServerNode(const rclcpp::NodeOptions& options);

private:
    double map_resolution_;
    int num_threads_;
    int map_size_x_, map_size_y_;
    bool enable_debug_;
    struct {
        int cloud_accumulate_frames;
        double roi_xy_radius_min;
        double roi_xy_radius_max;
        double roi_z_max;
        double roi_z_min;
        double downsample_voxel_size;
        double distance_threshold;
        int cell_obstacle_point_threshold;
        int dilate_kernel_size;
        int gaussian_blur_kernel_size;
        double gaussian_blur_sigma;
    } local_map_params_;

    cv::Mat global_direction_map_, global_cost_map_;
    small_gicp::PointCloud::Ptr global_point_cloud_;
    small_gicp::KdTree<small_gicp::PointCloud>::Ptr global_kdtree_;
    std::deque<small_gicp::PointCloud::Ptr> local_map_cloud_queue_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr direction_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr cost_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_dynamic_points_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr debug_local_cost_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_accumulated_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_global_cloud_pub_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    rclcpp::TimerBase::SharedPtr timer_;

    void local_map_cloud_callback(sensor_msgs::msg::PointCloud2::SharedPtr msg);
    small_gicp::PointCloud::Ptr preprocess_cloud(sensor_msgs::msg::PointCloud2::SharedPtr msg) const;
    small_gicp::PointCloud::Ptr convert_pcl_to_small_gicp(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud) const;
    cv::Mat dynamic_obstacle_analysis(const small_gicp::PointCloud& dynamic_points) const;
    void pub_direction_map(const cv::Mat& direction_map, const rclcpp::Time& stamp) const;
    void pub_cost_map(const cv::Mat& cost_map, const rclcpp::Time& stamp, rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher) const;
    void pub_cloud(const small_gicp::PointCloud& cloud, const rclcpp::Time& stamp, const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher) const;
};

MapServerNode::MapServerNode(const rclcpp::NodeOptions& options): Node("map_server", options) {
    // 参数加载
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    map_resolution_ = declare_parameter<double>("map_resolution");
    num_threads_ = declare_parameter<int>("num_threads");
    local_map_params_ = {
        .cloud_accumulate_frames = (int)declare_parameter<int>("local_map.cloud_accumulate_frames"),
        .roi_xy_radius_min = declare_parameter<double>("local_map.roi_xy_radius_min"),
        .roi_xy_radius_max = declare_parameter<double>("local_map.roi_xy_radius_max"),
        .roi_z_max = declare_parameter<double>("local_map.roi_z_max"),
        .roi_z_min = declare_parameter<double>("local_map.roi_z_min"),
        .downsample_voxel_size = declare_parameter<double>("local_map.downsample_voxel_size"),
        .distance_threshold = declare_parameter<double>("local_map.distance_threshold"),
        .cell_obstacle_point_threshold = (int)declare_parameter<int>("local_map.cell_obstacle_point_threshold"),
        .dilate_kernel_size = (int)declare_parameter<int>("local_map.dilate_kernel_size"),
        .gaussian_blur_kernel_size = (int)declare_parameter<int>("local_map.gaussian_blur_kernel_size"),
        .gaussian_blur_sigma = declare_parameter<double>("local_map.gaussian_blur_sigma")
    };
    enable_debug_ = declare_parameter<bool>("debug_mode.enable");
    if (enable_debug_) {
        std::string dynamic_points_pub_topic = declare_parameter<std::string>("debug_mode.dynamic_points_pub_topic");
        debug_dynamic_points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(dynamic_points_pub_topic, 1);
        std::string local_cost_map_pub_topic = declare_parameter<std::string>("debug_mode.local_cost_map_pub_topic");
        debug_local_cost_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(local_cost_map_pub_topic, 1);
        std::string accumulated_cloud_pub_topic = declare_parameter<std::string>("debug_mode.accumulated_cloud_pub_topic");
        debug_accumulated_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(accumulated_cloud_pub_topic, 1);
        std::string global_cloud_pub_topic = declare_parameter<std::string>("debug_mode.global_cloud_pub_topic");
        debug_global_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(global_cloud_pub_topic, 1);
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        RCLCPP_DEBUG(get_logger(), "Debug mode enabled");
    }

    // 加载全局地图
    std::string global_map_filename = declare_parameter<std::string>("global_map.map_filename");
    std::string global_map_path = ament_index_cpp::get_package_share_directory("map_server") + "/maps/" + global_map_filename;
    cv::Mat global_map = cv::imread(global_map_path, cv::IMREAD_COLOR);
    if (global_map.empty()) {
        RCLCPP_FATAL(get_logger(), "Failed to load global navmap from %s", global_map_path.c_str());
        throw std::runtime_error("Failed to load global navmap");
    }
    map_size_x_ = global_map.cols;
    map_size_y_ = global_map.rows;
    RCLCPP_INFO(get_logger(), "Loaded global navmap: size_x=%d, size_y=%d", map_size_x_, map_size_y_);
    std::vector<cv::Mat> channels;
    cv::split(global_map, channels);
    cv::merge(std::array{channels[0], channels[1]}, global_direction_map_); // 前两个通道表示台阶方向
    global_cost_map_ = channels[2]; // 第三个通道表示代价地图

    // 加载全局点云
    std::string global_cloud_filename = declare_parameter<std::string>("global_map.cloud_filename");
    std::string global_cloud_path = ament_index_cpp::get_package_share_directory("map_server") + "/maps/" + global_cloud_filename;
    auto global_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (pcl::io::loadPCDFile(global_cloud_path, *global_cloud) == -1) {
        RCLCPP_FATAL(get_logger(), "Failed to load global point cloud from %s", global_cloud_path.c_str());
        throw std::runtime_error("Failed to load global point cloud");
    }
    global_point_cloud_ = convert_pcl_to_small_gicp(global_cloud);
    RCLCPP_INFO(get_logger(), "Loaded global point cloud with %zu points", global_point_cloud_->size());
    double global_downsample_voxel_size = declare_parameter<double>("global_map.downsample_voxel_size");
    if (global_downsample_voxel_size > 0.0) {
        global_point_cloud_ = small_gicp::voxelgrid_sampling_omp(*global_point_cloud_, global_downsample_voxel_size, num_threads_);
    }
    global_kdtree_ = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(global_point_cloud_, small_gicp::KdTreeBuilderOMP(num_threads_));
    RCLCPP_INFO(get_logger(), "Downsampled global point cloud to %zu points", global_point_cloud_->size());

    // ROS相关
    std::string direction_map_pub_topic = declare_parameter<std::string>("direction_map_pub_topic");
    direction_map_pub_ = create_publisher<sensor_msgs::msg::Image>(direction_map_pub_topic, 1);
    std::string cost_map_pub_topic = declare_parameter<std::string>("cost_map_pub_topic");
    cost_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(cost_map_pub_topic, 1);
    std::string local_map_cloud_sub_topic = declare_parameter<std::string>("local_map.cloud_sub_topic");
    local_map_cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        local_map_cloud_sub_topic,
        rclcpp::QoS(1),
        [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { local_map_cloud_callback(msg); }
    );

    // 发布初始全局地图
    pub_cost_map(global_cost_map_, now(), cost_map_pub_);
    pub_direction_map(global_direction_map_, now());
}

void MapServerNode::local_map_cloud_callback(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 预处理点云，累积并下采样
    local_map_cloud_queue_.push_front(preprocess_cloud(msg));
    RCLCPP_DEBUG(get_logger(), "Inserted local cloud into queue with %zu points", local_map_cloud_queue_.front()->points.size());
    if (local_map_cloud_queue_.size() > local_map_params_.cloud_accumulate_frames) {
        local_map_cloud_queue_.pop_back();
    }
    small_gicp::PointCloud accumulated;
    for (const auto& cloud : local_map_cloud_queue_) {
        accumulated.points.insert(accumulated.points.end(), cloud->points.begin(), cloud->points.end());
    }
    RCLCPP_DEBUG(get_logger(), "Accumulated local cloud has %zu points", accumulated.points.size());
    if (local_map_params_.downsample_voxel_size > 0.0) {
        accumulated = *small_gicp::voxelgrid_sampling_omp(
            accumulated,
            local_map_params_.downsample_voxel_size,
            num_threads_
        );
        RCLCPP_DEBUG(get_logger(), "Downsampled local cloud has %zu points", accumulated.points.size());
    }

    // 局部点云减去全局点云，剩余的点认为是动态障碍物
    small_gicp::PointCloud dynamic_points;
    for (const auto & point : accumulated.points) {
        size_t index;
        double dist;
        global_kdtree_->knn_search(point, 1, &index, &dist);
        if (dist > local_map_params_.distance_threshold * local_map_params_.distance_threshold) {
            dynamic_points.points.push_back(point);
        }
    }
    RCLCPP_DEBUG(get_logger(), "Identified %zu dynamic obstacle points", dynamic_points.points.size());

    // 动态障碍物分析，生成代价地图
    cv::Mat local_cost_map = dynamic_obstacle_analysis(dynamic_points);
    cv::Mat cost_map = cv::max(global_cost_map_, local_cost_map);

    // 调试信息发布
    if (enable_debug_) {
        pub_cost_map(local_cost_map, msg->header.stamp, debug_local_cost_map_pub_);
        pub_cloud(dynamic_points, msg->header.stamp, debug_dynamic_points_pub_);
        pub_cloud(accumulated, msg->header.stamp, debug_accumulated_cloud_pub_);
        pub_cloud(*global_point_cloud_, msg->header.stamp, debug_global_cloud_pub_);
    }

    pub_cost_map(cost_map, msg->header.stamp, cost_map_pub_);
    pub_direction_map(global_direction_map_, msg->header.stamp);
}

small_gicp::PointCloud::Ptr MapServerNode::preprocess_cloud(sensor_msgs::msg::PointCloud2::SharedPtr msg) const {
    auto preprocessed = std::make_shared<small_gicp::PointCloud>();

    // 查找x, y, z字段的偏移量
    int offset_x = -1, offset_y = -1, offset_z = -1;
    for (const auto& field : msg->fields) {
        if (field.name == "x") offset_x = field.offset;
        else if (field.name == "y") offset_y = field.offset;
        else if (field.name == "z") offset_z = field.offset;
    }
    if (offset_x < 0 || offset_y < 0 || offset_z < 0) {
        RCLCPP_WARN(get_logger(), "PointCloud2 missing x/y/z fields");
        return preprocessed;
    }

    // 获取点云到地图的变换
    Eigen::Isometry3d cloud_to_map;
    try {
        cloud_to_map = utils::convert_to<Eigen::Isometry3d>(
            tf_buffer_->lookupTransform("map", msg->header.frame_id, tf2::TimePointZero).transform
        );
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(), "Failed to lookup %s to map: %s", msg->header.frame_id.c_str(), ex.what());
        return preprocessed;
    }

    // 获取激光雷达到地图的位移
    Eigen::Vector3d lidar_to_map;
    try {
        lidar_to_map = utils::convert_to<Eigen::Isometry3d>(
            tf_buffer_->lookupTransform("map", "lidar", tf2::TimePointZero).transform
        ).translation().head<3>();
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(), "Failed to lookup lidar to map: %s", ex.what());
        return preprocessed;
    }

    // 遍历点云，过滤并转换点
    const size_t point_step = msg->point_step;
    const size_t num_points = msg->width * msg->height;
    const uint8_t* data_ptr = msg->data.data();
    for (size_t i = 0; i < num_points; i++) {
        const Eigen::Vector3d pt = cloud_to_map * Eigen::Vector3d(
            *reinterpret_cast<const float*>(data_ptr + offset_x),
            *reinterpret_cast<const float*>(data_ptr + offset_y),
            *reinterpret_cast<const float*>(data_ptr + offset_z)
        );
        const Eigen::Vector3d pt_lidar = pt - lidar_to_map;
        const double distance_xy = std::hypot(pt_lidar.x(), pt_lidar.y());
        if (local_map_params_.roi_xy_radius_min < distance_xy && distance_xy < local_map_params_.roi_xy_radius_max &&
            local_map_params_.roi_z_min < pt_lidar.z() && pt_lidar.z() < local_map_params_.roi_z_max) {
            preprocessed->points.emplace_back(pt.x(), pt.y(), pt.z(), 1.0);
        }
        data_ptr += point_step;
    }
    return preprocessed;
}

cv::Mat MapServerNode::dynamic_obstacle_analysis(const small_gicp::PointCloud& dynamic_points) const {
    cv::Mat dynamic_points_count = cv::Mat::zeros(map_size_y_, map_size_x_, CV_8UC1);
    cv::Mat local_cost_map = cv::Mat::zeros(map_size_y_, map_size_x_, CV_8UC1);
    for (const auto& pt : dynamic_points.points) {
        const int map_x = pt.x() / map_resolution_;
        const int map_y = pt.y() / map_resolution_;
        if (map_x < 0 || map_x >= map_size_x_ || map_y < 0 || map_y >= map_size_y_) continue;
        uint8_t& cell = dynamic_points_count.at<uint8_t>(map_y, map_x);
        cell = (cell < 255) ? (cell + 1) : 255;
        if (cell >= local_map_params_.cell_obstacle_point_threshold) {
            local_cost_map.at<uint8_t>(map_y, map_x) = 255; // 标记为动态障碍物
        }
    }
    cv::dilate(local_cost_map, local_cost_map, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(local_map_params_.dilate_kernel_size, local_map_params_.dilate_kernel_size)));
    cv::GaussianBlur(local_cost_map, local_cost_map, cv::Size(local_map_params_.gaussian_blur_kernel_size, local_map_params_.gaussian_blur_kernel_size), local_map_params_.gaussian_blur_sigma);
    return local_cost_map;
}

small_gicp::PointCloud::Ptr MapServerNode::convert_pcl_to_small_gicp(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud) const {
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

void MapServerNode::pub_direction_map(const cv::Mat& direction_map, const rclcpp::Time& stamp) const {
    sensor_msgs::msg::Image::SharedPtr direction_map_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "8UC2", direction_map).toImageMsg();
    direction_map_msg->header.stamp = stamp;
    direction_map_msg->header.frame_id = "map";
    direction_map_pub_->publish(*direction_map_msg);
}

void MapServerNode::pub_cost_map(
    const cv::Mat& cost_map,
    const rclcpp::Time& stamp,
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher
) const {
    nav_msgs::msg::OccupancyGrid occupancy_grid;
    occupancy_grid.header.frame_id = "map";
    occupancy_grid.header.stamp = stamp;
    occupancy_grid.info.resolution = map_resolution_;
    occupancy_grid.info.height = map_size_y_;
    occupancy_grid.info.width = map_size_x_;
    occupancy_grid.data.resize(map_size_x_ * map_size_y_);
    std::copy(cost_map.data, cost_map.data + map_size_x_ * map_size_y_, occupancy_grid.data.data());
    publisher->publish(occupancy_grid);
}

void MapServerNode::pub_cloud(
    const small_gicp::PointCloud& cloud,
    const rclcpp::Time& stamp,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher
) const {
    const size_t num_points = cloud.size();
    const auto& points = cloud.points;
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.frame_id = "map";
    msg.header.stamp = stamp;
    msg.height = 1;
    msg.width = num_points;
    msg.is_dense = true;
    msg.point_step = 12;
    msg.row_step = 12 * num_points;
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
    for (int i = 0; i < num_points; i++) {
        Eigen::Vector3f pt = points[i](Eigen::seq(0, 2)).cast<float>();
        std::memcpy(msg.data.data() + i * 12, pt.data(), sizeof(pt));
    }
    publisher->publish(msg);
}
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(map_server::MapServerNode)