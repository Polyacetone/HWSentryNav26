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
#include <interfaces/msg/robot_status.hpp>
#include <interfaces/msg/cost_maps.hpp>
#include <common_utils/convert.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <map_server/object_tracker.hpp>
#include <map_server/utils.hpp>

namespace map_server {

class MapServerNode: public rclcpp::Node {
public:
    explicit MapServerNode(const rclcpp::NodeOptions& options);

private:
    double map_resolution_;
    double robot_color_wait_timeout_;
    int num_threads_;
    int map_size_x_, map_size_y_;
    bool enable_debug_;
    struct {
        size_t cloud_accumulate_frames;
        double roi_xy_radius_min;
        double roi_xy_radius_max;
        double roi_z_max;
        double roi_z_min;
        double downsample_voxel_size;
        struct {
            double distance_threshold;
        } with_global_cloud;
        struct {
            int normal_num_neighbors;
            double vertical_normal_abs_z_max;
            double direction_filter_threshold;
        } without_global_cloud;
        int sor_num_neighbors;
        double sor_std_mul;
        int cell_obstacle_point_threshold;
    } local_map_params_;
    bool bypass_dynamic_obstacle_;
    bool enable_prediction_with_cloud_;
    bool enable_prediction_without_cloud_;

    map_utils::MapInflationParams map_inflation_params_{};
    cv::Mat global_direction_map_, global_cost_map_;
    bool global_nav_map_initialized_ = false;
    std::chrono::steady_clock::time_point initialize_time_;
    small_gicp::PointCloud::Ptr global_point_cloud_;
    small_gicp::KdTree<small_gicp::PointCloud>::Ptr global_kdtree_;
    std::deque<small_gicp::PointCloud::Ptr> local_map_cloud_queue_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr local_cloud_sub_;
    rclcpp::Subscription<interfaces::msg::RobotStatus>::SharedPtr robot_status_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr global_direction_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_pub_;
    rclcpp::Publisher<interfaces::msg::CostMaps>::SharedPtr local_cost_maps_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_dynamic_points_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_accumulated_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_global_cloud_pub_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    rclcpp::TimerBase::SharedPtr timer_;

    // 目标跟踪预测
    ObjectTrackerParams tracker_params_;
    std::unique_ptr<ObjectTracker> object_tracker_;
    std::chrono::steady_clock::time_point last_tracker_update_time_;

    void timer_callback();
    void local_cloud_callback(sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void robot_status_callback(const interfaces::msg::RobotStatus::SharedPtr msg);
    small_gicp::PointCloud::Ptr preprocess_cloud(sensor_msgs::msg::PointCloud2::SharedPtr msg) const;
    small_gicp::PointCloud extract_dynamic_points_with_global_map(const small_gicp::PointCloud& cloud) const;
    small_gicp::PointCloud extract_dynamic_points_without_global_map(const small_gicp::PointCloud& cloud) const;
    small_gicp::PointCloud remove_statistical_outliers(const small_gicp::PointCloud& cloud) const;
    small_gicp::PointCloud::Ptr convert_pcl_to_small_gicp(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud) const;
    cv::Mat create_obstacle_mask(const small_gicp::PointCloud& dynamic_points) const;
    cv::Mat dynamic_obstacle_analysis(const small_gicp::PointCloud& dynamic_points) const;
    void fill_occupancy_grid(const cv::Mat& cost_map, const rclcpp::Time& stamp, nav_msgs::msg::OccupancyGrid& grid) const;
    void pub_direction_map(const cv::Mat& direction_map, const rclcpp::Time& stamp, const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher) const;
    void pub_cost_map(const cv::Mat& cost_map, const rclcpp::Time& stamp, const rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher) const;
    void pub_cloud(const small_gicp::PointCloud& cloud, const rclcpp::Time& stamp, const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher) const;
};

MapServerNode::MapServerNode(const rclcpp::NodeOptions& options): Node("map_server", options) {
    // 参数加载
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    map_resolution_ = declare_parameter<double>("map_resolution");
    num_threads_ = static_cast<int>(declare_parameter<int>("num_threads"));
    robot_color_wait_timeout_ = declare_parameter<double>("global_map.robot_color_wait_timeout");
    map_inflation_params_ = {
        .robot_radius_px = static_cast<int>(declare_parameter<int>("map_inflation.robot_radius_px")),
        .cutoff_radius_px = static_cast<int>(declare_parameter<int>("map_inflation.cutoff_radius_px")),
        .decay_alpha = declare_parameter<double>("map_inflation.decay_alpha")
    };

    local_map_params_ = {
        .cloud_accumulate_frames = (size_t)declare_parameter<int>("local_map.cloud_accumulate_frames"),
        .roi_xy_radius_min = declare_parameter<double>("local_map.roi_xy_radius_min"),
        .roi_xy_radius_max = declare_parameter<double>("local_map.roi_xy_radius_max"),
        .roi_z_max = declare_parameter<double>("local_map.roi_z_max"),
        .roi_z_min = declare_parameter<double>("local_map.roi_z_min"),
        .downsample_voxel_size = declare_parameter<double>("local_map.downsample_voxel_size"),
        .with_global_cloud = {
            .distance_threshold = declare_parameter<double>("local_map.with_global_cloud.distance_threshold")
        },
        .without_global_cloud = {
            .normal_num_neighbors = (int)declare_parameter<int>("local_map.without_global_cloud.normal_num_neighbors"),
            .vertical_normal_abs_z_max = declare_parameter<double>("local_map.without_global_cloud.vertical_normal_abs_z_max"),
            .direction_filter_threshold = declare_parameter<double>("local_map.without_global_cloud.direction_filter_threshold")
        },
        .sor_num_neighbors = (int)declare_parameter<int>("local_map.sor_num_neighbors"),
        .sor_std_mul = declare_parameter<double>("local_map.sor_std_mul"),
        .cell_obstacle_point_threshold = (int)declare_parameter<int>("local_map.cell_obstacle_point_threshold")
    };

    enable_debug_ = declare_parameter<bool>("debug.enable");
    if (enable_debug_) {
        std::string dynamic_points_pub_topic = declare_parameter<std::string>("debug.dynamic_points_pub_topic");
        debug_dynamic_points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(dynamic_points_pub_topic, 1);
        std::string accumulated_cloud_pub_topic = declare_parameter<std::string>("debug.accumulated_cloud_pub_topic");
        debug_accumulated_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(accumulated_cloud_pub_topic, 1);
        std::string global_cloud_pub_topic = declare_parameter<std::string>("debug.global_cloud_pub_topic");
        debug_global_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(global_cloud_pub_topic, 1);
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        RCLCPP_DEBUG(get_logger(), "Debug mode enabled");
    }

    // 目标跟踪预测参数
    tracker_params_ = {
        .morph_close_kernel_size = (int)declare_parameter<int>("local_map.object_tracker.morph_close_kernel_size"),
        .min_blob_area = (int)declare_parameter<int>("local_map.object_tracker.min_blob_area"),
        .local_grid_size = (int)declare_parameter<int>("local_map.object_tracker.local_grid_size"),
        .max_association_dist = declare_parameter<double>("local_map.object_tracker.max_association_dist"),
        .process_noise_std = declare_parameter<double>("local_map.object_tracker.process_noise_std"),
        .measurement_noise_std = declare_parameter<double>("local_map.object_tracker.measurement_noise_std"),
        .max_lost_frames = (int)declare_parameter<int>("local_map.object_tracker.max_lost_frames"),
        .min_hits_to_confirm = (int)declare_parameter<int>("local_map.object_tracker.min_hits_to_confirm"),
        .local_grid_decay = declare_parameter<double>("local_map.object_tracker.local_grid_decay"),
        .local_grid_render_threshold = declare_parameter<double>("local_map.object_tracker.local_grid_render_threshold"),
        .prediction_steps = (int)declare_parameter<int>("local_map.object_tracker.prediction_steps"),
        .prediction_dt = declare_parameter<double>("local_map.object_tracker.prediction_dt"),
        .num_threads = num_threads_
    };
    bypass_dynamic_obstacle_ = declare_parameter<bool>("local_map.bypass_dynamic_obstacle");
    enable_prediction_with_cloud_ = declare_parameter<bool>("local_map.with_global_cloud.enable_prediction");
    enable_prediction_without_cloud_ = declare_parameter<bool>("local_map.without_global_cloud.enable_prediction");
    std::string local_cost_maps_pub_topic = declare_parameter<std::string>("local_map.local_cost_maps_pub_topic");
    local_cost_maps_pub_ = create_publisher<interfaces::msg::CostMaps>(local_cost_maps_pub_topic, 1);

    // 加载全局点云
    std::string global_cloud_filename = declare_parameter<std::string>("global_map.cloud_filename");
    if (!global_cloud_filename.empty()) {
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
    } else {
        RCLCPP_WARN(get_logger(), "No global point cloud specified, enabling fallback dynamic obstacle detection from local cloud only");
    }

    // ROS相关
    std::string global_direction_map_pub_topic = declare_parameter<std::string>("global_map.direction_map_pub_topic");
    global_direction_map_pub_ = create_publisher<sensor_msgs::msg::Image>(global_direction_map_pub_topic, 1);
    std::string global_cost_map_pub_topic = declare_parameter<std::string>("global_map.cost_map_pub_topic");
    global_cost_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(global_cost_map_pub_topic, 1);
    double global_map_pub_freq = declare_parameter<double>("global_map.pub_freq");
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / global_map_pub_freq), [this] { timer_callback(); });
    std::string local_cloud_sub_topic = declare_parameter<std::string>("local_map.cloud_sub_topic");
    local_cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        local_cloud_sub_topic, 1,
        [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { local_cloud_callback(msg); }
    );
    std::string robot_status_sub_topic = declare_parameter<std::string>("global_map.robot_status_sub_topic");
    robot_status_sub_ = create_subscription<interfaces::msg::RobotStatus>(
        robot_status_sub_topic, 1,
        [this](interfaces::msg::RobotStatus::SharedPtr msg) { robot_status_callback(msg); }
    );

    initialize_time_ = std::chrono::steady_clock::now();
}

void MapServerNode::robot_status_callback(const interfaces::msg::RobotStatus::SharedPtr msg) {
    if (global_nav_map_initialized_) {
        robot_status_sub_.reset();
        return;
    }
    global_nav_map_initialized_ = true;

    std::string nav_map_filename;
    if (msg->robot_color) {
        RCLCPP_DEBUG(get_logger(), "Robot color: RED");
        nav_map_filename = declare_parameter<std::string>("global_map.red_nav_map_filename");
    } else {
        RCLCPP_DEBUG(get_logger(), "Robot color: BLUE");
        nav_map_filename = declare_parameter<std::string>("global_map.blue_nav_map_filename");
    }

    std::string nav_map_path = ament_index_cpp::get_package_share_directory("map_server") + "/maps/" + nav_map_filename;
    RCLCPP_INFO(get_logger(), "Loading terrain msgpack from %s", nav_map_path.c_str());

    try {
        auto terrain_data = map_utils::load_terrain_msgpack(nav_map_path);
        map_resolution_ = terrain_data.resolution;
        map_size_x_ = terrain_data.width;
        map_size_y_ = terrain_data.height;

        // 障碍物膨胀 → global cost map
        {
            cv::Mat obs_mask = cv::Mat::zeros(map_size_y_, map_size_x_, CV_8UC1);
            for (int y = 0; y < map_size_y_; y++) {
                for (int x = 0; x < map_size_x_; x++) {
                    if (terrain_data.terrain[static_cast<size_t>(y) * static_cast<size_t>(map_size_x_) + static_cast<size_t>(x)] == static_cast<uint8_t>(TerrainType::OBSTACLE)) {
                        obs_mask.at<uint8_t>(y, x) = 255;
                    }
                }
            }
            global_cost_map_ = map_utils::inflate_cost_map(obs_mask, map_inflation_params_);
        }

        // 方向场膨胀 → angle + magnitude
        cv::Mat angle, magnitude, terrain_labels;
        map_utils::inflate_direction_field(terrain_data, map_inflation_params_, angle, magnitude);

        // raw label 直通
        terrain_labels = cv::Mat(map_size_y_, map_size_x_, CV_8UC1, terrain_data.terrain.data()).clone();

        // 合并为 3 通道: ch0=angle, ch1=magnitude, ch2=label
        map_utils::build_terrain_3chan(angle, magnitude, terrain_labels, global_direction_map_);

        RCLCPP_INFO(
            get_logger(), "Loaded terrain msgpack (%s) size=%dx%d resolution=%.3f, "
            "cost_map=%s, direction_map=%s",
            msg->robot_color ? "RED" : "BLUE",
            map_size_x_, map_size_y_, terrain_data.resolution,
            global_cost_map_.empty() ? "empty" : "ok",
            global_direction_map_.empty() ? "empty" : "ok"
        );
    } catch (const std::exception& e) {
        RCLCPP_FATAL(get_logger(), "Failed to load/process navmap: %s", e.what());
        std::exit(EXIT_FAILURE);
    }
}

void MapServerNode::timer_callback() {
    if (!global_nav_map_initialized_) {
        const auto now_time = std::chrono::steady_clock::now();
        // 超时未收到机器人颜色，使用默认全局导航地图
        if (std::chrono::duration<double>(now_time - initialize_time_).count() > robot_color_wait_timeout_) {
            RCLCPP_ERROR(get_logger(), "Robot color not initialized within timeout %.2f seconds, loading default global navmap", robot_color_wait_timeout_);
            global_nav_map_initialized_ = true;
            std::string default_nav_map_filename = declare_parameter<std::string>("global_map.default_nav_map_filename");
            std::string default_nav_map_path = ament_index_cpp::get_package_share_directory("map_server") + "/maps/" + default_nav_map_filename;
            try {
                auto terrain_data = map_utils::load_terrain_msgpack(default_nav_map_path);
                map_resolution_ = terrain_data.resolution;
                map_size_x_ = terrain_data.width;
                map_size_y_ = terrain_data.height;
                {
                    cv::Mat obs_mask = cv::Mat::zeros(map_size_y_, map_size_x_, CV_8UC1);
                    for (int y = 0; y < map_size_y_; y++) {
                        for (int x = 0; x < map_size_x_; x++) {
                            if (terrain_data.terrain[static_cast<size_t>(y) * static_cast<size_t>(map_size_x_) + static_cast<size_t>(x)] == static_cast<uint8_t>(TerrainType::OBSTACLE)) {
                                obs_mask.at<uint8_t>(y, x) = 255;
                            }
                        }
                    }
                    global_cost_map_ = map_utils::inflate_cost_map(obs_mask, map_inflation_params_);
                }
                cv::Mat angle, magnitude, terrain_labels;
                map_utils::inflate_direction_field(terrain_data, map_inflation_params_, angle, magnitude);
                terrain_labels = cv::Mat(map_size_y_, map_size_x_, CV_8UC1, terrain_data.terrain.data()).clone();
                map_utils::build_terrain_3chan(angle, magnitude, terrain_labels, global_direction_map_);
                RCLCPP_INFO(get_logger(), "Loaded default terrain msgpack size=%dx%d", map_size_x_, map_size_y_);
            } catch (const std::exception& e) {
                RCLCPP_FATAL(get_logger(), "Failed to load default navmap: %s", e.what());
                std::exit(EXIT_FAILURE);
            }
        }
        return;
    }
    pub_cost_map(global_cost_map_, now(), global_cost_map_pub_);
    pub_direction_map(global_direction_map_, now(), global_direction_map_pub_);
    if (enable_debug_) {
        if (global_point_cloud_) {
            pub_cloud(*global_point_cloud_, now(), debug_global_cloud_pub_);
        }
    }
}

void MapServerNode::local_cloud_callback(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!global_nav_map_initialized_) return;

    if (bypass_dynamic_obstacle_) {
        interfaces::msg::CostMaps cm;
        cm.prediction_dt = tracker_params_.prediction_dt;
        cm.maps.resize(1);
        fill_occupancy_grid(cv::Mat::zeros(map_size_y_, map_size_x_, CV_8UC1), msg->header.stamp, cm.maps[0]);
        local_cost_maps_pub_->publish(cm);
        return;
    }

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

    // 先提取动态障碍物点，再进行离群点过滤
    small_gicp::PointCloud dynamic_points;
    if (global_kdtree_) {
        dynamic_points = extract_dynamic_points_with_global_map(accumulated);
    } else {
        dynamic_points = extract_dynamic_points_without_global_map(accumulated);
    }
    const small_gicp::PointCloud denoised_dynamic_points = remove_statistical_outliers(dynamic_points);
    RCLCPP_DEBUG(get_logger(), "Identified %zu dynamic obstacle points", denoised_dynamic_points.points.size());

    // 动态障碍物分析
    cv::Mat obstacle_mask = create_obstacle_mask(denoised_dynamic_points);
    cv::Mat local_cost_map = map_utils::inflate_cost_map(obstacle_mask, map_inflation_params_);

    // 根据当前模式（是否有全局点云）选择 prediction 开关
    const bool use_prediction = global_kdtree_
        ? enable_prediction_with_cloud_
        : enable_prediction_without_cloud_;

    if (use_prediction) {
        // 目标跟踪预测
        if (!object_tracker_) {
            object_tracker_ = std::make_unique<ObjectTracker>(map_size_x_, map_size_y_, map_resolution_, tracker_params_);
            last_tracker_update_time_ = std::chrono::steady_clock::now();
        }
        const auto now_time = std::chrono::steady_clock::now();
        double dt = std::clamp(std::chrono::duration<double>(now_time - last_tracker_update_time_).count(), 0.01, 0.5);
        last_tracker_update_time_ = now_time;

        const auto tracker_update_begin = std::chrono::steady_clock::now();
        auto prediction_result = object_tracker_->update(obstacle_mask, dt);
        const auto tracker_update_end = std::chrono::steady_clock::now();

        // 并行膨胀预测栅格
        const auto tracker_inflate_begin = std::chrono::steady_clock::now();
        #pragma omp parallel for num_threads(num_threads_) schedule(static)
        for (int i = 0; i < static_cast<int>(prediction_result.future_masks.size()); i++) {
            prediction_result.future_masks[static_cast<size_t>(i)] = map_utils::inflate_cost_map(prediction_result.future_masks[static_cast<size_t>(i)], map_inflation_params_);
        }
        const auto tracker_inflate_end = std::chrono::steady_clock::now();

        RCLCPP_DEBUG(
            get_logger(),
            "ObjectTracker timing: update=%.3f ms, inflate=%.3f ms, tracks=%zu, motion_tracks=%zu, fallback_cells=%d, obstacle_cells=%d, dt=%.3f s",
            std::chrono::duration<double, std::milli>(tracker_update_end - tracker_update_begin).count(),
            std::chrono::duration<double, std::milli>(tracker_inflate_end - tracker_inflate_begin).count(),
            object_tracker_->track_count(),
            prediction_result.motion_track_count,
            cv::countNonZero(prediction_result.static_fallback_mask),
            cv::countNonZero(obstacle_mask),
            dt
        );

        // 打包发布代价地图序列（含预测帧）
        interfaces::msg::CostMaps cm;
        cm.prediction_dt = tracker_params_.prediction_dt;
        cm.maps.resize(prediction_result.future_masks.size() + 1);
        fill_occupancy_grid(local_cost_map, msg->header.stamp, cm.maps[0]);
        for (size_t i = 0; i < prediction_result.future_masks.size(); i++) {
            fill_occupancy_grid(prediction_result.future_masks[i], msg->header.stamp, cm.maps[i + 1]);
        }
        local_cost_maps_pub_->publish(cm);
    } else {
        // 无预测：直接发布当前帧代价地图
        interfaces::msg::CostMaps cm;
        cm.prediction_dt = tracker_params_.prediction_dt;
        cm.maps.resize(1);
        fill_occupancy_grid(local_cost_map, msg->header.stamp, cm.maps[0]);
        local_cost_maps_pub_->publish(cm);
    }

    // 调试信息发布
    if (enable_debug_) {
        pub_cloud(denoised_dynamic_points, msg->header.stamp, debug_dynamic_points_pub_);
        pub_cloud(accumulated, msg->header.stamp, debug_accumulated_cloud_pub_);
    }
}

small_gicp::PointCloud::Ptr MapServerNode::preprocess_cloud(sensor_msgs::msg::PointCloud2::SharedPtr msg) const {
    auto preprocessed = std::make_shared<small_gicp::PointCloud>();

    // 查找x, y, z字段的偏移量
    size_t offset_x = static_cast<size_t>(-1), offset_y = static_cast<size_t>(-1), offset_z = static_cast<size_t>(-1);
    for (const auto& field : msg->fields) {
        if (field.name == "x") offset_x = field.offset;
        else if (field.name == "y") offset_y = field.offset;
        else if (field.name == "z") offset_z = field.offset;
    }
    if (offset_x == static_cast<size_t>(-1) || offset_y == static_cast<size_t>(-1) || offset_z == static_cast<size_t>(-1)) {
        RCLCPP_WARN(get_logger(), "PointCloud2 missing x/y/z fields");
        return preprocessed;
    }

    // 获取点云到map的变换（用于后续与全局点云对齐/做代价图）
    Eigen::Isometry3d cloud_to_map;
    try {
        cloud_to_map = utils::convert_to<Eigen::Isometry3d>(
            tf_buffer_->lookupTransform("map", msg->header.frame_id, tf2::TimePointZero).transform
        );
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(), "Failed to lookup %s to map: %s", msg->header.frame_id.c_str(), ex.what());
        return preprocessed;
    }

    // 使用 imu_world (z轴竖直) 来做ROI判定
    Eigen::Isometry3d cloud_to_imu_world;
    try {
        cloud_to_imu_world = utils::convert_to<Eigen::Isometry3d>(
            tf_buffer_->lookupTransform("imu_world", msg->header.frame_id, tf2::TimePointZero).transform
        );
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(), "Failed to lookup %s to imu_world: %s", msg->header.frame_id.c_str(), ex.what());
        return preprocessed;
    }

    Eigen::Vector3d lidar_pos_imu_world;
    try {
        lidar_pos_imu_world = utils::convert_to<Eigen::Isometry3d>(
            tf_buffer_->lookupTransform("imu_world", "lidar_link", tf2::TimePointZero).transform
        ).translation().head<3>();
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(), "Failed to lookup lidar_link to imu_world: %s", ex.what());
        return preprocessed;
    }

    // 遍历点云，过滤并转换点
    const size_t point_step = msg->point_step;
    const size_t num_points = msg->width * msg->height;
    const uint8_t* data_ptr = msg->data.data();
    for (size_t i = 0; i < num_points; i++) {
        const Eigen::Vector3d pt_cloud(
            *reinterpret_cast<const float*>(data_ptr + offset_x),
            *reinterpret_cast<const float*>(data_ptr + offset_y),
            *reinterpret_cast<const float*>(data_ptr + offset_z)
        );

        const Eigen::Vector3d pt_map = cloud_to_map * pt_cloud;
        const Eigen::Vector3d pt_imu_world = cloud_to_imu_world * pt_cloud;
        const Eigen::Vector3d pt_rel = pt_imu_world - lidar_pos_imu_world;
        const double distance_xy = std::hypot(pt_rel.x(), pt_rel.y());
        if (local_map_params_.roi_xy_radius_min < distance_xy && distance_xy < local_map_params_.roi_xy_radius_max &&
            local_map_params_.roi_z_min < pt_rel.z() && pt_rel.z() < local_map_params_.roi_z_max) {
            preprocessed->points.emplace_back(pt_map.x(), pt_map.y(), pt_map.z(), 1.0);
        }
        data_ptr += point_step;
    }
    return preprocessed;
}

small_gicp::PointCloud MapServerNode::extract_dynamic_points_with_global_map(const small_gicp::PointCloud& cloud) const {
    small_gicp::PointCloud dynamic_points;
    const double distance_sq_threshold = local_map_params_.with_global_cloud.distance_threshold * local_map_params_.with_global_cloud.distance_threshold;
    for (const auto& point : cloud.points) {
        size_t index = 0;
        double sq_dist = 0.0;
        global_kdtree_->knn_search(point, 1, &index, &sq_dist);
        if (sq_dist > distance_sq_threshold) {
            dynamic_points.points.push_back(point);
        }
    }
    return dynamic_points;
}

small_gicp::PointCloud MapServerNode::remove_statistical_outliers(const small_gicp::PointCloud& cloud) const {
    small_gicp::PointCloud filtered;
    if (cloud.empty()) {
        return filtered;
    }

    const int knn = std::max(1, local_map_params_.sor_num_neighbors);
    const size_t knn_query_size = static_cast<size_t>(knn) + 1;
    if (cloud.size() < knn_query_size) {
        return cloud;
    }

    auto cloud_ptr = std::make_shared<small_gicp::PointCloud>(cloud);
    small_gicp::KdTree<small_gicp::PointCloud> tree(cloud_ptr, small_gicp::KdTreeBuilderOMP(num_threads_));

    std::vector<double> mean_knn_dist(cloud.size(), 0.0);
    std::vector<size_t> knn_indices(knn_query_size);
    std::vector<double> knn_sq_dists(knn_query_size);
    for (size_t i = 0; i < cloud.size(); i++) {
        const size_t found = tree.knn_search(cloud.points[i], knn_query_size, knn_indices.data(), knn_sq_dists.data());
        if (found <= 1) {
            mean_knn_dist[i] = std::numeric_limits<double>::infinity();
            continue;
        }

        double sum = 0.0;
        size_t count = 0;
        for (size_t j = 0; j < found; j++) {
            if (knn_indices[j] == i) {
                continue;
            }
            sum += std::sqrt(std::max(knn_sq_dists[j], 0.0));
            count++;
        }
        mean_knn_dist[i] = count == 0 ? std::numeric_limits<double>::infinity() : (sum / static_cast<double>(count));
    }

    double mean = 0.0;
    size_t valid_count = 0;
    for (const double d : mean_knn_dist) {
        if (!std::isfinite(d)) {
            continue;
        }
        mean += d;
        valid_count++;
    }
    if (valid_count == 0) {
        return filtered;
    }
    mean /= static_cast<double>(valid_count);

    double var = 0.0;
    for (const double d : mean_knn_dist) {
        if (!std::isfinite(d)) {
            continue;
        }
        const double diff = d - mean;
        var += diff * diff;
    }
    var /= static_cast<double>(valid_count);
    const double stddev = std::sqrt(var);
    const double threshold = mean + local_map_params_.sor_std_mul * stddev;

    filtered.points.reserve(cloud.size());
    for (size_t i = 0; i < cloud.size(); i++) {
        if (mean_knn_dist[i] <= threshold) {
            filtered.points.push_back(cloud.points[i]);
        }
    }

    RCLCPP_DEBUG(
        get_logger(),
        "SOR kept %zu / %zu points (mean=%.4f, std=%.4f, threshold=%.4f)",
        filtered.points.size(),
        cloud.size(),
        mean,
        stddev,
        threshold
    );
    return filtered;
}

small_gicp::PointCloud MapServerNode::extract_dynamic_points_without_global_map(const small_gicp::PointCloud& cloud) const {
    small_gicp::PointCloud dynamic_points;
    if (cloud.empty()) {
        return dynamic_points;
    }

    const int normal_knn = std::max(5, local_map_params_.without_global_cloud.normal_num_neighbors);
    if (cloud.size() < static_cast<size_t>(normal_knn)) {
        return dynamic_points;
    }

    auto cloud_with_normals_ptr = std::make_shared<small_gicp::PointCloud>(cloud);
    auto local_tree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(cloud_with_normals_ptr, small_gicp::KdTreeBuilderOMP(num_threads_));
    small_gicp::estimate_normals_omp(*cloud_with_normals_ptr, *local_tree, normal_knn, num_threads_);

    const double max_abs_nz = std::clamp(local_map_params_.without_global_cloud.vertical_normal_abs_z_max, 0.0, 1.0);
    dynamic_points.points.reserve(cloud_with_normals_ptr->size());
    for (size_t i = 0; i < cloud_with_normals_ptr->size(); i++) {
        const Eigen::Vector3d normal = cloud_with_normals_ptr->normal(i).head<3>();
        if (!normal.allFinite() || normal.norm() < 1e-6) {
            continue;
        }
        if (std::abs(normal.z()) <= max_abs_nz) {
            dynamic_points.points.push_back(cloud_with_normals_ptr->point(i));
        }
    }

    RCLCPP_DEBUG(
        get_logger(),
        "Fallback normal filter kept %zu / %zu points (|nz| <= %.3f)",
        dynamic_points.points.size(),
        cloud_with_normals_ptr->size(),
        max_abs_nz
    );
    return dynamic_points;
}

cv::Mat MapServerNode::create_obstacle_mask(const small_gicp::PointCloud& dynamic_points) const {
    cv::Mat counts = cv::Mat::zeros(map_size_y_, map_size_x_, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(map_size_y_, map_size_x_, CV_8UC1);
    const bool use_direction_filter = !global_kdtree_
        && local_map_params_.without_global_cloud.direction_filter_threshold > 0.0
        && !global_direction_map_.empty();
    for (const auto& pt : dynamic_points.points) {
        const int map_x = static_cast<int>(pt.x() / map_resolution_);
        const int map_y = static_cast<int>(pt.y() / map_resolution_);
        if (map_x < 0 || map_x >= map_size_x_ || map_y < 0 || map_y >= map_size_y_) continue;

        if (use_direction_filter) {
            // terrain_map: ch0=angle, ch1=magnitude (0-255), ch2=label
            const auto& px = global_direction_map_.at<cv::Vec3b>(map_y, map_x);
            const double magnitude = px[1] / 255.0;
            if (magnitude > local_map_params_.without_global_cloud.direction_filter_threshold) {
                continue;
            }
        }

        uint8_t& cell = counts.at<uint8_t>(map_y, map_x);
        cell = (cell < 255) ? (cell + 1) : 255;
        if (cell >= local_map_params_.cell_obstacle_point_threshold) {
            mask.at<uint8_t>(map_y, map_x) = 255;
        }
    }
    return mask;
}

cv::Mat MapServerNode::dynamic_obstacle_analysis(const small_gicp::PointCloud& dynamic_points) const {
    return map_utils::inflate_cost_map(create_obstacle_mask(dynamic_points), map_inflation_params_);
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

void MapServerNode::pub_direction_map(
    const cv::Mat& direction_map,
    const rclcpp::Time& stamp,
    const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher
) const {
    // terrain_map 为 CV_8UC3:
    //   ch0: 膨胀后的方向角度 (0-255 → 0~2π)
    //   ch1: 膨胀后的方向模长 (0-255 → 0.0~1.0)
    //   ch2: 原始语义标签 (0-6)
    sensor_msgs::msg::Image::SharedPtr direction_map_msg = cv_bridge::CvImage(
        std_msgs::msg::Header(), "8UC3", direction_map
    ).toImageMsg();
    direction_map_msg->header.stamp = stamp;
    direction_map_msg->header.frame_id = "map";
    publisher->publish(*direction_map_msg);
}

void MapServerNode::fill_occupancy_grid(
    const cv::Mat& cost_map,
    const rclcpp::Time& stamp,
    nav_msgs::msg::OccupancyGrid& grid
) const {
    grid.header.frame_id = "map";
    grid.header.stamp = stamp;
    grid.info.resolution = static_cast<float>(map_resolution_);
    grid.info.height = static_cast<uint32_t>(map_size_y_);
    grid.info.width = static_cast<uint32_t>(map_size_x_);
    grid.data.resize(static_cast<size_t>(map_size_x_ * map_size_y_));
    std::copy(cost_map.data, cost_map.data + map_size_x_ * map_size_y_, grid.data.data());
}

void MapServerNode::pub_cost_map(
    const cv::Mat& cost_map,
    const rclcpp::Time& stamp,
    const rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher
) const {
    nav_msgs::msg::OccupancyGrid occupancy_grid;
    fill_occupancy_grid(cost_map, stamp, occupancy_grid);
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
        std::memcpy(msg.data.data() + i * 12, pt.data(), sizeof(pt));
    }
    publisher->publish(msg);
}
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(map_server::MapServerNode)