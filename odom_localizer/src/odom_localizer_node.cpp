#include <Eigen/Core>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <common_utils/convert.hpp>
#include <common_utils/ema_filter.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <interfaces/msg/robot_status.hpp>

namespace odom_localizer {
using namespace small_gicp;

namespace {
Eigen::Isometry3d transform_from_parameter(const std::vector<double>& values, const std::string& param_name) {
    if (values.size() != 7) {
        throw std::runtime_error(param_name + " must contain 7 elements: [x, y, z, qx, qy, qz, qw]");
    }

    Eigen::Quaterniond quaternion(values[6], values[3], values[4], values[5]);
    if (!std::isfinite(quaternion.norm()) || quaternion.norm() <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error(param_name + " contains an invalid quaternion");
    }
    quaternion.normalize();

    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translate(Eigen::Vector3d(values[0], values[1], values[2]));
    transform.rotate(quaternion);
    return transform;
}

double translation_distance(const Eigen::Isometry3d& lhs, const Eigen::Isometry3d& rhs) {
    return (lhs.translation() - rhs.translation()).norm();
}

double rotation_distance(const Eigen::Isometry3d& lhs, const Eigen::Isometry3d& rhs) {
    return Eigen::Quaterniond(lhs.rotation()).angularDistance(Eigen::Quaterniond(rhs.rotation()));
}

std::string transform_to_string(const Eigen::Isometry3d& transform) {
    const Eigen::Quaterniond q(transform.rotation());
    char buffer[160];
    std::snprintf(
        buffer, sizeof(buffer),
        "T=(%.3f, %.3f, %.3f) Q=(%.4f, %.4f, %.4f, %.4f)",
        transform.translation().x(), transform.translation().y(), transform.translation().z(),
        q.x(), q.y(), q.z(), q.w()
    );
    return std::string(buffer);
}
} // namespace

class OdomLocalizerNode: public rclcpp::Node {
public:
    explicit OdomLocalizerNode(const rclcpp::NodeOptions& options);

private:
    struct GeneralConfig {
        bool enable_debug;
        int num_threads;
        double publish_rate_hz;
    };

    struct FrameConfig {
        std::string map_frame;
        std::string odom_frame;
    };

    struct TopicConfig {
        std::string robot_status;
        std::string registered_cloud;
        std::string ivox_cloud;
    };

    struct StartupConfig {
        double robot_color_wait_timeout_s;
        double bootstrap_duration_s;
        Eigen::Isometry3d initial_transform_blue;
        Eigen::Isometry3d initial_transform_red;
        Eigen::Isometry3d initial_transform_default;
    };

    struct MapConfig {
        std::string cloud_filename;
        double downsample_resolution;
        int covariance_neighbors;
    };

    struct SourceConfig {
        size_t registered_window_frames;
        size_t min_points_raw;
        size_t min_points_downsampled;
        double downsample_resolution;
        int covariance_neighbors;
    };

    struct RegistrationConfig {
        double period_s;
        int max_iterations;
        double max_correspondence_distance;
        double translation_eps;
        double rotation_eps;
    };

    struct QualityConfig {
        double max_normalized_error;
        double min_overlap_ratio;
        double min_information_eigenvalue;
        double max_information_condition_number;
    };

    struct UpdateConfig {
        double ema_ratio;
        double max_translation_step;
        double max_rotation_step;
    };

    struct RegistrationSource {
        PointCloud::Ptr cloud;
        std::string label;
    };

    struct PreparedSource {
        PointCloud::Ptr cloud;
        size_t raw_points = 0;
        size_t downsampled_points = 0;
        std::string label;
    };

    struct RegistrationMetrics {
        bool converged = false;
        size_t iterations = 0;
        size_t num_inliers = 0;
        double normalized_error = std::numeric_limits<double>::infinity();
        double overlap_ratio = 0.0;
        double min_information_eigenvalue = 0.0;
        double max_information_eigenvalue = 0.0;
        double information_condition_number = std::numeric_limits<double>::infinity();
        std::array<double, 6> information_eigenvalues{};
    };

    struct RegistrationAssessment {
        bool accepted = false;
        Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
        RegistrationMetrics metrics;
        std::string reason;
    };

    GeneralConfig general_;
    FrameConfig frames_;
    TopicConfig topics_;
    StartupConfig startup_;
    MapConfig map_;
    SourceConfig source_;
    RegistrationConfig registration_;
    QualityConfig quality_;
    UpdateConfig update_;

    PointCloud::Ptr map_cloud_;
    KdTree<PointCloud>::Ptr map_kd_tree_;
    std::deque<PointCloud::Ptr> registered_cloud_window_;
    PointCloud::Ptr latest_ivox_cloud_;
    std::unique_ptr<utils::EMAFilter<Eigen::Isometry3d>> map_to_odom_filter_;
    std::optional<Eigen::Isometry3d> last_accepted_registration_transform_;
    bool initial_transform_initialized_ = false;
    bool has_successful_registration_ = false;
    std::chrono::steady_clock::time_point initialize_time_;

    rclcpp::CallbackGroup::SharedPtr non_registration_cb_group_;
    rclcpp::CallbackGroup::SharedPtr registration_cb_group_;
    mutable std::mutex transform_state_mutex_;
    mutable std::mutex source_data_mutex_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr registered_cloud_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr ivox_cloud_sub_;
    rclcpp::Subscription<interfaces::msg::RobotStatus>::SharedPtr robot_status_sub_;
    rclcpp::TimerBase::SharedPtr publish_timer_;
    rclcpp::TimerBase::SharedPtr registration_timer_;

    void load_parameters();
    void validate_parameters() const;
    void load_map();
    void create_callback_groups();
    void create_subscriptions();
    void create_timers();

    void publish_timer_callback();
    void registration_timer_callback();
    Eigen::Isometry3d get_current_map_to_odom() const;
    bool is_transform_initialized() const;
    bool has_map_cloud() const { return map_cloud_ != nullptr; }
    void disable_robot_status_subscription();
    void registered_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void ivox_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void robot_status_callback(const interfaces::msg::RobotStatus::SharedPtr msg);

    void maybe_initialize_default_transform();
    void initialize_transform(const Eigen::Isometry3d& transform, const char* reason);
    std::optional<RegistrationSource> select_registration_source();
    std::optional<PreparedSource> prepare_source_for_registration(const RegistrationSource& source) const;
    RegistrationAssessment align_to_map(const PreparedSource& source) const;
    bool evaluate_registration_result(
        const RegistrationResult& result,
        size_t source_points,
        RegistrationMetrics& metrics,
        std::string& reason
    ) const;
    bool apply_registration_update(const Eigen::Isometry3d& transform);
    void publish_transform();

    PointCloud::Ptr convert_pointcloud2_to_small_gicp(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) const;
};

OdomLocalizerNode::OdomLocalizerNode(const rclcpp::NodeOptions& options): Node("odom_localizer", options) {
    load_parameters();
    validate_parameters();

    if (general_.enable_debug) {
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        RCLCPP_DEBUG(get_logger(), "Debug mode enabled");
    }

    map_to_odom_filter_ = std::make_unique<utils::EMAFilter<Eigen::Isometry3d>>(update_.ema_ratio);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    create_callback_groups();

    load_map();
    create_subscriptions();
    create_timers();

    initialize_time_ = std::chrono::steady_clock::now();
}

void OdomLocalizerNode::load_parameters() {
    general_.enable_debug = declare_parameter<bool>("general.enable_debug");
    general_.num_threads = static_cast<int>(declare_parameter<int>("general.num_threads"));
    general_.publish_rate_hz = declare_parameter<double>("general.publish_rate_hz");

    frames_.map_frame = declare_parameter<std::string>("frames.map");
    frames_.odom_frame = declare_parameter<std::string>("frames.odom");

    topics_.robot_status = declare_parameter<std::string>("topics.robot_status");
    topics_.registered_cloud = declare_parameter<std::string>("topics.registered_cloud");
    topics_.ivox_cloud = declare_parameter<std::string>("topics.ivox_cloud");

    startup_.robot_color_wait_timeout_s = declare_parameter<double>("startup.robot_color_wait_timeout_s");
    startup_.bootstrap_duration_s = declare_parameter<double>("startup.bootstrap_duration_s");
    startup_.initial_transform_blue = transform_from_parameter(
        declare_parameter<std::vector<double>>("startup.initial_transform.blue"),
        "startup.initial_transform.blue"
    );
    startup_.initial_transform_red = transform_from_parameter(
        declare_parameter<std::vector<double>>("startup.initial_transform.red"),
        "startup.initial_transform.red"
    );
    startup_.initial_transform_default = transform_from_parameter(
        declare_parameter<std::vector<double>>("startup.initial_transform.default"),
        "startup.initial_transform.default"
    );

    map_.cloud_filename = declare_parameter<std::string>("map.cloud_filename");
    map_.downsample_resolution = declare_parameter<double>("map.downsample_resolution");
    map_.covariance_neighbors = static_cast<int>(declare_parameter<int>("map.covariance_neighbors"));

    source_.registered_window_frames = static_cast<size_t>(declare_parameter<int>("source.registered_window_frames"));
    source_.min_points_raw = static_cast<size_t>(declare_parameter<int>("source.min_points_raw"));
    source_.min_points_downsampled = static_cast<size_t>(declare_parameter<int>("source.min_points_downsampled"));
    source_.downsample_resolution = declare_parameter<double>("source.downsample_resolution");
    source_.covariance_neighbors = static_cast<int>(declare_parameter<int>("source.covariance_neighbors"));

    registration_.period_s = declare_parameter<double>("registration.period_s");
    registration_.max_iterations = static_cast<int>(declare_parameter<int>("registration.gicp.max_iterations"));
    registration_.max_correspondence_distance = declare_parameter<double>("registration.gicp.max_correspondence_distance");
    registration_.translation_eps = declare_parameter<double>("registration.gicp.translation_eps");
    registration_.rotation_eps = declare_parameter<double>("registration.gicp.rotation_eps");

    quality_.max_normalized_error = declare_parameter<double>("registration.quality.max_normalized_error");
    quality_.min_overlap_ratio = declare_parameter<double>("registration.quality.min_overlap_ratio");
    quality_.min_information_eigenvalue = declare_parameter<double>("registration.quality.min_information_eigenvalue");
    quality_.max_information_condition_number = declare_parameter<double>("registration.quality.max_information_condition_number");

    update_.ema_ratio = declare_parameter<double>("update.ema_ratio");
    update_.max_translation_step = declare_parameter<double>("update.max_translation_step");
    update_.max_rotation_step = declare_parameter<double>("update.max_rotation_step");
}

void OdomLocalizerNode::validate_parameters() const {
    const auto require_positive = [](const double value, const char* name) {
        if (!(value > 0.0)) {
            throw std::runtime_error(std::string(name) + " must be > 0");
        }
    };
    const auto require_non_negative = [](const double value, const char* name) {
        if (value < 0.0) {
            throw std::runtime_error(std::string(name) + " must be >= 0");
        }
    };

    require_positive(static_cast<double>(general_.num_threads), "general.num_threads");
    require_positive(general_.publish_rate_hz, "general.publish_rate_hz");
    require_non_negative(startup_.robot_color_wait_timeout_s, "startup.robot_color_wait_timeout_s");
    require_non_negative(startup_.bootstrap_duration_s, "startup.bootstrap_duration_s");

    if (source_.registered_window_frames == 0) {
        throw std::runtime_error("source.registered_window_frames must be > 0");
    }
    if (source_.min_points_downsampled > source_.min_points_raw) {
        throw std::runtime_error("source.min_points_downsampled must be <= source.min_points_raw");
    }

    require_non_negative(map_.downsample_resolution, "map.downsample_resolution");
    require_non_negative(source_.downsample_resolution, "source.downsample_resolution");
    require_positive(registration_.period_s, "registration.period_s");
    require_positive(registration_.max_correspondence_distance, "registration.gicp.max_correspondence_distance");
    require_positive(registration_.translation_eps, "registration.gicp.translation_eps");
    require_positive(registration_.rotation_eps, "registration.gicp.rotation_eps");

    require_positive(static_cast<double>(registration_.max_iterations), "registration.gicp.max_iterations");
    require_positive(static_cast<double>(map_.covariance_neighbors), "map.covariance_neighbors");
    require_positive(static_cast<double>(source_.covariance_neighbors), "source.covariance_neighbors");
    if (update_.ema_ratio < 0.0 || update_.ema_ratio > 1.0) {
        throw std::runtime_error("update.ema_ratio must be within [0, 1]");
    }
    require_non_negative(update_.max_translation_step, "update.max_translation_step");
    require_non_negative(update_.max_rotation_step, "update.max_rotation_step");
    require_non_negative(quality_.min_overlap_ratio, "registration.quality.min_overlap_ratio");
    require_non_negative(quality_.min_information_eigenvalue, "registration.quality.min_information_eigenvalue");
    require_non_negative(quality_.max_information_condition_number, "registration.quality.max_information_condition_number");
}

void OdomLocalizerNode::load_map() {
    if (map_.cloud_filename.empty()) {
        RCLCPP_WARN(get_logger(), "No map cloud file specified, odom_localizer will only publish the initial transform");
        return;
    }

    const std::string map_cloud_path = ament_index_cpp::get_package_share_directory("odom_localizer") + "/maps/" + map_.cloud_filename;
    auto map_cloud_pcl = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (pcl::io::loadPCDFile(map_cloud_path, *map_cloud_pcl) == -1) {
        throw std::runtime_error("Failed to load map PCD: " + map_cloud_path);
    }

    map_cloud_ = std::make_shared<PointCloud>(
        utils::convert_to<PointCloud>(*map_cloud_pcl)
    );
    RCLCPP_INFO(get_logger(), "Loaded map point cloud with %zu points", map_cloud_->points.size());

    if (map_.downsample_resolution > 0.0) {
        map_cloud_ = voxelgrid_sampling_omp(*map_cloud_, map_.downsample_resolution, general_.num_threads);
        RCLCPP_INFO(get_logger(), "Downsampled map point cloud to %zu points", map_cloud_->points.size());
    }

    map_kd_tree_ = std::make_shared<KdTree<PointCloud>>(map_cloud_, KdTreeBuilderOMP(general_.num_threads));
    estimate_covariances_omp(*map_cloud_, *map_kd_tree_, map_.covariance_neighbors, general_.num_threads);
}

void OdomLocalizerNode::create_callback_groups() {
    non_registration_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    registration_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
}

void OdomLocalizerNode::create_subscriptions() {
    rclcpp::SubscriptionOptions non_registration_options;
    non_registration_options.callback_group = non_registration_cb_group_;

    robot_status_sub_ = create_subscription<interfaces::msg::RobotStatus>(
        topics_.robot_status,
        1,
        [this](const interfaces::msg::RobotStatus::SharedPtr msg) { robot_status_callback(msg); },
        non_registration_options
    );

    if (!has_map_cloud()) {
        return;
    }

    registered_cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        topics_.registered_cloud,
        rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { registered_cloud_callback(msg); },
        non_registration_options
    );

    ivox_cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        topics_.ivox_cloud,
        rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { ivox_cloud_callback(msg); },
        non_registration_options
    );
}

void OdomLocalizerNode::create_timers() {
    const auto publish_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / general_.publish_rate_hz)
    );
    publish_timer_ = create_wall_timer(publish_period, [this]() { publish_timer_callback(); }, non_registration_cb_group_);

    if (!has_map_cloud()) {
        return;
    }

    const auto registration_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(registration_.period_s)
    );
    registration_timer_ = create_wall_timer(registration_period, [this]() { registration_timer_callback(); }, registration_cb_group_);
}

void OdomLocalizerNode::publish_timer_callback() {
    maybe_initialize_default_transform();
    if (!is_transform_initialized()) {
        return;
    }

    publish_transform();
}

Eigen::Isometry3d OdomLocalizerNode::get_current_map_to_odom() const {
    std::lock_guard<std::mutex> lock(transform_state_mutex_);
    return map_to_odom_filter_->value();
}

bool OdomLocalizerNode::is_transform_initialized() const {
    std::lock_guard<std::mutex> lock(transform_state_mutex_);
    return initial_transform_initialized_;
}

void OdomLocalizerNode::disable_robot_status_subscription() {
    std::lock_guard<std::mutex> lock(transform_state_mutex_);
    robot_status_sub_.reset();
}

void OdomLocalizerNode::registration_timer_callback() {
    const auto start_time = std::chrono::steady_clock::now();

    maybe_initialize_default_transform();
    if (!is_transform_initialized() || !map_cloud_ || !map_kd_tree_) {
        return;
    }

    const auto source = select_registration_source();
    if (!source) {
        return;
    }

    const auto prepared_source = prepare_source_for_registration(*source);
    if (!prepared_source) {
        return;
    }

    const Eigen::Isometry3d current_map_to_odom = get_current_map_to_odom();
    const RegistrationAssessment assessment = align_to_map(*prepared_source);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();

    if (!assessment.accepted) {
        RCLCPP_WARN(
            get_logger(),
            "Registration failed (%s): %s [current=%s gicp=%s iters=%zu inliers=%zu "
            "error=%.4f overlap=%.4f min_eig=%.3e cond=%.1f] (%.2f s)",
            prepared_source->label.c_str(),
            assessment.reason.c_str(),
            transform_to_string(current_map_to_odom).c_str(),
            transform_to_string(assessment.transform).c_str(),
            assessment.metrics.iterations,
            assessment.metrics.num_inliers,
            assessment.metrics.normalized_error,
            assessment.metrics.overlap_ratio,
            assessment.metrics.min_information_eigenvalue,
            assessment.metrics.information_condition_number,
            elapsed
        );
        return;
    }

    if (!apply_registration_update(assessment.transform)) {
        return;
    }

    const Eigen::Isometry3d new_map_to_odom = get_current_map_to_odom();
    RCLCPP_INFO(
        get_logger(),
        "Accepted registration (%s): [current=%s gicp=%s update=%s inliers=%zu "
        "error=%.4f overlap=%.4f min_eig=%.3e cond=%.1f] (%.2f s)",
        prepared_source->label.c_str(),
        transform_to_string(current_map_to_odom).c_str(),
        transform_to_string(assessment.transform).c_str(),
        transform_to_string(new_map_to_odom).c_str(),
        assessment.metrics.num_inliers,
        assessment.metrics.normalized_error,
        assessment.metrics.overlap_ratio,
        assessment.metrics.min_information_eigenvalue,
        assessment.metrics.information_condition_number,
        elapsed
    );
}

void OdomLocalizerNode::registered_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto cloud = convert_pointcloud2_to_small_gicp(msg);
    if (cloud->points.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(source_data_mutex_);
    registered_cloud_window_.push_back(cloud);
    while (registered_cloud_window_.size() > source_.registered_window_frames) {
        registered_cloud_window_.pop_front();
    }
}

void OdomLocalizerNode::ivox_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto cloud = convert_pointcloud2_to_small_gicp(msg);
    if (cloud->points.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(source_data_mutex_);
    latest_ivox_cloud_ = std::move(cloud);
}

void OdomLocalizerNode::robot_status_callback(const interfaces::msg::RobotStatus::SharedPtr msg) {
    if (is_transform_initialized()) {
        disable_robot_status_subscription();
        return;
    }

    if (msg->robot_color) {
        initialize_transform(startup_.initial_transform_red, "robot color RED");
    } else {
        initialize_transform(startup_.initial_transform_blue, "robot color BLUE");
    }

    disable_robot_status_subscription();
}

void OdomLocalizerNode::maybe_initialize_default_transform() {
    if (is_transform_initialized()) {
        return;
    }

    const auto now_time = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now_time - initialize_time_).count();
    if (elapsed < startup_.robot_color_wait_timeout_s) {
        return;
    }

    RCLCPP_WARN(
        get_logger(),
        "Robot color not initialized within %.2f s, falling back to startup.initial_transform.default",
        startup_.robot_color_wait_timeout_s
    );
    initialize_transform(startup_.initial_transform_default, "robot color timeout");
    disable_robot_status_subscription();
}

void OdomLocalizerNode::initialize_transform(const Eigen::Isometry3d& transform, const char* reason) {
    {
        std::lock_guard<std::mutex> lock(transform_state_mutex_);
        if (initial_transform_initialized_) {
            return;
        }
        map_to_odom_filter_->initialize(transform);
        last_accepted_registration_transform_ = std::nullopt;
        has_successful_registration_ = false;
        initial_transform_initialized_ = true;
    }

    RCLCPP_INFO(
        get_logger(),
        "Initialized odom->map from %s: %s",
        reason,
        transform_to_string(transform).c_str()
    );
}

std::optional<OdomLocalizerNode::RegistrationSource> OdomLocalizerNode::select_registration_source() {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - initialize_time_).count();

    if (elapsed < startup_.bootstrap_duration_s) {
        std::deque<PointCloud::Ptr> registered_cloud_window_snapshot;
        {
            std::lock_guard<std::mutex> lock(source_data_mutex_);
            if (registered_cloud_window_.empty()) {
                RCLCPP_WARN(
                    get_logger(),
                    "Bootstrap registration skipped: no registered_cloud data in the sliding window"
                );
                return std::nullopt;
            }
            registered_cloud_window_snapshot = registered_cloud_window_;
        }

        PointCloud accumulated;
        for (const auto& cloud : registered_cloud_window_snapshot) {
            accumulated.points.insert(accumulated.points.end(), cloud->points.begin(), cloud->points.end());
        }

        return RegistrationSource {
            .cloud = std::make_shared<PointCloud>(std::move(accumulated)),
            .label = "registered_cloud bootstrap window",
        };
    }

    PointCloud::Ptr latest_ivox_cloud_snapshot;
    {
        std::lock_guard<std::mutex> lock(source_data_mutex_);
        registered_cloud_window_.clear();

        if (!latest_ivox_cloud_) {
            RCLCPP_WARN(
                get_logger(),
                "Periodic registration skipped: no ivox_cloud received yet"
            );
            return std::nullopt;
        }
        latest_ivox_cloud_snapshot = latest_ivox_cloud_;
    }

    return RegistrationSource {
        .cloud = std::make_shared<PointCloud>(*latest_ivox_cloud_snapshot),
        .label = "ivox_cloud periodic snapshot",
    };
}

std::optional<OdomLocalizerNode::PreparedSource> OdomLocalizerNode::prepare_source_for_registration(const RegistrationSource& source) const {
    const size_t raw_points = source.cloud->points.size();
    if (raw_points < source_.min_points_raw) {
        RCLCPP_WARN(
            get_logger(),
            "Registration skipped (%s): raw source cloud too small (%zu < %zu)",
            source.label.c_str(),
            raw_points,
            source_.min_points_raw
        );
        return std::nullopt;
    }

    PointCloud::Ptr downsampled;
    if (source_.downsample_resolution > 0.0) {
        downsampled = voxelgrid_sampling_omp(*source.cloud, source_.downsample_resolution, general_.num_threads);
    } else {
        downsampled = std::make_shared<PointCloud>(*source.cloud);
    }

    const size_t downsampled_points = downsampled->points.size();
    if (downsampled_points < source_.min_points_downsampled) {
        RCLCPP_WARN(
            get_logger(),
            "Registration skipped (%s): downsampled source cloud too small (%zu < %zu)",
            source.label.c_str(),
            downsampled_points,
            source_.min_points_downsampled
        );
        return std::nullopt;
    }

    auto source_kd_tree = std::make_shared<KdTree<PointCloud>>(downsampled, KdTreeBuilderOMP(general_.num_threads));
    estimate_covariances_omp(*downsampled, *source_kd_tree, source_.covariance_neighbors, general_.num_threads);

    return PreparedSource {.cloud = std::move(downsampled), .raw_points = raw_points, .downsampled_points = downsampled_points, .label = source.label};
}

OdomLocalizerNode::RegistrationAssessment OdomLocalizerNode::align_to_map(const PreparedSource& source) const {
    RegistrationAssessment assessment;

    Registration<GICPFactor, ParallelReductionOMP> registration;
    registration.reduction.num_threads = general_.num_threads;
    registration.optimizer.max_iterations = registration_.max_iterations;
    registration.criteria.translation_eps = registration_.translation_eps;
    registration.criteria.rotation_eps = registration_.rotation_eps;
    registration.rejector.max_dist_sq = registration_.max_correspondence_distance * registration_.max_correspondence_distance;

    const Eigen::Isometry3d initial_guess = get_current_map_to_odom();

    try {
        const RegistrationResult result = registration.align(*map_cloud_, *source.cloud, *map_kd_tree_, initial_guess);
        assessment.transform = result.T_target_source;
        assessment.accepted = evaluate_registration_result(result, source.cloud->points.size(), assessment.metrics, assessment.reason);
    } catch (const std::exception& exception) {
        assessment.reason = std::string("GICP exception: ") + exception.what();
    }

    return assessment;
}

bool OdomLocalizerNode::evaluate_registration_result(
    const RegistrationResult& result,
    const size_t source_points,
    RegistrationMetrics& metrics,
    std::string& reason
) const {
    metrics.converged = result.converged;
    metrics.iterations = result.iterations + 1;
    metrics.num_inliers = result.num_inliers;

    if (!result.converged) {
        reason = "solver did not converge";
        return false;
    }

    if (source_points == 0 || result.num_inliers == 0) {
        reason = "no valid inliers";
        return false;
    }

    if (!std::isfinite(result.error)) {
        reason = "non-finite registration error";
        return false;
    }

    metrics.normalized_error = result.error / static_cast<double>(result.num_inliers);
    metrics.overlap_ratio = static_cast<double>(result.num_inliers) / static_cast<double>(source_points);

    if (!std::isfinite(metrics.normalized_error)) {
        reason = "non-finite normalized error";
        return false;
    }
    if (metrics.normalized_error > quality_.max_normalized_error) {
        reason = "normalized error exceeds threshold";
        return false;
    }
    if (metrics.overlap_ratio < quality_.min_overlap_ratio) {
        reason = "overlap ratio below threshold";
        return false;
    }

    const Eigen::Matrix<double, 6, 6> information = 0.5 * (result.H + result.H.transpose());
    if (!information.allFinite()) {
        reason = "information matrix contains non-finite values";
        return false;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigen_solver(information);
    if (eigen_solver.info() != Eigen::Success) {
        reason = "failed to decompose information matrix";
        return false;
    }

    const Eigen::Matrix<double, 6, 1> eigenvalues = eigen_solver.eigenvalues();
    for (int i = 0; i < eigenvalues.size(); ++i) {
        metrics.information_eigenvalues[static_cast<size_t>(i)] = eigenvalues[i];
    }
    metrics.min_information_eigenvalue = eigenvalues.minCoeff();
    metrics.max_information_eigenvalue = eigenvalues.maxCoeff();

    if (!std::isfinite(metrics.min_information_eigenvalue) || !std::isfinite(metrics.max_information_eigenvalue)) {
        reason = "information eigenvalues are non-finite";
        return false;
    }
    if (metrics.min_information_eigenvalue <= 0.0) {
        reason = "information matrix is not positive definite";
        return false;
    }
    if (metrics.min_information_eigenvalue < quality_.min_information_eigenvalue) {
        reason = "minimum information eigenvalue below threshold";
        return false;
    }

    metrics.information_condition_number = metrics.max_information_eigenvalue / metrics.min_information_eigenvalue;
    if (!std::isfinite(metrics.information_condition_number)) {
        reason = "invalid information matrix condition number";
        return false;
    }
    if (
        quality_.max_information_condition_number > 0.0 &&
        metrics.information_condition_number > quality_.max_information_condition_number
    ) {
        reason = "information matrix is too ill-conditioned";
        return false;
    }

    return true;
}

bool OdomLocalizerNode::apply_registration_update(const Eigen::Isometry3d& transform) {
    {
        std::lock_guard<std::mutex> lock(transform_state_mutex_);

        if (!has_successful_registration_) {
            map_to_odom_filter_->initialize(transform);
            last_accepted_registration_transform_ = transform;
            has_successful_registration_ = true;
            RCLCPP_INFO(
                get_logger(),
                "Accepted first successful registration directly: %s",
                transform_to_string(transform).c_str()
            );
            return true;
        }
    }

    Eigen::Isometry3d previous_registration;
    double translation_delta = 0.0;
    double rotation_delta = 0.0;

    {
        std::lock_guard<std::mutex> lock(transform_state_mutex_);
        previous_registration = *last_accepted_registration_transform_;
        translation_delta = translation_distance(transform, previous_registration);
        rotation_delta = rotation_distance(transform, previous_registration);

        if (translation_delta <= update_.max_translation_step &&
            rotation_delta <= update_.max_rotation_step) {
            map_to_odom_filter_->update(transform);
            last_accepted_registration_transform_ = transform;
        }
    }

    if (translation_delta > update_.max_translation_step) {
        RCLCPP_WARN(
            get_logger(),
            "Discarded registration update: translation jump %.3f m exceeds threshold %.3f m "
            "[previous=%s gicp=%s]",
            translation_delta,
            update_.max_translation_step,
            transform_to_string(previous_registration).c_str(),
            transform_to_string(transform).c_str()
        );
        return false;
    }

    if (rotation_delta > update_.max_rotation_step) {
        RCLCPP_WARN(
            get_logger(),
            "Discarded registration update: rotation jump %.3f rad exceeds threshold %.3f rad "
            "[previous=%s gicp=%s]",
            rotation_delta,
            update_.max_rotation_step,
            transform_to_string(previous_registration).c_str(),
            transform_to_string(transform).c_str()
        );
        return false;
    }

    return true;
}

void OdomLocalizerNode::publish_transform() {
    geometry_msgs::msg::TransformStamped transform_msg;
    transform_msg.header.stamp = now() + rclcpp::Duration::from_seconds(1 / general_.publish_rate_hz + 0.1);
    transform_msg.header.frame_id = frames_.map_frame;
    transform_msg.child_frame_id = frames_.odom_frame;
    transform_msg.transform = utils::convert_to<geometry_msgs::msg::Transform>(get_current_map_to_odom());
    tf_broadcaster_->sendTransform(transform_msg);
}

PointCloud::Ptr OdomLocalizerNode::convert_pointcloud2_to_small_gicp(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) const {
    auto cloud = std::make_shared<PointCloud>();

    size_t offset_x = std::numeric_limits<size_t>::max();
    size_t offset_y = std::numeric_limits<size_t>::max();
    size_t offset_z = std::numeric_limits<size_t>::max();
    for (const auto& field : msg->fields) {
        if (field.name == "x") {
            offset_x = field.offset;
        } else if (field.name == "y") {
            offset_y = field.offset;
        } else if (field.name == "z") {
            offset_z = field.offset;
        }
    }

    if (offset_x == std::numeric_limits<size_t>::max() || offset_y == std::numeric_limits<size_t>::max() || offset_z == std::numeric_limits<size_t>::max()) {
        RCLCPP_WARN(get_logger(), "PointCloud2 missing x/y/z fields");
        return cloud;
    }

    const size_t point_step = msg->point_step;
    const size_t num_points = static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);
    const uint8_t* data_ptr = msg->data.data();
    bool has_invalid_points = false;

    cloud->points.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        float x, y, z;
        std::memcpy(&x, data_ptr + offset_x, sizeof(float));
        std::memcpy(&y, data_ptr + offset_y, sizeof(float));
        std::memcpy(&z, data_ptr + offset_z, sizeof(float));
        data_ptr += point_step;

        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            has_invalid_points = true;
            continue;
        }

        cloud->points.emplace_back(x, y, z, 1.0);
    }

    if (cloud->points.empty()) {
        RCLCPP_WARN(get_logger(), "Converted point cloud contains no valid points");
    }
    if (has_invalid_points) {
        RCLCPP_WARN(get_logger(), "Converted point cloud contains invalid points (NaN/Inf)");
    }

    return cloud;
}


} // namespace odom_localizer

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(odom_localizer::OdomLocalizerNode)
