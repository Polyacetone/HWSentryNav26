#pragma once

#include <random>
#include <vector>
#include <memory>
#include <Eigen/Core>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/gaussian_voxelmap.hpp>
#include <gtsam_points/factors/linear_damping_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_with_fallback.hpp>
#include <gtsam_points/ann/ivox.hpp>
#include <small_glim/odometry/initial_state_estimation.hpp>
#include <small_glim/odometry/estimation_frame.hpp>
#include <small_glim/odometry/imu_integration.hpp>
#include <small_glim/preprocess/cloud_deskewing.hpp>
#include <small_glim/preprocess/cloud_covariance_estimation.hpp>
#include <small_glim/common/config.hpp>

namespace small_glim {

/**
* @brief Parameters for OdometryEstimationCPU
*/
struct OdometryEstimationCPUParams {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit OdometryEstimationCPUParams(const Config::Ptr config);

public:
    // Sensor params
    bool fix_imu_bias;
    double imu_bias_noise;
    Eigen::Isometry3d T_lidar_imu;
    Eigen::Matrix<double, 6, 1> imu_bias;

    // Init state
    bool use_init_world_imu;
    Eigen::Isometry3d init_T_world_imu;
    Eigen::Vector3d init_v_world_imu;
    double init_pose_damping_scale;

    // Optimization params
    double smoother_lag;
    bool use_isam2_dogleg;
    double isam2_relinearize_skip;
    double isam2_relinearize_thresh;

    // MISC
    int num_threads; // Number of threads for preprocessing and per-factor parallelism

    // Registration params
    enum class RegistrationType { GICP, VGICP } registration_type; ///< Registration type (GICP or VGICP)
    int keyframe_window_size; ///< Local window size for pairwise registration
    int keyframe_delta; ///< Keyframe delta
    int lru_thresh; ///< LRU cache threshold
    double target_downsampling_rate; ///< Downsampling rate for points to be inserted into the target
    double ivox_resolution; ///< iVox resolution (for GICP)
    double ivox_min_dist; ///< Minimum distance between points in an iVox cell (for GICP)
    double vgicp_resolution; ///< Voxelmap resolution (for VGICP)
    int vgicp_voxelmap_levels; ///< Multi-resolution voxelmap levesl (for VGICP)
    double vgicp_voxelmap_scaling_factor; ///< Multi-resolution voxelmap scaling factor (for VGICP)
};

/**
* @brief CPU-based semi-tightly coupled LiDAR-IMU odometry
*/
class OdometryEstimationCPU {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit OdometryEstimationCPU(const Config::Ptr config);
    void insert_imu(const double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel);
    EstimationFrame::ConstPtr insert_frame(const PreprocessedFrame::Ptr frame, std::vector<EstimationFrame::ConstPtr>& marginalized_frames);
    std::vector<EstimationFrame::ConstPtr> get_remaining_frames();
    EstimationFrame::ConstPtr get_target_ivox_frame();

private:
    gtsam::NonlinearFactorGraph create_factors(
        const int current,
        const std::shared_ptr<gtsam::ImuFactor>& imu_factor,
        gtsam::Values& new_values
    );
    void fallback_smoother();
    void update_target(const int current, const Eigen::Isometry3d& T_target_imu);
    int select_target_update_frame(const int preferred) const;
    void update_frames(const int current, const gtsam::NonlinearFactorGraph& new_factors);
    void update_smoother(const gtsam::NonlinearFactorGraph& new_factors, const gtsam::Values& new_values, const std::map<std::uint64_t, double>& new_stamp, int update_count = 0);

private:
    std::unique_ptr<OdometryEstimationCPUParams> params;

    // Sensor extrinsic params
    Eigen::Isometry3d T_lidar_imu;
    Eigen::Isometry3d T_imu_lidar;

    // Frames & keyframes
    int marginalized_cursor;
    std::vector<EstimationFrame::Ptr> frames;

    // Utility classes
    std::unique_ptr<InitialStateEstimation> init_estimation;
    std::unique_ptr<IMUIntegration> imu_integration;
    std::unique_ptr<CloudDeskewing> deskewing;
    std::unique_ptr<CloudCovarianceEstimation> covariance_estimation;

    // Optimizer
    using FixedLagSmootherExt = gtsam_points::IncrementalFixedLagSmootherExtWithFallback;
    std::unique_ptr<FixedLagSmootherExt> smoother;

    // Registration params
    std::mt19937 mt; ///< RNG
    Eigen::Isometry3d last_T_target_imu; ///< Last IMU pose w.r.t. target model
    std::vector<std::shared_ptr<gtsam_points::GaussianVoxelMapCPU>> target_voxelmaps; ///< VGICP target voxelmap
    std::shared_ptr<gtsam_points::iVox> target_ivox; ///< GICP target iVox
    std::deque<int> active_keyframes; ///< Active keyframes for pairwise registration
};

}