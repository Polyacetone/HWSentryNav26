#pragma once

#include <random>
#include <vector>
#include <memory>
#include <Eigen/Core>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/factors/linear_damping_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
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

    // Logging params
    bool save_imu_rate_trajectory;

    // Number of threads for preprocessing and per-factor parallelism
    int num_threads;
    
    // GICP params
    double correspondence_distance;
    int full_connection_window_size;

    // Keyframe params
    enum class KeyframeUpdateStrategy { DISPLACEMENT, ENTROPY };
    KeyframeUpdateStrategy keyframe_strategy;
    int max_num_keyframes;
    double keyframe_delta_trans;
    double keyframe_delta_rot;
    double keyframe_entropy_thresh;
};

/**
* @brief CPU-based semi-tightly coupled LiDAR-IMU odometry
*/
class OdometryEstimationCPU {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit OdometryEstimationCPU(const Config::Ptr config);
    void insert_imu(const double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel);
    EstimationFrame::ConstPtr insert_frame(const PreprocessedFrame::Ptr& frame, std::vector<EstimationFrame::ConstPtr>& marginalized_frames);
    std::vector<EstimationFrame::ConstPtr> get_remaining_frames();
    EstimationFrame::ConstPtr get_target_ivox_frame();

private:
    gtsam::NonlinearFactorGraph create_factors(
        const int current,
        const std::shared_ptr<gtsam::ImuFactor>& imu_factor,
        gtsam::Values& new_values
    );
    void update_frames(const int current, const gtsam::NonlinearFactorGraph& new_factors);
    void update_smoother(const gtsam::NonlinearFactorGraph& new_factors, const gtsam::Values& new_values, const std::map<std::uint64_t, double>& new_stamp, int update_count = 0);
    void update_smoother(int update_count = 1);

    void update_keyframes_displacement(int current);
    void update_keyframes_entropy(const gtsam::NonlinearFactorGraph& matching_cost_factors, int current);

private:
    std::unique_ptr<OdometryEstimationCPUParams> params;

    // Sensor extrinsic params
    Eigen::Isometry3d T_lidar_imu;
    Eigen::Isometry3d T_imu_lidar;

    // Frames & keyframes
    int marginalized_cursor;
    std::vector<EstimationFrame::Ptr> frames;
    std::vector<EstimationFrame::Ptr> keyframes;

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
    
    // Entropy calculation
    int entropy_num_frames;
    double entropy_running_average;
};

}