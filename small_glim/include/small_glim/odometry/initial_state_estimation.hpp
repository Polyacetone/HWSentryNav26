#pragma once

#include <Eigen/Dense>
#include <small_glim/preprocess/preprocessed_frame.hpp>
#include <small_glim/preprocess/cloud_covariance_estimation.hpp>
#include <small_glim/odometry/estimation_frame.hpp>
#include <small_glim/odometry/imu_integration.hpp>
#include <small_glim/common/config.hpp>
#include <gtsam_points/ann/ivox.hpp>

namespace small_glim {

/**
 * @brief Initial state estimator used by odometry subsystem.
 *
 * The estimator has two operating modes controlled by configuration:
 *   - graph-based (default): builds a small factor graph combining LiDAR
 *     odometry and IMU measurements to estimate initial pose, velocity, and
 *     biases.  This is the original, more sophisticated algorithm.
 *   - naive: when `odometry_estimation.naive_initialization` is true the
 *     estimator simply averages the direction of accelerations collected over
 *     the initialization window and assumes gravity-aligned z axis with zero
 *     initial position.  This mode is cheap and sufficient for many cases where
 *     the platform is stationary at startup.
 */
class InitialStateEstimation {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    InitialStateEstimation(
        const Config::Ptr config,
        const Eigen::Isometry3d& T_lidar_imu,
        const Eigen::Matrix<double, 6, 1>& imu_bias
    );
    void insert_frame(const PreprocessedFrame::ConstPtr raw_frame);
    void insert_imu(
        double stamp,
        const Eigen::Vector3d& linear_acc,
        const Eigen::Vector3d& angular_vel
    );
    EstimationFrame::ConstPtr initial_pose();
    void set_init_state(
        const Eigen::Isometry3d& init_T_world_imu,
        const Eigen::Vector3d& init_v_world_imu
    );

private:
    int num_threads;
    double window_size;
    bool ready;
    double init_stamp;
    double stamp;
    Eigen::Vector3d sum_acc;
    const Eigen::Matrix<double, 6, 1> imu_bias;
    const Eigen::Isometry3d T_lidar_imu;
    bool force_init;
    Eigen::Vector3d init_v_world_imu;
    Eigen::Isometry3d init_T_world_imu;
    bool naive_mode;
    bool align_initial_odom_to_imu;

    std::unique_ptr<CloudCovarianceEstimation> covariance_estimation;
    std::shared_ptr<gtsam_points::iVox> target_ivox;
    std::vector<std::pair<double, Eigen::Isometry3d>> T_odom_lidar;
    std::unique_ptr<IMUIntegration> imu_integration;
};

}