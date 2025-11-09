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
* @brief Naive initial state estimator that simply calculates a pose that aligns linear acc with the gravity direction
*        Would not work well when the sensor is moving
*/
class InitialStateEstimation {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    InitialStateEstimation(
        const Config::Ptr config,
        const Eigen::Isometry3d& T_lidar_imu,
        const Eigen::Matrix<double, 6, 1>& imu_bias
    );
    void insert_frame(const PreprocessedFrame::ConstPtr& raw_frame);
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
    Eigen::Matrix<double, 6, 1> imu_bias;
    const Eigen::Isometry3d T_lidar_imu;
    bool force_init;
    Eigen::Vector3d init_v_world_imu;
    Eigen::Isometry3d init_T_world_imu;

    std::unique_ptr<CloudCovarianceEstimation> covariance_estimation;
    std::shared_ptr<gtsam_points::iVox> target_ivox;
    std::vector<std::pair<double, Eigen::Isometry3d>> T_odom_lidar;
    std::unique_ptr<IMUIntegration> imu_integration;
};

}