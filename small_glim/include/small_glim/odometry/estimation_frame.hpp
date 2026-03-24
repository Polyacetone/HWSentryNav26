#pragma once

#include <memory>
#include <array>
#include <Eigen/Dense>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/gaussian_voxelmap.hpp>
#include <small_glim/preprocess/preprocessed_frame.hpp>

namespace small_glim {

enum class FrameType { WORLD, LIDAR, IMU };

/**
* @brief Odometry estimation frame
*/
struct EstimationFrame {
    using Ptr = std::shared_ptr<EstimationFrame>;
    using ConstPtr = std::shared_ptr<const EstimationFrame>;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
    * @brief Make a clone of the estimation frame. (Points data are shallow copied)
    * @return EstimationFrame::Ptr   Cloned frame
    */
    EstimationFrame::Ptr clone() const;

    /**
    * @brief Make a clone of the estimation frame instance but without points data.
    * @return EstimationFrame::Ptr   Cloned frame without points
    */
    EstimationFrame::Ptr clone_wo_points() const;

    /**
    * @brief Get the sensor pose according to frame_type.
    * @return const Eigen::Isometry3d  Sensor pose
    */
    const Eigen::Isometry3d T_world_frame() const;

    /**
    * @brief Set the sensor pose.
    * @param frame_type  Sensor coodinate frame
    * @param T           Sensor pose
    */
    void set_T_world_frame(FrameType frame_type, const Eigen::Isometry3d& T);

public:
    size_t id; ///< Frame ID
    double stamp; ///< Timestamp

    Eigen::Isometry3d T_lidar_imu; ///< LiDAR-IMU transformation
    Eigen::Isometry3d T_world_lidar; ///< LiDAR pose in the world space
    Eigen::Isometry3d T_world_imu; ///< IMU pose in the world space

    Eigen::Vector3d v_world_imu; ///< IMU velocity in the world frame
    Eigen::Matrix<double, 6, 1> imu_bias; ///< IMU bias

    PreprocessedFrame::ConstPtr raw_frame; ///< Raw input point cloud (LiDAR frame)
    Eigen::Matrix<double, 8, -1> imu_rate_trajectory; ///< IMU-rate trajectory 8 x N  [t, x, y, z, qx, qy, qz, qw]

    // If true, this frame's deskewing relied on IMU measurements that saturated.
    // Used to avoid inserting heavily distorted scans into iVox.
    bool deskew_imu_saturated {false};
    std::array<bool, 3> deskew_acc_saturated_axes {false, false, false};
    std::array<bool, 3> deskew_gyro_saturated_axes {false, false, false};

    // If true, inter-scan IMU integration (between previous and this frame) detected saturation.
    bool interscan_imu_saturated {false};

    FrameType frame_type; ///< Coordinate center type of $frame
    gtsam_points::PointCloud::ConstPtr frame; ///< Deskewed points for state estimation
    std::vector<gtsam_points::GaussianVoxelMap::Ptr> voxelmaps; ///< Multi-resolution voxelmaps
};

}