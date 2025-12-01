#pragma once

#include <random>
#include <memory>
#include <small_glim/mapping/sub_map.hpp>
#include <small_glim/odometry/estimation_frame.hpp>
#include <small_glim/odometry/imu_integration.hpp>
#include <small_glim/preprocess/cloud_deskewing.hpp>
#include <small_glim/preprocess/cloud_covariance_estimation.hpp>

namespace gtsam {
class Values;
class NonlinearFactorGraph;
class PreintegratedImuMeasurements;
}

namespace small_glim {

/**
* @brief Sub mapping parameters
*/
struct SubMappingParams {
public:
    explicit SubMappingParams(const Config::Ptr config);

public:
    int num_threads;
    bool enable_imu;
    bool enable_optimization;
    // Keyframe update strategy params
    int max_num_keyframes;
    std::string keyframe_update_strategy;
    int keyframe_update_min_points;
    double keyframe_update_interval_rot;
    double keyframe_update_interval_trans;
    double max_keyframe_overlap;

    bool create_between_factors;
    std::string between_registration_type;

    std::string registration_error_factor_type;
    double keyframe_randomsampling_rate;
    double keyframe_voxel_resolution;
    int keyframe_voxelmap_levels;
    double keyframe_voxelmap_scaling_factor;

    double submap_downsample_resolution;
    int submap_target_num_points;
};

/**
* @brief Sub mapping
*/
class SubMapping {
public:
    explicit SubMapping(const Config::Ptr config);

    void insert_imu(
        const double stamp,
        const Eigen::Vector3d& linear_acc,
        const Eigen::Vector3d& angular_vel
    );
    void insert_frame(const EstimationFrame::ConstPtr& odom_frame);
    std::vector<SubMap::Ptr> get_submaps();
    std::vector<SubMap::Ptr> submit_end_of_sequence();

private:
    void insert_keyframe(const int current, const EstimationFrame::ConstPtr& odom_frame);
    SubMap::Ptr create_submap(bool force_create = false) const;

private:
    std::unique_ptr<SubMappingParams> params;

    std::mt19937 mt;
    int submap_count;

    std::unique_ptr<IMUIntegration> imu_integration;
    std::unique_ptr<CloudDeskewing> deskewing;
    std::unique_ptr<CloudCovarianceEstimation> covariance_estimation;

    std::deque<EstimationFrame::ConstPtr> delayed_input_queue;
    std::vector<EstimationFrame::ConstPtr> odom_frames;

    std::vector<int> keyframe_indices;
    std::vector<EstimationFrame::Ptr> keyframes;

    std::unique_ptr<gtsam::Values> values;
    std::unique_ptr<gtsam::NonlinearFactorGraph> graph;

    std::vector<SubMap::Ptr> submap_queue;
};

} // namespace glim
