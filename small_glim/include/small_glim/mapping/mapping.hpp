#pragma once

#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <small_glim/common/config.hpp>
#include <small_glim/odometry/estimation_frame.hpp>

namespace small_glim {

struct MappingParams {
    explicit MappingParams(const Config::Ptr config);

    int num_threads; // Number of threads for registration
    bool enable_optimization; // Enable pose graph optimization
    double downsample_resolution; // Downsample resolution when saving

    enum class KeyframeUpdateStrategy { DISPLACEMENT, OVERLAP } keyframe_update_strategy; // "DISPLACEMENT" or "OVERLAP"
    double keyframe_update_min_dist; // Minimum distance for keyframe update (DISPLACEMENT)
    double keyframe_update_min_rot; // Minimum rotation for keyframe update (DISPLACEMENT)
    double max_keyframe_overlap; // Maximum overlap for keyframe update (OVERLAP)
    
    enum class RegistrationType { GICP, VGICP } registration_type; // "GICP" or "VGICP"
    double keyframe_randomsampling_rate; // Random sampling rate for keyframes
    double max_correspondence_distance; // Maximum correspondence distance (GICP)
    double keyframe_voxel_resolution; // Voxel resolution for keyframes (VGICP)
    int keyframe_voxelmap_levels; // Multi-resolution voxelmap levels (VGICP)
    double keyframe_voxelmap_scaling_factor; // Multi-resolution voxelmap scaling factor (VGICP)
};

class Mapping {
public:
    using Ptr = std::shared_ptr<Mapping>;

    explicit Mapping(const Config::Ptr config);
    void insert_frame(const EstimationFrame::ConstPtr& frame);
    void save(const std::string& path);
    void save_raw_frames(const std::string& dir);

private:
    bool is_keyframe(const EstimationFrame::ConstPtr& frame);
    void add_keyframe(const EstimationFrame::ConstPtr& frame);
    void optimize();

    std::unique_ptr<MappingParams> params;
    std::mt19937 mt;

    std::mutex mutex;
    std::vector<EstimationFrame::ConstPtr> keyframes;
    std::vector<int> keyframe_indices;
    
    std::unique_ptr<gtsam::NonlinearFactorGraph> graph;
    std::unique_ptr<gtsam::Values> values;

    // For DISPLACEMENT strategy
    Eigen::Isometry3d last_keyframe_pose;
};

}
