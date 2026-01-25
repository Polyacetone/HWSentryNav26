#pragma once

#include <rclcpp/logger.hpp>

#include <gtsam/geometry/Pose3.h>

#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <vector>

namespace offline_mapping_optimizer {

struct LocalSubmapParams {
    // Split frames into submaps by a sliding window: [start, start + submap_size)
    int submap_size;

    // Fix the first frame pose in each submap to its initial estimate.
    double first_frame_prior_precision;

    // Odometry-based BetweenFactor between consecutive frames.
    bool enable_odometry_between;
    double odom_between_sigma; // isotropic sigma in se(3) tangent

    // VGICP registration error factors inside each submap.
    bool enable_vgicp_factors;
    int keyframe_stride; // 1 = use every frame as keyframe
    int max_vgicp_pairs_per_keyframe; // 0 = unlimited
    int vgicp_num_threads;

    // Voxelmap pyramid used as VGICP target.
    double voxel_resolution;
    int voxelmap_levels;
    double voxelmap_scaling_factor;

    // Optional gating: add a VGICP factor only when overlap is high.
    // Set <= 0 to disable.
    double min_overlap;

    // Skip frames with too few points.
    int min_frame_points;

    // Local optimizer settings
    bool enable_optimization;
    int max_iterations;
};

struct GlobalSubmapGraphParams {
    bool enable_optimization;
    int max_iterations;

    // Fix the first submap origin pose.
    double first_submap_prior_precision;

    // Adjacent submap VGICP alignment.
    bool enable_adjacent_vgicp;
    int vgicp_num_threads;

    // Optional odometry-like BetweenFactor between adjacent submaps (derived from initial submap origins).
    bool enable_odometry_between;
    double odom_between_sigma;

    // Loop candidates and gating.
    bool enable_loop_closures;
    double max_loop_distance;
    double min_loop_overlap;
    int max_loop_edges_per_submap; // 0 = unlimited

    // Voxelmap pyramid for each submap (built from its merged local map in submap frame).
    double voxel_resolution;
    int voxelmap_levels;
    double voxelmap_scaling_factor;
};

struct OfflineMappingOptimizationParams {
    int num_threads;
    LocalSubmapParams local;
    GlobalSubmapGraphParams global;
};

// Two-stage offline mapping optimization:
// 1) Local: split frames into submaps, add prior + odometry between + optional intra-submap VGICP, optimize each submap.
// 2) Global: treat each optimized submap as a node, add adjacent + loop VGICP (and optional odom between), optimize submap origins.
// Returns optimized poses for ALL original frames.
std::vector<gtsam::Pose3> optimize_poses_submap_graph(
    const std::vector<gtsam::Pose3>& initial_poses,
    const std::vector<gtsam_points::PointCloudCPU::Ptr>& frames,
    const OfflineMappingOptimizationParams& params,
    const rclcpp::Logger& logger
);

} // namespace offline_mapping_optimizer