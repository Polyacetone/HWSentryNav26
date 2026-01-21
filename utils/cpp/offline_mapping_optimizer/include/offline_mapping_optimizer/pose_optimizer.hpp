#pragma once

#include <rclcpp/logger.hpp>

#include <gtsam/geometry/Pose3.h>

#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <vector>

namespace offline_mapping_optimizer {

struct PoseOptimizerParams {
  int num_threads;

  // LM iterations per outer iteration
  int max_iterations;

  // Outer iterations: build global map -> optimize -> rebuild map ...
  int outer_iterations;

  // Skip frames with too few points (sparse frames are unstable)
  int min_frame_points;

  // Global map voxel resolution used for overlap gating + GICP-to-map target (iVox)
  double map_voxel_resolution;

  // Add a frame-to-map registration factor only if overlap >= threshold.
  // Set <= 0 to disable this gating.
  double map_overlap_threshold;

  // Optional: add pairwise (frame-frame) GICP factors only if overlap >= threshold
  bool enable_pairwise_factors;
  double pairwise_voxel_resolution;
  double pairwise_overlap_threshold;

  // Loop candidate distance threshold (in initial/updated pose space)
  double loop_dist_thres;
  int max_loops_per_frame;  // 0 means unlimited

  double gicp_max_correspondence_distance;
};

// Iterative offline optimization:
// 1) Build a global voxel map from current poses.
// 2) For each frame, add a frame-to-map registration factor (GICP-to-map) if overlap is high.
// 3) Optionally add pairwise frame-frame GICP factors (also overlap-gated).
// 4) Optimize all poses and repeat.
std::vector<gtsam::Pose3> optimize_poses_iterative(
  const std::vector<gtsam::Pose3>& initial_poses,
  const std::vector<gtsam_points::PointCloudCPU::Ptr>& frames,
  const PoseOptimizerParams& params,
  const rclcpp::Logger& logger
);

}  // namespace offline_mapping_optimizer
