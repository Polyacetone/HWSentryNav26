#pragma once

#include <rclcpp/logger.hpp>

#include <gtsam/geometry/Pose3.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <vector>

namespace offline_mapping_optimizer {

pcl::PointCloud<pcl::PointXYZ>::Ptr remove_dynamic_objects_raycasting(
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& merged,
  const std::vector<gtsam::Pose3>& poses,
  const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& pcl_frames,
  double voxel_resolution,
  int pass_through_threshold,
  int num_threads,
  bool use_cuda_raycasting,
  const rclcpp::Logger& logger
);

}  // namespace offline_mapping_optimizer
