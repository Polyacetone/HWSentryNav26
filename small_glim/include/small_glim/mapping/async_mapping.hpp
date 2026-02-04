#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <Eigen/Dense>

#include <small_glim/common/config.hpp>
#include <small_glim/odometry/estimation_frame.hpp>

namespace pcl {
template <typename PointT>
class PointCloud;
struct PointXYZ;
}

namespace small_glim {

struct AsyncMappingParams {
	bool save_raw_mapping_frames;
	std::string output_root;

	bool enable_imu_refine;
	int imu_refine_max_iterations;

	double keyframe_trans_thresh;
	double keyframe_rot_thresh;

	double map_voxel_leaf_size;
	int downsample_every_n_keyframes;

	explicit AsyncMappingParams(const Config::Ptr& config);
};

class AsyncMapping {
public:
	explicit AsyncMapping(const Config::Ptr& config);
	~AsyncMapping();

	AsyncMapping(const AsyncMapping&) = delete;
	AsyncMapping& operator=(const AsyncMapping&) = delete;

	void insert_frame(const EstimationFrame::ConstPtr& frame);
	void request_finish();
	void join();

	const std::string& output_directory() const { return output_dir_; }

private:
	void worker_loop();
	void process_frame(const EstimationFrame& frame, const EstimationFrame* next_frame);

	bool is_keyframe(const EstimationFrame& frame) const;
	void accept_keyframe(const EstimationFrame& frame, const pcl::PointCloud<pcl::PointXYZ>& keyframe_cloud_imu);

	void ensure_output_dir();
	std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> build_keyframe_cloud_imu(
		const EstimationFrame& frame,
		const EstimationFrame* next_frame
	);

	void save_keyframe_raw_cloud(const pcl::PointCloud<pcl::PointXYZ>& cloud_imu) const;
	void append_pose(const Eigen::Isometry3d& T_world_imu);
	void integrate_cloud_into_map(const pcl::PointCloud<pcl::PointXYZ>& cloud_imu, const Eigen::Isometry3d& T_world_imu);
	void downsample_map_if_needed();
	void save_final_map();

	static std::string make_timestamp_string();
	static double rotation_angle_deg(const Eigen::Matrix3d& R);

private:
	AsyncMappingParams params_;
	std::string output_dir_;

	std::atomic<bool> finish_requested_{false};
	std::atomic<bool> joined_{false};

	std::mutex mutex_;
	std::condition_variable cv_;
	std::deque<EstimationFrame::ConstPtr> queue_;
	std::thread worker_;

	bool has_last_keyframe_{false};
	Eigen::Isometry3d last_keyframe_T_world_frame_{Eigen::Isometry3d::Identity()};
	std::size_t keyframe_count_{0};

	std::ofstream poses_ofs_;

	std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> map_cloud_;
	size_t keyframes_since_downsample_{0};
};

}