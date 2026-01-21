#include <offline_mapping_optimizer/raycasting_filter.hpp>

#include <rclcpp/rclcpp.hpp>

#include <offline_mapping_optimizer/voxel_key.hpp>

#include <atomic>
#include <limits>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <rclcpp/logging.hpp>

#include <omp.h>

#if defined(MAP_OPTIMIZER_USE_CUDA)
    #include <offline_mapping_optimizer/raycasting_cuda.hpp>
#endif

namespace offline_mapping_optimizer {
namespace {

    // Amanatides & Woo style voxel traversal (3D DDA).
    // Traverses voxels from origin toward end_pt, excluding the end voxel.
    // For each traversed voxel that exists in voxel_to_index, increments the per-thread counter map.
    inline void traverse_voxels_dda(
        const Eigen::Vector3d& origin,
        const Eigen::Vector3d& end_pt,
        const double voxel_res,
        const double inv_voxel_res,
        const std::unordered_map<VoxelKey, int, VoxelKeyHash>& voxel_to_index,
        std::unordered_map<int, int>& local_counts
    ) {
        Eigen::Vector3d delta = end_pt - origin;
        const double distance = delta.norm();
        if (distance <= 1e-9)
            return;
        if (distance < voxel_res * 0.5)
            return;

        const Eigen::Vector3d dir = delta / distance;

        VoxelKey v = voxel_key_from_xyz(origin.x(), origin.y(), origin.z(), inv_voxel_res);
        const VoxelKey vend = voxel_key_from_xyz(end_pt.x(), end_pt.y(), end_pt.z(), inv_voxel_res);
        if (v == vend)
            return;

        const int step_x = (dir.x() > 0.0) ? 1 : ((dir.x() < 0.0) ? -1 : 0);
        const int step_y = (dir.y() > 0.0) ? 1 : ((dir.y() < 0.0) ? -1 : 0);
        const int step_z = (dir.z() > 0.0) ? 1 : ((dir.z() < 0.0) ? -1 : 0);

        const double inf = std::numeric_limits<double>::infinity();

        auto next_boundary = [&](int voxel_coord, int step) {
            return (step > 0) ? static_cast<double>(voxel_coord + 1) * voxel_res
                              : static_cast<double>(voxel_coord) * voxel_res;
        };

        double t_max_x = inf, t_max_y = inf, t_max_z = inf;
        double t_delta_x = inf, t_delta_y = inf, t_delta_z = inf;

        if (step_x != 0) {
            const double bx = next_boundary(v.x, step_x);
            t_max_x = (bx - origin.x()) / dir.x();
            t_delta_x = voxel_res / std::abs(dir.x());
        }
        if (step_y != 0) {
            const double by = next_boundary(v.y, step_y);
            t_max_y = (by - origin.y()) / dir.y();
            t_delta_y = voxel_res / std::abs(dir.y());
        }
        if (step_z != 0) {
            const double bz = next_boundary(v.z, step_z);
            t_max_z = (bz - origin.z()) / dir.z();
            t_delta_z = voxel_res / std::abs(dir.z());
        }

        while (!(v == vend)) {
            double t_next = 0.0;
            if (t_max_x < t_max_y) {
                if (t_max_x < t_max_z) {
                    t_next = t_max_x;
                    v.x += step_x;
                    t_max_x += t_delta_x;
                } else {
                    t_next = t_max_z;
                    v.z += step_z;
                    t_max_z += t_delta_z;
                }
            } else {
                if (t_max_y < t_max_z) {
                    t_next = t_max_y;
                    v.y += step_y;
                    t_max_y += t_delta_y;
                } else {
                    t_next = t_max_z;
                    v.z += step_z;
                    t_max_z += t_delta_z;
                }
            }

            if (t_next > distance - voxel_res)
                break;
            if (v == vend)
                break;

            auto it = voxel_to_index.find(v);
            if (it != voxel_to_index.end()) {
                local_counts[it->second] += 1;
            }
        }
    }

} // namespace

pcl::PointCloud<pcl::PointXYZ>::Ptr remove_dynamic_objects_raycasting(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& merged,
    const std::vector<gtsam::Pose3>& poses,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& pcl_frames,
    double voxel_resolution,
    int pass_through_threshold,
    int num_threads,
    bool use_cuda_raycasting,
    const rclcpp::Logger& logger
) {
    auto output = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (!merged || merged->empty()) {
        return output;
    }

    RCLCPP_INFO(logger, "Removing dynamic objects...");

    const double voxel_res = voxel_resolution;
    const double inv_voxel_res = 1.0 / voxel_res;

    std::unordered_map<VoxelKey, int, VoxelKeyHash> voxel_to_index;
    std::vector<VoxelKey> keys_by_index;

    // Build voxel index (parallel key generation -> sort/unique -> serial hash map build)
    {
        const int threads = std::max(1, num_threads);
        std::vector<VoxelKey> all_keys;
        all_keys.resize(merged->size());

#pragma omp parallel for num_threads(threads) schedule(static)
        for (int i = 0; i < static_cast<int>(merged->size()); i++) {
            const auto& pt = merged->points[i];
            all_keys[i] = voxel_key_from_xyz(pt.x, pt.y, pt.z, inv_voxel_res);
        }

        auto less_key = [](const VoxelKey& a, const VoxelKey& b) {
            if (a.x != b.x)
                return a.x < b.x;
            if (a.y != b.y)
                return a.y < b.y;
            return a.z < b.z;
        };

        std::sort(all_keys.begin(), all_keys.end(), less_key);
        all_keys.erase(std::unique(all_keys.begin(), all_keys.end()), all_keys.end());

        keys_by_index = std::move(all_keys);
        voxel_to_index.reserve(keys_by_index.size() + keys_by_index.size() / 3 + 1);
        for (int idx = 0; idx < static_cast<int>(keys_by_index.size()); idx++) {
            voxel_to_index.emplace(keys_by_index[idx], idx);
        }
    }

    std::vector<int> pass_through_counts(keys_by_index.size(), 0);

    bool used_gpu = false;
#if defined(MAP_OPTIMIZER_USE_CUDA)
    if (use_cuda_raycasting) {
        RCLCPP_INFO(logger, "Raycasting (CUDA) ...");

        std::vector<Float3> frame_origins;
        std::vector<float> frame_rotations_rowmajor;
        std::vector<Float3> points_local;
        std::vector<int> frame_offsets;

        frame_origins.resize(poses.size());
        frame_rotations_rowmajor.resize(poses.size() * 9);
        frame_offsets.resize(poses.size() + 1);
        frame_offsets[0] = 0;

        size_t total_points = 0;
        for (size_t i = 0; i < pcl_frames.size(); i++) {
            total_points += pcl_frames[i]->size();
            frame_offsets[i + 1] = static_cast<int>(total_points);
        }
        points_local.reserve(total_points);

        for (size_t i = 0; i < poses.size(); i++) {
            const gtsam::Pose3 pose = poses[i];
            const Eigen::Matrix3d R = pose.rotation().matrix();
            const auto t = pose.translation();
            frame_origins[i] = Float3 {static_cast<float>(t.x()), static_cast<float>(t.y()), static_cast<float>(t.z())};

            float* R9 = frame_rotations_rowmajor.data() + i * 9;
            R9[0] = static_cast<float>(R(0, 0));
            R9[1] = static_cast<float>(R(0, 1));
            R9[2] = static_cast<float>(R(0, 2));
            R9[3] = static_cast<float>(R(1, 0));
            R9[4] = static_cast<float>(R(1, 1));
            R9[5] = static_cast<float>(R(1, 2));
            R9[6] = static_cast<float>(R(2, 0));
            R9[7] = static_cast<float>(R(2, 1));
            R9[8] = static_cast<float>(R(2, 2));

            for (const auto& pt_local: *pcl_frames[i]) {
                points_local.push_back(Float3 {pt_local.x, pt_local.y, pt_local.z});
            }
        }

        std::string cuda_error;
        std::vector<int> cuda_counts;
        if (raycasting_cuda_compute_counts(
                keys_by_index,
                static_cast<float>(voxel_res),
                frame_origins,
                frame_rotations_rowmajor,
                points_local,
                frame_offsets,
                cuda_counts,
                &cuda_error
            ))
        {
            pass_through_counts = std::move(cuda_counts);
            used_gpu = true;
            RCLCPP_INFO(logger, "Raycasting (CUDA) done.");
        } else {
            RCLCPP_WARN(logger, "CUDA raycasting failed (%s). Falling back to CPU.", cuda_error.c_str());
        }
    }
#else
    if (use_cuda_raycasting) {
        RCLCPP_WARN(logger, "use_cuda_raycasting=true but CUDA support was not compiled. Falling back to CPU.");
    }
#endif

    if (!used_gpu) {
        const int threads = std::max(1, num_threads);
        std::vector<std::unordered_map<int, int>> thread_local_counts(threads);
        for (auto& m: thread_local_counts) {
            m.reserve(16384);
        }

        std::atomic<int> processed_frames = 0;

#pragma omp parallel for num_threads(threads) schedule(guided)
        for (size_t i = 0; i < poses.size(); i++) {
            const gtsam::Pose3 pose = poses[i];
            const Eigen::Matrix3d R = pose.rotation().matrix();
            const Eigen::Vector3d t(pose.translation().x(), pose.translation().y(), pose.translation().z());
            const Eigen::Vector3d origin = t;

            auto& local_counts = thread_local_counts[omp_get_thread_num()];
            for (const auto& pt_local: *pcl_frames[i]) {
                const Eigen::Vector3d pl(pt_local.x, pt_local.y, pt_local.z);
                const Eigen::Vector3d pt_global = R * pl + t;
                traverse_voxels_dda(origin, pt_global, voxel_res, inv_voxel_res, voxel_to_index, local_counts);
            }

            processed_frames.fetch_add(1);
            if (omp_get_thread_num() == 0) {
                RCLCPP_INFO(
                    logger,
                    "Raycasting progress: %.1f%% (%d / %zu)",
                    100.0 * processed_frames.load() / poses.size(),
                    processed_frames.load(),
                    poses.size()
                );
            }
        }

        for (const auto& local: thread_local_counts) {
            for (const auto& kv: local) {
                pass_through_counts[kv.first] += kv.second;
            }
        }
    }

    // Filter points
    {
        const int threads = std::max(1, num_threads);
        std::vector<std::vector<pcl::PointXYZ>> thread_local_points(threads);
        for (auto& v : thread_local_points) {
            v.reserve(std::max<size_t>(1024, merged->size() / (threads * 4 + 1)));
        }

#pragma omp parallel for num_threads(threads) schedule(guided)
        for (int i = 0; i < static_cast<int>(merged->size()); i++) {
            const auto& pt = merged->points[i];
            VoxelKey k = voxel_key_from_xyz(pt.x, pt.y, pt.z, inv_voxel_res);
            auto it = voxel_to_index.find(k);

            bool keep = false;
            if (it == voxel_to_index.end()) {
                keep = true;
            } else {
                keep = (pass_through_counts[it->second] <= pass_through_threshold);
            }

            if (keep) {
                thread_local_points[omp_get_thread_num()].push_back(pt);
            }
        }

        size_t total_kept = 0;
        for (const auto& v : thread_local_points) {
            total_kept += v.size();
        }

        output->points.clear();
        output->points.reserve(total_kept);
        for (auto& v : thread_local_points) {
            output->points.insert(output->points.end(), v.begin(), v.end());
        }
        output->width = static_cast<uint32_t>(output->points.size());
        output->height = 1;
        output->is_dense = true;
    }

    RCLCPP_INFO(logger, "Dynamic removal done. Kept %zu / %zu points", output->size(), merged->size());
    return output;
}

} // namespace offline_mapping_optimizer
