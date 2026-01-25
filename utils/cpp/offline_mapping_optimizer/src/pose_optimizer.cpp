#include <offline_mapping_optimizer/pose_optimizer.hpp>

#include <rclcpp/rclcpp.hpp>

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam_points/config.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>

#include <algorithm>
#include <cmath>
#include <omp.h>

using gtsam::symbol_shorthand::X;

namespace offline_mapping_optimizer {
namespace {
    gtsam::Key submap_key(int i) {
        return gtsam::Symbol('s', i);
    }

    struct Submap {
        int id = -1;
        int start = 0;   // inclusive
        int end = 0;     // exclusive
        int origin_index = 0;  // absolute frame index

        // Relative poses computed from local optimization
        std::vector<gtsam::Pose3> T_origin_frame;

        // Initial/optimized estimate of submap origin in world
        gtsam::Pose3 T_world_origin;

        // Submap representation in its own origin frame
        gtsam_points::PointCloudCPU::Ptr cloud;
        std::vector<std::shared_ptr<gtsam_points::GaussianVoxelMapCPU>> voxelmaps;
    };

    static std::vector<Submap> make_submaps(int num_frames, int submap_size) {
        std::vector<Submap> submaps;
        if (num_frames <= 0) {
            return submaps;
        }
        submap_size = std::max(1, submap_size);

        int id = 0;
        for (int start = 0; start < num_frames; start += submap_size) {
            const int end = std::min(num_frames, start + submap_size);
            if (end <= start) {
                break;
            }
            Submap sm;
            sm.id = id++;
            sm.start = start;
            sm.end = end;
            sm.origin_index = start + (end - start) / 2;
            submaps.push_back(std::move(sm));
            if (end == num_frames) {
                break;
            }
        }
        return submaps;
    }

    static std::vector<int> assign_frames_to_owner_submap(int num_frames, const std::vector<Submap>& submaps, int submap_size) {
        std::vector<int> owner(num_frames, -1);
        if (num_frames == 0 || submaps.empty()) {
            return owner;
        }

        // Ownership policy:
        // frame i is owned by the submap k such that i in [start_k, start_k + size),
        // except the last submap which owns up to its end.
        for (int k = 0; k < static_cast<int>(submaps.size()); k++) {
            const bool is_last = (k == static_cast<int>(submaps.size()) - 1);
            const int start = submaps[k].start;
            const int end = is_last ? submaps[k].end : std::min(submaps[k].end, start + std::max(1, submap_size));
            for (int i = start; i < end; i++) {
                if (i >= 0 && i < num_frames) {
                    owner[i] = k;
                }
            }
        }

        // Fill any gaps (can happen when submap_end < start+size due to tail).
        int last = 0;
        for (int i = 0; i < num_frames; i++) {
            if (owner[i] >= 0) {
                last = owner[i];
                continue;
            }
            owner[i] = last;
        }
        return owner;
    }

    static std::vector<std::shared_ptr<gtsam_points::GaussianVoxelMapCPU>>
    build_voxelmap_pyramid(
        const gtsam_points::PointCloudCPU::Ptr& cloud,
        double base_resolution,
        int levels,
        double scaling_factor
    ) {
        std::vector<std::shared_ptr<gtsam_points::GaussianVoxelMapCPU>> voxelmaps;
        if (!cloud || cloud->size() == 0 || levels <= 0) {
            return voxelmaps;
        }

        levels = std::max(1, levels);
        scaling_factor = std::max(1.0, scaling_factor);
        voxelmaps.reserve(levels);
        for (int l = 0; l < levels; l++) {
            const double res = base_resolution * std::pow(scaling_factor, l);
            auto vm = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(res);
            vm->insert(*cloud);
            voxelmaps.push_back(std::move(vm));
        }
        return voxelmaps;
    }

    // Build a submap cloud in the origin frame.
    // IMPORTANT: GaussianVoxelMap insertion requires per-point covariances.
    // Use gtsam_points::merge_frames which preserves points+covs.
    static gtsam_points::PointCloudCPU::Ptr build_submap_cloud(
        const std::vector<gtsam_points::PointCloudCPU::Ptr>& frames,
        int start,
        int end,
        const std::vector<gtsam::Pose3>& poses_world,
        const gtsam::Pose3& T_world_origin,
        int min_frame_points,
        double downsample_resolution
    ) {
        std::vector<Eigen::Isometry3d> poses;
        std::vector<gtsam_points::PointCloud::ConstPtr> clouds;

        poses.reserve(std::max(0, end - start));
        clouds.reserve(std::max(0, end - start));

        const gtsam::Pose3 T_origin_world = T_world_origin.inverse();
        for (int i = start; i < end; i++) {
            if (i < 0 || i >= static_cast<int>(frames.size())) {
                continue;
            }
            const auto& frame = frames[i];
            if (!frame || static_cast<int>(frame->size()) < min_frame_points) {
                continue;
            }
            if (!frame->has_covs()) {
                continue;
            }
            const gtsam::Pose3 T_origin_frame = T_origin_world * poses_world[i];
            poses.emplace_back(T_origin_frame.matrix());
            clouds.emplace_back(frame);
        }

        auto out = std::make_shared<gtsam_points::PointCloudCPU>();
        if (clouds.empty()) {
            return out;
        }

        const double ds = std::max(1e-3, downsample_resolution);
        auto merged = gtsam_points::merge_frames(poses, clouds, ds);
        if (!merged || merged->size() == 0) {
            return out;
        }

        auto merged_cpu = std::dynamic_pointer_cast<gtsam_points::PointCloudCPU>(merged);
        if (!merged_cpu) {
            merged_cpu = gtsam_points::PointCloudCPU::clone(*merged);
        }

        if (!merged_cpu->has_covs()) {
            std::vector<Eigen::Matrix4d> covs(merged_cpu->size(), Eigen::Matrix4d::Zero());
            for (auto& c : covs) {
                c.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 1e-3;
            }
            merged_cpu->add_covs(covs);
        }

        return merged_cpu;
    }

    static gtsam::noiseModel::Base::shared_ptr make_isotropic_precision6(double precision) {
        const double p = std::max(1e-9, precision);
        return gtsam::noiseModel::Isotropic::Precision(6, p);
    }

    static gtsam::noiseModel::Base::shared_ptr make_isotropic_sigma6(double sigma) {
        const double s = std::max(1e-9, sigma);
        return gtsam::noiseModel::Isotropic::Sigma(6, s);
    }
} // namespace

std::vector<gtsam::Pose3> optimize_poses_submap_graph(
    const std::vector<gtsam::Pose3>& initial_poses,
    const std::vector<gtsam_points::PointCloudCPU::Ptr>& frames,
    const OfflineMappingOptimizationParams& params,
    const rclcpp::Logger& logger
) {
    if (initial_poses.size() != frames.size()) {
        throw std::runtime_error("poses/frames size mismatch");
    }
    if (initial_poses.empty()) {
        return {};
    }

    const int num_frames = static_cast<int>(initial_poses.size());
    auto submaps = make_submaps(num_frames, params.local.submap_size);
    if (submaps.empty()) {
        return initial_poses;
    }
    const auto owner = assign_frames_to_owner_submap(num_frames, submaps, params.local.submap_size);

    // Stage 1) Local optimization per submap
    std::vector<gtsam::Pose3> local_poses_world = initial_poses;

    RCLCPP_INFO(logger, "Local stage: %zu submaps", submaps.size());

    for (auto& sm : submaps) {
        const int start = sm.start;
        const int end = sm.end;
        const int m = end - start;
        if (m <= 0) {
            continue;
        }

        gtsam::NonlinearFactorGraph graph;
        gtsam::Values values;
        for (int i = 0; i < m; i++) {
            values.insert(X(i), local_poses_world[start + i]);
        }

        // Fix first frame in the submap
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            X(0),
            local_poses_world[start],
            make_isotropic_precision6(params.local.first_frame_prior_precision)
        );

        // Odometry between factors from initial estimate
        if (params.local.enable_odometry_between) {
            auto noise = make_isotropic_sigma6(params.local.odom_between_sigma);
            for (int i = 1; i < m; i++) {
                const gtsam::Pose3 delta = local_poses_world[start + i - 1].inverse() * local_poses_world[start + i];
                graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(i - 1), X(i), delta, noise);
            }
        }

        // VGICP registration error factors (keyframe -> other frames)
        int vgicp_factor_count = 0;
        if (params.local.enable_vgicp_factors) {
            const int key_stride = std::max(1, params.local.keyframe_stride);
            const int max_pairs = params.local.max_vgicp_pairs_per_keyframe > 0 ? params.local.max_vgicp_pairs_per_keyframe : std::numeric_limits<int>::max();
            const bool use_overlap_gate = params.local.min_overlap > 0.0;
            const int num_threads = std::max(1, params.local.vgicp_num_threads);

            for (int ki = 0; ki < m; ki += key_stride) {
                const int k_abs = start + ki;
                if (!frames[k_abs] || static_cast<int>(frames[k_abs]->size()) < params.local.min_frame_points) {
                    continue;
                }

                auto k_voxelmaps = build_voxelmap_pyramid(
                    frames[k_abs],
                    params.local.voxel_resolution,
                    params.local.voxelmap_levels,
                    params.local.voxelmap_scaling_factor
                );
                if (k_voxelmaps.empty()) {
                    continue;
                }

                int added = 0;
                for (int j = ki + 1; j < m; j++) {
                    if (max_pairs > 0 && added >= max_pairs) {
                        break;
                    }
                    const int j_abs = start + j;
                    if (!frames[j_abs] || static_cast<int>(frames[j_abs]->size()) < params.local.min_frame_points) {
                        continue;
                    }

                    if (use_overlap_gate) {
                        const gtsam::Pose3 T_k_j_init = local_poses_world[k_abs].inverse() * local_poses_world[j_abs];
                        const double ov = gtsam_points::overlap(k_voxelmaps.back(), frames[j_abs], Eigen::Isometry3d(T_k_j_init.matrix()));
                        if (ov < params.local.min_overlap) {
                            continue;
                        }
                    }

                    for (const auto& vm : k_voxelmaps) {
                        if (!vm || !vm->has_points()) {
                            continue;
                        }
                        // Share the same voxelmap instance across factors to avoid huge per-edge memory overhead.
                        auto factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(X(ki), X(j), vm, frames[j_abs]);
                        factor->set_num_threads(num_threads);
                        graph.add(factor);
                        vgicp_factor_count++;
                    }
                    added++;
                }
            }
        }

        if (params.local.enable_optimization) {
            RCLCPP_INFO(logger, "Local submap %d [%d,%d): factors=%zu (vgicp=%d), optimizing...", sm.id, start, end, graph.size(), vgicp_factor_count);
            try {
                gtsam::LevenbergMarquardtParams lm;
                lm.setMaxIterations(std::max(1, params.local.max_iterations));
                gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm);
                gtsam::Values result = optimizer.optimize();
                for (int i = 0; i < m; i++) {
                    local_poses_world[start + i] = result.at<gtsam::Pose3>(X(i));
                }
            } catch (const gtsam::IndeterminantLinearSystemException& e) {
                RCLCPP_WARN(logger, "Local submap %d optimization failed (indeterminant). Keeping initial poses. nearby=%s", sm.id, std::string(gtsam::Symbol(e.nearbyVariable())).c_str());
            } catch (const std::exception& e) {
                RCLCPP_WARN(logger, "Local submap %d optimization failed (%s). Keeping initial poses.", sm.id, e.what());
            }
        }

        // Prepare submap state for global stage
        sm.T_world_origin = local_poses_world[sm.origin_index];
        sm.T_origin_frame.resize(m);
        const gtsam::Pose3 T_origin_world = sm.T_world_origin.inverse();
        for (int i = 0; i < m; i++) {
            sm.T_origin_frame[i] = T_origin_world * local_poses_world[start + i];
        }

        sm.cloud = build_submap_cloud(
            frames,
            start,
            end,
            local_poses_world,
            sm.T_world_origin,
            params.local.min_frame_points,
            params.global.voxel_resolution
        );
        sm.voxelmaps = build_voxelmap_pyramid(
            sm.cloud,
            params.global.voxel_resolution,
            params.global.voxelmap_levels,
            params.global.voxelmap_scaling_factor
        );
    }

    // Stage 2) Global submap graph optimization
    RCLCPP_INFO(logger, "Global stage: building submap graph...");
    gtsam::NonlinearFactorGraph global_graph;
    gtsam::Values global_values;

    for (int i = 0; i < static_cast<int>(submaps.size()); i++) {
        global_values.insert(submap_key(i), submaps[i].T_world_origin);
    }
    global_graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        submap_key(0),
        submaps[0].T_world_origin,
        make_isotropic_precision6(params.global.first_submap_prior_precision)
    );

    // Adjacent edges
    if (params.global.enable_odometry_between) {
        auto noise = make_isotropic_sigma6(params.global.odom_between_sigma);
        for (int i = 1; i < static_cast<int>(submaps.size()); i++) {
            const gtsam::Pose3 delta = submaps[i - 1].T_world_origin.inverse() * submaps[i].T_world_origin;
            global_graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(submap_key(i - 1), submap_key(i), delta, noise);
        }
    }

    int adjacent_vgicp_edges = 0;
    if (params.global.enable_adjacent_vgicp) {
        const int threads = std::max(1, params.global.vgicp_num_threads);
        for (int i = 1; i < static_cast<int>(submaps.size()); i++) {
            const auto& target = submaps[i - 1];
            const auto& source = submaps[i];
            if (target.voxelmaps.empty() || !target.voxelmaps.back() || !target.voxelmaps.back()->has_points()) {
                continue;
            }
            if (!source.cloud || source.cloud->size() == 0) {
                continue;
            }
            for (const auto& vm : target.voxelmaps) {
                if (!vm || !vm->has_points()) continue;
                // Avoid deep-copying voxelmaps per edge; factors can share the same read-only target voxelmap.
                auto factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(submap_key(i - 1), submap_key(i), vm, source.cloud);
                factor->set_num_threads(threads);
                global_graph.add(factor);
            }
            adjacent_vgicp_edges++;
        }
    }

    // Loop edges
    int loop_edges = 0;
    if (params.global.enable_loop_closures) {
        const int threads = std::max(1, params.global.vgicp_num_threads);
        const double max_dist2 = params.global.max_loop_distance * params.global.max_loop_distance;
        const int max_edges_per = params.global.max_loop_edges_per_submap > 0 ? params.global.max_loop_edges_per_submap : std::numeric_limits<int>::max();

        std::vector<int> edges_added(submaps.size(), 0);
        for (int i = 0; i < static_cast<int>(submaps.size()); i++) {
            for (int j = i + 2; j < static_cast<int>(submaps.size()); j++) {
                if (max_edges_per > 0 && edges_added[j] >= max_edges_per) {
                    continue;
                }
                const Eigen::Vector3d dt = submaps[i].T_world_origin.translation() - submaps[j].T_world_origin.translation();
                if (dt.squaredNorm() > max_dist2) {
                    continue;
                }
                const auto& target = submaps[i];
                const auto& source = submaps[j];
                if (target.voxelmaps.empty() || !target.voxelmaps.back() || !target.voxelmaps.back()->has_points()) {
                    continue;
                }
                if (!source.cloud || source.cloud->size() == 0) {
                    continue;
                }

                const gtsam::Pose3 T_i_j_init = target.T_world_origin.inverse() * source.T_world_origin;
                const double ov = gtsam_points::overlap(target.voxelmaps.back(), source.cloud, Eigen::Isometry3d(T_i_j_init.matrix()));
                if (ov < params.global.min_loop_overlap) {
                    continue;
                }

                for (const auto& vm : target.voxelmaps) {
                    if (!vm || !vm->has_points()) continue;
                    // Avoid deep-copying voxelmaps per edge; factors can share the same read-only target voxelmap.
                    auto factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(submap_key(i), submap_key(j), vm, source.cloud);
                    factor->set_num_threads(threads);
                    global_graph.add(factor);
                }
                edges_added[j]++;
                loop_edges++;
            }
        }
    }

    RCLCPP_INFO(
        logger,
        "Global graph: nodes=%zu factors=%zu (adj_vgicp=%d loop=%d).",
        submaps.size(),
        global_graph.size(),
        adjacent_vgicp_edges,
        loop_edges
    );

    gtsam::Values global_result = global_values;
    if (params.global.enable_optimization) {
        try {
            gtsam::LevenbergMarquardtParams lm;
            lm.setMaxIterations(std::max(1, params.global.max_iterations));
            gtsam::LevenbergMarquardtOptimizer optimizer(global_graph, global_values, lm);
            global_result = optimizer.optimize();
        } catch (const gtsam::IndeterminantLinearSystemException& e) {
            RCLCPP_WARN(logger, "Global optimization failed (indeterminant). Keeping initial submap poses. nearby=%s", std::string(gtsam::Symbol(e.nearbyVariable())).c_str());
            global_result = global_values;
        } catch (const std::exception& e) {
            RCLCPP_WARN(logger, "Global optimization failed (%s). Keeping initial submap poses.", e.what());
            global_result = global_values;
        }
    }

    // Stage 3) Compose final per-frame poses from global submap origins + local relative poses.
    std::vector<gtsam::Pose3> final_poses(num_frames);
    for (int i = 0; i < num_frames; i++) {
        final_poses[i] = initial_poses[i];
    }

    for (int k = 0; k < static_cast<int>(submaps.size()); k++) {
        const auto& sm = submaps[k];
        const gtsam::Pose3 T_world_origin_opt = global_result.at<gtsam::Pose3>(submap_key(k));
        for (int i = sm.start; i < sm.end; i++) {
            if (i < 0 || i >= num_frames) continue;
            if (owner[i] != k) continue;
            const int local_i = i - sm.start;
            if (local_i < 0 || local_i >= static_cast<int>(sm.T_origin_frame.size())) continue;
            final_poses[i] = T_world_origin_opt * sm.T_origin_frame[local_i];
        }
    }

    return final_poses;
}

} // namespace offline_mapping_optimizer