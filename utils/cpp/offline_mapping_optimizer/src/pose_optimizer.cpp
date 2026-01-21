#include <offline_mapping_optimizer/pose_optimizer.hpp>

#include <rclcpp/rclcpp.hpp>

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/types/point_cloud.hpp>

#include <atomic>

#include <algorithm>
#include <omp.h>

using gtsam::symbol_shorthand::X;

namespace offline_mapping_optimizer {
namespace {
    std::vector<std::shared_ptr<gtsam_points::GaussianVoxelMapCPU>>
    build_local_voxelmaps(const std::vector<gtsam_points::PointCloudCPU::Ptr>& frames, const double resolution) {
        std::vector<std::shared_ptr<gtsam_points::GaussianVoxelMapCPU>> voxelmaps;
        voxelmaps.resize(frames.size());

#pragma omp parallel for schedule(guided, 4)
        for (int i = 0; i < static_cast<int>(frames.size()); i++) {
            auto vm = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(resolution);
            if (frames[i] && frames[i]->size() > 0) {
                vm->insert(*frames[i]);
            }
            voxelmaps[i] = std::move(vm);
        }

        return voxelmaps;
    }

    double symmetric_overlap_between_frames(
        const std::shared_ptr<gtsam_points::GaussianVoxelMapCPU>& target_i,
        const gtsam_points::PointCloudCPU::Ptr& frame_i,
        const std::shared_ptr<gtsam_points::GaussianVoxelMapCPU>& target_j,
        const gtsam_points::PointCloudCPU::Ptr& frame_j,
        const gtsam::Pose3& T_w_i,
        const gtsam::Pose3& T_w_j
    ) {
        if (!target_i || !target_j || !frame_i || !frame_j) {
            return 0.0;
        }
        if (!target_i->has_points() || !target_j->has_points()) {
            return 0.0;
        }

        const Eigen::Isometry3d T_i_j((T_w_i.inverse() * T_w_j).matrix());
        const Eigen::Isometry3d T_j_i((T_w_j.inverse() * T_w_i).matrix());

        const double o_i_j = gtsam_points::overlap(target_i, frame_j, T_i_j);
        const double o_j_i = gtsam_points::overlap(target_j, frame_i, T_j_i);
        return 0.5 * (o_i_j + o_j_i);
    }

    std::shared_ptr<gtsam_points::GaussianVoxelMapCPU> build_global_voxelmap(
        const std::vector<gtsam::Pose3>& poses,
        const std::vector<gtsam_points::PointCloudCPU::Ptr>& frames,
        const double resolution,
        const int parity_keep // 0 => even only, 1 => odd only
    ) {
        auto map = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(resolution);

        for (size_t i = 0; i < poses.size(); i++) {
            if (static_cast<int>(i % 2) != parity_keep) {
                continue;
            }
            if (!frames[i] || frames[i]->size() == 0) {
                continue;
            }
            const Eigen::Isometry3d T_w_i(poses[i].matrix());
            auto transformed = gtsam_points::transform(frames[i], T_w_i);
            map->insert(*transformed);
        }

        return map;
    }

    std::shared_ptr<gtsam_points::iVox> build_global_ivox(
        const std::vector<gtsam::Pose3>& poses,
        const std::vector<gtsam_points::PointCloudCPU::Ptr>& frames,
        const double resolution,
        const int parity_keep // 0 => even only, 1 => odd only
    ) {
        auto target = std::make_shared<gtsam_points::iVox>(resolution);
        target->set_neighbor_voxel_mode(1);

        for (size_t i = 0; i < poses.size(); i++) {
            if (static_cast<int>(i % 2) != parity_keep) {
                continue;
            }
            if (!frames[i] || frames[i]->size() == 0) {
                continue;
            }
            const Eigen::Isometry3d T_w_i(poses[i].matrix());
            auto transformed = gtsam_points::transform(frames[i], T_w_i);
            target->insert(*transformed);
        }

        return target;
    }
} // namespace

std::vector<gtsam::Pose3> optimize_poses_iterative(
    const std::vector<gtsam::Pose3>& initial_poses,
    const std::vector<gtsam_points::PointCloudCPU::Ptr>& frames,
    const PoseOptimizerParams& params,
    const rclcpp::Logger& logger
) {
    if (initial_poses.size() != frames.size()) {
        throw std::runtime_error("poses/frames size mismatch");
    }
    if (initial_poses.empty()) {
        return {};
    }

    std::vector<gtsam::Pose3> poses = initial_poses;

    // Prebuild per-frame voxelmaps for overlap gating of pairwise factors.
    std::vector<std::shared_ptr<gtsam_points::GaussianVoxelMapCPU>> local_voxelmaps;
    if (params.enable_pairwise_factors) {
        local_voxelmaps = build_local_voxelmaps(frames, params.pairwise_voxel_resolution);
    }

    const auto prior_noise = gtsam::noiseModel::Isotropic::Precision(6, 1);

    const int outer_iters = std::max(1, params.outer_iterations);
    for (int outer = 0; outer < outer_iters; outer++) {
        // Parity split: for frame i, use the opposite-parity global map as target
        // - GaussianVoxelMap is used for overlap gating (fast overlap query)
        // - iVox is used as the GICP target structure
        auto map_even = build_global_voxelmap(poses, frames, params.map_voxel_resolution, 0);
        auto map_odd = build_global_voxelmap(poses, frames, params.map_voxel_resolution, 1);
        auto ivox_even = build_global_ivox(poses, frames, params.map_voxel_resolution, 0);
        auto ivox_odd = build_global_ivox(poses, frames, params.map_voxel_resolution, 1);

        gtsam::NonlinearFactorGraph graph;
        gtsam::Values values;

        for (size_t i = 0; i < poses.size(); i++) {
            values.insert(X(i), poses[i]);
        }
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), poses[0], prior_noise));

        const int threads = std::max(1, params.num_threads);
        const bool use_map_overlap_gate = (params.map_overlap_threshold > 0.0);
        const bool use_pair_overlap_gate = params.enable_pairwise_factors && (params.pairwise_overlap_threshold > 0.0);

        std::vector<std::vector<gtsam::NonlinearFactor::shared_ptr>> thread_local_factors(threads);
        for (auto& v : thread_local_factors) {
            v.reserve(2048);
        }

        std::atomic<int> map_factor_count = 0;
        std::atomic<int> pair_factor_count = 0;
        std::atomic<int> loop_factor_count = 0;

        // Frame-to-global-map factors (GICP-to-map), optional overlap-gated
#pragma omp parallel for num_threads(threads) schedule(guided)
        for (int i = 0; i < static_cast<int>(poses.size()); i++) {
            if (!frames[i] || static_cast<int>(frames[i]->size()) < params.min_frame_points) {
                continue;
            }

            const bool i_is_even = (i % 2 == 0);
            const auto& gate_target = i_is_even ? map_odd : map_even;
            const auto& reg_target = i_is_even ? ivox_odd : ivox_even;

            if (!reg_target || !reg_target->has_points()) {
                continue;
            }
            if (use_map_overlap_gate) {
                if (!gate_target || !gate_target->has_points()) {
                    continue;
                }
                const Eigen::Isometry3d T_w_i(poses[i].matrix());
                const double ov = gtsam_points::overlap(gate_target, frames[i], T_w_i);
                if (ov < params.map_overlap_threshold) {
                    continue;
                }
            }

            // Clone target to keep factor cost stable
            auto reg_target_copy = std::make_shared<gtsam_points::iVox>(*reg_target);
            auto factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor_<gtsam_points::iVox, gtsam_points::PointCloud>>(
                gtsam::Pose3(),
                X(i),
                reg_target_copy,
                frames[i],
                reg_target_copy
            );
            factor->set_num_threads(params.num_threads);
            factor->set_max_correspondence_distance(params.gicp_max_correspondence_distance);
            thread_local_factors[omp_get_thread_num()].push_back(factor);
            map_factor_count.fetch_add(1, std::memory_order_relaxed);
        }

        // Optional: overlap-gated pairwise GICP factors
        if (params.enable_pairwise_factors) {
            // Sequential edges
#pragma omp parallel for num_threads(threads) schedule(guided)
            for (int i = 1; i < static_cast<int>(poses.size()); i++) {
                if (!frames[i - 1] || !frames[i])
                    continue;
                if (static_cast<int>(frames[i - 1]->size()) < params.min_frame_points)
                    continue;
                if (static_cast<int>(frames[i]->size()) < params.min_frame_points)
                    continue;

                if (use_pair_overlap_gate) {
                    const double ov = symmetric_overlap_between_frames(
                        local_voxelmaps[i - 1],
                        frames[i - 1],
                        local_voxelmaps[i],
                        frames[i],
                        poses[i - 1],
                        poses[i]
                    );
                    if (ov < params.pairwise_overlap_threshold) {
                        continue;
                    }
                }

                auto factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(X(i - 1), X(i), frames[i - 1], frames[i]);
                factor->set_num_threads(params.num_threads);
                factor->set_max_correspondence_distance(params.gicp_max_correspondence_distance);
                thread_local_factors[omp_get_thread_num()].push_back(factor);
                pair_factor_count.fetch_add(1, std::memory_order_relaxed);
            }

            // Loop closures (distance + optional overlap gating)
            if (params.loop_dist_thres > 0.0) {
#pragma omp parallel for num_threads(threads) schedule(guided)
                for (int i = 0; i < static_cast<int>(poses.size()); i++) {
                    int added_for_i = 0;
                    for (int j = i + 2; j < static_cast<int>(poses.size()); j++) {
                        if (params.max_loops_per_frame > 0 && added_for_i >= params.max_loops_per_frame) {
                            break;
                        }
                        if (!frames[i] || !frames[j])
                            continue;
                        if (static_cast<int>(frames[i]->size()) < params.min_frame_points)
                            continue;
                        if (static_cast<int>(frames[j]->size()) < params.min_frame_points)
                            continue;

                        const double dist = (poses[i].translation() - poses[j].translation()).norm();
                        if (dist > params.loop_dist_thres) {
                            continue;
                        }

                        if (use_pair_overlap_gate) {
                            const double ov = symmetric_overlap_between_frames(
                                local_voxelmaps[i],
                                frames[i],
                                local_voxelmaps[j],
                                frames[j],
                                poses[i],
                                poses[j]
                            );
                            if (ov < params.pairwise_overlap_threshold) {
                                continue;
                            }
                        }

                        auto factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(X(i), X(j), frames[i], frames[j]);
                        factor->set_num_threads(params.num_threads);
                        factor->set_max_correspondence_distance(params.gicp_max_correspondence_distance);
                        thread_local_factors[omp_get_thread_num()].push_back(factor);
                        loop_factor_count.fetch_add(1, std::memory_order_relaxed);
                        added_for_i++;
                    }
                }
            }
        }

        // Merge into graph (serial)
        for (const auto& v : thread_local_factors) {
            for (const auto& f : v) {
                graph.add(f);
            }
        }

        RCLCPP_INFO(
            logger,
            "Iter %d/%d: factors(map=%d, seq=%d, loop=%d, total=%zu), optimizing...",
            outer + 1,
            outer_iters,
            map_factor_count.load(),
            pair_factor_count.load(),
            loop_factor_count.load(),
            graph.size()
        );

        gtsam::LevenbergMarquardtParams lm;
        lm.setMaxIterations(params.max_iterations);
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm);
        gtsam::Values result = optimizer.optimize();

        for (size_t i = 0; i < poses.size(); i++) {
            poses[i] = result.at<gtsam::Pose3>(X(i));
        }

        RCLCPP_INFO(logger, "Iter %d/%d: done", outer + 1, outer_iters);
    }

    return poses;
}

} // namespace offline_mapping_optimizer