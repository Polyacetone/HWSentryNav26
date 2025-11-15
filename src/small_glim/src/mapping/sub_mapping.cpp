#include <small_glim/mapping/sub_mapping.hpp>

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <gtsam_points/config.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <small_glim/common/config.hpp>
#include <small_glim/common/logger.hpp>
#include <small_glim/odometry/imu_integration.hpp>
#include <small_glim/preprocess/cloud_deskewing.hpp>
#include <small_glim/preprocess/cloud_covariance_estimation.hpp>

namespace small_glim {

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

SubMappingParams::SubMappingParams(const Config::Ptr config) {
    num_threads = config->param<int>("sub_mapping.num_threads");
    enable_imu = config->param<bool>("sub_mapping.enable_imu");
    enable_optimization = config->param<bool>("sub_mapping.enable_optimization");

    max_num_keyframes = config->param<int>("sub_mapping.max_num_keyframes");

    keyframe_update_strategy = config->param<std::string>("sub_mapping.keyframe_update_strategy");
    keyframe_update_min_points = config->param<int>("sub_mapping.keyframe_update_min_points");
    keyframe_update_interval_rot = config->param<double>("sub_mapping.keyframe_update_interval_rot");
    keyframe_update_interval_trans = config->param<double>("sub_mapping.keyframe_update_interval_trans");
    max_keyframe_overlap = config->param<double>("sub_mapping.max_keyframe_overlap");

    create_between_factors = config->param<bool>("sub_mapping.create_between_factors");
    between_registration_type = config->param<std::string>("sub_mapping.between_registration_type");

    registration_error_factor_type = config->param<std::string>("sub_mapping.registration_error_factor_type");
    keyframe_randomsampling_rate = config->param<double>("sub_mapping.keyframe_randomsampling_rate");
    keyframe_voxel_resolution = config->param<double>("sub_mapping.keyframe_voxel_resolution");
    keyframe_voxelmap_levels = config->param<int>("sub_mapping.keyframe_voxelmap_levels");
    keyframe_voxelmap_scaling_factor = config->param<double>("sub_mapping.keyframe_voxelmap_scaling_factor");

    submap_downsample_resolution = config->param<double>("sub_mapping.submap_downsample_resolution");
    submap_target_num_points = config->param<int>("sub_mapping.submap_target_num_points");
}

SubMapping::SubMapping(const Config::Ptr config) {
    params = std::make_unique<SubMappingParams>(config);
    submap_count = 0;
    imu_integration = std::make_unique<IMUIntegration>(config);
    deskewing = std::make_unique<CloudDeskewing>();
    covariance_estimation = std::make_unique<CloudCovarianceEstimation>(params->num_threads);
    values = std::make_unique<gtsam::Values>();
    graph = std::make_unique<gtsam::NonlinearFactorGraph>();
}

void SubMapping::insert_imu(
    const double stamp,
    const Eigen::Vector3d& linear_acc,
    const Eigen::Vector3d& angular_vel
) {
    if (params->enable_imu) {
        imu_integration->insert_imu(stamp, linear_acc, angular_vel);
    }
}

void SubMapping::insert_frame(const EstimationFrame::ConstPtr& odom_frame_) {
    logger::debug("sub_mapping", "insert_frame frame_id={} stamp={}", odom_frame_->id, odom_frame_->stamp);

    delayed_input_queue.emplace_back(odom_frame_);
    if (delayed_input_queue.size() < 2) {
        return;
    }

    EstimationFrame::Ptr odom_frame = delayed_input_queue.front()->clone();
    delayed_input_queue.pop_front();
    EstimationFrame::ConstPtr next_frame = delayed_input_queue.front();

    if (params->enable_imu) {
        logger::debug("sub_mapping", "smoothing trajectory");
        // Smoothing IMU-based pose estimation
        gtsam::NavState nav_world_imu(
            gtsam::Pose3(odom_frame->T_world_imu.matrix()),
            odom_frame->v_world_imu
        );
        gtsam::imuBias::ConstantBias imu_bias(odom_frame->imu_bias);

        std::vector<double> imu_stamps;
        std::vector<Eigen::Isometry3d> imu_poses;
        imu_integration->integrate_imu(
            odom_frame->stamp,
            next_frame->stamp,
            nav_world_imu,
            imu_bias,
            imu_stamps,
            imu_poses
        );

        gtsam::Values values;
        for (int i = 0; i < imu_stamps.size(); i++) {
            values.insert(X(i), gtsam::Pose3(imu_poses[i].matrix()));
        }

        gtsam::NonlinearFactorGraph graph;
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            X(0),
            gtsam::Pose3(odom_frame->T_world_imu.matrix()),
            gtsam::noiseModel::Isotropic::Sigma(6, 1e-5)
        );
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            X(imu_stamps.size() - 1),
            gtsam::Pose3(next_frame->T_world_imu.matrix()),
            gtsam::noiseModel::Isotropic::Sigma(6, 1e-5)
        );
        for (int i = 1; i < imu_stamps.size(); i++) {
            const double dt =
                (imu_stamps[i] - imu_stamps[i - 1]) / (next_frame->stamp - odom_frame->stamp);
            const Eigen::Isometry3d T_last_current = imu_poses[i - 1].inverse() * imu_poses[i];
            graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(i - 1),
                X(i),
                gtsam::Pose3(T_last_current.matrix()),
                gtsam::noiseModel::Isotropic::Sigma(6, dt + 1e-2)
            );
        }

        gtsam::LevenbergMarquardtParams lm_params;
        lm_params.setAbsoluteErrorTol(1e-6);
        lm_params.setRelativeErrorTol(1e-6);
        lm_params.setMaxIterations(5);

        values = gtsam::LevenbergMarquardtOptimizer(graph, values, lm_params).optimize();

        odom_frame->imu_rate_trajectory.resize(8, imu_stamps.size());
        for (int i = 0; i < imu_stamps.size(); i++) {
            const Eigen::Vector3d trans(imu_poses[i].translation());
            const Eigen::Quaterniond quat(imu_poses[i].linear());
            odom_frame->imu_rate_trajectory.col(i) << imu_stamps[i], trans, quat.x(), quat.y(), quat.z(), quat.w();
        }
    }

    const int current = odom_frames.size();
    const int last = current - 1;
    odom_frames.push_back(odom_frame);
    values->insert(X(current), gtsam::Pose3(odom_frame->T_world_sensor().matrix()));

    if (params->enable_imu && odom_frame->frame_id != FrameID::IMU) {
        logger::warn("sub_mapping", "odom frames are not estimated in the IMU frame while sub_mapping requires IMU estimation");
    }

    // Fix the first frame
    if (current == 0) {
        logger::debug("sub_mapping", "first frame in submap");
        graph->emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            X(0),
            values->at<gtsam::Pose3>(X(0)),
            gtsam::noiseModel::Isotropic::Precision(6, 1e8)
        );
    }
    // Create a relative pose factor between consecutive frames
    else if (params->create_between_factors) {
        logger::debug("sub_mapping", "create between factors");
        const Eigen::Isometry3d delta =
            odom_frames[last]->T_world_sensor().inverse() * odom_frame->T_world_sensor();

        if (params->between_registration_type == "GICP") {
            const auto& last_frame = odom_frames[last]->frame;
            const auto& current_frame = odom_frames[current]->frame;

            gtsam::noiseModel::Base::shared_ptr noise_model;
            if (last_frame->size() < 500 || current_frame->size() < 500) {
                logger::warn(
                    "sub_mapping",
                    "use an identity covariance because either of last or current frames have too few points (last={} current={})",
                    last_frame->size(),
                    current_frame->size()
                );
                noise_model = gtsam::noiseModel::Isotropic::Precision(6, 1e3);
            } else {
                auto factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(
                    X(last),
                    X(current),
                    last_frame,
                    current_frame
                );
                auto linearized = factor->linearize(*values);
                // graph->emplace_shared<gtsam::LinearContainerFactor>(linearized, *values);

                auto H = linearized->hessianBlockDiagonal()[X(current)];
                noise_model = gtsam::noiseModel::Gaussian::Information(H);
            }

            graph->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(last),
                X(current),
                gtsam::Pose3(delta.matrix()),
                noise_model
            );
        } else if (params->between_registration_type == "NONE") {
            graph->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(last),
                X(current),
                gtsam::Pose3(delta.matrix()),
                gtsam::noiseModel::Isotropic::Precision(6, 1e3)
            );
        } else {
            logger::fatal(
                "sub_mapping",
                "unknown between registration type ({})",
                params->between_registration_type
            );
            abort();
        }
    }

    // Create an IMU preintegration factor
    if (params->enable_imu) {
        logger::debug("sub_mapping", "create IMU factor");
        const gtsam::imuBias::ConstantBias imu_bias(odom_frame->imu_bias);

        values->insert(V(current), odom_frame->v_world_imu);
        values->insert(B(current), imu_bias);

        graph->emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
            V(current),
            odom_frame->v_world_imu,
            gtsam::noiseModel::Isotropic::Precision(3, 1e3)
        );
        graph->emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
            B(current),
            imu_bias,
            gtsam::noiseModel::Isotropic::Precision(6, 1e6)
        );

        if (current != 0) {
            int num_integrated = 0;
            const int imu_read_cursor = imu_integration->integrate_imu(
                odom_frames[last]->stamp,
                odom_frames[current]->stamp,
                imu_bias,
                &num_integrated
            );
            imu_integration->erase_imu_data(imu_read_cursor);

            graph->emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
                B(last),
                B(current),
                gtsam::imuBias::ConstantBias(),
                gtsam::noiseModel::Isotropic::Precision(6, 1e6)
            );
            if (num_integrated >= 2) {
                graph->emplace_shared<gtsam::ImuFactor>(
                    X(last),
                    V(last),
                    X(current),
                    V(current),
                    B(last),
                    imu_integration->integrated_measurements()
                );
            } else {
                logger::warn("sub_mapping", "insufficient IMU data between LiDAR frames!! (sub_mapping)");
                graph->emplace_shared<gtsam::BetweenFactor<gtsam::Vector3>>(
                    V(last),
                    V(current),
                    gtsam::Vector3::Zero(),
                    gtsam::noiseModel::Isotropic::Precision(3, 1.0)
                );
            }
        }
    }

    bool insert_as_keyframe = keyframes.empty();
    if (!insert_as_keyframe && odom_frame->frame
        && odom_frame->frame->size() > params->keyframe_update_min_points)
    {
        if (params->keyframe_update_strategy == "OVERLAP") { // Overlap-based keyframe update
            if (keyframes.back()->voxelmaps.empty() || odom_frame->frame->size() < 10) {
                logger::warn(
                    "sub_mapping",
                    "voxelmap or odom_frame is empty!! (voxelmap={} odom_frame={})",
                    keyframes.back()->voxelmaps.size(),
                    odom_frame->frame->size()
                );
            } else {
                const double overlap = gtsam_points::overlap_auto(
                    keyframes.back()->voxelmaps.back(),
                    odom_frame->frame,
                    keyframes.back()->T_world_sensor().inverse() * odom_frame->T_world_sensor()
                );
                insert_as_keyframe = overlap < params->max_keyframe_overlap;
            }
        } else if (params->keyframe_update_strategy == "DISPLACEMENT") { // Displacement-based keyframe update
            const Eigen::Isometry3d delta_from_keyframe =
                keyframes.back()->T_world_sensor().inverse() * odom_frame->T_world_sensor();
            const double delta_trans = delta_from_keyframe.translation().norm();
            const double delta_angle = Eigen::AngleAxisd(delta_from_keyframe.linear()).angle();

            insert_as_keyframe = delta_trans > params->keyframe_update_interval_trans
                || delta_angle > params->keyframe_update_interval_rot;
        } else {
            logger::fatal("sub_mapping", "unknown keyframe update strategy ({})", params->keyframe_update_strategy);
            abort();
        }
    }

    // Create a new keyframe
    if (insert_as_keyframe) {
        logger::debug("sub_mapping", "insert frame as keyframe");
        insert_keyframe(current, odom_frame);

        // Create registration error factors (fully connected)
        for (int i = 0; i < keyframes.size() - 1; i++) {
            if (keyframes[i]->frame->size() == 0 || keyframes.back()->frame->size() == 0) {
                logger::warn(
                    "sub_mapping",
                    "skip creation of registration error factors because keyframe has no points (keyframe[i]={}, keyframe[-1]={})",
                    keyframes[i]->frame->size(),
                    keyframes.back()->frame->size()
                );
            }

            if (params->registration_error_factor_type == "VGICP") {
                for (const auto& voxelmap: keyframes[i]->voxelmaps) {
                    if (!voxelmap) {
                        logger::warn("sub_mapping", "voxelmap is empty!");
                        continue;
                    }

                    graph->emplace_shared<gtsam_points::IntegratedVGICPFactor>(
                        X(keyframe_indices[i]),
                        X(current),
                        voxelmap,
                        keyframes.back()->frame
                    );
                }
            } else {
                logger::fatal(
                    "sub_mapping",
                    "unknown registration error factor type ({})",
                    params->registration_error_factor_type
                );
                abort();
            }
        }
    }

    if (odom_frames.size() >= 2) {
        // Drop unnecessary points data
        // The last frame may be required to compute the relative pose factor
        odom_frames[odom_frames.size() - 2] = odom_frames[odom_frames.size() - 2]->clone_wo_points();
    }

    auto new_submap = create_submap();

    if (new_submap) {
        new_submap->id = submap_count++;
        submap_queue.push_back(new_submap);

        odom_frames.clear();
        keyframes.clear();
        keyframe_indices.clear();
        values = std::make_unique<gtsam::Values>();
        graph = std::make_unique<gtsam::NonlinearFactorGraph>();
    }
}

void SubMapping::insert_keyframe(const int current, const EstimationFrame::ConstPtr& odom_frame) {
    gtsam_points::PointCloud::ConstPtr deskewed_frame = odom_frame->frame;

    // Re-perform deskewing with smoothed IMU poses
    if (params->enable_imu && odom_frame->raw_frame && odom_frame->imu_rate_trajectory.cols() >= 2) {
        if (std::abs(odom_frame->stamp - odom_frame->imu_rate_trajectory(0, 0)) > 1e-3) {
            logger::warn(
                "sub_mapping",
                "inconsistent frame stamp and imu_rate stamp!! (odom_frame={} imu_rate_trajectory={})",
                odom_frame->stamp,
                odom_frame->imu_rate_trajectory(0, 0)
            );
        }
        if (odom_frame->raw_frame->scan_end_time
            > odom_frame->imu_rate_trajectory.rightCols<1>()[0] + 1e-3)
        {
            logger::warn(
                "sub_mapping",
                "imu_rate stamp does not cover the scan duration range!! (imu_rate_end={} scan_end={})",
                odom_frame->imu_rate_trajectory.rightCols<1>()[0],
                odom_frame->raw_frame->scan_end_time
            );
        }

        std::vector<double> imu_pred_times(odom_frame->imu_rate_trajectory.cols());
        std::vector<Eigen::Isometry3d> imu_pred_poses(odom_frame->imu_rate_trajectory.cols());
        for (int i = 0; i < odom_frame->imu_rate_trajectory.cols(); i++) {
            const Eigen::Matrix<double, 8, 1> imu =
                odom_frame->imu_rate_trajectory.col(i).transpose();
            imu_pred_times[i] = imu[0];
            imu_pred_poses[i].setIdentity();
            imu_pred_poses[i].translation() << imu[1], imu[2], imu[3];
            imu_pred_poses[i].linear() =
                Eigen::Quaterniond(imu[7], imu[4], imu[5], imu[6]).toRotationMatrix();
        }

        auto deskewed = deskewing->deskew(
            odom_frame->T_lidar_imu.inverse(),
            imu_pred_times,
            imu_pred_poses,
            odom_frame->raw_frame->stamp,
            odom_frame->raw_frame->times,
            odom_frame->raw_frame->points
        );

        auto frame = std::make_shared<gtsam_points::PointCloudCPU>(deskewed);
        for (int i = 0; i < frame->size(); i++) {
            frame->points[i] = odom_frame->T_lidar_imu.inverse() * frame->points[i];
        }
        frame->add_covs(
            covariance_estimation->estimate(frame->points_storage, odom_frame->raw_frame->neighbors)
        );

        deskewed_frame = frame;
    }

    // Random sampling for registration error factors
    gtsam_points::PointCloud::Ptr subsampled_frame =
        gtsam_points::random_sampling(deskewed_frame, params->keyframe_randomsampling_rate, mt);

    EstimationFrame::Ptr keyframe = std::make_shared<EstimationFrame>();
    *keyframe = *odom_frame;

    keyframe->voxelmaps.clear();
    for (int i = 0; i < params->keyframe_voxelmap_levels; i++) {
        const double resolution = params->keyframe_voxel_resolution
            * std::pow(params->keyframe_voxelmap_scaling_factor, i);
        auto voxelmap = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(resolution);
        voxelmap->insert(*keyframe->frame);
        keyframe->voxelmaps.push_back(voxelmap);
    }

    keyframe->frame = subsampled_frame;
    keyframes.push_back(keyframe);
    keyframe_indices.push_back(current);
}

SubMap::Ptr SubMapping::create_submap(bool force_create) const {
    logger::debug("sub_mapping", "|keyframes|={}", keyframes.size());
    if (keyframes.size() < params->max_num_keyframes && !force_create) {
        return nullptr;
    }

    logger::debug("sub_mapping", "create_submap");
    // Optimization
    gtsam_points::LevenbergMarquardtExtParams lm_params;
    lm_params.setMaxIterations(20);
    gtsam_points::LevenbergMarquardtOptimizerExt optimizer(*graph, *values, lm_params);
    if (params->enable_optimization) {
        try {
            gtsam::Values optimized = optimizer.optimize();
            *values = optimized;
        } catch (const gtsam::IndeterminantLinearSystemException& e) {
            logger::error("sub_mapping", "an indeterminant lienar system exception was caught during sub map optimization");
            logger::error("sub_mapping", "{}", e.what());
        }
    }

    // Create a submap by merging optimized frames
    SubMap::Ptr submap = std::make_shared<SubMap>();
    submap->id = 0;

    const int center = odom_frames.size() / 2;
    submap->T_world_origin = Eigen::Isometry3d(values->at<gtsam::Pose3>(X(center)).matrix());
    submap->T_origin_endpoint_L = submap->T_world_origin.inverse() * Eigen::Isometry3d(values->at<gtsam::Pose3>(X(0)).matrix());
    submap->T_origin_endpoint_R = submap->T_world_origin.inverse() * Eigen::Isometry3d(values->at<gtsam::Pose3>(X(odom_frames.size() - 1)).matrix());

    submap->odom_frames = odom_frames;
    submap->frames.resize(odom_frames.size());
    for (int i = 0; i < odom_frames.size(); i++) {
        EstimationFrame::Ptr frame = std::make_shared<EstimationFrame>();
        *frame = *odom_frames[i];

        const Eigen::Isometry3d T_world_sensor(values->at<gtsam::Pose3>(X(i)).matrix());
        frame->set_T_world_sensor(odom_frames[i]->frame_id, T_world_sensor);

        if (params->enable_imu) {
            frame->v_world_imu = values->at<gtsam::Vector3>(V(i));
            frame->imu_bias = values->at<gtsam::imuBias::ConstantBias>(B(i)).vector();
        }

        submap->frames[i] = frame;
    }

    logger::debug("sub_mapping", "merge frames");
    std::vector<gtsam_points::PointCloud::ConstPtr> keyframes_to_merge(keyframes.size());
    std::vector<Eigen::Isometry3d> poses_to_merge(keyframes.size());
    for (int i = 0; i < keyframes.size(); i++) {
        keyframes_to_merge[i] = keyframes[i]->frame;
        poses_to_merge[i] = submap->T_world_origin.inverse()
            * Eigen::Isometry3d(values->at<gtsam::Pose3>(X(keyframe_indices[i])).matrix());
    }

    if (submap->frame == nullptr) {
        submap->frame = gtsam_points::merge_frames_auto(
            poses_to_merge,
            keyframes_to_merge,
            params->submap_downsample_resolution
        );
    }
    logger::debug("sub_mapping", "|merged_submap|={}", submap->frame->size());

    if (params->submap_target_num_points > 0
        && submap->frame->size() > params->submap_target_num_points)
    {
        std::mt19937 mt(
            submap_count * 643145 + submap->frame->size() * 4312
        ); // Just a random-like seed
        submap->frame = gtsam_points::random_sampling(
            submap->frame,
            static_cast<double>(params->submap_target_num_points) / submap->frame->size(),
            mt
        );
        logger::debug("sub_mapping", "|subsampled_submap|={}", submap->frame->size());
    }

    return submap;
}

std::vector<SubMap::Ptr> SubMapping::get_submaps() {
    std::vector<SubMap::Ptr> submaps;
    submap_queue.swap(submaps);
    return submaps;
}

std::vector<SubMap::Ptr> SubMapping::submit_end_of_sequence() {
    std::vector<SubMap::Ptr> submaps;
    if (!odom_frames.empty()) {
        auto new_submap = create_submap(true);

        if (new_submap) {
            new_submap->id = submap_count++;
            submaps.push_back(new_submap);
        }
    }

    return submaps;
}

}