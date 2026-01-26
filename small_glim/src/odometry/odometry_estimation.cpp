#include <small_glim/odometry/odometry_estimation.hpp>

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>

#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/linear_damping_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_with_fallback.hpp>

#include <small_glim/common/config.hpp>
#include <small_glim/common/logger.hpp>
#include <small_glim/common/convert_to_string.hpp>
#include <small_glim/odometry/imu_integration.hpp>
#include <small_glim/preprocess/cloud_deskewing.hpp>
#include <small_glim/preprocess/cloud_covariance_estimation.hpp>

namespace small_glim {

using gtsam::symbol_shorthand::B; // IMU bias
using gtsam::symbol_shorthand::V; // IMU velocity   (v_world_imu)
using gtsam::symbol_shorthand::X; // IMU pose       (T_world_imu)

OdometryEstimationCPUParams::OdometryEstimationCPUParams(const Config::Ptr config) {
    // sensor config
    T_lidar_imu = config->param<Eigen::Isometry3d>("sensors.T_lidar_imu");
    imu_bias_noise = config->param<double>("sensors.imu_bias_noise");
    auto bias = config->param<std::vector<double>>("sensors.imu_bias");
    if (bias.size() == 6) {
        imu_bias = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(bias.data());
    } else {
        logger::fatal("odometry_estimation", "sensors.imu_bias need 6 parameters");
        std::exit(EXIT_FAILURE);
    }

    // odometry config
    fix_imu_bias = config->param<bool>("odometry_estimation.fix_imu_bias");
    use_init_world_imu = config->param<bool>("odometry_estimation.use_init_world_imu");
    if (use_init_world_imu) {
        init_T_world_imu = config->param<Eigen::Isometry3d>("odometry_estimation.init_T_world_imu");
        init_v_world_imu = config->param<Eigen::Vector3d>("odometry_estimation.init_v_world_imu");
    }
    init_pose_damping_scale = config->param<double>("odometry_estimation.init_pose_damping_scale");
    smoother_lag = config->param<double>("odometry_estimation.smoother_lag");
    use_isam2_dogleg = config->param<bool>("odometry_estimation.use_isam2_dogleg");
    isam2_relinearize_skip = config->param<int>("odometry_estimation.isam2_relinearize_skip");
    isam2_relinearize_thresh = config->param<double>("odometry_estimation.isam2_relinearize_thresh");
    num_threads = config->param<int>("odometry_estimation.num_threads");

    // odometry config
    std::string reg_type = config->param<std::string>("odometry_estimation.registration_type");
    if (reg_type == "GICP") {
        registration_type = RegistrationType::GICP;
    } else if (reg_type == "VGICP") {
        registration_type = RegistrationType::VGICP;
    } else {
        logger::fatal("odometry_estimation", "unknown registration type for odometry_estimation: {}", reg_type);
        std::exit(EXIT_FAILURE);
    }
    keyframe_window_size = config->param<int>("odometry_estimation.keyframe_window_size");
    keyframe_delta = config->param<int>("odometry_estimation.keyframe_delta");
    lru_thresh = config->param<int>("odometry_estimation.lru_thresh");
    target_downsampling_rate = config->param<double>("odometry_estimation.target_downsampling_rate");
    ivox_resolution = config->param<double>("odometry_estimation.ivox_resolution");
    ivox_min_dist = config->param<double>("odometry_estimation.ivox_min_dist");
    vgicp_resolution = config->param<double>("odometry_estimation.vgicp_resolution");
    vgicp_voxelmap_levels = config->param<int>("odometry_estimation.vgicp_voxelmap_levels");
    vgicp_voxelmap_scaling_factor = config->param<double>("odometry_estimation.vgicp_voxelmap_scaling_factor");
}

OdometryEstimationCPU::OdometryEstimationCPU(const Config::Ptr config) {
    marginalized_cursor = 0;
    T_lidar_imu.setIdentity();
    T_imu_lidar.setIdentity();
    params = std::make_unique<OdometryEstimationCPUParams>(config);

    init_estimation = std::make_unique<InitialStateEstimation>(config, params->T_lidar_imu, params->imu_bias);
    if (params->use_init_world_imu) {
        init_estimation->set_init_state(params->init_T_world_imu, params->init_v_world_imu);
    }

    imu_integration = std::make_unique<IMUIntegration>(config);
    deskewing = std::make_unique<CloudDeskewing>();
    covariance_estimation = std::make_unique<CloudCovarianceEstimation>(params->num_threads);

    gtsam::ISAM2Params isam2_params;
    if (params->use_isam2_dogleg) {
        isam2_params.setOptimizationParams(gtsam::ISAM2DoglegParams());
    }
    isam2_params.relinearizeSkip = params->isam2_relinearize_skip;
    isam2_params.setRelinearizeThreshold(params->isam2_relinearize_thresh);
    smoother = std::make_unique<FixedLagSmootherExt>(params->smoother_lag, isam2_params);

    last_T_target_imu.setIdentity();
    switch (params->registration_type) {
        case OdometryEstimationCPUParams::RegistrationType::GICP: {
            target_ivox = std::make_shared<gtsam_points::iVox>(params->ivox_resolution);
            target_ivox->voxel_insertion_setting().set_min_dist_in_cell(params->ivox_min_dist);
            target_ivox->set_lru_horizon(params->lru_thresh);
            target_ivox->set_neighbor_voxel_mode(1);
            break;
        }
        case OdometryEstimationCPUParams::RegistrationType::VGICP: {
            target_voxelmaps.resize(params->vgicp_voxelmap_levels);
            for (int i = 0; i < params->vgicp_voxelmap_levels; i++) {
                const double resolution = params->vgicp_resolution * std::pow(params->vgicp_voxelmap_scaling_factor, i);
                target_voxelmaps[i] = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(resolution);
                target_voxelmaps[i]->set_lru_horizon(params->lru_thresh);
            }
            break;
        }
    }
}

gtsam::NonlinearFactorGraph OdometryEstimationCPU::create_factors(
    const int current,
    const std::shared_ptr<gtsam::ImuFactor>& imu_factor,
    gtsam::Values& new_values
) {
    const int last = current - 1;

    if (current == 0) {
        last_T_target_imu = frames[current]->T_world_imu;
        // update_target(current, frames[current]->T_world_imu);
        return gtsam::NonlinearFactorGraph();
    }

    gtsam::NonlinearFactorGraph factors;

    // Create frame-to-model matching factors and add directly to the main factor graph
    switch (params->registration_type) {
        case OdometryEstimationCPUParams::RegistrationType::GICP: {
            if (target_ivox->has_points()) {
                // Clone the target to ensure the factor's cost function remains stable during optimization
                // even if the global map is updated later.
                auto target_ivox_copy = std::make_shared<gtsam_points::iVox>(*target_ivox);
                auto gicp_factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor_<gtsam_points::iVox, gtsam_points::PointCloud>>(
                    gtsam::Pose3(),
                    X(current),
                    target_ivox_copy,
                    frames[current]->frame,
                    target_ivox_copy
                );
                gicp_factor->set_max_correspondence_distance(params->ivox_resolution * 2.0);
                gicp_factor->set_num_threads(params->num_threads);
                factors.add(gicp_factor);
            }
            break;
        }
        case OdometryEstimationCPUParams::RegistrationType::VGICP: {
            for (const auto& voxelmap: target_voxelmaps) {
                if (!voxelmap->has_points()) continue;
                // Clone the target to ensure the factor's cost function remains stable
                auto voxelmap_copy = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(*voxelmap);
                auto vgicp_factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(
                    gtsam::Pose3(),
                    X(current),
                    voxelmap_copy,
                    frames[current]->frame
                );
                vgicp_factor->set_num_threads(params->num_threads);
                factors.add(vgicp_factor);
            }
            break;
        }
    }

    // Create pairwise matching factors
    for (const int target : active_keyframes) {
        if (target < 0 || target < marginalized_cursor) continue;
        if (frames[target]->frame->size() < 20) continue;
        switch (params->registration_type) {
            case OdometryEstimationCPUParams::RegistrationType::GICP: {
                auto gicp_factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(
                    X(target),
                    X(current),
                    frames[target]->frame,
                    frames[current]->frame
                );
                gicp_factor->set_max_correspondence_distance(params->ivox_resolution * 2.0);
                gicp_factor->set_num_threads(params->num_threads);
                factors.add(gicp_factor);
                break;
            }
            case OdometryEstimationCPUParams::RegistrationType::VGICP: {
                if (frames[target]->voxelmaps.empty()) continue;
                for (const auto& voxelmap : frames[target]->voxelmaps) {
                    auto vgicp_factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(
                        X(target),
                        X(current),
                        voxelmap,
                        frames[current]->frame
                    );
                    vgicp_factor->set_num_threads(params->num_threads);
                    factors.add(vgicp_factor);
                }
                break;
            }
        }
    }

    return factors;
}

int OdometryEstimationCPU::select_target_update_frame(const int preferred) const {
    if (preferred < 0 || preferred >= static_cast<int>(frames.size())) return -1;

    auto is_valid = [&](int idx) -> bool {
        if (idx < 0 || idx >= static_cast<int>(frames.size())) return false;
        if (!frames[idx]) return false;
        if (!frames[idx]->frame) return false;
        return !frames[idx]->deskew_imu_saturated;
    };

    if (is_valid(preferred)) return preferred;

    // Search nearby frames (prefer closest in time/index).
    constexpr int search_radius = 5;
    for (int d = 1; d <= search_radius; d++) {
        const int left = preferred - d;
        const int right = preferred + d;
        if (is_valid(left)) return left;
        if (is_valid(right)) return right;
    }

    return -1;
}

void OdometryEstimationCPU::update_target(
    const int current,
    const Eigen::Isometry3d& T_target_imu
) {
    auto frame = frames[current]->frame;
    if (current >= 5) {
        frame = gtsam_points::random_sampling(
            frames[current]->frame,
            params->target_downsampling_rate,
            mt
        );
    }

    auto transformed = gtsam_points::transform(frame, T_target_imu);
    switch (params->registration_type) {
        case OdometryEstimationCPUParams::RegistrationType::GICP: {
            target_ivox->insert(*transformed);
            break;
        }
        case OdometryEstimationCPUParams::RegistrationType::VGICP: {
            for (auto& target_voxelmap: target_voxelmaps) {
                target_voxelmap->insert(*transformed);
            }
            break;
        }
    }
}

void OdometryEstimationCPU::insert_imu(
    const double stamp,
    const Eigen::Vector3d& linear_acc,
    const Eigen::Vector3d& angular_vel
) {
    if (init_estimation) {
        init_estimation->insert_imu(stamp, linear_acc, angular_vel);
    }
    imu_integration->insert_imu(stamp, linear_acc, angular_vel);
}

EstimationFrame::ConstPtr OdometryEstimationCPU::insert_frame(
    const PreprocessedFrame::Ptr raw_frame,
    std::vector<EstimationFrame::ConstPtr>& marginalized_frames
) {
    if (raw_frame->size()) {
        logger::debug(
            "odom_estimation",
            "insert_frame points={} times={} ~ {}",
            raw_frame->size(),
            raw_frame->times.front(),
            raw_frame->times.back()
        );
    } else {
        logger::warn("odom_estimation", "insert_frame points={}", raw_frame->size());
    }

    const int current = frames.size();
    const int last = current - 1;

    // The very first frame
    if (frames.empty()) {
        EstimationFrame::ConstPtr init_state;
        init_estimation->insert_frame(raw_frame);
        init_state = init_estimation->initial_pose();

        if (init_state == nullptr) {
            logger::debug("odom_estimation", "waiting for initial IMU state estimation to be finished");
            return nullptr;
        }
        init_estimation.reset();

        logger::info("odom_estimation", "initial IMU state estimation result");
        logger::info("odom_estimation", "T_world_imu={}", convert_to_string(init_state->T_world_imu));
        logger::info("odom_estimation", "v_world_imu={}", convert_to_string(init_state->v_world_imu));
        logger::info("odom_estimation", "imu_bias={}", convert_to_string(init_state->imu_bias));

        // Initialize the first frame
        EstimationFrame::Ptr new_frame = std::make_shared<EstimationFrame>();
        new_frame->id = current;
        new_frame->stamp = raw_frame->stamp;

        T_lidar_imu = init_state->T_lidar_imu;
        T_imu_lidar = T_lidar_imu.inverse();

        new_frame->T_lidar_imu = init_state->T_lidar_imu;
        new_frame->T_world_lidar = init_state->T_world_lidar;
        new_frame->T_world_imu = init_state->T_world_imu;

        new_frame->v_world_imu = init_state->v_world_imu;
        new_frame->imu_bias = init_state->imu_bias;
        new_frame->raw_frame = raw_frame;

        // Transform points into IMU frame
        std::vector<Eigen::Vector4d> points_imu(raw_frame->size());
        for (int i = 0; i < raw_frame->size(); i++) {
            points_imu[i] = T_imu_lidar * raw_frame->points[i];
        }

        std::vector<Eigen::Vector4d> normals;
        std::vector<Eigen::Matrix4d> covs;
        covariance_estimation->estimate(points_imu, raw_frame->neighbors, normals, covs);

        auto frame = std::make_shared<gtsam_points::PointCloudCPU>(points_imu);
        if (raw_frame->intensities.size()) {
            frame->add_intensities(raw_frame->intensities);
        }
        frame->add_covs(covs);
        frame->add_normals(normals);
        new_frame->frame = frame;
        new_frame->frame_type = FrameType::IMU;
        frames.push_back(new_frame);

        // Initialize the estimator
        gtsam::Values new_values;
        gtsam::NonlinearFactorGraph new_factors;
        gtsam::FixedLagSmootherKeyTimestampMap new_stamps;

        new_stamps[X(0)] = raw_frame->stamp;
        new_stamps[V(0)] = raw_frame->stamp;
        new_stamps[B(0)] = raw_frame->stamp;

        new_values.insert(X(0), gtsam::Pose3(new_frame->T_world_imu.matrix()));
        new_values.insert(V(0), new_frame->v_world_imu);
        new_values.insert(B(0), gtsam::imuBias::ConstantBias(new_frame->imu_bias));

        // Prior for initial IMU states
        new_factors.emplace_shared<gtsam_points::LinearDampingFactor>(
            X(0),
            6,
            params->init_pose_damping_scale
        );
        new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
            V(0),
            init_state->v_world_imu,
            gtsam::noiseModel::Isotropic::Precision(3, 1.0)
        );
        new_factors.emplace_shared<gtsam_points::LinearDampingFactor>(B(0), 6, 1e6);
        new_factors.add(create_factors(current, nullptr, new_values));

        update_smoother(new_factors, new_values, new_stamps);
        update_frames(current, new_factors);

        if (current % params->keyframe_delta == 0) {
            active_keyframes.push_back(current);
        }

        return frames.back();
    }

    gtsam::Values new_values;
    gtsam::NonlinearFactorGraph new_factors;
    gtsam::FixedLagSmootherKeyTimestampMap new_stamps;

    const double last_stamp = frames[last]->stamp;
    const auto last_T_world_imu_ = smoother->calculateEstimate<gtsam::Pose3>(X(last));
    const auto last_T_world_imu = gtsam::Pose3(last_T_world_imu_.rotation().normalized(), last_T_world_imu_.translation());
    const auto last_v_world_imu = smoother->calculateEstimate<gtsam::Vector3>(V(last));
    const auto last_imu_bias = smoother->calculateEstimate<gtsam::imuBias::ConstantBias>(B(last));
    const gtsam::NavState last_nav_world_imu(last_T_world_imu, last_v_world_imu);

    // IMU integration between LiDAR scans (inter-scan)
    int num_imu_integrated = 0;
    const int imu_read_cursor = imu_integration->integrate_imu(last_stamp, raw_frame->stamp, last_imu_bias, &num_imu_integrated);
    imu_integration->erase_imu_data(imu_read_cursor);
    logger::debug("odom_estimation", "num_imu_integrated={}", num_imu_integrated);

    // IMU state prediction
    const gtsam::NavState predicted_nav_world_imu = imu_integration->integrated_measurements().predict(last_nav_world_imu, last_imu_bias);
    gtsam::Pose3 predicted_T_world_imu = predicted_nav_world_imu.pose();
    gtsam::Vector3 predicted_v_world_imu = predicted_nav_world_imu.velocity();

    // Overwrite the predicted state with the last states if no IMU data is available
    if (num_imu_integrated < 2 && last > 1) {
        const Eigen::Isometry3d T_delta = frames[last - 1]->T_lidar_imu.inverse() * frames[last]->T_lidar_imu;
        predicted_T_world_imu = gtsam::Pose3((frames[last]->T_world_imu * T_delta).matrix());
        predicted_v_world_imu = frames[last]->v_world_imu;
    }

    new_stamps[X(current)] = raw_frame->stamp;
    new_stamps[V(current)] = raw_frame->stamp;
    new_stamps[B(current)] = raw_frame->stamp;

    new_values.insert(X(current), predicted_T_world_imu);
    new_values.insert(V(current), predicted_v_world_imu);
    new_values.insert(B(current), last_imu_bias);

    // Constant IMU bias assumption
    new_factors.add(
        gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
            B(last),
            B(current),
            gtsam::imuBias::ConstantBias(),
            gtsam::noiseModel::Isotropic::Sigma(6, params->imu_bias_noise)
        )
    );
    if (params->fix_imu_bias) {
        new_factors.add(
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(
                B(current),
                gtsam::imuBias::ConstantBias(params->imu_bias),
                gtsam::noiseModel::Isotropic::Precision(6, 1e3)
            )
        );
    }

    // Create IMU factor
    gtsam::ImuFactor::shared_ptr imu_factor;
    if (num_imu_integrated >= 2) {
        imu_factor = gtsam::make_shared<gtsam::ImuFactor>(
            X(last),
            V(last),
            X(current),
            V(current),
            B(last),
            imu_integration->integrated_measurements()
        );
        new_factors.add(imu_factor);
    } else {
        logger::warn("odom_estimation", "insufficient number of IMU data between LiDAR scans!!");
        logger::warn(
            "odom_estimation",
            "t_last={:.6f} t_current={:.6f} num_imu={}",
            last_stamp,
            raw_frame->stamp,
            num_imu_integrated
        );
        new_factors.add(
            gtsam::BetweenFactor<gtsam::Vector3>(
                V(last),
                V(current),
                gtsam::Vector3::Zero(),
                gtsam::noiseModel::Isotropic::Sigma(3, 1.0)
            )
        );
    }

    // Motion prediction for deskewing (intra-scan)
    std::vector<double> pred_imu_times;
    std::vector<Eigen::Isometry3d> pred_imu_poses;
    IMUSaturationStatus deskew_sat;
    imu_integration->integrate_imu(
        raw_frame->stamp,
        raw_frame->scan_end_time,
        predicted_nav_world_imu,
        last_imu_bias,
        pred_imu_times,
        pred_imu_poses,
        &deskew_sat
    );

    // Create EstimationFrame
    EstimationFrame::Ptr new_frame = std::make_shared<EstimationFrame>();
    new_frame->id = current;
    new_frame->stamp = raw_frame->stamp;

    new_frame->T_lidar_imu = T_lidar_imu;
    new_frame->T_world_imu = Eigen::Isometry3d(predicted_T_world_imu.matrix());
    new_frame->T_world_lidar = Eigen::Isometry3d(predicted_T_world_imu.matrix()) * T_imu_lidar;
    new_frame->v_world_imu = predicted_v_world_imu;
    new_frame->imu_bias = last_imu_bias.vector();
    new_frame->raw_frame = raw_frame;

    new_frame->deskew_imu_saturated = deskew_sat.any();
    new_frame->deskew_acc_saturated_axes = deskew_sat.acc_axes;
    new_frame->deskew_gyro_saturated_axes = deskew_sat.gyro_axes;

    // Store IMU-rate trajectory used for deskewing
    new_frame->imu_rate_trajectory.resize(8, pred_imu_times.size());
    for (int i = 0; i < pred_imu_times.size(); i++) {
        const Eigen::Vector3d trans = pred_imu_poses[i].translation();
        const Eigen::Quaterniond quat(pred_imu_poses[i].linear());
        new_frame->imu_rate_trajectory.col(i) << pred_imu_times[i], trans, quat.x(), quat.y(), quat.z(), quat.w();
    }

    // Deskew and tranform points into IMU frame
    auto deskewed = deskewing->deskew(
        T_imu_lidar,
        pred_imu_times,
        pred_imu_poses,
        raw_frame->stamp,
        raw_frame->times,
        raw_frame->points
    );
    for (auto& pt: deskewed) {
        pt = T_imu_lidar * pt;
    }

    std::vector<Eigen::Vector4d> deskewed_normals;
    std::vector<Eigen::Matrix4d> deskewed_covs;
    covariance_estimation->estimate(deskewed, raw_frame->neighbors, deskewed_normals, deskewed_covs);

    auto frame = std::make_shared<gtsam_points::PointCloudCPU>(deskewed);
    if (raw_frame->intensities.size()) {
        frame->add_intensities(raw_frame->intensities);
    }
    frame->add_covs(deskewed_covs);
    frame->add_normals(deskewed_normals);
    new_frame->frame = frame;
    new_frame->frame_type = FrameType::IMU;

    if (params->registration_type == OdometryEstimationCPUParams::RegistrationType::VGICP) {
        new_frame->voxelmaps.resize(params->vgicp_voxelmap_levels);
        for (int i = 0; i < params->vgicp_voxelmap_levels; i++) {
            double resolution = params->vgicp_resolution * std::pow(params->vgicp_voxelmap_scaling_factor, i);
            new_frame->voxelmaps[i] = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(resolution);
            new_frame->voxelmaps[i]->insert(*frame);
        }
    }

    frames.push_back(new_frame);

    new_factors.add(create_factors(current, imu_factor, new_values));

    // Update smoother
    update_smoother(new_factors, new_values, new_stamps);

    // Find out marginalized frames
    while (marginalized_cursor < current) {
        bool is_active_keyframe = false;
        for (int k : active_keyframes) {
            if (k == marginalized_cursor) {
                is_active_keyframe = true;
                break;
            }
        }
        if (is_active_keyframe) {
            break;
        }

        double span = frames[current]->stamp - frames[marginalized_cursor]->stamp;
        if (span < params->smoother_lag - 0.1) {
            break;
        }

        marginalized_frames.push_back(frames[marginalized_cursor]);
        frames[marginalized_cursor].reset();
        marginalized_cursor++;
    }

    // Update frames
    update_frames(current, new_factors);

    if (current % params->keyframe_delta == 0) {
        active_keyframes.push_back(current);
        while (active_keyframes.size() > params->keyframe_window_size) {
            int old_keyframe = active_keyframes.front();
            active_keyframes.pop_front();

            if (frames[old_keyframe]) {
                const int chosen = select_target_update_frame(old_keyframe);
                if (chosen < 0) {
                    logger::warn(
                        "odom_estimation",
                        "skip iVox target update: no valid neighbor found (preferred={}, time={})",
                        old_keyframe,
                        frames[old_keyframe]->stamp
                    );
                } else {
                    if (chosen != old_keyframe) {
                        logger::warn(
                            "odom_estimation",
                            "replace saturated target frame: preferred={} -> chosen={} (preferred_time={}, chosen_time={})",
                            old_keyframe,
                            chosen,
                            frames[old_keyframe]->stamp,
                            frames[chosen]->stamp
                        );
                    }
                    update_target(chosen, frames[chosen]->T_world_imu);
                    last_T_target_imu = frames[chosen]->T_world_imu;
                }
            }
        }
    }

    if (smoother->fallbackHappened()) {
        logger::warn("odom_estimation", "odometry estimation smoother fallback happened (time={})", raw_frame->stamp);
    }

    return frames[current];
}

std::vector<EstimationFrame::ConstPtr> OdometryEstimationCPU::get_remaining_frames() {
    // Perform a few optimization iterations at the end
    // for(int i=0; i<5; i++) {
    //   smoother->update();
    // }
    // OdometryEstimationCPU::update_frames(frames.size() - 1, gtsam::NonlinearFactorGraph());

    std::vector<EstimationFrame::ConstPtr> marginalized_frames;
    for (int i = marginalized_cursor; i < frames.size(); i++) {
        marginalized_frames.push_back(frames[i]);
    }

    return marginalized_frames;
}

EstimationFrame::ConstPtr OdometryEstimationCPU::get_target_ivox_frame() {
    if (frames.empty()) return nullptr;
    EstimationFrame::Ptr target_frame = std::make_shared<EstimationFrame>();
    target_frame->id = frames.size() - 1;
    target_frame->stamp = frames.back()->stamp;
    target_frame->T_lidar_imu = frames.back()->T_lidar_imu;
    target_frame->T_world_lidar = target_frame->T_lidar_imu.inverse();
    target_frame->T_world_imu.setIdentity();
    target_frame->v_world_imu.setZero();
    target_frame->imu_bias.setZero();
    target_frame->frame_type = FrameType::WORLD;
    switch (params->registration_type) {
        case OdometryEstimationCPUParams::RegistrationType::GICP: {
            target_frame->frame = std::make_shared<gtsam_points::PointCloudCPU>(target_ivox->voxel_points());
            break;
        }
        case OdometryEstimationCPUParams::RegistrationType::VGICP: {
            target_frame->frame = std::make_shared<gtsam_points::PointCloudCPU>(target_voxelmaps[0]->voxel_points());
            break;
        }
    }
    return target_frame;
}

void OdometryEstimationCPU::update_frames(
    int current,
    const gtsam::NonlinearFactorGraph& new_factors
) {
    logger::debug("odom_estimation", "update frames current={} marginalized_cursor={}", current, marginalized_cursor);

    gtsam::Values values = smoother->calculateEstimate();
    for (int i = marginalized_cursor; i < frames.size(); i++) {
        if (!values.exists(X(i))) continue;
        try {
            Eigen::Isometry3d T_world_imu = Eigen::Isometry3d(values.at<gtsam::Pose3>(X(i)).matrix());
            Eigen::Vector3d v_world_imu = values.at<gtsam::Vector3>(V(i));
            Eigen::Matrix<double, 6, 1> imu_bias = values.at<gtsam::imuBias::ConstantBias>(B(i)).vector();
            frames[i]->T_world_imu = T_world_imu;
            frames[i]->T_world_lidar = T_world_imu * T_imu_lidar;
            frames[i]->v_world_imu = v_world_imu;
            frames[i]->imu_bias = imu_bias;
        } catch (std::out_of_range& e) {
            logger::error("odom_estimation", "caught {}", e.what());
            logger::error("odom_estimation", "current={}", current);
            logger::error("odom_estimation", "marginalized_cursor={}", marginalized_cursor);
            break;
        }
    }
}

void OdometryEstimationCPU::update_smoother(
    const gtsam::NonlinearFactorGraph& new_factors,
    const gtsam::Values& new_values,
    const std::map<std::uint64_t, double>& new_stamp,
    int update_count
) {
    smoother->update(new_factors, new_values, new_stamp);
    for (int i = 0; i < update_count; i++) {
        smoother->update();
    }
}

}