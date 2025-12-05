#include <small_glim/odometry/odometry_estimation.hpp>

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>

#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/linear_damping_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
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
        exit(EXIT_FAILURE);
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
    save_imu_rate_trajectory = config->param<bool>("odometry_estimation.save_imu_rate_trajectory");
    num_threads = config->param<int>("odometry_estimation.num_threads");

    // GICP params
    correspondence_distance = config->param<double>("odometry_estimation.correspondence_distance");
    full_connection_window_size = config->param<int>("odometry_estimation.full_connection_window_size");

    // Keyframe params
    const std::string strategy = config->param<std::string>("odometry_estimation.keyframe_update_strategy");
    if (strategy == "DISPLACEMENT") {
        keyframe_strategy = KeyframeUpdateStrategy::DISPLACEMENT;
    } else if (strategy == "ENTROPY") {
        keyframe_strategy = KeyframeUpdateStrategy::ENTROPY;
    } else {
        logger::fatal("odom_estimation", "unknown keyframe update strategy: {}", strategy);
        exit(EXIT_FAILURE);
    }

    max_num_keyframes = config->param<int>("odometry_estimation.max_num_keyframes");
    keyframe_delta_trans = config->param<double>("odometry_estimation.keyframe_delta_trans");
    keyframe_delta_rot = config->param<double>("odometry_estimation.keyframe_delta_rot");
    keyframe_entropy_thresh = config->param<double>("odometry_estimation.keyframe_entropy_thresh");
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

    entropy_num_frames = 0;
    entropy_running_average = 0.0;
}

gtsam::NonlinearFactorGraph OdometryEstimationCPU::create_factors(
    const int current,
    const std::shared_ptr<gtsam::ImuFactor>& imu_factor,
    gtsam::Values& new_values
) {
    const int last = current - 1;

    if (current == 0 || !frames[current]->frame->size()) {
        return gtsam::NonlinearFactorGraph();
    }

    const auto create_binary_factor = [this](
        gtsam::NonlinearFactorGraph& factors,
        gtsam::Key target_key,
        gtsam::Key source_key,
        const EstimationFrame::ConstPtr& target,
        const EstimationFrame::ConstPtr& source
    ) {
        auto factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(
            target_key,
            source_key,
            target->frame,
            source->frame
        );
        factor->set_max_correspondence_distance(params->correspondence_distance);
        factor->set_num_threads(params->num_threads);
        factor->set_fused_cov_cache_mode(gtsam_points::FusedCovCacheMode::FULL);
        factors.add(factor);
    };

    gtsam::NonlinearFactorGraph factors;

    // There must be at least one factor between consecutive frames
    for (int target = current - params->full_connection_window_size; target < current; target++) {
        if (target < 0) {
            continue;
        }
        create_binary_factor(factors, X(target), X(current), frames[target], frames[current]);
    }

    for (const auto& keyframe : keyframes) {
        if (keyframe->id >= current - params->full_connection_window_size) {
            // Already connected in the previous loop
            continue;
        }

        double span = frames[current]->stamp - keyframe->stamp;
        if (span > params->smoother_lag - 0.1 || !frames[keyframe->id]) {
            // Keyframe is already marginalized
            continue; 
        } else {
            create_binary_factor(factors, X(keyframe->id), X(current), keyframe, frames[current]);
        }
    }
    
    return factors;
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
    const PreprocessedFrame::Ptr& raw_frame,
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
        new_frame->frame_id = FrameID::IMU; 
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
    imu_integration->integrate_imu(
        raw_frame->stamp,
        raw_frame->scan_end_time,
        predicted_nav_world_imu,
        last_imu_bias,
        pred_imu_times,
        pred_imu_poses
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

    if (params->save_imu_rate_trajectory) {
        new_frame->imu_rate_trajectory.resize(8, pred_imu_times.size());

        for (int i = 0; i < pred_imu_times.size(); i++) {
            const Eigen::Vector3d trans = pred_imu_poses[i].translation();
            const Eigen::Quaterniond quat(pred_imu_poses[i].linear());
            new_frame->imu_rate_trajectory.col(i) << pred_imu_times[i], trans, quat.x(), quat.y(), quat.z(), quat.w();
        }
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
    for (auto& pt : deskewed) {
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
    new_frame->frame_id = FrameID::IMU;
    frames.push_back(new_frame);

    new_factors.add(create_factors(current, imu_factor, new_values));

    // Update smoother
    update_smoother(new_factors, new_values, new_stamps, 1);

    // Find out marginalized frames
    while (marginalized_cursor < current) {
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
    logger::debug("odom_estimation", "frames updated");

    if (smoother->fallbackHappened()) {
        logger::warn("odom_estimation", "odometry estimation smoother fallback happened (time={})", raw_frame->stamp);
    }

    return frames[current];
}

std::vector<EstimationFrame::ConstPtr> OdometryEstimationCPU::get_remaining_frames() {
    std::vector<EstimationFrame::ConstPtr> marginalized_frames;
    for (int i = marginalized_cursor; i < frames.size(); i++) {
        marginalized_frames.push_back(frames[i]);
    }
    return marginalized_frames;
}

EstimationFrame::ConstPtr OdometryEstimationCPU::get_target_ivox_frame() {
    if (keyframes.empty()) return nullptr;
    std::vector<gtsam_points::PointCloud::ConstPtr> frames;
    std::vector<Eigen::Isometry3d> poses;
    for (const auto& keyframe : keyframes) {
        frames.push_back(keyframe->frame);
        poses.push_back(keyframe->T_world_imu);
    }
    auto merged = gtsam_points::merge_frames(poses, frames, 0.25);
    EstimationFrame::Ptr ivox_frame = std::make_shared<EstimationFrame>();
    ivox_frame->id = 0;
    ivox_frame->stamp = keyframes.back()->stamp;
    ivox_frame->frame_id = FrameID::WORLD;
    ivox_frame->T_world_lidar = Eigen::Isometry3d::Identity();
    ivox_frame->T_world_imu = Eigen::Isometry3d::Identity();
    ivox_frame->frame = merged;
    return ivox_frame;
}

void OdometryEstimationCPU::update_frames(
    int current,
    const gtsam::NonlinearFactorGraph& new_factors
) {
    logger::debug("odom_estimation", "update frames current={} marginalized_cursor={}", current, marginalized_cursor);

    for (int i = marginalized_cursor; i < frames.size(); i++) {
        try {
            Eigen::Isometry3d T_world_imu = Eigen::Isometry3d(smoother->calculateEstimate<gtsam::Pose3>(X(i)).matrix());
            Eigen::Vector3d v_world_imu = smoother->calculateEstimate<gtsam::Vector3>(V(i));
            Eigen::Matrix<double, 6, 1> imu_bias = smoother->calculateEstimate<gtsam::imuBias::ConstantBias>(B(i)).vector();
            frames[i]->T_world_imu = T_world_imu;
            frames[i]->T_world_lidar = T_world_imu * T_imu_lidar;
            frames[i]->v_world_imu = v_world_imu;
            frames[i]->imu_bias = imu_bias;
        } catch (std::out_of_range& e) {
            logger::error("odom_estimation", "caught {}", e.what());
            break;
        }
    }

    switch (params->keyframe_strategy) {
        case OdometryEstimationCPUParams::KeyframeUpdateStrategy::DISPLACEMENT: {
            update_keyframes_displacement(current);
            break;
        }
        case OdometryEstimationCPUParams::KeyframeUpdateStrategy::ENTROPY: {
            update_keyframes_entropy(new_factors, current);
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

void OdometryEstimationCPU::update_smoother(int count) {
    if (count <= 0) {
        return;
    }

    update_smoother(
        gtsam::NonlinearFactorGraph(),
        gtsam::Values(),
        std::map<std::uint64_t, double>(),
        count - 1
    );
}

void OdometryEstimationCPU::update_keyframes_displacement(int current) {
    if (keyframes.empty()) {
        keyframes.push_back(frames[current]);
        return;
    }

    const Eigen::Isometry3d delta_from_last = keyframes.back()->T_world_imu.inverse() * frames[current]->T_world_imu;
    const double delta_trans = delta_from_last.translation().norm();
    const double delta_rot = Eigen::AngleAxisd(delta_from_last.linear()).angle();

    if (delta_trans < params->keyframe_delta_trans && delta_rot < params->keyframe_delta_rot) {
        return;
    }

    const auto& new_keyframe = frames[current];
    keyframes.push_back(new_keyframe);

    if (keyframes.size() <= params->max_num_keyframes) {
        return;
    }

    // Calculate score for each keyframe based on distance
    const int leave_window = 2;
    const double eps = 1e-3;
    std::vector<double> scores(keyframes.size() - 1, 0.0);
    for (int i = leave_window; i < keyframes.size() - 1; i++) {
        double sum_inv_dist = 0.0;
        for (int j = 0; j < keyframes.size() - 1; j++) {
            if (i == j) {
                continue;
            }

            const double dist = (keyframes[i]->T_world_imu.translation() - keyframes[j]->T_world_imu.translation()).norm();
            sum_inv_dist += 1.0 / (dist + eps);
        }

        const double d0 = (keyframes[i]->T_world_imu.translation() - new_keyframe->T_world_imu.translation()).norm();
        scores[i] = std::sqrt(d0) * sum_inv_dist;
    }

    const auto max_score_loc = std::max_element(scores.begin(), scores.end());
    const int max_score_index = std::distance(scores.begin(), max_score_loc);

    keyframes.erase(keyframes.begin() + max_score_index);
}

void OdometryEstimationCPU::update_keyframes_entropy(const gtsam::NonlinearFactorGraph& matching_cost_factors, int current) {
    gtsam::Values values = smoother->calculateEstimate();

    gtsam::NonlinearFactorGraph valid_factors;
    for (const auto& factor : matching_cost_factors) {
        bool valid = std::all_of(factor->keys().begin(), factor->keys().end(), [&](const gtsam::Key key) { return values.exists(key); });
        if (!valid) {
            continue;
        }

        valid_factors.push_back(factor->clone());
    }

    if (valid_factors.empty()) {
        logger::warn("odom_estimation", "no valid factors for entropy calculation");
        update_keyframes_displacement(current);
        return;
    }

    auto linearized = valid_factors.linearize(values);
    gtsam::Matrix6 H = linearized->hessianBlockDiagonal()[X(current)];
    double negative_entropy = std::log(H.determinant());

    entropy_num_frames++;
    entropy_running_average += (negative_entropy - entropy_running_average) / entropy_num_frames;

    if (!keyframes.empty() && negative_entropy > entropy_running_average * params->keyframe_entropy_thresh) {
        return;
    }

    entropy_num_frames = 0;
    entropy_running_average = 0.0;

    const auto& new_keyframe = frames[current];
    keyframes.push_back(new_keyframe);

    if (keyframes.size() <= params->max_num_keyframes) {
        return;
    }

    keyframes.erase(keyframes.begin());
}

}