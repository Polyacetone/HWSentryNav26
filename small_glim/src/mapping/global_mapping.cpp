#include <small_glim/mapping/global_mapping.hpp>

#include <filesystem>
#include <gtsam/base/serialization.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PoseRotationPrior.h>
#include <gtsam/slam/PoseTranslationPrior.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_points/config.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/factors/linear_damping_factor.hpp>
#include <gtsam_points/factors/rotate_vector3_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/optimizers/isam2_ext.hpp>
#include <gtsam_points/optimizers/isam2_ext_dummy.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <small_glim/common/config.hpp>
#include <small_glim/common/logger.hpp>
#include <small_glim/odometry/imu_integration.hpp>

namespace small_glim {

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::E;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

GlobalMappingParams::GlobalMappingParams(const Config::Ptr config) {
    enable_imu = config->param<bool>("global_mapping.enable_imu");
    enable_optimization = config->param<bool>("global_mapping.enable_optimization");

    enable_between_factors = config->param<bool>("global_mapping.create_between_factors");
    between_registration_type = config->param<std::string>("global_mapping.between_registration_type");
    registration_error_factor_type = config->param<std::string>("global_mapping.registration_error_factor_type");
    submap_voxel_resolution = config->param<double>("global_mapping.submap_voxel_resolution");
    submap_voxel_resolution_max = config->param<double>("global_mapping.submap_voxel_resolution_max");
    submap_voxel_resolution_dmin = config->param<double>("global_mapping.submap_voxel_resolution_dmin");
    submap_voxel_resolution_dmax = config->param<double>("global_mapping.submap_voxel_resolution_dmax");

    submap_voxelmap_levels = config->param<int>("global_mapping.submap_voxelmap_levels");
    submap_voxelmap_scaling_factor = config->param<double>("global_mapping.submap_voxelmap_scaling_factor");

    randomsampling_rate = config->param<double>("global_mapping.randomsampling_rate");
    max_implicit_loop_distance = config->param<double>("global_mapping.max_implicit_loop_distance");
    min_implicit_loop_overlap = config->param<double>("global_mapping.min_implicit_loop_overlap");

    use_isam2_dogleg = config->param<bool>("global_mapping.use_isam2_dogleg");
    isam2_relinearize_skip = config->param<int>("global_mapping.isam2_relinearize_skip");
    isam2_relinearize_thresh = config->param<double>("global_mapping.isam2_relinearize_thresh");

    init_pose_damping_scale = config->param<double>("global_mapping.init_pose_damping_scale");
}

GlobalMapping::GlobalMapping(const Config::Ptr config) {
    params = std::make_unique<GlobalMappingParams>(config);

    session_id = 0;
    imu_integration = std::make_unique<IMUIntegration>(config);

    new_values = std::make_unique<gtsam::Values>();
    new_factors = std::make_unique<gtsam::NonlinearFactorGraph>();

    gtsam::ISAM2Params isam2_params;
    if (params->use_isam2_dogleg) {
        gtsam::ISAM2DoglegParams dogleg_params;
        isam2_params.setOptimizationParams(dogleg_params);
    }
    isam2_params.relinearizeSkip = params->isam2_relinearize_skip;
    isam2_params.setRelinearizeThreshold(params->isam2_relinearize_thresh);

    if (params->enable_optimization) {
        isam2 = std::make_unique<gtsam_points::ISAM2Ext>(isam2_params);
    } else {
        isam2 = std::make_unique<gtsam_points::ISAM2ExtDummy>(isam2_params);
    }
}

void GlobalMapping::insert_imu(
    const double stamp,
    const Eigen::Vector3d& linear_acc,
    const Eigen::Vector3d& angular_vel
) {
    if (params->enable_imu) {
        imu_integration->insert_imu(stamp, linear_acc, angular_vel);
    }
}

void GlobalMapping::insert_submap(const SubMap::Ptr& submap) {
    logger::debug("global_mapping", "insert_submap id={} |frame|={}", submap->id, submap->frame->size());

    const int current = submaps.size();
    const int last = current - 1;
    insert_submap(current, submap);

    gtsam::Pose3 current_T_world_submap = gtsam::Pose3::Identity();
    gtsam::Pose3 last_T_world_submap = gtsam::Pose3::Identity();

    if (current != 0) {
        if (isam2->valueExists(X(last))) {
            last_T_world_submap = isam2->calculateEstimate<gtsam::Pose3>(X(last));
        } else {
            last_T_world_submap = new_values->at<gtsam::Pose3>(X(last));
        }

        const Eigen::Isometry3d T_origin0_endpointR0 = submaps[last]->T_origin_endpoint_R;
        const Eigen::Isometry3d T_origin1_endpointL1 = submaps[current]->T_origin_endpoint_L;
        const Eigen::Isometry3d T_endpointR0_endpointL1 =
            submaps[last]->odom_frames.back()->T_world_sensor().inverse()
            * submaps[current]->odom_frames.front()->T_world_sensor();
        const Eigen::Isometry3d T_origin0_origin1 =
            T_origin0_endpointR0 * T_endpointR0_endpointL1 * T_origin1_endpointL1.inverse();

        current_T_world_submap = last_T_world_submap * gtsam::Pose3(T_origin0_origin1.matrix());
    } else {
        current_T_world_submap = gtsam::Pose3(submap->T_world_origin.matrix());
    }

    new_values->insert(X(current), current_T_world_submap);
    submap->T_world_origin = Eigen::Isometry3d(current_T_world_submap.matrix());

    submap->drop_frame_points();

    if (current == 0) {
        new_factors->emplace_shared<gtsam_points::LinearDampingFactor>(X(0), 6, params->init_pose_damping_scale);
    } else {
        new_factors->add(*create_between_factors(current));
        new_factors->add(*create_matching_cost_factors(current));
    }

    if (params->enable_imu) {
        logger::debug("global_mapping", "create IMU factor");
        if (submap->odom_frames.front()->frame_id != FrameID::IMU) {
            logger::warn("global_mapping", "odom frames are not estimated in the IMU frame while global mapping requires IMU estimation");
        }

        // Local velocities
        const gtsam::imuBias::ConstantBias imu_biasL(submap->frames.front()->imu_bias);
        const gtsam::imuBias::ConstantBias imu_biasR(submap->frames.back()->imu_bias);

        const Eigen::Vector3d v_origin_imuL = submap->T_world_origin.linear().inverse() * submap->frames.front()->v_world_imu;
        const Eigen::Vector3d v_origin_imuR = submap->T_world_origin.linear().inverse() * submap->frames.back()->v_world_imu;

        const auto prior_noise3 = gtsam::noiseModel::Isotropic::Precision(3, 1e6);
        const auto prior_noise6 = gtsam::noiseModel::Isotropic::Precision(6, 1e6);

        if (current > 0) {
            new_values->insert(
                E(current * 2),
                gtsam::Pose3((submap->T_world_origin * submap->T_origin_endpoint_L).matrix())
            );
            new_values->insert(
                V(current * 2),
                (submap->T_world_origin.linear() * v_origin_imuL).eval()
            );
            new_values->insert(B(current * 2), imu_biasL);

            new_factors->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(current),
                E(current * 2),
                gtsam::Pose3(submap->T_origin_endpoint_L.matrix()),
                prior_noise6
            );
            new_factors->emplace_shared<gtsam_points::RotateVector3Factor>(
                X(current),
                V(current * 2),
                v_origin_imuL,
                prior_noise3
            );
            new_factors->emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                B(current * 2),
                imu_biasL,
                prior_noise6
            );
            new_factors->emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
                B(current * 2),
                B(current * 2 + 1),
                gtsam::imuBias::ConstantBias(),
                prior_noise6
            );
        }

        new_values->insert(
            E(current * 2 + 1),
            gtsam::Pose3((submap->T_world_origin * submap->T_origin_endpoint_R).matrix())
        );
        new_values->insert(
            V(current * 2 + 1),
            (submap->T_world_origin.linear() * v_origin_imuR).eval()
        );
        new_values->insert(B(current * 2 + 1), imu_biasR);

        new_factors->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(current),
            E(current * 2 + 1),
            gtsam::Pose3(submap->T_origin_endpoint_R.matrix()),
            prior_noise6
        );
        new_factors->emplace_shared<gtsam_points::RotateVector3Factor>(
            X(current),
            V(current * 2 + 1),
            v_origin_imuR,
            prior_noise3
        );
        new_factors->emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
            B(current * 2 + 1),
            imu_biasR,
            prior_noise6
        );

        if (current != 0) {
            const double stampL = submaps[last]->frames.back()->stamp;
            const double stampR = submaps[current]->frames.front()->stamp;

            int num_integrated;
            const int imu_read_cursor =
                imu_integration->integrate_imu(stampL, stampR, imu_biasL, &num_integrated);
            imu_integration->erase_imu_data(imu_read_cursor);

            if (num_integrated < 2) {
                logger::warn("global_mapping", "insufficient IMU data between submaps (global_mapping)!");
                new_factors->emplace_shared<gtsam::BetweenFactor<gtsam::Vector3>>(
                    V(last * 2 + 1),
                    V(current * 2),
                    gtsam::Vector3::Zero(),
                    gtsam::noiseModel::Isotropic::Precision(3, 1.0)
                );
            } else {
                new_factors->emplace_shared<gtsam::ImuFactor>(
                    E(last * 2 + 1),
                    V(last * 2 + 1),
                    E(current * 2),
                    V(current * 2),
                    B(last * 2 + 1),
                    imu_integration->integrated_measurements()
                );
            }
        }
    }

    logger::debug("global_mapping", "|new_factors|={} |new_values|={}", new_factors->size(), new_values->size());

    auto result = update_isam2(*new_factors, *new_values);

    new_values = std::make_unique<gtsam::Values>();
    new_factors = std::make_unique<gtsam::NonlinearFactorGraph>();

    update_submaps();
}

void GlobalMapping::insert_submap(int current, const SubMap::Ptr& submap) {
    submap->voxelmaps.clear();

    // Adaptively determine the voxel resolution based on the median distance
    const int max_scan_count = 256;
    const double dist_median = gtsam_points::median_distance(submap->frame, max_scan_count);
    const double p = std::max(
        0.0,
        std::min(
            1.0,
            (dist_median - params->submap_voxel_resolution_dmin)
                / (params->submap_voxel_resolution_dmax - params->submap_voxel_resolution_dmin)
        )
    );
    const double base_resolution = params->submap_voxel_resolution
        + p * (params->submap_voxel_resolution_max - params->submap_voxel_resolution);

    // Create frame and voxelmaps
    gtsam_points::PointCloud::ConstPtr subsampled_submap;
    if (params->randomsampling_rate > 0.99) {
        subsampled_submap = submap->frame;
    } else {
        subsampled_submap =
            gtsam_points::random_sampling(submap->frame, params->randomsampling_rate, mt);
    }

    if (submap->voxelmaps.empty()) {
        for (int i = 0; i < params->submap_voxelmap_levels; i++) {
            const double resolution =
                base_resolution * std::pow(params->submap_voxelmap_scaling_factor, i);
            auto voxelmap = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(resolution);
            voxelmap->insert(*subsampled_submap);
            submap->voxelmaps.push_back(voxelmap);
        }
    }

    submaps.push_back(submap);
    subsampled_submaps.push_back(subsampled_submap);
}

void GlobalMapping::find_overlapping_submaps(double min_overlap) {
    if (submaps.empty()) {
        return;
    }

    // Between factors are Vector2i actually. A bad use of Vector3i
    std::unordered_set<Eigen::Vector3i, gtsam_points::Vector3iHash> existing_factors;
    for (const auto& factor: isam2->getFactorsUnsafe()) {
        if (factor == nullptr) {
            continue;
        }
        if (factor->keys().size() != 2) {
            continue;
        }

        gtsam::Symbol sym1(factor->keys()[0]);
        gtsam::Symbol sym2(factor->keys()[1]);
        if (sym1.chr() != 'x' || sym2.chr() != 'x') {
            continue;
        }

        existing_factors.emplace(sym1.index(), sym2.index(), 0);
    }

    double squared_max_implicit_loop_distance = params->max_implicit_loop_distance * params->max_implicit_loop_distance;
    for (int i = 0; i < submaps.size(); i++) {
        for (int j = i + 1; j < submaps.size(); j++) {
            if (existing_factors.count(Eigen::Vector3i(i, j, 0))) {
                continue;
            }

            const Eigen::Isometry3d delta =
                submaps[i]->T_world_origin.inverse() * submaps[j]->T_world_origin;
            const double squared_dist = delta.translation().squaredNorm();
            if (squared_dist > squared_max_implicit_loop_distance) {
                continue;
            }

            const double overlap = gtsam_points::overlap_auto(
                submaps[i]->voxelmaps.back(),
                subsampled_submaps[j],
                delta
            );
            if (overlap < min_overlap) {
                continue;
            }

            for (const auto& voxelmap: submaps[i]->voxelmaps) {
                new_factors->emplace_shared<gtsam_points::IntegratedVGICPFactor>(
                    X(i),
                    X(j),
                    voxelmap,
                    subsampled_submaps[j]
                );
            }
        }
    }

    logger::info("global_mapping", "new overlapping {} submap pairs found", new_factors->size());

    auto result = update_isam2(*new_factors, *new_values);

    new_factors->resize(0);
    new_values->clear();

    update_submaps();
}

void GlobalMapping::optimize() {
    if (isam2->empty()) {
        return;
    }

    logger::debug("global_mapping", "|new_factors|={} |new_values|={}", new_factors->size(), new_values->size());

    auto result = update_isam2(*new_factors, *new_values);

    new_factors = std::make_unique<gtsam::NonlinearFactorGraph>();
    new_values = std::make_unique<gtsam::Values>();

    update_submaps();
}

std::shared_ptr<gtsam::NonlinearFactorGraph>
GlobalMapping::create_between_factors(int current) const {
    auto factors = gtsam::make_shared<gtsam::NonlinearFactorGraph>();
    if (current == 0 || !params->enable_between_factors) {
        return factors;
    }

    const int last = current - 1;
    const gtsam::Pose3 init_delta = gtsam::Pose3(
        (submaps[last]->T_world_origin.inverse() * submaps[current]->T_world_origin).matrix()
    );

    if (params->between_registration_type == "NONE") {
        factors->add(
            gtsam::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(last),
                X(current),
                init_delta,
                gtsam::noiseModel::Isotropic::Precision(6, 1e6)
            )
        );
        return factors;
    } else if (params->between_registration_type == "GICP") {
        gtsam::Values values;
        values.insert(X(0), gtsam::Pose3::Identity());
        values.insert(X(1), init_delta);

        gtsam::NonlinearFactorGraph graph;
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            X(0),
            gtsam::Pose3::Identity(),
            gtsam::noiseModel::Isotropic::Precision(6, 1e6)
        );

        auto factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(
            X(0),
            X(1),
            submaps[last]->frame,
            submaps[current]->frame
        );
        factor->set_max_correspondence_distance(0.5);
        factor->set_num_threads(2);
        graph.add(factor);

        logger::debug("global_mapping", "--- LM optimization ---");
        gtsam_points::LevenbergMarquardtExtParams lm_params;
        lm_params.setlambdaInitial(1e-12);
        lm_params.setMaxIterations(10);
        lm_params.callback = [this](const auto& status, const auto& values) {
            logger::debug("global_mapping", "{}", status.to_string());
        };

        gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
        values = optimizer.optimize();

        const gtsam::Pose3 estimated_delta = values.at<gtsam::Pose3>(X(1));
        const auto linearized = factor->linearize(values);
        const auto H = linearized->hessianBlockDiagonal()[X(1)] + 1e6 * gtsam::Matrix6::Identity();

        factors->add(
            gtsam::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(last),
                X(current),
                estimated_delta,
                gtsam::noiseModel::Gaussian::Information(H)
            )
        );
        return factors;
    } else {
        logger::fatal("global_mapping", "unknown between registration type ({})", params->between_registration_type);
        exit(EXIT_FAILURE);
    }
}

std::shared_ptr<gtsam::NonlinearFactorGraph>
GlobalMapping::create_matching_cost_factors(int current) const {
    auto factors = gtsam::make_shared<gtsam::NonlinearFactorGraph>();
    if (current == 0) {
        return factors;
    }

    const auto& current_submap = submaps.back();

    double previous_overlap = 0.0;
    double squared_max_implicit_loop_distance = params->max_implicit_loop_distance * params->max_implicit_loop_distance;

    for (int i = 0; i < current; i++) {
        const double squared_dist = (submaps[i]->T_world_origin.translation() - current_submap->T_world_origin.translation()).squaredNorm();
        if (squared_dist > squared_max_implicit_loop_distance) {
            continue;
        }

        const Eigen::Isometry3d delta = submaps[i]->T_world_origin.inverse() * current_submap->T_world_origin;
        const double overlap = gtsam_points::overlap_auto(submaps[i]->voxelmaps.back(), current_submap->frame, delta);

        previous_overlap = i == current - 1 ? overlap : previous_overlap;
        if (overlap < params->min_implicit_loop_overlap) {
            continue;
        }

        if (params->registration_error_factor_type == "VGICP") {
            for (const auto& voxelmap: submaps[i]->voxelmaps) {
                factors->emplace_shared<gtsam_points::IntegratedVGICPFactor>(
                    X(i),
                    X(current),
                    voxelmap,
                    subsampled_submaps[current]
                );
            }
        } else {
            logger::fatal("global_mapping", "unknown registration error type ({})", params->registration_error_factor_type);
            exit(EXIT_FAILURE);
        }
    }

    if (previous_overlap < std::max(0.25, params->min_implicit_loop_overlap)) {
        logger::warn(
            "global_mapping",
            "previous submap has only a small overlap with the current submap ({})",
            previous_overlap
        );
        logger::warn("global_mapping", "create a between factor to prevent the submap from being isolated");
        const int last = current - 1;
        const gtsam::Pose3 init_delta = gtsam::Pose3(
            (submaps[last]->T_world_origin.inverse() * submaps[current]->T_world_origin).matrix()
        );
        factors->add(
            gtsam::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(last),
                X(current),
                init_delta,
                gtsam::noiseModel::Isotropic::Precision(6, 1e6)
            )
        );
    }

    return factors;
}

void GlobalMapping::update_submaps() {
    for (int i = 0; i < submaps.size(); i++) {
        submaps[i]->T_world_origin = Eigen::Isometry3d(isam2->calculateEstimate<gtsam::Pose3>(X(i)).matrix());
    }
}

gtsam_points::ISAM2ResultExt GlobalMapping::update_isam2(
    const gtsam::NonlinearFactorGraph& new_factors,
    const gtsam::Values& new_values
) {
    gtsam_points::ISAM2ResultExt result;

    gtsam::Key indeterminant_nearby_key = 0;
    try {
        result = isam2->update(new_factors, new_values);
    } catch (const gtsam::IndeterminantLinearSystemException& e) {
        logger::error("global_mapping", "an indeterminant lienar system exception was caught during global map optimization!!");
        logger::error("global_mapping", "{}", e.what());
        indeterminant_nearby_key = e.nearbyVariable();
    }

    if (indeterminant_nearby_key != 0) {
        const gtsam::Symbol symbol(indeterminant_nearby_key);
        if (symbol.chr() == 'v' || symbol.chr() == 'b' || symbol.chr() == 'e') {
            indeterminant_nearby_key = X(symbol.index() / 2);
        }
        logger::warn(
            "global_mapping",
            "insert a damping factor at {} to prevent corruption",
            std::string(gtsam::Symbol(indeterminant_nearby_key))
        );

        gtsam::Values values = isam2->getLinearizationPoint();
        gtsam::NonlinearFactorGraph factors = isam2->getFactorsUnsafe();
        factors.emplace_shared<gtsam_points::LinearDampingFactor>(indeterminant_nearby_key, 6, 1e3);

        gtsam::ISAM2Params isam2_params;
        if (params->use_isam2_dogleg) {
            gtsam::ISAM2DoglegParams dogleg_params;
            isam2_params.setOptimizationParams(dogleg_params);
        }
        isam2_params.relinearizeSkip = params->isam2_relinearize_skip;
        isam2_params.setRelinearizeThreshold(params->isam2_relinearize_thresh);

        if (params->enable_optimization) {
            isam2 = std::make_unique<gtsam_points::ISAM2Ext>(isam2_params);
        } else {
            isam2 = std::make_unique<gtsam_points::ISAM2ExtDummy>(isam2_params);
        }

        logger::warn("global_mapping", "reset isam2");
        return update_isam2(factors, values);
    }

    return result;
}

void GlobalMapping::save(const std::string& path) {
    optimize();

    std::filesystem::create_directories(path);

    gtsam::NonlinearFactorGraph serializable_factors;
    std::unordered_map<std::string, gtsam::NonlinearFactor::shared_ptr> matching_cost_factors;

    for (const auto& factor: isam2->getFactorsUnsafe()) {
        bool serializable = !dynamic_cast<gtsam_points::IntegratedMatchingCostFactor*>(factor.get());

        if (serializable) {
            serializable_factors.push_back(factor);
        } else {
            const gtsam::Symbol symbol0(factor->keys()[0]);
            const gtsam::Symbol symbol1(factor->keys()[1]);
            const std::string key = std::to_string(symbol0.index()) + "_" + std::to_string(symbol1.index());
            matching_cost_factors[key] = factor;
        }
    }

    logger::info("global_mapping", "serializing factor graph to {}/graph.bin", path);
    serializeToBinaryFile(serializable_factors, path + "/graph.bin");
    serializeToBinaryFile(isam2->calculateEstimate(), path + "/values.bin");

    std::ofstream ofs(path + "/graph.txt");
    ofs << "num_submaps: " << submaps.size() << std::endl;
    ofs << "num_all_frames: " << std::accumulate(
        submaps.begin(),
        submaps.end(),
        0,
        [](int sum, const SubMap::ConstPtr& submap) { return sum + submap->frames.size(); }
    ) << std::endl;

    ofs << "num_matching_cost_factors: " << matching_cost_factors.size() << std::endl;
    for (const auto& factor: matching_cost_factors) {
        std::string type;

        if (dynamic_cast<gtsam_points::IntegratedGICPFactor*>(factor.second.get())) {
            type = "gicp";
        } else if (dynamic_cast<gtsam_points::IntegratedVGICPFactor*>(factor.second.get())) {
            type = "vgicp";
        }

        gtsam::Symbol symbol0(factor.second->keys()[0]);
        gtsam::Symbol symbol1(factor.second->keys()[1]);
        ofs << "matching_cost " << type << " " << symbol0.index() << " " << symbol1.index() << std::endl;
    }

    std::ofstream odom_lidar_ofs(path + "/odom_lidar.txt");
    std::ofstream traj_lidar_ofs(path + "/traj_lidar.txt");

    std::ofstream odom_imu_ofs(path + "/odom_imu.txt");
    std::ofstream traj_imu_ofs(path + "/traj_imu.txt");

    const auto write_tum_frame =
        [](std::ofstream& ofs, const double stamp, const Eigen::Isometry3d& pose) {
            const Eigen::Quaterniond quat(pose.linear());
            const Eigen::Vector3d trans(pose.translation());
            ofs << std::format(
                "{:.9f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}",
                stamp, trans.x(), trans.y(), trans.z(),
                quat.x(), quat.y(), quat.z(), quat.w()
            ) << std::endl;
        };

    for (int i = 0; i < submaps.size(); i++) {
        for (const auto& frame: submaps[i]->odom_frames) {
            write_tum_frame(odom_lidar_ofs, frame->stamp, frame->T_world_lidar);
            write_tum_frame(odom_imu_ofs, frame->stamp, frame->T_world_imu);
        }

        const Eigen::Isometry3d T_world_endpoint_L = submaps[i]->T_world_origin * submaps[i]->T_origin_endpoint_L;
        const Eigen::Isometry3d T_odom_lidar0 = submaps[i]->frames.front()->T_world_lidar;
        const Eigen::Isometry3d T_odom_imu0 = submaps[i]->frames.front()->T_world_imu;

        for (const auto& frame: submaps[i]->frames) {
            const Eigen::Isometry3d T_world_imu = T_world_endpoint_L * T_odom_imu0.inverse() * frame->T_world_imu;
            const Eigen::Isometry3d T_world_lidar = T_world_imu * frame->T_lidar_imu.inverse();

            write_tum_frame(traj_imu_ofs, frame->stamp, T_world_imu);
            write_tum_frame(traj_lidar_ofs, frame->stamp, T_world_lidar);
        }

        submaps[i]->save(std::format("{}/{:06d}", path, i));
    }
}

std::vector<Eigen::Vector4d> GlobalMapping::export_points() {
    int num_all_points = 0;
    for (const auto& submap: submaps) {
        num_all_points += submap->frame->size();
    }

    std::vector<Eigen::Vector4d> all_points;
    all_points.reserve(num_all_points);

    for (const auto& submap: submaps) {
        std::transform(
            submap->frame->points,
            submap->frame->points + submap->frame->size(),
            std::back_inserter(all_points),
            [&](const Eigen::Vector4d& p) { return submap->T_world_origin * p; }
        );
    }

    return all_points;
}

bool GlobalMapping::load(const std::string& path) {
    std::ifstream ifs(path + "/graph.txt");
    if (!ifs) {
        logger::error("global_mapping", "failed to open {}/graph.txt", path);
        return false;
    }

    const int start_from_frame_id = submaps.size();

    std::string token;
    int num_submaps, num_all_frames, num_matching_cost_factors;

    ifs >> token >> num_submaps;
    ifs >> token >> num_all_frames;
    ifs >> token >> num_matching_cost_factors;

    std::vector<std::tuple<std::string, int, int>> matching_cost_factors(num_matching_cost_factors);
    for (int i = 0; i < num_matching_cost_factors; i++) {
        auto& factor = matching_cost_factors[i];
        ifs >> token >> std::get<0>(factor) >> std::get<1>(factor) >> std::get<2>(factor);
    }

    logger::info("global_mapping", "Load submaps (session_id={})", session_id);
    submaps.reserve(submaps.size() + num_submaps);
    subsampled_submaps.reserve(submaps.size() + num_submaps);
    for (int i = 0; i < num_submaps; i++) {
        auto submap = SubMap::load(std::format("{}/{:06d}", path, i));
        if (!submap) {
            return false;
        }
        submap->id += start_from_frame_id;
        submap->session_id = session_id;

        // Adaptively determine the voxel resolution based on the median distance
        const int max_scan_count = 256;
        const double dist_median = gtsam_points::median_distance(submap->frame, max_scan_count);
        const double p = std::max(
            0.0,
            std::min(
                1.0,
                (dist_median - params->submap_voxel_resolution_dmin) / (params->submap_voxel_resolution_dmax - params->submap_voxel_resolution_dmin)
            )
        );
        const double base_resolution = params->submap_voxel_resolution
            + p * (params->submap_voxel_resolution_max - params->submap_voxel_resolution);

        gtsam_points::PointCloud::Ptr subsampled_submap;
        if (params->randomsampling_rate > 0.99) {
            subsampled_submap = submap->frame;
        } else {
            subsampled_submap = gtsam_points::random_sampling(submap->frame, params->randomsampling_rate, mt);
        }

        submaps.push_back(submap);
        submaps.back()->voxelmaps.clear();
        subsampled_submaps.push_back(subsampled_submap);

        for (int j = 0; j < params->submap_voxelmap_levels; j++) {
            const double resolution = base_resolution * std::pow(params->submap_voxelmap_scaling_factor, j);
            auto voxelmap = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(resolution);
            voxelmap->insert(*subsampled_submaps.back());
            submaps.back()->voxelmaps.push_back(voxelmap);
        }
    }

    gtsam::Values values, loaded_values;
    gtsam::NonlinearFactorGraph graph, loaded_graph;
    bool needs_recover = false;

    logger::info("global_mapping", "deserializing factor graph");
    try {
        gtsam::deserializeFromBinaryFile(path + "/graph.bin", loaded_graph);
    } catch (boost::archive::archive_exception e) {
        logger::error("global_mapping", "failed to deserialize factor graph!!");
        logger::error("global_mapping", "{}", e.what());
    } catch (std::exception& e) {
        logger::error("global_mapping", "failed to deserialize factor graph!!");
        logger::error("global_mapping", "{}", e.what());
        needs_recover = true;
    }

    logger::info("global_mapping", "deserializing values");
    try {
        gtsam::deserializeFromBinaryFile(path + "/values.bin", loaded_values);
    } catch (boost::archive::archive_exception e) {
        logger::error("global_mapping", "failed to deserialize values!!");
        logger::error("global_mapping", "{}", e.what());
    } catch (std::exception& e) {
        logger::error("global_mapping", "failed to deserialize values!!");
        logger::error("global_mapping", "{}", e.what());
        needs_recover = true;
    }

    // remap keys in graph and values if dump previously loaded
    if (start_from_frame_id > 0) {
        std::map<gtsam::Key, gtsam::Key> rekey_mapping;
        for (int i = 0; i < num_submaps; i++) {
            rekey_mapping[X(i)] = X(i + start_from_frame_id);
            rekey_mapping[E(i * 2)] = E((i + start_from_frame_id) * 2);
            rekey_mapping[E(i * 2 + 1)] = E((i + start_from_frame_id) * 2 + 1);
            rekey_mapping[B(i * 2)] = B((i + start_from_frame_id) * 2);
            rekey_mapping[B(i * 2 + 1)] = B((i + start_from_frame_id) * 2 + 1);
            rekey_mapping[V(i * 2)] = V((i + start_from_frame_id) * 2);
            rekey_mapping[V(i * 2 + 1)] = V((i + start_from_frame_id) * 2 + 1);
        }

        logger::info("global_mapping", "removing translation prior factors");
        auto remove_loc = std::remove_if(loaded_graph.begin(), loaded_graph.end(), [](const auto& factor) {
            return dynamic_cast<gtsam::PoseTranslationPrior<gtsam::Pose3>*>(factor.get()) != nullptr;
        });
        logger::info("global_mapping", "removed {} prior factors", std::distance(remove_loc, loaded_graph.end()));
        loaded_graph.erase(remove_loc, loaded_graph.end());

        logger::info("global_mapping", "removing damping factors");
        remove_loc = std::remove_if(loaded_graph.begin(), loaded_graph.end(), [](const auto& factor) {
            return dynamic_cast<gtsam_points::LinearDampingFactor*>(factor.get()) != nullptr;
        });
        logger::info("global_mapping", "removed {} prior factors", std::distance(remove_loc, loaded_graph.end()));
        loaded_graph.erase(remove_loc, loaded_graph.end());

        logger::info("global_mapping", "removing prior factors");
        remove_loc = std::remove_if(loaded_graph.begin(), loaded_graph.end(), [](const auto& factor) {
            return dynamic_cast<gtsam::PriorFactor<gtsam::Pose3>*>(factor.get()) != nullptr;
        });
        logger::info("global_mapping", "removed {} prior factors", std::distance(remove_loc, loaded_graph.end()));
        loaded_graph.erase(remove_loc, loaded_graph.end());

        // rekey graph
        graph = loaded_graph;
        logger::info("global_mapping", "rekeying factors");
        graph = graph.rekey(rekey_mapping);

        // rekey values
        for (auto it = loaded_values.begin(); it != loaded_values.end(); ++it) {
            auto matched_key = rekey_mapping.find(it->key);
            if (matched_key != rekey_mapping.end()) {
                values.insert(matched_key->second, it->value);
            } else {
                logger::warn(
                    "global_mapping",
                    "No remapping found for Value with key {}, keeping it as is",
                    gtsam::Symbol(it->key).string()
                );
                values.insert(it->key, it->value);
            }
        }
    } else {
        graph = loaded_graph;
        values = loaded_values;
    }

    logger::info("global_mapping", "creating matching cost factors");
    for (const auto& factor: matching_cost_factors) {
        const auto type = std::get<0>(factor);
        const auto first = std::get<1>(factor) + start_from_frame_id;
        const auto second = std::get<2>(factor) + start_from_frame_id;

        if (type == "vgicp") {
            for (const auto& voxelmap: submaps[first]->voxelmaps) {
                graph.emplace_shared<gtsam_points::IntegratedVGICPFactor>(
                    X(first),
                    X(second),
                    voxelmap,
                    subsampled_submaps[second]
                );
            }
        } else {
            logger::fatal("global_mapping", "unsupported matching cost factor type ({})", type);
            exit(EXIT_FAILURE);
        }
    }

    const size_t num_factors_before = graph.size();
    const auto remove_loc = std::remove_if(graph.begin(), graph.end(), [](const auto& factor) {
        return factor == nullptr;
    });
    graph.erase(remove_loc, graph.end());
    if (graph.size() != num_factors_before) {
        logger::warn("global_mapping", "removed {} invalid factors", num_factors_before - graph.size());
        needs_recover = true;
    }

    if (needs_recover) {
        logger::warn("global_mapping", "recovering factor graph");
        const auto recovered = recover_graph(graph, values, start_from_frame_id);
        logger::warn("global_mapping", "add {} factors and {} values", recovered.first.size(), recovered.second.size());

        graph.add(recovered.first);
        values.insert_or_assign(recovered.second);
    }

    if (start_from_frame_id <= 0) {
        logger::info("global_mapping", "optimize");
        auto result = update_isam2(graph, values);
        update_submaps();
    } else {
        logger::info("global_mapping", "skip optimization");
        this->new_factors->add(graph);
        this->new_values->insert(values);
    }

    logger::info("global_mapping", "done");
    session_id++;

    return true;
}

void GlobalMapping::recover_graph() {
    const auto recovered = recover_graph(isam2->getFactorsUnsafe(), isam2->calculateEstimate(), 0);
    update_isam2(recovered.first, recovered.second);
}

// Recover the graph by adding missing values and factors
std::pair<gtsam::NonlinearFactorGraph, gtsam::Values> GlobalMapping::recover_graph(
    const gtsam::NonlinearFactorGraph& graph,
    const gtsam::Values& values,
    int start_from_frame_id
) const {
    logger::info("global_mapping", "recovering graph");
    bool enable_imu = false;
    for (const auto& value: values) {
        const char chr = gtsam::Symbol(value.key).chr();
        enable_imu |= (chr == 'e' || chr == 'v' || chr == 'b');
    }
    for (const auto& factor: graph) {
        enable_imu |= dynamic_cast<gtsam::ImuFactor*>(factor.get()) != nullptr;
    }

    logger::info("global_mapping", "enable_imu={}", enable_imu);

    logger::info("global_mapping", "creating connectivity map");
    bool prior_exists = false;
    std::unordered_map<gtsam::Key, std::set<gtsam::Key>> connectivity_map;
    for (const auto& factor: graph) {
        if (!factor) {
            continue;
        }

        for (const auto key: factor->keys()) {
            for (const auto key2: factor->keys()) {
                connectivity_map[key].insert(key2);
            }
        }

        if (factor->keys().size() == 1 && factor->keys()[0] == X(0)) {
            prior_exists |=
                dynamic_cast<gtsam_points::LinearDampingFactor*>(factor.get()) != nullptr;
        }
    }

    logger::info("global_mapping", "fixing missing values and factors");
    const auto prior_noise3 = gtsam::noiseModel::Isotropic::Precision(3, 1e6);
    const auto prior_noise6 = gtsam::noiseModel::Isotropic::Precision(6, 1e6);

    gtsam::NonlinearFactorGraph new_factors;
    gtsam::Values new_values;

    if (!prior_exists) {
        logger::warn("global_mapping", "X0 prior is missing");
        new_factors.emplace_shared<gtsam_points::LinearDampingFactor>(
            X(0),
            6,
            params->init_pose_damping_scale
        );
    }

    for (int i = start_from_frame_id; i < submaps.size(); i++) {
        if (!values.exists(X(i))) {
            logger::warn("global_mapping", "X{} is missing", i);
            new_values.insert(X(i), gtsam::Pose3(submaps[i]->T_world_origin.matrix()));
        }

        if (connectivity_map[X(i)].count(X(i + 1)) == 0 && i != submaps.size() - 1) {
            logger::warn("global_mapping", "X{} -> X{} is missing", i, i + 1);

            const Eigen::Isometry3d delta =
                submaps[i]->origin_odom_frame()->T_world_sensor().inverse()
                * submaps[i + 1]->origin_odom_frame()->T_world_sensor();
            new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(i),
                X(i + 1),
                gtsam::Pose3(delta.matrix()),
                prior_noise6
            );
        }

        if (!enable_imu) {
            continue;
        }

        const auto submap = submaps[i];
        const gtsam::imuBias::ConstantBias imu_biasL(submap->frames.front()->imu_bias);
        const gtsam::imuBias::ConstantBias imu_biasR(submap->frames.back()->imu_bias);
        const Eigen::Vector3d v_origin_imuL = submap->T_world_origin.linear().inverse() * submap->frames.front()->v_world_imu;
        const Eigen::Vector3d v_origin_imuR = submap->T_world_origin.linear().inverse() * submap->frames.back()->v_world_imu;

        if (i != 0) {
            if (!values.exists(E(i * 2))) {
                logger::warn("global_mapping", "E{} is missing", i * 2);
                new_values.insert(
                    E(i * 2),
                    gtsam::Pose3((submap->T_world_origin * submap->T_origin_endpoint_L).matrix())
                );
            }
            if (!values.exists(V(i * 2))) {
                logger::warn("global_mapping", "V{} is missing", i * 2);
                new_values.insert(
                    V(i * 2),
                    (submap->T_world_origin.linear() * v_origin_imuL).eval()
                );
            }
            if (!values.exists(B(i * 2))) {
                logger::warn("global_mapping", "B{} is missing", i * 2);
                new_values.insert(B(i * 2), imu_biasL);
            }

            if (connectivity_map[X(i)].count(E(i * 2)) == 0) {
                logger::warn("global_mapping", "X{} -> E{} is missing", i, i * 2);
                new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                    X(i),
                    E(i * 2),
                    gtsam::Pose3(submap->T_origin_endpoint_L.matrix()),
                    prior_noise6
                );
            }
            if (connectivity_map[X(i)].count(V(i * 2)) == 0) {
                logger::warn("global_mapping", "X{} -> V{} is missing", i, i * 2);
                new_factors.emplace_shared<gtsam_points::RotateVector3Factor>(
                    X(i),
                    V(i * 2),
                    v_origin_imuL,
                    prior_noise3
                );
            }
            if (connectivity_map[B(i * 2)].count(B(i * 2)) == 0) {
                logger::warn("global_mapping", "B{} -> B{} is missing", i * 2, i * 2);
                new_factors.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                    B(i * 2),
                    imu_biasL,
                    prior_noise6
                );
            }

            if (connectivity_map[B(i * 2)].count(B(i * 2 + 1)) == 0) {
                logger::warn("global_mapping", "B{} -> B{} is missing", i * 2, i * 2 + 1);
                new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
                    B(i * 2),
                    B(i * 2 + 1),
                    gtsam::imuBias::ConstantBias(),
                    prior_noise6
                );
            }
        }

        if (!values.exists(E(i * 2 + 1))) {
            logger::warn("global_mapping", "E{} is missing", i * 2 + 1);
            new_values.insert(
                E(i * 2 + 1),
                gtsam::Pose3((submap->T_world_origin * submap->T_origin_endpoint_R).matrix())
            );
        }
        if (!values.exists(V(i * 2 + 1))) {
            logger::warn("global_mapping", "V{} is missing", i * 2 + 1);
            new_values.insert(
                V(i * 2 + 1),
                (submap->T_world_origin.linear() * v_origin_imuR).eval()
            );
        }
        if (!values.exists(B(i * 2 + 1))) {
            logger::warn("global_mapping", "B{} is missing", i * 2 + 1);
            new_values.insert(B(i * 2 + 1), imu_biasR);
        }

        if (connectivity_map[X(i)].count(E(i * 2 + 1)) == 0) {
            logger::warn("global_mapping", "X{} -> E{} is missing", i, i * 2 + 1);
            new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(i),
                E(i * 2 + 1),
                gtsam::Pose3(submap->T_origin_endpoint_R.matrix()),
                prior_noise6
            );
        }
        if (connectivity_map[X(i)].count(V(i * 2 + 1)) == 0) {
            logger::warn("global_mapping", "X{} -> V{} is missing", i, i * 2 + 1);
            new_factors.emplace_shared<gtsam_points::RotateVector3Factor>(
                X(i),
                V(i * 2 + 1),
                v_origin_imuR,
                prior_noise3
            );
        }
        if (connectivity_map[B(i * 2 + 1)].count(B(i * 2 + 1)) == 0) {
            logger::warn("global_mapping", "B{} -> B{} is missing", i * 2 + 1, i * 2 + 1);
            new_factors.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                B(i * 2 + 1),
                imu_biasR,
                prior_noise6
            );
        }
    }

    logger::info("global_mapping", "recovering done");

    return {new_factors, new_values};
}

}