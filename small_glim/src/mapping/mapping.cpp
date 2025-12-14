#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam_points/config.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>

#include <small_glim/mapping/mapping.hpp>
#include <small_glim/common/logger.hpp>
#include <small_glim/common/convert_to_string.hpp>

namespace small_glim {

using gtsam::symbol_shorthand::X;

MappingParams::MappingParams(const Config::Ptr config) {
    num_threads = config->param<int>("mapping.num_threads");
    enable_optimization = config->param<bool>("mapping.enable_optimization");
    downsample_resolution = config->param<double>("mapping.downsample_resolution");
    std::string strategy = config->param<std::string>("mapping.keyframe_update_strategy");
    if (strategy == "DISPLACEMENT") {
        keyframe_update_strategy = KeyframeUpdateStrategy::DISPLACEMENT;
    } else if (strategy == "OVERLAP") {
        keyframe_update_strategy = KeyframeUpdateStrategy::OVERLAP;
    } else {
        logger::fatal("mapping", "unknown keyframe update strategy: {}", strategy);
        exit(EXIT_FAILURE);
    }
    keyframe_update_min_dist = config->param<double>("mapping.keyframe_update_min_dist");
    keyframe_update_min_rot = config->param<double>("mapping.keyframe_update_min_rot");
    max_keyframe_overlap = config->param<double>("mapping.max_keyframe_overlap");
    std::string reg_type = config->param<std::string>("mapping.registration_type");
    if (reg_type == "GICP") {
        registration_type = RegistrationType::GICP;
    } else if (reg_type == "VGICP") {
        registration_type = RegistrationType::VGICP;
    } else {
        logger::fatal("mapping", "unknown registration type: {}", reg_type);
        exit(EXIT_FAILURE);
    }
    max_correspondence_distance = config->param<double>("mapping.max_correspondence_distance");
    keyframe_randomsampling_rate = config->param<double>("mapping.keyframe_randomsampling_rate");
    keyframe_voxel_resolution = config->param<double>("mapping.keyframe_voxel_resolution");
    keyframe_voxelmap_levels = config->param<int>("mapping.keyframe_voxelmap_levels");
    keyframe_voxelmap_scaling_factor = config->param<double>("mapping.keyframe_voxelmap_scaling_factor");
}

Mapping::Mapping(const Config::Ptr config) {
    params = std::make_unique<MappingParams>(config);
    graph = std::make_unique<gtsam::NonlinearFactorGraph>();
    values = std::make_unique<gtsam::Values>();
    last_keyframe_pose = Eigen::Isometry3d::Identity();
}

void Mapping::insert_frame(const EstimationFrame::ConstPtr& frame) {
    if (frame->frame_type != FrameType::IMU) {
        logger::fatal("mapping", "only IMU frames can be inserted into mapping");
        exit(EXIT_FAILURE);
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (is_keyframe(frame)) {
        logger::debug("mapping", "adding new keyframe, total keyframes: {}", keyframes.size() + 1);
        add_keyframe(frame);
    }
}

bool Mapping::is_keyframe(const EstimationFrame::ConstPtr& frame) {
    if (keyframes.empty()) return true;
    switch (params->keyframe_update_strategy) {
        case MappingParams::KeyframeUpdateStrategy::DISPLACEMENT: {
            Eigen::Isometry3d delta = last_keyframe_pose.inverse() * frame->T_world_frame();
            double dist = delta.translation().norm();
            double rot = Eigen::AngleAxisd(delta.linear()).angle();
            return dist > params->keyframe_update_min_dist || rot > params->keyframe_update_min_rot;
        }
        case MappingParams::KeyframeUpdateStrategy::OVERLAP: {
            if (params->registration_type != MappingParams::RegistrationType::VGICP) {
                logger::fatal("mapping", "OVERLAP strategy is only supported with VGICP");
                exit(EXIT_FAILURE);
            }
            const double overlap = gtsam_points::overlap(
                keyframes.back()->voxelmaps.back(),
                frame->frame,
                keyframes.back()->T_world_frame().inverse() * frame->T_world_frame()
            );
            return overlap < params->max_keyframe_overlap;
        }
    }
    __builtin_unreachable();
}

void Mapping::add_keyframe(const EstimationFrame::ConstPtr& frame) {    
    // Random sampling
    auto subsampled = gtsam_points::random_sampling(frame->frame, params->keyframe_randomsampling_rate, mt);

    auto keyframe = std::make_shared<EstimationFrame>(*frame);
    keyframe->frame = subsampled;

    // Create voxelmaps for VGICP
    if (params->registration_type == MappingParams::RegistrationType::VGICP) {
        keyframe->voxelmaps.clear();
        for (int i = 0; i < params->keyframe_voxelmap_levels; i++) {
            const double resolution = params->keyframe_voxel_resolution * std::pow(params->keyframe_voxelmap_scaling_factor, i);
            auto voxelmap = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(resolution);
            voxelmap->insert(*keyframe->frame);
            keyframe->voxelmaps.push_back(voxelmap);
        }
    }
    
    int current_idx = keyframes.size();
    keyframes.push_back(keyframe);
    keyframe_indices.push_back(current_idx);
    
    values->insert(X(current_idx), gtsam::Pose3(frame->T_world_frame().matrix()));
    last_keyframe_pose = frame->T_world_frame();

    if (current_idx == 0) {
        auto noise = gtsam::noiseModel::Isotropic::Precision(6, 1e6);
        graph->add(gtsam::PriorFactor<gtsam::Pose3>(X(0), gtsam::Pose3(frame->T_world_frame().matrix()), noise));
    } else {
        // Factor to the first keyframe (Global consistency)
        auto target = keyframes[0];
        auto source = keyframe;
        
        switch (params->registration_type) {
            case MappingParams::RegistrationType::GICP: {
                auto factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(
                    X(0), X(current_idx), target->frame, source->frame
                );
                factor->set_max_correspondence_distance(params->max_correspondence_distance);
                factor->set_num_threads(params->num_threads);
                graph->add(factor);
                break;
            }
            case MappingParams::RegistrationType::VGICP: {
                for (int level = 0; level < params->keyframe_voxelmap_levels; level++) {
                    auto factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(
                        X(0), X(current_idx), target->voxelmaps[level], source->frame
                    );
                    factor->set_num_threads(params->num_threads);
                    graph->add(factor);
                }
                break;
            }
        }
        
        // Factor to the previous keyframe (Odometry consistency)
        gtsam::Pose3 T_prev_curr( (keyframes[current_idx-1]->T_world_frame().inverse() * frame->T_world_frame()).matrix() );
        auto odom_noise = gtsam::noiseModel::Isotropic::Precision(6, 100.0);
        graph->add(gtsam::BetweenFactor<gtsam::Pose3>(X(current_idx-1), X(current_idx), T_prev_curr, odom_noise));
    }

    if (params->enable_optimization) {
        optimize();
    }
}

void Mapping::optimize() {
    if (graph->empty()) return;
    
    gtsam::LevenbergMarquardtParams lm_params;
    lm_params.setMaxIterations(5);
    gtsam::LevenbergMarquardtOptimizer optimizer(*graph, *values, lm_params);
    *values = optimizer.optimize();
}

void Mapping::save(const std::string& path) {
    mutex.lock();    
    std::vector<gtsam_points::PointCloud::ConstPtr> frames;
    std::vector<Eigen::Isometry3d> poses;
    for (const auto& keyframe : keyframes) {
        frames.push_back(keyframe->frame);
        poses.push_back(keyframe->T_world_frame());
    }
    mutex.unlock();

    auto merged = gtsam_points::merge_frames(poses, frames, params->downsample_resolution);
    auto pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl_cloud->resize(merged->size());
    for (int i = 0; i < merged->size(); i++) {
        const auto& pt = merged->points[i];
        pcl_cloud->at(i).x = pt.x();
        pcl_cloud->at(i).y = pt.y();
        pcl_cloud->at(i).z = pt.z();
    }
    pcl_cloud->width = merged->size();
    pcl_cloud->height = 1;
    pcl_cloud->is_dense = false;
    pcl::io::savePCDFileBinary(path, *pcl_cloud);
    logger::info("mapping", "saved cloud with {} points to {}", pcl_cloud->size(), path);
}

void Mapping::save_raw_frames(const std::string& dir) {
    mutex.lock();
    std::vector<gtsam_points::PointCloud::ConstPtr> frames;
    for (const auto& keyframe : keyframes) {
        frames.push_back(keyframe->frame);
    }
    mutex.unlock();

    std::fstream fs;
    fs.open(dir + "/poses.txt", std::ios::out);
    for (int i = 0; i < keyframes.size(); i++) {
        const auto& keyframe = keyframes[i];
        const auto& T = keyframe->T_world_frame();
        fs << convert_to_string(T) << std::endl;
        
        std::string filename = dir + "/frame_" + std::to_string(i) + ".pcd";
        auto pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl_cloud->resize(keyframe->frame->size());
        for (int j = 0; j < keyframe->frame->size(); j++) {
            const auto& pt = keyframe->frame->points[j];
            pcl_cloud->at(j).x = pt.x();
            pcl_cloud->at(j).y = pt.y();
            pcl_cloud->at(j).z = pt.z();
        }
        pcl_cloud->width = keyframe->frame->size();
        pcl_cloud->height = 1;
        pcl_cloud->is_dense = false;
        pcl::io::savePCDFileBinary(filename, *pcl_cloud);
    }
    fs.close();
    logger::info("mapping", "saved {} raw frames to {}", keyframes.size(), dir);
}

}