#include <small_glim/preprocess/cloud_preprocessor.hpp>
#include <small_glim/common/logger.hpp>
#include <small_glim/common/convert_to_string.hpp>
#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/util/parallelism.hpp>

namespace small_glim {

CloudPreprocessorParams::CloudPreprocessorParams(const Config::Ptr config) {
    global_shutter = config->param<bool>("sensors.global_shutter_lidar");
    distance_near_thresh = config->param<double>("preprocess.distance_near_thresh");
    distance_far_thresh = config->param<double>("preprocess.distance_far_thresh");
    use_random_grid_downsampling = config->param<bool>("preprocess.use_random_grid_downsampling");
    downsample_resolution = config->param<double>("preprocess.downsample_resolution");
    downsample_target = config->param<int>("preprocess.random_downsample_target");
    downsample_rate = config->param<double>("preprocess.random_downsample_rate");
    enable_outlier_removal = config->param<bool>("preprocess.enable_outlier_removal");
    outlier_removal_k = config->param<int>("preprocess.outlier_removal_k");
    outlier_std_mul_factor = config->param<double>("preprocess.outlier_std_mul_factor");
    enable_cropbox_filter = config->param<bool>("preprocess.enable_cropbox_filter");

    if (enable_cropbox_filter) {
        Eigen::Isometry3d T_lidar_imu = config->param<Eigen::Isometry3d>("sensors.T_lidar_imu");
        T_imu_lidar = T_lidar_imu.inverse();
        crop_bbox_min = config->param<Eigen::Vector3d>("preprocess.crop_bbox_min");
        crop_bbox_max = config->param<Eigen::Vector3d>("preprocess.crop_bbox_max");
        std::string crop_bbox_frame_str = config->param<std::string>("preprocess.crop_bbox_frame");
        if (crop_bbox_frame_str == "lidar") {
            crop_bbox_frame = CropBBoxFrame::LIDAR;
        } else if (crop_bbox_frame_str == "imu") {
            crop_bbox_frame = CropBBoxFrame::IMU;
        } else {
            logger::fatal("cloud_preprocess", "Unsupported crop bbox frame: {}", crop_bbox_frame_str);
            std::exit(EXIT_FAILURE);
        }
        if ((crop_bbox_min.array() > crop_bbox_max.array()).any()) {
            logger::fatal(
                "cloud_preprocess",
                "Misconfigured bbox: min=[{}], max=[{}]",
                convert_to_string(crop_bbox_min),
                convert_to_string(crop_bbox_max)
            );
            std::exit(EXIT_FAILURE);
        }
    }

    k_correspondences = config->param<int>("preprocess.k_correspondences");
    num_threads = config->param<int>("preprocess.num_threads");
}

CloudPreprocessor::CloudPreprocessor(const Config::Ptr config) {
    params = std::make_unique<CloudPreprocessorParams>(config);
}

PreprocessedFrame::Ptr CloudPreprocessor::preprocess(const RawPoints::ConstPtr raw_points) {
    auto frame = std::make_shared<gtsam_points::PointCloud>();
    frame->num_points = raw_points->size();
    frame->times = const_cast<double*>(raw_points->times.data());
    frame->points = const_cast<Eigen::Vector4d*>(raw_points->points.data());
    if (raw_points->intensities.size()) {
        frame->intensities = const_cast<double*>(raw_points->intensities.data());
    }

    // Downsampling
    if (params->use_random_grid_downsampling) {
        const double rate = params->downsample_target > 0
            ? static_cast<double>(params->downsample_target) / static_cast<double>(frame->size())
            : params->downsample_rate;
        frame = gtsam_points::randomgrid_sampling(
            frame,
            params->downsample_resolution,
            rate,
            mt,
            params->num_threads
        );
    } else {
        frame = gtsam_points::voxelgrid_sampling(
            frame,
            params->downsample_resolution,
            params->num_threads
        );
    }

    if (frame->size() < 100) {
        logger::warn("cloud_preprocess", "Too few points in the downsampled cloud ({} points)", frame->size());
    }

    // Distance filter
    std::vector<int> indices;
    indices.reserve(frame->size());
    double squared_distance_near_thresh = params->distance_near_thresh * params->distance_near_thresh;
    double squared_distance_far_thresh = params->distance_far_thresh * params->distance_far_thresh;

    for (size_t i = 0; i < frame->size(); i++) {
        const bool is_finite = frame->points[i].allFinite();
        const double squared_dist = (Eigen::Vector4d() << frame->points[i].head<3>(), 0.0).finished().squaredNorm();
        if (squared_dist > squared_distance_near_thresh && squared_dist < squared_distance_far_thresh && is_finite) {
            indices.push_back(static_cast<int>(i));
        }
    }

    if (indices.size() < 100) {
        logger::warn("cloud_preprocess", "Too few points in the filtered cloud ({} points)", indices.size());
    }

    // Sort by time
    std::sort(indices.begin(), indices.end(), [&](const int lhs, const int rhs) {
        return frame->times[lhs] < frame->times[rhs];
    });
    frame = gtsam_points::sample(frame, indices);

    if (params->global_shutter) {
        std::fill(frame->times, frame->times + frame->size(), 0.0);
    }

    // Cropbox filter
    if (params->enable_cropbox_filter) {
        if (params->crop_bbox_frame == CloudPreprocessorParams::CropBBoxFrame::LIDAR) {
            auto is_inside_bbox = [&](const Eigen::Vector3d& p_lidar) {
                return (p_lidar.array() >= params->crop_bbox_min.array()).all()
                    && (p_lidar.array() <= params->crop_bbox_max.array()).all();
            };

            frame = gtsam_points::filter(frame, [&](const auto& pt) {
                return !is_inside_bbox(pt.template head<3>());
            });
        } else if (params->crop_bbox_frame == CloudPreprocessorParams::CropBBoxFrame::IMU) {
            auto is_inside_bbox = [&](const Eigen::Vector3d& p_lidar) {
                const auto p_imu = params->T_imu_lidar * p_lidar;
                return (p_imu.array() >= params->crop_bbox_min.array()).all()
                    && (p_imu.array() <= params->crop_bbox_max.array()).all();
            };

            frame = gtsam_points::filter(frame, [&](const auto& pt) {
                return !is_inside_bbox(pt.template head<3>());
            });
        }
    }

    // Outlier removal
    if (params->enable_outlier_removal) {
        frame = gtsam_points::remove_outliers(
            frame,
            params->outlier_removal_k,
            params->outlier_std_mul_factor,
            params->num_threads
        );
    }

    // Create a preprocessed frame
    auto preprocessed = std::make_shared<PreprocessedFrame>();
    preprocessed->stamp = raw_points->stamp;
    preprocessed->scan_end_time = frame->size() ? raw_points->stamp + frame->times[frame->size() - 1] : raw_points->stamp;

    preprocessed->times.assign(frame->times, frame->times + frame->size());
    preprocessed->points.assign(frame->points, frame->points + frame->size());
    if (frame->intensities) {
        preprocessed->intensities.assign(frame->intensities, frame->intensities + frame->size());
    }

    preprocessed->k_neighbors = static_cast<size_t>(params->k_correspondences);
    preprocessed->neighbors = find_neighbors(frame->points, frame->size(), static_cast<size_t>(params->k_correspondences));

    logger::debug("cloud_preprocess", "Preprocessed: {} -> {} points", raw_points->size(), preprocessed->size());
    return preprocessed;
}

std::vector<size_t> CloudPreprocessor::find_neighbors(
    const Eigen::Vector4d* points,
    const size_t num_points,
    const size_t k
) const {
    gtsam_points::KdTree tree(points, static_cast<int>(num_points));
    std::vector<size_t> neighbors(num_points * k);

    const auto perpoint_task = [&](size_t i) {
        std::vector<size_t> k_indices(k);
        std::vector<double> k_sq_dists(k);
        tree.knn_search(points[i].data(), k, k_indices.data(), k_sq_dists.data());
        std::copy(k_indices.begin(), k_indices.end(), neighbors.begin() + static_cast<int64_t>(i) * static_cast<int64_t>(k));
    };

    #pragma omp parallel for num_threads(params->num_threads) schedule(guided, 8)
    for (size_t i = 0; i < num_points; i++) {
        perpoint_task(i);
    }

    return neighbors;
}

}