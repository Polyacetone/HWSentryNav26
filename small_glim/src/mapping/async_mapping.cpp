#include <small_glim/mapping/async_mapping.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <format>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <small_glim/common/convert_to_string.hpp>
#include <small_glim/common/logger.hpp>
#include <small_glim/preprocess/cloud_deskewing.hpp>

namespace fs = std::filesystem;

namespace {
using gtsam::symbol_shorthand::X;

inline Eigen::Array3i fast_floor(const Eigen::Vector3d& pt) {
    Eigen::Array3d arr = pt.array();
    Eigen::Array3i ncoord = arr.cast<int>();
    return ncoord - (arr < ncoord.cast<double>()).cast<int>();
}

struct ImuTrajectory {
    std::vector<double> times;
    std::vector<Eigen::Isometry3d> poses;
};

std::optional<ImuTrajectory> parse_imu_rate_trajectory(const Eigen::Matrix<double, 8, -1>& traj) {
    if (traj.cols() < 2) {
        return std::nullopt;
    }

    ImuTrajectory out;
    out.times.resize(traj.cols());
    out.poses.resize(traj.cols());
    for (int i = 0; i < traj.cols(); i++) {
        const Eigen::Matrix<double, 8, 1> imu = traj.col(i);
        out.times[i] = imu[0];
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() << imu[1], imu[2], imu[3];
        T.linear() = Eigen::Quaterniond(imu[7], imu[4], imu[5], imu[6]).toRotationMatrix();
        out.poses[i] = T;
    }

    // Basic monotonicity check
    for (std::size_t i = 1; i < out.times.size(); i++) {
        if (!(out.times[i] > out.times[i - 1])) {
            return std::nullopt;
        }
    }

    return out;
}

std::optional<ImuTrajectory> optimize_imu_trajectory_with_end_priors(
    const ImuTrajectory& initial,
    const Eigen::Isometry3d& T_world_imu_begin,
    const Eigen::Isometry3d& T_world_imu_end,
    const int max_iterations
) {
    if (initial.times.size() < 2 || initial.times.size() != initial.poses.size()) {
        return std::nullopt;
    }

    gtsam::Values values;
    for (int i = 0; i < static_cast<int>(initial.times.size()); i++) {
        values.insert(X(i), gtsam::Pose3(initial.poses[i].matrix()));
    }

    gtsam::NonlinearFactorGraph graph;
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        X(0),
        gtsam::Pose3(T_world_imu_begin.matrix()),
        gtsam::noiseModel::Isotropic::Sigma(6, 1e-5)
    );
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        X(static_cast<int>(initial.times.size() - 1)),
        gtsam::Pose3(T_world_imu_end.matrix()),
        gtsam::noiseModel::Isotropic::Sigma(6, 1e-5)
    );

    const double total_dt = std::max(1e-6, initial.times.back() - initial.times.front());
    for (int i = 1; i < static_cast<int>(initial.times.size()); i++) {
        const double dt_norm = (initial.times[i] - initial.times[i - 1]) / total_dt;
        const Eigen::Isometry3d T_last_curr = initial.poses[i - 1].inverse() * initial.poses[i];
        graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(i - 1),
            X(i),
            gtsam::Pose3(T_last_curr.matrix()),
            gtsam::noiseModel::Isotropic::Sigma(6, dt_norm + 1e-2)
        );
    }

    gtsam::LevenbergMarquardtParams lm_params;
    lm_params.setAbsoluteErrorTol(1e-6);
    lm_params.setRelativeErrorTol(1e-6);
    lm_params.setMaxIterations(std::max(1, max_iterations));

    gtsam::Values optimized = gtsam::LevenbergMarquardtOptimizer(graph, values, lm_params).optimize();

    ImuTrajectory out;
    out.times = initial.times;
    out.poses.resize(initial.poses.size());
    for (int i = 0; i < static_cast<int>(out.poses.size()); i++) {
        out.poses[i] = Eigen::Isometry3d(optimized.at<gtsam::Pose3>(X(i)).matrix());
    }

    return out;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr voxelgrid_sampling(const pcl::PointCloud<pcl::PointXYZ>& input, double leaf_size) {
    if (input.empty()) {
        return std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }

    const double inv_leaf_size = 1.0 / leaf_size;

    constexpr std::uint64_t invalid_coord = std::numeric_limits<std::uint64_t>::max();
    constexpr int coord_bit_size = 21; // 21 bits per axis → 63 bits total
    constexpr std::uint64_t coord_bit_mask = (1ULL << coord_bit_size) - 1;
    constexpr int coord_offset = 1 << (coord_bit_size - 1); // to make coords non-negative

    std::vector<std::pair<std::uint64_t, size_t>> coord_pt;
    coord_pt.reserve(input.size());

    for (size_t i = 0; i < input.size(); i++) {
        const auto& p = input.points[i];
        // Skip NaN points
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            coord_pt.emplace_back(invalid_coord, i);
            continue;
        }

        Eigen::Vector3d pt(p.x, p.y, p.z);
        Eigen::Array3i coord = fast_floor(pt * inv_leaf_size) + coord_offset;

        if ((coord < 0).any() || (coord > static_cast<int>(coord_bit_mask)).any()) {
            std::cerr << "Warning: voxel coordinate out of range!" << std::endl;
            coord_pt.emplace_back(invalid_coord, i);
            continue;
        }

        // Pack x, y, z into uint64_t: [unused(1b)][z(21b)][y(21b)][x(21b)]
        std::uint64_t bits = (static_cast<std::uint64_t>(coord[0] & coord_bit_mask) << (0 * coord_bit_size))
            | (static_cast<std::uint64_t>(coord[1] & coord_bit_mask) << (1 * coord_bit_size))
            | (static_cast<std::uint64_t>(coord[2] & coord_bit_mask) << (2 * coord_bit_size));

        coord_pt.emplace_back(bits, i);
    }

    // Sort by voxel key
    std::sort(coord_pt.begin(), coord_pt.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    auto output = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    output->reserve(input.size());

    size_t i = 0;
    while (i < coord_pt.size()) {
        if (coord_pt[i].first == invalid_coord) {
            i++;
            continue;
        }

        std::uint64_t current_voxel = coord_pt[i].first;
        Eigen::Vector3d sum(0, 0, 0);
        int count = 0;

        // Accumulate all points in the same voxel
        while (i < coord_pt.size() && coord_pt[i].first == current_voxel) {
            const auto& p = input.points[coord_pt[i].second];
            sum += Eigen::Vector3d(p.x, p.y, p.z);
            count++;
            i++;
        }

        // Compute centroid
        sum /= static_cast<double>(count);
        output->push_back(
            pcl::PointXYZ(static_cast<float>(sum.x()), static_cast<float>(sum.y()), static_cast<float>(sum.z()))
        );
    }

    return output;
}
}

namespace small_glim {

AsyncMappingParams::AsyncMappingParams(const Config::Ptr& config) {
    save_raw_mapping_frames = config->param<bool>("mapping.save_raw_mapping_frames");
    output_root = config->param<std::string>("mapping.output_root");

    enable_imu_refine = config->param<bool>("mapping.enable_imu_refine");
    imu_refine_max_iterations = config->param<int>("mapping.imu_refine_max_iterations");

    keyframe_trans_thresh = config->param<double>("mapping.keyframe_trans_thresh");
    keyframe_rot_thresh = config->param<double>("mapping.keyframe_rot_thresh");
    map_voxel_leaf_size = config->param<double>("mapping.map_voxel_leaf_size");
    downsample_every_n_keyframes = config->param<int>("mapping.downsample_every_n_keyframes");
}

AsyncMapping::AsyncMapping(const Config::Ptr& config):
    params_(config),
    output_dir_(fs::path(params_.output_root) / ("mapping_" + make_timestamp_string())),
    map_cloud_(std::make_shared<pcl::PointCloud<pcl::PointXYZ>>()) {
    ensure_output_dir();

    if (params_.save_raw_mapping_frames) {
        poses_ofs_.open(fs::path(output_dir_) / "poses.txt", std::ios::out);
        if (!poses_ofs_) {
            throw std::runtime_error(std::format("failed to open poses file in {}", output_dir_));
        }
    }

    logger::info("mapping", "output_dir={} save_raw_mapping_frames={}", output_dir_, params_.save_raw_mapping_frames);
    worker_ = std::thread([this]() { worker_loop(); });
}

AsyncMapping::~AsyncMapping() {
    request_finish();
    join();
}

void AsyncMapping::insert_frame(const EstimationFrame::ConstPtr& frame) {
    if (!frame) return;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(frame);
    }
    cv_.notify_one();
}

void AsyncMapping::request_finish() {
    finish_requested_.store(true);
    cv_.notify_all();
}

void AsyncMapping::join() {
    if (joined_.exchange(true)) {
        return;
    }
    if (worker_.joinable()) {
        worker_.join();
    }
}

void AsyncMapping::worker_loop() {
    std::deque<EstimationFrame::ConstPtr> delayed_frames;

    while (true) {
        EstimationFrame::ConstPtr frame;
        {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_.wait(lk, [&]() { return finish_requested_.load() || !queue_.empty(); });
            if (!queue_.empty()) {
                frame = queue_.front();
                queue_.pop_front();
            } else if (finish_requested_.load()) {
                break;
            }
        }

        if (frame) {
            delayed_frames.push_back(frame);
        }

        while (delayed_frames.size() >= 2) {
            const auto current = delayed_frames.front();
            const auto next = delayed_frames[1];
            process_frame(*current, next.get());
            delayed_frames.pop_front();
        }
    }

    // Flush the last frame(s) without a "next" constraint
    while (!delayed_frames.empty()) {
        process_frame(*delayed_frames.front(), nullptr);
        delayed_frames.pop_front();
    }

    downsample_map_if_needed();
    save_final_map();
    if (poses_ofs_.is_open()) {
        poses_ofs_.flush();
        poses_ofs_.close();
    }
}

void AsyncMapping::process_frame(const EstimationFrame& frame, const EstimationFrame* next_frame) {
    if (frame.deskew_imu_saturated) {
        logger::info("mapping", "skip saturated frame (id={}, stamp={:.6f})", frame.id, frame.stamp);
        return;
    }

    const bool has_points = frame.frame && frame.frame->size() > 0;
    const bool has_raw = frame.raw_frame && !frame.raw_frame->points.empty();
    if (!has_points && !has_raw) {
        logger::warn("mapping", "skip empty frame (id={}, stamp={:.6f})", frame.id, frame.stamp);
        return;
    }

    if (!is_keyframe(frame)) {
        return;
    }

    auto cloud_imu = build_keyframe_cloud_imu(frame, next_frame);
    if (!cloud_imu || cloud_imu->empty()) {
        logger::warn("mapping", "skip keyframe because cloud is empty (id={}, stamp={:.6f})", frame.id, frame.stamp);
        return;
    }

    accept_keyframe(frame, *cloud_imu);
}

bool AsyncMapping::is_keyframe(const EstimationFrame& frame) const {
    const Eigen::Isometry3d T_world_imu = frame.T_world_imu;

    if (!has_last_keyframe_) {
        return true;
    }

    const Eigen::Isometry3d T_last_curr = last_keyframe_T_world_frame_.inverse() * T_world_imu;
    const double trans = T_last_curr.translation().norm();
    const double rot_deg = rotation_angle_deg(T_last_curr.linear());

    return (trans > params_.keyframe_trans_thresh) || (rot_deg > params_.keyframe_rot_thresh);
}

void AsyncMapping::accept_keyframe(const EstimationFrame& frame, const pcl::PointCloud<pcl::PointXYZ>& keyframe_cloud_imu) {
    if (frame.frame_type != FrameType::IMU) {
        logger::fatal("mapping", "only IMU frames are supported for mapping; skip frame_id={}", frame.id);
        std::exit(EXIT_FAILURE);
    }

    const Eigen::Isometry3d T_world_imu = frame.T_world_imu;

    if (params_.save_raw_mapping_frames) {
        save_keyframe_raw_cloud(keyframe_cloud_imu);
        append_pose(T_world_imu);
    }

    integrate_cloud_into_map(keyframe_cloud_imu, T_world_imu);
    keyframes_since_downsample_++;
    downsample_map_if_needed();

    has_last_keyframe_ = true;
    last_keyframe_T_world_frame_ = T_world_imu;
    keyframe_count_++;

    logger::debug(
        "mapping",
        "accept keyframe idx={} (frame_id={}, stamp={:.6f}) map_points={}",
        keyframe_count_ - 1,
        frame.id,
        frame.stamp,
        map_cloud_->size()
    );
}

void AsyncMapping::ensure_output_dir() {
    fs::path out_dir(output_dir_);
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) {
        throw std::runtime_error(std::format("failed to create mapping output dir {} ({})", output_dir_, ec.message()));
    }
}

void AsyncMapping::save_keyframe_raw_cloud(const pcl::PointCloud<pcl::PointXYZ>& cloud_imu) const {
    const fs::path filepath = fs::path(output_dir_) / std::format("frame_{}.pcd", keyframe_count_);
    if (pcl::io::savePCDFileBinary(filepath.string(), cloud_imu) != 0) {
        logger::warn("mapping", "failed to save {}", filepath.string());
    }
}

void AsyncMapping::append_pose(const Eigen::Isometry3d& T_world_imu) {
    poses_ofs_ << convert_to_string(T_world_imu) << "\n";
}

void AsyncMapping::integrate_cloud_into_map(const pcl::PointCloud<pcl::PointXYZ>& cloud_imu, const Eigen::Isometry3d& T_world_imu) {
    map_cloud_->reserve(map_cloud_->size() + cloud_imu.size());
    for (const auto& p_imu : cloud_imu.points) {
        const Eigen::Vector3d p_world = T_world_imu * Eigen::Vector3d(p_imu.x, p_imu.y, p_imu.z);
        map_cloud_->emplace_back(static_cast<float>(p_world.x()), static_cast<float>(p_world.y()), static_cast<float>(p_world.z()));
    }
}

std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> AsyncMapping::build_keyframe_cloud_imu(
    const EstimationFrame& frame,
    const EstimationFrame* next_frame
) {
    // Prefer refined redeskewing (raw_frame + imu_rate_trajectory + next_frame constraint)
    if (params_.enable_imu_refine && next_frame && frame.raw_frame) {
        if (frame.imu_rate_trajectory.cols() >= 2 && next_frame->stamp > frame.stamp) {
            // Only apply end prior if trajectory end is reasonably close to next frame stamp.
            const auto traj_opt = parse_imu_rate_trajectory(frame.imu_rate_trajectory);
            if (traj_opt) {
                const double end_mismatch = std::abs(traj_opt->times.back() - next_frame->stamp);
                if (end_mismatch < 0.05) {
                    try {
                        const auto optimized = optimize_imu_trajectory_with_end_priors(
                            *traj_opt,
                            frame.T_world_imu,
                            next_frame->T_world_imu,
                            params_.imu_refine_max_iterations
                        );
                        if (optimized) {
                            if (frame.raw_frame->scan_end_time > optimized->times.back() + 1e-3) {
                                logger::warn(
                                    "mapping",
                                    "imu_refine trajectory does not cover scan duration (frame_id={} imu_end={:.6f} scan_end={:.6f}); fallback",
                                    frame.id,
                                    optimized->times.back(),
                                    frame.raw_frame->scan_end_time
                                );
                            } else {
                                CloudDeskewing deskewing;
                                const Eigen::Isometry3d T_imu_lidar = frame.T_lidar_imu.inverse();
                                auto deskewed_lidar = deskewing.deskew(
                                    T_imu_lidar,
                                    optimized->times,
                                    optimized->poses,
                                    frame.raw_frame->stamp,
                                    frame.raw_frame->times,
                                    frame.raw_frame->points
                                );

                                auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                                cloud->reserve(deskewed_lidar.size());
                                for (const auto& p_lidar : deskewed_lidar) {
                                    const Eigen::Vector4d p_imu = T_imu_lidar * p_lidar;
                                    cloud->emplace_back(static_cast<float>(p_imu.x()), static_cast<float>(p_imu.y()), static_cast<float>(p_imu.z()));
                                }
                                return cloud;
                            }
                        }
                    } catch (const std::exception& e) {
                        logger::warn("mapping", "imu_refine failed (frame_id={}): {}", frame.id, e.what());
                    }
                } else {
                    logger::debug(
                        "mapping",
                        "skip imu_refine due to end stamp mismatch (frame_id={} traj_end={:.6f} next_stamp={:.6f})",
                        frame.id,
                        traj_opt->times.back(),
                        next_frame->stamp
                    );
                }
            }
        }
    }

    // Fallback: use existing deskewed estimation cloud (expected to be in IMU frame)
    if (!frame.frame || frame.frame->size() <= 0) {
        return std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->reserve(frame.frame->size());
    const auto& points = frame.frame->points;
    for (int i = 0; i < frame.frame->size(); i++) {
        const Eigen::Vector4d p = points[i];
        cloud->emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z()));
    }
    return cloud;
}

void AsyncMapping::downsample_map_if_needed() {
    if (map_cloud_->empty()) return;
    if (params_.downsample_every_n_keyframes <= 0) return;
    if (keyframes_since_downsample_ < params_.downsample_every_n_keyframes) return;
    if (params_.map_voxel_leaf_size <= 0.0) return;

    const std::size_t before = map_cloud_->size();
    map_cloud_ = voxelgrid_sampling(*map_cloud_, params_.map_voxel_leaf_size);
    const std::size_t after = map_cloud_->size();

    logger::debug(
        "mapping",
        "downsample map leaf={:.3f} points={} -> {}",
        params_.map_voxel_leaf_size,
        before,
        after
    );

    keyframes_since_downsample_ = 0;
}

void AsyncMapping::save_final_map() {
    if (map_cloud_->empty()) {
        logger::warn("mapping", "final map is empty; skip saving mapping.pcd");
        return;
    }

    const std::size_t before = map_cloud_->size();
    map_cloud_ = voxelgrid_sampling(*map_cloud_, params_.map_voxel_leaf_size);
    const std::size_t after = map_cloud_->size();

    logger::debug(
        "mapping",
        "downsample map leaf={:.3f} points={} -> {}",
        params_.map_voxel_leaf_size,
        before,
        after
    );

    const fs::path filepath = fs::path(output_dir_) / "mapping.pcd";
    if (pcl::io::savePCDFileBinary(filepath.string(), *map_cloud_) != 0) {
        logger::warn("mapping", "failed to save {}", filepath.string());
    } else {
        logger::debug("mapping", "saved final map {} (points={})", filepath.string(), map_cloud_->size());
    }
}

std::string AsyncMapping::make_timestamp_string() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&tt, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

double AsyncMapping::rotation_angle_deg(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd aa(R);
    double angle = aa.angle();
    if (angle > M_PI) {
        angle = 2.0 * M_PI - angle;
    }
    return angle * 180.0 / M_PI;
}

}