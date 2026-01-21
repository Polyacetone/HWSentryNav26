#include <small_glim/mapping/async_mapping.hpp>

#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <format>
#include <iomanip>
#include <sstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <small_glim/common/convert_to_string.hpp>
#include <small_glim/common/logger.hpp>

namespace fs = std::filesystem;

namespace {
inline Eigen::Array3i fast_floor(const Eigen::Vector3d& pt) {
    Eigen::Array3d arr = pt.array();
    Eigen::Array3i ncoord = arr.cast<int>();
    return ncoord - (arr < ncoord.cast<double>()).cast<int>();
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

    for (size_t i = 0; i < input.size(); ++i) {
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
            ++i;
            continue;
        }

        std::uint64_t current_voxel = coord_pt[i].first;
        Eigen::Vector3d sum(0, 0, 0);
        int count = 0;

        // Accumulate all points in the same voxel
        while (i < coord_pt.size() && coord_pt[i].first == current_voxel) {
            const auto& p = input.points[coord_pt[i].second];
            sum += Eigen::Vector3d(p.x, p.y, p.z);
            ++count;
            ++i;
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

        if (!frame) {
            continue;
        }

        if (frame->deskew_imu_saturated) {
            logger::info("mapping", "skip saturated frame (id={}, stamp={:.6f})", frame->id, frame->stamp);
            continue;
        }

        if (!frame->frame || frame->frame->size() <= 0) {
            logger::warn("mapping", "skip empty frame (id={}, stamp={:.6f})", frame->id, frame->stamp);
            continue;
        }

        if (!is_keyframe(*frame)) {
            continue;
        }

        accept_keyframe(*frame);
    }

    downsample_map_if_needed();
    save_final_map();
    if (poses_ofs_.is_open()) {
        poses_ofs_.flush();
        poses_ofs_.close();
    }
}

bool AsyncMapping::is_keyframe(const EstimationFrame& frame) const {
    const Eigen::Isometry3d T_world_frame = frame.T_world_frame();

    if (!has_last_keyframe_) {
        return true;
    }

    const Eigen::Isometry3d T_last_curr = last_keyframe_T_world_frame_.inverse() * T_world_frame;
    const double trans = T_last_curr.translation().norm();
    const double rot_deg = rotation_angle_deg(T_last_curr.linear());

    return (trans > params_.keyframe_trans_thresh) || (rot_deg > params_.keyframe_rot_thresh);
}

void AsyncMapping::accept_keyframe(const EstimationFrame& frame) {
    const Eigen::Isometry3d T_world_frame = frame.T_world_frame();

    if (params_.save_raw_mapping_frames) {
        save_keyframe_raw(frame);
        append_pose(T_world_frame);
    }

    integrate_into_map(frame);
    keyframes_since_downsample_++;
    downsample_map_if_needed();

    has_last_keyframe_ = true;
    last_keyframe_T_world_frame_ = T_world_frame;
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

void AsyncMapping::save_keyframe_raw(const EstimationFrame& frame) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(frame.frame->size());
    const auto& points = frame.frame->points;
    for (int i = 0; i < frame.frame->size(); i++) {
        Eigen::Vector4d p = points[i];
        cloud.emplace_back(p.x(), p.y(), p.z());
    }

    const fs::path filepath = fs::path(output_dir_) / std::format("frame_{}.pcd", keyframe_count_);
    if (pcl::io::savePCDFileBinary(filepath.string(), cloud) != 0) {
        logger::warn("mapping", "failed to save {}", filepath.string());
    }
}

void AsyncMapping::append_pose(const Eigen::Isometry3d& T_world_frame) {
    poses_ofs_ << convert_to_string(T_world_frame) << "\n";
}

void AsyncMapping::integrate_into_map(const EstimationFrame& frame) {
    if (frame.frame_type != FrameType::IMU) {
        logger::fatal("mapping", "only IMU frames are supported for raw saving; skip frame_id={}", frame.id);
        std::exit(EXIT_FAILURE);
    }

    const auto& points = frame.frame->points;
    map_cloud_->reserve(map_cloud_->size() + frame.frame->size());
    const Eigen::Isometry3d T_world_frame = frame.T_world_frame();
    for (int i = 0; i < frame.frame->size(); i++) {
        Eigen::Vector4d p = points[i];
        p = T_world_frame * p;
        map_cloud_->emplace_back(p.x(), p.y(), p.z());
    }
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