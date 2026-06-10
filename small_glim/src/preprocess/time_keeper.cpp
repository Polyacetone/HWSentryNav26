#include <small_glim/preprocess/time_keeper.hpp>
#include <small_glim/common/logger.hpp>
#include <algorithm>
#include <cmath>

namespace small_glim {

PerPointTimeSettings::PerPointTimeSettings(const Config::Ptr config) {
    autoconf = config->param<bool>("sensors.autoconf_perpoint_times");
    if (autoconf) {
        relative_time = true;
        prefer_frame_time = false;
        point_time_scale = 1.0;
    } else {
        relative_time = config->param<bool>("sensors.perpoint_relative_time");
        prefer_frame_time = config->param<bool>("sensors.autoconf_prefer_frame_time");
        point_time_scale = config->param<double>("sensors.perpoint_time_scale");
    }
    max_scan_duration = config->param<double>("sensors.max_scan_duration");
    max_frame_time_gap = config->param<double>("sensors.max_frame_time_gap");
    max_imu_time_gap = config->param<double>("sensors.max_imu_time_gap");
    max_invalid_time_ratio = config->param<double>("sensors.max_invalid_time_ratio");
}

TimeKeeper::TimeKeeper(const Config::Ptr config) {
    settings = std::make_unique<PerPointTimeSettings>(config);
    last_points_stamp = -1.0;
    last_imu_stamp = -1.0;
    estimated_scan_duration = -1.0;
    point_time_offset = 0.0;
}

bool TimeKeeper::validate_imu_stamp(const double imu_stamp) {
    const double imu_diff = imu_stamp - last_imu_stamp;
    if (!std::isfinite(imu_stamp)) {
        logger::warn("time_keeper", "non-finite IMU timestamp detected!! stamp={:.6f}", imu_stamp);
        return false;
    } else if (last_imu_stamp < 0.0) {
        // First IMU frame
    } else if (imu_stamp < last_imu_stamp) {
        logger::warn("time_keeper", "IMU timestamp rewind detected!!");
        logger::warn("time_keeper", "current={:.6f} last={:.6f} diff={:.6f}", imu_stamp, last_imu_stamp, imu_diff);
        return false;
    } else if (imu_stamp - last_imu_stamp > settings->max_imu_time_gap) {
        logger::warn("time_keeper", "large time gap between consecutive IMU data!!");
        logger::warn("time_keeper", "current={:.6f} last={:.6f} diff={:.6f}", imu_stamp, last_imu_stamp, imu_diff);
    }
    last_imu_stamp = imu_stamp;

    const double points_diff = imu_stamp - last_points_stamp;
    if (last_points_stamp > 0.0 && std::abs(points_diff) > 1.0) {
        logger::warn("time_keeper", "large time difference between points and imu!!");
        logger::warn("time_keeper", 
            "points={:.6f} imu={:.6f} diff={:.6f}",
            last_points_stamp,
            imu_stamp,
            points_diff
        );
    }

    return true;
}

bool TimeKeeper::process(const small_glim::RawPoints::Ptr points) {
    replace_points_stamp(points);

    if (points->points.size() != points->times.size()) {
        logger::warn(
            "time_keeper",
            "inconsistent # of points and # of timestamps found after time conversion!! |points|={} |times|={}",
            points->points.size(),
            points->times.size()
        );
        return false;
    }
    if (points->points.empty()) {
        logger::warn("time_keeper", "drop empty LiDAR frame");
        return false;
    }

    const size_t original_size = points->size();
    std::vector<Eigen::Vector4d> valid_points;
    std::vector<double> valid_times;
    std::vector<double> valid_intensities;
    std::vector<uint32_t> valid_rings;
    std::vector<Eigen::Vector4d> valid_colors;
    valid_points.reserve(original_size);
    valid_times.reserve(original_size);
    if (!points->intensities.empty()) valid_intensities.reserve(original_size);
    if (!points->rings.empty()) valid_rings.reserve(original_size);
    if (!points->colors.empty()) valid_colors.reserve(original_size);

    for (size_t i = 0; i < original_size; i++) {
        const auto& point = points->points[i];
        const double time = points->times[i];
        const bool valid = std::isfinite(time)
            && time >= -1e-6
            && time <= settings->max_scan_duration
            && point.allFinite();
        if (!valid) {
            continue;
        }
        valid_points.push_back(point);
        valid_times.push_back(std::max(0.0, time));
        if (!points->intensities.empty()) valid_intensities.push_back(points->intensities[i]);
        if (!points->rings.empty()) valid_rings.push_back(points->rings[i]);
        if (!points->colors.empty()) valid_colors.push_back(points->colors[i]);
    }

    const size_t num_dropped = original_size - valid_points.size();
    if (num_dropped != 0) {
        const double invalid_ratio = static_cast<double>(num_dropped) / static_cast<double>(original_size);
        logger::warn(
            "time_keeper",
            "drop {} invalid points by timestamp/finite check (ratio={:.4f}, max_scan_duration={:.3f})",
            num_dropped,
            invalid_ratio,
            settings->max_scan_duration
        );
        if (invalid_ratio > settings->max_invalid_time_ratio || valid_points.empty()) {
            logger::warn("time_keeper", "drop LiDAR frame due to too many invalid point timestamps");
            return false;
        }
        points->points = std::move(valid_points);
        points->times = std::move(valid_times);
        points->intensities = std::move(valid_intensities);
        points->rings = std::move(valid_rings);
        points->colors = std::move(valid_colors);
    }

    const auto minmax_times = std::minmax_element(points->times.begin(), points->times.end());
    if (*minmax_times.first < -1e-6 || *minmax_times.second > settings->max_scan_duration) {
        logger::warn(
            "time_keeper",
            "drop LiDAR frame with invalid per-point timestamp range min={:.6f} max={:.6f}",
            *minmax_times.first,
            *minmax_times.second
        );
        return false;
    }
    if (points->stamp < 0.0) {
        logger::warn("time_keeper", "frame timestamp is negative!! frame={:.6f}", points->stamp);
    }
    if (points->stamp > 3000000000) {
        logger::warn("time_keeper", 
            "frame timestamp is wrong (or GLIM has been used for over 40 years)!! frame={:.6f}",
            points->stamp
        );
    }

    const double time_diff = points->stamp - last_points_stamp;
    if (last_points_stamp < 0.0) {
        // First LiDAR frame
    } else if (time_diff < 0.0) {
        logger::warn("time_keeper", "point timestamp rewind detected!!");
        logger::warn("time_keeper", 
            "current={:.6f} last={:.6f} diff={:.6f}",
            points->stamp,
            last_points_stamp,
            time_diff
        );
        return false;
    } else if (time_diff > settings->max_frame_time_gap) {
        logger::warn("time_keeper", "large time gap between consecutive LiDAR frames!!");
        logger::warn("time_keeper", 
            "current={:.6f} last={:.6f} diff={:.6f}",
            points->stamp,
            last_points_stamp,
            time_diff
        );
    }

    last_points_stamp = points->stamp;

    return true;
}

void TimeKeeper::replace_points_stamp(const small_glim::RawPoints::Ptr points) {
    // No per-point timestamps
    // Assign timestamps based on the estimated scan duration
    if (points->times.empty()) {
        static bool first_warning = true;
        if (first_warning) {
            logger::warn("time_keeper", "per-point timestamps are not given!!");
            logger::warn("time_keeper", "use pseudo per-point timestamps based on the order of points");
            first_warning = false;
        }

        points->times.resize(points->size(), 0.0);
        const double scan_duration = estimate_scan_duration(points->stamp);
        if (scan_duration > 0.0) {
            for (size_t i = 0; i < points->size(); i++) {
                points->times[i] = scan_duration * static_cast<double>(i) / static_cast<double>(points->size());
            }
        }

        return;
    }

    // Check the number of timestamps
    if (points->times.size() != points->size()) {
        logger::warn("time_keeper", "# of timestamps and # of points mismatch!!");
        points->times.resize(points->size(), 0.0);
        return;
    }

    const auto minmax_times = std::minmax_element(points->times.begin(), points->times.end());
    const double min_time = *minmax_times.first;
    const double max_time = *minmax_times.second;

    if (settings->autoconf) {
        settings->autoconf = false;

        if (min_time < 0.0) {
            logger::warn("time_keeper", 
                "negative per-point timestamp is found!! min={:.6f} max={:.6f}",
                min_time,
                max_time
            );

            if (settings->prefer_frame_time) {
                logger::warn("time_keeper", "use frame timestamp as is!!");
            } else {
                logger::warn("time_keeper", 
                    "add an offset to the frame timestamp to make per-point ones positive!!"
                );
            }
        }

        if (max_time < 1.0) {
            settings->relative_time = true;
        } else {
            settings->relative_time = false;
            logger::info("time_keeper", 
                "large point timestamp (min={:.6f} max={:.6f} > 1.0) found",
                min_time,
                max_time
            );
            logger::info("time_keeper", "assume that point times are absolute and convert them to relative");

            if (min_time > 1e16) {
                logger::info("time_keeper", 
                    "too large point timestamp (min={:.6f} max={:.6f} > 1e16) found",
                    min_time,
                    max_time
                );
                logger::info("time_keeper", 
                    "maybe using a Livox LiDAR that use FLOAT64 nanosec per-point timestamps"
                );
                settings->point_time_scale = 1e-9;
            }

            if (settings->prefer_frame_time) {
                logger::info("time_keeper", 
                    "frame timestamp will be prioritized over the first point timestamp"
                );
            } else {
                logger::info("time_keeper", "frame timestamp will be overwritten by the first point timestamp");
            }
        }
    }

    // Per-point timestamps are relative to the first one
    if (settings->relative_time) {
        // Make per-point timestamps positive
        if (min_time < 0.0) {
            if (!settings->prefer_frame_time) {
                // Shift the frame timestamp to keep the consistency
                points->stamp += min_time * settings->point_time_scale;
            }

            for (auto& time: points->times) {
                time -= min_time;
            }
        }

        if (std::abs(settings->point_time_scale - 1.0) > 1e-6) {
            // Convert timestamps to seconds
            for (auto& time: points->times) {
                time *= settings->point_time_scale;
            }
        }

        return;
    }

    // Per-point timestamps are absolute

    if (!settings->prefer_frame_time) {
        // Overwrite the frame timestamp with the first point timestamp
        points->stamp = min_time * settings->point_time_scale;
    }

    // Make per-point timestamps relative to the frame timestamp
    for (auto& time: points->times) {
        time = (time - min_time) * settings->point_time_scale;
    }
}

double TimeKeeper::estimate_scan_duration(const double stamp) {
    if (estimated_scan_duration > 0.0) {
        return estimated_scan_duration;
    }

    if (last_points_stamp < 0) {
        return -1.0;
    }

    scan_duration_history.emplace_back(stamp - last_points_stamp);
    std::nth_element(
        scan_duration_history.begin(),
        scan_duration_history.begin() + scan_duration_history.size() / 2,
        scan_duration_history.end()
    );
    double scan_duration = scan_duration_history[scan_duration_history.size() / 2];

    if (scan_duration_history.size() == 1000) {
        logger::info("time_keeper", "estimated scan duration: {}", scan_duration);
        estimated_scan_duration = scan_duration;
        scan_duration_history.clear();
        scan_duration_history.shrink_to_fit();
    }

    if (scan_duration < 0.01 || scan_duration > 1.0) {
        logger::warn("time_keeper", "invalid scan duration estimate: {}", scan_duration);
        scan_duration = -1.0;
    }

    return scan_duration;
}

}
