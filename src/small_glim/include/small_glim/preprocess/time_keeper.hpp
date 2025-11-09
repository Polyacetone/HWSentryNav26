#pragma once

#include <small_glim/common/raw_points.hpp>
#include <small_glim/common/config.hpp>

namespace small_glim {

/// @brief Parameters for per-point timestamp management
struct PerPointTimeSettings {
public:
    explicit PerPointTimeSettings(const Config::Ptr config);

    bool autoconf; ///< If true, load parameters from config file
    bool relative_time; ///< If true, per-point timestamps are relative to the first point. Otherwise, absolute.
    /// @brief If true, frame timestamp will never be overwritten by antyhing.
    ///        If false,
    ///          when per-point timestamps are absolute, overwrite the frame timestamp with the first point timestamp.
    ///          when per-point timestamps are relative and negative, add an offset to the frame timestamp to make per-point ones positive.
    bool prefer_frame_time;
    double point_time_scale; ///< Scale factor to convert per-point timestamps to seconds.
};

/**
 * @brief Utility class to unify timestamp convension
 */
class TimeKeeper {
public:
    explicit TimeKeeper(const Config::Ptr config);

    /**
    * @brief  Replace frame and point timestamps
    * @note   Frame timestamp must be the one at the moment when the first point is acquired
    * @note   Point timestamps must be relative with respect to the first point
    * @return If the timestamp of the point cloud is valid
    */
    bool process(const small_glim::RawPoints::Ptr& points);

    /**
    * @brief Check if IMU and LiDAR data are (very roughly) synchronized
    * @return If the IMU data is invalid and should be skipped, return false
    */
    bool validate_imu_stamp(const double imu_stamp);

private:
    void replace_points_stamp(const small_glim::RawPoints::Ptr& points);
    double estimate_scan_duration(const double stamp);

private:
    std::unique_ptr<PerPointTimeSettings> settings;

    double last_points_stamp; ///< Timestamp of the last LiDAR frame
    double last_imu_stamp; ///< Timestamp of the last IMU data

    // Scan duration estimation
    double estimated_scan_duration; ///< Estimated scan duration
    std::vector<double> scan_duration_history; ///< History of scan durations

    double point_time_offset; ///< Offset to correct time shift of point times
};

}