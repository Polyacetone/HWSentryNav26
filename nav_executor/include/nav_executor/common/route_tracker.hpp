#pragma once

#include <chrono>
#include <optional>

#include <Eigen/Core>

#include <nav_executor/common/annotated_path.hpp>

namespace nav_executor {

struct RouteTrackerParams {
    double initial_search_distance;
    double max_progress_rate;
    double progress_tolerance;
    double max_cross_track_error;
};

enum class RouteTrackingStatus : uint8_t {
    TRACKED = 0,
    LOST = 1,
};

struct RouteEstimate {
    AnnotatedPath::ConstPtr path;
    RouteTrackingStatus status = RouteTrackingStatus::LOST;
    double u = 0.0;
    double arc_length = 0.0;
    double remaining_length = 0.0;
    Eigen::Vector2d projected_position = Eigen::Vector2d::Zero();
    double cross_track_error = 0.0;
};

class RouteTracker {
public:
    explicit RouteTracker(RouteTrackerParams params);

    std::optional<RouteEstimate> update(
        AnnotatedPath::ConstPtr path,
        const Eigen::Vector3d& chassis_pose_map,
        double chassis_velocity,
        std::chrono::steady_clock::time_point stamp
    );

    void reset();

private:
    RouteTrackerParams params_;
    AnnotatedPath::ConstPtr path_;
    std::optional<RouteEstimate> accepted_estimate_;
    std::chrono::steady_clock::time_point last_stamp_{};
};

} // namespace nav_executor
