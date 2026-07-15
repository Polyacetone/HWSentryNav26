#pragma once

#include <chrono>
#include <optional>

#include <Eigen/Core>

#include <nav_executor/common/annotated_path.hpp>

namespace nav_executor {

struct RouteTrackerParams {
    double initial_search_distance;
    double max_tracking_error;
    GovernedClockParams governed_clock;
};

enum class RouteTrackingStatus : uint8_t {
    TRACKED = 0,
    LOST = 1,
};

struct RouteEstimate {
    AnnotatedPath::ConstPtr path;
    RouteTrackingStatus status = RouteTrackingStatus::LOST;
    double tau = 0.0;
    double arc_length = 0.0;
    double remaining_length = 0.0;
    Eigen::Vector2d reference_position = Eigen::Vector2d::Zero();
    double tracking_error = 0.0;
};

class RouteTracker {
public:
    explicit RouteTracker(RouteTrackerParams params);

    std::optional<RouteEstimate> update(
        AnnotatedPath::ConstPtr path,
        const Eigen::Vector3d& chassis_pose_map,
        std::chrono::steady_clock::time_point stamp,
        bool advance_clock
    );

    void reset();

private:
    RouteTrackerParams params_;
    AnnotatedPath::ConstPtr path_;
    std::optional<RouteEstimate> estimate_;
    std::chrono::steady_clock::time_point last_stamp_{};
};

} // namespace nav_executor
