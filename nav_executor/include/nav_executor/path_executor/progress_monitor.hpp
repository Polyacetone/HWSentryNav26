#pragma once

#include <chrono>
#include <rclcpp/logger.hpp>

#include <nav_executor/path_executor/state_machine.hpp>

namespace nav_executor {

struct NoProgressGuardParams {
    double arc_length_landmark_spacing;
    double timeout;
};

class ProgressMonitor {
public:
    explicit ProgressMonitor(rclcpp::Logger logger);

    bool update_and_check_no_progress(
        double current_arc_length,
        const NoProgressGuardParams& params,
        MotionState current_state,
        MotionState prev_state,
        std::chrono::steady_clock::time_point stamp
    );

    void reset();
private:
    int max_arc_length_landmark_index_ = -1;
    std::chrono::steady_clock::time_point max_arc_length_landmark_time_ = {};
    MotionState last_no_progress_check_state_ = MotionState::IDLE;
    rclcpp::Logger logger_;
};

} // namespace nav_executor
