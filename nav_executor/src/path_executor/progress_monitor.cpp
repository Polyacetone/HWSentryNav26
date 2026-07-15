#include <nav_executor/path_executor/progress_monitor.hpp>
#include <algorithm>
#include <cmath>
#include <rclcpp/logging.hpp>

namespace nav_executor {

ProgressMonitor::ProgressMonitor(rclcpp::Logger logger) : logger_(logger) {}

bool ProgressMonitor::update_and_check_no_progress(
    const double current_arc_length,
    const NoProgressGuardParams& params,
    const MotionState current_state,
    const MotionState prev_state,
    const std::chrono::steady_clock::time_point stamp
) {
    if (prev_state != MotionState::FOLLOW && prev_state != MotionState::STEPPING) return false;
    const double spacing = std::max(params.landmark_spacing, 0.01);
    const int current_landmark_idx = static_cast<int>(std::floor(std::max(0.0, current_arc_length) / spacing));

    if (current_state != last_no_progress_check_state_) {
        last_no_progress_check_state_ = current_state;
        follow_max_landmark_time_ = stamp;
    }

    if (follow_max_landmark_idx_ < 0 || current_landmark_idx > follow_max_landmark_idx_) {
        follow_max_landmark_idx_ = current_landmark_idx;
        follow_max_landmark_time_ = stamp;
        return false;
    }

    const double elapsed = std::chrono::duration<double>(stamp - follow_max_landmark_time_).count();
    if (elapsed >= params.timeout) {
        const char* mode_name = (current_state == MotionState::STEPPING) ? "Stepping" : "Follow";
        RCLCPP_WARN(logger_,
            "%s replan: no progress beyond %.2f m (current=%.2f m) for %.1fs",
            mode_name, static_cast<double>(follow_max_landmark_idx_) * spacing, current_arc_length, elapsed);
        return true;
    }
    return false;
}

void ProgressMonitor::reset() {
    follow_max_landmark_idx_ = -1;
    follow_max_landmark_time_ = {};
    last_no_progress_check_state_ = MotionState::IDLE;
}

} // namespace nav_executor
