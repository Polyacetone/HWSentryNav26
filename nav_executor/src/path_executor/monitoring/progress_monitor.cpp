#include <nav_executor/path_executor/monitoring/progress_monitor.hpp>
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
    const double spacing = std::max(params.arc_length_landmark_spacing, 1e-4);
    const double arc_length = std::max(current_arc_length, 0.0);
    const int current_landmark_idx = static_cast<int>(std::floor(arc_length / spacing));

    if (current_state != last_no_progress_check_state_) {
        last_no_progress_check_state_ = current_state;
        max_arc_length_landmark_time_ = stamp;
    }

    if (max_arc_length_landmark_index_ < 0
        || current_landmark_idx > max_arc_length_landmark_index_) {
        max_arc_length_landmark_index_ = current_landmark_idx;
        max_arc_length_landmark_time_ = stamp;
        return false;
    }

    const double elapsed = std::chrono::duration<double>(
        stamp - max_arc_length_landmark_time_
    ).count();
    if (elapsed >= params.timeout) {
        const char* mode_name = (current_state == MotionState::STEPPING) ? "Stepping" : "Follow";
        RCLCPP_WARN(logger_,
            "%s replan: path progress did not cross %.3f m (current=%.3f m) for %.1fs",
            mode_name,
            static_cast<double>(max_arc_length_landmark_index_) * spacing,
            arc_length,
            elapsed);
        return true;
    }
    return false;
}

void ProgressMonitor::reset() {
    max_arc_length_landmark_index_ = -1;
    max_arc_length_landmark_time_ = {};
    last_no_progress_check_state_ = MotionState::IDLE;
}

} // namespace nav_executor
