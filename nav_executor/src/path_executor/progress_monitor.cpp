#include <nav_executor/path_executor/progress_monitor.hpp>
#include <algorithm>
#include <cmath>
#include <rclcpp/logging.hpp>

namespace nav_executor {

ProgressMonitor::ProgressMonitor(rclcpp::Logger logger) : logger_(logger) {}

bool ProgressMonitor::update_and_check_no_progress(
    const double current_tau,
    const NoProgressGuardParams& params,
    const MotionState current_state,
    const MotionState prev_state,
    const std::chrono::steady_clock::time_point stamp
) {
    if (prev_state != MotionState::FOLLOW && prev_state != MotionState::STEPPING) return false;
    const double spacing = std::clamp(params.tau_landmark_spacing, 1e-4, 1.0);
    const double tau = std::clamp(current_tau, 0.0, 1.0);
    const int current_landmark_idx = static_cast<int>(std::floor(tau / spacing));

    if (current_state != last_no_progress_check_state_) {
        last_no_progress_check_state_ = current_state;
        max_tau_landmark_time_ = stamp;
    }

    if (max_tau_landmark_index_ < 0 || current_landmark_idx > max_tau_landmark_index_) {
        max_tau_landmark_index_ = current_landmark_idx;
        max_tau_landmark_time_ = stamp;
        return false;
    }

    const double elapsed = std::chrono::duration<double>(stamp - max_tau_landmark_time_).count();
    if (elapsed >= params.timeout) {
        const char* mode_name = (current_state == MotionState::STEPPING) ? "Stepping" : "Follow";
        RCLCPP_WARN(logger_,
            "%s replan: tau did not cross %.4f (current=%.4f) for %.1fs",
            mode_name, static_cast<double>(max_tau_landmark_index_) * spacing, tau, elapsed);
        return true;
    }
    return false;
}

void ProgressMonitor::reset() {
    max_tau_landmark_index_ = -1;
    max_tau_landmark_time_ = {};
    last_no_progress_check_state_ = MotionState::IDLE;
}

} // namespace nav_executor
