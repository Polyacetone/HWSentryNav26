#include <path_follower/path/progress_monitor.hpp>
#include <rclcpp/logging.hpp>

namespace path_follower {

ProgressMonitor::ProgressMonitor(rclcpp::Logger logger) : logger_(logger) {}

void ProgressMonitor::recompute_landmarks(const SplinePath& path, const double spacing) {
    follow_landmarks_u_.clear();

    constexpr int ESTIMATE_SAMPLES = 100;
    double est_length = 0.0;
    Eigen::Vector2d est_prev = path.position(0.0);
    for (int i = 1; i <= ESTIMATE_SAMPLES; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(ESTIMATE_SAMPLES);
        const Eigen::Vector2d cur = path.position(u);
        est_length += (cur - est_prev).norm();
        est_prev = cur;
    }

    const double sample_density = std::max(
        10.0, 1.0 / std::max(spacing, 0.01)
    );
    const int N = std::max(10, static_cast<int>(std::ceil(est_length * sample_density)));

    double arc = 0.0;
    double next_threshold = 0.0;
    Eigen::Vector2d prev = path.position(0.0);

    for (int i = 0; i <= N; i++) {
        const double u = static_cast<double>(i) / static_cast<double>(N);
        const Eigen::Vector2d cur = path.position(u);
        if (i > 0) arc += (cur - prev).norm();
        prev = cur;
        if (arc >= next_threshold) {
            follow_landmarks_u_.push_back(u);
            next_threshold += spacing;
        }
    }

    if (follow_landmarks_u_.empty() || follow_landmarks_u_.back() < 1.0) {
        follow_landmarks_u_.push_back(1.0);
    }
}

bool ProgressMonitor::update_and_check_no_progress(
    const double current_u,
    const NoProgressGuardParams& params,
    const FsmState current_state,
    const FsmState prev_state,
    const std::chrono::steady_clock::time_point stamp
) {
    if (prev_state != FsmState::FOLLOW && prev_state != FsmState::STEPPING) return false;
    if (follow_landmarks_u_.empty()) return false;

    const int n = static_cast<int>(follow_landmarks_u_.size());

    if (current_state != last_no_progress_check_state_) {
        last_no_progress_check_state_ = current_state;
        follow_max_landmark_time_ = stamp;
    }

    int new_max = follow_max_landmark_idx_;
    for (int i = std::max(0, new_max + 1); i < n; i++) {
        if (current_u >= follow_landmarks_u_[static_cast<size_t>(i)]) {
            new_max = i;
        } else {
            break;
        }
    }

    if (follow_max_landmark_idx_ < 0 || new_max > follow_max_landmark_idx_) {
        follow_max_landmark_idx_ = new_max;
        follow_max_landmark_time_ = stamp;
        return false;
    }

    const double elapsed = std::chrono::duration<double>(stamp - follow_max_landmark_time_).count();
    if (elapsed >= params.timeout) {
        const double landmark_u = follow_max_landmark_idx_ >= 0 ? follow_landmarks_u_[static_cast<size_t>(follow_max_landmark_idx_)] : -1.0;
        const char* mode_name = (current_state == FsmState::STEPPING) ? "Stepping" : "Follow";
        RCLCPP_WARN(logger_,
            "%s replan: no progress at landmark %d/%d (landmark_u=%.3f, progress_u=%.3f) for %.1fs",
            mode_name, follow_max_landmark_idx_, n - 1, landmark_u, current_u, elapsed);
        return true;
    }
    return false;
}

void ProgressMonitor::reset() {
    follow_max_landmark_idx_ = -1;
    follow_max_landmark_time_ = {};
}

void ProgressMonitor::clear() {
    follow_landmarks_u_.clear();
    follow_max_landmark_idx_ = -1;
}

} // namespace path_follower
