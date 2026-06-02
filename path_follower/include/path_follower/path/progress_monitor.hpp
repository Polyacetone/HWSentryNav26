#pragma once

#include <chrono>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <path_follower/path/spline_path.hpp>
#include <path_follower/control/state_machine.hpp>

namespace path_follower {

struct NoProgressGuardParams {
    double landmark_spacing;
    double timeout;
};

class ProgressMonitor {
public:
    explicit ProgressMonitor(rclcpp::Logger logger);

    void recompute_landmarks(const SplinePath& path, double spacing);

    bool update_and_check_no_progress(
        double current_u,
        const NoProgressGuardParams& params,
        FsmState current_state,
        FsmState prev_state,
        std::chrono::steady_clock::time_point stamp
    );

    void reset();
    void clear();

private:
    std::vector<double> follow_landmarks_u_;
    int follow_max_landmark_idx_ = -1;
    std::chrono::steady_clock::time_point follow_max_landmark_time_ = {};
    FsmState last_no_progress_check_state_ = FsmState::IDLE;
    rclcpp::Logger logger_;
};

} // namespace path_follower
