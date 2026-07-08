#pragma once

#include <chrono>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/path/spline_path.hpp>
#include <nav_executor/executor/state_machine.hpp>

namespace nav_executor {

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
        MotionState current_state,
        MotionState prev_state,
        std::chrono::steady_clock::time_point stamp
    );

    void reset();
    void clear();

private:
    std::vector<double> follow_landmarks_u_;
    int follow_max_landmark_idx_ = -1;
    std::chrono::steady_clock::time_point follow_max_landmark_time_ = {};
    MotionState last_no_progress_check_state_ = MotionState::IDLE;
    rclcpp::Logger logger_;
};

} // namespace nav_executor
