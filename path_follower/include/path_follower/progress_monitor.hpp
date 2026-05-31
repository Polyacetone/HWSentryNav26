#pragma once

#include <chrono>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <path_follower/spline_path.hpp>
#include <path_follower/state_machine.hpp>

namespace path_follower {

// ═══════════════════════ 无进度检测参数 ═══════════════════

struct NoProgressGuardParams {
    double landmark_spacing;
    double timeout;
};

// ═══════════════════════ 进度监视器 ═══════════════════════

class ProgressMonitor {
public:
    explicit ProgressMonitor(rclcpp::Logger logger);

    /// 基于路径弧长重新计算路标点
    void recompute_landmarks(const SplinePath& path, double spacing);

    /// 检查是否无进度（路标点方式），返回 true 表示检测到无进度
    bool check_no_progress(
        double current_u,
        const NoProgressGuardParams& params,
        FsmState current_state,
        FsmState prev_state,
        std::chrono::steady_clock::time_point stamp
    );

    /// 重置路标点进度（离开 follow-like 状态时）
    void reset();

    /// 清空路标数据（路径被清除时）
    void clear();

private:
    std::vector<double> follow_landmarks_u_;
    int follow_max_landmark_idx_ = -1;
    std::chrono::steady_clock::time_point follow_max_landmark_time_ = {};
    FsmState last_no_progress_check_state_ = FsmState::IDLE;
    rclcpp::Logger logger_;
};

} // namespace path_follower
