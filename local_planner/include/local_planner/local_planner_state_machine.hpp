#pragma once

#include <chrono>
#include <cstdint>
#include <rclcpp/logger.hpp>

namespace local_planner {

// ═══════════════════════ 状态枚举 ═══════════════════════════

enum class PlannerState : uint8_t {
    IDLE = 0,
    TRACK = 1,
    SPIN = 2,
    STOP_TRANSITION = 3,
    HOLD_FIXED = 4,
    REVERSE = 5,
};

// ═══════════════════════ 参数 ═══════════════════════════════

struct TransitionParams {
    double follow_to_spin_vel_max;
    double spin_to_follow_omega_max;
    double to_idle_vel_max;
    double to_idle_omega_max;
    double stopping_timeout;
};

struct StuckParams {
    bool enable;
    double cmd_vel_threshold;
    double timeout;
    double max_displacement;
    double reverse_speed;
    double reverse_duration;
};

struct LocalPlannerFsmParams {
    TransitionParams transition;
    StuckParams stuck;
};

// ═══════════════════════ FSM 输入 ═══════════════════════════

struct LocalPlannerFsmInput {
    bool has_path = false;
    bool has_new_path = false;
    bool fixed_goal_flag = false;
    bool reach_goal = false;

    bool spin_requested = false;
    bool spin_high_priority = false;

    bool is_stuck = false;

    // 速度判定量（用于 STOP_TRANSITION 退出判断）
    double velocity = 0.0;
    double omega = 0.0;

    std::chrono::steady_clock::time_point stamp;
};

// ═══════════════════════ FSM 输出 ═══════════════════════════

struct LocalPlannerFsmOutput {
    PlannerState state = PlannerState::IDLE;
    bool consume_global_path = false;
};

// ═══════════════════════ 状态机 ═══════════════════════════

/// V1 的 local_planner 状态机：用显式 switch 实现。
/// 不含 HAZARD_RECOVERY（V2 再补）。
class LocalPlannerStateMachine {
public:
    explicit LocalPlannerStateMachine(const LocalPlannerFsmParams& params, rclcpp::Logger logger);

    LocalPlannerFsmOutput update(const LocalPlannerFsmInput& input);
    [[nodiscard]] PlannerState state() const { return state_; }

private:
    enum class StopDest { IDLE, FIXED, SPIN, TRACK };

    void enter_state(PlannerState new_state);
    bool stopping_ready(const LocalPlannerFsmInput& in) const;

    LocalPlannerFsmParams params_;
    rclcpp::Logger logger_;

    PlannerState state_ = PlannerState::IDLE;
    StopDest stop_dest_ = StopDest::IDLE;
    std::chrono::steady_clock::time_point stopping_start_time_;
    std::chrono::steady_clock::time_point reverse_start_time_;
};

} // namespace local_planner
