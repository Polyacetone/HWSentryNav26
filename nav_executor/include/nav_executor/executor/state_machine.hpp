#pragma once

#include <chrono>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

namespace nav_executor {

// ═══════════════════════════ 参数 ═══════════════════════════

// 正常模式切换（Follow ↔ Spin ↔ Idle/Fixed）速度门槛
struct TransitionParams {
    double follow_to_spin_vel_max;   // →SPIN: |v| 须低于
    double spin_to_follow_omega_max; // →FOLLOW: |ω| 须低于
    double to_idle_vel_max;          // →IDLE/FIXED: |v| 须低于
    double to_idle_omega_max;        // →IDLE/FIXED: |ω| 须低于
    double stopping_timeout;         // STOPPING 状态超时 (s)，超过强制切出
};

// 危险恢复参数
struct RecoveryParams {
    double hazard_cost_threshold;
    double hazard_step_norm_threshold;
    double safe_cost_threshold;
    double safe_step_norm_threshold;
    double recovery_cost_threshold;
    double radius_min;
    double radius_max;
    int radius_samples;
    int angle_samples;
    double path_integral_resolution;
    double path_integral_cost_weight;
    double path_integral_step_weight;
    double step_ascent_penalty_weight;
    double step_ascent_penalty_norm_threshold;
    double step_ascent_penalty_dot_threshold;
    double safe_hold_time;
    double goal_timeout;
};

// 卡住检测 + 倒车参数
struct StuckParams {
    double cmd_vel_threshold;
    double timeout;
    double max_displacement;
    double reverse_speed;
    double reverse_displacement;
    double reverse_timeout;
};

struct FsmParams {
    TransitionParams transition;
    RecoveryParams recovery;
    StuckParams stuck;
};

// ═══════════════════════════ 运动状态 ═══════════════════════
//
// 暴露给任务层的 motion_state。可抢占集合与不可抢占集合见
// is_preemptible()。DEAD 由 PathExecutor 外部拦截，不参与 FSM 流转。

enum class MotionState : uint8_t {
    DEAD = 0,             // 底盘失效：由 PathExecutor 外部拦截
    IDLE = 1,             // 无路径且不旋转
    FOLLOW = 2,           // 跟随全局路径
    SPIN = 3,             // 小陀螺
    STOPPING = 4,         // 模式间平滑减速
    HAZARD_RECOVERY = 5,  // 危险恢复
    STUCK_REVERSE = 6,    // 倒车脱困
    FIXED = 7,            // 固定保持目标点
    STEPPING = 8,         // 上下台阶（不可抢占）
};

// 可抢占集合
[[nodiscard]] inline bool is_preemptible(const MotionState s) {
    switch (s) {
        case MotionState::FOLLOW:
        case MotionState::IDLE:
        case MotionState::STOPPING:
        case MotionState::FIXED:
            return true;
        default:
            return false;
    }
}

// ═══════════════════════════ 输入 / 输出 ═══════════════════

struct FsmInput {
    // 任务输入（由顶层每周期暴露的三输入推导得到）
    bool has_path = false;       // 当前存在 active_path
    bool has_hold_goal = false;  // 当前存在 hold_goal（应进入 FIXED 保持）
    bool reach_goal = false;     // 路径终点已到达（dist/u 阈值）
    bool step_active = false;

    // 外部请求
    bool spin_requested = false;
    bool spin_high_priority = false;

    // stuck-like 检测信号（follow/stepping 无进度 + 卡住）
    bool no_progress_detected = false;
    bool is_stuck = false;

    // 危险 / 恢复安全
    bool is_hazard = false;
    bool is_recovery_safe = false;

    // 底盘不可控（FLIGHT/JUMP/STEP 物理过程），不响应速度指令
    bool command_blocked = false;

    Eigen::Vector2d chassis_pos_map = Eigen::Vector2d::Zero();
    double velocity = 0.0;
    double omega = 0.0;
    std::chrono::steady_clock::time_point stamp;
};

struct FsmOutput {
    MotionState state = MotionState::IDLE;

    // 路径终点到达事实（one-shot，仅路径阶段上报）
    bool goal_reached = false;

    // 恢复链结束后请求顶层丢 path 并 replan（one-shot）
    bool executor_replan_event = false;
};

// ═══════════════════════ 运动层状态机 ═══════════════════════

class StateMachine {
public:
    explicit StateMachine(const FsmParams& params, rclcpp::Logger logger);

    FsmOutput update(const FsmInput& input);
    [[nodiscard]] MotionState state() const { return active_state_; }

private:
    FsmOutput on_idle(const FsmInput& in);
    FsmOutput on_fixed(const FsmInput& in);
    FsmOutput on_follow(const FsmInput& in);
    FsmOutput on_stepping(const FsmInput& in);
    FsmOutput on_spin(const FsmInput& in);
    FsmOutput on_stopping(const FsmInput& in);
    FsmOutput on_stuck_reverse(const FsmInput& in);
    FsmOutput on_hazard_recovery(const FsmInput& in);

    FsmOutput transition_to(MotionState next);
    FsmOutput route_to_terminal(const FsmInput& in);
    FsmOutput exit_reverse(const FsmInput& in, double displacement, double mature_elapsed);
    FsmOutput finish_recovery_chain(const FsmInput& in);

    // 目标终态类型（仅用于选择 STOPPING 退出速度门槛）
    enum class Terminal : uint8_t { IDLE, FIXED, SPIN, FOLLOW };
    [[nodiscard]] Terminal terminal_target(const FsmInput& in) const;
    [[nodiscard]] bool stopping_ready(const FsmInput& in, Terminal target) const;

    FsmParams params_;
    rclcpp::Logger logger_;

    MotionState active_state_ = MotionState::IDLE;

    std::chrono::steady_clock::time_point stopping_start_time_;

    // 恢复链结束后是否应发出 one-shot executor_replan_event
    bool replan_after_recovery_ = false;

    std::chrono::steady_clock::time_point pending_reverse_start_time_;
    Eigen::Vector2d pending_reverse_start_pos_ = Eigen::Vector2d::Zero();

    bool reverse_entry_initialized_ = false;
    double reverse_mature_accumulated_ = 0.0;
    Eigen::Vector2d reverse_entry_pos_ = Eigen::Vector2d::Zero();
    std::chrono::steady_clock::time_point reverse_last_mature_stamp_;
};

} // namespace nav_executor
