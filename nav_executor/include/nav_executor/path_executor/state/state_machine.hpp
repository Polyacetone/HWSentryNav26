#pragma once

#include <cstddef>
#include <chrono>
#include <optional>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

namespace nav_executor {

// ═══════════════════════════ 参数 ═══════════════════════════

// 下位机 SPIN 控制器接管前的安全互锁门槛。
struct PrepareSpinParams {
    double command_velocity_max;
    double command_omega_max;
    double measured_velocity_max;
    double measured_omega_max;
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
    PrepareSpinParams prepare_spin;
    RecoveryParams recovery;
    StuckParams stuck;
};

// ═══════════════════════════ 运动状态 ═══════════════════════
// 暴露给任务层的 motion_state。可抢占性判定在 PathExecutor::preemptible() 中统一处理。

// STEPPING 父状态内部的有序子状态。同一台阶段只允许沿
// PREPARING → ARMED → COMMITTED 前进；相邻台阶段会开启新的子状态生命周期。
enum class StepPhase : uint8_t {
    NONE = 0,       // 当前不处于 STEPPING 父状态。

    // 已进入台阶生命周期，但尚未向下位机发送台阶模式。
    // 本阶段开始渐变 capability，仍允许目标抢占、路径替换、RouteMonitor、
    // MPC lethal 检查，并沿用普通 FOLLOW 的 no-progress 策略。
    PREPARING = 1,

    // 已向下位机发送可撤销的台阶提示，但尚未越过 commit 弧长。
    // 本阶段仍可打断；一旦当前路径失效或目标改变，应立即取消台阶模式并减速，
    // 新路径就绪后可以直接从当前运动状态接管。
    ARMED = 2,

    // 已越过 commit 弧长，当前台阶路径不可再被普通任务或重规划结果替换。
    // 保持台阶模式直到 release 弧长，并使用 committed 阶段专属的监控策略。
    COMMITTED = 3,
};

[[nodiscard]] const char* step_phase_str(StepPhase phase);

[[nodiscard]] constexpr bool is_step_phase_active(const StepPhase phase) {
    return phase != StepPhase::NONE;
}

[[nodiscard]] constexpr bool is_step_phase_precommit(const StepPhase phase) {
    return phase == StepPhase::PREPARING || phase == StepPhase::ARMED;
}

[[nodiscard]] constexpr bool step_phase_activates_chassis_mode(const StepPhase phase) {
    return phase == StepPhase::ARMED || phase == StepPhase::COMMITTED;
}

enum class MotionState : uint8_t {
    DEAD = 0,             // 底盘失效：由 PathExecutor 外部拦截
    IDLE = 1,             // 无路径且不旋转
    FOLLOW = 2,           // 跟随全局路径
    SPIN = 3,             // 小陀螺
    PREPARE_SPIN = 4,     // NORMAL 模式下制动，准备向下位机 SPIN 控制器交权
    HAZARD_RECOVERY = 5,  // 危险恢复
    STUCK_REVERSE = 6,    // 倒车脱困
    FIXED = 7,            // 固定保持目标点
    STEPPING = 8,         // 上下台阶父状态；是否可抢占由 StepPhase 决定
};

// ═══════════════════════════ 输入 / 输出 ═══════════════════

struct FsmInput {
    bool has_active_path = false; // 任务层仍持有 active_path，不代表本周期投影成功
    bool route_tracked = false;   // active_path 存在且 RouteTracker 本周期投影有效
    bool has_hold_goal = false;  // 当前存在 hold_goal（应进入 FIXED 保持）
    bool reach_goal = false;     // 终点距离与路径剩余弧长均满足阈值
    StepPhase step_phase = StepPhase::NONE;
    uint64_t step_path_epoch = 0;
    std::optional<size_t> step_segment_index;
    bool resumed_from_stopped = false;

    // 外部请求
    bool spin_requested = false;
    bool spin_high_priority = false;

    // stuck-like 检测信号（follow/stepping 无进度 + 卡住）
    bool no_progress_detected = false;
    bool is_stuck = false;

    // 危险 / 恢复安全
    bool is_hazard_now = false;
    bool is_recovery_safe = false;

    // 底盘不可控（FLIGHT/JUMP/STEP 物理过程），不响应速度指令
    bool command_blocked = false;
    bool command_state_tracked = false;

    Eigen::Vector2d chassis_pos_map = Eigen::Vector2d::Zero();
    double command_velocity = 0.0;
    double command_omega = 0.0;
    double measured_velocity = 0.0;
    double measured_omega = 0.0;
    std::chrono::steady_clock::time_point stamp;
};

struct FsmOutput {
    MotionState state = MotionState::IDLE;
    bool goal_reached = false;           // 路径终点到达事实（one-shot，仅路径阶段上报）
    bool executor_replan_event = false;   // 恢复链结束后请求顶层丢 path 并 replan（one-shot）
    bool step_cancelled = false;          // 可撤销台阶段被打断；本周期必须下发 NORMAL 模式
};

// ═══════════════════════ 运动层状态机 ═══════════════════════

class StateMachine {
public:
    explicit StateMachine(const FsmParams& params, rclcpp::Logger logger);

    FsmOutput update(const FsmInput& input);
    [[nodiscard]] MotionState state() const { return active_state_; }
    [[nodiscard]] StepPhase step_phase() const { return step_phase_; }
    [[nodiscard]] uint64_t step_path_epoch() const { return step_path_epoch_; }
    [[nodiscard]] std::optional<size_t> step_segment_index() const {
        return step_segment_index_;
    }

private:
    FsmOutput on_idle(const FsmInput& in);
    FsmOutput on_fixed(const FsmInput& in);
    FsmOutput on_follow(const FsmInput& in);
    FsmOutput on_stepping(const FsmInput& in);
    FsmOutput on_spin(const FsmInput& in);
    FsmOutput on_prepare_spin(const FsmInput& in);
    FsmOutput on_stuck_reverse(const FsmInput& in);
    FsmOutput on_hazard_recovery(const FsmInput& in);

    FsmOutput transition_to(MotionState next);
    FsmOutput enter_stepping(
        StepPhase phase,
        uint64_t path_epoch,
        std::optional<size_t> segment_index
    );
    void synchronize_step_phase(
        StepPhase observed_phase,
        uint64_t path_epoch,
        std::optional<size_t> segment_index
    );
    FsmOutput route_to_requested_state(const FsmInput& in);
    FsmOutput route_to_normal_state(const FsmInput& in);
    FsmOutput exit_reverse(const FsmInput& in, double displacement, double mature_elapsed);
    FsmOutput finish_recovery_chain(const FsmInput& in);
    [[nodiscard]] bool should_start_resume_hazard_recovery(const FsmInput& in) const;

    [[nodiscard]] bool spin_authorized(const FsmInput& in) const;
    [[nodiscard]] bool prepare_spin_ready(const FsmInput& in) const;

    FsmParams params_;
    rclcpp::Logger logger_;

    MotionState active_state_ = MotionState::IDLE;
    StepPhase step_phase_ = StepPhase::NONE;
    uint64_t step_path_epoch_ = 0;
    std::optional<size_t> step_segment_index_;

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
