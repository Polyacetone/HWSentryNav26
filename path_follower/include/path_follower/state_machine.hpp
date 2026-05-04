#pragma once

#include <chrono>
#include <memory>

#include <rclcpp/logger.hpp>

namespace path_follower {

// ═══════════════════════════ 参数 ═══════════════════════════

// 正常模式切换（Follow ↔ Spin ↔ Idle）速度门槛
struct TransitionParams {
    double follow_to_spin_vel_max;   // Follow→Stopping→Spin: |v| 须低于
    double spin_to_follow_omega_max; // Spin→Stopping→Follow: |ω| 须低于
    double to_idle_vel_max;          // →Idle: |v| 须低于
    double to_idle_omega_max;        // →Idle: |ω| 须低于
    double stopping_timeout;         // Stopping 状态超时 (s)，超过该时间仍未切出停止则强制切出
};

// 危险恢复参数
struct RecoveryParams {
    bool enable;

    // 危险判定（当前位置）
    double hazard_cost_threshold;
    double hazard_step_norm_threshold;

    // 可通行点判定（目标点）
    double safe_cost_threshold;
    double safe_step_norm_threshold;

    // 搜索可通行点
    double recovery_cost_threshold;
    double radius_min;
    double radius_max;
    int radius_samples;
    int angle_samples;
    double path_integral_resolution;

    // 退出条件
    double safe_hold_time;
    double goal_timeout;
};

// 卡住检测 + 倒车参数
struct StuckParams {
    bool enable;
    double cmd_vel_threshold;
    double timeout;
    double max_displacement;
    double reverse_speed;
    double reverse_duration;
};

// 所有 FSM 参数的聚合
struct FsmParams {
    TransitionParams transition;
    RecoveryParams recovery;
    StuckParams stuck;
};

// ═══════════════════════════ 状态枚举 ═══════════════════════

enum class FsmState : uint8_t {
    DEAD = 0,             // 底盘失效（Dead/Recovery/Abnormal）：由 MainController 外部拦截
    IDLE = 1,             // 无路径且不旋转
    FOLLOW = 2,           // 跟随全局路径
    SPIN = 3,             // 小陀螺模式
    STOPPING = 4,         // 正常模式间的平滑过渡减速
    HAZARD_RECOVERY = 5,  // 危险恢复（向安全点移动）
    STUCK_REVERSE = 6,    // 倒车脱困
    FIXED = 7,            // 固定在目标点位（持续 MPC 保持位置）
    STEP_RUNUP = 8        // 上台阶助跑（先去助跑点，再恢复 FOLLOW）
};

// ═══════════════════════════ 输入 / 输出 ═══════════════════

// 每个控制周期由 MainController 填写、传入 FSM
struct FsmInput {
    // 任务输入
    bool has_path = false;      // 当前是否仍有未完成路径
    bool has_new_path = false;  // 本周期是否收到新路径
    bool fixed_goal_flag = false;
    bool reach_goal = false;
    bool step_runup_requested = false;
    bool step_runup_done = false;

    // 外部请求
    bool spin_requested = false;
    bool spin_high_priority = false;

    // 安全布尔（全部在 FSM 外计算）
    bool is_hazard = false;
    bool is_stuck = false;
    bool is_recovery_safe = false;

    // 速度判定量（用于 STOPPING 退出判断，当前由 MainController 传入指令速度）
    double velocity = 0.0;   // 线速度
    double omega = 0.0;      // 角速度

    // 时间戳
    std::chrono::steady_clock::time_point stamp;
};

// FSM 每次 update() 的输出
struct FsmOutput {
    FsmState state = FsmState::IDLE;

    // 状态机要求上层消费当前路径（例如到达终点或脱困链重置任务）
    bool consume_global_path = false;

    // HAZARD_RECOVERY 完成标记
    bool recovery_finished = false;
};

// ═══════════════════════ 统一控制状态机 ═════════════════════

class StateMachine {
public:
    explicit StateMachine(const FsmParams& params, rclcpp::Logger logger);
    ~StateMachine();

    StateMachine(StateMachine&&) noexcept;
    StateMachine& operator=(StateMachine&&) noexcept;

    StateMachine(const StateMachine&) = delete;
    StateMachine& operator=(const StateMachine&) = delete;

    // 每个控制周期调用一次，返回状态决策结果
    FsmOutput update(const FsmInput& input);

    // 查询当前状态
    [[nodiscard]] FsmState state() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
