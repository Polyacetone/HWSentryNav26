#pragma once

#include <chrono>

#include <Eigen/Core>
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
    double wait_replan_timeout;      // WAIT_REPLAN 等待新路径超时 (s)
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

// 所有 FSM 参数的聚合
struct FsmParams {
    TransitionParams transition;
    RecoveryParams recovery;
    StuckParams stuck;
};

// 目标状态枚举（Stopping 阶段需要记录目的地）
enum class DestState : uint8_t {
    IDLE = 0,
    FIXED = 1,
    SPIN = 2,
    FOLLOW = 3,
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
    WAIT_REPLAN = 8,      // 等待 path_planner 重规划
    STEPPING = 9,         // 正在上下台阶，不允许外部任务打断
};

// ═══════════════════════════ 输入 / 输出 ═══════════════════

// 每个控制周期由 MainController 填写、传入 FSM
struct FsmInput {
    // 任务输入
    bool has_path = false;      // 当前是否仍有未完成路径
    bool has_new_path = false;  // 本周期是否收到新路径
    bool fixed_goal_flag = false;
    bool reach_goal = false;
    bool step_active = false;
    bool replan_requested = false;
    bool replan_failed = false;     // 重规划请求收到了空路径（planner 失败）

    // 外部请求
    bool spin_requested = false;
    bool spin_high_priority = false;

    // 信号：无进度检测触发（Follow/Stepping 模式，由路标点方式判定）
    bool no_progress_detected = false;

    // 底盘当前位置（map 坐标系，用于 STUCK_REVERSE 位移判定）
    Eigen::Vector2d chassis_pos_map = Eigen::Vector2d::Zero();

    // 安全布尔（全部在 FSM 外计算）
    bool is_hazard = false;
    bool is_stuck = false;
    bool is_recovery_safe = false;

    // 底盘控制状态（由 MainController 根据腿模式 + 比赛阶段判断）
    bool command_blocked = false; // 底盘处于 FLIGHT/JUMP/STEP 等状态，不响应速度指令

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

    // 请求 path_planner 立即重规划
    bool request_replan = false;
};

// ═══════════════════════ 统一控制状态机 ═════════════════════

class StateMachine {
public:
    explicit StateMachine(const FsmParams& params, rclcpp::Logger logger);

    FsmOutput update(const FsmInput& input);
    FsmState state() const { return active_state_; }

private:
    FsmOutput on_idle(const FsmInput& in);
    FsmOutput on_fixed(const FsmInput& in);
    FsmOutput on_follow(const FsmInput& in);
    FsmOutput on_stepping(const FsmInput& in);
    FsmOutput on_spin(const FsmInput& in);
    FsmOutput on_stopping(const FsmInput& in);
    FsmOutput on_stuck_reverse(const FsmInput& in);
    FsmOutput on_hazard_recovery(const FsmInput& in);
    FsmOutput on_wait_replan(const FsmInput& in);

    bool stopping_ready(const FsmInput& in) const;
    FsmOutput route_to_terminal(const FsmInput& in);
    FsmOutput exit_reverse(const FsmInput& in, double displacement, double mature_elapsed);

    FsmParams params_;
    rclcpp::Logger logger_;

    FsmState active_state_ = FsmState::IDLE;

    DestState stopping_dest_ = DestState::IDLE;
    std::chrono::steady_clock::time_point stopping_start_time_;

    std::chrono::steady_clock::time_point pending_wait_replan_start_time_;

    bool replan_after_recovery_ = false;

    std::chrono::steady_clock::time_point pending_reverse_start_time_;
    Eigen::Vector2d pending_reverse_start_pos_ = Eigen::Vector2d::Zero();

    double reverse_mature_accumulated_ = 0.0;
    Eigen::Vector2d reverse_entry_pos_ = Eigen::Vector2d::Zero();
    std::chrono::steady_clock::time_point reverse_last_mature_stamp_;
};

}
