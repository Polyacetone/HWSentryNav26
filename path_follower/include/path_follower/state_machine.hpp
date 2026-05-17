#pragma once

#include <chrono>
#include <memory>

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
    double path_integral_cost_weight;
    double path_integral_step_weight;
    double step_ascent_penalty_weight;
    double step_ascent_penalty_norm_threshold;
    double step_ascent_penalty_dot_threshold;

    // 退出条件
    double safe_hold_time;
    double goal_timeout;
};

// 卡住检测 + 倒车参数
struct StuckParams {
    double cmd_vel_threshold;        // 指令速度超过此值才开始检测卡住
    double timeout;                  // 卡住判定持续时间阈值 (s)
    double max_displacement;         // 位移小于此值判定卡住 (m)
    double reverse_speed;            // 倒车速度 (m/s)
    double reverse_displacement;     // 倒车退出位移阈值 (m) —— 里程计从 entry 起计
    double reverse_timeout;          // 倒车安全网超时 (s) —— MATURE 累计时间，超时则 RCLCPP_ERROR
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
    WAIT_REPLAN = 8,      // 等待 path_planner 重规划
    STEPPING = 9,        // 正在上下台阶，不允许外部任务打断
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

    // 外部请求
    bool spin_requested = false;
    bool spin_high_priority = false;

    // 信号：step_latch_ttl 超时（用于直接从 STEPPING → STUCK_REVERSE）
    bool step_ttl_expired = false;

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

    // HAZARD_RECOVERY 完成标记
    bool recovery_finished = false;

    // 请求 path_planner 立即重规划
    bool request_replan = false;
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
