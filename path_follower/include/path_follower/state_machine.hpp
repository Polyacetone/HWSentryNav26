#pragma once

#include <memory>
#include <optional>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>
#include <rclcpp/time.hpp>

#include <path_follower/nav_map.hpp>

namespace path_follower {

// ═══════════════════════════ 参数 ═══════════════════════════

// 正常模式切换（Follow ↔ Spin ↔ Idle）速度门槛
struct TransitionParams {
    double follow_to_spin_vel_max;   // Follow→Stopping→Spin: |v| 须低于
    double spin_to_follow_omega_max; // Spin→Stopping→Follow: |ω| 须低于
    double to_idle_vel_max;          // →Idle: |v| 须低于
    double to_idle_omega_max;        // →Idle: |ω| 须低于
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
    double circ_radius;
    int circ_angle_samples;
    int circ_radius_samples;

    // 退出条件
    double goal_reached_dist;
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

enum class FsmState {
    IDLE,             // 无路径且不旋转
    FOLLOW,           // 跟随全局路径
    SPIN,             // 小陀螺模式
    STOPPING,         // 正常模式间的平滑过渡减速
    HAZARD_RECOVERY,  // 危险恢复（向安全点移动）
    STUCK_REVERSE,    // 倒车脱困
};

// ═══════════════════════════ 输入 / 输出 ═══════════════════

// 每个控制周期由 MainController 填写、传入 FSM
struct FsmInput {
    // 外部请求
    bool has_path = false;
    bool spin_requested = false;
    bool spin_high_priority = false;

    // 底盘实际状态
    double velocity = 0.0;   // 线速度
    double omega = 0.0;      // 角速度

    // 位姿 / 地图（恢复模式的危险检测和目标搜索需要）
    Eigen::Vector3d chassis_pose_map = Eigen::Vector3d::Zero();
    const CostMap* merged_cost_map = nullptr;
    const DirectionMap* global_direction_map = nullptr;

    // 恢复目标点（由 MainController 维护/选择，FSM 仅用于退出判定）
    std::optional<Eigen::Vector2d> recovery_goal_map;

    // 时间戳
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
};

// FSM 每次 update() 的输出
struct FsmOutput {
    FsmState state = FsmState::IDLE;

    // STUCK_REVERSE 完成后要求清除全局路径
    bool clear_global_path = false;

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

    // 外部发布底盘指令后回调（用于卡住检测）
    void on_chassis_cmd_published(double velocity, double omega, const rclcpp::Time& stamp);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}