#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>
#include <rclcpp/time.hpp>

#include <path_follower/state_machine.hpp>
#include <path_follower/mpc_solver.hpp>
#include <path_follower/nav_map.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

// ═══════════════════════ 控制器输入 ═══════════════════════════

/// 每个控制周期由 Node 填写、传入 MainController
struct ControlInput {
    // ─── 路径 ───
    std::optional<SplineD> global_path;

    // ─── 底盘状态 ───
    Eigen::Vector3d chassis_pose_map = Eigen::Vector3d::Zero();  // (x, y, theta) in map
    Eigen::Vector2d chassis_status = Eigen::Vector2d::Zero();    // (v_act, ω_act)

    // ─── 小陀螺请求 ───
    bool spin_requested = false;
    bool spin_high_priority = false;
    bool spin_slow = false;
    bool spin_fast = false;

    // ─── 地图 ───
    const CostMap* merged_cost_map = nullptr;
    const DirectionMap* global_direction_map = nullptr;

    // ─── 时间 ───
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
};

// ═══════════════════════ 控制器输出 ═══════════════════════════

/// MainController 每个周期的输出
struct ControlOutput {
    // ─── 底盘指令 ───
    double velocity = 0.0;
    double omega = 0.0;
    bool step_up_ahead = false;
    bool step_down_ahead = false;
    bool slow_spin = false;
    bool fast_spin = false;

    // ─── 状态信息 ───
    FsmState fsm_state = FsmState::IDLE;
    bool path_cleared = false;          // 通知 Node 清除全局路径

    // ─── 调试 ───
    std::optional<std::vector<Eigen::Vector2d>> predicted_path_map;

    // ─── 有效性 ───
    bool valid = false;                 // false 时 Node 不应发布指令
};

// ═══════════════════════ 控制器参数 ═══════════════════════════

struct NavigationParams {
    double stop_threshold_dist;
    double stop_threshold_u;

    // 台阶检测
    double step_check_back;
    double step_check_front;
    double step_check_sample_step;
};

// ═══════════════════ MainController ═════════════════════

/// 控制逻辑层：接收传感器/状态数据，调用 FSM 决策 + MPC 计算，输出底盘指令。
/// 不依赖 ROS 通信，便于测试和维护。
class MainController {
public:
    MainController(
        const NavigationParams& nav_params,
        const FsmParams& fsm_params,
        std::shared_ptr<MPCSolver> mpc_controller,
        rclcpp::Logger logger
    );

    /// 每个控制周期调用一次
    ControlOutput update(const ControlInput& input);

    /// 查询当前 FSM 状态
    [[nodiscard]] FsmState fsm_state() const;

    /// 获取当前 spline 投影 u（供 Node 判定到达）
    [[nodiscard]] double last_reference_u() const { return last_reference_u_; }

private:
    // ─── 各状态的执行函数 ───
    ControlOutput execute_idle(const ControlInput& input);
    ControlOutput execute_follow(const ControlInput& input);
    ControlOutput execute_spin(const ControlInput& input);
    ControlOutput execute_stop(const ControlInput& input);
    ControlOutput execute_recovery(const ControlInput& input);
    ControlOutput execute_stuck_reverse(const ControlInput& input);

    void on_state_transition(FsmState prev, FsmState next);
    void update_recovery_goal_if_needed(const ControlInput& input);

    // ─── 工具函数 ───
    std::tuple<bool, bool> detect_steps_on_spline(const ControlInput& input, double u0) const;

    static double wrap_pi(double a) {
        return std::atan2(std::sin(a), std::cos(a));
    }

    // ─── 核心组件 ───
    std::unique_ptr<StateMachine> control_fsm_;
    std::shared_ptr<MPCSolver> mpc_controller_;
    rclcpp::Logger logger_;

    // ─── 参数 ───
    NavigationParams nav_params_;
    FsmParams fsm_params_;

    // ─── 内部状态 ───
    double last_reference_u_ = 0.0;
    std::optional<Eigen::Vector2d> recovery_goal_map_;
    rclcpp::Time recovery_goal_set_time_{0, 0, RCL_ROS_TIME};
    FsmState last_fsm_state_ = FsmState::IDLE;
};

}