#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>
#include <cstdint>

#include <path_follower/chassis_defs.hpp>
#include <path_follower/state_machine.hpp>
#include <path_follower/mpc_solver.hpp>
#include <path_follower/nav_map.hpp>
#include <path_follower/step_controller.hpp>
#include <path_follower/progress_monitor.hpp>
#include <path_follower/safety_monitor.hpp>

namespace path_follower {

// ═══════════════════════ 控制器输入 ═══════════════════════════

/// 每个控制周期由 Node 填写、传入 MainController
struct ControlInput {
    // ─── 路径 ───
    std::optional<SplinePath> global_path;
    bool path_updated;  // 本周期是否收到路径更新（包括空路径）

    // ─── fixed 目标 ───
    bool fixed_goal;                   // 当前目标为 fixed 类型
    Eigen::Vector2d fixed_goal_pos;    // fixed 目标位置 (map)

    // ─── 底盘状态 ───
    Eigen::Vector3d chassis_pose_map;    // (x, y, theta) in map
    ChassisMotionState chassis_state;    // (v_act, ω_act, leg_h, leg_psi)

    // ─── 能量状态 ───
    double remaining_energy;     // 电容剩余可用电量 (J)
    double rfr_pwr_limit;        // 裁判系统最大取电功率 (W)

    // ─── 底盘模式/比赛状态 ───
    uint8_t chassis_leg_mode;    // 参考 interfaces/msg/ChassisStatus.msg
    uint8_t comp_stage;          // 参考 interfaces/msg/CompStage.msg

    // ─── 小陀螺请求 ───
    bool spin_requested;
    bool spin_high_priority;
    bool spin_slow;
    bool spin_fast;

    // ─── 地图 ───
    const CostMap* const final_cost_map; // 全局先验代价地图 + 台阶掩码 + 动态障碍物
    const CostMap* const masked_global_cost_map; // 全局先验代价地图 + 台阶掩码（不含动态障碍物）
    const DirectionMap* const masked_direction_map; // 全局先验方向场 - 台阶掩码
    const DirectionMap* const base_direction_map = nullptr; // 原始（未掩码）台阶方向场，用于恢复上台阶感知
    const CostMap* const current_dynamic_cost_map; // 当前时刻动态障碍物代价地图（通常为 local_cost_map）

    // ─── 逐步预测代价地图 ───
    std::vector<const CostMap*> per_step_cost_maps; // 每个预测时间步的代价地图（可为空）
    std::vector<const CostMap*> per_step_dynamic_cost_maps; // 每个预测时间步的动态障碍物代价地图（非空=预测模式）
    double prediction_dt = 0.0; // 预测步长 (s)

    // ─── 时间 ───
    std::chrono::steady_clock::time_point stamp;
};

// ═══════════════════════ 控制器输出 ═══════════════════════════

/// MainController 每个周期的输出
struct ControlOutput {
    // ─── 底盘指令 ───
    double velocity = 0.0;
    double omega = 0.0;
    uint8_t mode = chassis_mode::NORMAL;
    uint8_t step_dist_cm = 0;

    // ─── 状态信息 ───
    FsmState fsm_state = FsmState::IDLE;
    bool consume_global_path = false;   // 通知 Node 消费当前全局路径，并据此重建台阶擦除地图
    bool keep_goal_on_path_consume = false; // 消费当前路径但保留目标，等待 planner 返回新路径
    bool request_replan = false;        // 通知 Node 向 path_planner 发布一次重规划触发

    // ─── 调试 ───
    std::optional<std::vector<Eigen::Vector2d>> predicted_path_map;
    std::optional<std::vector<double>> predicted_v;
    std::optional<std::vector<double>> predicted_w;
    std::optional<std::vector<std::vector<Eigen::Vector2d>>> mppi_rollouts;

    // ─── 有效性 ───
    bool valid = false;                 // false 时 Node 不应发布指令
};

// ═══════════════════════ 控制器参数 ═══════════════════════════

struct FollowProjectionGuardParams {
    double dist_max;
    double cost_max;
    int cost_samples;
};

struct NavigationParams {
    double stop_threshold_dist;
    double stop_threshold_u;

    FollowProjectionGuardParams follow_proj_guard;
    StepDetectionParams step_detection;
    NoProgressGuardParams follow_no_progress_guard;
    NoProgressGuardParams stepping_no_progress_guard;
    StepBlockReplanParams step_block_replan;

    double step_dist_offset;
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
    ControlOutput execute_fixed(const ControlInput& input);
    void sync_mpc_context(const ControlInput& input, bool allow_observer_update);
    void reset_all_mpc_warm_start();
    void reset_all_mpc_observer();
    void apply_held_command(ControlOutput& output) const;
    void remember_command_output(const ControlOutput& output);

    void on_state_transition(FsmState prev, FsmState next, bool allow_warm_start_reset);

    [[nodiscard]] double project_path_u(const ControlInput& input, const SplinePath& path, double seed_u) const;
    [[nodiscard]] bool check_follow_projection_guard(const ControlInput& input, const SplinePath& path, double current_u) const;

    static double wrap_pi(double a) {
        return std::atan2(std::sin(a), std::cos(a));
    }

    // ─── 核心组件 ───
    std::unique_ptr<StateMachine> control_fsm_;
    std::shared_ptr<MPCSolver> mpc_controller_;
    StepController step_controller_;
    ProgressMonitor progress_monitor_;
    SafetyMonitor safety_monitor_;
    rclcpp::Logger logger_;

    // ─── 参数 ───
    NavigationParams nav_params_;
    FsmParams fsm_params_;

    // ─── 内部状态（仅编排层持有） ───
    double last_reference_u_ = 0.0;
    FsmState last_fsm_state_ = FsmState::IDLE;
    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero();

    // ─── 最近一次实际下发到底盘的控制指令 ───
    ControlOutput last_command_output_;
    bool has_last_command_output_ = false;

    bool follow_stop_and_wait_replan_pending_ = false;
    bool last_cycle_chassis_controllable_ = false;
};

} // namespace path_follower
