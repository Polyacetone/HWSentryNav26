#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>
#include <cstdint>

#include <path_follower/chassis_mode.hpp>
#include <path_follower/state_machine.hpp>
#include <path_follower/mpc_solver.hpp>
#include <path_follower/nav_map.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

enum class StepDirection : uint8_t {
    UP,
    DOWN,
};

// ═══════════════════════ 控制器输入 ═══════════════════════════

/// 每个控制周期由 Node 填写、传入 MainController
struct ControlInput {
    // ─── 路径 ───
    std::optional<SplineD> global_path;
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
    ChassisMode mode = ChassisMode::NORMAL;
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

struct StepDetectionParams {
    double detect_norm_threshold;
    double detect_dot_threshold;
    double path_sample_resolution;
    double lookahead_distance;
    double exit_advance_distance;
    double step_engage_distance;
};

struct NoProgressGuardParams {
    double landmark_spacing;
    double timeout;
};

struct StepBlockReplanParams {
    bool enable;
    double lookahead_distance;
    double sample_resolution;
    double step_norm_threshold;
    double obstacle_cost_threshold;
    double predicted_obstacle_ratio_threshold;
};

struct NavigationParams {
    double stop_threshold_dist;
    double stop_threshold_u;

    FollowProjectionGuardParams follow_proj_guard;
    StepDetectionParams step_detection;
    NoProgressGuardParams follow_no_progress_guard;
    NoProgressGuardParams stepping_no_progress_guard;
    StepBlockReplanParams step_block_replan;

    double step_dist_offset; // 补偿代价地图膨胀导致的台阶检测距离偏差 (m)
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
    void update_recovery_goal_if_needed(const ControlInput& input);
    bool check_stuck(const ControlInput& input);
    bool compute_is_hazard(const ControlInput& input) const;
    bool update_recovery_safe_flag(const ControlInput& input);
    void recompute_follow_landmarks(const SplineD& path);
    bool check_no_progress(const ControlInput& input, double current_u, const NoProgressGuardParams& params, FsmState current_state);

    // ─── 工具函数 ───
    struct StepPlanSegment {
        int path_version = 0;
        double stepping_enter_u = 0.0;
        double mode_enter_u = 0.0;
        double step_enter_u = 0.0;
        double step_exit_u = 0.0;
        double stepping_exit_u = 0.0;
        Eigen::Vector2d step_enter_pos_map = Eigen::Vector2d::Zero();
        Eigen::Vector2d step_exit_pos_map = Eigen::Vector2d::Zero();
        Eigen::Vector2d dir_map = Eigen::Vector2d::Zero();
        StepDirection direction = StepDirection::UP;
        ActiveStepMode command;
    };

    [[nodiscard]] double project_path_u(const ControlInput& input, const SplineD& path, double seed_u) const;
    double advance_path_u_by_distance(const SplineD& path, double start_u, double distance) const;
    double retreat_path_u_by_distance(const SplineD& path, double start_u, double distance) const;
    [[nodiscard]] bool check_follow_projection_guard(const ControlInput& input, const SplineD& path, double current_u) const;
    [[nodiscard]] bool check_step_block_replan(const ControlInput& input, const SplineD& path, double current_u) const;
    std::vector<StepPlanSegment> build_step_plan(const SplineD& path, const DirectionMap& direction_map) const;
    void clear_step_runtime_state();
    void clear_step_plan();
    void update_step_plan_for_path_change(bool has_new_path, const std::optional<SplineD>& path, const DirectionMap* direction_map);
    void update_active_step_segment(const ControlInput& input, double current_u);
    [[nodiscard]] std::optional<size_t> find_active_step_segment_index(double current_u) const;
    [[nodiscard]] const StepPlanSegment* active_step_segment(double current_u) const;
    [[nodiscard]] const StepPlanSegment* current_step_command_segment(double current_u) const;
    [[nodiscard]] uint8_t compute_step_distance_cm(const SplineD& path, double current_u) const;
    [[nodiscard]] bool should_engage_step_mode(double current_u) const;
    std::optional<ActiveStepMode> build_step_command(
        StepDirection direction,
        const Eigen::Vector2d& step_enter_pos_map,
        double step_enter_u,
        const DirectionMap& direction_map
    ) const;
    double step_speed_from_level(uint8_t speed_level) const;
    std::optional<ActiveStepMode> current_active_step_mode(double current_u) const;
    bool is_step_active(double current_u) const;

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
    std::optional<std::chrono::steady_clock::time_point> recovery_goal_set_time_;
    std::optional<std::chrono::steady_clock::time_point> recovery_safe_since_;
    FsmState last_fsm_state_ = FsmState::IDLE;
    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero();
    int path_version_ = 0;

    // ─── 外部安全观测状态 ───
    bool stuck_active_ = false;
    std::chrono::steady_clock::time_point stuck_start_time_;
    Eigen::Vector2d stuck_start_pos_ = Eigen::Vector2d::Zero();

    // ─── 台阶检测 / 锁存 / 执行状态 ───
    std::vector<StepPlanSegment> step_plan_;
    std::optional<size_t> active_step_segment_index_;
    std::optional<SplineD> step_locked_path_;
    bool step_locked_fixed_goal_ = false;
    Eigen::Vector2d step_locked_fixed_goal_pos_ = Eigen::Vector2d::Zero();
    bool deferred_external_path_update_ = false;

    // ─── 最近一次实际下发到底盘的控制指令 ───
    ControlOutput last_command_output_;
    bool has_last_command_output_ = false;

    bool follow_stop_and_wait_replan_pending_ = false;

    bool last_cycle_chassis_controllable_ = false;

    // ─── Follow/Stepping 路标点无进度检测状态 ───
    std::vector<double> follow_landmarks_u_;                 // 每隔 ~landmark_spacing 的路径参数 u
    int follow_max_landmark_idx_ = -1;                       // 已到达的最高路标点索引（-1=未初始化）
    std::chrono::steady_clock::time_point follow_max_landmark_time_;  // 最后一次路标更新时刻
    FsmState last_no_progress_check_state_ = FsmState::IDLE; // 上次调用 check_no_progress 的状态，用于模式切换时重置计时
};

}
