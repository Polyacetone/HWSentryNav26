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

// ═══════════════════════ 控制器输入 ═══════════════════════════

/// 每个控制周期由 Node 填写、传入 MainController
struct ControlInput {
    // ─── 路径 ───
    std::optional<SplineD> global_path;
    bool path_updated;

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

    // ─── 逐步预测代价地图 ───
    std::vector<const CostMap*> per_step_cost_maps; // 每个预测时间步的代价地图（可为空）
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

    // ─── 状态信息 ───
    FsmState fsm_state = FsmState::IDLE;
    bool consume_global_path = false;   // 通知 Node 消费当前全局路径，并据此重建台阶擦除地图

    // ─── 调试 ───
    std::optional<std::vector<Eigen::Vector2d>> predicted_path_map;
    std::optional<std::vector<double>> predicted_v;
    std::optional<std::vector<double>> predicted_w;
    std::optional<std::vector<Eigen::Vector2d>> step_rollout_path_map;

    // ─── 有效性 ───
    bool valid = false;                 // false 时 Node 不应发布指令
};

// ═══════════════════════ 控制器参数 ═══════════════════════════

struct FollowProjectionGuardParams {
    double dist_max;
    double cost_max;
    int cost_samples;
};

struct StepLookaheadParams {
    double rollout_length_filter_alpha;
    double fixed_extension_distance;
    double min_distance;
};

struct StepDetectionParams {
    double detect_norm_threshold;
    double detect_dot_threshold;
    double path_sample_resolution;
    double target_match_distance;
    int latch_threshold;
    double release_distance;
    StepLookaheadParams lookahead;
};

struct StepRunupSearchParams {
    double radius_min;
    double radius_max;
    int radius_samples;
    double sector_half_angle_rad;
    int angle_samples;
    double candidate_cost_max;
    double line_cost_max;
    double safe_step_norm_threshold;
    double path_integral_resolution;
    double path_integral_cost_weight;
    double path_integral_step_weight;
    double radius_preference_weight;
};

struct StepRunupParams {
    double velocity_deficit_threshold;
    double goal_tolerance;
    StepRunupSearchParams search;
};

struct NoProgressGuardParams {
    bool enable;
    double landmark_spacing;
    double timeout;
};

struct NavigationParams {
    double stop_threshold_dist;
    double stop_threshold_u;

    FollowProjectionGuardParams follow_proj_guard;
    StepDetectionParams step_detection;
    StepRunupParams step_runup;
    NoProgressGuardParams no_progress_guard;
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
    ControlOutput execute_step_runup(const ControlInput& input);
    void sync_mpc_context(const ControlInput& input);
    void reset_all_mpc_warm_start();
    void reset_all_mpc_observer();

    void on_state_transition(FsmState prev, FsmState next);
    void update_recovery_goal_if_needed(const ControlInput& input);
    bool check_stuck(const ControlInput& input);
    bool compute_is_hazard(const ControlInput& input) const;
    bool update_recovery_safe_flag(const ControlInput& input);
    void recompute_follow_landmarks(const SplineD& path);
    bool check_no_progress(const ControlInput& input);

    // ─── 工具函数 ───
    enum class StepDirection : uint8_t {
        UP,
        DOWN,
    };

    struct PathStepTarget {
        int path_version = 0;
        double path_u = 0.0;
        double release_u = 0.0;
        Eigen::Vector2d pos_map = Eigen::Vector2d::Zero();
        Eigen::Vector2d dir_map = Eigen::Vector2d::Zero();
        StepDirection direction = StepDirection::UP;
    };

    struct StepRunupContext {
        PathStepTarget step_target;
        ActiveStepMode step_command;
        Eigen::Vector2d goal_map = Eigen::Vector2d::Zero();
        double preferred_backoff_distance = 0.0;
        double predicted_arrival_velocity = 0.0;
        double velocity_deficit = 0.0;
    };

    struct StepRunupDecision {
        bool cancel_path = false;
        std::optional<StepRunupContext> context;
        std::optional<std::vector<Eigen::Vector2d>> debug_rollout_path_map;
    };

    std::optional<PathStepTarget> detect_step_target_on_path(
        const SplineD& path,
        double start_u,
        const DirectionMap& direction_map
    ) const;
    static double prediction_path_length(const MPCPrediction& prediction);
    void update_step_lookahead_distance(const MPCPrediction& prediction);
    [[nodiscard]] double current_step_lookahead_distance() const;
    double advance_path_u_by_distance(const SplineD& path, double start_u, double distance) const;
    bool is_same_step_target(const PathStepTarget& lhs, const PathStepTarget& rhs) const;
    void clear_step_state();
    void clear_step_runup_state(bool clear_last_completed_target = false);
    void update_step_state_for_path_change(bool has_new_path);
    void update_step_release(const SplineD& path, double current_u);
    std::optional<PathStepTarget> try_latch_step_target(const SplineD& path, double current_u, const DirectionMap& direction_map);
    std::optional<ActiveStepMode> build_step_command(const PathStepTarget& target, const DirectionMap& direction_map) const;
    StepRunupDecision evaluate_step_runup(
        const ControlInput& input,
        const SplineD& path,
        double current_u,
        const PathStepTarget& target,
        const ActiveStepMode& step_command
    ) const;
    std::optional<Eigen::Vector2d> find_step_runup_goal(
        const ControlInput& input,
        const PathStepTarget& target,
        double preferred_backoff_distance
    ) const;
    bool is_step_runup_goal_reached(const ControlInput& input) const;
    double step_speed_from_level(uint8_t speed_level) const;
    std::optional<ActiveStepMode> current_active_step_mode() const;
    bool is_step_active() const;

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

    double filtered_follow_rollout_length_ = 0.0;
    double step_lookahead_distance_ = 0.0;

    // ─── 外部安全观测状态 ───
    bool stuck_active_ = false;
    std::chrono::steady_clock::time_point stuck_start_time_;
    Eigen::Vector2d stuck_start_pos_ = Eigen::Vector2d::Zero();

    // ─── 台阶检测 / 锁存 / 执行状态 ───
    std::optional<PathStepTarget> pending_step_target_detection_;
    int pending_step_target_on_count_ = 0;
    std::optional<PathStepTarget> active_step_target_;
    std::optional<ActiveStepMode> active_step_command_;
    bool step_runup_request_pending_ = false;
    bool step_runup_goal_reached_ = false;
    std::optional<StepRunupContext> step_runup_context_;
    std::optional<PathStepTarget> last_completed_step_runup_target_;

    // ─── 复活检测（底盘 Dead -> Mature） ───
    bool last_cycle_chassis_dead_ = false;
    uint8_t last_leg_mode_ = 0;

    // ─── Follow 路标点无进度检测状态 ───
    std::vector<double> follow_landmarks_u_;                 // 每隔 ~landmark_spacing 的路径参数 u
    int follow_max_landmark_idx_ = -1;                       // 已到达的最高路标点索引（-1=未初始化）
    std::chrono::steady_clock::time_point follow_max_landmark_time_;  // 最后一次路标更新时刻
};

}
