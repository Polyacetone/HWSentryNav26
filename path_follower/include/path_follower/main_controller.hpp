#pragma once

#include <memory>
#include <optional>
#include <vector>
#include <deque>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

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
    std::optional<std::vector<Eigen::Vector2d>> step_preview_path_map;

    // ─── 有效性 ───
    bool valid = false;                 // false 时 Node 不应发布指令
};

// ═══════════════════════ 控制器参数 ═══════════════════════════

struct NavigationParams {
    double stop_threshold_dist;
    double stop_threshold_u;

    // Follow 任务取消：投影守卫
    double follow_proj_dist_max;      // 当前位置到样条投影点距离过大则取消 (m)
    double follow_proj_cost_max;      // 当前位置到投影点连线最大代价超过则取消 (0~255)
    int follow_proj_cost_samples;     // 连线采样数（用于取最大代价）

    // 台阶检测（基于MPC预测轨迹）
    double step_detect_norm_threshold;    // 方向场模长阈值，>=该值认为经过台阶区域
    double step_detect_dot_threshold;     // 朝向与方向场点积阈值；< -dot_threshold 为下台阶，> dot_threshold 为上台阶
    double step_edge_norm_threshold;      // 台阶边缘判定方向场模长阈值（上台阶空间检测使用）
    int step_on_count_threshold;          // [仅下台阶] 连续检测到台阶轨迹的次数才设置 step_down_flag
    int step_off_count_threshold;         // [仅下台阶] 连续未检测到台阶轨迹的次数才清除 step_down_flag

    // 上台阶空间锁存与预评估
    double step_path_lookahead_distance;  // 沿路径向前扫查台阶目标的固定距离 (m)
    double step_path_sample_resolution;   // 沿路径采样分辨率 (m)
    double step_target_match_distance;    // 连续检测时视为同一台阶目标的位置阈值 (m)
    int step_latch_threshold;             // 连续检测到同一台阶目标的次数才锁存
    double step_preview_hit_distance;     // 长预测命中已锁存台阶目标的位置阈值 (m)
    double step_preview_speed_error_threshold;
    double step_preview_heading_error_threshold;
    double step_preview_speed_error_weight;
    double step_preview_heading_error_weight;
    double step_release_distance;

    // 上台阶失败兜底（防止在 Dead↔Follow 恢复循环中卡死）
    bool step_up_failsafe_enable;
    int step_up_failsafe_similar_attempts;   // 连续相近位置出现次数阈值
    double step_up_failsafe_similar_dist;    // 位置相近判定距离 (m)

    // Follow 路标点无进度检测（防止在障碍物前蠕动不前）
    bool no_progress_enable;
    double no_progress_landmark_spacing;     // 路标点间距 (m)
    double no_progress_timeout;              // 无进度超时 (s)

    // 上台阶助跑点搜索参数
    double step_runup_radius_min;               // 采样半径下界 (m)，以台阶边缘点 A 为圆心
    double step_runup_radius_max;               // 采样半径上界 (m)，以台阶边缘点 A 为圆心
    double step_runup_angle_half_range;         // 扇区半角 (rad)，以台阶边缘点 A 的方向为扇区中心，逆时针为正
    int step_runup_radius_samples;              // 半径维采样数
    int step_runup_angle_samples;               // 角度维采样数
    double step_runup_cost_threshold;           // 助跑点代价阈值（0~255），超过则认为不可行
    double step_runup_path_integral_resolution; // 助跑点到台阶边缘点 A 连线的路径积分采样分辨率 (m)
    int step_runup_line_check_samples;          // 助跑点到台阶边缘点 A 连线的代价检查采样数（用于取最大代价）
    double step_runup_cost_weight;              // 助跑点代价权重（0~1），用于与距离/角度成本加权求和，选取总成本最低的点
    double step_runup_step_dist_weight;         // 助跑点到台阶边缘点 A 的距离权重（0~1），用于与代价/角度成本加权求和
    double step_runup_angle_weight;             // 助跑点与台阶边缘点 A 的连线与台阶边缘方向夹角权重（0~1），用于与距离/代价成本加权求和
    double step_runup_robot_dist_weight;        // 助跑点到机器人当前位置的距离权重（0~1），用于与代价/角度成本加权求和，鼓励选择更靠近机器人的助跑点，避免过长的助跑路径
    double step_runup_robot_path_cost_weight;   // 助跑点到机器人当前位置的路径代价权重（0~1），用于与距离/角度成本加权求和，鼓励选择路径代价更低的助跑点，避免过于复杂的助跑路径
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

    void on_state_transition(FsmState prev, FsmState next);
    void update_recovery_goal_if_needed(const ControlInput& input);
    bool check_stuck(const ControlInput& input);
    bool compute_is_hazard(const ControlInput& input) const;
    bool update_recovery_safe_flag(const ControlInput& input);
    void recompute_follow_landmarks(const SplineD& path);
    bool check_no_progress(const ControlInput& input);
    std::optional<Eigen::Vector2d> select_step_runup_point(const ControlInput& input) const;
    bool is_step_runup_segment_feasible(const CostMap& cost_map, const Eigen::Vector2d& from_map, const Eigen::Vector2d& to_map) const;

    // ─── 工具函数 ───
    struct StepDetectResult {
        bool step_down_rising = false;
        bool step_down = false;
    };

    struct PathStepTarget {
        int path_version = 0;
        double path_u = 0.0;
        double release_u = 0.0;
        Eigen::Vector2d pos_map = Eigen::Vector2d::Zero();
        Eigen::Vector2d dir_map = Eigen::Vector2d::Zero();
    };

    struct StepPreviewEvaluation {
        ChassisMode mode = ChassisMode::NORMAL;
        size_t hit_index = 0;
        double predicted_speed = 0.0;
        double speed_error = 0.0;
        double heading_error = 0.0;
        double score = 0.0;
        std::vector<Eigen::Vector2d> path_map;
    };

    StepDetectResult detect_steps_on_prediction_with_edges(
        const MPCPrediction& prediction,
        const DirectionMap& direction_map
    );
    std::optional<PathStepTarget> detect_step_target_on_path(
        const SplineD& path,
        double start_u,
        const DirectionMap& direction_map
    ) const;
    double advance_path_u_by_distance(const SplineD& path, double start_u, double distance) const;
    bool is_same_step_target(const PathStepTarget& lhs, const PathStepTarget& rhs) const;
    void clear_step_up_decision();
    void update_step_up_state_for_path_change(bool has_new_path);
    void update_step_up_release(const SplineD& path, double current_u);
    std::optional<StepPreviewEvaluation> evaluate_step_preview_candidate(
        MPCSolver& preview_controller,
        const ControlInput& input,
        const SplineD& path,
        const PathStepTarget& target,
        double start_u,
        ChassisMode preview_mode
    );
    double step_target_speed(ChassisMode mode) const;
    std::optional<PathStepTarget> try_latch_step_up_target(const SplineD& path, double current_u, const DirectionMap& direction_map);
    void finalize_step_up_mode_selection(
        const ControlInput& input,
        std::optional<StepPreviewEvaluation> leg_eval,
        std::optional<StepPreviewEvaluation> jump_eval
    );

    void clear_step_up_attempt_history();
    void clear_step_up_attempt_history_if_needed(bool has_path, bool has_new_path);
    bool register_step_up_attempt_and_should_cancel(const Eigen::Vector2d& pos_map);

    static double wrap_pi(double a) {
        return std::atan2(std::sin(a), std::cos(a));
    }

    // ─── 核心组件 ───
    std::unique_ptr<StateMachine> control_fsm_;
    std::shared_ptr<MPCSolver> mpc_controller_;
    std::shared_ptr<MPCSolver> preview_leg_controller_;
    std::shared_ptr<MPCSolver> preview_jump_controller_;
    rclcpp::Logger logger_;

    // ─── 参数 ───
    NavigationParams nav_params_;
    FsmParams fsm_params_;

    // ─── 内部状态 ───
    double last_reference_u_ = 0.0;
    std::optional<Eigen::Vector2d> recovery_goal_map_;
    std::optional<std::chrono::steady_clock::time_point> recovery_goal_set_time_;
    std::optional<std::chrono::steady_clock::time_point> recovery_safe_since_;
    std::optional<PathStepTarget> pending_step_runup_target_;
    std::optional<Eigen::Vector2d> step_runup_goal_map_;
    std::optional<std::vector<Eigen::Vector2d>> debug_step_preview_path_map_;
    FsmState last_fsm_state_ = FsmState::IDLE;
    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero();
    int path_version_ = 0;

    // ─── 外部安全观测状态 ───
    bool stuck_active_ = false;
    std::chrono::steady_clock::time_point stuck_start_time_;
    Eigen::Vector2d stuck_start_pos_ = Eigen::Vector2d::Zero();

    // ─── 台阶检测防抖状态 ───
    int step_down_on_count_ = 0;
    int step_down_off_count_ = 0;
    bool step_down_flag_ = false;

    // ─── 上台阶目标锁存与模式决策 ───
    std::optional<PathStepTarget> pending_step_target_detection_;
    int pending_step_target_on_count_ = 0;
    std::optional<PathStepTarget> latched_step_target_;
    std::optional<ChassisMode> latched_step_up_mode_;

    // ─── 复活检测（底盘 Dead -> Mature） ───
    bool last_cycle_chassis_dead_ = false;

    // ─── 上台阶失败兜底状态 ───
    std::deque<Eigen::Vector2d> step_up_attempt_positions_;  // 仅保存最近 N 次
    bool pending_cancel_follow_task_ = false;                // 由 FOLLOW 内检测触发，下一周期执行取消
    bool pending_step_runup_ = false;
    bool step_runup_done_ = false;

    // ─── Follow 路标点无进度检测状态 ───
    std::vector<double> follow_landmarks_u_;                 // 每隔 ~landmark_spacing 的路径参数 u
    int follow_max_landmark_idx_ = -1;                       // 已到达的最高路标点索引（-1=未初始化）
    std::chrono::steady_clock::time_point follow_max_landmark_time_;  // 最后一次路标更新时刻
};

}
