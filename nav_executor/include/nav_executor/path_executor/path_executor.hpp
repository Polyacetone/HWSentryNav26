#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/common/chassis_defs.hpp>
#include <nav_executor/path_executor/state_machine.hpp>
#include <nav_executor/path_executor/step_controller.hpp>
#include <nav_executor/path_executor/progress_monitor.hpp>
#include <nav_executor/path_executor/safety_monitor.hpp>
#include <nav_executor/common/annotated_path.hpp>
#include <nav_executor/path_planner/nav_map.hpp>
#include <nav_executor/path_executor/solver/mpc_solver.hpp>

namespace nav_executor {

struct PathExecutorParams {
    double stop_threshold_dist;
    double stop_threshold_u;
    double step_dist_offset;
    NoProgressGuardParams follow_no_progress_guard;
    NoProgressGuardParams stepping_no_progress_guard;
};

struct MotionIntent {
    AnnotatedPath::ConstPtr active_path;
    std::optional<Eigen::Vector2d> hold_goal;
    bool spin_requested = false;
    bool spin_high_priority = false;
    bool spin_fast = false;
};

struct MotionObservation {
    Eigen::Vector3d chassis_pose_map = Eigen::Vector3d::Zero();
    ChassisMotionState chassis_state;
    double remaining_energy = 0.0;
    double rfr_pwr_limit = 0.0;
    uint8_t chassis_leg_mode = 4;
    uint8_t comp_stage = 4;
    std::chrono::steady_clock::time_point stamp;
};

struct MotionEnvironment {
    const CostMap* final_cost_map = nullptr;          // masked_global + current_dynamic
    const CostMap* masked_global_cost_map = nullptr;  // global + step_cost_layer
    const DirectionMap* masked_direction_map = nullptr;
    const DirectionMap* base_direction_map = nullptr;
    const CostMap* current_dynamic_cost_map = nullptr;
    std::vector<const CostMap*> per_step_cost_maps;
    std::vector<const CostMap*> per_step_dynamic_cost_maps;
    double prediction_dt = 0.0;
};

struct ExecutorInput {
    MotionIntent intent;
    MotionObservation observation;
    MotionEnvironment environment;
};

struct ExecutorOutput {
    double velocity = 0.0;
    double omega = 0.0;
    uint8_t mode = chassis_mode::NORMAL;
    uint8_t step_dist_cm = 0;
    bool valid = false;

    MotionState motion_state = MotionState::IDLE;

    // one-shot 事实/事件
    bool goal_reached = false;
    bool executor_replan_event = false;
    bool mpc_lethal = false;

    // 当前投影 u（供顶层 RouteMonitor 复用为下周期 seed 及诊断）
    double current_u = 0.0;

    // 调试
    std::optional<std::vector<Eigen::Vector2d>> mpc_path_map;
    std::optional<std::vector<double>> predicted_v;
    std::optional<std::vector<double>> predicted_w;
    std::optional<std::vector<std::vector<Eigen::Vector2d>>> global_search_rollouts;
};

// 运动控制编排：持有 FSM / MPC / 台阶运行时 / stuck 检测与恢复链，消费顶层每周期传入的 active_path 与 hold_goal。
class PathExecutor {
public:
    PathExecutor(
        const PathExecutorParams& params,
        const FsmParams& fsm_params,
        std::shared_ptr<MPCSolver> mpc_controller,
        const CapabilityProfile& normal_profile,
        const std::array<CapabilityProfile, 3>& capability_profiles,
        const ProfileBlendParams& blend_params,
        rclcpp::Logger logger
    );

    ExecutorOutput update(const ExecutorInput& input);

    [[nodiscard]] MotionState motion_state() const { return control_fsm_->state(); }

    [[nodiscard]] bool preemptible() const {
        const MotionState s = control_fsm_->state();
        switch (s) {
            case MotionState::SPIN:
                return !last_spin_high_priority_;
            case MotionState::FOLLOW:
            case MotionState::IDLE:
            case MotionState::STOPPING:
            case MotionState::FIXED:
                return true;
            default:
                return false;
        }
    }

private:
    ExecutorOutput execute_idle();
    ExecutorOutput execute_follow(const ExecutorInput& input, bool check_lethal_status);
    ExecutorOutput execute_spin(const ExecutorInput& input);
    ExecutorOutput execute_stop(const ExecutorInput& input);
    ExecutorOutput execute_recovery(const ExecutorInput& input);
    ExecutorOutput execute_stuck_reverse();
    ExecutorOutput execute_fixed(const ExecutorInput& input);

    void sync_mpc_context(const ExecutorInput& input, bool allow_observer_update);
    void apply_held_command(ExecutorOutput& output) const;
    void remember_command_output(const ExecutorOutput& output);
    void on_state_transition(MotionState prev, MotionState next, bool allow_warm_start_reset);

    [[nodiscard]] double project_path_u(const ExecutorInput& input, const SplinePath& path, double seed_u) const;

    std::unique_ptr<StateMachine> control_fsm_;
    std::shared_ptr<MPCSolver> mpc_controller_;
    StepController step_controller_;
    ProgressMonitor progress_monitor_;
    SafetyMonitor safety_monitor_;
    rclcpp::Logger logger_;

    PathExecutorParams params_;
    FsmParams fsm_params_;

    // 当前绑定的 path（身份用于检测切换）
    AnnotatedPath::ConstPtr bound_path_;

    double last_reference_u_ = 0.0;
    bool last_spin_high_priority_ = false;
    MotionState last_motion_state_ = MotionState::IDLE;
    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero();

    ExecutorOutput last_command_output_;
    bool has_last_command_output_ = false;

    bool mpc_lethal_pending_ = false;
    ChassisControlState last_cycle_chassis_control_state_ = ChassisControlState::STOPPED;
    bool last_cycle_chassis_controllable_ = false;
};

} // namespace nav_executor
