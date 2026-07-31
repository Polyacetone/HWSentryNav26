#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/common/chassis_defs.hpp>
#include <nav_executor/common/tracking/route_tracker.hpp>
#include <nav_executor/path_executor/state/state_machine.hpp>
#include <nav_executor/path_executor/state/step_controller.hpp>
#include <nav_executor/path_executor/monitoring/progress_monitor.hpp>
#include <nav_executor/path_executor/monitoring/safety_monitor.hpp>
#include <nav_executor/common/trajectory/annotated_path.hpp>
#include <nav_executor/common/environment/nav_map.hpp>
#include <nav_executor/common/environment/obstacle_semantics.hpp>
#include <nav_executor/path_executor/mpc/mpc_solver.hpp>

namespace nav_executor {

struct PathExecutorParams {
    double stop_threshold_dist;
    double stop_threshold_remaining_distance;
    double step_dist_offset;
    double command_history_timeout;
    NoProgressGuardParams follow_no_progress_guard;
    NoProgressGuardParams stepping_no_progress_guard;
};

enum class CommandStatus : uint8_t {
    NOT_EVALUATED = 0,
    PUBLISHED = 1,
    HELD_PREVIOUS = 2,
    INVALID = 3,
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
    uint64_t chassis_state_sequence = 0;
    uint8_t chassis_leg_mode = 4;
    uint8_t comp_stage = 4;
    std::chrono::steady_clock::time_point stamp;
};

struct MotionEnvironment {
    const FollowerObstacleView* obstacles = nullptr;
};

struct ExecutorInput {
    MotionIntent intent;
    MotionObservation observation;
    MotionEnvironment environment;
    std::optional<RouteEstimate> route;
};

struct ExecutorOutput {
    double velocity = 0.0;
    double omega = 0.0;
    uint8_t mode = chassis_mode::NORMAL;
    uint8_t step_dist_cm = 0;
    bool valid = false;
    CommandStatus command_status = CommandStatus::NOT_EVALUATED;

    MotionState motion_state = MotionState::IDLE;
    StepPhase step_phase = StepPhase::NONE;

    // one-shot 事实/事件
    bool goal_reached = false;
    bool executor_replan_event = false;
    bool mpc_lethal = false;

    // 调试
    ObserverDiagnostics observer_diagnostics;
    std::optional<std::vector<Eigen::Vector2d>> mpc_path_map;
    std::optional<MPCDiagnostics> mpc_diagnostics;

    // 仅供 PathExecutor 内部维护 MPC command state，不属于外部诊断契约。
    bool mpc_generated_command = false;
};

struct StepExecutionPreview {
    StepPhase phase = StepPhase::NONE;
    bool preemptible = true;
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
    [[nodiscard]] StepPhase step_phase() const { return control_fsm_->step_phase(); }
    [[nodiscard]] StepExecutionPreview preview_step_execution(
        const AnnotatedPath::ConstPtr& path,
        double path_progress,
        bool route_tracked
    ) const;

    [[nodiscard]] bool preemptible() const {
        const MotionState s = control_fsm_->state();
        switch (s) {
            case MotionState::SPIN:
                return !last_spin_high_priority_;
            case MotionState::STEPPING:
                return is_step_phase_precommit(control_fsm_->step_phase());
            case MotionState::FOLLOW:
            case MotionState::IDLE:
            case MotionState::PREPARE_SPIN:
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
    ExecutorOutput execute_prepare_spin(const ExecutorInput& input);
    ExecutorOutput execute_recovery(const ExecutorInput& input);
    ExecutorOutput execute_stuck_reverse();
    ExecutorOutput execute_fixed(const ExecutorInput& input);

    void sync_mpc_context(const ExecutorInput& input, bool allow_observer_update);
    void reset_mpc_observer(
        ObserverResetReason reason = ObserverResetReason::EXPLICIT_REQUEST
    );
    void reanchor_mpc_command_state(const ChassisMotionState& chassis_state);
    void invalidate_mpc_command_history(ObserverResetReason reason);
    void apply_held_command(ExecutorOutput& output) const;
    void remember_command_output(
        const ExecutorOutput& output,
        std::chrono::steady_clock::time_point stamp
    );
    void on_state_transition(MotionState prev, MotionState next, bool allow_warm_start_reset);
    [[nodiscard]] static bool state_uses_mpc(MotionState state);

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
    uint64_t bound_path_epoch_ = 0;

    bool last_spin_high_priority_ = false;
    MotionState last_motion_state_ = MotionState::IDLE;
    enum class MpcCommandHistory : uint8_t { TRACKED, NEEDS_REANCHOR };
    Eigen::Vector2d mpc_command_state_ = Eigen::Vector2d::Zero();
    Eigen::Vector2d mpc_command_rate_ = Eigen::Vector2d::Zero();
    MpcCommandHistory mpc_command_history_ = MpcCommandHistory::NEEDS_REANCHOR;
    std::optional<std::chrono::steady_clock::time_point> last_update_stamp_;
    std::optional<std::chrono::steady_clock::time_point> last_command_output_stamp_;
    std::optional<uint64_t> last_observer_state_sequence_;

    ExecutorOutput last_command_output_;
    bool has_last_command_output_ = false;

    bool mpc_lethal_pending_ = false;
    ChassisControlState last_cycle_chassis_control_state_ = ChassisControlState::STOPPED;
    bool last_cycle_chassis_controllable_ = false;
};

} // namespace nav_executor
