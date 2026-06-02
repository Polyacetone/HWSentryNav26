#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>
#include <cstdint>

#include <path_follower/common/chassis_defs.hpp>
#include <path_follower/control/state_machine.hpp>
#include <path_follower/solver/mpc_solver.hpp>
#include <path_follower/common/nav_map.hpp>
#include <path_follower/step/step_controller.hpp>
#include <path_follower/path/progress_monitor.hpp>
#include <path_follower/safety/safety_monitor.hpp>

namespace path_follower {

struct ControlInput {
    std::optional<SplinePath> global_path;
    bool path_updated;

    bool fixed_goal;
    Eigen::Vector2d fixed_goal_pos;

    Eigen::Vector3d chassis_pose_map;
    ChassisMotionState chassis_state;

    double remaining_energy;
    double rfr_pwr_limit;

    uint8_t chassis_leg_mode;
    uint8_t comp_stage;

    bool spin_requested;
    bool spin_high_priority;
    bool spin_slow;
    bool spin_fast;

    const CostMap* const final_cost_map;
    const CostMap* const masked_global_cost_map;
    const DirectionMap* const masked_direction_map;
    const DirectionMap* const base_direction_map = nullptr;
    const CostMap* const current_dynamic_cost_map;

    std::vector<const CostMap*> per_step_cost_maps;
    std::vector<const CostMap*> per_step_dynamic_cost_maps;
    double prediction_dt = 0.0;

    std::chrono::steady_clock::time_point stamp;
};

struct ControlOutput {
    double velocity = 0.0;
    double omega = 0.0;
    uint8_t mode = chassis_mode::NORMAL;
    uint8_t step_dist_cm = 0;

    FsmState fsm_state = FsmState::IDLE;
    bool consume_global_path = false;
    bool keep_goal_on_path_consume = false;
    bool request_replan = false;

    std::optional<std::vector<Eigen::Vector2d>> predicted_path_map;
    std::optional<std::vector<double>> predicted_v;
    std::optional<std::vector<double>> predicted_w;
    std::optional<std::vector<std::vector<Eigen::Vector2d>>> mppi_rollouts;

    bool valid = false;
};

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

class MainController {
public:
    MainController(
        const NavigationParams& nav_params,
        const FsmParams& fsm_params,
        std::shared_ptr<MPCSolver> mpc_controller,
        const CapabilityProfile& normal_profile,
        const std::array<CapabilityProfile, 3>& capability_profiles,
        const ProfileBlendParams& blend_params,
        rclcpp::Logger logger
    );

    ControlOutput update(const ControlInput& input);

    [[nodiscard]] FsmState fsm_state() const;
    [[nodiscard]] double last_reference_u() const { return last_reference_u_; }

private:
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

    std::unique_ptr<StateMachine> control_fsm_;
    std::shared_ptr<MPCSolver> mpc_controller_;
    StepController step_controller_;
    ProgressMonitor progress_monitor_;
    SafetyMonitor safety_monitor_;
    rclcpp::Logger logger_;

    NavigationParams nav_params_;
    FsmParams fsm_params_;

    double last_reference_u_ = 0.0;
    FsmState last_fsm_state_ = FsmState::IDLE;
    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero();

    ControlOutput last_command_output_;
    bool has_last_command_output_ = false;

    bool follow_stop_and_wait_replan_pending_ = false;
    bool last_cycle_chassis_controllable_ = false;
};

} // namespace path_follower
