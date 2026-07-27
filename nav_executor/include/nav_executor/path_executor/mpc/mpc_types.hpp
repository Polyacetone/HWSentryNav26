#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <nav_executor/common/trajectory/step_plan.hpp>
#include <nav_executor/common/environment/nav_map.hpp>

namespace nav_executor {

constexpr int MPC_HORIZON = 40;
constexpr double MPC_DT = 0.05;
constexpr int MPC_NX = 12;
constexpr int MPC_NU = 3;

constexpr double SOLVER_TOL_GRAD = 1e-6;

namespace ix {
    enum {
        X = 0,
        Y,
        THETA,
        XH,
        V,
        W,
        V_CMD,
        W_CMD,
        V_CMD_RATE,
        W_CMD_RATE,
        PATH_PROGRESS,
        PATH_SPEED,
    };
}

namespace iu {
    // 前两项是下位机速度指令的变化率，不是底盘实测物理加速度。
    enum { V_CMD_RATE = 0, W_CMD_RATE, PATH_SPEED_CMD };
}

struct MPCStartCommandLimits {
    double vel_cmd_act_gap_max;
    double omega_cmd_act_gap_max;
};

struct MPCCommandDynamicsWeights {
    double lateral_acceleration;
};

struct MPCFollowTrackingWeights {
    double contour;
    double lag;
    double heading;
    double velocity;
    double angular_velocity;
};

struct MPCFollowProgressParams {
    double progress_reward;
    double speed_tracking_weight;
    double speed_smoothness_weight;
};

struct MPCFollowTerminalWeights {
    double position;
    double heading;
    double velocity;
    double angular_velocity;
    double remaining_progress;
};

struct MPCFollowCommandWeights {
    double r_v;
    double r_omega;
    double r_dv;
    double r_domega;
    double r_jerk_v;
    double r_jerk_omega;
};

struct MPCTraversalTargetWeights {
    double velocity;
    double direction;
    double angular_velocity_command;
    double angular_velocity_predicted;
    double velocity_command_smoothness;
    double velocity_predicted_smoothness;
};

struct MPCFollowEnvironmentWeights {
    double obstacle;
};

struct ChassisMotionState {
    double velocity = 0.0;
    double omega = 0.0;
    double leg_h = 0.0;
    double leg_psi = 0.0;
};

struct LPVKinematicModelParams {
    double z_ref;
    double z_scale;
    double rho_clip;
    double sgn_eps;

    double ca00;
    double ca01;
    double ca10;
    double ca11;
    double cb0;
    double cb1;

    double dca00;
    double dca01;
    double dca10;
    double dca11;
    double dcb0;
    double dcb1;

    double gxh;
    double gv;
    double cf1;
    double cf2;

    double w_lam0;
    double w_k0;
    double w_cf0;
    double w_lam1;
    double w_k1;
    double w_cf1;

    double psi_bias;
    double psi_gain;
    double psi_v;
    double obs_lv;
    double obs_v_correction_clip;
    double obs_v_reset_threshold;
};

struct LPVDiscreteModel {
    double rho;
    double ad00;
    double ad01;
    double ad10;
    double ad11;
    double bd0;
    double bd1;
    double gd0;
    double gd1;
    double alpha_w;
    double beta_w;
    double gamma_w;
    double sgn_eps;
    double cf1;
    double cf2;
};

struct MPCFollowRolloutSafetyParams {
    bool enable_lethal_obstacle_check;
    double lethal_obstacle_threshold;
    int fddp_lethal_consecutive_threshold;
};

struct MPCFollowAncillaryFeedbackParams {
    bool enable;
    double velocity_error_gain;
    double command_error_gain;
    double velocity_error_reanchor_threshold;
    double velocity_command_margin;
    double velocity_command_rate_margin;
};

struct MPCFollowParams {
    MPCStartCommandLimits start_command;
    CapabilityProfile normal_profile;
    std::array<CapabilityProfile, 3> capability_profiles;
    MPCFollowTrackingWeights tracking_weights;
    MPCFollowCommandWeights command_weights;
    MPCCommandDynamicsWeights command_dynamics_weights;
    MPCTraversalTargetWeights traversal_target_weights;
    MPCFollowEnvironmentWeights environment_weights;
    MPCFollowRolloutSafetyParams rollout_safety;
    MPCFollowAncillaryFeedbackParams ancillary_feedback;
    MPCFollowProgressParams progress;
    MPCFollowTerminalWeights terminal_weights;
    int max_iters;
};

struct MPCStopCommandWeights {
    double r_v;
    double r_omega;
    double r_dv;
    double r_domega;
    double r_jerk_v;
    double r_jerk_omega;
};

struct MPCStopEnvironmentWeights {
    double obstacle;
};

struct MPCStopTerminalWeights {
    double obstacle_terminal;
};

struct MPCStopParams {
    CapabilityProfile profile;
    MPCStartCommandLimits start_command;
    MPCStopCommandWeights command_weights;
    MPCCommandDynamicsWeights command_dynamics_weights;
    MPCStopEnvironmentWeights environment_weights;
    MPCStopTerminalWeights terminal_weights;
    int max_iters;
};

struct MPCHoldGoalWeights {
    double q_goal_xy;
    double q_goal_theta;
    double goal_deadzone;
};

struct MPCHoldCommandWeights {
    double r_v;
    double r_omega;
    double r_dv;
    double r_domega;
    double r_jerk_v;
    double r_jerk_omega;
};

struct MPCHoldEnvironmentWeights {
    double obstacle;
};

struct MPCHoldTerminalWeights {
    double q_goal_xy_terminal;
    double obstacle_terminal;
};

struct MPCHoldParams {
    CapabilityProfile profile;
    MPCStartCommandLimits start_command;
    MPCHoldGoalWeights goal_weights;
    MPCHoldCommandWeights command_weights;
    MPCCommandDynamicsWeights command_dynamics_weights;
    MPCHoldEnvironmentWeights environment_weights;
    MPCHoldTerminalWeights terminal_weights;
    int max_iters;
};

struct RolloutLethalObstacleInfo {
    int state_index = -1;
    Eigen::Vector2d position_map = Eigen::Vector2d::Zero();
    double sampled_cost = 0.0;
};

struct MPCPrediction {
    std::vector<Eigen::Vector2d> path_map;
    std::vector<double> headings;
    std::vector<double> v_pred;
    std::vector<double> w_pred;
    std::vector<double> path_progress_pred;
    std::vector<double> path_speed_pred;
};

enum class MPCSolverMode : uint8_t {
    NONE = 0,
    FOLLOW = 1,
    STOP = 2,
    HOLD = 3,
};

enum class ObserverUpdateEvent : uint8_t {
    NONE = 0,
    RESET = 1,
    INITIALIZED = 2,
    CORRECTED = 3,
    MODEL_STRESS = 4,
};

enum class ObserverResetReason : uint8_t {
    NONE = 0,
    EXPLICIT_REQUEST = 1,
    NONFINITE_INPUT = 2,
    MODEL_DEGENERATE = 3,
    NONFINITE_PREDICTION = 4,
    VELOCITY_INNOVATION = 5,
    STATE_SEQUENCE_GAP = 6,
    CONTROL_UPDATE_GAP = 7,
    COMMAND_RESYNCHRONIZED = 8,
    CONTROL_UNAVAILABLE = 9,
    CONTROL_OUTPUT_INVALID = 10,
};

struct ObserverDiagnostics {
    ObserverUpdateEvent event = ObserverUpdateEvent::NONE;
    ObserverResetReason last_reset_reason = ObserverResetReason::NONE;
    bool initialized = false;
    bool validated = false;
    bool prediction_available = false;
    bool auxiliary_prediction_available = false;
    bool velocity_correction_clipped = false;

    uint64_t state_sequence = 0;
    uint64_t reset_count = 0;
    uint64_t active_run_length = 0;
    uint64_t revalidation_latency_updates = 0;

    double hidden_state_estimate = 0.0;
    double predicted_hidden_state = 0.0;
    double predicted_velocity = 0.0;
    double velocity_innovation = 0.0;
    double predicted_angular_velocity = 0.0;
    double angular_velocity_innovation = 0.0;
    double predicted_leg_psi = 0.0;
    double leg_psi_innovation = 0.0;
    double input_command_velocity = 0.0;
    double input_command_angular_velocity = 0.0;
};

struct MPCDiagnostics {
    MPCSolverMode solver_mode = MPCSolverMode::NONE;
    bool solve_succeeded = false;
    std::string solve_error;

    bool ancillary_enabled = false;
    bool ancillary_active = false;
    bool nominal_reanchored = false;
    bool first_command_tube_feasible = false;

    Eigen::Vector2d measured_velocity = Eigen::Vector2d::Zero();
    Eigen::Vector2d previous_command = Eigen::Vector2d::Zero();
    Eigen::Vector2d nominal_command = Eigen::Vector2d::Zero();
    Eigen::Vector2d nominal_command_rate = Eigen::Vector2d::Zero();
    Eigen::Vector2d applied_command_rate = Eigen::Vector2d::Zero();

    std::vector<double> reference_path_progress;
    std::vector<double> reference_path_speed;
    std::vector<double> trajectory_nominal_velocity;
    std::vector<double> trajectory_nominal_angular_velocity;
    std::vector<double> reference_velocity;
    std::vector<double> reference_angular_velocity;

    MPCPrediction nominal_prediction;
    MPCPrediction applied_prediction;
};

struct MPCParams {
    MPCFollowParams follow;
    MPCStopParams stop;
    MPCHoldParams hold;

    LPVKinematicModelParams kinematic_model;
};

struct GridInfo {
    double origin_x, origin_y, inv_resolution;
    int width, height;
};

template<typename MapT>
inline GridInfo make_grid_info(const MapT& map) {
    return GridInfo {map.origin_x, map.origin_y, 1.0 / map.resolution, map.width, map.height};
}

struct CostMapGridView {
    explicit CostMapGridView(const CostMap& map): map_(map) {}

    double value_at_clamped(int row, int col) const {
        row = std::max(0, std::min(row, map_.height - 1));
        col = std::max(0, std::min(col, map_.width - 1));
        return static_cast<double>(map_.data[static_cast<size_t>(row * map_.width + col)]);
    }

    const CostMap& map_;
};

struct CostSample {
    double value;
    double dx, dy;
};

using StateVec = Eigen::Matrix<double, MPC_NX, 1>;
using ControlVec = Eigen::Matrix<double, MPC_NU, 1>;
using MatXX = Eigen::Matrix<double, MPC_NX, MPC_NX>;
using MatXU = Eigen::Matrix<double, MPC_NX, MPC_NU>;

struct MPCControlBounds {
    ControlVec lower;
    ControlVec upper;
    Eigen::Matrix<double, MPC_NU, MPC_NX> lower_state_jacobian =
        Eigen::Matrix<double, MPC_NU, MPC_NX>::Zero();
    Eigen::Matrix<double, MPC_NU, MPC_NX> upper_state_jacobian =
        Eigen::Matrix<double, MPC_NU, MPC_NX>::Zero();
};

} // namespace nav_executor
