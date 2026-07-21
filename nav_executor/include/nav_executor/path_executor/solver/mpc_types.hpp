#pragma once

#include <array>
#include <vector>
#include <Eigen/Dense>
#include <nav_executor/common/chassis_defs.hpp>
#include <nav_executor/common/minco_trajectory.hpp>
#include <nav_executor/path_planner/nav_map.hpp>

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
        PHASE_TIME,
        PHASE_RATE,
    };
}

namespace iu {
    // 前两项是下位机速度指令的变化率，不是底盘实测物理加速度。
    enum { V_CMD_RATE = 0, W_CMD_RATE, PHASE_RATE_CMD };
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
    double tangent_blend_speed_scale;
};

struct MPCFollowPhaseParams {
    double rate_min;
    double rate_max;
    double nominal_rate;
    double progress_reward;
    double rate_tracking_weight;
    double rate_smoothness_weight;
    double overshoot_weight;
};

struct MPCFollowTerminalWeights {
    double position;
    double heading;
    double velocity;
    double angular_velocity;
    double remaining_phase;
    double overshoot;
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
    double angular_velocity;
    double velocity_smoothness;
    double angular_velocity_smoothness;
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

    double xh0_bias;
    double xh0_psi;
    double xh0_v;
    double psi_bias;
    double psi_gain;
    double psi_v;
    double obs_lv;
    double obs_lpsi;
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
    MPCFollowPhaseParams phase;
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

// 路径台阶约束。仅供 MPC 按预测相位对应的 tau 查询，不携带底盘模式或 FSM 语义。
//
// 锚点语义（沿路径 u 从小到大）：
//   commit_u ≤ step_enter_u ≤ exit_u
//   - commit_u：上位机视角的台阶起点（物理边缘上游回退 run_up）。速度窗/方向对齐等
//     “助跑期建立约束”自 commit_u 起施加，实现起跳前的提前达速与对齐。
//   - step_enter_u：物理台阶边缘（真实起跳点）。这是“助跑是否还来得及”的物理截止点，
//     可达包络/入口速度地板以此为参考终点，可行性因子 f 也以此判定。
struct StepTraversalConstraint {
    TraversalVelocityWindow velocity_window;
    double commit_u = 0.0;
    double step_enter_u = 0.0;
    double exit_u = 1.0;
    double gate_start_u = 0.0;
    double gate_end_u = 1.0;
    Eigen::Vector2d dir_map = Eigen::Vector2d::Zero();

    bool operator==(const StepTraversalConstraint&) const = default;
};

// 按路径 u 查询的不可变台阶约束表。规划期负责保证 gate 区间不重叠。
class StepConstraintSchedule {
public:
    explicit StepConstraintSchedule(std::vector<StepTraversalConstraint> constraints)
        : constraints_(std::move(constraints)) {}

    [[nodiscard]] const StepTraversalConstraint* constraint_at(const double path_u) const {
        for (const auto& constraint : constraints_) {
            if (path_u < constraint.gate_start_u) break;
            if (path_u <= constraint.gate_end_u) return &constraint;
        }
        return nullptr;
    }

private:
    std::vector<StepTraversalConstraint> constraints_;
};

// 底盘台阶命令。仅供执行器决定底盘模式与能力档，不参与 MPC 台阶代价。
struct StepChassisCommand {
    uint8_t mode = chassis_mode::NORMAL;
    CapabilityLevel capability = CapabilityLevel::LOW;

    bool operator==(const StepChassisCommand&) const = default;
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
    std::vector<double> phase_time_pred;
    std::vector<double> phase_rate_pred;
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
