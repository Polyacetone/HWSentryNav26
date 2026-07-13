#pragma once

#include <array>
#include <expected>
#include <optional>
#include <string>
#include <Eigen/Dense>
#include <nav_executor/common/chassis_defs.hpp>
#include <nav_executor/path_planner/nav_map.hpp>

namespace nav_executor {

constexpr int MPC_HORIZON = 60;
constexpr double MPC_DT = 0.05;
constexpr int MPC_NX = 10;
constexpr int MPC_NU = 2;
constexpr int PWR_N = 12;

constexpr double SOLVER_TOL_GRAD = 1e-6;
constexpr double SOLVER_TOL_COST = 1e-8;

namespace ix {
    enum { X = 0, Y, THETA, XH, V, W, DV, DW, PATH_U, ENERGY };
}

struct MPCStartCommandLimits {
    double vel_cmd_act_gap_max;
    double omega_cmd_act_gap_max;
};

struct MPCMotionConstraintWeights {
    double acc_limit;
    double alpha_limit;
    double lat_acc;
};

struct MPCFollowTrackingWeights {
    double q_y;            // 横向误差管廊外权重（tube 外二次惩罚）
    double q_theta;        // 航向误差权重
    double q_u;            // 逐步进度势权重（基于投影弧长的 cost-to-go 梯度）
    double y_tube;         // 横向误差管廊半宽 (m)：|ey| < y_tube 内不惩罚，允许横向腾挪
    double q_term_prog;    // 终端进度势权重：terminal_cost += q_term_prog * s_remaining(u*)
    double q_term_lateral; // 终端横向势权重：terminal_cost += q_term_lateral * |ey|，补偿投影势的 cross-track 平坦性
};

struct MPCFollowCommandWeights {
    double r_v;
    double r_omega;
    double r_dv;
    double r_domega;
};

struct MPCFollowTerrainLimits {
    double step_reachability_guide_acc;
};

struct MPCFollowTerrainWeights {
    double step_vel_weight;
    double step_reachability_lo;
    double step_reachability_hi;
    double direction;
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

struct PowerModelParams {
    std::array<double, PWR_N> coeffs {};
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

struct MPCFollowTerminalWeights {
    double q_v_final;
    double a_brake;
    double slow_down_target_vel;
};

struct MPCFollowProjection {
    int proj_num_samples;
    double proj_search_window;
    double local_search_lazy_distance;
};

struct GlobalSearchSamplingStd {
    double velocity;
    double omega;
};

struct GlobalSearchNoiseSmoothing {
    int window;
    int passes;
};

struct GlobalSearchParams {
    bool enable;
    int num_threads;
    int batch_size;
    int iteration_count;
    double elite_fraction;
    GlobalSearchSamplingStd sampling_std;
    GlobalSearchNoiseSmoothing noise_smoothing;
    bool include_nominal_trajectory;
    int candidate_count = 2;
    int min_period_ms = 200;
    int max_seed_age_ticks = 8;
    double improvement_margin = 0.05;
    int hysteresis_count = 2;
    int refinement_iterations = 8;
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
    MPCMotionConstraintWeights motion_constraint_weights;
    MPCFollowTerrainLimits terrain_limits;
    MPCFollowTerrainWeights terrain_weights;
    MPCFollowEnvironmentWeights environment_weights;
    MPCFollowTerminalWeights terminal_weights;
    MPCFollowProjection projection;
    GlobalSearchParams global_search;
    MPCFollowRolloutSafetyParams rollout_safety;
    int max_iters;
};

struct MPCStopCommandWeights {
    double r_v;
    double r_omega;
    double r_dv;
    double r_domega;
};

struct MPCStopEnvironmentWeights {
    double obstacle;
};

struct MPCStopTerminalWeights {
    double obstacle_terminal;
};

struct MPCStopParams {
    MPCCommandBounds command_bounds;
    MPCStartCommandLimits start_command;
    MPCMotionConstraints motion_constraints;
    MPCStopCommandWeights command_weights;
    MPCMotionConstraintWeights motion_constraint_weights;
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
};

struct MPCHoldEnvironmentWeights {
    double obstacle;
};

struct MPCHoldTerminalWeights {
    double q_goal_xy_terminal;
    double obstacle_terminal;
};

struct MPCHoldParams {
    MPCCommandBounds command_bounds;
    MPCStartCommandLimits start_command;
    MPCMotionConstraints motion_constraints;
    MPCHoldGoalWeights goal_weights;
    MPCHoldCommandWeights command_weights;
    MPCMotionConstraintWeights motion_constraint_weights;
    MPCHoldEnvironmentWeights environment_weights;
    MPCHoldTerminalWeights terminal_weights;
    int max_iters;
};

struct EnergyParams {
    bool enable;
    double threshold;
    double weight;
};

// 运行时台阶约束参数。所有 u 锚点语义（prepare ≤ active ≤ commit ≤ 物理边缘 ≤ exit ≤ release）
// 在规划期由 step_annotator 计算并冻结。
//
// commit_u 是「上位机视角的台阶起点」：引入助跑提前量后，速度可达包络、速度窗、方向对齐
// 全部提前到 commit_u 施加，而非物理台阶边缘。物理边缘（真实起跳点，上报底盘用）由
// StepPlanSegment.step_enter_u 单独持有，二者不混用。
struct ActiveStepMode {
    uint8_t mode = chassis_mode::NORMAL;
    CapabilityLevel capability = CapabilityLevel::LOW;
    double speed_min = 0.0;
    double speed_max = 0.0;
    double prepare_u = 0.0;    // 能力档 profile blend 起点
    double active_u = 0.0;     // 底盘台阶模式激活（抬腿指令）起点
    double commit_u = 0.0;     // 约束锚点：可达包络靶点 + 约束窗内边界起点
    double exit_u = 1.0;       // 约束窗内边界终点（物理台阶出口 u）
    double gate_start_u = 0.0; // 约束窗软起点（commit_u 上游 transition 处，门控 0→1）
    double gate_end_u = 1.0;   // 约束窗软终点（exit_u 下游 transition 处，门控 1→0）
    Eigen::Vector2d dir_map = Eigen::Vector2d::Zero(); // 归一化台阶穿越方向（航向对齐目标）
    double release_u = 1.0;

    bool operator==(const ActiveStepMode&) const = default;
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

    std::vector<std::vector<Eigen::Vector2d>> rollout_paths;
};

struct MPCParams {
    MPCFollowParams follow;
    MPCStopParams stop;
    MPCHoldParams hold;

    EnergyParams energy;
    LPVKinematicModelParams kinematic_model;
    PowerModelParams power_model;
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

} // namespace nav_executor
