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
    double q_y;
    double q_theta;
    double q_u;
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

// Anytime MHA* 局部搜索播种器参数（在 FDDP 前做全局较优粗搜，跳出非凸局部最优）
struct FollowSearchParams {
    bool enable;

    // ── 离散化 ──
    double dt;                  // 搜索步长 (s)，粗于 MPC_DT
    int horizon_steps;          // 搜索时域步数上限（dt × steps 覆盖 MPC 时域）
    int theta_bins;             // 航向离散档数
    double v_bin_size;          // 速度离散分辨率 (m/s)

    // ── 运动基元（相对 command_bounds 的比例，v 含负值支持后退）──
    std::vector<double> v_primitive_fracs;
    std::vector<double> omega_primitive_fracs;
    double tau_v;               // 速度一阶滞后时间常数 (s)；<=0 时启动期由 LPV 模型自动推导

    // ── Anytime 预算 ──
    double budget_ms;
    int max_expansions;

    // ── 可行性 ──
    double collision_threshold; // 代价 >= 该值视为不可通行
    double goal_tolerance;      // 到达前瞻目标的距离阈值 (m)
    double lookahead_distance;  // 沿样条设定前瞻目标点的弧长 (m)

    // ── MHA* 权重 ──
    double w_anchor;            // w1：不可采纳启发式膨胀系数
    double w_inadmissible;      // w2：anchor 有界次优系数
    double spline_bias;         // H1：偏向贴近全局样条的强度

    // ── 边代价权重（与 FDDP running cost 主项同构）──
    double w_time;              // 每步时间/进度基代价
    double w_obstacle;          // 避障（↔ environment_weights.obstacle）
    double w_lateral;           // Frenet 横向误差（↔ tracking_weights.q_y）
    double w_step_align;        // 台阶方向对齐（↔ terrain_weights.direction）
    double w_step_reach;        // 台阶入口可达速度（↔ terrain_weights.step_reachability_*）
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
    FollowSearchParams search;
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

struct ActiveStepMode {
    uint8_t mode = chassis_mode::NORMAL;
    CapabilityLevel capability = CapabilityLevel::LOW;
    double speed_min = 0.0;
    double speed_max = 0.0;
    std::optional<double> step_entry_u = std::nullopt;
    double prepare_u = 0.0;
    double active_u = 0.0;
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

    // 调试：本周期 MHA* 搜索得到的粗路径（map 坐标）
    std::vector<Eigen::Vector2d> search_path;
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

struct DirectionMapGridView {
    explicit DirectionMapGridView(const DirectionMap& map): map_(map) {}

    Eigen::Vector2d value_at_clamped(int row, int col) const {
        row = std::max(0, std::min(row, map_.height - 1));
        col = std::max(0, std::min(col, map_.width - 1));
        return map_.data[static_cast<size_t>(row * map_.width + col)];
    }

    const DirectionMap& map_;
};

struct CostSample {
    double value;
    double dx, dy;
};

struct DirSample {
    Eigen::Vector2d value;
    Eigen::Matrix2d J;
};

using StateVec = Eigen::Matrix<double, MPC_NX, 1>;
using ControlVec = Eigen::Matrix<double, MPC_NU, 1>;
using MatXX = Eigen::Matrix<double, MPC_NX, MPC_NX>;
using MatXU = Eigen::Matrix<double, MPC_NX, MPC_NU>;

} // namespace nav_executor
