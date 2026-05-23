#pragma once

#include <array>
#include <expected>
#include <optional>
#include <string>
#include <Eigen/Dense>
#include <path_follower/chassis_defs.hpp>
#include <path_follower/fddp_solver.hpp>
#include <path_follower/nav_map.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

// ═══════════════════════════════════════════════════════════════
//  MPC 编译期常量
// ═══════════════════════════════════════════════════════════════

constexpr int MPC_HORIZON = 60;
constexpr double MPC_DT = 0.05;
constexpr int MPC_NX = 9; // [x, y, theta, x_h, v_act, w_act, dv, dw, path_u]
constexpr int MPC_NU = 2; // [v_cmd, omega_cmd]
constexpr int PWR_N = 12; // P = c0 + c1*v*a + c2*w*alpha + c3*a^2 + c4*alpha^2 + c5*|v| + c6*|w| + c7*v^2 + c8*w^2 + c9*|a| + c10*|alpha| + c11*|v*w|

constexpr double SOLVER_TOL_GRAD = 1e-6;
constexpr double SOLVER_TOL_COST = 1e-8;

// ═══════════════════════════════════════════════════════════════
//  State / control vector indexing
// ═══════════════════════════════════════════════════════════════

// State indices: x=0, y=1, theta=2, x_h=3, v_act=4, w_act=5, dv=6, dw=7, path_u=8
namespace ix {
    enum { X = 0, Y, THETA, XH, V, W, DV, DW, PATH_U };
}

// ═══════════════════════════════════════════════════════════════
//  Parameter structs
// ═══════════════════════════════════════════════════════════════

struct MPCCommandBounds {
    double vel_max;
    double vel_min;
    double omega_max;
    double omega_min;
};

struct MPCStartCommandLimits {
    double vel_cmd_act_gap_max;
    double omega_cmd_act_gap_max;
};

struct MPCMotionConstraints {
    double acc_max;
    double alpha_max;
    double a_lat_max;
};

struct MPCMotionConstraintWeights {
    double acc_limit;
    double alpha_limit;
    double lat_acc;
};

struct MPCFollowModeProfile {
    MPCCommandBounds command_bounds;
    MPCMotionConstraints motion_constraints;
};

struct MPCFollowModeProfiles {
    MPCFollowModeProfile normal;
    struct {
        MPCFollowModeProfile jump;
        MPCFollowModeProfile short_leg;
        MPCFollowModeProfile long_leg;
    } up;
    struct {
        MPCFollowModeProfile jump;
        MPCFollowModeProfile short_leg;
    } down;
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
    std::array<double, 4> step_speed_levels;
    double step_vel_deadzone;
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

struct MPPISamplingStd {
    double velocity;
    double omega;
};

struct MPPINoiseSmoothing {
    int window;
    int passes;
};

struct MPCFollowMPPIParams {
    bool enable;
    int num_threads;
    int batch_size;
    int iteration_count;
    double temperature;
    double gamma;
    MPPISamplingStd sampling_std;
    MPPINoiseSmoothing noise_smoothing;
    bool include_nominal_trajectory;
    bool fallback_to_best_sample;
};

struct MPCFollowRolloutSafetyParams {
    bool enable_lethal_obstacle_check;
    double lethal_obstacle_threshold;
    int fddp_lethal_consecutive_threshold;
};

struct MPCFollowParams {
    MPCStartCommandLimits start_command;
    MPCFollowModeProfiles mode_profiles;
    MPCFollowTrackingWeights tracking_weights;
    MPCFollowCommandWeights command_weights;
    MPCMotionConstraintWeights motion_constraint_weights;
    MPCFollowTerrainLimits terrain_limits;
    MPCFollowTerrainWeights terrain_weights;
    MPCFollowEnvironmentWeights environment_weights;
    MPCFollowTerminalWeights terminal_weights;
    MPCFollowProjection projection;
    MPCFollowMPPIParams mppi;
    MPCFollowRolloutSafetyParams rollout_safety;
    int max_iters;
};

struct MPCStopCommandWeights {
    double q_v;
    double q_omega;
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
    ChassisMode mode = ChassisMode::NORMAL;
    double target_velocity = 0.0;
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

/// MPC 预测轨迹
struct MPCPrediction {
    std::vector<Eigen::Vector2d> path_map;
    std::vector<double> headings;
    std::vector<double> v_pred;
    std::vector<double> w_pred;

    /// MPPI 采样 rollout 调试轨迹（仅 debug 模式填充）
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

// ═══════════════════════════════════════════════════════════════
//  代价地图零拷贝视图（双线性采样 + 梯度）
// ═══════════════════════════════════════════════════════════════

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

/// 双线性采样代价地图值 + 梯度（∂cost/∂x, ∂cost/∂y）
struct CostSample {
    double value;
    double dx, dy; // ∂value/∂x_map, ∂value/∂y_map
};

CostSample eval_cost_bilinear(const CostMapGridView& grid, const GridInfo& info, double x_map, double y_map);

/// 双线性采样方向场值 + 雅可比（2×2: ∂dir/∂(x,y)）
struct DirSample {
    Eigen::Vector2d value;
    Eigen::Matrix2d J; // ∂value/∂(x_map, y_map)
};

DirSample eval_dir_bilinear(const DirectionMapGridView& grid, const GridInfo& info, double x_map, double y_map);

/// 方向场双线性插值（仅值，无雅可比）—— 给 cost-only 路径用
Eigen::Vector2d eval_dir_bilinear_value_only(const DirectionMapGridView& grid, const GridInfo& info, double x_map, double y_map);

// ═══════════════════════════════════════════════════════════════
//  共享动力学模型（supply analytic Jacobians for FDDP）
// ═══════════════════════════════════════════════════════════════

using StateVec = Eigen::Matrix<double, MPC_NX, 1>;
using ControlVec = Eigen::Matrix<double, MPC_NU, 1>;
using MatXX = Eigen::Matrix<double, MPC_NX, MPC_NX>;
using MatXU = Eigen::Matrix<double, MPC_NX, MPC_NU>;

template<int Horizon>
class FollowProblemT {
public:
    FollowProblemT(
        const SplinePath& spline,
        const MPCParams& params,
        const std::vector<CostMapGridView>& per_step_cost_grids,
        const GridInfo& cost_info,
        const CostMapGridView& masked_global_grid,
        double prediction_dt,
        double schedule_rho,
        const DirectionMapGridView& dir_grid,
        const GridInfo& dir_info,
        double remaining_energy,
        double rfr_pwr_limit,
        std::optional<ActiveStepMode> active_step_mode,
        double current_path_u
    );

    StateVec dynamics(int k, const StateVec& x, const ControlVec& u) const;
    void dynamics_jacobians(int k, const StateVec& x, const ControlVec& u, MatXX& fx, MatXU& fu) const;

    double running_cost(int k, const StateVec& x, const ControlVec& u) const;
    double running_cost_value_only(int k, const StateVec& x, const ControlVec& u, double* cached_cost_value = nullptr) const;
    void running_cost_derivatives(
        int k,
        const StateVec& x,
        const ControlVec& u,
        StateVec& lx,
        ControlVec& lu,
        MatXX& lxx,
        Eigen::Matrix<double, MPC_NU, MPC_NX>& lux,
        Eigen::Matrix<double, MPC_NU, MPC_NU>& luu
    ) const;

    double terminal_cost(const StateVec& x) const;
    void terminal_cost_derivatives(const StateVec& x, StateVec& lfx, MatXX& lfxx) const;

    ControlVec u_lower() const;
    ControlVec u_upper() const;

    [[nodiscard]] std::optional<RolloutLethalObstacleInfo> detect_lethal_obstacle(int state_index, const StateVec& x, double* out_cost_value = nullptr) const;
    [[nodiscard]] const MPCParams& params() const;
    [[nodiscard]] FollowProblemT<Horizon> with_reference_path(const SplinePath& spline) const;

private:
    const CostMapGridView& cost_grid_for_step(int k) const;

    Eigen::Vector2d goal_xy_;
    SplinePath spline_;
    const MPCParams& p_;
    const std::vector<CostMapGridView>& step_cost_grids_;
    GridInfo cost_info_;
    const CostMapGridView& masked_global_grid_;
    double prediction_dt_;
    LPVDiscreteModel model_ {};
    const DirectionMapGridView& dir_grid_;
    GridInfo dir_info_;
    double remaining_energy_;
    double rfr_pwr_limit_;
    std::optional<ActiveStepMode> active_step_mode_;
    double current_path_u_;
};

using FollowProblem = FollowProblemT<MPC_HORIZON>;

// ═══════════════════════════════════════════════════════════════
//  FDDP Problem 类型 — Stop
// ═══════════════════════════════════════════════════════════════

class StopProblem {
public:
    StopProblem(
        const MPCParams& params,
        const CostMapGridView& cost_grid,
        const GridInfo& cost_info,
        double schedule_rho,
        double remaining_energy,
        double rfr_pwr_limit
    );

    StateVec dynamics(int k, const StateVec& x, const ControlVec& u) const;
    void dynamics_jacobians(int k, const StateVec& x, const ControlVec& u, MatXX& fx, MatXU& fu) const;

    double running_cost(int k, const StateVec& x, const ControlVec& u) const;
    void running_cost_derivatives(
        int k,
        const StateVec& x,
        const ControlVec& u,
        StateVec& lx,
        ControlVec& lu,
        MatXX& lxx,
        Eigen::Matrix<double, MPC_NU, MPC_NX>& lux,
        Eigen::Matrix<double, MPC_NU, MPC_NU>& luu
    ) const;

    double terminal_cost(const StateVec& x) const;
    void terminal_cost_derivatives(const StateVec& x, StateVec& lfx, MatXX& lfxx) const;

    ControlVec u_lower() const;
    ControlVec u_upper() const;

private:
    const MPCParams& p_;
    const CostMapGridView& cost_grid_;
    GridInfo cost_info_;
    LPVDiscreteModel model_ {};
    double remaining_energy_;
    double rfr_pwr_limit_;
};

// ═══════════════════════════════════════════════════════════════
//  FDDP Problem 类型 — Hold
// ═══════════════════════════════════════════════════════════════

class HoldProblem {
public:
    HoldProblem(
        const Eigen::Vector2d& goal_map,
        const MPCParams& params,
        const CostMapGridView& cost_grid,
        const GridInfo& cost_info,
        double schedule_rho,
        double remaining_energy,
        double rfr_pwr_limit
    );

    StateVec dynamics(int k, const StateVec& x, const ControlVec& u) const;
    void dynamics_jacobians(int k, const StateVec& x, const ControlVec& u, MatXX& fx, MatXU& fu) const;

    double running_cost(int k, const StateVec& x, const ControlVec& u) const;
    void running_cost_derivatives(
        int k,
        const StateVec& x,
        const ControlVec& u,
        StateVec& lx,
        ControlVec& lu,
        MatXX& lxx,
        Eigen::Matrix<double, MPC_NU, MPC_NX>& lux,
        Eigen::Matrix<double, MPC_NU, MPC_NU>& luu
    ) const;

    double terminal_cost(const StateVec& x) const;
    void terminal_cost_derivatives(const StateVec& x, StateVec& lfx, MatXX& lfxx) const;

    ControlVec u_lower() const;
    ControlVec u_upper() const;

private:
    Eigen::Vector2d goal_;
    const MPCParams& p_;
    const CostMapGridView& cost_grid_;
    GridInfo cost_info_;
    LPVDiscreteModel model_ {};
    double remaining_energy_;
    double rfr_pwr_limit_;
};

// ═══════════════════════════════════════════════════════════════
} // namespace path_follower

// ── FDDP Dims specializations ──
namespace fddp {
template<int Horizon>
struct Dims<path_follower::FollowProblemT<Horizon>> {
    static constexpr int NX = path_follower::MPC_NX;
    static constexpr int NU = path_follower::MPC_NU;
    static constexpr int N = Horizon;
};
template<>
struct Dims<path_follower::StopProblem> {
    static constexpr int NX = path_follower::MPC_NX;
    static constexpr int NU = path_follower::MPC_NU;
    static constexpr int N = path_follower::MPC_HORIZON;
};
template<>
struct Dims<path_follower::HoldProblem> {
    static constexpr int NX = path_follower::MPC_NX;
    static constexpr int NU = path_follower::MPC_NU;
    static constexpr int N = path_follower::MPC_HORIZON;
};
}

namespace path_follower {

// ═══════════════════════════════════════════════════════════════
//  MPCSolver — 对外接口（与原有 API 兼容）
// ═══════════════════════════════════════════════════════════════

class MPCSolver {
public:
    enum class FollowSolveStatus : uint8_t {
        FOLLOW = 0,
        STOP_AND_WAIT_REPLAN = 1,
    };

    struct FollowSolveResult {
        Eigen::Vector2d command = Eigen::Vector2d::Zero();
        MPCPrediction prediction;
        FollowSolveStatus status = FollowSolveStatus::FOLLOW;
        std::optional<RolloutLethalObstacleInfo> lethal_obstacle;
    };

    explicit MPCSolver(const MPCParams& params);

    void set_last_cmd(const Eigen::Vector2d& cmd);
    void reset_warm_start();

    void update_observer(const ChassisMotionState& chassis_state);
    void reset_observer();
    void set_energy_state(double remaining_energy, double rfr_pwr_limit);

    [[nodiscard]] double hidden_state_estimate() const {
        return x_h_hat_;
    }

    std::expected<FollowSolveResult, std::string> solve_follow(
        const SplinePath& global_path,
        const Eigen::Vector3d& chassis_pose_map,
        const ChassisMotionState& chassis_state,
        const CostMap& cost_map,
        const CostMap& masked_global_map,
        const std::vector<const CostMap*>& per_step_cost_maps,
        double prediction_dt,
        const DirectionMap& direction_map,
        std::optional<ActiveStepMode> active_step_mode
    );

    std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> solve_stop(
        const Eigen::Vector3d& chassis_pose_map,
        const ChassisMotionState& chassis_state,
        const CostMap& cost_map
    );

    std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> solve_hold(
        const Eigen::Vector2d& goal_map,
        const Eigen::Vector3d& chassis_pose_map,
        const ChassisMotionState& chassis_state,
        const CostMap& cost_map
    );

    [[nodiscard]] const MPCParams& params() const {
        return params_;
    }

private:
    MPCParams params_;
    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero();
    double last_u_ = 0.0;

    fddp::Solver<FollowProblem> follow_solver_;
    fddp::Solver<StopProblem> stop_solver_;
    fddp::Solver<HoldProblem> hold_solver_;
    bool follow_warm_ = false;
    bool stop_warm_ = false;
    bool hold_warm_ = false;

    // 复用每步代价图视图，避免 solve_follow 中反复分配
    std::vector<CostMapGridView> step_cost_grids_cache_;

    // 路径变更检测
    std::optional<SplinePath> prev_ref_control_points_;

    // ── Hidden-state Luenberger observer ──
    double x_h_hat_ = 0.0;
    double prev_v_act_ = 0.0;
    double prev_w_act_ = 0.0;
    double prev_schedule_rho_ = 0.0;
    bool observer_initialized_ = false;

    // ── Energy state ──
    double remaining_energy_ = 200.0;
    double rfr_pwr_limit_ = 90.0;

    // ── FDDP rollout 连续致命障碍物帧计数器 ──
    int fddp_lethal_consecutive_count_ = 0;

    // ── 辅助 ──
    StateVec make_initial_state(
        const Eigen::Vector3d& pose,
        const ChassisMotionState& chassis_state,
        const Eigen::Vector2d& cmd_clamped,
        double path_u
    ) const;
};

}
