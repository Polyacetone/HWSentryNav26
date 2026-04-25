#pragma once

#include <array>
#include <expected>
#include <string>
#include <Eigen/Dense>
#include <path_follower/fddp_solver.hpp>
#include <path_follower/nav_map.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

// ═══════════════════════════════════════════════════════════════
//  MPC 编译期常量
// ═══════════════════════════════════════════════════════════════

constexpr int MPC_HORIZON = 30;
constexpr double MPC_DT = 0.05;
constexpr int MPC_NX = 9; // [x, y, theta, x_h, v_act, w_act, dv, dw, path_u]
constexpr int MPC_NU = 2; // [v_cmd, omega_cmd]

// ── ZOH-discretized model constants (auto-generated) ──
constexpr double SGN_EPS   = 0.05;
constexpr double CF1       = 0.117776965182103;
constexpr double CF2       = -0.07934663936437283;
constexpr double CF3       = -0.035922362597109334;
constexpr double XH0       = -0.1741781626557545;
// v-subsystem (2×2 ZOH via matrix exponential)
constexpr double A00       = 0.7843676355861557;
constexpr double A01       = -0.09648336780318062;
constexpr double A03       = 0.0037319689718094814;
constexpr double A10       = 0.3987787787629378;
constexpr double A11       = 1.1336902129030175;
constexpr double A13       = 0.0351881685023933;
// nonlinear gains (ZOH): Gnl = G·[0;1]
constexpr double GNL_XH    = -0.002444300627739186;
constexpr double GNL_V     = 0.05342869358100494;
// ω-channel (1st-order ZOH exact): pole = exp(-dt/τ) = 0.553737 (positive!)
constexpr double A22       = 0.5537368531002395;
constexpr double A24       = 0.44626314689976054;
constexpr double GAMMA_W   = 0.03775072274445112;
// hidden-state observer gain (target pole = 0.6)
constexpr double OBS_L     = 0.46233060886060057;

// ── Power model coefficients (auto-generated from identification) ──
constexpr int PWR_N = 12;
constexpr double PWR_EPS2 = 0.05 * 0.05;
constexpr double PWR_C[PWR_N] = {
    2.1819886751e-00,  // c0: 1 (bias)
    3.7554288560e+01,  // c1: v·a
    9.0037590352e-01,  // c2: ω·α
    3.7473397433e+00,  // c3: a²
    5.3351521229e-02,  // c4: α²
    1.3085009538e+01,  // c5: |v|
    3.7559074150e+00,  // c6: |ω|
    7.7240176266e+00,  // c7: v²
    2.8778608500e-01,  // c8: ω²
    1.0022176768e+01,  // c9: |a|
    0.0000000000e+00,  // c10: |α|
    4.7233251821e+00  // c11: |v·ω|
};

// 弧长查找表类型
constexpr int ARCLENGTH_TABLE_SIZE = 128;
using ArclengthTable = std::array<double, ARCLENGTH_TABLE_SIZE + 1>;

// ═══════════════════════════════════════════════════════════════
//  State / control vector indexing
// ═══════════════════════════════════════════════════════════════

// State indices: x=0, y=1, theta=2, x_h=3, v_act=4, w_act=5, dv=6, dw=7, path_u=8
namespace ix {
    enum { X = 0, Y, THETA, XH, V, W, DV, DW, PATH_U };
}

// ═══════════════════════════════════════════════════════════════
//  Parameter structs (unchanged public interface)
// ═══════════════════════════════════════════════════════════════

struct MPCFollowLimits {
    double vel_max;
    double vel_min;
    double omega_max;
    double omega_min;
    double start_vel_cmd_act_diff_max;
    double start_omega_cmd_act_diff_max;
    double acc_max;
    double alpha_max;

    double vel_step_up;
    double vel_step_down;
    double vel_step_deadzone;
    double a_lat_max;

    double slow_down_deceleration;
    double slow_down_target_vel;
    int slow_down_num_samples;
};

struct MPCFollowWeights {
    double q_y;
    double q_theta;
    double q_u;
    double q_v_final;
    double r_v;
    double r_omega;
    double r_dv;
    double r_domega;

    double acc_limit;
    double alpha_limit;
    double lat_acc;

    double vel_on_step;
    double obstacle;
    double direction;
};

struct MPCFollowProjection {
    int proj_num_samples;
    double proj_search_window;
    double local_search_lazy_distance;
};

struct MPCStopLimits {
    double vel_max;
    double omega_max;
    double omega_min;

    double start_vel_cmd_act_diff_max;
    double start_omega_cmd_act_diff_max;
    double acc_max;
    double alpha_max;

    double vel_step_up;
    double vel_step_down;
    double vel_step_deadzone;
    double a_lat_max;
};

struct MPCStopWeights {
    double q_v;
    double q_omega;
    double r_dv;
    double r_domega;

    double acc_limit;
    double alpha_limit;
    double lat_acc;
    double vel_on_step;

    double obstacle;
    double obstacle_terminal;
    double direction;
    double step_terminal;
};

struct MPCRecoveryLimits {
    double vel_max;
    double vel_min;
    double omega_max;
    double omega_min;

    double start_vel_cmd_act_diff_max;
    double start_omega_cmd_act_diff_max;
    double acc_max;
    double alpha_max;
    double a_lat_max;
};

struct MPCRecoveryWeights {
    double q_goal_xy;
    double q_goal_theta;
    double r_v;
    double r_omega;
    double r_dv;
    double r_domega;

    double acc_limit;
    double alpha_limit;
    double lat_acc;

    double obstacle;
    double step;

    double q_goal_xy_terminal;
    double obstacle_terminal;
    double step_terminal;
};

struct MPCFixedLimits {
    double vel_max;
    double vel_min;
    double omega_max;
    double omega_min;

    double start_vel_cmd_act_diff_max;
    double start_omega_cmd_act_diff_max;
    double acc_max;
    double alpha_max;
    double a_lat_max;
};

struct MPCFixedWeights {
    double q_goal_xy;
    double q_goal_theta;
    double goal_deadzone;
    double r_v;
    double r_omega;
    double r_dv;
    double r_domega;

    double acc_limit;
    double alpha_limit;
    double lat_acc;

    double obstacle;
    double step;

    double q_goal_xy_terminal;
    double obstacle_terminal;
    double step_terminal;
};

struct EnergyParams {
    bool enable;
    double threshold;
    double weight;
    double softplus_beta;
};

struct MultiHypothesisParams {
    bool enable;
    double lateral_offset;
    double target_ey_penalty;
};

/// MPC 预测轨迹
struct MPCPrediction {
    std::vector<Eigen::Vector2d> path_map;
    std::vector<double> headings;
    std::vector<double> v_pred;
    std::vector<double> w_pred;
};

struct MPCParams {
    MPCFollowLimits follow_limits;
    MPCFollowWeights follow_weights;
    MPCFollowProjection follow_projection;

    MPCStopLimits stop_limits;
    MPCStopWeights stop_weights;

    MPCRecoveryLimits recovery_limits;
    MPCRecoveryWeights recovery_weights;

    MPCFixedLimits fixed_limits;
    MPCFixedWeights fixed_weights;

    EnergyParams energy;
    MultiHypothesisParams mh_params;
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

// ═══════════════════════════════════════════════════════════════
//  共享动力学模型（supply analytic Jacobians for FDDP）
// ═══════════════════════════════════════════════════════════════

using StateVec = Eigen::Matrix<double, MPC_NX, 1>;
using ControlVec = Eigen::Matrix<double, MPC_NU, 1>;
using MatXX = Eigen::Matrix<double, MPC_NX, MPC_NX>;
using MatXU = Eigen::Matrix<double, MPC_NX, MPC_NU>;

/// 通用离散动力学，车辆状态按模型推进，PATH_U 默认保持不变。
StateVec mpc_dynamics(const StateVec& x, const ControlVec& u);

/// 动力学雅可比: fx = ∂f/∂x, fu = ∂f/∂u
void mpc_dynamics_jacobians(const StateVec& x, const ControlVec& u, MatXX& fx, MatXU& fu);

// ═══════════════════════════════════════════════════════════════
//  FDDP Problem 类型 — Follow
// ═══════════════════════════════════════════════════════════════

class FollowProblem {
public:
    FollowProblem(
        const std::vector<Eigen::Vector2d>& ref_control_points,
        const MPCParams& params,
        const std::vector<CostMapGridView>& per_step_cost_grids,
        const GridInfo& cost_info,
        double prediction_dt,
        const DirectionMapGridView& dir_grid,
        const GridInfo& dir_info,
        const ArclengthTable& arclength_table,
        double remaining_energy,
        double rfr_pwr_limit,
        double target_ey = 0.0
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
    const CostMapGridView& cost_grid_for_step(int k) const;

    const std::vector<Eigen::Vector2d>& ref_cps_;
    const MPCParams& p_;
    const std::vector<CostMapGridView>& step_cost_grids_;
    GridInfo cost_info_;
    double prediction_dt_;
    const DirectionMapGridView& dir_grid_;
    GridInfo dir_info_;
    const ArclengthTable& arc_table_;
    double remaining_energy_;
    double rfr_pwr_limit_;
    double target_ey_;
};

// ═══════════════════════════════════════════════════════════════
//  FDDP Problem 类型 — Stop
// ═══════════════════════════════════════════════════════════════

class StopProblem {
public:
    StopProblem(
        const MPCParams& params,
        const CostMapGridView& cost_grid,
        const GridInfo& cost_info,
        const DirectionMapGridView& dir_grid,
        const GridInfo& dir_info,
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
    const DirectionMapGridView& dir_grid_;
    GridInfo dir_info_;
    double remaining_energy_;
    double rfr_pwr_limit_;
};

// ═══════════════════════════════════════════════════════════════
//  FDDP Problem 类型 — Recovery
// ═══════════════════════════════════════════════════════════════

class RecoveryProblem {
public:
    RecoveryProblem(
        const Eigen::Vector2d& goal_map,
        const MPCParams& params,
        const CostMapGridView& cost_grid,
        const GridInfo& cost_info,
        const DirectionMapGridView& dir_grid,
        const GridInfo& dir_info,
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
    const DirectionMapGridView& dir_grid_;
    GridInfo dir_info_;
    double remaining_energy_;
    double rfr_pwr_limit_;
};

// ═══════════════════════════════════════════════════════════════
//  FDDP Problem 类型 — Fixed（固定在目标点）
// ═══════════════════════════════════════════════════════════════

class FixedProblem {
public:
    FixedProblem(
        const Eigen::Vector2d& goal_map,
        const MPCParams& params,
        const CostMapGridView& cost_grid,
        const GridInfo& cost_info,
        const DirectionMapGridView& dir_grid,
        const GridInfo& dir_info,
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
    const DirectionMapGridView& dir_grid_;
    GridInfo dir_info_;
    double remaining_energy_;
    double rfr_pwr_limit_;
};

} // namespace path_follower

// ── FDDP Dims specializations ──
namespace fddp {
template<>
struct Dims<path_follower::FollowProblem> {
    static constexpr int NX = path_follower::MPC_NX;
    static constexpr int NU = path_follower::MPC_NU;
    static constexpr int N = path_follower::MPC_HORIZON;
};
template<>
struct Dims<path_follower::StopProblem> {
    static constexpr int NX = path_follower::MPC_NX;
    static constexpr int NU = path_follower::MPC_NU;
    static constexpr int N = path_follower::MPC_HORIZON;
};
template<>
struct Dims<path_follower::RecoveryProblem> {
    static constexpr int NX = path_follower::MPC_NX;
    static constexpr int NU = path_follower::MPC_NU;
    static constexpr int N = path_follower::MPC_HORIZON;
};
template<>
struct Dims<path_follower::FixedProblem> {
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
    explicit MPCSolver(const MPCParams& params);

    void set_last_cmd(const Eigen::Vector2d& cmd);
    void reset_warm_start();

    void update_observer(double v_act, double w_act);
    void set_energy_state(double remaining_energy, double rfr_pwr_limit);

    [[nodiscard]] double hidden_state_estimate() const {
        return x_h_hat_;
    }

    std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> follow_path(
        const SplineD& global_path,
        const Eigen::Vector3d& chassis_pose_map,
        const Eigen::Vector2d& chassis_status,
        const CostMap& cost_map,
        const std::vector<const CostMap*>& per_step_cost_maps,
        double prediction_dt,
        const DirectionMap& direction_map
    );

    std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> stop(
        const Eigen::Vector3d& chassis_pose_map,
        const Eigen::Vector2d& chassis_status,
        const CostMap& cost_map,
        const DirectionMap& direction_map
    );

    std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> recover_to_point(
        const Eigen::Vector2d& goal_map,
        const Eigen::Vector3d& chassis_pose_map,
        const Eigen::Vector2d& chassis_status,
        const CostMap& cost_map,
        const DirectionMap& direction_map
    );

    std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> hold_at_point(
        const Eigen::Vector2d& goal_map,
        const Eigen::Vector3d& chassis_pose_map,
        const Eigen::Vector2d& chassis_status,
        const CostMap& cost_map,
        const DirectionMap& direction_map
    );

    [[nodiscard]] const MPCParams& params() const {
        return params_;
    }

private:
    MPCParams params_;
    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero();
    double last_u_ = 0.0;

    // 主 FDDP solver（中心假设）
    fddp::Solver<FollowProblem> follow_solver_;
    fddp::Solver<StopProblem> stop_solver_;
    fddp::Solver<RecoveryProblem> recovery_solver_;
    fddp::Solver<FixedProblem> fixed_solver_;
    bool follow_warm_ = false;
    bool stop_warm_ = false;
    bool recovery_warm_ = false;
    bool fixed_warm_ = false;

    // 多假设 solver（左/右偏移，仅用于 follow_path）
    fddp::Solver<FollowProblem> follow_solver_left_;
    fddp::Solver<FollowProblem> follow_solver_right_;

    // 复用每步代价图视图，避免 follow_path 中反复分配
    std::vector<CostMapGridView> step_cost_grids_cache_;

    // 缓存的弧长查找表
    std::vector<Eigen::Vector2d> prev_ref_control_points_;
    int prev_arc_samples_ = -1;
    ArclengthTable prev_arclength_table_ {};

    // ── Hidden-state Luenberger observer ──
    double x_h_hat_ = 0.0;
    double prev_v_act_ = 0.0;
    double prev_w_act_ = 0.0;
    bool observer_initialized_ = false;

    // ── Energy state ──
    double remaining_energy_ = 200.0;
    double rfr_pwr_limit_ = 90.0;

    // ── 辅助 ──
    StateVec make_initial_state(
        const Eigen::Vector3d& pose,
        const Eigen::Vector2d& status,
        const Eigen::Vector2d& cmd_clamped,
        double path_u
    ) const;
    static MPCPrediction extract_prediction(const StateVec* xs, int n);
};

}