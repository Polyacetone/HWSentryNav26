#pragma once

#include <array>
#include <expected>
#include <string>
#include <Eigen/Dense>
#include <path_follower/nav_map.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

// ── MPC 编译期常量 ──
constexpr int MPC_HORIZON = 20;
constexpr double MPC_DT = 0.05;
constexpr int MPC_MAX_ITERATIONS = 60;
constexpr int MPC_CONTROL_SIZE = 2;
constexpr int MPC_STATE_SIZE = 10;
constexpr int MPC_PARAM_SIZE = MPC_CONTROL_SIZE * MPC_HORIZON;

constexpr int ARCLENGTH_TABLE_SIZE = 128;

constexpr double SGN_EPS   = 0.05;
constexpr double CF1       = 0.026829611608016123;
constexpr double CF2       = -0.2010928891690858;
constexpr double CF3       = -0.5171878055992227;
constexpr double XH0       = -3.9704078719129248;
constexpr double A00       = 0.9483617718409404;
constexpr double A01       = -2.4476243257126464;
constexpr double A03       = 2.798730623635046;
constexpr double A10       = 0.0033272418652128084;
constexpr double A11       = 0.9605359128145212;
constexpr double A13       = 0.014581231780666865;
constexpr double GNL_XH    = -0.06210638535453685;
constexpr double GNL_V     = 0.04904295219367288;
constexpr double A22       = 0.42595701520417945;
constexpr double A24       = 0.5740429847958206;
constexpr double GAMMA_W   = 0.0336320398873947;
constexpr double OBS_L     = 104.69986431799704;

constexpr int    PWR_N      = 12;
constexpr double PWR_EPS2   = 0.05 * 0.05;
constexpr double PWR_C[PWR_N] = {
    3.1183599570e+00,
    3.4172476463e+01,
    1.0359111933e+00,
    3.6371494354e+00,
    2.3486803448e-02,
    2.7300289323e+01,
    2.6315570711e+00,
    1.8359691253e+00,
    1.1200532785e+00,
    2.6043584920e-01,
    5.2574769643e-02,
    0.0000000000e+00
};

constexpr std::array<double, MPC_STATE_SIZE> DYNAMICS_WEIGHTS = {
    2000.0, 2000.0, 1000.0, // X, Y, Theta
    500.0,  500.0, 500.0,   // XH, V_ACT, W_ACT
    500.0, 500.0,  // V_CMD_Z1, W_CMD_Z1
    1000.0, 10.0   // PATH_U, ENERGY
};

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
    double a_lat_max;

    double slow_down_deceleration;
    int slow_down_num_samples;

    double step_norm_threshold;
    double step_norm_transition;
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
    double step;
};

struct MPCFollowProjection {
    int proj_num_samples;
    double proj_search_window;
    double max_correspondence_distance;
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
    double a_lat_max;

    double step_norm_threshold;
    double step_norm_transition;
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

    double step_norm_threshold;
    double step_norm_transition;
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

struct EnergyParams {
    bool enable = false;
    double threshold = 60.0;     // 电容惩罚阈值 (J)
    double weight = 2.0;         // 软约束权重
    double softplus_beta = 10.0; // softplus 斜率(越大越接近 ReLU)，建议 5~30
};

/// MPC 预测轨迹（灰箱模型输出）
struct MPCPrediction {
    std::vector<Eigen::Vector2d> path_map;   ///< (x, y) 位置, N+1 points（含初始点）
    std::vector<double> headings;            ///< theta 朝向 (rad), N+1
    std::vector<double> v_pred;              ///< 预测线速度响应, N+1
    std::vector<double> w_pred;              ///< 预测角速度响应, N+1
};

struct MPCParams {
    MPCFollowLimits follow_limits;
    MPCFollowWeights follow_weights;
    MPCFollowProjection follow_projection;

    MPCStopLimits stop_limits;
    MPCStopWeights stop_weights;

    MPCRecoveryLimits recovery_limits;
    MPCRecoveryWeights recovery_weights;

    EnergyParams energy;
};

class MPCSolver {
public:
    explicit MPCSolver(const MPCParams& params);

    void set_last_cmd(const Eigen::Vector2d& cmd);
    void reset_warm_start();

    /// Update the hidden-state observer with the latest measured velocities.
    /// Must be called once per control cycle before any solve call.
    void update_observer(double v_act, double w_act);

    /// Set current energy state (remaining capacitor energy + charge power limit).
    void set_energy_state(double remaining_energy, double rfr_pwr_limit);

    /// Current hidden-state estimate (pitch proxy).
    [[nodiscard]] double hidden_state_estimate() const { return x_h_hat_; }

    std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> follow_path(
        const SplineD& global_path,
        const Eigen::Vector3d& chassis_pose_map,
        const Eigen::Vector2d& chassis_status,
        const CostMap& merged_cost_map,
        const DirectionMap& global_direction_map
    );

    std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> stop(
        const Eigen::Vector3d& chassis_pose_map,
        const Eigen::Vector2d& chassis_status,
        const CostMap& merged_cost_map,
        const DirectionMap& global_direction_map
    );

    std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> recover_to_point(
        const Eigen::Vector2d& goal_map,
        const Eigen::Vector3d& chassis_pose_map,
        const Eigen::Vector2d& chassis_status,
        const CostMap& merged_cost_map,
        const DirectionMap& global_direction_map
    );

    [[nodiscard]] const MPCParams& params() const { return params_; }

private:
    MPCParams params_;
    std::array<double, MPC_PARAM_SIZE> last_controls_{};
    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero();
    double last_u_ = 0.0;

    // ── Hidden-state Luenberger observer ──
    double x_h_hat_ = 0.0;        // current estimate of hidden pitch state
    double prev_v_act_ = 0.0;     // v_act at previous cycle (for prediction)
    double prev_w_act_ = 0.0;     // w_act at previous cycle (for nonlinear term)
    bool observer_initialized_ = false;

    // ── Energy state (set each cycle from ChassisStatus) ──
    double remaining_energy_ = 200.0;
    double rfr_pwr_limit_ = 90.0;
};

}