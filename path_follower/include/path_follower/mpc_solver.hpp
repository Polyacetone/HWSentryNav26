#pragma once

#include <expected>
#include <string>
#include <Eigen/Dense>
#include <path_follower/nav_map.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

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
    double threshold = 60.0;   // 缓冲电容惩罚阈值 (J)
    double weight = 2.0;       // 软约束权重
    double softplus_beta = 10.0; // softplus 斜率(越大越接近 ReLU)，建议 5~30
};

/// MPC 预测轨迹（灰箱模型输出）
struct MPCPrediction {
    std::vector<Eigen::Vector2d> path_map;  ///< (x, y) 位置, N+1 points（含初始点）
    std::vector<double> headings;            ///< theta 朝向 (rad), N+1
    std::vector<double> v_pred;              ///< 预测线速度响应, N+1
    std::vector<double> w_pred;              ///< 预测角速度响应, N+1
};

struct MPCParams {
    int horizon;
    double dt;
    int max_iterations;

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
    std::vector<Eigen::Vector2d> last_controls_;
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