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
};

class MPCSolver {
public:
    explicit MPCSolver(const MPCParams& params);

    void set_last_cmd(const Eigen::Vector2d& cmd);
    void reset_warm_start();

    std::expected<std::tuple<Eigen::Vector2d, std::vector<Eigen::Vector2d>>, std::string> follow_path(
        const SplineD& global_path,
        const Eigen::Vector3d& chassis_pose_map,
        const Eigen::Vector2d& chassis_status,
        const CostMap& merged_cost_map,
        const DirectionMap& global_direction_map
    );

    std::expected<std::tuple<Eigen::Vector2d, std::vector<Eigen::Vector2d>>, std::string> stop(
        const Eigen::Vector3d& chassis_pose_map,
        const Eigen::Vector2d& chassis_status,
        const CostMap& merged_cost_map,
        const DirectionMap& global_direction_map
    );

    std::expected<std::tuple<Eigen::Vector2d, std::vector<Eigen::Vector2d>>, std::string> recover_to_point(
        const Eigen::Vector2d& goal_map,
        const Eigen::Vector3d& chassis_pose_map,
        const Eigen::Vector2d& chassis_status,
        const CostMap& merged_cost_map,
        const DirectionMap& global_direction_map
    );

    [[nodiscard]] const MPCParams& params() const { return params_; }

private:
    /// 计算当前 x_h（pitch 隐藏状态）的 Luenberger 观测器估计值。
    /// 在每次求解前调用，基于上一周期存储的状态与本周期的 v 量测进行修正。
    double estimate_xh(double v_meas_now, double w_meas_now) const;

    /// 求解后存储本周期信息，供下一周期观测器使用。
    void store_observer_state(double xh_est, double v_meas, double w_meas, double dv_clamped);

    MPCParams params_;
    std::vector<Eigen::Vector2d> last_controls_;
    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero();
    double last_u_ = 0.0;

    // ── x_h Luenberger 观测器状态 ──
    struct XhObserver {
        double xh = 0.0;       // 当前 x_h 估计
        double v_prev = 0.0;   // 上一周期的 v 量测
        double w_prev = 0.0;   // 上一周期的 w 量测（用于非线性项）
        double dv_prev = 0.0;  // 上一周期的 dv（= 限幅后的 v_cmd_{k-1}）
        bool initialized = false;
    } xh_obs_;
};

}