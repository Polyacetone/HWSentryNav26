#pragma once

#include <expected>
#include <array>
#include <string>
#include <cctype>
#include <Eigen/Dense>
#include <path_follower/nav_map.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

struct MPCModel {
    // 离散 MIMO 状态空间模型（用于 MPC 内部预测 v_act/omega_act）
    // 状态：[ v(k), v(k-1), ..., v(k-n+1),  w(k),  v_cmd_z1=v_cmd(k-1),  w_cmd_z1=w_cmd(k-1) ]
    // 输入： [ v_cmd(k), w_cmd(k) ]
    // 输出： [ v_act=v(k), w_act=w(k) ]
    static constexpr int MAX_V_ORDER = 8;

    int v_order;
    std::array<double, MAX_V_ORDER> v_ar = {};  // a1..an, 仅使用 [0..v_order)
    double v_w_coeff;       // v(k+1) 中 w(k) 的系数
    double v_vcmd_z1_coeff; // v(k+1) 中 v_cmd(k-1) 的系数
    double w_alpha;         // w(k+1) = alpha*w(k) + beta*w_cmd(k-1)
    double w_beta;          // w(k+1) = alpha*w(k) + beta*w_cmd(k-1)
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

    MPCModel model;

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

    std::expected<std::tuple<Eigen::Vector3d, std::vector<Eigen::Vector2d>>, std::string> follow_path(
        const SplineD& global_path,
        const Eigen::Vector3d& chassis_pose_map,
        const Eigen::Vector2d& chassis_status,
        const CostMap& merged_cost_map,
        const DirectionMap& global_direction_map
    );

    std::expected<std::tuple<Eigen::Vector3d, std::vector<Eigen::Vector2d>>, std::string> stop(
        const Eigen::Vector3d& chassis_pose_map,
        const Eigen::Vector2d& chassis_status,
        const CostMap& merged_cost_map,
        const DirectionMap& global_direction_map
    );

    std::expected<std::tuple<Eigen::Vector3d, std::vector<Eigen::Vector2d>>, std::string> recover_to_point(
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
};

}