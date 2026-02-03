#pragma once

#include <expected>
#include <string>
#include <cctype>
#include <Eigen/Dense>
#include <path_follower/nav_map.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

enum class PredictionModel { NONE, LAG, LQR };

struct LagModel {
    // 线速度延迟模型
    // v_dot = a
    // a_dot = (1/tau_v) * (k_v * (v_cmd - v) - a)
    double tau_v;
    double k_v;

    // 角速度一阶惯性延迟模型
    // omega_dot = (1/tau_omega) * (omega_cmd - omega)
    double tau_omega;
};

struct LQRModel {
    // 离散化闭环矩阵（子步）: x_{k+1} = A_cl * x_k + B_ref * x_ref
    // 其中 A_cl = exp((A - B*K) * dt_sub), B_ref = \int_0^{dt_sub} exp((A - B*K)\tau) d\tau * (B*K)
    Eigen::Matrix<double, 10, 10> A_cl;   // 离散闭环状态矩阵（子步）
    Eigen::Matrix<double, 10, 10> B_ref;  // 离散参考输入矩阵（子步）
    int substeps;                         // 每个 MPC 步内的子步数
    double dt_sub;                        // LQR 子步时长 (s)
};

struct MPCFollowLimits {
    double vel_max;
    double vel_min;
    double omega_max;
    double omega_min;

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

    double acc_limit_weight;
    double alpha_limit_weight;
    double lat_acc_weight;

    double vel_on_step_weight;
    double obstacle_weight;
    double direction_weight;
    double step_weight;
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

    double acc_max;
    double alpha_max;

    double vel_step_up;
    double vel_step_down;
    double a_lat_max;
};

struct MPCStopWeights {
    double q_v;
    double q_omega;

    double acc_limit_weight;
    double alpha_limit_weight;
    double lat_acc_weight;
    double vel_on_step_weight;

    double obstacle_weight;
    double obstacle_terminal_weight;
    double direction_weight;
    double step_terminal_weight;
};

struct MPCParams {
    int horizon;
    double dt;
    int max_iterations;

    PredictionModel prediction_model;
    LagModel lag;
    LQRModel lqr;

    MPCFollowLimits follow_limits;
    MPCFollowWeights follow_weights;
    MPCFollowProjection follow_projection;

    MPCStopLimits stop_limits;
    MPCStopWeights stop_weights;
};

class MPCController {
public:
    explicit MPCController(const MPCParams& params);

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

private:
    MPCParams params_;
    std::vector<Eigen::Vector2d> last_controls_;
    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero();
    double last_u_ = 0.0;
};

}