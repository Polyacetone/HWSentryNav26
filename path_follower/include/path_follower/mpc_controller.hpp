#pragma once

#include <expected>
#include <Eigen/Dense>
#include <path_follower/nav_map.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

struct HybridModel {
    // 线速度二阶系统参数
    double wn_v;        // 自然频率 (rad/s)
    double zeta_v;      // 阻尼比

    // 角速度一阶系统参数
    double tau_omega;   // 时间常数 (s)

    // 预计算值
    double wn_v_sq() const { return wn_v * wn_v; }
    double two_zeta_wn_v() const { return 2.0 * zeta_v * wn_v; }
};

struct RobotStatus {
    double v;       // 线速度 (m/s)
    double v_dot;   // 线加速度 (m/s^2)
    double omega;   // 角速度 (rad/s)
};

struct MPCFollowLimits {
    double vel_max;
    double vel_min;
    double omega_max;
    double omega_min;

    double acc_max;
    double alpha_max;

    double phys_acc_max;
    double phys_alpha_max;

    double vel_on_step;
    double a_lat_max;

    double slow_down_distance;
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
    double phys_acc_limit_weight;
    double phys_alpha_limit_weight;
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
    double phys_acc_max;
    double phys_alpha_max;

    double vel_on_step;
    double a_lat_max;
};

struct MPCStopWeights {
    double q_v;
    double q_omega;

    double acc_limit_weight;
    double alpha_limit_weight;
    double phys_acc_limit_weight;
    double phys_alpha_limit_weight;
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

    HybridModel model;

    MPCFollowLimits follow_limits;
    MPCFollowWeights follow_weights;
    MPCFollowProjection follow_projection;

    MPCStopLimits stop_limits;
    MPCStopWeights stop_weights;
};

class MPCController {
public:
    explicit MPCController(const MPCParams& params);

    std::expected<std::tuple<Eigen::Vector2d, std::vector<Eigen::Vector2d>>, std::string> follow_path(
        const SplineD& global_path,
        const Eigen::Vector3d& current_pose_map,
        const RobotStatus& current_status,
        const CostMap& merged_cost_map,
        const DirectionMap& global_direction_map
    );

    std::expected<std::tuple<Eigen::Vector2d, std::vector<Eigen::Vector2d>>, std::string> stop(
        const Eigen::Vector3d& current_pose_map,
        const RobotStatus& current_status,
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