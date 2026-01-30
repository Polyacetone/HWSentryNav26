#pragma once

#include <expected>
#include <Eigen/Dense>
#include <path_follower/nav_map.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

struct MPCModel {
    // First-order lag time constants for the closed-loop velocity response.
    // v_act[k+1] = alpha*v_act[k] + (1-alpha)*v_cmd[k], alpha = exp(-dt/tau_v)
    double tau_v;
    double tau_omega;
};

struct MPCFollowLimits {
    double vel_max;
    double vel_min;
    double omega_max;
    double omega_min;

    // Command slew-rate limits (interface protection). Unit: per second.
    // Implemented as soft constraint on |v_cmd[k]-v_cmd[k-1]| <= acc_max*dt.
    double acc_max;
    double alpha_max;

    // Physical acceleration limits on actual states (safety). Unit: per second.
    double phys_acc_max;
    double phys_alpha_max;

    double vel_on_step;

    // Lateral acceleration limit: |v_act * omega_act| <= a_lat_max
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

    // Soft constraint weights
    // - acc/alpha_limit_weight: command slew-rate violation
    // - phys_*: physical accel violation on actual states
    // - lat_acc_weight: lateral acceleration violation on actual states
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

    // Command slew-rate limits (interface protection). Unit: per second.
    double acc_max;
    double alpha_max;

    // Physical acceleration limits on actual states (safety). Unit: per second.
    double phys_acc_max;
    double phys_alpha_max;

    double vel_on_step;

    // Lateral acceleration limit: |v_act * omega_act| <= a_lat_max
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

    MPCModel model;

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
        const Eigen::Vector2d& current_state,
        const CostMap& merged_cost_map,
        const DirectionMap& global_direction_map
    );

    std::expected<std::tuple<Eigen::Vector2d, std::vector<Eigen::Vector2d>>, std::string> stop(
        const Eigen::Vector3d& current_pose_map,
        const Eigen::Vector2d& current_state,
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