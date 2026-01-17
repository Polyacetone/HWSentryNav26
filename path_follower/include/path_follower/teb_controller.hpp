#pragma once

#include <Eigen/Dense>
#include <path_follower/nav_map.hpp>

namespace path_follower {

struct TebParams {
    int horizon;
    double dt;
    int max_iterations;
    int num_threads;

    double vel_max;
    double vel_min;
    double omega_max;
    double omega_min;

    double acc_max;
    double alpha_max;

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

    double obstacle_weight;
    double direction_weight;

    int proj_num_samples;
    double proj_search_window;
    double max_correspondence_distance;
};

class TebController {
public:
    struct Result {
        Eigen::Vector2d cmd_v_omega = Eigen::Vector2d::Zero();
        std::vector<Eigen::Vector2d> predicted_path_map;  // 预测轨迹（map坐标系）
        bool ok = false;
    };

    explicit TebController(const TebParams& params);

    void set_reference_path(std::vector<Eigen::Vector2d> path_points_map);
    bool has_reference_path() const;
    Eigen::Vector2d get_destination() const;

    // 输入：当前位姿(x,y,theta)、当前状态(v,omega)、地图（可为空）
    Result solve(
        const Eigen::Vector3d& current_pose_map,
        const Eigen::Vector2d& current_state,
        const CostMap* merged_cost_map,
        const DirectionMap* global_direction_map
    );

private:
    TebParams params_;
    std::vector<Eigen::Vector2d> ref_path_map_;
    std::vector<Eigen::Vector2d> last_controls_;
    double last_u_ = 0.0;
};

}