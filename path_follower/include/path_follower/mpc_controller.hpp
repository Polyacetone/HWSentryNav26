#pragma once

#include <Eigen/Dense>
#include <vector>

namespace path_follower {
struct MPCParams {
    int horizon;
    double dt;
    double target_vel;
    double vel_max, vel_min;
    double omega_max, omega_min;
    double acc_max, alpha_max;
    double pos_weight, angle_weight;
    double vel_smooth_weight, omega_smooth_weight;
    double acc_limit_weight, alpha_limit_weight;
};

class MPCController {
public:
    explicit MPCController(const MPCParams& params);

    void set_path(const std::vector<Eigen::Vector3d>& path_points, const std::vector<double>& path_dist);
    std::vector<Eigen::Vector3d> get_ref_traj(const Eigen::Vector3d& state) const;
    Eigen::Vector2d solve(const Eigen::Vector3d& current_pose, const Eigen::Vector2d& last_cmd_status);

private:
    MPCParams params_;
    std::vector<Eigen::Vector3d> path_points_;
    std::vector<double> path_dist_;
};
}