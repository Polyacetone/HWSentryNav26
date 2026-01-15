#pragma once

#include <Eigen/Dense>
#include <vector>

namespace path_follower {
struct MPCParams {
    int horizon;
    double dt;
    double vel_max, vel_min;
    double omega_max, omega_min;
    double acc_max, alpha_max;

    // 代价权重
    double q_y;
    double q_theta;
    double q_u;
    double q_v_final;

    // 速度/角速度正则
    double r_v;
    double r_omega;

    // 控制增量平滑
    double r_dv;
    double r_domega;

    // 加速度/角加速度限制
    double acc_limit_weight;
    double alpha_limit_weight;

    // 路径参数投影
    int proj_num_samples;
    double proj_search_window;
    double max_correspondence_distance;
};

class MPCController {
public:
    explicit MPCController(const MPCParams& params);

    void set_control_points(const std::vector<Eigen::Vector2d>& control_points);
    bool has_control_points() const;
    Eigen::Vector2d get_destination() const;
    Eigen::Vector2d step(const Eigen::Vector3d& current_pose);

private:
    const MPCParams params_;
    std::vector<Eigen::Vector2d> control_points_;
    Eigen::Vector2d last_cmd_status_ = Eigen::Vector2d::Zero();
    double last_u_ = 0;
};
}