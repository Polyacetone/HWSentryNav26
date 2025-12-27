#include <path_follower/mpc_controller.hpp>
#include <ceres/ceres.h>

namespace path_follower {
struct MPCCostFunctor {
    MPCCostFunctor(
        const Eigen::Vector3d& start_state,
        const std::vector<Eigen::Vector3d>& ref_traj,
        const Eigen::Vector2d& last_u,
        const MPCParams& params
    ): start_state_(start_state), ref_traj_(ref_traj), last_u_(last_u), params_(params) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        T x = T(start_state_.x());
        T y = T(start_state_.y());
        T theta = T(start_state_.z());
        T last_v = T(last_u_.x());
        T last_omega = T(last_u_.y());

        int res_idx = 0;
        for (int i = 0; i < params_.horizon; i++) {
            const T* u = parameters[i];
            T v = u[0];
            T omega = u[1];

            // 控制增量
            const T dv = v - last_v;
            const T domega = omega - last_omega;

            // 动力学
            x += v * cos(theta) * T(params_.dt);
            y += v * sin(theta) * T(params_.dt);
            theta += omega * T(params_.dt);

            // 位置误差
            residuals[res_idx++] = T(sqrt(params_.pos_weight)) * (x - T(ref_traj_[i].x()));
            residuals[res_idx++] = T(sqrt(params_.pos_weight)) * (y - T(ref_traj_[i].y()));

            // 朝向误差
            T theta_err = theta - T(ref_traj_[i].z());
            residuals[res_idx++] = T(sqrt(params_.angle_weight)) * atan2(sin(theta_err), cos(theta_err));

            // 平滑项
            residuals[res_idx++] = T(sqrt(params_.vel_smooth_weight)) * dv;
            residuals[res_idx++] = T(sqrt(params_.omega_smooth_weight)) * domega;

            // 最大加速度/角加速度惩罚
            const T dv_limit = T(params_.acc_max * params_.dt);
            const T domega_limit = T(params_.alpha_max * params_.dt);
            const T dv_excess = ceres::fmax(T(0.0), ceres::abs(dv) - dv_limit);
            const T domega_excess = ceres::fmax(T(0.0), ceres::abs(domega) - domega_limit);
            residuals[res_idx++] = T(sqrt(params_.acc_limit_weight)) * dv_excess;
            residuals[res_idx++] = T(sqrt(params_.alpha_limit_weight)) * domega_excess;

            last_v = v;
            last_omega = omega;
        }
        return true;
    }

    const Eigen::Vector3d& start_state_;
    const std::vector<Eigen::Vector3d>& ref_traj_;
    const Eigen::Vector2d& last_u_;
    const MPCParams& params_;
};
}

namespace path_follower {
MPCController::MPCController(const MPCParams& params) : params_(params) {}

void MPCController::set_path(const std::vector<Eigen::Vector3d>& path_points, const std::vector<double>& path_dist) {
    path_points_ = path_points;
    path_dist_ = path_dist;
}

std::vector<Eigen::Vector3d> MPCController::get_ref_traj(const Eigen::Vector3d& state) const {
    std::vector<Eigen::Vector3d> ref_traj;
    if (path_points_.empty()) return ref_traj;

    // 找到最近点
    double min_dist = std::numeric_limits<double>::max();
    size_t min_idx = 0;
    for (size_t i = 0; i < path_points_.size(); i++) {
        double dist = (path_points_[i].head<2>() - state.head<2>()).norm();
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i;
        }
    }

    double current_s = path_dist_[min_idx];
    const double total_dist = path_dist_.back();

    for (int k = 0; k < params_.horizon; k++) {
        double dist_to_end = total_dist - current_s;
        if (dist_to_end < 0) dist_to_end = 0;
        
        // 到目的地时减速
        double v_limit = std::sqrt(2.0 * params_.acc_max * dist_to_end);
        double ref_v = std::min(params_.target_vel, v_limit);
        
        double ds = ref_v * params_.dt;
        current_s += ds;
        
        // 插值
        Eigen::Vector3d pt;
        if (current_s > path_dist_.back()) {
            pt = path_points_.back();
        } else {
            auto it = std::lower_bound(path_dist_.begin(), path_dist_.end(), current_s);
            size_t idx = std::distance(path_dist_.begin(), it);
            if (idx == 0) {
                pt = path_points_[0];
            } else {
                double s_prev = path_dist_[idx-1];
                double s_next = path_dist_[idx];
                double ratio = (current_s - s_prev) / (s_next - s_prev);
                
                Eigen::Vector3d p1 = path_points_[idx-1];
                Eigen::Vector3d p2 = path_points_[idx];
                
                pt.x() = p1.x() + ratio * (p2.x() - p1.x());
                pt.y() = p1.y() + ratio * (p2.y() - p1.y());
                
                double diff = p2.z() - p1.z();
                diff = atan2(sin(diff), cos(diff));
                pt.z() = p1.z() + ratio * diff;
            }
        }
        ref_traj.push_back(pt);
    }
    return ref_traj;
}

Eigen::Vector2d MPCController::solve(const Eigen::Vector3d& current_pose, const Eigen::Vector2d& last_cmd_status) {
    const std::vector<Eigen::Vector3d> ref_traj = get_ref_traj(current_pose);
    
    ceres::Problem problem;
    auto* cost_function = new ceres::DynamicAutoDiffCostFunction<MPCCostFunctor>(
        new MPCCostFunctor(current_pose, ref_traj, last_cmd_status, params_)
    );

    std::vector<std::vector<double>> controls(params_.horizon, std::vector<double>(2));
    
    // 初始解给上一次的控制量
    for (int i = 0; i < params_.horizon; i++) {
        controls[i][0] = last_cmd_status.x();
        controls[i][1] = last_cmd_status.y();
        cost_function->AddParameterBlock(2);
    }
    
    cost_function->SetNumResiduals(7 * params_.horizon);
    
    std::vector<double*> parameter_blocks;
    for (auto& c : controls) parameter_blocks.push_back(c.data());
    
    problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);

    for (int i = 0; i < params_.horizon; i++) {
        problem.SetParameterLowerBound(controls[i].data(), 0, params_.vel_min);
        problem.SetParameterUpperBound(controls[i].data(), 0, params_.vel_max);
        problem.SetParameterLowerBound(controls[i].data(), 1, params_.omega_min);
        problem.SetParameterUpperBound(controls[i].data(), 1, params_.omega_max);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Vector2d cmd_status(controls[0][0], controls[0][1]);
    return cmd_status;

}
}