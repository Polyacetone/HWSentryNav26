#include <path_follower/mpc_controller.hpp>
#include <rclcpp/logging.hpp>
#include <ceres/ceres.h>
#include <uniform_bspline/uniform_bspline.hpp>
#include <uniform_bspline_ceres/uniform_bspline_ceres_generator.hpp>

namespace path_follower {
template <typename ValueType>
using SplineT = ubs::UniformBSpline<ValueType, 2, ValueType, Eigen::Matrix<ValueType, 2, 1>, std::vector<Eigen::Matrix<ValueType, 2, 1>>>;
using SplineD = ubs::UniformBSpline<double, 2, double, Eigen::Vector2d, std::vector<Eigen::Vector2d>>;

struct MPCCostFuncor {
    MPCCostFuncor(
        ubs::UniformBSplineCeresGenerator<SplineT> generator,
        int num_control_points,
        Eigen::Vector3d start_point,
        double u0,
        Eigen::Vector2d start_state,
        MPCParams params
    ):  generator_(generator),
        num_control_points_(num_control_points),
        start_point_(start_point),
        u0_(u0),
        start_state_(start_state),
        params_(params) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        // 参数块布局：
        // [0 .. horizon-1] : 控制 (v, omega)
        // [horizon .. horizon+M-1] : 控制点 (x, y)
        T x = T(start_point_.x());
        T y = T(start_point_.y());
        T theta = T(start_point_.z());
        T u = T(u0_);

        T last_v = T(start_state_.x());
        T last_omega = T(start_state_.y());

        int res_idx = 0;
        const T* const* control_points_raw = parameters + params_.horizon;
        auto spline = generator_.template generate<T>(control_points_raw, true);
        for (int k = 0; k < params_.horizon; k++) {
            const T* uk = parameters[k];
            const T v = uk[0];
            const T omega = uk[1];

            const T dv = v - last_v;
            const T domega = omega - last_omega;

            // 参考点（由u决定）
            u = ceres::fmin(ceres::fmax(u, T(0.0)), T(1.0));
            const Eigen::Matrix<T, 2, 1> pr = spline.evaluate(u);
            const Eigen::Matrix<T, 2, 1> d1 = spline.derivative(u, 1);
            const Eigen::Matrix<T, 2, 1> d2 = spline.derivative(u, 2);

            const T dx = d1.x();
            const T dy = d1.y();
            const T ddx = d2.x();
            const T ddy = d2.y();

            const T thetar = atan2(dy, dx);
            const T dsdu = ceres::sqrt(dx * dx + dy * dy) + T(1e-6);
            const T kappa = (dx * ddy - dy * ddx) / (dsdu * dsdu * dsdu);

            const T ex = x - pr.x();
            const T ey_world = y - pr.y();
            const T ey = -ex * sin(thetar) + ey_world * cos(thetar);
            T etheta = theta - thetar;
            etheta = atan2(sin(etheta), cos(etheta));

            // 偏离路径代价
            residuals[res_idx++] = T(std::sqrt(params_.q_y)) * ey;
            residuals[res_idx++] = T(std::sqrt(params_.q_theta)) * etheta;

            // 进度项
            residuals[res_idx++] = T(std::sqrt(params_.q_u)) * (T(1.0) - u);

            // 控制正则
            residuals[res_idx++] = T(std::sqrt(params_.r_v)) * v;
            residuals[res_idx++] = T(std::sqrt(params_.r_omega)) * omega;

            // 控制平滑
            residuals[res_idx++] = T(std::sqrt(params_.r_dv)) * dv;
            residuals[res_idx++] = T(std::sqrt(params_.r_domega)) * domega;

            // 终点速度为0约束
            T terminal_weight = ceres::fmax(T(0.0), ceres::fmin(T(1.0), (u - T(0.99)) / T(0.01)));
            residuals[res_idx++] = T(std::sqrt(params_.q_v_final)) * terminal_weight * v;

            // 软加速度限制
            const T dv_limit = T(params_.acc_max * params_.dt);
            const T domega_limit = T(params_.alpha_max * params_.dt);
            const T dv_excess = ceres::fmax(T(0.0), ceres::abs(dv) - dv_limit);
            const T domega_excess = ceres::fmax(T(0.0), ceres::abs(domega) - domega_limit);
            residuals[res_idx++] = T(std::sqrt(params_.acc_limit_weight)) * dv_excess;
            residuals[res_idx++] = T(std::sqrt(params_.alpha_limit_weight)) * domega_excess;

            // 单车模型动力学
            x += v * cos(theta) * T(params_.dt);
            y += v * sin(theta) * T(params_.dt);
            theta += omega * T(params_.dt);

            // 进度动力学（将 ds/dt 转成 du/dt）
            T denom = T(1.0) - kappa * ey;
            const T denom_abs = ceres::abs(denom);
            // 保持符号，把 |denom| 下限抬到 0.1，避免除零/数值爆炸
            denom = denom / (denom_abs + T(1e-6)) * ceres::fmax(denom_abs, T(0.1));
            const T dsdt = v * cos(etheta) / denom;
            const T dudt = dsdt / dsdu;
            u += dudt * T(params_.dt);
            u = ceres::fmin(ceres::fmax(u, T(0.0)), T(1.0));

            last_v = v;
            last_omega = omega;
        }

        return true;
    }

    ubs::UniformBSplineCeresGenerator<SplineT> generator_;
    int num_control_points_{0};
    Eigen::Vector3d start_point_;
    double u0_{0.0};
    Eigen::Vector2d start_state_;
    MPCParams params_;
};
}

namespace path_follower {
MPCController::MPCController(const MPCParams& params) : params_(params) {}

double project_to_spline_u(
    const SplineD& spline,
    const Eigen::Vector2d& pos,
    double u_hint,
    int num_samples,
    double search_window,
    double max_correspondence_distance
) {
    const auto search = [&](double a, double b, int n) {
        double best_u = a;
        double best_d2 = std::numeric_limits<double>::infinity();
        for (int i = 0; i <= n; i++) {
            const double u = a + (b - a) * (double(i) / double(n));
            const Eigen::Vector2d p = spline.evaluate(u);
            const double d2 = (p - pos).squaredNorm();
            if (d2 < best_d2) {
                best_d2 = d2;
                best_u = u;
            }
        }
        return best_u;
    };

    // 先在窗口内找
    double u_best = search(
        std::clamp(u_hint - search_window, 0.0, 1.0),
        std::clamp(u_hint + search_window, 0.0, 1.0),
        num_samples
    );

    // 如果距离小于阈值，认为找到了
    Eigen::Vector2d p_best = spline.evaluate(u_best);
    if ((p_best - pos).norm() <= max_correspondence_distance) {
        return std::clamp(u_best, 0.0, 1.0);
    }

    // 否则在全域找
    u_best = search(0.0, 1.0, num_samples);
    u_best = search(
        std::clamp(u_best - search_window, 0.0, 1.0),
        std::clamp(u_best + search_window, 0.0, 1.0),
        num_samples
    );
    p_best = spline.evaluate(u_best);
    if ((p_best - pos).norm() <= max_correspondence_distance) {
        return std::clamp(u_best, 0.0, 1.0);
    }

    // 仍然找不到
    return -1.0;
}

Eigen::Vector2d MPCController::solve(
    const std::vector<Eigen::Vector2d>& control_points,
    const Eigen::Vector3d& current_pose,
    const Eigen::Vector2d& last_cmd_status
) {
    SplineD spline(control_points);
    spline.setExtrapolate(true);

    const double u0 = project_to_spline_u(
        spline,
        current_pose.head<2>(),
        last_u_,
        params_.proj_num_samples,
        params_.proj_search_window,
        params_.max_correspondence_distance
    );
    if (u0 < 0.0) {
        RCLCPP_ERROR(rclcpp::get_logger("path_follower"), "Failed to project current position to spline!");
        return Eigen::Vector2d::Zero();
    }
    last_u_ = u0;

    // 控制点作为常量参数块传入 cost functor，允许在 Jet 下评估样条
    const int num_cps = static_cast<int>(control_points.size());
    std::vector<std::array<double, 2>> control_point_blocks;
    control_point_blocks.reserve(num_cps);
    for (const auto& p : control_points) {
        control_point_blocks.push_back({p.x(), p.y()});
    }

    std::array<int, 1> shape{{num_cps}};
    ubs::UniformBSplineCeresGenerator<SplineT> generator(0.0, 1.0, shape);
    
    ceres::Problem problem;
    auto* cost_function = new ceres::DynamicAutoDiffCostFunction<MPCCostFuncor>(
        new MPCCostFuncor(generator, num_cps, current_pose, u0, last_cmd_status, params_)
    );

    std::vector<std::vector<double>> controls(params_.horizon, std::vector<double>(2));
    
    // 初始解给上一次的控制量
    for (int i = 0; i < params_.horizon; i++) {
        controls[i][0] = last_cmd_status.x();
        controls[i][1] = last_cmd_status.y();
        cost_function->AddParameterBlock(2);
    }

    for (int i = 0; i < num_cps; i++) {
        cost_function->AddParameterBlock(2);
    }
    
    // 每步残差：ey, etheta, (1-u), v, omega, dv, domega, v_final, dv_excess, domega_excess
    cost_function->SetNumResiduals(10 * params_.horizon);
    
    std::vector<double*> parameter_blocks;
    for (auto& c : controls) parameter_blocks.push_back(c.data());

    for (auto& cp : control_point_blocks) {
        problem.AddParameterBlock(cp.data(), 2);
        problem.SetParameterBlockConstant(cp.data());
        parameter_blocks.push_back(cp.data());
    }
    
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

Eigen::Vector2d MPCController::get_end_position(const std::vector<Eigen::Vector2d>& control_points) const {
    SplineD spline(control_points);
    spline.setExtrapolate(true);
    return spline.evaluate(1.0);
}
}