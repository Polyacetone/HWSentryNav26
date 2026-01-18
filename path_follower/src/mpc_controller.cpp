#include <ceres/ceres.h>
#include <uniform_bspline/uniform_bspline.hpp>
#include <uniform_bspline_ceres/uniform_bspline_ceres_generator.hpp>
#include <path_follower/mpc_controller.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

// --------- Jet 兼容的小工具：取标量部分用于索引 ---------

inline double scalar_value(double v) { return v; }

template <typename S, int N>
inline double scalar_value(const ceres::Jet<S, N>& v) { return static_cast<double>(v.a); }

// --------- cost map / direction map 的可微插值 ---------

template <typename T>
T interpolate_cost_map(const CostMap& cost_map, const T& x_map, const T& y_map) {
    const T gx = (x_map - T(cost_map.origin_x)) / T(cost_map.resolution);
    const T gy = (y_map - T(cost_map.origin_y)) / T(cost_map.resolution);

    const double gxs = scalar_value(gx);
    const double gys = scalar_value(gy);

    const int x0 = static_cast<int>(std::floor(gxs));
    const int y0 = static_cast<int>(std::floor(gys));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    if (x0 < 0 || y0 < 0 || x1 >= cost_map.width || y1 >= cost_map.height) {
        return T(255.0);
    }

    const T dx = gx - T(x0);
    const T dy = gy - T(y0);

    const double c00 = static_cast<double>(cost_map.at({x0, y0}));
    const double c10 = static_cast<double>(cost_map.at({x1, y0}));
    const double c01 = static_cast<double>(cost_map.at({x0, y1}));
    const double c11 = static_cast<double>(cost_map.at({x1, y1}));

    return (T(1.0) - dx) * (T(1.0) - dy) * T(c00) + dx * (T(1.0) - dy) * T(c10) + (T(1.0) - dx) * dy * T(c01) + dx * dy * T(c11);
}

template <typename T>
Eigen::Matrix<T, 2, 1> interpolate_direction_map(const DirectionMap& dir_map, const T& x_map, const T& y_map) {
    const T gx = (x_map - T(dir_map.origin_x)) / T(dir_map.resolution);
    const T gy = (y_map - T(dir_map.origin_y)) / T(dir_map.resolution);

    const double gxs = scalar_value(gx);
    const double gys = scalar_value(gy);

    const int x0 = static_cast<int>(std::floor(gxs));
    const int y0 = static_cast<int>(std::floor(gys));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    if (x0 < 0 || y0 < 0 || x1 >= dir_map.width || y1 >= dir_map.height) {
        return Eigen::Matrix<T, 2, 1>(T(0.0), T(0.0));
    }

    const T dx = gx - T(x0);
    const T dy = gy - T(y0);

    const auto at = [&](int x, int y) {
        const Eigen::Vector2d v = dir_map.data[static_cast<size_t>(y * dir_map.width + x)];
        return Eigen::Matrix<T, 2, 1>(T(v.x()), T(v.y()));
    };

    const auto v00 = at(x0, y0);
    const auto v10 = at(x1, y0);
    const auto v01 = at(x0, y1);
    const auto v11 = at(x1, y1);

    return (T(1.0) - dx) * (T(1.0) - dy) * v00 + dx * (T(1.0) - dy) * v10 + (T(1.0) - dx) * dy * v01 + dx * dy * v11;
}

// --------- 剩余弧长估计 ---------

template <typename T, typename Spline>
T estimate_remaining_arclength(const Spline& spline, const T& u_in, int num_samples) {
    const T u = ceres::fmin(ceres::fmax(u_in, T(0.0)), T(1.0));
    const T one_minus_u = T(1.0) - u;

    if (num_samples <= 1) {
        const auto d1 = spline.derivative(u, 1);
        const T dsdu = ceres::sqrt(d1.squaredNorm() + T(1e-12));
        return one_minus_u * dsdu;
    }

    T length = T(0.0);
    T u_prev = u;
    const auto d1_prev = spline.derivative(u_prev, 1);
    T dsdu_prev = ceres::sqrt(d1_prev.squaredNorm() + T(1e-12));

    for (int i = 1; i <= num_samples; i++) {
        const T ui = u + one_minus_u * (T(i) / T(num_samples));
        const auto d1 = spline.derivative(ui, 1);
        const T dsdu = ceres::sqrt(d1.squaredNorm() + T(1e-12));
        const T du = ui - u_prev;
        length += (dsdu_prev + dsdu) * T(0.5) * du;
        u_prev = ui;
        dsdu_prev = dsdu;
    }

    return length;
}

// --------- Ceres cost functor：优化 (v,omega) 序列 ---------

struct MPCPathCostFunctor {
    MPCPathCostFunctor(
        const ubs::UniformBSplineCeresGenerator<SplineT>& generator,
        const int num_control_points,
        const Eigen::Vector3d& start_pose,
        const double u0,
        const Eigen::Vector2d& start_state,
        const MPCParams& params,
        const CostMap& merged_cost_map,
        const DirectionMap& direction_map
    ) : generator_(generator),
        num_control_points_(num_control_points),
        start_pose_(start_pose),
        u0_(u0),
        start_state_(start_state),
        params_(params),
        merged_cost_map_(merged_cost_map),
        direction_map_(direction_map) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        // 参数块布局：
        // [0 .. horizon-1] : 控制 (v, omega)
        // [horizon .. horizon+M-1] : 参考路径点 (x, y)（常量参数块）

        T x = T(start_pose_.x());
        T y = T(start_pose_.y());
        T theta = T(start_pose_.z());
        T u = T(u0_);

        T last_v = T(start_state_.x());
        T last_omega = T(start_state_.y());

        // 每一步的 residual 个数：
        // ey(1) + etheta(1) + progress_u(1) + reg(v,omega)(2) + smooth(dv,domega)(2)
        // + goal_slow_down(1) + soft_limits(2) + v_omega_product(1) + obstacle(1) + direction(1) + step_speed(1) = 14
        const int expected_residuals = 14 * params_.horizon;

        int res_idx = 0;

        const T* const* control_points_raw = parameters + params_.horizon;
        auto spline = generator_.template generate<T>(control_points_raw, true);

        for (int k = 0; k < params_.horizon; k++) {
            const T* uk = parameters[k];
            const T v = uk[0];
            const T omega = uk[1];
            const T dv = v - last_v;
            const T domega = omega - last_omega;

            // 单车模型动力学
            x += v * cos(theta) * T(params_.dt);
            y += v * sin(theta) * T(params_.dt);
            theta += omega * T(params_.dt);

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

            // Frenet 横向误差
            const T ex = x - pr.x();
            const T ey_world = y - pr.y();
            const T ey = -ex * sin(thetar) + ey_world * cos(thetar);
            T etheta = theta - thetar;
            etheta = atan2(sin(etheta), cos(etheta));

            // 贴合全局路径
            residuals[res_idx++] = T(params_.path_follow.q_y) * ey;
            residuals[res_idx++] = T(params_.path_follow.q_theta) * etheta;

            // 推进项（鼓励u增长）
            residuals[res_idx++] = T(params_.path_follow.q_u) * (T(1.0) - u);

            // 控制正则
            residuals[res_idx++] = T(params_.path_follow.r_v) * v;
            residuals[res_idx++] = T(params_.path_follow.r_omega) * omega;

            // 控制平滑
            residuals[res_idx++] = T(params_.path_follow.r_dv) * dv;
            residuals[res_idx++] = T(params_.path_follow.r_domega) * domega;

            // 软加速度限制
            const T dv_limit = T(params_.path_follow.acc_max * params_.dt);
            const T domega_limit = T(params_.path_follow.alpha_max * params_.dt);
            const T dv_excess = ceres::fmax(T(0.0), ceres::abs(dv) - dv_limit);
            const T domega_excess = ceres::fmax(T(0.0), ceres::abs(domega) - domega_limit);
            residuals[res_idx++] = T(params_.path_follow.acc_limit_weight) * dv_excess;
            residuals[res_idx++] = T(params_.path_follow.alpha_limit_weight) * domega_excess;

            // 限制 |v * omega| 软约束
            const T vomega_excess = ceres::fmax(T(0.0), ceres::abs(v * omega) - T(params_.path_follow.v_omega_product_max));
            residuals[res_idx++] = T(params_.path_follow.v_omega_product_weight) * vomega_excess;

            // 避障（使用合并代价地图：全局+局部）
            const T cost = interpolate_cost_map(merged_cost_map_, x, y);
            // 归一化到 [0,1]，并进行惩罚
            residuals[res_idx++] = T(params_.path_follow.obstacle_weight) * (cost / T(255.0));

            // 对齐台阶方向（方向地图在障碍附近/台阶上才有意义；无方向则不惩罚）
            const auto dir = interpolate_direction_map(direction_map_, x, y);
            const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-9));
            const auto dir_normalized = dir / dir_norm;
            const Eigen::Matrix<T, 2, 1> heading(cos(theta), sin(theta));
            const T heading_dot_dir = heading.dot(dir_normalized);
            // 惩罚 1 - |cos|，即鼓励朝向与方向场对齐（允许正反向都对齐）
            residuals[res_idx++] = T(params_.path_follow.direction_weight) * (T(1.0) - ceres::abs(heading_dot_dir));

            // 台阶区域限速：当方向场非0（台阶）时，惩罚超过 step_speed_max 的速度
            // 使用 direction norm 做软开关，避免边界插值处不连续
            const T gate_step = ceres::fmin(T(1.0), dir_norm * 2.0); // [0,1]
            const T v_excess_step = ceres::fmax(T(0.0), v - T(params_.path_follow.vel_max_on_step));
            residuals[res_idx++] = T(params_.path_follow.vel_max_on_step_weight) * gate_step * v_excess_step;

            // 终点减速：基于剩余弧长在 goal_slow_down_distance 内逐渐限速
            // 设计为软约束形式：只惩罚 v 超过 v_limit_goal(remaining_distance)
            const T s_remain = estimate_remaining_arclength(spline, u, params_.path_follow.slow_down_num_samples);
            const T slow_dist = T(params_.path_follow.slow_down_distance);
            const T s_remain_ratio = ceres::fmin(T(1.0), s_remain / (slow_dist + T(1e-6))); // 1: far, 0: at goal
            const T gate_goal = ceres::fmin(T(1.0), ceres::fmax(T(0.0), (slow_dist - s_remain) / (slow_dist + T(1e-6))));
            // v_limit_goal: far -> vel_max, near -> vel_min
            const T v_limit_goal = T(params_.path_follow.vel_min) + (T(params_.path_follow.vel_max) - T(params_.path_follow.vel_min)) * s_remain_ratio;
            const T v_excess_goal = ceres::fmax(T(0.0), v - v_limit_goal);
            residuals[res_idx++] = T(params_.path_follow.q_v_final) * gate_goal * v_excess_goal;

            // 进度动力学（将 ds/dt 转成 du/dt）
            T denom = T(1.0) - kappa * ey;
            const T denom_abs = ceres::abs(denom);
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

    const ubs::UniformBSplineCeresGenerator<SplineT>& generator_;
    const int num_control_points_;
    const Eigen::Vector3d& start_pose_;
    const double u0_;
    const Eigen::Vector2d& start_state_;
    const MPCParams& params_;
    const CostMap& merged_cost_map_;
    const DirectionMap& direction_map_;
};

// --------- Ceres cost functor：原地旋转（小陀螺） ---------

struct MPCSpinCostFunctor {
    MPCSpinCostFunctor(
        const Eigen::Vector3d& start_pose,
        const Eigen::Vector2d& start_state,
        const MPCParams& params,
        const double target_omega
    ) : start_pose_(start_pose),
        start_state_(start_state),
        params_(params),
        target_omega_(target_omega) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        T theta = T(start_pose_.z());
        T last_v = T(start_state_.x());
        T last_omega = T(start_state_.y());

        int res_idx = 0;
        for (int k = 0; k < params_.horizon; k++) {
            const T* uk = parameters[k];
            const T v = uk[0];
            const T omega = uk[1];
            const T dv = v - last_v;
            const T domega = omega - last_omega;

            // 单车模型动力学
            theta += omega * T(params_.dt);

            // 跟踪目标角速度
            residuals[res_idx++] = T(params_.spin_follow.q_omega) * (omega - T(target_omega_));

            // 控制正则
            residuals[res_idx++] = T(params_.spin_follow.r_v) * v;

            // 软加速度限制
            const T dv_limit = T(params_.spin_follow.acc_max * params_.dt);
            const T domega_limit = T(params_.spin_follow.alpha_max * params_.dt);
            const T dv_excess = ceres::fmax(T(0.0), ceres::abs(dv) - dv_limit);
            const T domega_excess = ceres::fmax(T(0.0), ceres::abs(domega) - domega_limit);
            residuals[res_idx++] = T(params_.spin_follow.acc_limit_weight) * dv_excess;
            residuals[res_idx++] = T(params_.spin_follow.alpha_limit_weight) * domega_excess;

            // 限制 |v * omega| 软约束
            const T vomega_excess = ceres::fmax(T(0.0), ceres::abs(v * omega) - T(params_.spin_follow.v_omega_product_max));
            residuals[res_idx++] = T(params_.spin_follow.v_omega_product_weight) * vomega_excess;

            last_v = v;
            last_omega = omega;
        }

        return true;
    }

    const Eigen::Vector3d& start_pose_;
    const Eigen::Vector2d& start_state_;
    const MPCParams& params_;
    const double target_omega_;
};

}

namespace path_follower {

MPCController::MPCController(const MPCParams& params) : params_(params) {
    last_controls_.assign(static_cast<size_t>(std::max(1, params_.horizon)), Eigen::Vector2d::Zero());
}

std::expected<std::tuple<Eigen::Vector2d, std::vector<Eigen::Vector2d>>, std::string> MPCController::solve_path(
    const SplineD& global_path,
    const Eigen::Vector3d& current_pose_map,
    const Eigen::Vector2d& current_state,
    const CostMap& merged_cost_map,
    const DirectionMap& global_direction_map
) {
    const double u0 = project_to_spline_u(
        global_path,
        current_pose_map.head<2>(),
        last_u_,
        params_.projection.num_samples,
        params_.projection.search_window,
        params_.projection.max_correspondence_distance
    );

    last_u_ = u0;

    // 将参考路径点作为常量参数块，支持 Jet 下评估样条
    const int num_pts = static_cast<int>(global_path.getControlPoints().size());
    std::vector<std::array<double, 2>> ref_point_blocks;
    ref_point_blocks.reserve(static_cast<size_t>(num_pts));
    for (const auto& p : global_path.getControlPoints()) {
        ref_point_blocks.push_back({p.x(), p.y()});
    }

    ubs::UniformBSplineCeresGenerator<SplineT> generator(0.0, 1.0, {num_pts});

    // 决策变量：控制序列 (v, omega)
    std::vector<std::vector<double>> controls(static_cast<size_t>(params_.horizon), std::vector<double>(2, 0.0));

    // warm start：把上次解往前移一格，否则用当前状态
    if (last_controls_.size() == static_cast<size_t>(params_.horizon)) {
        for (int i = 0; i < params_.horizon; i++) {
            const Eigen::Vector2d init = (i + 1 < params_.horizon) ? last_controls_[static_cast<size_t>(i + 1)] : last_controls_.back();
            controls[static_cast<size_t>(i)][0] = init.x();
            controls[static_cast<size_t>(i)][1] = init.y();
        }
    } else {
        for (int i = 0; i < params_.horizon; i++) {
            controls[static_cast<size_t>(i)][0] = current_state.x();
            controls[static_cast<size_t>(i)][1] = current_state.y();
        }
    }

    ceres::Problem problem;

    auto* cost_function = new ceres::DynamicAutoDiffCostFunction<MPCPathCostFunctor>(
        new MPCPathCostFunctor(
            generator,
            num_pts,
            current_pose_map,
            u0,
            current_state,
            params_,
            merged_cost_map,
            global_direction_map
        )
    );

    // 参数块：控制 + 参考路径点
    for (int i = 0; i < params_.horizon; i++) {
        cost_function->AddParameterBlock(2);
    }
    for (int i = 0; i < num_pts; i++) {
        cost_function->AddParameterBlock(2);
    }

    // residual 总数：每步 14 个
    cost_function->SetNumResiduals(14 * params_.horizon);

    std::vector<double*> parameter_blocks;
    parameter_blocks.reserve(static_cast<size_t>(params_.horizon + num_pts));

    for (auto& c : controls) parameter_blocks.push_back(c.data());

    for (auto& rp : ref_point_blocks) {
        problem.AddParameterBlock(rp.data(), 2);
        problem.SetParameterBlockConstant(rp.data());
        parameter_blocks.push_back(rp.data());
    }

    problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);

    for (int i = 0; i < params_.horizon; i++) {
        problem.SetParameterLowerBound(controls[static_cast<size_t>(i)].data(), 0, params_.path_follow.vel_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(i)].data(), 0, params_.path_follow.vel_max);
        problem.SetParameterLowerBound(controls[static_cast<size_t>(i)].data(), 1, params_.path_follow.omega_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(i)].data(), 1, params_.path_follow.omega_max);
    }

    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = params_.max_iterations;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable()) {
        return std::unexpected(summary.BriefReport());
    }

    // 输出第一步控制
    Eigen::Vector2d cmd_v_omega(controls[0][0], controls[0][1]);

    // 保存 warm start
    last_controls_.resize(static_cast<size_t>(params_.horizon));
    for (int i = 0; i < params_.horizon; i++) {
        last_controls_[static_cast<size_t>(i)] = Eigen::Vector2d(
            controls[static_cast<size_t>(i)][0],
            controls[static_cast<size_t>(i)][1]
        );
    }

    // 生成预测轨迹用于调试展示
    std::vector<Eigen::Vector2d> predicted_path_map;
    predicted_path_map.reserve(static_cast<size_t>(params_.horizon + 1));
    Eigen::Vector3d pose = current_pose_map;
    predicted_path_map.push_back(pose.head<2>());
    for (int i = 0; i < params_.horizon; i++) {
        const double v = controls[static_cast<size_t>(i)][0];
        const double w = controls[static_cast<size_t>(i)][1];
        pose.x() += v * std::cos(pose.z()) * params_.dt;
        pose.y() += v * std::sin(pose.z()) * params_.dt;
        pose.z() += w * params_.dt;
        predicted_path_map.push_back(pose.head<2>());
    }

    return std::tuple{cmd_v_omega, predicted_path_map};
}

std::expected<std::tuple<Eigen::Vector2d, std::vector<Eigen::Vector2d>>, std::string> MPCController::solve_spin(
    const Eigen::Vector3d& current_pose_map,
    const Eigen::Vector2d& current_state,
    double target_omega
) {
    // 决策变量：控制序列 (v, omega)
    std::vector<std::vector<double>> controls(static_cast<size_t>(params_.horizon), std::vector<double>(2, 0.0));

    // warm start：沿用上次解（不区分模式）
    if (last_controls_.size() == static_cast<size_t>(params_.horizon)) {
        for (int i = 0; i < params_.horizon; i++) {
            const Eigen::Vector2d init = (i + 1 < params_.horizon) ? last_controls_[static_cast<size_t>(i + 1)] : last_controls_.back();
            controls[static_cast<size_t>(i)][0] = init.x();
            controls[static_cast<size_t>(i)][1] = init.y();
        }
    } else {
        for (int i = 0; i < params_.horizon; i++) {
            controls[static_cast<size_t>(i)][0] = 0.0;
            controls[static_cast<size_t>(i)][1] = target_omega;
        }
    }

    ceres::Problem problem;

    auto* cost_function = new ceres::DynamicAutoDiffCostFunction<MPCSpinCostFunctor>(
        new MPCSpinCostFunctor(
            current_pose_map,
            current_state,
            params_,
            target_omega
        )
    );

    for (int i = 0; i < params_.horizon; i++) {
        cost_function->AddParameterBlock(2);
    }

    // residual 总数：每步 5 个
    // q_omega(1) + r_v(1) + soft_limits(2) + v_omega_product(1)
    cost_function->SetNumResiduals(5 * params_.horizon);

    std::vector<double*> parameter_blocks;
    parameter_blocks.reserve(static_cast<size_t>(params_.horizon));
    for (auto& c : controls) parameter_blocks.push_back(c.data());

    problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);

    for (int i = 0; i < params_.horizon; i++) {
        problem.SetParameterLowerBound(controls[static_cast<size_t>(i)].data(), 0, 0.0);
        // 注意：spin 模式不对 v 施加上界硬约束，避免从路径跟随切换时速度被卡在上界导致突变
        // 通过 r_v 与 soft constraints 让优化器自行把 v 收敛到 0
        problem.SetParameterLowerBound(controls[static_cast<size_t>(i)].data(), 1, params_.spin_follow.omega_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(i)].data(), 1, params_.spin_follow.omega_max);
    }

    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = params_.max_iterations;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable()) {
        return std::unexpected(summary.BriefReport());
    }

    Eigen::Vector2d cmd_v_omega(controls[0][0], controls[0][1]);

    last_controls_.resize(static_cast<size_t>(params_.horizon));
    for (int i = 0; i < params_.horizon; i++) {
        last_controls_[static_cast<size_t>(i)] = Eigen::Vector2d(
            controls[static_cast<size_t>(i)][0],
            controls[static_cast<size_t>(i)][1]
        );
    }

    std::vector<Eigen::Vector2d> predicted_path_map;
    predicted_path_map.reserve(static_cast<size_t>(params_.horizon + 1));
    Eigen::Vector3d pose = current_pose_map;
    predicted_path_map.push_back(pose.head<2>());
    for (int i = 0; i < params_.horizon; i++) {
        const double v = controls[static_cast<size_t>(i)][0];
        const double w = controls[static_cast<size_t>(i)][1];
        pose.x() += v * std::cos(pose.z()) * params_.dt;
        pose.y() += v * std::sin(pose.z()) * params_.dt;
        pose.z() += w * params_.dt;
        predicted_path_map.push_back(pose.head<2>());
    }

    return std::tuple{cmd_v_omega, predicted_path_map};
}

}