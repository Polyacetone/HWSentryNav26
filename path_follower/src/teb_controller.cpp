#include <ceres/ceres.h>
#include <rclcpp/logging.hpp>
#include <uniform_bspline/uniform_bspline.hpp>
#include <uniform_bspline_ceres/uniform_bspline_ceres_generator.hpp>
#include <path_follower/teb_controller.hpp>

namespace path_follower {

template <typename ValueType>
using SplineT = ubs::UniformBSpline<ValueType, 2, ValueType, Eigen::Matrix<ValueType, 2, 1>, std::vector<Eigen::Matrix<ValueType, 2, 1>>>;
using SplineD = ubs::UniformBSpline<double, 2, double, Eigen::Vector2d, std::vector<Eigen::Vector2d>>;

// --------- Jet 兼容的小工具：取标量部分用于索引 ---------
inline double scalar_value(double v) { return v; }

template <typename S, int N>
inline double scalar_value(const ceres::Jet<S, N>& v) { return static_cast<double>(v.a); }

// --------- cost map / direction map 的可微(近似)插值 ---------
// 说明：索引(x0,y0)使用标量部分做 floor，因此对格子边界处的导数并不严格，
//       但在这类离散地图代价里通常是可以接受的近似。

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

// --------- 将当前位置投影到样条参数u ---------
static double project_to_spline_u(
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

    double u_best = search(
        std::clamp(u_hint - search_window, 0.0, 1.0),
        std::clamp(u_hint + search_window, 0.0, 1.0),
        num_samples
    );

    if ((spline.evaluate(u_best) - pos).norm() <= max_correspondence_distance) {
        return std::clamp(u_best, 0.0, 1.0);
    }

    u_best = search(0.0, 1.0, num_samples);
    u_best = search(
        std::clamp(u_best - search_window, 0.0, 1.0),
        std::clamp(u_best + search_window, 0.0, 1.0),
        num_samples
    );

    if ((spline.evaluate(u_best) - pos).norm() <= max_correspondence_distance) {
        return std::clamp(u_best, 0.0, 1.0);
    }

    return -1.0;
}

// --------- Ceres cost functor：优化 (v,omega) 序列 ---------
struct TebCostFunctor {
    TebCostFunctor(
        ubs::UniformBSplineCeresGenerator<SplineT> generator,
        int num_control_points,
        Eigen::Vector3d start_pose,
        double u0,
        Eigen::Vector2d start_state,
        TebParams params,
        const CostMap* merged_cost_map,
        const DirectionMap* direction_map
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
        // + v_final(1) + soft_limits(2) + obstacle(1) + direction(1) = 12
        const int expected_residuals = 12 * params_.horizon;

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

            // Frenet 横向误差
            const T ex = x - pr.x();
            const T ey_world = y - pr.y();
            const T ey = -ex * sin(thetar) + ey_world * cos(thetar);
            T etheta = theta - thetar;
            etheta = atan2(sin(etheta), cos(etheta));

            // 贴合全局路径
            residuals[res_idx++] = T(std::sqrt(params_.q_y)) * ey;
            residuals[res_idx++] = T(std::sqrt(params_.q_theta)) * etheta;

            // 推进项（鼓励u增长）
            residuals[res_idx++] = T(std::sqrt(params_.q_u)) * (T(1.0) - u);

            // 控制正则
            residuals[res_idx++] = T(std::sqrt(params_.r_v)) * v;
            residuals[res_idx++] = T(std::sqrt(params_.r_omega)) * omega;

            // 控制平滑
            residuals[res_idx++] = T(std::sqrt(params_.r_dv)) * dv;
            residuals[res_idx++] = T(std::sqrt(params_.r_domega)) * domega;

            // 终点速度为0（u接近1时生效）
            const T terminal_weight = ceres::fmax(T(0.0), ceres::fmin(T(1.0), (u - T(0.999)) / T(0.001)));
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

            // 避障（使用合并代价地图：全局+局部）
            if (merged_cost_map_ && params_.obstacle_weight > 0.0) {
                const T c = interpolate_cost_map(*merged_cost_map_, x, y);
                // 归一化到 [0,1]，并进行惩罚
                residuals[res_idx++] = T(std::sqrt(params_.obstacle_weight)) * (c / T(255.0));
            } else {
                residuals[res_idx++] = T(0.0);
            }

            // 对齐台阶方向（方向地图在障碍附近/台阶上才有意义；无方向则不惩罚）
            if (direction_map_ && params_.direction_weight > 0.0) {
                const auto dir = interpolate_direction_map(*direction_map_, x, y);
                const T n = ceres::sqrt(dir.squaredNorm() + T(1e-9));
                const auto dirn = dir / n;
                const Eigen::Matrix<T, 2, 1> heading(cos(theta), sin(theta));
                const T dot = heading.dot(dirn);
                // 惩罚 1 - |cos|，即鼓励朝向与方向场对齐（允许正反向都对齐）
                residuals[res_idx++] = T(std::sqrt(params_.direction_weight)) * (T(1.0) - ceres::abs(dot));
            } else {
                residuals[res_idx++] = T(0.0);
            }

            // 进度动力学（将 ds/dt 转成 du/dt），保持与现有实现一致
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

        // 防御性兜底：确保所有 residual 都被写入，避免 Ceres 读取未初始化值。
        for (; res_idx < expected_residuals; ++res_idx) {
            residuals[res_idx] = T(0.0);
        }

        return true;
    }

    const ubs::UniformBSplineCeresGenerator<SplineT> generator_;
    const int num_control_points_;
    const Eigen::Vector3d start_pose_;
    const double u0_;
    const Eigen::Vector2d start_state_;
    const TebParams params_;
    const CostMap* merged_cost_map_;
    const DirectionMap* direction_map_;
};

}

namespace path_follower {

TebController::TebController(const TebParams& params) : params_(params) {
    last_controls_.assign(static_cast<size_t>(std::max(1, params_.horizon)), Eigen::Vector2d::Zero());
}

void TebController::set_reference_path(std::vector<Eigen::Vector2d> path_points_map) {
    ref_path_map_ = path_points_map;
    last_u_ = 0.0;
}

bool TebController::has_reference_path() const {
    return ref_path_map_.size() >= 3;
}

Eigen::Vector2d TebController::get_destination() const {
    if (ref_path_map_.size() < 3) return Eigen::Vector2d::Zero();
    SplineD spline(ref_path_map_);
    spline.setExtrapolate(true);
    return spline.evaluate(1.0);
}

TebController::Result TebController::solve(
    const Eigen::Vector3d& current_pose_map,
    const Eigen::Vector2d& current_state,
    const CostMap* merged_cost_map,
    const DirectionMap* global_direction_map
) {
    Result out;
    if (!has_reference_path()) {
        out.ok = false;
        out.cmd_v_omega.setZero();
        return out;
    }

    // 参考路径：使用 uniform_bspline 作为连续参考，便于计算切线与曲率
    SplineD spline(ref_path_map_);
    spline.setExtrapolate(true);

    const double u0 = project_to_spline_u(
        spline,
        current_pose_map.head<2>(),
        last_u_,
        params_.proj_num_samples,
        params_.proj_search_window,
        params_.max_correspondence_distance
    );

    if (u0 < 0.0) {
        RCLCPP_ERROR(rclcpp::get_logger("path_follower.teb_controller"), "无法将当前位置投影到全局参考路径上，停止输出。");
        out.ok = false;
        out.cmd_v_omega.setZero();
        return out;
    }
    last_u_ = u0;

    // 将参考路径点作为常量参数块，支持 Jet 下评估样条
    const int num_pts = static_cast<int>(ref_path_map_.size());
    std::vector<std::array<double, 2>> ref_point_blocks;
    ref_point_blocks.reserve(static_cast<size_t>(num_pts));
    for (const auto& p : ref_path_map_) {
        ref_point_blocks.push_back({p.x(), p.y()});
    }

    std::array<int, 1> shape{{num_pts}};
    ubs::UniformBSplineCeresGenerator<SplineT> generator(0.0, 1.0, shape);

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

    auto* cost_function = new ceres::DynamicAutoDiffCostFunction<TebCostFunctor>(
        new TebCostFunctor(
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

    // residual 总数：每步 12 个（见 TebCostFunctor::operator() 里的说明）
    cost_function->SetNumResiduals(12 * params_.horizon);

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
        problem.SetParameterLowerBound(controls[static_cast<size_t>(i)].data(), 0, params_.vel_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(i)].data(), 0, params_.vel_max);
        problem.SetParameterLowerBound(controls[static_cast<size_t>(i)].data(), 1, params_.omega_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(i)].data(), 1, params_.omega_max);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = params_.max_iterations;
    options.num_threads = params_.num_threads;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出：第一步控制
    out.cmd_v_omega = Eigen::Vector2d(controls[0][0], controls[0][1]);
    out.ok = summary.IsSolutionUsable();

    // 保存 warm start
    last_controls_.resize(static_cast<size_t>(params_.horizon));
    for (int i = 0; i < params_.horizon; i++) {
        last_controls_[static_cast<size_t>(i)] = Eigen::Vector2d(controls[static_cast<size_t>(i)][0], controls[static_cast<size_t>(i)][1]);
    }

    // 生成预测轨迹用于调试展示
    out.predicted_path_map.clear();
    out.predicted_path_map.reserve(static_cast<size_t>(params_.horizon + 1));
    Eigen::Vector3d pose = current_pose_map;
    out.predicted_path_map.push_back(pose.head<2>());
    for (int i = 0; i < params_.horizon; i++) {
        const double v = controls[static_cast<size_t>(i)][0];
        const double w = controls[static_cast<size_t>(i)][1];
        pose.x() += v * std::cos(pose.z()) * params_.dt;
        pose.y() += v * std::sin(pose.z()) * params_.dt;
        pose.z() += w * params_.dt;
        out.predicted_path_map.push_back(pose.head<2>());
    }

    return out;
}

}