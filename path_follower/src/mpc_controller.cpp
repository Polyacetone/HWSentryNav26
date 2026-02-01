#include <ceres/ceres.h>
#include <algorithm>
#include <uniform_bspline/uniform_bspline.hpp>
#include <uniform_bspline_ceres/uniform_bspline_ceres_generator.hpp>
#include <path_follower/mpc_controller.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

// ============================================================================
//                                  工具函数
// ============================================================================

inline double scalar_value(double v) {
    return v;
}

template<typename S, int N>
inline double scalar_value(const ceres::Jet<S, N>& v) {
    return static_cast<double>(v.a);
}

// ============================================================================
//                           LQR离散模型（10维状态）
// ============================================================================

namespace {
constexpr int LQR_STATE_SIZE = 10;
constexpr int LQR_INPUT_SIZE = 4;

constexpr int IDX_S = 0;
constexpr int IDX_DS = 1;
constexpr int IDX_PHI = 2;
constexpr int IDX_DPHI = 3;
constexpr int IDX_THETA_L_L = 4;
constexpr int IDX_DTHETA_L_L = 5;
constexpr int IDX_THETA_L_R = 6;
constexpr int IDX_DTHETA_L_R = 7;
constexpr int IDX_THETA_B = 8;
constexpr int IDX_DTHETA_B = 9;

/**
 * @brief LQR闭环步进函数（使用预计算的离散闭环矩阵，带子步迭代）
 * 
 * 离散闭环动力学（子步）：
 *   x_{k+1} = A_cl * x_k + B_ref * x_ref
 * 
 * 其中：
 *   - A_cl = exp((A - B*K) * dt_sub) 是离散闭环状态矩阵
 *   - B_ref 是离散参考输入矩阵
 *   - x_ref 是参考状态，其中 s, phi 跟踪当前值，ds, dphi 是指令速度
 *   - dt_sub = dt_mpc / substeps
 * 
 * 通过子步迭代，可以正确处理 s_ref, phi_ref 跟踪当前状态的效果
 */
template<typename T>
inline void lqr_closed_loop_step(
    const LQRModel& model,
    Eigen::Matrix<T, LQR_STATE_SIZE, 1>& x,
    const T& v_ref,
    const T& w_ref
) {
    // 子步迭代
    for (int i = 0; i < model.substeps; i++) {
        // 构建参考状态（每个子步都更新 s_ref 和 phi_ref）
        Eigen::Matrix<T, LQR_STATE_SIZE, 1> x_ref = Eigen::Matrix<T, LQR_STATE_SIZE, 1>::Zero();
        x_ref(IDX_S) = x(IDX_S);       // s_ref 跟踪当前位置
        x_ref(IDX_DS) = v_ref;          // ds_ref 是指令速度
        x_ref(IDX_PHI) = x(IDX_PHI);   // phi_ref 跟踪当前角度
        x_ref(IDX_DPHI) = w_ref;        // dphi_ref 是指令角速度

        // 使用预计算的离散闭环矩阵进行状态更新
        x = model.A_cl.cast<T>() * x + model.B_ref.cast<T>() * x_ref;
    }
}
}

// ============================================================================
//                         代价地图/方向地图插值（Jet兼容）
// ============================================================================

template<typename T>
inline T interpolate_cost_map(const CostMap& cost_map, const T& x_map, const T& y_map) {
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

template<typename T>
inline T smoothstep01(const T& t_in) {
    const T t = ceres::fmin(T(1.0), ceres::fmax(T(0.0), t_in));
    return t * t * (T(3.0) - T(2.0) * t);
}

template<typename T>
inline T smoothstep(const T& x, const T& edge0, const T& edge1) {
    const T denom = ceres::fmax(T(1e-12), edge1 - edge0);
    return smoothstep01((x - edge0) / denom);
}

template<typename T>
inline Eigen::Matrix<T, 2, 1> interpolate_direction_map(const DirectionMap& dir_map, const T& x_map, const T& y_map) {
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
        const Eigen::Vector2d v = dir_map.data[y * dir_map.width + x];
        return Eigen::Matrix<T, 2, 1>(T(v.x()), T(v.y()));
    };

    const auto v00 = at(x0, y0);
    const auto v10 = at(x1, y0);
    const auto v01 = at(x0, y1);
    const auto v11 = at(x1, y1);

    return (T(1.0) - dx) * (T(1.0) - dy) * v00 + dx * (T(1.0) - dy) * v10 + (T(1.0) - dx) * dy * v01 + dx * dy * v11;
}

template<typename T, typename Spline>
inline T estimate_remaining_arclength(const Spline& spline, const T& u_in, int num_samples) {
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

// ============================================================================
//                         Follow模式 Cost Functor
// ============================================================================

struct FollowMPCCostFunctor {
    FollowMPCCostFunctor(
        const ubs::UniformBSplineCeresGenerator<SplineT>& generator,
        const int num_control_points,
        const Eigen::Vector3d& start_pose,
        const double u0,
        const RobotStatus& start_status,
        const Eigen::Vector2d& start_cmd,
        const MPCParams& params,
        const CostMap& merged_cost_map,
        const DirectionMap& direction_map
    ):
        generator_(generator),
        num_control_points_(num_control_points),
        start_pose_(start_pose),
        u0_(u0),
        start_status_(start_status),
        start_cmd_(start_cmd),
        params_(params),
        merged_cost_map_(merged_cost_map),
        direction_map_(direction_map),
        lqr_(params.lqr) {}

    template<typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        // 初始位姿
        T x = T(start_pose_.x());
        T y = T(start_pose_.y());
        T theta = T(start_pose_.z());
        T u = T(u0_);

        // 初始动态状态
        T v_act = T(start_status_.v);
        T omega_act = T(start_status_.omega);

        Eigen::Matrix<T, LQR_STATE_SIZE, 1> x_state;
        x_state.setZero();
        x_state(IDX_DS) = T(start_status_.v);
        x_state(IDX_DPHI) = T(start_status_.omega);
        x_state(IDX_PHI) = T(start_pose_.z());

        // 上一时刻指令（用于平滑约束）
        T last_v_cmd = T(start_cmd_.x());
        T last_omega_cmd = T(start_cmd_.y());

        int res_idx = 0;

        // 生成样条
        const T* const* control_points_raw = parameters + params_.horizon;
        auto spline = generator_.template generate<T>(control_points_raw, true);

        for (int k = 0; k < params_.horizon; k++) {
            const T* uk = parameters[k];
            const T v_cmd = uk[0];
            const T omega_cmd = uk[1];

            // 指令变化
            const T dv_cmd = v_cmd - last_v_cmd;
            const T domega_cmd = omega_cmd - last_omega_cmd;

            // 保存上一时刻状态用于计算变化率
            const T v_act_prev = v_act;
            const T omega_act_prev = omega_act;

            lqr_closed_loop_step(lqr_, x_state, v_cmd, omega_cmd);
            const T v_now = x_state(IDX_DS);
            const T theta_now = x_state(IDX_PHI);
            x += v_now * ceres::cos(theta_now) * T(params_.dt);
            y += v_now * ceres::sin(theta_now) * T(params_.dt);
            theta = theta_now;
            v_act = v_now;
            omega_act = x_state(IDX_DPHI);

            // 实际状态变化（用于物理约束）
            const T dv_act = v_act - v_act_prev;
            const T domega_act = omega_act - omega_act_prev;

            // 样条上的参考点
            u = ceres::fmin(ceres::fmax(u, T(0.0)), T(1.0));
            const Eigen::Matrix<T, 2, 1> pr = spline.evaluate(u);
            const Eigen::Matrix<T, 2, 1> d1 = spline.derivative(u, 1);
            const Eigen::Matrix<T, 2, 1> d2 = spline.derivative(u, 2);

            const T dx = d1.x();
            const T dy = d1.y();
            const T ddx = d2.x();
            const T ddy = d2.y();

            const T thetar = ceres::atan2(dy, dx);
            const T dsdu = ceres::sqrt(dx * dx + dy * dy) + T(1e-6);
            const T kappa = (dx * ddy - dy * ddx) / (dsdu * dsdu * dsdu);

            const T s_remain = estimate_remaining_arclength(spline, u, params_.follow_limits.slow_down_num_samples);
            const T slow_dist = T(params_.follow_limits.slow_down_distance);

            // Frenet误差
            const T ex = x - pr.x();
            const T ey_world = y - pr.y();
            const T ey = -ex * ceres::sin(thetar) + ey_world * ceres::cos(thetar);
            T etheta = theta - thetar;
            etheta = ceres::atan2(ceres::sin(etheta), ceres::cos(etheta));

            // 1. 路径跟踪
            residuals[res_idx++] = T(params_.follow_weights.q_y) * ey;
            residuals[res_idx++] = T(params_.follow_weights.q_theta) * etheta;

            // 2. 进度推进
            // 使用剩余距离的平滑门控来调整进度权重，避免接近终点时过度推进
            const T progress_scale = smoothstep(s_remain, T(0.0), slow_dist);
            residuals[res_idx++] = T(params_.follow_weights.q_u) * progress_scale * (T(1.0) - u);

            // 3. 控制正则化
            residuals[res_idx++] = T(params_.follow_weights.r_v) * v_cmd;
            residuals[res_idx++] = T(params_.follow_weights.r_omega) * omega_cmd;

            // 4. 控制平滑
            residuals[res_idx++] = T(params_.follow_weights.r_dv) * dv_cmd;
            residuals[res_idx++] = T(params_.follow_weights.r_domega) * domega_cmd;

            // 5. 指令变化率软约束
            const T dv_cmd_limit = T(params_.follow_limits.acc_max * params_.dt);
            const T domega_cmd_limit = T(params_.follow_limits.alpha_max * params_.dt);
            residuals[res_idx++] = T(params_.follow_weights.acc_limit_weight) * ceres::fmax(T(0.0), ceres::abs(dv_cmd) - dv_cmd_limit);
            residuals[res_idx++] = T(params_.follow_weights.alpha_limit_weight) * ceres::fmax(T(0.0), ceres::abs(domega_cmd) - domega_cmd_limit);

            // 6. 物理加速度软约束
            const T dv_act_limit = T(params_.follow_limits.phys_acc_max * params_.dt);
            const T domega_act_limit = T(params_.follow_limits.phys_alpha_max * params_.dt);
            residuals[res_idx++] = T(params_.follow_weights.phys_acc_limit_weight) * ceres::fmax(T(0.0), ceres::abs(dv_act) - dv_act_limit);
            residuals[res_idx++] = T(params_.follow_weights.phys_alpha_limit_weight) * ceres::fmax(T(0.0), ceres::abs(domega_act) - domega_act_limit);

            // 7. 侧向加速度约束
            const T a_lat = ceres::abs(v_act * omega_act);
            residuals[res_idx++] = T(params_.follow_weights.lat_acc_weight) * ceres::fmax(T(0.0), a_lat - T(params_.follow_limits.a_lat_max));

            // 8. 避障
            const T cost = interpolate_cost_map(merged_cost_map_, x, y);
            residuals[res_idx++] = T(params_.follow_weights.obstacle_weight) * (cost / T(255.0));

            // 9. 台阶处理
            const auto dir = interpolate_direction_map(direction_map_, x, y);
            const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-9));
            const T step_gate = smoothstep(
                dir_norm,
                T(params_.follow_limits.step_norm_threshold),
                T(params_.follow_limits.step_norm_threshold + params_.follow_limits.step_norm_transition)
            );

            // 台阶惩罚使用接近门控，避免不必要地接近台阶
            residuals[res_idx++] = T(params_.follow_weights.step_weight) * step_gate;

            // 台阶方向对齐和台阶区域速度保持使用方向场模长门控
            const Eigen::Matrix<T, 2, 1> heading(ceres::cos(theta), ceres::sin(theta));
            const T heading_cross_dir = heading.x() * dir.y() - heading.y() * dir.x();
            residuals[res_idx++] = T(params_.follow_weights.direction_weight) * dir_norm * ceres::abs(heading_cross_dir);
            residuals[res_idx++] = T(params_.follow_weights.vel_on_step_weight) * dir_norm * ceres::abs(v_act - T(params_.follow_limits.vel_on_step));

            // 10. 终点减速
            const T gate_goal = ceres::fmin(T(1.0), ceres::fmax(T(0.0), (slow_dist - s_remain) / (slow_dist + T(1e-6))));
            const T deceleration = T(params_.follow_limits.acc_max); // 期望的减速加速度
            const T v_dec_profile = ceres::sqrt(T(2.0) * deceleration * s_remain + T(0.01)); // 基于物理的限速 (v^2 = 2 * a * s)
            residuals[res_idx++] = T(params_.follow_weights.q_v_final) * ceres::fmax(T(0.0), v_act - v_dec_profile); // 惩罚超速

            // 进度动力学
            T denom = T(1.0) - kappa * ey;
            const T denom_abs = ceres::abs(denom);
            denom = denom / (denom_abs + T(1e-6)) * ceres::fmax(denom_abs, T(0.1));
            const T dsdt = v_act * ceres::cos(etheta) / denom;
            const T dudt = dsdt / dsdu;
            u += dudt * T(params_.dt);
            u = ceres::fmin(ceres::fmax(u, T(0.0)), T(1.0));
            last_v_cmd = v_cmd;
            last_omega_cmd = omega_cmd;
        }

        return true;
    }

    const ubs::UniformBSplineCeresGenerator<SplineT>& generator_;
    const int num_control_points_;
    const Eigen::Vector3d& start_pose_;
    const double u0_;
    const RobotStatus& start_status_;
    const Eigen::Vector2d& start_cmd_;
    const MPCParams& params_;
    const CostMap& merged_cost_map_;
    const DirectionMap& direction_map_;

    const LQRModel& lqr_;
};

// ============================================================================
//                            Stop模式 Cost Functor
// ============================================================================

struct StopMPCCostFunctor {
    StopMPCCostFunctor(
        const Eigen::Vector3d& start_pose,
        const RobotStatus& start_status,
        const Eigen::Vector2d& start_cmd,
        const MPCParams& params,
        const CostMap& merged_cost_map,
        const DirectionMap& direction_map
    ):
        start_pose_(start_pose),
        start_status_(start_status),
        start_cmd_(start_cmd),
        params_(params),
        merged_cost_map_(merged_cost_map),
        direction_map_(direction_map),
        lqr_(params.lqr) {}

    template<typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        T x = T(start_pose_.x());
        T y = T(start_pose_.y());
        T theta = T(start_pose_.z());

        T v_act = T(start_status_.v);
        T omega_act = T(start_status_.omega);

        Eigen::Matrix<T, LQR_STATE_SIZE, 1> x_state;
        x_state.setZero();
        x_state(IDX_DS) = T(start_status_.v);
        x_state(IDX_DPHI) = T(start_status_.omega);
        x_state(IDX_PHI) = T(start_pose_.z());

        T last_v_cmd = T(start_cmd_.x());
        T last_omega_cmd = T(start_cmd_.y());

        int res_idx = 0;

        for (int k = 0; k < params_.horizon; k++) {
            const T* uk = parameters[k];
            const T v_cmd = uk[0];
            const T omega_cmd = uk[1];
            const T dv_cmd = v_cmd - last_v_cmd;
            const T domega_cmd = omega_cmd - last_omega_cmd;

            const T v_act_prev = v_act;
            const T omega_act_prev = omega_act;

            lqr_closed_loop_step(lqr_, x_state, v_cmd, omega_cmd);
            const T v_now = x_state(IDX_DS);
            const T theta_now = x_state(IDX_PHI);
            x += v_now * ceres::cos(theta_now) * T(params_.dt);
            y += v_now * ceres::sin(theta_now) * T(params_.dt);
            theta = theta_now;
            v_act = v_now;
            omega_act = x_state(IDX_DPHI);

            const T dv_act = v_act - v_act_prev;
            const T domega_act = omega_act - omega_act_prev;

            // 1. 速度正则化（希望停止）
            residuals[res_idx++] = T(params_.stop_weights.q_v) * v_act;
            residuals[res_idx++] = T(params_.stop_weights.q_omega) * omega_act;

            // 2. 指令变化率约束
            const T dv_cmd_limit = T(params_.stop_limits.acc_max * params_.dt);
            const T domega_cmd_limit = T(params_.stop_limits.alpha_max * params_.dt);
            residuals[res_idx++] = T(params_.stop_weights.acc_limit_weight) * ceres::fmax(T(0.0), ceres::abs(dv_cmd) - dv_cmd_limit);
            residuals[res_idx++] = T(params_.stop_weights.alpha_limit_weight) * ceres::fmax(T(0.0), ceres::abs(domega_cmd) - domega_cmd_limit);

            // 3. 物理加速度约束
            const T dv_act_limit = T(params_.stop_limits.phys_acc_max * params_.dt);
            const T domega_act_limit = T(params_.stop_limits.phys_alpha_max * params_.dt);
            residuals[res_idx++] = T(params_.stop_weights.phys_acc_limit_weight) * ceres::fmax(T(0.0), ceres::abs(dv_act) - dv_act_limit);
            residuals[res_idx++] = T(params_.stop_weights.phys_alpha_limit_weight) * ceres::fmax(T(0.0), ceres::abs(domega_act) - domega_act_limit);

            // 4. 侧向加速度约束
            const T a_lat = ceres::abs(v_act * omega_act);
            residuals[res_idx++] = T(params_.stop_weights.lat_acc_weight) * ceres::fmax(T(0.0), a_lat - T(params_.stop_limits.a_lat_max));

            // 5. 避障
            const T cost = interpolate_cost_map(merged_cost_map_, x, y);
            residuals[res_idx++] = T(params_.stop_weights.obstacle_weight) * (cost / T(255.0));

            // 6. 台阶处理
            const auto dir = interpolate_direction_map(direction_map_, x, y);
            const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-9));
            const T gate_step = ceres::fmin(T(1.0), dir_norm * T(2.0));
            const auto dir_normalized = dir / dir_norm;
            const Eigen::Matrix<T, 2, 1> heading(ceres::cos(theta), ceres::sin(theta));
            const T heading_dot_dir = heading.dot(dir_normalized);
            // 这里统一使用接近门控，避免停止模式下不必要地接近台阶
            residuals[res_idx++] = T(params_.stop_weights.vel_on_step_weight) * gate_step * ceres::abs(v_act - T(params_.stop_limits.vel_on_step));
            residuals[res_idx++] = T(params_.stop_weights.direction_weight) * gate_step * (T(1.0) - ceres::abs(heading_dot_dir));
            
            last_v_cmd = v_cmd;
            last_omega_cmd = omega_cmd;
        }

        // 终端约束
        const T cost_terminal = interpolate_cost_map(merged_cost_map_, x, y);
        residuals[res_idx++] = T(params_.stop_weights.obstacle_terminal_weight) * (cost_terminal / T(255.0));

        const auto dir_t = interpolate_direction_map(direction_map_, x, y);
        const T dir_t_norm = ceres::sqrt(dir_t.squaredNorm() + T(1e-9));
        const T gate_step_t = ceres::fmin(T(1.0), dir_t_norm * T(2.0));
        residuals[res_idx++] = T(params_.stop_weights.step_terminal_weight) * gate_step_t;

        return true;
    }

    const Eigen::Vector3d& start_pose_;
    const RobotStatus& start_status_;
    const Eigen::Vector2d& start_cmd_;
    const MPCParams& params_;
    const CostMap& merged_cost_map_;
    const DirectionMap& direction_map_;

    const LQRModel& lqr_;
};

// ============================================================================
//                            MPCController 实现
// ============================================================================

MPCController::MPCController(const MPCParams& params): params_(params) {
    last_controls_.assign(std::max(1, params_.horizon), Eigen::Vector2d::Zero());
}

std::expected<std::tuple<Eigen::Vector2d, std::vector<Eigen::Vector2d>>, std::string> MPCController::follow_path(
    const SplineD& global_path,
    const Eigen::Vector3d& current_pose_map,
    const RobotStatus& current_status,
    const CostMap& merged_cost_map,
    const DirectionMap& global_direction_map
) {
    // 投影到样条
    const double u0 = project_to_spline_u(
        global_path,
        current_pose_map.head<2>(),
        last_u_,
        params_.follow_projection.proj_num_samples,
        params_.follow_projection.proj_search_window,
        params_.follow_projection.max_correspondence_distance
    );
    last_u_ = u0;

    // 参考路径控制点
    const int num_pts = static_cast<int>(global_path.getControlPoints().size());
    std::vector<std::array<double, 2>> ref_point_blocks;
    ref_point_blocks.reserve(num_pts);
    for (const auto& p: global_path.getControlPoints()) {
        ref_point_blocks.push_back({p.x(), p.y()});
    }

    ubs::UniformBSplineCeresGenerator<SplineT> generator(0.0, 1.0, {num_pts});

    // 决策变量
    std::vector<std::vector<double>> controls(params_.horizon, std::vector<double>(2, 0.0));

    // Warm start
    if (last_controls_.size() == params_.horizon) {
        for (int i = 0; i < params_.horizon; i++) {
            const Eigen::Vector2d init = (i + 1 < params_.horizon) ? last_controls_[i + 1] : last_controls_.back();
            controls[i][0] = init.x();
            controls[i][1] = init.y();
        }
    } else {
        for (int i = 0; i < params_.horizon; i++) {
            controls[i][0] = current_status.v;
            controls[i][1] = current_status.omega;
        }
    }

    // 构建优化问题
    ceres::Problem problem;

    auto* cost_function = new ceres::DynamicAutoDiffCostFunction<FollowMPCCostFunctor>(new FollowMPCCostFunctor(
        generator,
        num_pts,
        current_pose_map,
        u0,
        current_status,
        last_cmd_,
        params_,
        merged_cost_map,
        global_direction_map
    ));

    // 添加参数块
    for (int i = 0; i < params_.horizon; i++) {
        cost_function->AddParameterBlock(2);
    }
    for (int i = 0; i < num_pts; i++) {
        cost_function->AddParameterBlock(2);
    }

    // 每步17个残差
    cost_function->SetNumResiduals(17 * params_.horizon);

    std::vector<double*> parameter_blocks;
    parameter_blocks.reserve(params_.horizon + num_pts);

    for (auto& c: controls) {
        parameter_blocks.push_back(c.data());
    }

    for (auto& rp: ref_point_blocks) {
        problem.AddParameterBlock(rp.data(), 2);
        problem.SetParameterBlockConstant(rp.data());
        parameter_blocks.push_back(rp.data());
    }

    problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);

    // 设置边界
    for (int i = 0; i < params_.horizon; i++) {
        problem.SetParameterLowerBound(controls[i].data(), 0, params_.follow_limits.vel_min);
        problem.SetParameterUpperBound(controls[i].data(), 0, params_.follow_limits.vel_max);
        problem.SetParameterLowerBound(controls[i].data(), 1, params_.follow_limits.omega_min);
        problem.SetParameterUpperBound(controls[i].data(), 1, params_.follow_limits.omega_max);
    }

    // 求解
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

    // 输出
    Eigen::Vector2d cmd_v_omega(controls[0][0], controls[0][1]);
    last_cmd_ = cmd_v_omega;

    // 保存warm start
    last_controls_.resize(params_.horizon);
    for (int i = 0; i < params_.horizon; i++) {
        last_controls_[i] = Eigen::Vector2d(controls[i][0], controls[i][1]);
    }

    // 生成预测轨迹（使用固定LQR离散模型）
    std::vector<Eigen::Vector2d> predicted_path_map;
    predicted_path_map.reserve(params_.horizon + 1);

    Eigen::Vector3d pose = current_pose_map;
    predicted_path_map.push_back(pose.head<2>());

    Eigen::Matrix<double, LQR_STATE_SIZE, 1> x_state = Eigen::Matrix<double, LQR_STATE_SIZE, 1>::Zero();
    x_state(IDX_DS) = current_status.v;
    x_state(IDX_DPHI) = current_status.omega;
    x_state(IDX_PHI) = pose.z();

    for (int i = 0; i < params_.horizon; i++) {
        const double v_cmd = controls[i][0];
        const double w_cmd = controls[i][1];

        lqr_closed_loop_step(params_.lqr, x_state, v_cmd, w_cmd);
        const double v_now = x_state(IDX_DS);
        const double theta_now = x_state(IDX_PHI);
        pose.x() += v_now * std::cos(theta_now) * params_.dt;
        pose.y() += v_now * std::sin(theta_now) * params_.dt;
        pose.z() = theta_now;

        predicted_path_map.push_back(pose.head<2>());
    }

    return std::tuple {cmd_v_omega, predicted_path_map};
}

std::expected<std::tuple<Eigen::Vector2d, std::vector<Eigen::Vector2d>>, std::string> MPCController::stop(
    const Eigen::Vector3d& current_pose_map,
    const RobotStatus& current_status,
    const CostMap& merged_cost_map,
    const DirectionMap& global_direction_map
) {
    // 决策变量
    std::vector<std::vector<double>> controls(params_.horizon, std::vector<double>(2, 0.0));

    // Warm start
    if (last_controls_.size() == params_.horizon) {
        for (int i = 0; i < params_.horizon; i++) {
            const Eigen::Vector2d init = (i + 1 < params_.horizon) ? last_controls_[i + 1] : last_controls_.back();
            controls[i][0] = std::max(0.0, init.x());
            controls[i][1] = init.y();
        }
    } else {
        for (int i = 0; i < params_.horizon; i++) {
            controls[i][0] = std::max(0.0, current_status.v);
            controls[i][1] = current_status.omega;
        }
    }

    ceres::Problem problem;

    auto* cost_function = new ceres::DynamicAutoDiffCostFunction<StopMPCCostFunctor>(new StopMPCCostFunctor(
        current_pose_map,
        current_status,
        last_cmd_,
        params_,
        merged_cost_map,
        global_direction_map
    ));

    for (int i = 0; i < params_.horizon; i++) {
        cost_function->AddParameterBlock(2);
    }

    // 每步10个残差 + 终端2个
    cost_function->SetNumResiduals(10 * params_.horizon + 2);

    std::vector<double*> parameter_blocks;
    parameter_blocks.reserve(params_.horizon);
    for (auto& c: controls) {
        parameter_blocks.push_back(c.data());
    }
    problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);

    // 边界
    for (int i = 0; i < params_.horizon; i++) {
        problem.SetParameterLowerBound(controls[i].data(), 0, 0.0);
        problem.SetParameterUpperBound(controls[i].data(), 0, params_.stop_limits.vel_max);
        problem.SetParameterLowerBound(controls[i].data(), 1, params_.stop_limits.omega_min);
        problem.SetParameterUpperBound(controls[i].data(), 1, params_.stop_limits.omega_max);
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
    last_cmd_ = cmd_v_omega;

    // 保存warm start
    last_controls_.resize(params_.horizon);
    for (int i = 0; i < params_.horizon; i++) {
        last_controls_[i] = Eigen::Vector2d(controls[i][0], controls[i][1]);
    }

    // 预测轨迹（使用固定LQR离散模型）
    std::vector<Eigen::Vector2d> predicted_path_map;
    predicted_path_map.reserve(params_.horizon + 1);

    Eigen::Vector3d pose = current_pose_map;
    predicted_path_map.push_back(pose.head<2>());

    Eigen::Matrix<double, LQR_STATE_SIZE, 1> x_state = Eigen::Matrix<double, LQR_STATE_SIZE, 1>::Zero();
    x_state(IDX_DS) = current_status.v;
    x_state(IDX_DPHI) = current_status.omega;
    x_state(IDX_PHI) = pose.z();

    for (int i = 0; i < params_.horizon; i++) {
        const double v_cmd = controls[i][0];
        const double w_cmd = controls[i][1];

        lqr_closed_loop_step(params_.lqr, x_state, v_cmd, w_cmd);
        const double v_now = x_state(IDX_DS);
        const double theta_now = x_state(IDX_PHI);
        pose.x() += v_now * std::cos(theta_now) * params_.dt;
        pose.y() += v_now * std::sin(theta_now) * params_.dt;
        pose.z() = theta_now;

        predicted_path_map.push_back(pose.head<2>());
    }

    return std::tuple {cmd_v_omega, predicted_path_map};
}

}