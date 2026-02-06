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

inline int clamp_int(const int v, const int lo, const int hi) {
    return std::max(lo, std::min(v, hi));
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

template <typename T>
inline void eval_quadratic_uniform_bspline_2d(
    const std::vector<Eigen::Vector2d>& control_points,
    const T& u_in,
    Eigen::Matrix<T, 2, 1>* p,
    Eigen::Matrix<T, 2, 1>* d1,
    Eigen::Matrix<T, 2, 1>* d2
) {
    const int n = static_cast<int>(control_points.size());
    if (n < 3) {
        if (p) {
            *p = Eigen::Matrix<T, 2, 1>(T(0.0), T(0.0));
        }
        if (d1) {
            *d1 = Eigen::Matrix<T, 2, 1>(T(0.0), T(0.0));
        }
        if (d2) {
            *d2 = Eigen::Matrix<T, 2, 1>(T(0.0), T(0.0));
        }
        return;
    }

    // 对应 ubs::ControlPointsContainer::updatedShape()：scale = (n - Degree) / (upper - lower)
    // 这里 Degree=2, bounds=[0,1]，所以 scale = n - 2。
    const double scale_d = static_cast<double>(n - 2);

    const T u = ceres::fmin(ceres::fmax(u_in, T(0.0)), T(1.0));
    const T base_x = u * T(scale_d);

    const int xi_unclamped = static_cast<int>(std::floor(scalar_value(base_x)));
    const int xi = clamp_int(xi_unclamped, 0, n - 3);
    const T t = base_x - T(xi);

    const T one_minus_t = T(1.0) - t;
    const T b0 = T(0.5) * one_minus_t * one_minus_t;
    const T b1 = T(0.5) * (T(-2.0) * t * t + T(2.0) * t + T(1.0));
    const T b2 = T(0.5) * t * t;

    const T db0_dt = -one_minus_t;
    const T db1_dt = T(-2.0) * t + T(1.0);
    const T db2_dt = t;

    const Eigen::Vector2d& p0d = control_points[static_cast<size_t>(xi + 0)];
    const Eigen::Vector2d& p1d = control_points[static_cast<size_t>(xi + 1)];
    const Eigen::Vector2d& p2d = control_points[static_cast<size_t>(xi + 2)];

    const Eigen::Matrix<T, 2, 1> p0(T(p0d.x()), T(p0d.y()));
    const Eigen::Matrix<T, 2, 1> p1(T(p1d.x()), T(p1d.y()));
    const Eigen::Matrix<T, 2, 1> p2(T(p2d.x()), T(p2d.y()));

    if (p) {
        *p = b0 * p0 + b1 * p1 + b2 * p2;
    }

    // 一阶/二阶导数：先对局部参数 t 求导，再乘以 dt/du。
    // base_x = u * scale, t = base_x - xi，所以 dt/du = scale（xi 对 u 的导数为 0）。
    const T scale = T(scale_d);

    if (d1) {
        const Eigen::Matrix<T, 2, 1> dp_dt = db0_dt * p0 + db1_dt * p1 + db2_dt * p2;
        *d1 = dp_dt * scale;
    }

    if (d2) {
        // 二阶基函数对 t 的导数常数：b0''=1, b1''=-2, b2''=1
        const Eigen::Matrix<T, 2, 1> d2p_dt2 = p0 - T(2.0) * p1 + p2;
        *d2 = d2p_dt2 * (scale * scale);
    }
}

inline double estimate_remaining_arclength(
    const std::vector<Eigen::Vector2d>& control_points,
    const double u_in,
    const int num_samples
) {
    const double u = std::min(1.0, std::max(0.0, u_in));
    const double one_minus_u = 1.0 - u;

    auto dsdu_at = [&](double uu) {
        Eigen::Matrix<double, 2, 1> d1;
        eval_quadratic_uniform_bspline_2d<double>(control_points, uu, nullptr, &d1, nullptr);
        return std::sqrt(d1.squaredNorm() + 1e-12);
    };

    if (num_samples <= 1) {
        return one_minus_u * dsdu_at(u);
    }

    double length = 0.0;
    double u_prev = u;
    double dsdu_prev = dsdu_at(u_prev);

    for (int i = 1; i <= num_samples; i++) {
        const double ui = u + one_minus_u * (static_cast<double>(i) / static_cast<double>(num_samples));
        const double dsdu = dsdu_at(ui);
        const double du = ui - u_prev;
        length += (dsdu_prev + dsdu) * 0.5 * du;
        u_prev = ui;
        dsdu_prev = dsdu;
    }

    return length;
}

// ============================================================================
//                                 MPC预测模型
// ============================================================================

namespace {
constexpr int LQR_STATE_SIZE = 10;
constexpr int IDX_S = 0;
constexpr int IDX_DS = 1;
constexpr int IDX_PHI = 2;
constexpr int IDX_DPHI = 3;

template<typename T>
inline T wrap_to_pi(const T& a) {
    return ceres::atan2(ceres::sin(a), ceres::cos(a));
}

template<typename T>
struct PredictorState {
    T x, y, theta;
    T v_act, omega_act;

    // LAG模式专用
    T a_act;

    // LQR模式专用
    Eigen::Matrix<T, LQR_STATE_SIZE, 1> x_lqr;
};

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
    const T& w_ref,
    const T& theta_ref
) {
    // 子步迭代
    for (int i = 0; i < model.substeps; i++) {
        // 构建参考状态（每个子步都更新 s_ref 和 phi_ref）
        Eigen::Matrix<T, LQR_STATE_SIZE, 1> x_ref = Eigen::Matrix<T, LQR_STATE_SIZE, 1>::Zero();
        x_ref(IDX_S) = x(IDX_S);       // s_ref 跟踪当前位置
        x_ref(IDX_DS) = v_ref;         // ds_ref 是指令速度
        // phi_ref：使用理想指令外推得到的下一时刻角度作为参考，并做最短角差处理避免跨 ±pi 跳变
        x_ref(IDX_PHI) = x(IDX_PHI) + wrap_to_pi(theta_ref - x(IDX_PHI));
        x_ref(IDX_DPHI) = w_ref;       // dphi_ref 是指令角速度

        // 使用预计算的离散闭环矩阵进行状态更新
        x = model.A_cl.cast<T>() * x + model.B_ref.cast<T>() * x_ref;
    }
}

template<typename T>
inline void prediction_step(
    const MPCParams& params,
    PredictorState<T>& st,
    const T& v_cmd,
    const T& omega_cmd,
    T& theta_cmd
) {
    const T dt = T(params.dt);
    const T theta_cmd_next = wrap_to_pi(theta_cmd + omega_cmd * dt);
    switch (params.prediction_model) {
        case PredictionModel::NONE: {
            st.v_act = v_cmd;
            st.omega_act = omega_cmd;
            st.theta = wrap_to_pi(st.theta + st.omega_act * dt);
            st.x += st.v_act * ceres::cos(st.theta) * dt;
            st.y += st.v_act * ceres::sin(st.theta) * dt;
            theta_cmd = theta_cmd_next;
            break;
        }
        case PredictionModel::LAG: {
            const T tau_v = T(params.lag.tau_v);
            const T k_v = T(params.lag.k_v);
            const T tau_omega = T(params.lag.tau_omega);

            const T a_dot = (k_v * (v_cmd - st.v_act) - st.a_act) / tau_v;
            st.a_act = st.a_act + a_dot * dt;
            st.v_act = st.v_act + st.a_act * dt;

            const T omega_dot = (omega_cmd - st.omega_act) / tau_omega;
            st.omega_act = st.omega_act + omega_dot * dt;

            st.theta = wrap_to_pi(st.theta + st.omega_act * dt);
            st.x += st.v_act * ceres::cos(st.theta) * dt;
            st.y += st.v_act * ceres::sin(st.theta) * dt;
            theta_cmd = theta_cmd_next;
            break;
        }
        case PredictionModel::LQR: {
            lqr_closed_loop_step(params.lqr, st.x_lqr, v_cmd, omega_cmd, theta_cmd_next);
            st.v_act = st.x_lqr(IDX_DS);
            st.omega_act = st.x_lqr(IDX_DPHI);
            st.theta = st.x_lqr(IDX_PHI);

            st.x += st.v_act * ceres::cos(st.theta) * dt;
            st.y += st.v_act * ceres::sin(st.theta) * dt;
            theta_cmd = theta_cmd_next;
            break;
        }
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
        const Eigen::Vector2d v = dir_map.data[static_cast<size_t>(y * dir_map.width + x)];
        return Eigen::Matrix<T, 2, 1>(T(v.x()), T(v.y()));
    };

    const auto v00 = at(x0, y0);
    const auto v10 = at(x1, y0);
    const auto v01 = at(x0, y1);
    const auto v11 = at(x1, y1);

    return (T(1.0) - dx) * (T(1.0) - dy) * v00 + dx * (T(1.0) - dy) * v10 + (T(1.0) - dx) * dy * v01 + dx * dy * v11;
}

// ============================================================================
//                         Follow 模式 Cost Functor
// ============================================================================

struct FollowMPCCostFunctor {
    FollowMPCCostFunctor(
        const std::vector<Eigen::Vector2d>& ref_control_points,
        const Eigen::Vector3d& start_pose,
        const double u0,
        const Eigen::Vector2d& start_status,
        const Eigen::Vector2d& start_cmd,
        const MPCParams& params,
        const CostMap& merged_cost_map,
        const DirectionMap& direction_map
    ):
        ref_control_points_(ref_control_points),
        start_pose_(start_pose),
        u0_(u0),
        start_status_(start_status),
        start_cmd_(start_cmd),
        params_(params),
        merged_cost_map_(merged_cost_map),
        direction_map_(direction_map) {}

    template<typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        PredictorState<T> st;
        st.x = T(start_pose_.x());
        st.y = T(start_pose_.y());
        st.theta = T(start_pose_.z());
        T u = T(u0_);
        T theta_cmd = T(start_pose_.z());
        st.v_act = T(start_status_.x());
        st.omega_act = T(start_status_.y());
        st.a_act = T(0.0);
        st.x_lqr.setZero();
        st.x_lqr(IDX_DS) = T(start_status_.x());
        st.x_lqr(IDX_DPHI) = T(start_status_.y());
        st.x_lqr(IDX_PHI) = T(start_pose_.z());

        // 上一时刻指令（经过起始指令与实际状态差值限幅），用于平滑约束
        T last_v_cmd = ceres::fmax(
            ceres::fmin(T(start_cmd_.x()), T(start_status_.x() + params_.follow_limits.start_vel_cmd_act_diff_max - params_.follow_limits.acc_max * params_.dt)),
            T(start_status_.x() - params_.follow_limits.start_vel_cmd_act_diff_max + params_.follow_limits.acc_max * params_.dt)
        );
        T last_omega_cmd = ceres::fmax(
            ceres::fmin(T(start_cmd_.y()), T(start_status_.y() + params_.follow_limits.start_omega_cmd_act_diff_max - params_.follow_limits.alpha_max * params_.dt)),
            T(start_status_.y() - params_.follow_limits.start_omega_cmd_act_diff_max + params_.follow_limits.alpha_max * params_.dt)
        );

        size_t res_idx = 0;
        for (int k = 0; k < params_.horizon; k++) {
            const T* uk = parameters[k];
            const T v_cmd = uk[0];
            const T omega_cmd = uk[1];

            // 指令变化
            const T dv_cmd = v_cmd - last_v_cmd;
            const T domega_cmd = omega_cmd - last_omega_cmd;

            prediction_step(params_, st, v_cmd, omega_cmd, theta_cmd);

            // 样条上的参考点
            u = ceres::fmin(ceres::fmax(u, T(0.0)), T(1.0));
            Eigen::Matrix<T, 2, 1> pr;
            Eigen::Matrix<T, 2, 1> d1;
            Eigen::Matrix<T, 2, 1> d2;
            eval_quadratic_uniform_bspline_2d<T>(ref_control_points_, u, &pr, &d1, &d2);

            const T dx = d1.x();
            const T dy = d1.y();
            const T ddx = d2.x();
            const T ddy = d2.y();

            const T thetar = ceres::atan2(dy, dx);
            const T dsdu = ceres::sqrt(dx * dx + dy * dy) + T(1e-6);
            const T kappa = (dx * ddy - dy * ddx) / (dsdu * dsdu * dsdu);

            // 终点剩余里程：只需要数值用于限速，不需要其导数。用 double 做数值积分，并将结果提升为 T，显式阻断 AutoDiff 的巨大循环计算图。
            const double s_remain_d = estimate_remaining_arclength(
                ref_control_points_,
                scalar_value(u),
                params_.follow_limits.slow_down_num_samples
            );
            const T s_remain = T(s_remain_d);

            // Frenet误差
            const T ex = st.x - pr.x();
            const T ey_world = st.y - pr.y();
            const T ey = -ex * ceres::sin(thetar) + ey_world * ceres::cos(thetar);
            T etheta = st.theta - thetar;
            etheta = ceres::atan2(ceres::sin(etheta), ceres::cos(etheta));

            // 1. 路径跟踪
            residuals[res_idx++] = T(params_.follow_weights.q_y) * ey;
            residuals[res_idx++] = T(params_.follow_weights.q_theta) * etheta;

            // 2. 进度推进
            residuals[res_idx++] = T(params_.follow_weights.q_u) * (T(1.0) - u);

            // 3. 控制正则化
            residuals[res_idx++] = T(params_.follow_weights.r_v) * v_cmd;
            residuals[res_idx++] = T(params_.follow_weights.r_omega) * omega_cmd;

            // 4. 控制平滑
            residuals[res_idx++] = T(params_.follow_weights.r_dv) * dv_cmd;
            residuals[res_idx++] = T(params_.follow_weights.r_domega) * domega_cmd;

            // 5. 指令变化率软约束
            const T dv_cmd_limit = T(params_.follow_limits.acc_max * params_.dt);
            const T domega_cmd_limit = T(params_.follow_limits.alpha_max * params_.dt);
            residuals[res_idx++] = T(params_.follow_weights.acc_limit) * ceres::fmax(T(0.0), ceres::abs(dv_cmd) - dv_cmd_limit);
            residuals[res_idx++] = T(params_.follow_weights.alpha_limit) * ceres::fmax(T(0.0), ceres::abs(domega_cmd) - domega_cmd_limit);

            // 6. 侧向加速度约束
            const T a_lat = ceres::abs(st.v_act * st.omega_act);
            residuals[res_idx++] = T(params_.follow_weights.lat_acc) * ceres::fmax(T(0.0), a_lat - T(params_.follow_limits.a_lat_max));

            // 7. 避障
            const T cost = interpolate_cost_map(merged_cost_map_, st.x, st.y);
            residuals[res_idx++] = T(params_.follow_weights.obstacle) * (cost / T(255.0));

            // 8. 台阶处理
            const auto dir = interpolate_direction_map(direction_map_, st.x, st.y);
            const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-10));
            const auto dir_unit = dir / dir_norm;

            // 台阶惩罚使用接近门控，避免不必要地接近台阶
            const T step_gate = smoothstep(
                dir_norm,
                T(params_.follow_limits.step_norm_threshold),
                T(params_.follow_limits.step_norm_threshold + params_.follow_limits.step_norm_transition)
            );
            residuals[res_idx++] = T(params_.follow_weights.step) * step_gate;

            // 台阶方向对齐
            const Eigen::Matrix<T, 2, 1> heading(ceres::cos(st.theta), ceres::sin(st.theta));
            const T heading_cross_dir = heading.x() * dir.y() - heading.y() * dir.x();
            residuals[res_idx++] = T(params_.follow_weights.direction) * ceres::abs(heading_cross_dir);

            // 台阶区域速度保持
            const T cos_theta = heading.dot(dir_unit);
            const T weight_up = (cos_theta + T(1.0)) / T(2.0); // weight_up 为 1 时表示完全上坡，为 0 时表示完全下坡
            const T target_vel_step = weight_up * T(params_.follow_limits.vel_step_up) + (T(1.0) - weight_up) * T(params_.follow_limits.vel_step_down);
            residuals[res_idx++] = T(params_.follow_weights.vel_on_step) * dir_norm * ceres::abs(st.v_act - target_vel_step);

            // 9. 终点减速
            const T deceleration = T(params_.follow_limits.slow_down_deceleration); // 期望的减速加速度
            const T v_dec_profile = ceres::sqrt(T(2.0) * deceleration * s_remain + T(0.01)); // 基于物理的限速 (v^2 = 2 * a * s)
            residuals[res_idx++] = T(params_.follow_weights.q_v_final) * ceres::fmax(T(0.0), st.v_act - v_dec_profile); // 惩罚超速

            // 进度动力学
            T denom = T(1.0) - kappa * ey;
            const T denom_abs = ceres::abs(denom);
            denom = denom / (denom_abs + T(1e-6)) * ceres::fmax(denom_abs, T(0.1));
            const T dsdt = st.v_act * ceres::cos(etheta) / denom;
            const T dudt = dsdt / dsdu;
            u += dudt * T(params_.dt);
            u = ceres::fmin(ceres::fmax(u, T(0.0)), T(1.0));
            last_v_cmd = v_cmd;
            last_omega_cmd = omega_cmd;
        }

        return true;
    }

    const std::vector<Eigen::Vector2d>& ref_control_points_;
    const Eigen::Vector3d& start_pose_;
    const double u0_;
    const Eigen::Vector2d& start_status_;
    const Eigen::Vector2d& start_cmd_;
    const MPCParams& params_;
    const CostMap& merged_cost_map_;
    const DirectionMap& direction_map_;
};

// ============================================================================
//                            Stop 模式 Cost Functor
// ============================================================================

struct StopMPCCostFunctor {
    StopMPCCostFunctor(
        const Eigen::Vector3d& start_pose,
        const Eigen::Vector2d& start_status,
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
        direction_map_(direction_map) {}

    template<typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        PredictorState<T> st;
        st.x = T(start_pose_.x());
        st.y = T(start_pose_.y());
        st.theta = T(start_pose_.z());
        T theta_cmd = T(start_pose_.z());
        st.v_act = T(start_status_.x());
        st.omega_act = T(start_status_.y());
        st.a_act = T(0.0);
        st.x_lqr.setZero();
        st.x_lqr(IDX_DS) = T(start_status_.x());
        st.x_lqr(IDX_DPHI) = T(start_status_.y());
        st.x_lqr(IDX_PHI) = T(start_pose_.z());

        T last_v_cmd = ceres::fmax(
            ceres::fmin(T(start_cmd_.x()), T(start_status_.x() + params_.stop_limits.start_vel_cmd_act_diff_max - params_.stop_limits.acc_max * params_.dt)),
            T(start_status_.x() - params_.stop_limits.start_vel_cmd_act_diff_max + params_.stop_limits.acc_max * params_.dt)
        );
        T last_omega_cmd = ceres::fmax(
            ceres::fmin(T(start_cmd_.y()), T(start_status_.y() + params_.stop_limits.start_omega_cmd_act_diff_max - params_.stop_limits.alpha_max * params_.dt)),
            T(start_status_.y() - params_.stop_limits.start_omega_cmd_act_diff_max + params_.stop_limits.alpha_max * params_.dt)
        );

        size_t res_idx = 0;
        for (int k = 0; k < params_.horizon; k++) {
            const T* uk = parameters[k];
            const T v_cmd = uk[0];
            const T omega_cmd = uk[1];
            const T dv_cmd = v_cmd - last_v_cmd;
            const T domega_cmd = omega_cmd - last_omega_cmd;

            prediction_step(params_, st, v_cmd, omega_cmd, theta_cmd);

            // 1. 速度正则化（希望停止）
            residuals[res_idx++] = T(params_.stop_weights.q_v) * st.v_act;
            residuals[res_idx++] = T(params_.stop_weights.q_omega) * st.omega_act;

            // 2. 指令变化率约束
            const T dv_cmd_limit = T(params_.stop_limits.acc_max * params_.dt);
            const T domega_cmd_limit = T(params_.stop_limits.alpha_max * params_.dt);
            residuals[res_idx++] = T(params_.stop_weights.acc_limit) * ceres::fmax(T(0.0), ceres::abs(dv_cmd) - dv_cmd_limit);
            residuals[res_idx++] = T(params_.stop_weights.alpha_limit) * ceres::fmax(T(0.0), ceres::abs(domega_cmd) - domega_cmd_limit);

            // 3. 侧向加速度约束
            const T a_lat = ceres::abs(st.v_act * st.omega_act);
            residuals[res_idx++] = T(params_.stop_weights.lat_acc) * ceres::fmax(T(0.0), a_lat - T(params_.stop_limits.a_lat_max));

            // 4. 避障
            const T cost = interpolate_cost_map(merged_cost_map_, st.x, st.y);
            residuals[res_idx++] = T(params_.stop_weights.obstacle) * (cost / T(255.0));

            // 5. 台阶处理
            const auto dir = interpolate_direction_map(direction_map_, st.x, st.y);
            const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-10));
            const auto dir_unit = dir / dir_norm;

            // 台阶方向对齐
            const Eigen::Matrix<T, 2, 1> heading(ceres::cos(st.theta), ceres::sin(st.theta));
            const T heading_cross_dir = heading.x() * dir.y() - heading.y() * dir.x();
            residuals[res_idx++] = T(params_.follow_weights.direction) * ceres::abs(heading_cross_dir);

            // 台阶区域速度保持
            const T cos_theta = heading.dot(dir_unit);
            const T weight_up = (cos_theta + T(1.0)) / T(2.0); // weight_up 为 1 时表示完全上坡，为 0 时表示完全下坡
            const T target_vel_step = weight_up * T(params_.follow_limits.vel_step_up) + (T(1.0) - weight_up) * T(params_.follow_limits.vel_step_down);
            residuals[res_idx++] = T(params_.follow_weights.vel_on_step) * dir_norm * ceres::abs(st.v_act - target_vel_step);
            
            last_v_cmd = v_cmd;
            last_omega_cmd = omega_cmd;
        }

        // 终端约束：不碰撞且不在台阶区域
        const T cost_terminal = interpolate_cost_map(merged_cost_map_, st.x, st.y);
        residuals[res_idx++] = T(params_.stop_weights.obstacle_terminal) * (cost_terminal / T(255.0));
        const auto dir = interpolate_direction_map(direction_map_, st.x, st.y);
        const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-10));
        const T step_gate = smoothstep(
            dir_norm,
            T(params_.follow_limits.step_norm_threshold),
            T(params_.follow_limits.step_norm_threshold + params_.follow_limits.step_norm_transition)
        );
        residuals[res_idx++] = T(params_.stop_weights.step_terminal) * step_gate;

        return true;
    }

    const Eigen::Vector3d& start_pose_;
    const Eigen::Vector2d& start_status_;
    const Eigen::Vector2d& start_cmd_;
    const MPCParams& params_;
    const CostMap& merged_cost_map_;
    const DirectionMap& direction_map_;
};

// ============================================================================
//                       Recovery-To-Point 模式 Cost Functor
// ============================================================================

struct RecoveryMPCCostFunctor {
    RecoveryMPCCostFunctor(
        const Eigen::Vector2d& goal_map,
        const Eigen::Vector3d& start_pose,
        const Eigen::Vector2d& start_status,
        const Eigen::Vector2d& start_cmd,
        const MPCParams& params,
        const CostMap& merged_cost_map,
        const DirectionMap& direction_map
    ):
        goal_map_(goal_map),
        start_pose_(start_pose),
        start_status_(start_status),
        start_cmd_(start_cmd),
        params_(params),
        merged_cost_map_(merged_cost_map),
        direction_map_(direction_map) {}

    template<typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        PredictorState<T> st;
        st.x = T(start_pose_.x());
        st.y = T(start_pose_.y());
        st.theta = T(start_pose_.z());
        T theta_cmd = T(start_pose_.z());
        st.v_act = T(start_status_.x());
        st.omega_act = T(start_status_.y());
        st.a_act = T(0.0);
        st.x_lqr.setZero();
        st.x_lqr(IDX_DS) = T(start_status_.x());
        st.x_lqr(IDX_DPHI) = T(start_status_.y());
        st.x_lqr(IDX_PHI) = T(start_pose_.z());

        T last_v_cmd = ceres::fmax(
            ceres::fmin(T(start_cmd_.x()), T(start_status_.x() + params_.recovery_limits.start_vel_cmd_act_diff_max - params_.recovery_limits.acc_max * params_.dt)),
            T(start_status_.x() - params_.recovery_limits.start_vel_cmd_act_diff_max + params_.recovery_limits.acc_max * params_.dt)
        );
        T last_omega_cmd = ceres::fmax(
            ceres::fmin(T(start_cmd_.y()), T(start_status_.y() + params_.recovery_limits.start_omega_cmd_act_diff_max - params_.recovery_limits.alpha_max * params_.dt)),
            T(start_status_.y() - params_.recovery_limits.start_omega_cmd_act_diff_max + params_.recovery_limits.alpha_max * params_.dt)
        );

        const T gx = T(goal_map_.x());
        const T gy = T(goal_map_.y());

        size_t res_idx = 0;
        for (int k = 0; k < params_.horizon; k++) {
            const T* uk = parameters[k];
            const T v_cmd = uk[0];
            const T omega_cmd = uk[1];
            const T dv_cmd = v_cmd - last_v_cmd;
            const T domega_cmd = omega_cmd - last_omega_cmd;

            prediction_step(params_, st, v_cmd, omega_cmd, theta_cmd);

            // 1. 目标点吸引
            const T dx = st.x - gx;
            const T dy = st.y - gy;
            residuals[res_idx++] = T(params_.recovery_weights.q_goal_xy) * dx;
            residuals[res_idx++] = T(params_.recovery_weights.q_goal_xy) * dy;

            // 2. 朝向目标（不要求台阶对齐）
            const T theta_goal = ceres::atan2(gy - st.y, gx - st.x);
            T etheta = wrap_to_pi(st.theta - theta_goal);
            residuals[res_idx++] = T(params_.recovery_weights.q_goal_theta) * etheta;

            // 3. 指令正则
            residuals[res_idx++] = T(params_.recovery_weights.r_v) * v_cmd;
            residuals[res_idx++] = T(params_.recovery_weights.r_omega) * omega_cmd;

            // 4. 指令平滑
            residuals[res_idx++] = T(params_.recovery_weights.r_dv) * dv_cmd;
            residuals[res_idx++] = T(params_.recovery_weights.r_domega) * domega_cmd;

            // 5. 指令变化率硬约束（软惩罚实现）
            const T dv_cmd_limit = T(params_.recovery_limits.acc_max * params_.dt);
            const T domega_cmd_limit = T(params_.recovery_limits.alpha_max * params_.dt);
            residuals[res_idx++] = T(params_.recovery_weights.acc_limit) * ceres::fmax(T(0.0), ceres::abs(dv_cmd) - dv_cmd_limit);
            residuals[res_idx++] = T(params_.recovery_weights.alpha_limit) * ceres::fmax(T(0.0), ceres::abs(domega_cmd) - domega_cmd_limit);

            // 6. 侧向加速度约束
            const T a_lat = ceres::abs(st.v_act * st.omega_act);
            residuals[res_idx++] = T(params_.recovery_weights.lat_acc) * ceres::fmax(T(0.0), a_lat - T(params_.recovery_limits.a_lat_max));

            // 7. 避障
            const T cost = interpolate_cost_map(merged_cost_map_, st.x, st.y);
            residuals[res_idx++] = T(params_.recovery_weights.obstacle) * (cost / T(255.0));

            // 8. 台阶方向场：视为不可通行区域
            const auto dir = interpolate_direction_map(direction_map_, st.x, st.y);
            const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-10));
            const T step_gate = smoothstep(
                dir_norm,
                T(params_.follow_limits.step_norm_threshold),
                T(params_.follow_limits.step_norm_threshold + params_.follow_limits.step_norm_transition)
            );
            residuals[res_idx++] = T(params_.recovery_weights.step) * step_gate;

            last_v_cmd = v_cmd;
            last_omega_cmd = omega_cmd;
        }

        // 终端：更强地要求到达目标且不在危险区域
        const T dxT = st.x - T(goal_map_.x());
        const T dyT = st.y - T(goal_map_.y());
        residuals[res_idx++] = T(params_.recovery_weights.q_goal_xy_terminal) * dxT;
        residuals[res_idx++] = T(params_.recovery_weights.q_goal_xy_terminal) * dyT;

        const T cost_terminal = interpolate_cost_map(merged_cost_map_, st.x, st.y);
        residuals[res_idx++] = T(params_.recovery_weights.obstacle_terminal) * (cost_terminal / T(255.0));

        const auto dirT = interpolate_direction_map(direction_map_, st.x, st.y);
        const T dir_normT = ceres::sqrt(dirT.squaredNorm() + T(1e-10));
        const T step_gateT = smoothstep(
            dir_normT,
            T(params_.follow_limits.step_norm_threshold),
            T(params_.follow_limits.step_norm_threshold + params_.follow_limits.step_norm_transition)
        );
        residuals[res_idx++] = T(params_.recovery_weights.step_terminal) * step_gateT;

        return true;
    }

    const Eigen::Vector2d& goal_map_;
    const Eigen::Vector3d& start_pose_;
    const Eigen::Vector2d& start_status_;
    const Eigen::Vector2d& start_cmd_;
    const MPCParams& params_;
    const CostMap& merged_cost_map_;
    const DirectionMap& direction_map_;
};

inline std::vector<Eigen::Vector2d> generate_predicted_path_map(
    const MPCParams& params,
    const Eigen::Vector3d& chassis_pose_map,
    const Eigen::Vector2d& chassis_status,
    const std::vector<std::array<double, 2>>& controls
) {
    std::vector<Eigen::Vector2d> predicted_path_map;
    predicted_path_map.reserve(static_cast<size_t>(params.horizon) + 1);

    PredictorState<double> st;
    st.x = chassis_pose_map.x();
    st.y = chassis_pose_map.y();
    st.theta = chassis_pose_map.z();
    st.v_act = chassis_status.x();
    st.omega_act = chassis_status.y();
    st.a_act = 0.0;
    st.x_lqr = Eigen::Matrix<double, LQR_STATE_SIZE, 1>::Zero();
    st.x_lqr(IDX_DS) = chassis_status.x();
    st.x_lqr(IDX_DPHI) = chassis_status.y();
    st.x_lqr(IDX_PHI) = chassis_pose_map.z();

    predicted_path_map.emplace_back(st.x, st.y);

    double theta_cmd = chassis_pose_map.z();
    for (int i = 0; i < params.horizon; i++) {
        const double v_cmd = controls[static_cast<size_t>(i)][0];
        const double w_cmd = controls[static_cast<size_t>(i)][1];
        prediction_step(params, st, v_cmd, w_cmd, theta_cmd);
        predicted_path_map.emplace_back(st.x, st.y);
    }

    return predicted_path_map;
}

// ============================================================================
//                            MPCController 实现
// ============================================================================

MPCController::MPCController(const MPCParams& params): params_(params) {
    last_controls_.assign(static_cast<size_t>(params_.horizon), Eigen::Vector2d::Zero());
}

std::expected<std::tuple<Eigen::Vector3d, std::vector<Eigen::Vector2d>>, std::string> MPCController::follow_path(
    const SplineD& global_path,
    const Eigen::Vector3d& chassis_pose_map,
    const Eigen::Vector2d& chassis_status,
    const CostMap& merged_cost_map,
    const DirectionMap& global_direction_map
) {
    // 投影到样条
    const double u0 = project_to_spline_u(
        global_path,
        chassis_pose_map.head<2>(),
        last_u_,
        params_.follow_projection.proj_num_samples,
        params_.follow_projection.proj_search_window,
        params_.follow_projection.max_correspondence_distance
    );
    last_u_ = u0;

    // 决策变量
    std::vector<std::array<double, 2>> controls(static_cast<size_t>(params_.horizon), {0.0, 0.0});

    // Warm start
    if (last_controls_.size() == static_cast<size_t>(params_.horizon)) {
        for (int i = 0; i < params_.horizon; i++) {
            const Eigen::Vector2d init = (i + 1 < params_.horizon) ? last_controls_[static_cast<size_t>(i + 1)] : last_controls_.back();
            controls[static_cast<size_t>(i)][0] = init.x();
            controls[static_cast<size_t>(i)][1] = init.y();
        }
    } else {
        for (int i = 0; i < params_.horizon; i++) {
            controls[static_cast<size_t>(i)][0] = chassis_status.x();
            controls[static_cast<size_t>(i)][1] = chassis_status.y();
        }
    }

    // 构建优化问题
    ceres::Problem problem;

    auto* cost_function = new ceres::DynamicAutoDiffCostFunction<FollowMPCCostFunctor>(new FollowMPCCostFunctor(
        global_path.getControlPoints(),
        chassis_pose_map,
        u0,
        chassis_status,
        last_cmd_,
        params_,
        merged_cost_map,
        global_direction_map
    ));

    // 添加参数块
    for (int i = 0; i < params_.horizon; i++) {
        cost_function->AddParameterBlock(2);
    }
    // 每步15个残差
    cost_function->SetNumResiduals(15 * params_.horizon);

    std::vector<double*> parameter_blocks;
    parameter_blocks.reserve(static_cast<size_t>(params_.horizon));

    for (auto& c: controls) {
        parameter_blocks.push_back(c.data());
    }

    problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);

    // 设置边界
    for (int i = 0; i < params_.horizon; i++) {
        problem.SetParameterLowerBound(controls[static_cast<size_t>(i)].data(), 0, params_.follow_limits.vel_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(i)].data(), 0, params_.follow_limits.vel_max);
        problem.SetParameterLowerBound(controls[static_cast<size_t>(i)].data(), 1, params_.follow_limits.omega_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(i)].data(), 1, params_.follow_limits.omega_max);
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

    const double theta_cmd_map = std::atan2(
        std::sin(chassis_pose_map.z() + controls[0][1] * params_.dt),
        std::cos(chassis_pose_map.z() + controls[0][1] * params_.dt)
    );

    // 保存warm start
    last_controls_.resize(static_cast<size_t>(params_.horizon));
    for (int i = 0; i < params_.horizon; i++) {
        last_controls_[static_cast<size_t>(i)] = Eigen::Vector2d(controls[static_cast<size_t>(i)][0], controls[static_cast<size_t>(i)][1]);
    }

    const std::vector<Eigen::Vector2d> predicted_path_map = generate_predicted_path_map(params_, chassis_pose_map, chassis_status, controls);
    return std::make_tuple(Eigen::Vector3d(cmd_v_omega.x(), theta_cmd_map, cmd_v_omega.y()), predicted_path_map);
}

std::expected<std::tuple<Eigen::Vector3d, std::vector<Eigen::Vector2d>>, std::string> MPCController::stop(
    const Eigen::Vector3d& chassis_pose_map,
    const Eigen::Vector2d& chassis_status,
    const CostMap& merged_cost_map,
    const DirectionMap& global_direction_map
) {
    // 决策变量
    std::vector<std::array<double, 2>> controls(static_cast<size_t>(params_.horizon), {0.0, 0.0});

    // Warm start
    if (last_controls_.size() == static_cast<size_t>(params_.horizon)) {
        for (int i = 0; i < params_.horizon; i++) {
            const Eigen::Vector2d init = (i + 1 < params_.horizon) ? last_controls_[static_cast<size_t>(i + 1)] : last_controls_.back();
            controls[static_cast<size_t>(i)][0] = std::max(0.0, init.x());
            controls[static_cast<size_t>(i)][1] = init.y();
        }
    } else {
        for (int i = 0; i < params_.horizon; i++) {
            controls[static_cast<size_t>(i)][0] = std::max(0.0, chassis_status.x());
            controls[static_cast<size_t>(i)][1] = chassis_status.y();
        }
    }

    ceres::Problem problem;

    auto* cost_function = new ceres::DynamicAutoDiffCostFunction<StopMPCCostFunctor>(new StopMPCCostFunctor(
        chassis_pose_map,
        chassis_status,
        last_cmd_,
        params_,
        merged_cost_map,
        global_direction_map
    ));

    for (int i = 0; i < params_.horizon; i++) {
        cost_function->AddParameterBlock(2);
    }

    // 每步8个残差 + 终端2个
    cost_function->SetNumResiduals(8 * params_.horizon + 2);

    std::vector<double*> parameter_blocks;
    parameter_blocks.reserve(static_cast<size_t>(params_.horizon));
    for (auto& c: controls) {
        parameter_blocks.push_back(c.data());
    }
    problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);

    // 边界
    for (int i = 0; i < params_.horizon; i++) {
        problem.SetParameterLowerBound(controls[static_cast<size_t>(i)].data(), 0, 0.0);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(i)].data(), 0, params_.stop_limits.vel_max);
        problem.SetParameterLowerBound(controls[static_cast<size_t>(i)].data(), 1, params_.stop_limits.omega_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(i)].data(), 1, params_.stop_limits.omega_max);
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

    const double theta_cmd_map = std::atan2(
        std::sin(chassis_pose_map.z() + controls[0][1] * params_.dt),
        std::cos(chassis_pose_map.z() + controls[0][1] * params_.dt)
    );

    // 保存warm start
    last_controls_.resize(static_cast<size_t>(params_.horizon));
    for (int i = 0; i < params_.horizon; i++) {
        last_controls_[static_cast<size_t>(i)] = Eigen::Vector2d(controls[static_cast<size_t>(i)][0], controls[static_cast<size_t>(i)][1]);
    }

    const std::vector<Eigen::Vector2d> predicted_path_map = generate_predicted_path_map(params_, chassis_pose_map, chassis_status, controls);
    return std::make_tuple(Eigen::Vector3d(cmd_v_omega.x(), theta_cmd_map, cmd_v_omega.y()), predicted_path_map);
}

std::expected<std::tuple<Eigen::Vector3d, std::vector<Eigen::Vector2d>>, std::string> MPCController::recover_to_point(
    const Eigen::Vector2d& goal_map,
    const Eigen::Vector3d& chassis_pose_map,
    const Eigen::Vector2d& chassis_status,
    const CostMap& merged_cost_map,
    const DirectionMap& global_direction_map
) {
    // 决策变量
    std::vector<std::array<double, 2>> controls(static_cast<size_t>(params_.horizon), {0.0, 0.0});

    // Warm start
    if (last_controls_.size() == static_cast<size_t>(params_.horizon)) {
        for (int i = 0; i < params_.horizon; i++) {
            const Eigen::Vector2d init = (i + 1 < params_.horizon) ? last_controls_[static_cast<size_t>(i + 1)] : last_controls_.back();
            controls[static_cast<size_t>(i)][0] = std::max(0.0, init.x());
            controls[static_cast<size_t>(i)][1] = init.y();
        }
    } else {
        for (int i = 0; i < params_.horizon; i++) {
            controls[static_cast<size_t>(i)][0] = std::max(0.0, chassis_status.x());
            controls[static_cast<size_t>(i)][1] = chassis_status.y();
        }
    }

    ceres::Problem problem;

    auto* cost_function = new ceres::DynamicAutoDiffCostFunction<RecoveryMPCCostFunctor>(new RecoveryMPCCostFunctor(
        goal_map,
        chassis_pose_map,
        chassis_status,
        last_cmd_,
        params_,
        merged_cost_map,
        global_direction_map
    ));

    for (int i = 0; i < params_.horizon; i++) {
        cost_function->AddParameterBlock(2);
    }

    // 每步12个残差 + 终端4个
    cost_function->SetNumResiduals(12 * params_.horizon + 4);

    std::vector<double*> parameter_blocks;
    parameter_blocks.reserve(static_cast<size_t>(params_.horizon));
    for (auto& c: controls) {
        parameter_blocks.push_back(c.data());
    }
    problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);

    // 边界
    for (int i = 0; i < params_.horizon; i++) {
        problem.SetParameterLowerBound(controls[static_cast<size_t>(i)].data(), 0, params_.recovery_limits.vel_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(i)].data(), 0, params_.recovery_limits.vel_max);
        problem.SetParameterLowerBound(controls[static_cast<size_t>(i)].data(), 1, params_.recovery_limits.omega_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(i)].data(), 1, params_.recovery_limits.omega_max);
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

    const double theta_cmd_map = std::atan2(
        std::sin(chassis_pose_map.z() + controls[0][1] * params_.dt),
        std::cos(chassis_pose_map.z() + controls[0][1] * params_.dt)
    );

    // 保存warm start
    last_controls_.resize(static_cast<size_t>(params_.horizon));
    for (int i = 0; i < params_.horizon; i++) {
        last_controls_[static_cast<size_t>(i)] = Eigen::Vector2d(controls[static_cast<size_t>(i)][0], controls[static_cast<size_t>(i)][1]);
    }

    const std::vector<Eigen::Vector2d> predicted_path_map = generate_predicted_path_map(params_, chassis_pose_map, chassis_status, controls);
    return std::make_tuple(Eigen::Vector3d(cmd_v_omega.x(), theta_cmd_map, cmd_v_omega.y()), predicted_path_map);
}

}