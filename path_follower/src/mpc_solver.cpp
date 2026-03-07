#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <algorithm>
#include <array>
#include <uniform_bspline/uniform_bspline.hpp>
#include <uniform_bspline_ceres/uniform_bspline_ceres_generator.hpp>
#include <path_follower/mpc_solver.hpp>
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

template<typename T>
inline T softplus(const T& x) {
    // Numerically-stable softplus: log(1 + exp(x))
    // Use mild branching to avoid overflow while keeping derivatives well-behaved.
    if (x > T(20.0)) {
        return x;
    }
    if (x < T(-20.0)) {
        return ceres::exp(x);
    }
    return ceres::log(T(1.0) + ceres::exp(x));
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
//                          弧长查找表预计算
// ============================================================================

inline bool same_control_points(
    const std::vector<Eigen::Vector2d>& lhs,
    const std::vector<Eigen::Vector2d>& rhs
) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs.size(); i++) {
        if (lhs[i].x() != rhs[i].x() || lhs[i].y() != rhs[i].y()) {
            return false;
        }
    }

    return true;
}

inline ArclengthTable build_arclength_table(
    const std::vector<Eigen::Vector2d>& control_points,
    const int num_samples
) {
    ArclengthTable table{};
    for (int i = 0; i <= ARCLENGTH_TABLE_SIZE; i++) {
        const double u = static_cast<double>(i) / static_cast<double>(ARCLENGTH_TABLE_SIZE);
        table[static_cast<size_t>(i)] = estimate_remaining_arclength(control_points, u, num_samples);
    }
    return table;
}

inline double lookup_remaining_arclength(
    const ArclengthTable& table,
    double u
) {
    u = std::max(0.0, std::min(1.0, u));
    const double idx = u * static_cast<double>(ARCLENGTH_TABLE_SIZE);
    const int i0 = std::min(static_cast<int>(std::floor(idx)), ARCLENGTH_TABLE_SIZE - 1);
    const int i1 = i0 + 1;
    const double t = idx - static_cast<double>(i0);
    return (1.0 - t) * table[static_cast<size_t>(i0)] + t * table[static_cast<size_t>(i1)];
}

// ============================================================================
//                       零拷贝栅格视图与双线性采样
// ============================================================================

struct GridInfo {
    double origin_x, origin_y, inv_resolution;
    int width, height;
};

template<typename MapT>
inline GridInfo make_grid_info(const MapT& map) {
    return GridInfo{map.origin_x, map.origin_y, 1.0 / map.resolution, map.width, map.height};
}

struct CostMapGridView {
    enum { DATA_DIMENSION = 1 };

    explicit CostMapGridView(const CostMap& map): map_(map) {}

    EIGEN_STRONG_INLINE double value_at_clamped(const int row, const int col) const {
        const int row_idx = clamp_int(row, 0, map_.height - 1);
        const int col_idx = clamp_int(col, 0, map_.width - 1);
        return static_cast<double>(map_.data[static_cast<size_t>(row_idx * map_.width + col_idx)]);
    }

    const CostMap& map_;
};

struct DirectionMapGridView {
    enum { DATA_DIMENSION = 2 };

    explicit DirectionMapGridView(const DirectionMap& map): map_(map) {}

    EIGEN_STRONG_INLINE Eigen::Vector2d value_at_clamped(const int row, const int col) const {
        const int row_idx = clamp_int(row, 0, map_.height - 1);
        const int col_idx = clamp_int(col, 0, map_.width - 1);
        return map_.data[static_cast<size_t>(row_idx * map_.width + col_idx)];
    }

    const DirectionMap& map_;
};

template<typename T>
inline T eval_cost_bilinear(
    const CostMapGridView& grid,
    const GridInfo& info,
    const T& x_map, const T& y_map
) {
    if (info.width < 2 || info.height < 2) {
        return T(255.0);
    }

    const T gx = (x_map - T(info.origin_x)) * T(info.inv_resolution);
    const T gy = (y_map - T(info.origin_y)) * T(info.inv_resolution);
    const double gxs = scalar_value(gx);
    const double gys = scalar_value(gy);
    if (gxs < 0.0 || gys < 0.0 ||
        gxs >= static_cast<double>(info.width - 1) ||
        gys >= static_cast<double>(info.height - 1)) {
        return T(255.0);
    }

    const int x0 = static_cast<int>(std::floor(gxs));
    const int y0 = static_cast<int>(std::floor(gys));
    const T dx = gx - T(x0);
    const T dy = gy - T(y0);

    const T v00 = T(grid.value_at_clamped(y0, x0));
    const T v01 = T(grid.value_at_clamped(y0, x0 + 1));
    const T v10 = T(grid.value_at_clamped(y0 + 1, x0));
    const T v11 = T(grid.value_at_clamped(y0 + 1, x0 + 1));

    const T one_minus_dx = T(1.0) - dx;
    const T one_minus_dy = T(1.0) - dy;
    return one_minus_dy * (one_minus_dx * v00 + dx * v01)
        + dy * (one_minus_dx * v10 + dx * v11);
}

template<typename T>
inline Eigen::Matrix<T, 2, 1> eval_dir_bilinear(
    const DirectionMapGridView& grid,
    const GridInfo& info,
    const T& x_map, const T& y_map
) {
    if (info.width < 2 || info.height < 2) {
        return Eigen::Matrix<T, 2, 1>(T(0.0), T(0.0));
    }

    const T gx = (x_map - T(info.origin_x)) * T(info.inv_resolution);
    const T gy = (y_map - T(info.origin_y)) * T(info.inv_resolution);
    const double gxs = scalar_value(gx);
    const double gys = scalar_value(gy);
    if (gxs < 0.0 || gys < 0.0 ||
        gxs >= static_cast<double>(info.width - 1) ||
        gys >= static_cast<double>(info.height - 1)) {
        return Eigen::Matrix<T, 2, 1>(T(0.0), T(0.0));
    }

    const int x0 = static_cast<int>(std::floor(gxs));
    const int y0 = static_cast<int>(std::floor(gys));
    const T dx = gx - T(x0);
    const T dy = gy - T(y0);

    const Eigen::Vector2d v00d = grid.value_at_clamped(y0, x0);
    const Eigen::Vector2d v01d = grid.value_at_clamped(y0, x0 + 1);
    const Eigen::Vector2d v10d = grid.value_at_clamped(y0 + 1, x0);
    const Eigen::Vector2d v11d = grid.value_at_clamped(y0 + 1, x0 + 1);

    const Eigen::Matrix<T, 2, 1> v00(T(v00d.x()), T(v00d.y()));
    const Eigen::Matrix<T, 2, 1> v01(T(v01d.x()), T(v01d.y()));
    const Eigen::Matrix<T, 2, 1> v10(T(v10d.x()), T(v10d.y()));
    const Eigen::Matrix<T, 2, 1> v11(T(v11d.x()), T(v11d.y()));

    const T one_minus_dx = T(1.0) - dx;
    const T one_minus_dy = T(1.0) - dy;
    return one_minus_dy * (one_minus_dx * v00 + dx * v01)
        + dy * (one_minus_dx * v10 + dx * v11);
}

// ============================================================================
//                                 MPC预测模型
// ============================================================================

namespace {
template<typename T>
inline T sabs(const T& x) { return ceres::sqrt(x * x + T(PWR_EPS2)); }

template<typename T>
inline T predict_power(const T& v, const T& w, const T& a, const T& alpha) {
    return T(PWR_C[0])
        + T(PWR_C[1]) * v * a + T(PWR_C[2]) * w * alpha
        + T(PWR_C[3]) * a * a + T(PWR_C[4]) * alpha * alpha
        + T(PWR_C[5]) * sabs(v) + T(PWR_C[6]) * sabs(w)
        + T(PWR_C[7]) * v * v + T(PWR_C[8]) * w * w
        + T(PWR_C[9]) * sabs(a) + T(PWR_C[10]) * sabs(alpha)
        + T(PWR_C[11]) * sabs(v * w);
}

template<typename T>
struct PredictorState {
    T x, y, theta;
    T v_act, omega_act;

    // greybox 模型内部状态 x = [x_h, v_act, w_act, dv, dw]
    Eigen::Matrix<T, MODEL_NX, 1> x_model;
};

template<typename T>
inline void model_init(
    PredictorState<T>& st,
    const T& v_meas,
    const T& w_meas,
    const T& v_cmd_prev,
    const T& w_cmd_prev,
    const T& x_h_init
) {
    st.x_model.setZero();
    st.x_model(0) = x_h_init;
    st.x_model(1) = v_meas;
    st.x_model(2) = w_meas;
    st.x_model(3) = v_cmd_prev; // dv = v_cmd[k-1]
    st.x_model(4) = w_cmd_prev; // dw = w_cmd[k-1]

    st.v_act = v_meas;
    st.omega_act = w_meas;
}

template<typename T>
inline T smooth_sgn(const T& x, const double eps) {
    const T e = ceres::fmax(T(1e-6), T(eps));
    return ceres::tanh(x / e);
}

template<typename T>
inline void model_step(
    PredictorState<T>& st,
    const T& v_cmd,
    const T& w_cmd
) {
    const T xh = st.x_model(0);
    const T v = st.x_model(1);
    const T w = st.x_model(2);
    const T dv = st.x_model(3);
    const T dw = st.x_model(4);

    // ZOH linear part
    T xh_next = T(A00) * xh + T(A01) * v + T(A03) * dv;
    T v_next = T(A10) * xh + T(A11) * v + T(A13) * dv;
    T w_next = T(A22) * w + T(A24) * dw;
    const T dv_next = v_cmd;
    const T dw_next = w_cmd;

    // ZOH nonlinear: gains from G matrix instead of plain dt
    const T sv = smooth_sgn(v, SGN_EPS);
    const T sw = smooth_sgn(w, SGN_EPS);
    const T nl_v = T(CF1) * sv + T(CF2) * v * ceres::abs(w);

    xh_next += T(GNL_XH) * nl_v;
    v_next  += T(GNL_V)  * nl_v;
    w_next  += -T(GAMMA_W) * T(CF3) * sw;

    st.x_model(0) = xh_next;
    st.x_model(1) = v_next;
    st.x_model(2) = w_next;
    st.x_model(3) = dv_next;
    st.x_model(4) = dw_next;

    // y = [v_act, w_act]
    st.v_act = v_next;
    st.omega_act = w_next;
}

template<typename T>
inline void prediction_step(
    PredictorState<T>& st,
    const T& v_cmd,
    const T& omega_cmd
) {
    const T dt = T(MPC_DT);

    const T theta0 = st.theta;
    const T v0 = st.v_act;
    const T w0 = st.omega_act;

    // 先更新执行器动态，再用 v_act/omega_act 推进位姿
    model_step(st, v_cmd, omega_cmd);

    const T v1 = st.v_act;
    const T w1 = st.omega_act;
    const T theta1 = theta0 + (w0 + w1) * (dt * T(0.5));

    st.x += (v0 * ceres::cos(theta0) + v1 * ceres::cos(theta1)) * (dt * T(0.5));
    st.y += (v0 * ceres::sin(theta0) + v1 * ceres::sin(theta1)) * (dt * T(0.5));
    st.theta = theta1;
}
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
        const Eigen::Vector2d& start_cmd_clamped,
        const double x_h_init,
        const MPCParams& params,
        const CostMapGridView& cost_grid,
        const GridInfo& cost_info,
        const DirectionMapGridView& dir_grid,
        const GridInfo& dir_info,
        const ArclengthTable& arclength_table,
        const double remaining_energy,
        const double rfr_pwr_limit
    ):
        ref_control_points_(ref_control_points),
        start_pose_(start_pose),
        u0_(u0),
        start_status_(start_status),
        start_cmd_clamped_(start_cmd_clamped),
        x_h_init_(x_h_init),
        params_(params),
        cost_grid_(cost_grid),
        cost_info_(cost_info),
        dir_grid_(dir_grid),
        dir_info_(dir_info),
        arclength_table_(arclength_table),
        remaining_energy_(remaining_energy),
        rfr_pwr_limit_(rfr_pwr_limit) {}

    template<typename T>
    bool operator()(const T* const parameters, T* residuals) const {
        PredictorState<T> st;
        st.x = T(start_pose_.x());
        st.y = T(start_pose_.y());
        st.theta = T(start_pose_.z());
        T u = T(u0_);
        T last_v_cmd = T(start_cmd_clamped_.x());
        T last_omega_cmd = T(start_cmd_clamped_.y());

        model_init(
            st,
            T(start_status_.x()),
            T(start_status_.y()),
            last_v_cmd,
            last_omega_cmd,
            T(x_h_init_)
        );

        T energy = T(remaining_energy_);
        T v_act_prev = T(start_status_.x());
        T w_act_prev = T(start_status_.y());

        size_t res_idx = 0;
        for (int k = 0; k < MPC_HORIZON; k++) {
            const T v_cmd = parameters[2 * k];
            const T omega_cmd = parameters[2 * k + 1];

            // 指令变化
            const T dv_cmd = v_cmd - last_v_cmd;
            const T domega_cmd = omega_cmd - last_omega_cmd;

            prediction_step(st, v_cmd, omega_cmd);

            // 能量传播
            const T a_k = (st.v_act - v_act_prev) / T(MPC_DT);
            const T alpha_k = (st.omega_act - w_act_prev) / T(MPC_DT);
            energy += (T(rfr_pwr_limit_) - predict_power(st.v_act, st.omega_act, a_k, alpha_k)) * T(MPC_DT);

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

            // 终点剩余里程：O(1) 查表替代循环积分
            const double s_remain_d = lookup_remaining_arclength(arclength_table_, scalar_value(u));
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
            const T dv_cmd_limit = T(params_.follow_limits.acc_max * MPC_DT);
            const T domega_cmd_limit = T(params_.follow_limits.alpha_max * MPC_DT);
            residuals[res_idx++] = T(params_.follow_weights.acc_limit) * ceres::fmax(T(0.0), ceres::abs(dv_cmd) - dv_cmd_limit);
            residuals[res_idx++] = T(params_.follow_weights.alpha_limit) * ceres::fmax(T(0.0), ceres::abs(domega_cmd) - domega_cmd_limit);

            // 6. 侧向加速度约束
            const T a_lat = ceres::abs(st.v_act * st.omega_act);
            residuals[res_idx++] = T(params_.follow_weights.lat_acc) * ceres::fmax(T(0.0), a_lat - T(params_.follow_limits.a_lat_max));

            // 7. 避障（零拷贝双线性采样）
            const T cost = eval_cost_bilinear(cost_grid_, cost_info_, st.x, st.y);
            residuals[res_idx++] = T(params_.follow_weights.obstacle) * (cost / T(255.0));

            // 8. 台阶处理（零拷贝双线性采样）
            const auto dir = eval_dir_bilinear(dir_grid_, dir_info_, st.x, st.y);
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
            const T v_dec_profile = ceres::sqrt(T(2.0) * deceleration * s_remain + T(params_.follow_limits.slow_down_target_vel * params_.follow_limits.slow_down_target_vel)); // 基于物理的限速 (v1^2 - v0^2 = 2 * a * s)
            residuals[res_idx++] = T(params_.follow_weights.q_v_final) * ceres::fmax(T(0.0), st.v_act - v_dec_profile); // 惩罚超速

            // 10. 能量约束
            const T w_e = T(params_.energy.enable ? params_.energy.weight : 0.0);
            const T thr = T(std::max(params_.energy.threshold, 1.0));
            const T beta = T(std::max(params_.energy.softplus_beta, 1e-6));
            const T vio = (T(params_.energy.threshold) - energy) / thr;
            // Make penalty EXACTLY zero for vio <= 0 (energy above threshold).
            // softplus(beta*vio)/beta is > 0 even for vio < 0, which can cause a slow drift
            // towards minimum speed just to keep "charging" in the prediction model.
            const T relu_soft = softplus(beta * vio) / beta - softplus(T(0.0)) / beta;
            residuals[res_idx++] = w_e * ceres::fmax(T(0.0), relu_soft);

            // 进度动力学
            T denom = T(1.0) - kappa * ey;
            const T denom_abs = ceres::abs(denom);
            denom = denom / (denom_abs + T(1e-6)) * ceres::fmax(denom_abs, T(0.1));
            const T dsdt = st.v_act * ceres::cos(etheta) / denom;
            const T dudt = dsdt / dsdu;
            u += dudt * T(MPC_DT);
            u = ceres::fmin(ceres::fmax(u, T(0.0)), T(1.0));

            last_v_cmd = v_cmd;
            last_omega_cmd = omega_cmd;
            v_act_prev = st.v_act;
            w_act_prev = st.omega_act;
        }

        return true;
    }

    const std::vector<Eigen::Vector2d>& ref_control_points_;
    const Eigen::Vector3d& start_pose_;
    const double u0_;
    const Eigen::Vector2d& start_status_;
    const Eigen::Vector2d start_cmd_clamped_;
    const double x_h_init_;
    const MPCParams& params_;
    const CostMapGridView& cost_grid_;
    const GridInfo cost_info_;
    const DirectionMapGridView& dir_grid_;
    const GridInfo dir_info_;
    const ArclengthTable& arclength_table_;
    const double remaining_energy_;
    const double rfr_pwr_limit_;
};

// ============================================================================
//                            Stop 模式 Cost Functor
// ============================================================================

struct StopMPCCostFunctor {
    StopMPCCostFunctor(
        const Eigen::Vector3d& start_pose,
        const Eigen::Vector2d& start_status,
        const Eigen::Vector2d& start_cmd_clamped,
        const double x_h_init,
        const MPCParams& params,
        const CostMapGridView& cost_grid,
        const GridInfo& cost_info,
        const DirectionMapGridView& dir_grid,
        const GridInfo& dir_info,
        const double remaining_energy,
        const double rfr_pwr_limit
    ):
        start_pose_(start_pose),
        start_status_(start_status),
        start_cmd_clamped_(start_cmd_clamped),
        x_h_init_(x_h_init),
        params_(params),
        cost_grid_(cost_grid),
        cost_info_(cost_info),
        dir_grid_(dir_grid),
        dir_info_(dir_info),
        remaining_energy_(remaining_energy),
        rfr_pwr_limit_(rfr_pwr_limit) {}

    template<typename T>
    bool operator()(const T* const parameters, T* residuals) const {
        PredictorState<T> st;
        st.x = T(start_pose_.x());
        st.y = T(start_pose_.y());
        st.theta = T(start_pose_.z());
        T last_v_cmd = T(start_cmd_clamped_.x());
        T last_omega_cmd = T(start_cmd_clamped_.y());

        model_init(
            st,
            T(start_status_.x()),
            T(start_status_.y()),
            last_v_cmd,
            last_omega_cmd,
            T(x_h_init_)
        );

        T energy = T(remaining_energy_);
        T v_act_prev = T(start_status_.x());
        T w_act_prev = T(start_status_.y());

        size_t res_idx = 0;
        for (int k = 0; k < MPC_HORIZON; k++) {
            const T v_cmd = parameters[2 * k];
            const T omega_cmd = parameters[2 * k + 1];
            const T dv_cmd = v_cmd - last_v_cmd;
            const T domega_cmd = omega_cmd - last_omega_cmd;

            prediction_step(st, v_cmd, omega_cmd);

            // 能量传播
            const T a_k = (st.v_act - v_act_prev) / T(MPC_DT);
            const T alpha_k = (st.omega_act - w_act_prev) / T(MPC_DT);
            energy += (T(rfr_pwr_limit_) - predict_power(st.v_act, st.omega_act, a_k, alpha_k)) * T(MPC_DT);

            // 1. 速度正则化
            residuals[res_idx++] = T(params_.stop_weights.q_v) * st.v_act;
            residuals[res_idx++] = T(params_.stop_weights.q_omega) * st.omega_act;

            // 2 指令平滑
            residuals[res_idx++] = T(params_.stop_weights.r_dv) * dv_cmd;
            residuals[res_idx++] = T(params_.stop_weights.r_domega) * domega_cmd;

            // 3. 指令变化率约束
            const T dv_cmd_limit = T(params_.stop_limits.acc_max * MPC_DT);
            const T domega_cmd_limit = T(params_.stop_limits.alpha_max * MPC_DT);
            residuals[res_idx++] = T(params_.stop_weights.acc_limit) * ceres::fmax(T(0.0), ceres::abs(dv_cmd) - dv_cmd_limit);
            residuals[res_idx++] = T(params_.stop_weights.alpha_limit) * ceres::fmax(T(0.0), ceres::abs(domega_cmd) - domega_cmd_limit);

            // 4. 侧向加速度约束
            const T a_lat = ceres::abs(st.v_act * st.omega_act);
            residuals[res_idx++] = T(params_.stop_weights.lat_acc) * ceres::fmax(T(0.0), a_lat - T(params_.stop_limits.a_lat_max));

            // 5. 避障
            const T cost = eval_cost_bilinear(cost_grid_, cost_info_, st.x, st.y);
            residuals[res_idx++] = T(params_.stop_weights.obstacle) * (cost / T(255.0));

            // 6. 台阶处理
            const auto dir = eval_dir_bilinear(dir_grid_, dir_info_, st.x, st.y);
            const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-10));
            const auto dir_unit = dir / dir_norm;

            // 方向场在非台阶区域可能存在小噪声。用 step_norm_threshold 做门控，避免"停止模式"在接近 0 速度时被拉回一个小正速度。
            const T step_gate = smoothstep(
                dir_norm,
                T(params_.stop_limits.step_norm_threshold),
                T(params_.stop_limits.step_norm_threshold + params_.stop_limits.step_norm_transition)
            );

            // 台阶方向对齐
            const Eigen::Matrix<T, 2, 1> heading(ceres::cos(st.theta), ceres::sin(st.theta));
            const T heading_cross_dir = heading.x() * dir.y() - heading.y() * dir.x();
            residuals[res_idx++] = T(params_.stop_weights.direction) * step_gate * ceres::abs(heading_cross_dir);

            // 台阶区域速度保持
            const T cos_theta = heading.dot(dir_unit);
            const T weight_up = (cos_theta + T(1.0)) / T(2.0); // weight_up 为 1 时表示完全上坡，为 0 时表示完全下坡
            const T target_vel_step = weight_up * T(params_.stop_limits.vel_step_up) + (T(1.0) - weight_up) * T(params_.stop_limits.vel_step_down);
            residuals[res_idx++] = T(params_.stop_weights.vel_on_step) * step_gate * dir_norm * ceres::abs(st.v_act - target_vel_step);

            // 能量约束
            const T w_e = T(params_.energy.enable ? params_.energy.weight : 0.0);
            const T thr = T(std::max(params_.energy.threshold, 1.0));
            const T beta = T(std::max(params_.energy.softplus_beta, 1e-6));
            const T vio = (T(params_.energy.threshold) - energy) / thr;
            const T relu_soft = softplus(beta * vio) / beta - softplus(T(0.0)) / beta;
            residuals[res_idx++] = w_e * ceres::fmax(T(0.0), relu_soft);
            
            last_v_cmd = v_cmd;
            last_omega_cmd = omega_cmd;
            v_act_prev = st.v_act;
            w_act_prev = st.omega_act;
        }

        // 终端约束：不碰撞且不在台阶区域
        const T cost_terminal = eval_cost_bilinear(cost_grid_, cost_info_, st.x, st.y);
        residuals[res_idx++] = T(params_.stop_weights.obstacle_terminal) * (cost_terminal / T(255.0));
        const auto dir = eval_dir_bilinear(dir_grid_, dir_info_, st.x, st.y);
        const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-10));
        const T step_gate = smoothstep(
            dir_norm,
            T(params_.stop_limits.step_norm_threshold),
            T(params_.stop_limits.step_norm_threshold + params_.stop_limits.step_norm_transition)
        );
        residuals[res_idx++] = T(params_.stop_weights.step_terminal) * step_gate;

        return true;
    }

    const Eigen::Vector3d& start_pose_;
    const Eigen::Vector2d& start_status_;
    const Eigen::Vector2d start_cmd_clamped_;
    const double x_h_init_;
    const MPCParams& params_;
    const CostMapGridView& cost_grid_;
    const GridInfo cost_info_;
    const DirectionMapGridView& dir_grid_;
    const GridInfo dir_info_;
    const double remaining_energy_;
    const double rfr_pwr_limit_;
};

// ============================================================================
//                       Recovery-To-Point 模式 Cost Functor
// ============================================================================

struct RecoveryMPCCostFunctor {
    RecoveryMPCCostFunctor(
        const Eigen::Vector2d& goal_map,
        const Eigen::Vector3d& start_pose,
        const Eigen::Vector2d& start_status,
        const Eigen::Vector2d& start_cmd_clamped,
        const double x_h_init,
        const MPCParams& params,
        const CostMapGridView& cost_grid,
        const GridInfo& cost_info,
        const DirectionMapGridView& dir_grid,
        const GridInfo& dir_info,
        const double remaining_energy,
        const double rfr_pwr_limit
    ):
        goal_map_(goal_map),
        start_pose_(start_pose),
        start_status_(start_status),
        start_cmd_clamped_(start_cmd_clamped),
        x_h_init_(x_h_init),
        params_(params),
        cost_grid_(cost_grid),
        cost_info_(cost_info),
        dir_grid_(dir_grid),
        dir_info_(dir_info),
        remaining_energy_(remaining_energy),
        rfr_pwr_limit_(rfr_pwr_limit) {}

    template<typename T>
    bool operator()(const T* const parameters, T* residuals) const {
        PredictorState<T> st;
        st.x = T(start_pose_.x());
        st.y = T(start_pose_.y());
        st.theta = T(start_pose_.z());
        T last_v_cmd = T(start_cmd_clamped_.x());
        T last_omega_cmd = T(start_cmd_clamped_.y());

        model_init(
            st,
            T(start_status_.x()),
            T(start_status_.y()),
            last_v_cmd,
            last_omega_cmd,
            T(x_h_init_)
        );

        const T gx = T(goal_map_.x());
        const T gy = T(goal_map_.y());

        T energy = T(remaining_energy_);
        T v_act_prev = T(start_status_.x());
        T w_act_prev = T(start_status_.y());

        size_t res_idx = 0;
        for (int k = 0; k < MPC_HORIZON; k++) {
            const T v_cmd = parameters[2 * k];
            const T omega_cmd = parameters[2 * k + 1];
            const T dv_cmd = v_cmd - last_v_cmd;
            const T domega_cmd = omega_cmd - last_omega_cmd;

            prediction_step(st, v_cmd, omega_cmd);

            // 能量传播
            const T a_k = (st.v_act - v_act_prev) / T(MPC_DT);
            const T alpha_k = (st.omega_act - w_act_prev) / T(MPC_DT);
            energy += (T(rfr_pwr_limit_) - predict_power(st.v_act, st.omega_act, a_k, alpha_k)) * T(MPC_DT);

            // 1. 目标点吸引
            const T ddx = st.x - gx;
            const T ddy = st.y - gy;
            residuals[res_idx++] = T(params_.recovery_weights.q_goal_xy) * ddx;
            residuals[res_idx++] = T(params_.recovery_weights.q_goal_xy) * ddy;

            // 2. 朝向目标（前后朝向均可，使用sin）
            const T desired_theta = ceres::atan2(gy - st.y, gx - st.x);
            const T heading_cross_desired = ceres::sin(st.theta - desired_theta);
            residuals[res_idx++] = T(params_.recovery_weights.q_goal_theta) * ceres::abs(heading_cross_desired);

            // 3. 指令正则
            residuals[res_idx++] = T(params_.recovery_weights.r_v) * v_cmd;
            residuals[res_idx++] = T(params_.recovery_weights.r_omega) * omega_cmd;

            // 4. 指令平滑
            residuals[res_idx++] = T(params_.recovery_weights.r_dv) * dv_cmd;
            residuals[res_idx++] = T(params_.recovery_weights.r_domega) * domega_cmd;

            // 5. 指令变化率硬约束（软惩罚实现）
            const T dv_cmd_limit = T(params_.recovery_limits.acc_max * MPC_DT);
            const T domega_cmd_limit = T(params_.recovery_limits.alpha_max * MPC_DT);
            residuals[res_idx++] = T(params_.recovery_weights.acc_limit) * ceres::fmax(T(0.0), ceres::abs(dv_cmd) - dv_cmd_limit);
            residuals[res_idx++] = T(params_.recovery_weights.alpha_limit) * ceres::fmax(T(0.0), ceres::abs(domega_cmd) - domega_cmd_limit);

            // 6. 侧向加速度约束
            const T a_lat = ceres::abs(st.v_act * st.omega_act);
            residuals[res_idx++] = T(params_.recovery_weights.lat_acc) * ceres::fmax(T(0.0), a_lat - T(params_.recovery_limits.a_lat_max));

            // 7. 避障
            const T cost = eval_cost_bilinear(cost_grid_, cost_info_, st.x, st.y);
            residuals[res_idx++] = T(params_.recovery_weights.obstacle) * (cost / T(255.0));

            // 8. 台阶方向场：视为不可通行区域
            const auto dir = eval_dir_bilinear(dir_grid_, dir_info_, st.x, st.y);
            const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-10));
            const T step_gate = smoothstep(
                dir_norm,
                T(params_.recovery_limits.step_norm_threshold),
                T(params_.recovery_limits.step_norm_threshold + params_.recovery_limits.step_norm_transition)
            );
            residuals[res_idx++] = T(params_.recovery_weights.step) * step_gate;

            // 能量约束
            const T w_e = T(params_.energy.enable ? params_.energy.weight : 0.0);
            const T thr = T(std::max(params_.energy.threshold, 1.0));
            const T beta_e = T(std::max(params_.energy.softplus_beta, 1e-6));
            const T vio = (T(params_.energy.threshold) - energy) / thr;
            const T relu_soft = softplus(beta_e * vio) / beta_e - softplus(T(0.0)) / beta_e;
            residuals[res_idx++] = w_e * ceres::fmax(T(0.0), relu_soft);

            last_v_cmd = v_cmd;
            last_omega_cmd = omega_cmd;
            v_act_prev = st.v_act;
            w_act_prev = st.omega_act;
        }

        // 终端：更强地要求到达目标且不在危险区域
        const T dxT = st.x - T(goal_map_.x());
        const T dyT = st.y - T(goal_map_.y());
        residuals[res_idx++] = T(params_.recovery_weights.q_goal_xy_terminal) * dxT;
        residuals[res_idx++] = T(params_.recovery_weights.q_goal_xy_terminal) * dyT;

        const T cost_terminal = eval_cost_bilinear(cost_grid_, cost_info_, st.x, st.y);
        residuals[res_idx++] = T(params_.recovery_weights.obstacle_terminal) * (cost_terminal / T(255.0));

        const auto dirT = eval_dir_bilinear(dir_grid_, dir_info_, st.x, st.y);
        const T dir_normT = ceres::sqrt(dirT.squaredNorm() + T(1e-10));
        const T step_gateT = smoothstep(
            dir_normT,
            T(params_.recovery_limits.step_norm_threshold),
            T(params_.recovery_limits.step_norm_threshold + params_.recovery_limits.step_norm_transition)
        );
        residuals[res_idx++] = T(params_.recovery_weights.step_terminal) * step_gateT;

        return true;
    }

    const Eigen::Vector2d& goal_map_;
    const Eigen::Vector3d& start_pose_;
    const Eigen::Vector2d& start_status_;
    const Eigen::Vector2d start_cmd_clamped_;
    const double x_h_init_;
    const MPCParams& params_;
    const CostMapGridView& cost_grid_;
    const GridInfo cost_info_;
    const DirectionMapGridView& dir_grid_;
    const GridInfo dir_info_;
    const double remaining_energy_;
    const double rfr_pwr_limit_;
};

// ============================================================================
//                          生成预测轨迹
// ============================================================================

inline MPCPrediction generate_predicted_path_map(
    const Eigen::Vector3d& chassis_pose_map,
    const Eigen::Vector2d& chassis_status,
    const Eigen::Vector2d& cmd_prev,
    const double x_h_init,
    const double* controls
) {
    MPCPrediction pred;
    constexpr size_t n = static_cast<size_t>(MPC_HORIZON) + 1;
    pred.path_map.reserve(n);
    pred.headings.reserve(n);
    pred.v_pred.reserve(n);
    pred.w_pred.reserve(n);

    PredictorState<double> st;
    st.x = chassis_pose_map.x();
    st.y = chassis_pose_map.y();
    st.theta = chassis_pose_map.z();

    model_init(
        st,
        chassis_status.x(),
        chassis_status.y(),
        cmd_prev.x(),
        cmd_prev.y(),
        x_h_init
    );

    pred.path_map.emplace_back(st.x, st.y);
    pred.headings.push_back(st.theta);
    pred.v_pred.push_back(st.v_act);
    pred.w_pred.push_back(st.omega_act);

    for (int i = 0; i < MPC_HORIZON; i++) {
        const double v_cmd = controls[2 * i];
        const double w_cmd = controls[2 * i + 1];
        prediction_step(st, v_cmd, w_cmd);
        pred.path_map.emplace_back(st.x, st.y);
        pred.headings.push_back(st.theta);
        pred.v_pred.push_back(st.v_act);
        pred.w_pred.push_back(st.omega_act);
    }

    return pred;
}

inline double clamp_prev_cmd(
    const double cmd_prev,
    const double status,
    const double cmd_act_diff_max,
    const double rate_max,
    const double dt
) {
    return std::max(
        std::min(cmd_prev, status + cmd_act_diff_max - rate_max * dt),
        status - cmd_act_diff_max + rate_max * dt
    );
}

// ============================================================================
//                            MPCSolver 实现
// ============================================================================

MPCSolver::MPCSolver(const MPCParams& params): params_(params) {
    last_controls_.fill(0.0);
}

void MPCSolver::set_last_cmd(const Eigen::Vector2d& cmd) {
    last_cmd_ = cmd;
}

void MPCSolver::reset_warm_start() {
    last_controls_.fill(0.0);
}

void MPCSolver::set_energy_state(double remaining_energy, double rfr_pwr_limit) {
    remaining_energy_ = remaining_energy;
    rfr_pwr_limit_ = rfr_pwr_limit;
}

void MPCSolver::update_observer(const double v_act, const double w_act) {
    if (!observer_initialized_) {
        // First call: no previous data, use default x_h
        x_h_hat_ = XH0;
        prev_v_act_ = v_act;
        prev_w_act_ = w_act;
        observer_initialized_ = true;
        return;
    }

    const double v_prev = prev_v_act_;
    const double w_prev = prev_w_act_;
    const double vc_prev = last_cmd_.x();  // v_cmd sent at previous cycle

    // Nonlinear term at previous step
    const double sv_prev = std::tanh(v_prev / SGN_EPS);
    const double nl_prev = CF1 * sv_prev + CF2 * v_prev * std::abs(w_prev);

    // Predict x_h and v using ZOH model
    const double xh_pred = A00 * x_h_hat_ + A01 * v_prev + A03 * vc_prev + GNL_XH * nl_prev;
    const double v_pred  = A10 * x_h_hat_ + A11 * v_prev + A13 * vc_prev + GNL_V  * nl_prev;

    // Innovation: observed v_act vs predicted v
    const double innovation = v_act - v_pred;

    // Correct hidden state
    x_h_hat_ = xh_pred + OBS_L * innovation;

    // Store for next cycle
    prev_v_act_ = v_act;
    prev_w_act_ = w_act;
}

std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> MPCSolver::follow_path(
    const SplineD& global_path,
    const Eigen::Vector3d& chassis_pose_map,
    const Eigen::Vector2d& chassis_status,
    const CostMap& merged_cost_map,
    const DirectionMap& global_direction_map
) {
    // 投影到样条
    const auto& ref_control_points = global_path.getControlPoints();
    const double u0 = project_to_spline_u(
        global_path,
        chassis_pose_map.head<2>(),
        last_u_,
        params_.follow_projection.proj_num_samples,
        params_.follow_projection.proj_search_window,
        params_.follow_projection.max_correspondence_distance
    );
    last_u_ = u0;

    const Eigen::Vector2d start_cmd = last_cmd_;
    const Eigen::Vector2d start_cmd_clamped(
        clamp_prev_cmd(start_cmd.x(), chassis_status.x(), params_.follow_limits.start_vel_cmd_act_diff_max, params_.follow_limits.acc_max, MPC_DT),
        clamp_prev_cmd(start_cmd.y(), chassis_status.y(), params_.follow_limits.start_omega_cmd_act_diff_max, params_.follow_limits.alpha_max, MPC_DT)
    );

    const CostMapGridView cost_grid(merged_cost_map);
    const GridInfo cost_info = make_grid_info(merged_cost_map);
    const DirectionMapGridView dir_grid(global_direction_map);
    const GridInfo dir_info = make_grid_info(global_direction_map);

    // ── 路径未变化时复用弧长查找表 ──
    const ArclengthTable& arclength_table = [&]() -> const ArclengthTable& {
        const int samples = params_.follow_limits.slow_down_num_samples;
        if (prev_arc_samples_ != samples || !same_control_points(prev_ref_control_points_, ref_control_points)) {
            prev_arclength_table_ = build_arclength_table(ref_control_points, samples);
            prev_ref_control_points_ = ref_control_points;
            prev_arc_samples_ = samples;
        }
        return prev_arclength_table_;
    }();

    // 决策变量（单一平坦参数块）
    std::array<double, MPC_PARAM_SIZE> controls{};

    // Warm start
    for (int i = 0; i < MPC_HORIZON; i++) {
        const int src = (i + 1 < MPC_HORIZON) ? (i + 1) : (MPC_HORIZON - 1);
        controls[static_cast<size_t>(2 * i)]     = last_controls_[static_cast<size_t>(2 * src)];
        controls[static_cast<size_t>(2 * i + 1)] = last_controls_[static_cast<size_t>(2 * src + 1)];
    }

    // 构建优化问题
    ceres::Problem problem;

    constexpr int FOLLOW_NUM_RESIDUALS = 16 * MPC_HORIZON;
    auto* cost_function = new ceres::AutoDiffCostFunction<
        FollowMPCCostFunctor, FOLLOW_NUM_RESIDUALS, MPC_PARAM_SIZE>(
        new FollowMPCCostFunctor(
            ref_control_points,
            chassis_pose_map,
            u0,
            chassis_status,
            start_cmd_clamped,
            x_h_hat_,
            params_,
            cost_grid,
            cost_info,
            dir_grid,
            dir_info,
            arclength_table,
            remaining_energy_,
            rfr_pwr_limit_
        )
    );

    problem.AddResidualBlock(cost_function, nullptr, controls.data());

    // 设置边界
    for (int i = 0; i < MPC_HORIZON; i++) {
        problem.SetParameterLowerBound(controls.data(), 2 * i,     params_.follow_limits.vel_min);
        problem.SetParameterUpperBound(controls.data(), 2 * i,     params_.follow_limits.vel_max);
        problem.SetParameterLowerBound(controls.data(), 2 * i + 1, params_.follow_limits.omega_min);
        problem.SetParameterUpperBound(controls.data(), 2 * i + 1, params_.follow_limits.omega_max);
    }

    // 求解
    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.max_solver_time_in_seconds = MPC_MAX_SOLVER_TIME;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable()) {
        return std::unexpected(summary.BriefReport());
    }

    // 输出
    Eigen::Vector2d cmd_v_omega(controls[0], controls[1]);
    last_cmd_ = cmd_v_omega;

    // 保存warm start
    last_controls_ = controls;

    const MPCPrediction prediction = generate_predicted_path_map(
        chassis_pose_map, chassis_status, start_cmd_clamped, x_h_hat_, controls.data()
    );
    return std::tuple{cmd_v_omega, prediction};
}

std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> MPCSolver::stop(
    const Eigen::Vector3d& chassis_pose_map,
    const Eigen::Vector2d& chassis_status,
    const CostMap& merged_cost_map,
    const DirectionMap& global_direction_map
) {
    const Eigen::Vector2d start_cmd = last_cmd_;
    const Eigen::Vector2d start_cmd_clamped(
        clamp_prev_cmd(start_cmd.x(), chassis_status.x(), params_.stop_limits.start_vel_cmd_act_diff_max, params_.stop_limits.acc_max, MPC_DT),
        clamp_prev_cmd(start_cmd.y(), chassis_status.y(), params_.stop_limits.start_omega_cmd_act_diff_max, params_.stop_limits.alpha_max, MPC_DT)
    );

    const CostMapGridView cost_grid(merged_cost_map);
    const GridInfo cost_info = make_grid_info(merged_cost_map);
    const DirectionMapGridView dir_grid(global_direction_map);
    const GridInfo dir_info = make_grid_info(global_direction_map);

    // 决策变量
    std::array<double, MPC_PARAM_SIZE> controls{};

    // Warm start
    for (int i = 0; i < MPC_HORIZON; i++) {
        const int src = (i + 1 < MPC_HORIZON) ? (i + 1) : (MPC_HORIZON - 1);
        controls[static_cast<size_t>(2 * i)]     = std::max(0.0, last_controls_[static_cast<size_t>(2 * src)]);
        controls[static_cast<size_t>(2 * i + 1)] = last_controls_[static_cast<size_t>(2 * src + 1)];
    }

    ceres::Problem problem;

    constexpr int STOP_NUM_RESIDUALS = 11 * MPC_HORIZON + 2;
    auto* cost_function = new ceres::AutoDiffCostFunction<
        StopMPCCostFunctor, STOP_NUM_RESIDUALS, MPC_PARAM_SIZE>(
        new StopMPCCostFunctor(
            chassis_pose_map,
            chassis_status,
            start_cmd_clamped,
            x_h_hat_,
            params_,
            cost_grid,
            cost_info,
            dir_grid,
            dir_info,
            remaining_energy_,
            rfr_pwr_limit_
        )
    );

    problem.AddResidualBlock(cost_function, nullptr, controls.data());

    // 边界
    for (int i = 0; i < MPC_HORIZON; i++) {
        problem.SetParameterLowerBound(controls.data(), 2 * i,     0.0);
        problem.SetParameterUpperBound(controls.data(), 2 * i,     params_.stop_limits.vel_max);
        problem.SetParameterLowerBound(controls.data(), 2 * i + 1, params_.stop_limits.omega_min);
        problem.SetParameterUpperBound(controls.data(), 2 * i + 1, params_.stop_limits.omega_max);
    }

    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.max_solver_time_in_seconds = MPC_MAX_SOLVER_TIME;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable()) {
        return std::unexpected(summary.BriefReport());
    }

    Eigen::Vector2d cmd_v_omega(controls[0], controls[1]);
    last_cmd_ = cmd_v_omega;

    // 保存warm start
    last_controls_ = controls;

    const MPCPrediction prediction = generate_predicted_path_map(
        chassis_pose_map, chassis_status, start_cmd_clamped, x_h_hat_, controls.data()
    );
    return std::tuple{cmd_v_omega, prediction};
}

std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> MPCSolver::recover_to_point(
    const Eigen::Vector2d& goal_map,
    const Eigen::Vector3d& chassis_pose_map,
    const Eigen::Vector2d& chassis_status,
    const CostMap& merged_cost_map,
    const DirectionMap& global_direction_map
) {
    const Eigen::Vector2d start_cmd = last_cmd_;
    const Eigen::Vector2d start_cmd_clamped(
        clamp_prev_cmd(start_cmd.x(), chassis_status.x(), params_.recovery_limits.start_vel_cmd_act_diff_max, params_.recovery_limits.acc_max, MPC_DT),
        clamp_prev_cmd(start_cmd.y(), chassis_status.y(), params_.recovery_limits.start_omega_cmd_act_diff_max, params_.recovery_limits.alpha_max, MPC_DT)
    );

    const CostMapGridView cost_grid(merged_cost_map);
    const GridInfo cost_info = make_grid_info(merged_cost_map);
    const DirectionMapGridView dir_grid(global_direction_map);
    const GridInfo dir_info = make_grid_info(global_direction_map);

    // 决策变量
    std::array<double, MPC_PARAM_SIZE> controls{};

    // Warm start
    for (int i = 0; i < MPC_HORIZON; i++) {
        const int src = (i + 1 < MPC_HORIZON) ? (i + 1) : (MPC_HORIZON - 1);
        controls[static_cast<size_t>(2 * i)]     = std::max(0.0, last_controls_[static_cast<size_t>(2 * src)]);
        controls[static_cast<size_t>(2 * i + 1)] = last_controls_[static_cast<size_t>(2 * src + 1)];
    }

    ceres::Problem problem;

    constexpr int RECOVERY_NUM_RESIDUALS = 13 * MPC_HORIZON + 4;
    auto* cost_function = new ceres::AutoDiffCostFunction<
        RecoveryMPCCostFunctor, RECOVERY_NUM_RESIDUALS, MPC_PARAM_SIZE>(
        new RecoveryMPCCostFunctor(
            goal_map,
            chassis_pose_map,
            chassis_status,
            start_cmd_clamped,
            x_h_hat_,
            params_,
            cost_grid,
            cost_info,
            dir_grid,
            dir_info,
            remaining_energy_,
            rfr_pwr_limit_
        )
    );

    problem.AddResidualBlock(cost_function, nullptr, controls.data());

    // 边界
    for (int i = 0; i < MPC_HORIZON; i++) {
        problem.SetParameterLowerBound(controls.data(), 2 * i,     params_.recovery_limits.vel_min);
        problem.SetParameterUpperBound(controls.data(), 2 * i,     params_.recovery_limits.vel_max);
        problem.SetParameterLowerBound(controls.data(), 2 * i + 1, params_.recovery_limits.omega_min);
        problem.SetParameterUpperBound(controls.data(), 2 * i + 1, params_.recovery_limits.omega_max);
    }

    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.max_solver_time_in_seconds = MPC_MAX_SOLVER_TIME;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable()) {
        return std::unexpected(summary.BriefReport());
    }

    Eigen::Vector2d cmd_v_omega(controls[0], controls[1]);
    last_cmd_ = cmd_v_omega;

    // 保存warm start
    last_controls_ = controls;

    const MPCPrediction prediction = generate_predicted_path_map(
        chassis_pose_map, chassis_status, start_cmd_clamped, x_h_hat_, controls.data()
    );
    return std::tuple{cmd_v_omega, prediction};
}

}