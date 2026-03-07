#include <ceres/ceres.h>

#include <algorithm>
#include <array>
#include <cmath>

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
inline T clamp01(const T& v) {
    return ceres::fmin(T(1.0), ceres::fmax(T(0.0), v));
}

template<typename T>
inline T normalize_angle(const T& angle) {
    return ceres::atan2(ceres::sin(angle), ceres::cos(angle));
}

template<typename T>
inline T smoothstep01(const T& t_in) {
    const T t = clamp01(t_in);
    return t * t * (T(3.0) - T(2.0) * t);
}

template<typename T>
inline T smoothstep(const T& x, const T& edge0, const T& edge1) {
    const T denom = ceres::fmax(T(1e-12), edge1 - edge0);
    return smoothstep01((x - edge0) / denom);
}

template<typename T>
inline T softplus(const T& x) {
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

    const double scale_d = static_cast<double>(n - 2);

    const T u = clamp01(u_in);
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

    const T scale = T(scale_d);

    if (d1) {
        const Eigen::Matrix<T, 2, 1> dp_dt = db0_dt * p0 + db1_dt * p1 + db2_dt * p2;
        *d1 = dp_dt * scale;
    }

    if (d2) {
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

using ArclengthTable = std::array<double, ARCLENGTH_TABLE_SIZE + 1>;

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
//                           零拷贝双线性采样
// ============================================================================

struct GridInfo {
    double origin_x;
    double origin_y;
    double resolution;
    int width;
    int height;
};

template<typename T>
inline T eval_cost_bilinear(
    const CostMap& cost_map,
    const GridInfo& info,
    const T& x_map,
    const T& y_map
) {
    if (info.width < 2 || info.height < 2) {
        return T(255.0);
    }

    const T gx = (x_map - T(info.origin_x)) / T(info.resolution);
    const T gy = (y_map - T(info.origin_y)) / T(info.resolution);
    const double gxv = scalar_value(gx);
    const double gyv = scalar_value(gy);
    if (gxv < 0.0 || gyv < 0.0 ||
        gxv >= static_cast<double>(info.width - 1) ||
        gyv >= static_cast<double>(info.height - 1)) {
        return T(255.0);
    }

    const int x0 = clamp_int(static_cast<int>(std::floor(gxv)), 0, info.width - 2);
    const int y0 = clamp_int(static_cast<int>(std::floor(gyv)), 0, info.height - 2);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const T dx = gx - T(x0);
    const T dy = gy - T(y0);

    const auto at = [&](const int x, const int y) {
        return T(cost_map.data[static_cast<size_t>(y * info.width + x)]);
    };

    return (T(1.0) - dx) * (T(1.0) - dy) * at(x0, y0)
        + dx * (T(1.0) - dy) * at(x1, y0)
        + (T(1.0) - dx) * dy * at(x0, y1)
        + dx * dy * at(x1, y1);
}

template<typename T>
inline Eigen::Matrix<T, 2, 1> eval_dir_bilinear(
    const DirectionMap& dir_map,
    const GridInfo& info,
    const T& x_map,
    const T& y_map
) {
    if (info.width < 2 || info.height < 2) {
        return Eigen::Matrix<T, 2, 1>(T(0.0), T(0.0));
    }

    const T gx = (x_map - T(info.origin_x)) / T(info.resolution);
    const T gy = (y_map - T(info.origin_y)) / T(info.resolution);
    const double gxv = scalar_value(gx);
    const double gyv = scalar_value(gy);
    if (gxv < 0.0 || gyv < 0.0 ||
        gxv >= static_cast<double>(info.width - 1) ||
        gyv >= static_cast<double>(info.height - 1)) {
        return Eigen::Matrix<T, 2, 1>(T(0.0), T(0.0));
    }

    const int x0 = clamp_int(static_cast<int>(std::floor(gxv)), 0, info.width - 2);
    const int y0 = clamp_int(static_cast<int>(std::floor(gyv)), 0, info.height - 2);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const T dx = gx - T(x0);
    const T dy = gy - T(y0);

    const auto at = [&](const int x, const int y) {
        const Eigen::Vector2d& dir = dir_map.data[static_cast<size_t>(y * info.width + x)];
        return Eigen::Matrix<T, 2, 1>(T(dir.x()), T(dir.y()));
    };

    return (T(1.0) - dx) * (T(1.0) - dy) * at(x0, y0)
        + dx * (T(1.0) - dy) * at(x1, y0)
        + (T(1.0) - dx) * dy * at(x0, y1)
        + dx * dy * at(x1, y1);
}

// ============================================================================
//                                 MPC预测模型
// ============================================================================

namespace {
enum StateIndex : int {
    IDX_X = 0,
    IDX_Y = 1,
    IDX_THETA = 2,
    IDX_XH = 3,
    IDX_V_ACT = 4,
    IDX_W_ACT = 5,
    IDX_V_CMD_Z1 = 6,
    IDX_W_CMD_Z1 = 7,
    IDX_PATH_U = 8,
    IDX_ENERGY = 9
};

using ControlBlock = std::array<double, MPC_CONTROL_SIZE>;
using ControlTrajectory = std::array<ControlBlock, MPC_HORIZON>;
using StateBlock = std::array<double, MPC_STATE_SIZE>;
using StateTrajectory = std::array<StateBlock, MPC_HORIZON + 1>;

template<typename T>
using ShootingState = Eigen::Matrix<T, MPC_STATE_SIZE, 1>;

template<typename T>
inline ShootingState<T> load_state(const T* block) {
    ShootingState<T> state;
    for (int i = 0; i < MPC_STATE_SIZE; i++) {
        state(i) = block[i];
    }
    return state;
}

template<typename T>
inline void store_state(const ShootingState<T>& state, T* block) {
    for (int i = 0; i < MPC_STATE_SIZE; i++) {
        block[i] = state(i);
    }
}

template<typename T>
inline T sabs(const T& x) {
    return ceres::sqrt(x * x + T(PWR_EPS2));
}

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
inline T smooth_sgn(const T& x, const double eps) {
    const T e = ceres::fmax(T(1e-6), T(eps));
    return ceres::tanh(x / e);
}

template<typename T>
inline void model_step(
    const T& xh,
    const T& v,
    const T& w,
    const T& v_cmd_z1,
    const T& w_cmd_z1,
    T* xh_next,
    T* v_next,
    T* w_next
) {
    *xh_next = T(A00) * xh + T(A01) * v + T(A03) * v_cmd_z1;
    *v_next = T(A10) * xh + T(A11) * v + T(A13) * v_cmd_z1;
    *w_next = T(A22) * w + T(A24) * w_cmd_z1;

    const T sv = smooth_sgn(v, SGN_EPS);
    const T sw = smooth_sgn(w, SGN_EPS);
    const T nl_v = T(CF1) * sv + T(CF2) * v * ceres::abs(w);

    *xh_next += T(GNL_XH) * nl_v;
    *v_next  += T(GNL_V)  * nl_v;
    *w_next  += -T(GAMMA_W) * T(CF3) * sw;
}

template<typename T>
struct PathReferenceSample {
    Eigen::Matrix<T, 2, 1> position;
    Eigen::Matrix<T, 2, 1> d1;
    Eigen::Matrix<T, 2, 1> d2;
    T theta;
    T dsdu;
    T kappa;
};

template<typename T>
inline PathReferenceSample<T> sample_path_reference(
    const std::vector<Eigen::Vector2d>& ref_control_points,
    const T& u_in
) {
    PathReferenceSample<T> sample;
    const T u = clamp01(u_in);
    eval_quadratic_uniform_bspline_2d<T>(ref_control_points, u, &sample.position, &sample.d1, &sample.d2);
    const T dx = sample.d1.x();
    const T dy = sample.d1.y();
    const T ddx = sample.d2.x();
    const T ddy = sample.d2.y();
    sample.theta = ceres::atan2(dy, dx);
    sample.dsdu = ceres::sqrt(dx * dx + dy * dy) + T(1e-6);
    sample.kappa = (dx * ddy - dy * ddx) / (sample.dsdu * sample.dsdu * sample.dsdu);
    return sample;
}

template<typename T>
inline ShootingState<T> propagate_state(
    const ShootingState<T>& current,
    const T& v_cmd,
    const T& w_cmd,
    const std::vector<Eigen::Vector2d>* ref_control_points,
    const double rfr_pwr_limit
) {
    ShootingState<T> next = current;

    T xh_next;
    T v_next;
    T w_next;
    model_step(
        current(IDX_XH),
        current(IDX_V_ACT),
        current(IDX_W_ACT),
        current(IDX_V_CMD_Z1),
        current(IDX_W_CMD_Z1),
        &xh_next,
        &v_next,
        &w_next
    );

    const T theta0 = current(IDX_THETA);
    const T theta1 = theta0 + (current(IDX_W_ACT) + w_next) * (T(MPC_DT) * T(0.5));

    next(IDX_X) = current(IDX_X) + (current(IDX_V_ACT) * ceres::cos(theta0) + v_next * ceres::cos(theta1)) * (T(MPC_DT) * T(0.5));
    next(IDX_Y) = current(IDX_Y) + (current(IDX_V_ACT) * ceres::sin(theta0) + v_next * ceres::sin(theta1)) * (T(MPC_DT) * T(0.5));
    next(IDX_THETA) = theta1;
    next(IDX_XH) = xh_next;
    next(IDX_V_ACT) = v_next;
    next(IDX_W_ACT) = w_next;
    next(IDX_V_CMD_Z1) = v_cmd;
    next(IDX_W_CMD_Z1) = w_cmd;

    const T a_k = (v_next - current(IDX_V_ACT)) / T(MPC_DT);
    const T alpha_k = (w_next - current(IDX_W_ACT)) / T(MPC_DT);
    next(IDX_ENERGY) = current(IDX_ENERGY) + (T(rfr_pwr_limit) - predict_power(v_next, w_next, a_k, alpha_k)) * T(MPC_DT);

    if (ref_control_points && ref_control_points->size() >= 3) {
        const T u = clamp01(current(IDX_PATH_U));
        const auto ref = sample_path_reference(*ref_control_points, u);
        const T ex = next(IDX_X) - ref.position.x();
        const T ey_world = next(IDX_Y) - ref.position.y();
        const T ey = -ex * ceres::sin(ref.theta) + ey_world * ceres::cos(ref.theta);
        const T etheta = normalize_angle(next(IDX_THETA) - ref.theta);
        T denom = T(1.0) - ref.kappa * ey;
        const T denom_abs = ceres::abs(denom);
        denom = denom / (denom_abs + T(1e-6)) * ceres::fmax(denom_abs, T(0.1));
        const T dsdt = next(IDX_V_ACT) * ceres::cos(etheta) / denom;
        const T dudt = dsdt / ref.dsdu;
        next(IDX_PATH_U) = clamp01(u + dudt * T(MPC_DT));
    }

    return next;
}

inline ControlTrajectory warm_start_controls(const std::array<double, MPC_PARAM_SIZE>& last_controls, const bool nonnegative_v) {
    ControlTrajectory controls{};
    for (int i = 0; i < MPC_HORIZON; i++) {
        const int src = (i + 1 < MPC_HORIZON) ? (i + 1) : (MPC_HORIZON - 1);
        controls[static_cast<size_t>(i)][0] = last_controls[static_cast<size_t>(2 * src)];
        controls[static_cast<size_t>(i)][1] = last_controls[static_cast<size_t>(2 * src + 1)];
        if (nonnegative_v) {
            controls[static_cast<size_t>(i)][0] = std::max(0.0, controls[static_cast<size_t>(i)][0]);
        }
    }
    return controls;
}

inline void flatten_controls(const ControlTrajectory& controls, std::array<double, MPC_PARAM_SIZE>* flat) {
    for (int i = 0; i < MPC_HORIZON; i++) {
        (*flat)[static_cast<size_t>(2 * i)] = controls[static_cast<size_t>(i)][0];
        (*flat)[static_cast<size_t>(2 * i + 1)] = controls[static_cast<size_t>(i)][1];
    }
}

inline StateTrajectory initialize_states_from_controls(
    const StateBlock& initial_state,
    const ControlTrajectory& controls,
    const std::vector<Eigen::Vector2d>* ref_control_points,
    const double rfr_pwr_limit
) {
    StateTrajectory states{};
    states[0] = initial_state;

    for (int k = 0; k < MPC_HORIZON; k++) {
        ShootingState<double> current = load_state(states[static_cast<size_t>(k)].data());
        const auto next = propagate_state(
            current,
            controls[static_cast<size_t>(k)][0],
            controls[static_cast<size_t>(k)][1],
            ref_control_points,
            rfr_pwr_limit
        );
        store_state(next, states[static_cast<size_t>(k + 1)].data());
    }

    return states;
}

inline MPCPrediction prediction_from_states(const StateTrajectory& states) {
    MPCPrediction pred;
    pred.path_map.reserve(states.size());
    pred.headings.reserve(states.size());
    pred.v_pred.reserve(states.size());
    pred.w_pred.reserve(states.size());

    for (const auto& state : states) {
        pred.path_map.emplace_back(state[IDX_X], state[IDX_Y]);
        pred.headings.push_back(state[IDX_THETA]);
        pred.v_pred.push_back(state[IDX_V_ACT]);
        pred.w_pred.push_back(state[IDX_W_ACT]);
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

inline StateBlock build_initial_state(
    const Eigen::Vector3d& chassis_pose_map,
    const Eigen::Vector2d& chassis_status,
    const Eigen::Vector2d& start_cmd_clamped,
    const double x_h_init,
    const double u0,
    const double remaining_energy
) {
    StateBlock initial{};
    initial[IDX_X] = chassis_pose_map.x();
    initial[IDX_Y] = chassis_pose_map.y();
    initial[IDX_THETA] = chassis_pose_map.z();
    initial[IDX_XH] = x_h_init;
    initial[IDX_V_ACT] = chassis_status.x();
    initial[IDX_W_ACT] = chassis_status.y();
    initial[IDX_V_CMD_Z1] = start_cmd_clamped.x();
    initial[IDX_W_CMD_Z1] = start_cmd_clamped.y();
    initial[IDX_PATH_U] = u0;
    initial[IDX_ENERGY] = remaining_energy;
    return initial;
}

inline ceres::Solver::Options make_solver_options() {
    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = MPC_MAX_ITERATIONS;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;
    options.num_threads = 4;
    return options;
}

struct DynamicsFunctor {
    DynamicsFunctor(
        const std::vector<Eigen::Vector2d>* ref_control_points,
        const double rfr_pwr_limit
    ):
        ref_control_points_(ref_control_points),
        rfr_pwr_limit_(rfr_pwr_limit) {}

    template<typename T>
    bool operator()(const T* const state_k, const T* const control_k, const T* const state_k1, T* residuals) const {
        const auto current = load_state(state_k);
        const auto predicted = propagate_state(current, control_k[0], control_k[1], ref_control_points_, rfr_pwr_limit_);
        const auto next = load_state(state_k1);

        for (int i = 0; i < MPC_STATE_SIZE; i++) {
            residuals[i] = T(DYNAMICS_WEIGHTS[static_cast<size_t>(i)]) * (next(i) - predicted(i));
        }
        return true;
    }

    const std::vector<Eigen::Vector2d>* ref_control_points_;
    const double rfr_pwr_limit_;
};

struct FollowCostFunctor {
    FollowCostFunctor(
        const std::vector<Eigen::Vector2d>& ref_control_points,
        const MPCParams& params,
        const CostMap& cost_map,
        const GridInfo& cost_info,
        const DirectionMap& dir_map,
        const GridInfo& dir_info,
        const ArclengthTable& arclength_table
    ):
        ref_control_points_(ref_control_points),
        params_(params),
        cost_map_(cost_map),
        cost_info_(cost_info),
        dir_map_(dir_map),
        dir_info_(dir_info),
        arclength_table_(arclength_table) {}

    template<typename T>
    bool operator()(const T* const state_k, const T* const control_k, const T* const state_k1, T* residuals) const {
        const auto current = load_state(state_k);
        const auto next = load_state(state_k1);
        const T v_cmd = control_k[0];
        const T omega_cmd = control_k[1];
        const T dv_cmd = v_cmd - current(IDX_V_CMD_Z1);
        const T domega_cmd = omega_cmd - current(IDX_W_CMD_Z1);

        const T u = clamp01(current(IDX_PATH_U));
        const auto ref = sample_path_reference(ref_control_points_, u);
        const double s_remain_d = lookup_remaining_arclength(arclength_table_, scalar_value(u));
        const T s_remain = T(s_remain_d);

        const T ex = next(IDX_X) - ref.position.x();
        const T ey_world = next(IDX_Y) - ref.position.y();
        const T ey = -ex * ceres::sin(ref.theta) + ey_world * ceres::cos(ref.theta);
        const T etheta = normalize_angle(next(IDX_THETA) - ref.theta);

        size_t res_idx = 0;
        residuals[res_idx++] = T(params_.follow_weights.q_y) * ey;
        residuals[res_idx++] = T(params_.follow_weights.q_theta) * etheta;
        residuals[res_idx++] = T(params_.follow_weights.q_u) * (T(1.0) - u);
        residuals[res_idx++] = T(params_.follow_weights.r_v) * v_cmd;
        residuals[res_idx++] = T(params_.follow_weights.r_omega) * omega_cmd;
        residuals[res_idx++] = T(params_.follow_weights.r_dv) * dv_cmd;
        residuals[res_idx++] = T(params_.follow_weights.r_domega) * domega_cmd;

        const T dv_cmd_limit = T(params_.follow_limits.acc_max * MPC_DT);
        const T domega_cmd_limit = T(params_.follow_limits.alpha_max * MPC_DT);
        residuals[res_idx++] = T(params_.follow_weights.acc_limit) * ceres::fmax(T(0.0), ceres::abs(dv_cmd) - dv_cmd_limit);
        residuals[res_idx++] = T(params_.follow_weights.alpha_limit) * ceres::fmax(T(0.0), ceres::abs(domega_cmd) - domega_cmd_limit);

        const T a_lat = ceres::abs(next(IDX_V_ACT) * next(IDX_W_ACT));
        residuals[res_idx++] = T(params_.follow_weights.lat_acc) * ceres::fmax(T(0.0), a_lat - T(params_.follow_limits.a_lat_max));

        const T cost = eval_cost_bilinear(cost_map_, cost_info_, next(IDX_X), next(IDX_Y));
        residuals[res_idx++] = T(params_.follow_weights.obstacle) * (cost / T(255.0));

        const auto dir = eval_dir_bilinear(dir_map_, dir_info_, next(IDX_X), next(IDX_Y));
        const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-10));
        const T step_gate = smoothstep(
            dir_norm,
            T(params_.follow_limits.step_norm_threshold),
            T(params_.follow_limits.step_norm_threshold + params_.follow_limits.step_norm_transition)
        );
        residuals[res_idx++] = T(params_.follow_weights.step) * step_gate;

        const Eigen::Matrix<T, 2, 1> heading(ceres::cos(next(IDX_THETA)), ceres::sin(next(IDX_THETA)));
        const T heading_cross_dir = heading.x() * dir.y() - heading.y() * dir.x();
        residuals[res_idx++] = T(params_.follow_weights.direction) * ceres::abs(heading_cross_dir);

        const auto dir_unit = dir / (dir_norm + T(1e-6));
        const T cos_theta = heading.dot(dir_unit);
        const T weight_up = (cos_theta + T(1.0)) / T(2.0);
        const T target_vel_step = weight_up * T(params_.follow_limits.vel_step_up)
                                + (T(1.0) - weight_up) * T(params_.follow_limits.vel_step_down);
        residuals[res_idx++] = T(params_.follow_weights.vel_on_step) * dir_norm * ceres::abs(next(IDX_V_ACT) - target_vel_step);

        const T deceleration = T(params_.follow_limits.slow_down_deceleration);
        const T v_dec_profile = ceres::sqrt(T(2.0) * deceleration * s_remain + T(0.01));
        residuals[res_idx++] = T(params_.follow_weights.q_v_final) * ceres::fmax(T(0.0), next(IDX_V_ACT) - v_dec_profile);

        const T w_e = T(params_.energy.enable ? params_.energy.weight : 0.0);
        const T thr = T(std::max(params_.energy.threshold, 1.0));
        const T beta = T(std::max(params_.energy.softplus_beta, 1e-6));
        const T vio = (T(params_.energy.threshold) - next(IDX_ENERGY)) / thr;
        const T relu_soft = softplus(beta * vio) / beta - softplus(T(0.0)) / beta;
        residuals[res_idx++] = w_e * ceres::fmax(T(0.0), relu_soft);
        return true;
    }

    const std::vector<Eigen::Vector2d>& ref_control_points_;
    const MPCParams& params_;
    const CostMap& cost_map_;
    const GridInfo cost_info_;
    const DirectionMap& dir_map_;
    const GridInfo dir_info_;
    const ArclengthTable arclength_table_;
};

struct StopCostFunctor {
    StopCostFunctor(
        const MPCParams& params,
        const CostMap& cost_map,
        const GridInfo& cost_info,
        const DirectionMap& dir_map,
        const GridInfo& dir_info
    ):
        params_(params),
        cost_map_(cost_map),
        cost_info_(cost_info),
        dir_map_(dir_map),
        dir_info_(dir_info) {}

    template<typename T>
    bool operator()(const T* const state_k, const T* const control_k, const T* const state_k1, T* residuals) const {
        const auto current = load_state(state_k);
        const auto next = load_state(state_k1);
        const T dv_cmd = control_k[0] - current(IDX_V_CMD_Z1);
        const T domega_cmd = control_k[1] - current(IDX_W_CMD_Z1);

        size_t res_idx = 0;
        residuals[res_idx++] = T(params_.stop_weights.q_v) * next(IDX_V_ACT);
        residuals[res_idx++] = T(params_.stop_weights.q_omega) * next(IDX_W_ACT);
        residuals[res_idx++] = T(params_.stop_weights.r_dv) * dv_cmd;
        residuals[res_idx++] = T(params_.stop_weights.r_domega) * domega_cmd;

        const T dv_cmd_limit = T(params_.stop_limits.acc_max * MPC_DT);
        const T domega_cmd_limit = T(params_.stop_limits.alpha_max * MPC_DT);
        residuals[res_idx++] = T(params_.stop_weights.acc_limit) * ceres::fmax(T(0.0), ceres::abs(dv_cmd) - dv_cmd_limit);
        residuals[res_idx++] = T(params_.stop_weights.alpha_limit) * ceres::fmax(T(0.0), ceres::abs(domega_cmd) - domega_cmd_limit);

        const T a_lat = ceres::abs(next(IDX_V_ACT) * next(IDX_W_ACT));
        residuals[res_idx++] = T(params_.stop_weights.lat_acc) * ceres::fmax(T(0.0), a_lat - T(params_.stop_limits.a_lat_max));

        const T cost = eval_cost_bilinear(cost_map_, cost_info_, next(IDX_X), next(IDX_Y));
        residuals[res_idx++] = T(params_.stop_weights.obstacle) * (cost / T(255.0));

        const auto dir = eval_dir_bilinear(dir_map_, dir_info_, next(IDX_X), next(IDX_Y));
        const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-10));
        const T step_gate = smoothstep(
            dir_norm,
            T(params_.stop_limits.step_norm_threshold),
            T(params_.stop_limits.step_norm_threshold + params_.stop_limits.step_norm_transition)
        );

        const Eigen::Matrix<T, 2, 1> heading(ceres::cos(next(IDX_THETA)), ceres::sin(next(IDX_THETA)));
        const T heading_cross_dir = heading.x() * dir.y() - heading.y() * dir.x();
        residuals[res_idx++] = T(params_.stop_weights.direction) * step_gate * ceres::abs(heading_cross_dir);

        const auto dir_unit = dir / (dir_norm + T(1e-6));
        const T cos_theta = heading.dot(dir_unit);
        const T weight_up = (cos_theta + T(1.0)) / T(2.0);
        const T target_vel_step = weight_up * T(params_.stop_limits.vel_step_up)
                                + (T(1.0) - weight_up) * T(params_.stop_limits.vel_step_down);
        residuals[res_idx++] = T(params_.stop_weights.vel_on_step) * step_gate * dir_norm * ceres::abs(next(IDX_V_ACT) - target_vel_step);

        const T w_e = T(params_.energy.enable ? params_.energy.weight : 0.0);
        const T thr = T(std::max(params_.energy.threshold, 1.0));
        const T beta = T(std::max(params_.energy.softplus_beta, 1e-6));
        const T vio = (T(params_.energy.threshold) - next(IDX_ENERGY)) / thr;
        const T relu_soft = softplus(beta * vio) / beta - softplus(T(0.0)) / beta;
        residuals[res_idx++] = w_e * ceres::fmax(T(0.0), relu_soft);
        return true;
    }

    const MPCParams& params_;
    const CostMap& cost_map_;
    const GridInfo cost_info_;
    const DirectionMap& dir_map_;
    const GridInfo dir_info_;
};

struct StopTerminalCostFunctor {
    StopTerminalCostFunctor(
        const MPCParams& params,
        const CostMap& cost_map,
        const GridInfo& cost_info,
        const DirectionMap& dir_map,
        const GridInfo& dir_info
    ):
        params_(params),
        cost_map_(cost_map),
        cost_info_(cost_info),
        dir_map_(dir_map),
        dir_info_(dir_info) {}

    template<typename T>
    bool operator()(const T* const state_n, T* residuals) const {
        const auto state = load_state(state_n);
        const T cost_terminal = eval_cost_bilinear(cost_map_, cost_info_, state(IDX_X), state(IDX_Y));
        residuals[0] = T(params_.stop_weights.obstacle_terminal) * (cost_terminal / T(255.0));

        const auto dir = eval_dir_bilinear(dir_map_, dir_info_, state(IDX_X), state(IDX_Y));
        const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-10));
        const T step_gate = smoothstep(
            dir_norm,
            T(params_.stop_limits.step_norm_threshold),
            T(params_.stop_limits.step_norm_threshold + params_.stop_limits.step_norm_transition)
        );
        residuals[1] = T(params_.stop_weights.step_terminal) * step_gate;
        return true;
    }

    const MPCParams& params_;
    const CostMap& cost_map_;
    const GridInfo cost_info_;
    const DirectionMap& dir_map_;
    const GridInfo dir_info_;
};

struct RecoveryCostFunctor {
    RecoveryCostFunctor(
        const Eigen::Vector2d& goal_map,
        const MPCParams& params,
        const CostMap& cost_map,
        const GridInfo& cost_info,
        const DirectionMap& dir_map,
        const GridInfo& dir_info
    ):
        goal_map_(goal_map),
        params_(params),
        cost_map_(cost_map),
        cost_info_(cost_info),
        dir_map_(dir_map),
        dir_info_(dir_info) {}

    template<typename T>
    bool operator()(const T* const state_k, const T* const control_k, const T* const state_k1, T* residuals) const {
        const auto current = load_state(state_k);
        const auto next = load_state(state_k1);
        const T dv_cmd = control_k[0] - current(IDX_V_CMD_Z1);
        const T domega_cmd = control_k[1] - current(IDX_W_CMD_Z1);
        const T gx = T(goal_map_.x());
        const T gy = T(goal_map_.y());

        size_t res_idx = 0;
        residuals[res_idx++] = T(params_.recovery_weights.q_goal_xy) * (next(IDX_X) - gx);
        residuals[res_idx++] = T(params_.recovery_weights.q_goal_xy) * (next(IDX_Y) - gy);

        const T desired_theta = ceres::atan2(gy - next(IDX_Y), gx - next(IDX_X));
        residuals[res_idx++] = T(params_.recovery_weights.q_goal_theta) * ceres::abs(ceres::sin(next(IDX_THETA) - desired_theta));

        residuals[res_idx++] = T(params_.recovery_weights.r_v) * control_k[0];
        residuals[res_idx++] = T(params_.recovery_weights.r_omega) * control_k[1];
        residuals[res_idx++] = T(params_.recovery_weights.r_dv) * dv_cmd;
        residuals[res_idx++] = T(params_.recovery_weights.r_domega) * domega_cmd;

        const T dv_cmd_limit = T(params_.recovery_limits.acc_max * MPC_DT);
        const T domega_cmd_limit = T(params_.recovery_limits.alpha_max * MPC_DT);
        residuals[res_idx++] = T(params_.recovery_weights.acc_limit) * ceres::fmax(T(0.0), ceres::abs(dv_cmd) - dv_cmd_limit);
        residuals[res_idx++] = T(params_.recovery_weights.alpha_limit) * ceres::fmax(T(0.0), ceres::abs(domega_cmd) - domega_cmd_limit);

        const T a_lat = ceres::abs(next(IDX_V_ACT) * next(IDX_W_ACT));
        residuals[res_idx++] = T(params_.recovery_weights.lat_acc) * ceres::fmax(T(0.0), a_lat - T(params_.recovery_limits.a_lat_max));

        const T cost = eval_cost_bilinear(cost_map_, cost_info_, next(IDX_X), next(IDX_Y));
        residuals[res_idx++] = T(params_.recovery_weights.obstacle) * (cost / T(255.0));

        const auto dir = eval_dir_bilinear(dir_map_, dir_info_, next(IDX_X), next(IDX_Y));
        const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-10));
        const T step_gate = smoothstep(
            dir_norm,
            T(params_.recovery_limits.step_norm_threshold),
            T(params_.recovery_limits.step_norm_threshold + params_.recovery_limits.step_norm_transition)
        );
        residuals[res_idx++] = T(params_.recovery_weights.step) * step_gate;

        const T w_e = T(params_.energy.enable ? params_.energy.weight : 0.0);
        const T thr = T(std::max(params_.energy.threshold, 1.0));
        const T beta = T(std::max(params_.energy.softplus_beta, 1e-6));
        const T vio = (T(params_.energy.threshold) - next(IDX_ENERGY)) / thr;
        const T relu_soft = softplus(beta * vio) / beta - softplus(T(0.0)) / beta;
        residuals[res_idx++] = w_e * ceres::fmax(T(0.0), relu_soft);
        return true;
    }

    const Eigen::Vector2d& goal_map_;
    const MPCParams& params_;
    const CostMap& cost_map_;
    const GridInfo cost_info_;
    const DirectionMap& dir_map_;
    const GridInfo dir_info_;
};

struct RecoveryTerminalCostFunctor {
    RecoveryTerminalCostFunctor(
        const Eigen::Vector2d& goal_map,
        const MPCParams& params,
        const CostMap& cost_map,
        const GridInfo& cost_info,
        const DirectionMap& dir_map,
        const GridInfo& dir_info
    ):
        goal_map_(goal_map),
        params_(params),
        cost_map_(cost_map),
        cost_info_(cost_info),
        dir_map_(dir_map),
        dir_info_(dir_info) {}

    template<typename T>
    bool operator()(const T* const state_n, T* residuals) const {
        const auto state = load_state(state_n);
        residuals[0] = T(params_.recovery_weights.q_goal_xy_terminal) * (state(IDX_X) - T(goal_map_.x()));
        residuals[1] = T(params_.recovery_weights.q_goal_xy_terminal) * (state(IDX_Y) - T(goal_map_.y()));

        const T cost_terminal = eval_cost_bilinear(cost_map_, cost_info_, state(IDX_X), state(IDX_Y));
        residuals[2] = T(params_.recovery_weights.obstacle_terminal) * (cost_terminal / T(255.0));

        const auto dir = eval_dir_bilinear(dir_map_, dir_info_, state(IDX_X), state(IDX_Y));
        const T dir_norm = ceres::sqrt(dir.squaredNorm() + T(1e-10));
        const T step_gate = smoothstep(
            dir_norm,
            T(params_.recovery_limits.step_norm_threshold),
            T(params_.recovery_limits.step_norm_threshold + params_.recovery_limits.step_norm_transition)
        );
        residuals[3] = T(params_.recovery_weights.step_terminal) * step_gate;
        return true;
    }

    const Eigen::Vector2d& goal_map_;
    const MPCParams& params_;
    const CostMap& cost_map_;
    const GridInfo cost_info_;
    const DirectionMap& dir_map_;
    const GridInfo dir_info_;
};

}  // namespace

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
        x_h_hat_ = XH0;
        prev_v_act_ = v_act;
        prev_w_act_ = w_act;
        observer_initialized_ = true;
        return;
    }

    const double v_prev = prev_v_act_;
    const double w_prev = prev_w_act_;
    const double vc_prev = last_cmd_.x();

    const double sv_prev = std::tanh(v_prev / SGN_EPS);
    const double nl_prev = CF1 * sv_prev + CF2 * v_prev * std::abs(w_prev);

    const double xh_pred = A00 * x_h_hat_ + A01 * v_prev + A03 * vc_prev + GNL_XH * nl_prev;
    const double v_pred  = A10 * x_h_hat_ + A11 * v_prev + A13 * vc_prev + GNL_V  * nl_prev;

    const double innovation = v_act - v_pred;
    x_h_hat_ = xh_pred + OBS_L * innovation;

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

    const GridInfo cost_info{merged_cost_map.origin_x, merged_cost_map.origin_y, merged_cost_map.resolution, merged_cost_map.width, merged_cost_map.height};
    const GridInfo dir_info{global_direction_map.origin_x, global_direction_map.origin_y, global_direction_map.resolution, global_direction_map.width, global_direction_map.height};
    const ArclengthTable arclength_table = build_arclength_table(global_path.getControlPoints(), params_.follow_limits.slow_down_num_samples);

    ControlTrajectory controls = warm_start_controls(last_controls_, false);
    const StateBlock initial_state = build_initial_state(
        chassis_pose_map,
        chassis_status,
        start_cmd_clamped,
        x_h_hat_,
        u0,
        remaining_energy_
    );
    const auto& ref_control_points = global_path.getControlPoints();
    StateTrajectory states = initialize_states_from_controls(initial_state, controls, &ref_control_points, rfr_pwr_limit_);

    ceres::Problem problem;
    problem.AddParameterBlock(states[0].data(), MPC_STATE_SIZE);
    problem.SetParameterBlockConstant(states[0].data());

    for (int k = 0; k < MPC_HORIZON; k++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<DynamicsFunctor, MPC_STATE_SIZE, MPC_STATE_SIZE, MPC_CONTROL_SIZE, MPC_STATE_SIZE>(
                new DynamicsFunctor(&ref_control_points, rfr_pwr_limit_)
            ),
            nullptr,
            states[static_cast<size_t>(k)].data(),
            controls[static_cast<size_t>(k)].data(),
            states[static_cast<size_t>(k + 1)].data()
        );

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<FollowCostFunctor, 16, MPC_STATE_SIZE, MPC_CONTROL_SIZE, MPC_STATE_SIZE>(
                new FollowCostFunctor(
                    ref_control_points,
                    params_,
                    merged_cost_map,
                    cost_info,
                    global_direction_map,
                    dir_info,
                    arclength_table
                )
            ),
            nullptr,
            states[static_cast<size_t>(k)].data(),
            controls[static_cast<size_t>(k)].data(),
            states[static_cast<size_t>(k + 1)].data()
        );

        problem.SetParameterLowerBound(controls[static_cast<size_t>(k)].data(), 0, params_.follow_limits.vel_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(k)].data(), 0, params_.follow_limits.vel_max);
        problem.SetParameterLowerBound(controls[static_cast<size_t>(k)].data(), 1, params_.follow_limits.omega_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(k)].data(), 1, params_.follow_limits.omega_max);
    }

    ceres::Solver::Summary summary;
    ceres::Solver::Options options = make_solver_options();
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable()) {
        return std::unexpected(summary.BriefReport());
    }

    flatten_controls(controls, &last_controls_);
    const Eigen::Vector2d cmd_v_omega(controls[0][0], controls[0][1]);
    last_cmd_ = cmd_v_omega;
    return std::tuple{cmd_v_omega, prediction_from_states(states)};
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

    const GridInfo cost_info{merged_cost_map.origin_x, merged_cost_map.origin_y, merged_cost_map.resolution, merged_cost_map.width, merged_cost_map.height};
    const GridInfo dir_info{global_direction_map.origin_x, global_direction_map.origin_y, global_direction_map.resolution, global_direction_map.width, global_direction_map.height};

    ControlTrajectory controls = warm_start_controls(last_controls_, true);
    const StateBlock initial_state = build_initial_state(
        chassis_pose_map,
        chassis_status,
        start_cmd_clamped,
        x_h_hat_,
        0.0,
        remaining_energy_
    );
    StateTrajectory states = initialize_states_from_controls(initial_state, controls, nullptr, rfr_pwr_limit_);

    ceres::Problem problem;
    problem.AddParameterBlock(states[0].data(), MPC_STATE_SIZE);
    problem.SetParameterBlockConstant(states[0].data());

    for (int k = 0; k < MPC_HORIZON; k++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<DynamicsFunctor, MPC_STATE_SIZE, MPC_STATE_SIZE, MPC_CONTROL_SIZE, MPC_STATE_SIZE>(
                new DynamicsFunctor(nullptr, rfr_pwr_limit_)
            ),
            nullptr,
            states[static_cast<size_t>(k)].data(),
            controls[static_cast<size_t>(k)].data(),
            states[static_cast<size_t>(k + 1)].data()
        );

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<StopCostFunctor, 11, MPC_STATE_SIZE, MPC_CONTROL_SIZE, MPC_STATE_SIZE>(
                new StopCostFunctor(params_, merged_cost_map, cost_info, global_direction_map, dir_info)
            ),
            nullptr,
            states[static_cast<size_t>(k)].data(),
            controls[static_cast<size_t>(k)].data(),
            states[static_cast<size_t>(k + 1)].data()
        );

        problem.SetParameterLowerBound(controls[static_cast<size_t>(k)].data(), 0, 0.0);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(k)].data(), 0, params_.stop_limits.vel_max);
        problem.SetParameterLowerBound(controls[static_cast<size_t>(k)].data(), 1, params_.stop_limits.omega_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(k)].data(), 1, params_.stop_limits.omega_max);
    }

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<StopTerminalCostFunctor, 2, MPC_STATE_SIZE>(
            new StopTerminalCostFunctor(params_, merged_cost_map, cost_info, global_direction_map, dir_info)
        ),
        nullptr,
        states.back().data()
    );

    ceres::Solver::Summary summary;
    ceres::Solver::Options options = make_solver_options();
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable()) {
        return std::unexpected(summary.BriefReport());
    }

    flatten_controls(controls, &last_controls_);
    const Eigen::Vector2d cmd_v_omega(controls[0][0], controls[0][1]);
    last_cmd_ = cmd_v_omega;
    return std::tuple{cmd_v_omega, prediction_from_states(states)};
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

    const GridInfo cost_info{merged_cost_map.origin_x, merged_cost_map.origin_y, merged_cost_map.resolution, merged_cost_map.width, merged_cost_map.height};
    const GridInfo dir_info{global_direction_map.origin_x, global_direction_map.origin_y, global_direction_map.resolution, global_direction_map.width, global_direction_map.height};

    ControlTrajectory controls = warm_start_controls(last_controls_, true);
    const StateBlock initial_state = build_initial_state(
        chassis_pose_map,
        chassis_status,
        start_cmd_clamped,
        x_h_hat_,
        0.0,
        remaining_energy_
    );
    StateTrajectory states = initialize_states_from_controls(initial_state, controls, nullptr, rfr_pwr_limit_);

    ceres::Problem problem;
    problem.AddParameterBlock(states[0].data(), MPC_STATE_SIZE);
    problem.SetParameterBlockConstant(states[0].data());

    for (int k = 0; k < MPC_HORIZON; k++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<DynamicsFunctor, MPC_STATE_SIZE, MPC_STATE_SIZE, MPC_CONTROL_SIZE, MPC_STATE_SIZE>(
                new DynamicsFunctor(nullptr, rfr_pwr_limit_)
            ),
            nullptr,
            states[static_cast<size_t>(k)].data(),
            controls[static_cast<size_t>(k)].data(),
            states[static_cast<size_t>(k + 1)].data()
        );

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RecoveryCostFunctor, 13, MPC_STATE_SIZE, MPC_CONTROL_SIZE, MPC_STATE_SIZE>(
                new RecoveryCostFunctor(goal_map, params_, merged_cost_map, cost_info, global_direction_map, dir_info)
            ),
            nullptr,
            states[static_cast<size_t>(k)].data(),
            controls[static_cast<size_t>(k)].data(),
            states[static_cast<size_t>(k + 1)].data()
        );

        problem.SetParameterLowerBound(controls[static_cast<size_t>(k)].data(), 0, params_.recovery_limits.vel_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(k)].data(), 0, params_.recovery_limits.vel_max);
        problem.SetParameterLowerBound(controls[static_cast<size_t>(k)].data(), 1, params_.recovery_limits.omega_min);
        problem.SetParameterUpperBound(controls[static_cast<size_t>(k)].data(), 1, params_.recovery_limits.omega_max);
    }

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<RecoveryTerminalCostFunctor, 4, MPC_STATE_SIZE>(
            new RecoveryTerminalCostFunctor(goal_map, params_, merged_cost_map, cost_info, global_direction_map, dir_info)
        ),
        nullptr,
        states.back().data()
    );

    ceres::Solver::Summary summary;
    ceres::Solver::Options options = make_solver_options();
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable()) {
        return std::unexpected(summary.BriefReport());
    }

    flatten_controls(controls, &last_controls_);
    const Eigen::Vector2d cmd_v_omega(controls[0][0], controls[0][1]);
    last_cmd_ = cmd_v_omega;
    return std::tuple{cmd_v_omega, prediction_from_states(states)};
}

}