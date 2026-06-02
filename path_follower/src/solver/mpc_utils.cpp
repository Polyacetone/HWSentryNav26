#include <path_follower/solver/mpc_utils.hpp>

namespace path_follower {

double smooth_sgn(double x, double eps) {
    return std::tanh(x / std::max(eps, 1e-6));
}

double smooth_sgn_deriv(double x, double eps) {
    const double s = smooth_sgn(x, eps);
    return (1.0 - s * s) / std::max(eps, 1e-6);
}

void zoh_v_matrices(
    double a00, double a01, double a10, double a11,
    double b0, double b1, double g0, double g1,
    double dt,
    double& ad00, double& ad01, double& ad10, double& ad11,
    double& bd0, double& bd1, double& gd0, double& gd1
) {
    const double m00 = a00 * dt;
    const double m01 = a01 * dt;
    const double m10 = a10 * dt;
    const double m11 = a11 * dt;
    const double tr_m = m00 + m11;
    const double det_m = m00 * m11 - m01 * m10;
    const double disc = tr_m * tr_m - 4.0 * det_m;
    constexpr double REG_EPS = 1e-8;

    double alpha = 0.0;
    double beta = 0.0;
    if (disc > REG_EPS) {
        const double s = std::sqrt(disc);
        const double lam1 = 0.5 * (tr_m + s);
        const double lam2 = 0.5 * (tr_m - s);
        const double delta = lam1 - lam2;
        const double el2 = std::exp(lam2);
        beta = el2 * std::expm1(delta) / delta;
        alpha = std::exp(lam1) - beta * lam1;
    } else if (disc < -REG_EPS) {
        const double p = 0.5 * tr_m;
        const double q = 0.5 * std::sqrt(-disc);
        const double ep = std::exp(p);
        beta = ep * std::sin(q) / q;
        alpha = ep * (std::cos(q) - p * std::sin(q) / q);
    } else {
        const double lam = 0.5 * tr_m;
        const double el = std::exp(lam);
        beta = el;
        alpha = el * (1.0 - lam);
    }

    ad00 = alpha + beta * m00;
    ad01 = beta * m01;
    ad10 = beta * m10;
    ad11 = alpha + beta * m11;

    const double det_a = a00 * a11 - a01 * a10;
    const double c = alpha - 1.0;
    double g00 = 0.0;
    double g01 = 0.0;
    double g10 = 0.0;
    double g11 = 0.0;
    constexpr double DET_EPS = 1e-6;
    if (std::abs(det_a) > DET_EPS) {
        const double inv_det = 1.0 / det_a;
        g00 = c * a11 * inv_det + beta * dt;
        g01 = c * (-a01) * inv_det;
        g10 = c * (-a10) * inv_det;
        g11 = c * a00 * inv_det + beta * dt;
    } else {
        g00 = dt + 0.5 * dt * dt * a00;
        g01 = 0.5 * dt * dt * a01;
        g10 = 0.5 * dt * dt * a10;
        g11 = dt + 0.5 * dt * dt * a11;
    }

    bd0 = g00 * b0 + g01 * b1;
    bd1 = g10 * b0 + g11 * b1;
    gd0 = g00 * g0 + g01 * g1;
    gd1 = g10 * g0 + g11 * g1;
}

LPVNonlinearEval evaluate_lpv_nonlinear(double v, double w, const LPVDiscreteModel& model) {
    const double sv = smooth_sgn(v, model.sgn_eps);
    const double dsv = smooth_sgn_deriv(v, model.sgn_eps);
    const double sw = smooth_sgn(w, model.sgn_eps);
    const double dsw = smooth_sgn_deriv(w, model.sgn_eps);
    const double absw = std::abs(w);
    return {
        .nl = model.cf1 * sv + model.cf2 * v * absw,
        .dnl_dv = model.cf1 * dsv + model.cf2 * absw,
        .dnl_dw = model.cf2 * v * sign_or_zero(w),
        .sw = sw,
        .dsw_dw = dsw,
    };
}

Eigen::Vector2d apply_goal_deadzone(const Eigen::Vector2d& delta, double deadzone) {
    if (deadzone <= 0.0) return delta;

    const double dist = delta.norm();
    if (dist <= 0.0) return Eigen::Vector2d::Zero();

    const double mag = positive_part(dist - deadzone);
    if (mag <= 0.0) {
        return Eigen::Vector2d::Zero();
    }
    return delta * (mag / dist);
}

LPVDiscreteModel build_lpv_discrete_model(const LPVKinematicModelParams& params, double rho) {
    LPVDiscreteModel model;
    model.rho = clamp_lpv_rho(rho, params.rho_clip);

    const double a00 = params.ca00 + model.rho * params.dca00;
    const double a01 = params.ca01 + model.rho * params.dca01;
    const double a10 = params.ca10 + model.rho * params.dca10;
    const double a11 = params.ca11 + model.rho * params.dca11;
    const double b0 = params.cb0 + model.rho * params.dcb0;
    const double b1 = params.cb1 + model.rho * params.dcb1;

    zoh_v_matrices(
        a00, a01, a10, a11, b0, b1, params.gxh, params.gv, MPC_DT,
        model.ad00, model.ad01, model.ad10, model.ad11,
        model.bd0, model.bd1, model.gd0, model.gd1
    );

    const double lam = std::max(params.w_lam0 + model.rho * params.w_lam1, 1e-5);
    const double kw = params.w_k0 + model.rho * params.w_k1;
    const double cf = params.w_cf0 + model.rho * params.w_cf1;
    model.alpha_w = std::exp(-lam * MPC_DT);
    const double integ_w = (1.0 - model.alpha_w) / lam;
    model.beta_w = integ_w * kw;
    model.gamma_w = integ_w * cf;
    model.sgn_eps = params.sgn_eps;
    model.cf1 = params.cf1;
    model.cf2 = params.cf2;
    return model;
}

double schedule_rho_from_state(const ChassisMotionState& chassis_state, const LPVKinematicModelParams& model_params) {
    const double z = lpv_schedule_z(chassis_state.leg_h, chassis_state.leg_psi);
    return clamp_lpv_rho((z - model_params.z_ref) / std::max(model_params.z_scale, 1e-6), model_params.rho_clip);
}

double select_follow_schedule_rho(
    const ChassisMotionState& chassis_state,
    const LPVKinematicModelParams& model_params
) {
    return schedule_rho_from_state(chassis_state, model_params);
}

} // namespace path_follower
