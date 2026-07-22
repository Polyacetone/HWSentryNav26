#include <nav_executor/path_executor/solver/mpc_utils.hpp>

namespace nav_executor {

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

MPCControlBounds command_rate_control_bounds(
    const StateVec& x,
    const CapabilityProfile& capability,
    const double path_speed_min,
    const double path_speed_max,
    const MPCStartCommandLimits* const start_command
) {
    const auto& envelope = capability.command_envelope;
    const auto& dynamics = capability.command_dynamics;
    const double inv_dt = 1.0 / MPC_DT;

    MPCControlBounds bounds;
    const auto set_command_rate_bounds = [&]<typename EnvelopeT>(
        const int control_index,
        const int command_state_index,
        const int actual_state_index,
        const EnvelopeT& command_envelope,
        const double rate_max,
        const double command_actual_gap_max
    ) {
        double target_min = command_envelope.min;
        double target_max = command_envelope.max;
        StateVec target_min_jacobian = StateVec::Zero();
        StateVec target_max_jacobian = StateVec::Zero();

        if (start_command) {
            const double gap_min = x(actual_state_index) - command_actual_gap_max;
            const double gap_max = x(actual_state_index) + command_actual_gap_max;
            const double intersected_min = std::max(target_min, gap_min);
            const double intersected_max = std::min(target_max, gap_max);
            // command envelope 是下位机安全边界，优先级高于启动时 command/actual 间隙。
            // 两者有交集时才收紧；无交集时由 command envelope 和 rate limit 负责可行恢复。
            if (intersected_min <= intersected_max) {
                if (gap_min > target_min) {
                    target_min = gap_min;
                    target_min_jacobian(actual_state_index) = 1.0;
                }
                if (gap_max < target_max) {
                    target_max = gap_max;
                    target_max_jacobian(actual_state_index) = 1.0;
                }
            }
        }

        const double command = x(command_state_index);
        const double max_delta = rate_max * MPC_DT;
        if (command < target_min - max_delta) {
            // 初始命令在目标窗外且一拍不可达：以最大合法速率向目标窗恢复。
            bounds.lower(control_index) = rate_max;
            bounds.upper(control_index) = rate_max;
            return;
        }
        if (command > target_max + max_delta) {
            bounds.lower(control_index) = -rate_max;
            bounds.upper(control_index) = -rate_max;
            return;
        }

        bounds.lower(control_index) = -rate_max;
        const double envelope_lower_rate = (target_min - command) * inv_dt;
        if (envelope_lower_rate >= bounds.lower(control_index)) {
            bounds.lower(control_index) = envelope_lower_rate;
            bounds.lower_state_jacobian.row(control_index) =
                target_min_jacobian.transpose() * inv_dt;
            bounds.lower_state_jacobian(control_index, command_state_index) -= inv_dt;
        }

        bounds.upper(control_index) = rate_max;
        const double envelope_upper_rate = (target_max - command) * inv_dt;
        if (envelope_upper_rate <= bounds.upper(control_index)) {
            bounds.upper(control_index) = envelope_upper_rate;
            bounds.upper_state_jacobian.row(control_index) =
                target_max_jacobian.transpose() * inv_dt;
            bounds.upper_state_jacobian(control_index, command_state_index) -= inv_dt;
        }
    };

    set_command_rate_bounds(
        iu::V_CMD_RATE,
        ix::V_CMD,
        ix::V,
        envelope.velocity,
        dynamics.velocity_rate_max,
        start_command ? start_command->vel_cmd_act_gap_max : 0.0
    );
    set_command_rate_bounds(
        iu::W_CMD_RATE,
        ix::W_CMD,
        ix::W,
        envelope.angular_velocity,
        dynamics.angular_velocity_rate_max,
        start_command ? start_command->omega_cmd_act_gap_max : 0.0
    );
    bounds.lower(iu::PATH_SPEED_CMD) = path_speed_min;
    bounds.upper(iu::PATH_SPEED_CMD) = path_speed_max;
    return bounds;
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

} // namespace nav_executor
