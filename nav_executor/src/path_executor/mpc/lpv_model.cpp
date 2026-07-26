#include <nav_executor/path_executor/mpc/lpv_model.hpp>
#include <nav_executor/path_executor/mpc/mpc_utils.hpp>

namespace nav_executor {

StateVec mpc_dynamics(const StateVec& x, const ControlVec& u, const LPVDiscreteModel& model) {
    const double theta = x(ix::THETA);
    const double xh = x(ix::XH), v = x(ix::V), w = x(ix::W);
    const double v_cmd = x(ix::V_CMD), w_cmd = x(ix::W_CMD);

    const auto nl_eval = evaluate_lpv_nonlinear(v, w, model);

    const double xh1 = model.ad00 * xh + model.ad01 * v + model.bd0 * v_cmd + model.gd0 * nl_eval.nl;
    const double v1 = model.ad10 * xh + model.ad11 * v + model.bd1 * v_cmd + model.gd1 * nl_eval.nl;
    const double w1 = model.alpha_w * w + model.beta_w * w_cmd - model.gamma_w * nl_eval.sw;

    const double dt = MPC_DT;
    const double theta1 = theta + (w + w1) * (dt * 0.5);
    const double ct0 = std::cos(theta), st0 = std::sin(theta);
    const double ct1 = std::cos(theta1), st1 = std::sin(theta1);

    StateVec xn;
    xn(ix::X) = x(ix::X) + (v * ct0 + v1 * ct1) * (dt * 0.5);
    xn(ix::Y) = x(ix::Y) + (v * st0 + v1 * st1) * (dt * 0.5);
    xn(ix::THETA) = theta1;
    xn(ix::XH) = xh1;
    xn(ix::V) = v1;
    xn(ix::W) = w1;
    xn(ix::V_CMD) = v_cmd + dt * u(iu::V_CMD_RATE);
    xn(ix::W_CMD) = w_cmd + dt * u(iu::W_CMD_RATE);
    xn(ix::V_CMD_RATE) = u(iu::V_CMD_RATE);
    xn(ix::W_CMD_RATE) = u(iu::W_CMD_RATE);
    xn(ix::PATH_PROGRESS) = x(ix::PATH_PROGRESS);
    xn(ix::PATH_SPEED) = x(ix::PATH_SPEED);
    return xn;
}

void mpc_dynamics_jacobians(const StateVec& x, const ControlVec& /*u*/, const LPVDiscreteModel& model, MatXX& fx, MatXU& fu) {
    const double theta = x(ix::THETA);
    const double xh = x(ix::XH), v = x(ix::V), w = x(ix::W);
    const double v_cmd = x(ix::V_CMD), w_cmd = x(ix::W_CMD);
    const double dt = MPC_DT, h = dt * 0.5;

    const auto nl_eval = evaluate_lpv_nonlinear(v, w, model);

    const double v1 = model.ad10 * xh + model.ad11 * v + model.bd1 * v_cmd + model.gd1 * nl_eval.nl;
    const double w1 = model.alpha_w * w + model.beta_w * w_cmd - model.gamma_w * nl_eval.sw;
    const double theta1 = theta + (w + w1) * h;

    const double dvn_dxh = model.ad10;
    const double dvn_dv = model.ad11 + model.gd1 * nl_eval.dnl_dv;
    const double dvn_dw = model.gd1 * nl_eval.dnl_dw;
    const double dvn_ddv = model.bd1;

    const double dwn_dw = model.alpha_w - model.gamma_w * nl_eval.dsw_dw;
    const double dwn_ddw = model.beta_w;

    const double dth1_dth = 1.0;
    const double dth1_dw = (1.0 + dwn_dw) * h;
    const double dth1_ddw = dwn_ddw * h;

    const double ct0 = std::cos(theta), st0 = std::sin(theta);
    const double ct1 = std::cos(theta1), st1 = std::sin(theta1);

    fx.setZero();
    fu.setZero();

    fx(ix::X, ix::X) = 1.0;
    fx(ix::X, ix::THETA) = (-v * st0 - v1 * st1 * dth1_dth) * h;
    fx(ix::X, ix::XH) = dvn_dxh * ct1 * h;
    fx(ix::X, ix::V) = (ct0 + dvn_dv * ct1) * h;
    fx(ix::X, ix::W) = (dvn_dw * ct1 - v1 * st1 * dth1_dw) * h;
    fx(ix::X, ix::V_CMD) = dvn_ddv * ct1 * h;
    fx(ix::X, ix::W_CMD) = -v1 * st1 * dth1_ddw * h;

    fx(ix::Y, ix::Y) = 1.0;
    fx(ix::Y, ix::THETA) = (v * ct0 + v1 * ct1 * dth1_dth) * h;
    fx(ix::Y, ix::XH) = dvn_dxh * st1 * h;
    fx(ix::Y, ix::V) = (st0 + dvn_dv * st1) * h;
    fx(ix::Y, ix::W) = (dvn_dw * st1 + v1 * ct1 * dth1_dw) * h;
    fx(ix::Y, ix::V_CMD) = dvn_ddv * st1 * h;
    fx(ix::Y, ix::W_CMD) = v1 * ct1 * dth1_ddw * h;

    fx(ix::THETA, ix::THETA) = dth1_dth;
    fx(ix::THETA, ix::W) = dth1_dw;
    fx(ix::THETA, ix::W_CMD) = dth1_ddw;

    fx(ix::XH, ix::XH) = model.ad00;
    fx(ix::XH, ix::V) = model.ad01 + model.gd0 * nl_eval.dnl_dv;
    fx(ix::XH, ix::W) = model.gd0 * nl_eval.dnl_dw;
    fx(ix::XH, ix::V_CMD) = model.bd0;

    fx(ix::V, ix::XH) = dvn_dxh;
    fx(ix::V, ix::V) = dvn_dv;
    fx(ix::V, ix::W) = dvn_dw;
    fx(ix::V, ix::V_CMD) = dvn_ddv;

    fx(ix::W, ix::W) = dwn_dw;
    fx(ix::W, ix::W_CMD) = dwn_ddw;

    fx(ix::V_CMD, ix::V_CMD) = 1.0;
    fx(ix::W_CMD, ix::W_CMD) = 1.0;

    fx(ix::PATH_PROGRESS, ix::PATH_PROGRESS) = 1.0;
    fx(ix::PATH_SPEED, ix::PATH_SPEED) = 1.0;

    fu(ix::V_CMD, iu::V_CMD_RATE) = dt;
    fu(ix::W_CMD, iu::W_CMD_RATE) = dt;
    fu(ix::V_CMD_RATE, iu::V_CMD_RATE) = 1.0;
    fu(ix::W_CMD_RATE, iu::W_CMD_RATE) = 1.0;
}

} // namespace nav_executor
