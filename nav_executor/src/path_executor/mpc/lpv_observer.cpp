#include <nav_executor/path_executor/mpc/lpv_observer.hpp>
#include <nav_executor/path_executor/mpc/mpc_utils.hpp>

#include <algorithm>
#include <cmath>
#include <rclcpp/logging.hpp>
#include <utility>

namespace nav_executor {

namespace {

constexpr double MIN_INITIALIZATION_COUPLING = 1e-3;

bool longitudinal_model_is_finite(const LPVDiscreteModel& model) {
    return std::isfinite(model.rho)
        && std::isfinite(model.ad00)
        && std::isfinite(model.ad01)
        && std::isfinite(model.ad10)
        && std::isfinite(model.ad11)
        && std::isfinite(model.bd0)
        && std::isfinite(model.bd1)
        && std::isfinite(model.gd0)
        && std::isfinite(model.gd1)
        && std::isfinite(model.sgn_eps)
        && std::isfinite(model.cf1)
        && std::isfinite(model.cf2);
}

} // namespace

LPVLongitudinalObserver::LPVLongitudinalObserver(
    const LPVKinematicModelParams& params,
    rclcpp::Logger logger
)
    : params_(params), logger_(std::move(logger)) {}

ObserverDiagnostics LPVLongitudinalObserver::base_diagnostics(
    const ObserverUpdateEvent event,
    const uint64_t state_sequence
) const {
    ObserverDiagnostics diagnostics;
    diagnostics.event = event;
    diagnostics.last_reset_reason = last_reset_reason_;
    diagnostics.initialized = initialized_;
    diagnostics.validated = validated_;
    diagnostics.state_sequence = state_sequence;
    diagnostics.reset_count = reset_count_;
    diagnostics.active_run_length = active_run_length_;
    diagnostics.revalidation_latency_updates = last_revalidation_latency_updates_;
    diagnostics.hidden_state_estimate = x_h_hat_;
    diagnostics.input_command_velocity = input_command_.x();
    diagnostics.input_command_angular_velocity = input_command_.y();
    return diagnostics;
}

void LPVLongitudinalObserver::clear_state() {
    x_h_hat_ = 0.0;
    prev_v_act_ = 0.0;
    prev_w_act_ = 0.0;
    prev_schedule_rho_ = 0.0;
    input_command_.setZero();
    initialized_ = false;
    validated_ = false;
    active_run_length_ = 0;
    updates_since_initialization_ = 0;
    last_revalidation_latency_updates_ = 0;
}

void LPVLongitudinalObserver::reject(
    const ObserverResetReason reason,
    ObserverDiagnostics diagnostics
) {
    if (initialized_ || validated_) ++reset_count_;
    clear_state();
    last_reset_reason_ = reason;

    diagnostics.event = ObserverUpdateEvent::RESET;
    diagnostics.last_reset_reason = reason;
    diagnostics.initialized = false;
    diagnostics.validated = false;
    diagnostics.reset_count = reset_count_;
    diagnostics.active_run_length = 0;
    diagnostics.revalidation_latency_updates = 0;
    diagnostics.hidden_state_estimate = 0.0;
    diagnostics_ = std::move(diagnostics);
    rejection_active_ = true;
}

void LPVLongitudinalObserver::reset(const ObserverResetReason reason) {
    const uint64_t state_sequence = diagnostics_.state_sequence;
    ObserverDiagnostics diagnostics = base_diagnostics(
        ObserverUpdateEvent::RESET,
        state_sequence
    );
    if (initialized_ || validated_) ++reset_count_;
    clear_state();
    last_reset_reason_ = reason;
    rejection_active_ = false;

    diagnostics.last_reset_reason = reason;
    diagnostics.initialized = false;
    diagnostics.validated = false;
    diagnostics.prediction_available = false;
    diagnostics.auxiliary_prediction_available = false;
    diagnostics.velocity_correction_clipped = false;
    diagnostics.reset_count = reset_count_;
    diagnostics.active_run_length = 0;
    diagnostics.revalidation_latency_updates = 0;
    diagnostics.hidden_state_estimate = 0.0;
    diagnostics_ = std::move(diagnostics);
}

const ObserverDiagnostics& LPVLongitudinalObserver::update(
    const ChassisMotionState& chassis_state,
    const Eigen::Vector2d& input_command,
    const uint64_t state_sequence
) {
    if (!std::isfinite(chassis_state.velocity)
        || !std::isfinite(chassis_state.omega)
        || !std::isfinite(chassis_state.leg_h)
        || !std::isfinite(chassis_state.leg_psi)
        || !input_command.allFinite()) {
        reject(
            ObserverResetReason::NONFINITE_INPUT,
            base_diagnostics(ObserverUpdateEvent::RESET, state_sequence)
        );
        return diagnostics_;
    }

    const double v_act = chassis_state.velocity;
    const double w_act = chassis_state.omega;
    const double rho_cur = schedule_rho_from_state(chassis_state, params_);

    if (!initialized_) {
        const LPVDiscreteModel model = build_lpv_discrete_model(params_, rho_cur);
        const LPVNonlinearEval nonlinear = evaluate_lpv_nonlinear(v_act, w_act, model);
        if (!longitudinal_model_is_finite(model)
            || !std::isfinite(nonlinear.nl)
            || std::abs(model.ad10) < MIN_INITIALIZATION_COUPLING) {
            const bool should_log = !rejection_active_;
            reject(
                ObserverResetReason::MODEL_DEGENERATE,
                base_diagnostics(ObserverUpdateEvent::RESET, state_sequence)
            );
            if (should_log) {
                RCLCPP_WARN(
                    logger_,
                    "Cannot initialize longitudinal LPV observer: ad10=%.6f, rho=%.3f",
                    model.ad10,
                    rho_cur
                );
            }
            return diagnostics_;
        }

        // Reconstruct xh so the same discrete model holds the measured velocity
        // for one step under the command captured at initialization.
        const double initial_xh = (
            v_act
            - model.ad11 * v_act
            - model.bd1 * input_command.x()
            - model.gd1 * nonlinear.nl
        ) / model.ad10;
        if (!std::isfinite(initial_xh)) {
            reject(
                ObserverResetReason::NONFINITE_PREDICTION,
                base_diagnostics(ObserverUpdateEvent::RESET, state_sequence)
            );
            return diagnostics_;
        }

        x_h_hat_ = initial_xh;
        prev_v_act_ = v_act;
        prev_w_act_ = w_act;
        prev_schedule_rho_ = rho_cur;
        input_command_ = input_command;
        initialized_ = true;
        validated_ = false;
        updates_since_initialization_ = 0;

        diagnostics_ = base_diagnostics(
            ObserverUpdateEvent::INITIALIZED,
            state_sequence
        );
        return diagnostics_;
    }

    ++updates_since_initialization_;
    const LPVDiscreteModel model = build_lpv_discrete_model(
        params_,
        0.5 * (prev_schedule_rho_ + rho_cur)
    );
    const LPVNonlinearEval nonlinear = evaluate_lpv_nonlinear(
        prev_v_act_,
        prev_w_act_,
        model
    );
    ObserverDiagnostics diagnostics = base_diagnostics(
        ObserverUpdateEvent::NONE,
        state_sequence
    );
    diagnostics.input_command_velocity = input_command_.x();
    diagnostics.input_command_angular_velocity = input_command_.y();

    if (!longitudinal_model_is_finite(model) || !std::isfinite(nonlinear.nl)) {
        const bool should_log = !rejection_active_;
        reject(ObserverResetReason::MODEL_DEGENERATE, std::move(diagnostics));
        if (should_log) {
            RCLCPP_WARN(logger_, "Rejecting longitudinal LPV observer: model is degenerate");
        }
        return diagnostics_;
    }

    // The identified model has a one-step input delay. input_command_ is the
    // command captured with the previous observer sample.
    const double xh_pred = model.ad00 * x_h_hat_
        + model.ad01 * prev_v_act_
        + model.bd0 * input_command_.x()
        + model.gd0 * nonlinear.nl;
    const double v_pred = model.ad10 * x_h_hat_
        + model.ad11 * prev_v_act_
        + model.bd1 * input_command_.x()
        + model.gd1 * nonlinear.nl;
    const double velocity_innovation = v_act - v_pred;

    diagnostics.predicted_hidden_state = xh_pred;
    diagnostics.predicted_velocity = v_pred;
    diagnostics.velocity_innovation = velocity_innovation;
    diagnostics.prediction_available = std::isfinite(xh_pred)
        && std::isfinite(v_pred)
        && std::isfinite(velocity_innovation);

    // Yaw and leg-angle residuals are shadow diagnostics only. They never
    // correct or invalidate the longitudinal observer.
    const double w_pred = model.alpha_w * prev_w_act_
        + model.beta_w * input_command_.y()
        - model.gamma_w * nonlinear.sw;
    const double angular_velocity_innovation = w_act - w_pred;
    const double psi_proxy_pred = params_.psi_bias
        + params_.psi_gain * xh_pred
        + params_.psi_v * v_pred;
    const double leg_psi_innovation = chassis_state.leg_psi - psi_proxy_pred;
    diagnostics.auxiliary_prediction_available = std::isfinite(w_pred)
        && std::isfinite(angular_velocity_innovation)
        && std::isfinite(psi_proxy_pred)
        && std::isfinite(leg_psi_innovation);
    if (diagnostics.auxiliary_prediction_available) {
        diagnostics.predicted_angular_velocity = w_pred;
        diagnostics.angular_velocity_innovation = angular_velocity_innovation;
        diagnostics.predicted_leg_psi = psi_proxy_pred;
        diagnostics.leg_psi_innovation = leg_psi_innovation;
    }

    if (!diagnostics.prediction_available) {
        const bool should_log = !rejection_active_;
        reject(ObserverResetReason::NONFINITE_PREDICTION, std::move(diagnostics));
        if (should_log) {
            RCLCPP_WARN(logger_, "Rejecting longitudinal LPV observer: non-finite prediction");
        }
        return diagnostics_;
    }

    const bool velocity_correction_clipped =
        std::abs(velocity_innovation) > params_.obs_v_correction_clip;
    if (std::abs(velocity_innovation) > params_.obs_v_reset_threshold) {
        const bool should_log = !rejection_active_;
        reject(ObserverResetReason::VELOCITY_INNOVATION, std::move(diagnostics));
        if (should_log) {
            RCLCPP_WARN(
                logger_,
                "Rejecting longitudinal LPV observer innovation: velocity=%.3f m/s",
                velocity_innovation
            );
        }
        return diagnostics_;
    }
    diagnostics.velocity_correction_clipped = velocity_correction_clipped;

    const double robust_velocity_innovation = std::clamp(
        velocity_innovation,
        -params_.obs_v_correction_clip,
        params_.obs_v_correction_clip
    );
    const double corrected_xh = xh_pred
        + params_.obs_lv * robust_velocity_innovation;
    if (!std::isfinite(corrected_xh)) {
        const bool should_log = !rejection_active_;
        reject(ObserverResetReason::NONFINITE_PREDICTION, std::move(diagnostics));
        if (should_log) {
            RCLCPP_WARN(logger_, "Rejecting longitudinal LPV observer: non-finite correction");
        }
        return diagnostics_;
    }

    const bool first_validation = !validated_;
    x_h_hat_ = corrected_xh;
    prev_v_act_ = v_act;
    prev_w_act_ = w_act;
    prev_schedule_rho_ = rho_cur;
    input_command_ = input_command;
    initialized_ = true;
    validated_ = true;
    if (first_validation) {
        active_run_length_ = 1;
        last_revalidation_latency_updates_ = updates_since_initialization_;
    } else {
        ++active_run_length_;
    }
    rejection_active_ = false;

    diagnostics.event = diagnostics.velocity_correction_clipped
        ? ObserverUpdateEvent::MODEL_STRESS
        : ObserverUpdateEvent::CORRECTED;
    diagnostics.initialized = true;
    diagnostics.validated = true;
    diagnostics.active_run_length = active_run_length_;
    diagnostics.revalidation_latency_updates = last_revalidation_latency_updates_;
    diagnostics.hidden_state_estimate = corrected_xh;
    diagnostics_ = std::move(diagnostics);
    return diagnostics_;
}

} // namespace nav_executor
