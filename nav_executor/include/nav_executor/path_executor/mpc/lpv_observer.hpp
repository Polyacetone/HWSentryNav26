#pragma once

#include <nav_executor/path_executor/mpc/mpc_types.hpp>
#include <rclcpp/logger.hpp>

namespace nav_executor {

class LPVLongitudinalObserver {
public:
    LPVLongitudinalObserver(
        const LPVKinematicModelParams& params,
        rclcpp::Logger logger
    );

    const ObserverDiagnostics& update(
        const ChassisMotionState& chassis_state,
        const Eigen::Vector2d& input_command,
        uint64_t state_sequence
    );
    void reset(ObserverResetReason reason);

    [[nodiscard]] bool validated() const { return validated_; }
    [[nodiscard]] double hidden_state_estimate() const { return x_h_hat_; }
    [[nodiscard]] const ObserverDiagnostics& diagnostics() const { return diagnostics_; }

private:
    ObserverDiagnostics base_diagnostics(
        ObserverUpdateEvent event,
        uint64_t state_sequence
    ) const;
    void clear_state();
    void reject(ObserverResetReason reason, ObserverDiagnostics diagnostics);

    LPVKinematicModelParams params_;
    rclcpp::Logger logger_;

    double x_h_hat_ = 0.0;
    double prev_v_act_ = 0.0;
    double prev_w_act_ = 0.0;
    double prev_schedule_rho_ = 0.0;
    Eigen::Vector2d input_command_ = Eigen::Vector2d::Zero();
    bool initialized_ = false;
    bool validated_ = false;
    bool rejection_active_ = false;

    uint64_t reset_count_ = 0;
    uint64_t active_run_length_ = 0;
    uint64_t updates_since_initialization_ = 0;
    uint64_t last_revalidation_latency_updates_ = 0;
    ObserverResetReason last_reset_reason_ = ObserverResetReason::NONE;
    ObserverDiagnostics diagnostics_;
};

} // namespace nav_executor
