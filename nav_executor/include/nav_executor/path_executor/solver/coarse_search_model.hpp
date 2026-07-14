#pragma once

#include <array>

#include <nav_executor/path_executor/solver/follow_problem.hpp>

namespace nav_executor {

inline constexpr int MAX_SEARCH_MACRO_STEPS = 8;

struct CoarseTransition {
    StateVec state = StateVec::Zero();
    std::array<ControlVec, MAX_SEARCH_MACRO_STEPS> controls {};
    int control_count = 0;
    double running_cost = 0.0;
    bool valid = false;
};

// Low-frequency search model. It preserves the identified velocity channels,
// but evaluates pose, path progress, and cost only once per macro step.
class CoarseSearchModel {
public:
    explicit CoarseSearchModel(const FollowProblem& problem): problem_(problem) {}

    [[nodiscard]] CoarseTransition transition(
        const StateVec& state,
        double velocity_acceleration,
        double omega_acceleration,
        int fine_step,
        int step_count
    ) const;

private:
    struct DynamicState {
        double xh = 0.0;
        double velocity = 0.0;
        double omega = 0.0;
    };

    struct DynamicTrace {
        std::array<DynamicState, MAX_SEARCH_MACRO_STEPS + 1> states {};
    };

    [[nodiscard]] DynamicTrace predict_dynamic_trace(
        const StateVec& state,
        const std::array<ControlVec, MAX_SEARCH_MACRO_STEPS>& controls,
        int step_count,
        double nonlinear_velocity,
        double nonlinear_omega
    ) const;
    [[nodiscard]] double path_progress_rate(const StateVec& state) const;

    const FollowProblem& problem_;
};

} // namespace nav_executor
