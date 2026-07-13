#pragma once

#include <nav_executor/path_executor/solver/trajectory_seed.hpp>

namespace nav_executor {

struct LongitudinalPlannerParams {
    int progress_cells = 81;
    int velocity_cells = 41;
    int acceleration_samples = 5;
    double reverse_clearance = 0.6;
    double step_speed_violation_weight = 40.0;
    double progress_weight = 1.0;
    double control_change_weight = 0.1;
};

class LongitudinalPlanner {
public:
    explicit LongitudinalPlanner(LongitudinalPlannerParams params = {}): params_(params) {}

    [[nodiscard]] TrajectorySeed plan(
        const FollowProblem& problem,
        const SplinePath& path,
        const StateVec& x0,
        const StepConstraintSchedule& step_constraint_schedule,
        uint64_t sequence
    ) const;

private:
    LongitudinalPlannerParams params_;
};

} // namespace nav_executor
