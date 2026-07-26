#pragma once

#include <array>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/annotated_path.hpp>
#include <nav_executor/path_planner/admm_qp_solver.hpp>

namespace nav_executor {

class SpeedProfileOptimizer {
public:
    struct Params {
        struct Discretization {
            double max_spacing;
            double step_max_spacing;
            double curvature_refine_threshold;
        } discretization;
        struct Objective {
            double traversal_window;
            double global_speed_reward;
            double velocity_scale;
        } objective;
        struct Validation {
            double sample_spacing;
            double velocity_tolerance;
            double acceleration_tolerance;
            double angular_velocity_tolerance;
            double lateral_acceleration_tolerance;
        } validation;

        AdmmQpSolver::Params solver;
        CapabilityProfile normal_profile;
        std::array<CapabilityProfile, 3> step_profiles;
        double trajectory_velocity_max;
        double trajectory_acceleration_max;
        double trajectory_angular_velocity_max;
        double trajectory_lateral_acceleration_max;
    };

    struct StepWindowViolation {
        size_t segment_index = 0;
        double max_under_speed = 0.0;
        double max_over_speed = 0.0;
        double arc_length = 0.0;
        double hard_velocity_upper = 0.0;
        TraversalVelocityWindow target;
    };

    struct Diagnostics {
        int node_count = 0;
        int variable_count = 0;
        int constraint_count = 0;
        int soft_window_node_count = 0;
        int iterations = 0;
        int rho_updates = 0;
        double seed_total_time = 0.0;
        double result_total_time = 0.0;
        double speed_reward_cost = 0.0;
        double traversal_window_cost = 0.0;
        double primal_residual = 0.0;
        double dual_residual = 0.0;
        double max_constraint_violation = 0.0;
        double factorization_ms = 0.0;
        double iteration_ms = 0.0;
        bool polish_attempted = false;
        bool polish_accepted = false;
        bool used_fallback = false;
        std::vector<StepWindowViolation> step_violations;
    };

    struct Result {
        bool success = false;
        PathSpeedProfile profile;
        Diagnostics diagnostics;
        std::string error;
    };

    explicit SpeedProfileOptimizer(Params params) : params_(std::move(params)) {}

    [[nodiscard]] Result optimize(
        const MincoTrajectory& geometry,
        const std::vector<StepPlanSegment>& step_segments,
        const Eigen::Vector2d& current_velocity_map
    ) const;

private:
    Params params_;
};

} // namespace nav_executor
