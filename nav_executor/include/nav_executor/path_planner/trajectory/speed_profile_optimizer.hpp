#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/trajectory/annotated_path.hpp>

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
        enum class Selection {
            OPTIMAL,
            SEED_OPTIMAL_NO_WINDOW,
            FALLBACK,
        };

        int node_count = 0;
        // 软速度窗约束条数；同一节点被多个台阶段覆盖时会重复计数。
        int soft_window_constraint_count = 0;
        double seed_total_time = 0.0;
        double result_total_time = 0.0;
        double speed_reward_cost = 0.0;
        double traversal_window_cost = 0.0;
        int max_breakpoints = 0;
        double solve_ms = 0.0;
        Selection selection = Selection::FALLBACK;
        std::string fallback_reason;
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
