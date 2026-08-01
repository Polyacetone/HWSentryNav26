#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

#include <nav_executor/common/trajectory/annotated_path.hpp>

namespace nav_executor {

// 固定几何上的唯一运动时标。能力边界与 MPC Follow 共用 CapabilityProfile，
// 以 z(s)=v(s)² 为决策变量在优化问题内施加硬约束：
//   角速度      |κ|·√z ≤ ω_max
//   切向加速度  |dz/ds|/2 ≤ a_t_max
//   角加速度    |κ'·z + κ·(dz/ds)/2| ≤ α_max
// 侧向加速度 |κ|·z ≤ a_lat_max 和台阶速度窗与 MPC 一致，均为软约束。
// 发布只检查 profile 的有限性和弧长/时间参数化契约；动力学包络用于塑形，不能
// 因其离散近似误差把已经得到的优化解再次拒绝。
class SpeedProfileOptimizer {
public:
    struct Params {
        struct Discretization {
            double max_spacing;
            double step_max_spacing;
            double curvature_refine_threshold;
            double envelope_sample_spacing;
        } discretization;
        struct Objective {
            double traversal_window;
            double lateral_acceleration;
            double global_speed_reward;
            double velocity_scale;
        } objective;
        double stationary_velocity_threshold;
        GeometryLimits geometry;

        CapabilityProfile normal_profile;
        std::array<CapabilityProfile, 3> step_profiles;
    };

    struct StepWindowViolation {
        size_t segment_index = 0;
        double max_under_speed = 0.0;
        double max_over_speed = 0.0;
        double arc_length = 0.0;
        TraversalVelocityWindow target;
    };

    struct Diagnostics {
        enum class Selection {
            OPTIMAL,
            CLOSED_FORM_NO_SOFT_CONSTRAINT,
        };

        int node_count = 0;
        // 同一节点被多个台阶段覆盖时，台阶速度窗会重复计数。
        int traversal_window_constraint_count = 0;
        int lateral_acceleration_constraint_count = 0;
        double seed_total_time = 0.0;
        double result_total_time = 0.0;
        double speed_reward_cost = 0.0;
        double traversal_window_cost = 0.0;
        double lateral_acceleration_cost = 0.0;
        int max_breakpoints = 0;
        double solve_ms = 0.0;
        Selection selection = Selection::OPTIMAL;
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
