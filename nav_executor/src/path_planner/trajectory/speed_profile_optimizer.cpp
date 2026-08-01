#include <nav_executor/path_planner/trajectory/speed_profile_optimizer.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include <nav_executor/path_planner/numerics/piecewise_quadratic_chain_solver.hpp>

namespace nav_executor {

namespace {

constexpr double EPS = 1e-9;
constexpr double INF = std::numeric_limits<double>::infinity();

// 角加速度 dω/dt = κ'·z + κ·(dz/ds)/2 是两个可加项。链式求解器只接受
// 「节点上界」和「相邻差分上界」两类约束，因此把角加速度预算等分给两项：
//   κ'·z        → 节点上界 z ≤ SHARE·α_max/|κ'|
//   κ·(dz/ds)/2 → 差分上界，等价于把切向加速度收紧到 SHARE·α_max/|κ|
// 等分保证两项之和恒不超过 α_max，以保守包络塑造速度解。
constexpr double ANGULAR_ACCELERATION_SHARE = 0.5;

struct NodeLimit {
    double speed_squared_upper = 0.0;
    double acceleration = 0.0;
};

struct TraversalWindowNode {
    size_t node_index = 0;
    double integration_weight = 0.0;
    double gate = 0.0;
    TraversalVelocityWindow target;
};

struct LateralAccelerationNode {
    size_t node_index = 0;
    double integration_weight = 0.0;
    double curvature = 0.0;
    double speed_squared_upper = 0.0;
};

// 逐点动力学包络，全部由同一份真实几何曲率导出。
struct LocalEnvelope {
    double speed_squared_upper = 0.0;
    double acceleration_upper = 0.0;
};

CapabilityProfile interpolate_profile(
    const CapabilityProfile& normal,
    const CapabilityProfile& step,
    const double blend
) {
    const auto interpolate = [blend](const double a, const double b) {
        return std::lerp(a, b, blend);
    };
    return {
        .command_envelope = {
            .velocity = {
                .min = interpolate(normal.command_envelope.velocity.min, step.command_envelope.velocity.min),
                .max = interpolate(normal.command_envelope.velocity.max, step.command_envelope.velocity.max),
            },
            .angular_velocity = {
                .min = interpolate(normal.command_envelope.angular_velocity.min, step.command_envelope.angular_velocity.min),
                .max = interpolate(normal.command_envelope.angular_velocity.max, step.command_envelope.angular_velocity.max),
            },
        },
        .command_dynamics = {
            .velocity_rate_max = interpolate(normal.command_dynamics.velocity_rate_max, step.command_dynamics.velocity_rate_max),
            .angular_velocity_rate_max = interpolate(normal.command_dynamics.angular_velocity_rate_max, step.command_dynamics.angular_velocity_rate_max),
            .lateral_acceleration_max = interpolate(normal.command_dynamics.lateral_acceleration_max, step.command_dynamics.lateral_acceleration_max),
        },
    };
}

double capability_blend(const StepPlanSegment& segment, const double progress) {
    if (progress < segment.prepare_arc_length || progress > segment.release_arc_length) return 0.0;
    if (progress < segment.commit_arc_length) {
        const double width = segment.commit_arc_length - segment.prepare_arc_length;
        return width > EPS
            ? std::clamp((progress - segment.prepare_arc_length) / width, 0.0, 1.0)
            : 1.0;
    }
    if (progress <= segment.step_exit_arc_length) return 1.0;
    const double width = segment.release_arc_length - segment.step_exit_arc_length;
    return width > EPS
        ? std::clamp((segment.release_arc_length - progress) / width, 0.0, 1.0)
        : 0.0;
}

CapabilityProfile capability_at(
    const SpeedProfileOptimizer::Params& params,
    const std::vector<StepPlanSegment>& segments,
    const double progress
) {
    std::optional<CapabilityProfile> effective;
    for (const StepPlanSegment& segment : segments) {
        const double blend = capability_blend(segment, progress);
        if (blend <= 0.0) continue;
        const CapabilityProfile candidate = interpolate_profile(
            params.normal_profile,
            params.step_profiles[static_cast<size_t>(segment.chassis_command.capability)],
            blend
        );
        if (!effective) {
            effective = candidate;
            continue;
        }
        auto& envelope = effective->command_envelope;
        auto& dynamics = effective->command_dynamics;
        envelope.velocity.max = std::min(
            envelope.velocity.max, candidate.command_envelope.velocity.max
        );
        envelope.angular_velocity.min = std::max(
            envelope.angular_velocity.min, candidate.command_envelope.angular_velocity.min
        );
        envelope.angular_velocity.max = std::min(
            envelope.angular_velocity.max, candidate.command_envelope.angular_velocity.max
        );
        dynamics.velocity_rate_max = std::min(
            dynamics.velocity_rate_max, candidate.command_dynamics.velocity_rate_max
        );
        dynamics.angular_velocity_rate_max = std::min(
            dynamics.angular_velocity_rate_max,
            candidate.command_dynamics.angular_velocity_rate_max
        );
        dynamics.lateral_acceleration_max = std::min(
            dynamics.lateral_acceleration_max,
            candidate.command_dynamics.lateral_acceleration_max
        );
    }
    return effective.value_or(params.normal_profile);
}

double angular_velocity_magnitude_max(const CapabilityProfile& capability) {
    return std::min(
        capability.command_envelope.angular_velocity.max,
        -capability.command_envelope.angular_velocity.min
    );
}

LocalEnvelope local_envelope(
    const SpeedProfileOptimizer::Params& params,
    const std::vector<StepPlanSegment>& segments,
    const TrajSample& sample,
    const double progress
) {
    const CapabilityProfile capability = capability_at(params, segments, progress);
    const auto& command = capability.command_envelope;
    const auto& dynamics = capability.command_dynamics;
    const double curvature = std::abs(sample.kappa);
    // MINCO 已用同一几何上界塑形 κ'。速度包络仍对数值超调做物理限幅，避免
    // 单个离散尖峰通过 1/|κ'| 把相邻两个执行节点压到近零速度。
    const double curvature_rate = std::min(
        std::abs(sample.kappa_rate), params.geometry.curvature_rate_max
    );
    const double angular_budget = ANGULAR_ACCELERATION_SHARE
        * dynamics.angular_velocity_rate_max;

    LocalEnvelope envelope;
    envelope.speed_squared_upper = command.velocity.max * command.velocity.max;
    envelope.acceleration_upper = dynamics.velocity_rate_max;
    if (curvature > EPS) {
        // |κ|√z ≤ ω_max。侧向加速度与 MPC 一致，作为软约束单独进入目标函数。
        const double angular_velocity_max = angular_velocity_magnitude_max(capability);
        envelope.speed_squared_upper = std::min(
            envelope.speed_squared_upper,
            angular_velocity_max * angular_velocity_max / (curvature * curvature)
        );
        // |κ|·|dz/ds|/2 ≤ SHARE·α_max，其中 dz/ds = 2·a_t
        envelope.acceleration_upper = std::min(
            envelope.acceleration_upper, angular_budget / curvature
        );
    }
    if (curvature_rate > EPS) {
        // |κ'|z ≤ SHARE·α_max
        envelope.speed_squared_upper = std::min(
            envelope.speed_squared_upper, angular_budget / curvature_rate
        );
    }
    envelope.speed_squared_upper = std::max(envelope.speed_squared_upper, 0.0);
    envelope.acceleration_upper = std::max(envelope.acceleration_upper, 0.0);
    return envelope;
}

std::vector<double> build_nodes(
    const SpeedProfileOptimizer::Params& params,
    const MincoTrajectory& geometry,
    const std::vector<StepPlanSegment>& segments
) {
    const double total_length = geometry.total_arc_length();
    std::vector<double> mandatory {0.0, total_length};
    for (int boundary = 0; boundary <= geometry.segment_count(); ++boundary) {
        mandatory.push_back(geometry.segment_boundary_arc_length(boundary));
    }
    for (const StepPlanSegment& segment : segments) {
        mandatory.insert(mandatory.end(), {
            segment.prepare_arc_length,
            segment.active_arc_length,
            segment.commit_arc_length,
            segment.step_enter_arc_length,
            segment.step_exit_arc_length,
            segment.release_arc_length,
        });
    }
    std::sort(mandatory.begin(), mandatory.end());
    mandatory.erase(std::unique(
        mandatory.begin(), mandatory.end(),
        [](const double a, const double b) { return std::abs(a - b) <= EPS; }
    ), mandatory.end());

    std::vector<double> nodes;
    for (size_t interval = 0; interval + 1 < mandatory.size(); ++interval) {
        const double begin = std::clamp(mandatory[interval], 0.0, total_length);
        const double end = std::clamp(mandatory[interval + 1], 0.0, total_length);
        if (nodes.empty() || begin > nodes.back() + EPS) nodes.push_back(begin);
        double spacing = params.discretization.max_spacing;
        for (const StepPlanSegment& segment : segments) {
            if (end >= segment.prepare_arc_length - EPS
                && begin <= segment.release_arc_length + EPS) {
                spacing = std::min(spacing, params.discretization.step_max_spacing);
            }
        }
        const int pieces = std::max(1, static_cast<int>(std::ceil((end - begin) / spacing)));
        for (int piece = 1; piece <= pieces; ++piece) {
            nodes.push_back(std::lerp(
                begin, end, static_cast<double>(piece) / static_cast<double>(pieces)
            ));
        }
    }

    // 曲率变化大的区间递归二分，最小间距由常规间距的 1/8 限制。
    const double minimum_spacing = params.discretization.max_spacing / 8.0;
    for (int pass = 0; pass < 8; ++pass) {
        bool refined = false;
        std::vector<double> next;
        next.reserve(nodes.size() * 2);
        next.push_back(nodes.front());
        for (size_t i = 0; i + 1 < nodes.size(); ++i) {
            const double begin = nodes[i];
            const double end = nodes[i + 1];
            const double midpoint = 0.5 * (begin + end);
            const double k0 = geometry.eval_arc_length(begin).kappa;
            const double km = geometry.eval_arc_length(midpoint).kappa;
            const double k1 = geometry.eval_arc_length(end).kappa;
            const double variation = std::max({
                std::abs(km - k0), std::abs(k1 - km), std::abs(k1 - k0)
            });
            if (end - begin > 2.0 * minimum_spacing
                && variation > params.discretization.curvature_refine_threshold) {
                next.push_back(midpoint);
                refined = true;
            }
            next.push_back(end);
        }
        nodes = std::move(next);
        if (!refined) break;
    }
    return nodes;
}

std::vector<NodeLimit> build_limits(
    const SpeedProfileOptimizer::Params& params,
    const MincoTrajectory& geometry,
    const std::vector<StepPlanSegment>& segments,
    const std::vector<double>& nodes
) {
    const auto envelope_at = [&](const double progress) {
        return local_envelope(
            params, segments, geometry.eval_arc_length(progress), progress
        );
    };

    std::vector<NodeLimit> limits(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        const LocalEnvelope envelope = envelope_at(nodes[i]);
        limits[i].speed_squared_upper = envelope.speed_squared_upper;
        limits[i].acceleration = envelope.acceleration_upper;
    }
    // 每个区间内更密采样，把几何峰值转换为保守的端点上限与区间加速度上限。
    for (size_t i = 0; i + 1 < nodes.size(); ++i) {
        const double distance = nodes[i + 1] - nodes[i];
        const int samples = std::max(
            1, static_cast<int>(std::ceil(
                distance / params.discretization.envelope_sample_spacing
            ))
        );
        double interval_upper = INF;
        double interval_acceleration = INF;
        for (int sample = 0; sample <= samples; ++sample) {
            const double progress = std::lerp(
                nodes[i], nodes[i + 1],
                static_cast<double>(sample) / static_cast<double>(samples)
            );
            const LocalEnvelope envelope = envelope_at(progress);
            interval_upper = std::min(interval_upper, envelope.speed_squared_upper);
            interval_acceleration = std::min(
                interval_acceleration, envelope.acceleration_upper
            );
        }
        limits[i].speed_squared_upper = std::min(
            limits[i].speed_squared_upper, interval_upper
        );
        limits[i + 1].speed_squared_upper = std::min(
            limits[i + 1].speed_squared_upper, interval_upper
        );
        limits[i].acceleration = interval_acceleration;
    }
    return limits;
}

std::optional<std::vector<double>> reachable_seed(
    const std::vector<double>& nodes,
    const std::vector<NodeLimit>& limits,
    const double initial_speed_squared,
    std::string& error
) {
    if (nodes.size() < 2 || nodes.size() != limits.size()) {
        error = "speed profile has insufficient nodes";
        return std::nullopt;
    }
    std::vector<double> speed_squared(nodes.size(), 0.0);
    speed_squared[0] = initial_speed_squared;
    for (size_t i = 0; i + 1 < nodes.size(); ++i) {
        const double distance = nodes[i + 1] - nodes[i];
        if (!std::isfinite(distance) || distance <= 0.0
            || !std::isfinite(limits[i].acceleration)
            || limits[i].acceleration <= 0.0) {
            error = "speed profile nodes or acceleration limits are invalid";
            return std::nullopt;
        }
        speed_squared[i + 1] = std::min(
            limits[i + 1].speed_squared_upper,
            speed_squared[i] + 2.0 * limits[i].acceleration * distance
        );
    }
    speed_squared.back() = 0.0;
    for (size_t reverse = nodes.size() - 1; reverse > 0; --reverse) {
        const size_t i = reverse - 1;
        const double distance = nodes[i + 1] - nodes[i];
        speed_squared[i] = std::min(
            speed_squared[i],
            speed_squared[i + 1] + 2.0 * limits[i].acceleration * distance
        );
    }
    if (speed_squared.front() + EPS < initial_speed_squared) {
        error = "initial speed cannot reach the terminal or local speed envelope";
        return std::nullopt;
    }
    speed_squared.front() = initial_speed_squared;
    return speed_squared;
}

PathSpeedProfile make_profile(
    const std::vector<double>& nodes,
    const std::vector<double>& speed_squared
) {
    std::vector<SpeedProfileState> states(nodes.size());
    double time = 0.0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const double velocity = std::sqrt(std::max(speed_squared[i], 0.0));
        states[i] = {
            .arc_length = nodes[i],
            .time = time,
            .velocity = velocity,
        };
        if (i + 1 < nodes.size()) {
            const double next_velocity = std::sqrt(std::max(speed_squared[i + 1], 0.0));
            const double velocity_sum = velocity + next_velocity;
            if (velocity_sum <= EPS) return {};
            time += 2.0 * (nodes[i + 1] - nodes[i]) / velocity_sum;
        }
    }
    return PathSpeedProfile(std::move(states));
}

std::vector<TraversalWindowNode> collect_traversal_windows(
    const std::vector<double>& nodes,
    const std::vector<StepPlanSegment>& segments
) {
    std::vector<TraversalWindowNode> windows;
    for (const StepPlanSegment& segment : segments) {
        for (size_t node_index = 0; node_index < nodes.size(); ++node_index) {
            const double gate = step_window_gate(
                nodes[node_index], segment.traversal_constraint
            );
            if (gate <= 0.0) continue;
            const double left = node_index > 0
                ? 0.5 * (nodes[node_index] - nodes[node_index - 1]) : 0.0;
            const double right = node_index + 1 < nodes.size()
                ? 0.5 * (nodes[node_index + 1] - nodes[node_index]) : 0.0;
            windows.push_back({
                .node_index = node_index,
                .integration_weight = left + right,
                .gate = gate,
                .target = segment.traversal_constraint.velocity_window,
            });
        }
    }
    return windows;
}

std::vector<LateralAccelerationNode> collect_lateral_acceleration_nodes(
    const SpeedProfileOptimizer::Params& params,
    const MincoTrajectory& geometry,
    const std::vector<StepPlanSegment>& segments,
    const std::vector<double>& nodes
) {
    std::vector<LateralAccelerationNode> constraints;
    if (params.objective.lateral_acceleration == 0.0) return constraints;

    constraints.reserve(nodes.size());
    for (size_t node_index = 0; node_index < nodes.size(); ++node_index) {
        const double curvature = std::abs(
            geometry.eval_arc_length(nodes[node_index]).kappa
        );
        if (curvature <= EPS) continue;
        const CapabilityProfile capability = capability_at(
            params, segments, nodes[node_index]
        );
        const double left = node_index > 0
            ? 0.5 * (nodes[node_index] - nodes[node_index - 1]) : 0.0;
        const double right = node_index + 1 < nodes.size()
            ? 0.5 * (nodes[node_index + 1] - nodes[node_index]) : 0.0;
        constraints.push_back({
            .node_index = node_index,
            .integration_weight = left + right,
            .curvature = curvature,
            .speed_squared_upper =
                capability.command_dynamics.lateral_acceleration_max / curvature,
        });
    }
    return constraints;
}

ChainProblem build_chain_problem(
    const SpeedProfileOptimizer::Params& params,
    const std::vector<double>& nodes,
    const std::vector<NodeLimit>& limits,
    const std::vector<TraversalWindowNode>& windows,
    const std::vector<LateralAccelerationNode>& lateral_constraints,
    const double initial_speed_squared
) {
    const double speed_squared_scale = params.objective.velocity_scale
        * params.objective.velocity_scale;
    ChainProblem problem;
    problem.linear_reward.resize(nodes.size());
    problem.node_upper.resize(nodes.size());
    problem.step_limit.resize(nodes.size() - 1);
    problem.initial_value = initial_speed_squared / speed_squared_scale;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const double left = i > 0 ? 0.5 * (nodes[i] - nodes[i - 1]) : 0.0;
        const double right = i + 1 < nodes.size()
            ? 0.5 * (nodes[i + 1] - nodes[i]) : 0.0;
        problem.linear_reward[i] = params.objective.global_speed_reward * (left + right);
        problem.node_upper[i] = limits[i].speed_squared_upper / speed_squared_scale;
        if (i + 1 < nodes.size()) {
            problem.step_limit[i] = 2.0 * limits[i].acceleration
                * (nodes[i + 1] - nodes[i]) / speed_squared_scale;
        }
    }
    problem.soft_windows.reserve(windows.size() + lateral_constraints.size());
    for (const TraversalWindowNode& window : windows) {
        problem.soft_windows.push_back({
            .node_index = window.node_index,
            .lower = window.target.min * window.target.min / speed_squared_scale,
            .upper = window.target.max * window.target.max / speed_squared_scale,
            .weight = params.objective.traversal_window * window.integration_weight
                * window.gate * window.gate,
        });
    }
    for (const LateralAccelerationNode& constraint : lateral_constraints) {
        // q=z/scale²。将 (|κ|z-a_lat,max)² 的物理违规准确换算到 q 空间。
        const double acceleration_per_q = constraint.curvature * speed_squared_scale;
        problem.soft_windows.push_back({
            .node_index = constraint.node_index,
            .lower = 0.0,
            .upper = constraint.speed_squared_upper / speed_squared_scale,
            .weight = params.objective.lateral_acceleration
                * constraint.integration_weight
                * acceleration_per_q * acceleration_per_q,
        });
    }
    return problem;
}

// 发布契约只保证弧长时标在数学上可用。动力学包络负责塑造优化问题，不能把其
// 离散近似或轻微物理超限再次放大为发布拒绝。
bool validate_profile_contract(
    const MincoTrajectory& geometry,
    const PathSpeedProfile& profile,
    const double initial_velocity,
    std::string& error
) {
    if (profile.empty() || !std::isfinite(profile.total_time())
        || profile.total_time() <= 0.0) {
        error = "speed profile has invalid or zero traversal time";
        return false;
    }
    const auto& states = profile.states();
    const double total_length = geometry.total_arc_length();
    const double numerical_tolerance = 64.0
        * std::numeric_limits<double>::epsilon();
    const double length_tolerance = numerical_tolerance
        * std::max(total_length, 1.0);
    const double velocity_tolerance = numerical_tolerance
        * std::max(initial_velocity, 1.0);
    if (std::abs(states.front().arc_length) > length_tolerance
        || std::abs(states.back().arc_length - total_length) > length_tolerance) {
        error = "speed profile does not cover the complete trajectory arc length";
        return false;
    }
    if (std::abs(states.front().time) > numerical_tolerance) {
        error = "speed profile does not start at zero time";
        return false;
    }
    if (std::abs(states.front().velocity - initial_velocity) > velocity_tolerance
        || std::abs(states.back().velocity) > velocity_tolerance) {
        error = "speed profile violates an endpoint velocity constraint";
        return false;
    }
    for (size_t i = 0; i < states.size(); ++i) {
        if (!std::isfinite(states[i].arc_length) || !std::isfinite(states[i].time)
            || !std::isfinite(states[i].velocity)
            || states[i].velocity < 0.0
            || (i > 0 && (states[i].arc_length <= states[i - 1].arc_length
                || states[i].time <= states[i - 1].time))) {
            error = "speed profile contains non-finite or non-monotone states";
            return false;
        }
    }
    return true;
}

std::vector<SpeedProfileOptimizer::StepWindowViolation> step_violations(
    const std::vector<StepPlanSegment>& segments,
    const PathSpeedProfile& profile
) {
    std::vector<SpeedProfileOptimizer::StepWindowViolation> diagnostics;
    diagnostics.reserve(segments.size());
    for (size_t segment_index = 0; segment_index < segments.size(); ++segment_index) {
        const StepPlanSegment& segment = segments[segment_index];
        SpeedProfileOptimizer::StepWindowViolation violation;
        violation.segment_index = segment_index;
        violation.target = segment.traversal_constraint.velocity_window;
        for (const SpeedProfileState& state : profile.states()) {
            if (state.arc_length + EPS < segment.commit_arc_length
                || state.arc_length - EPS > segment.step_exit_arc_length) {
                continue;
            }
            const double under = std::max(violation.target.min - state.velocity, 0.0);
            const double over = std::max(state.velocity - violation.target.max, 0.0);
            if (std::max(under, over)
                > std::max(violation.max_under_speed, violation.max_over_speed)) {
                violation.arc_length = state.arc_length;
            }
            violation.max_under_speed = std::max(violation.max_under_speed, under);
            violation.max_over_speed = std::max(violation.max_over_speed, over);
        }
        diagnostics.push_back(violation);
    }
    return diagnostics;
}

} // anonymous namespace

SpeedProfileOptimizer::Result SpeedProfileOptimizer::optimize(
    const MincoTrajectory& geometry,
    const std::vector<StepPlanSegment>& step_segments,
    const Eigen::Vector2d& current_velocity_map
) const {
    Result result;
    if (geometry.empty() || !std::isfinite(geometry.total_arc_length())
        || geometry.total_arc_length() <= 0.0 || !current_velocity_map.allFinite()) {
        result.error = "speed profile received invalid geometry or initial velocity";
        return result;
    }
    const std::vector<double> nodes = build_nodes(params_, geometry, step_segments);
    std::vector<NodeLimit> limits = build_limits(params_, geometry, step_segments, nodes);

    // 起点速度取当前速度在有向路径切线上的前向投影。
    const TrajSample start = geometry.eval_arc_length(0.0);
    const Eigen::Vector2d tangent(std::cos(start.theta), std::sin(start.theta));
    const double projected_initial_velocity = std::max(
        0.0, current_velocity_map.dot(tangent)
    );
    const double initial_velocity = projected_initial_velocity
            <= params_.stationary_velocity_threshold
        ? 0.0
        : projected_initial_velocity;
    const double initial_speed_squared = initial_velocity * initial_velocity;
    // 起点是不可修改的当前事实；局部包络从下一空间位置开始约束。
    limits.front().speed_squared_upper = std::max(
        limits.front().speed_squared_upper, initial_speed_squared
    );

    std::string seed_error;
    const auto seed_squared = reachable_seed(
        nodes, limits, initial_speed_squared, seed_error
    );
    if (!seed_squared) {
        result.error = seed_error;
        return result;
    }
    const PathSpeedProfile seed_profile = make_profile(nodes, *seed_squared);
    std::string contract_error;
    if (!validate_profile_contract(
            geometry, seed_profile, initial_velocity, contract_error
        )) {
        result.error = "reachable solution violates the profile contract: "
            + contract_error;
        return result;
    }

    const std::vector<TraversalWindowNode> windows = collect_traversal_windows(
        nodes, step_segments
    );
    const std::vector<LateralAccelerationNode> lateral_constraints =
        collect_lateral_acceleration_nodes(params_, geometry, step_segments, nodes);
    result.diagnostics.node_count = static_cast<int>(nodes.size());
    result.diagnostics.traversal_window_constraint_count = static_cast<int>(windows.size());
    result.diagnostics.lateral_acceleration_constraint_count =
        static_cast<int>(lateral_constraints.size());
    result.diagnostics.seed_total_time = seed_profile.total_time();

    PathSpeedProfile selected = seed_profile;
    if (windows.empty() && lateral_constraints.empty()) {
        result.diagnostics.selection =
            Diagnostics::Selection::CLOSED_FORM_NO_SOFT_CONSTRAINT;
    } else {
        const ChainProblem problem = build_chain_problem(
            params_, nodes, limits, windows, lateral_constraints, initial_speed_squared
        );
        const PiecewiseQuadraticChainSolver::Result optimized_result =
            PiecewiseQuadraticChainSolver::solve(problem);
        result.diagnostics.max_breakpoints = optimized_result.max_breakpoints;
        result.diagnostics.solve_ms = optimized_result.solve_ms;

        if (optimized_result.status != PiecewiseQuadraticChainSolver::Status::OPTIMAL) {
            result.error = optimized_result.error.empty()
                ? "chain solver did not return an optimal result"
                : optimized_result.error;
            return result;
        } else if (optimized_result.value.size() != nodes.size()) {
            result.error = "chain solution dimension does not match the node count";
            return result;
        } else if (!std::all_of(
                optimized_result.value.begin(), optimized_result.value.end(),
                [](const double value) {
                    return std::isfinite(value) && value >= -EPS;
                }
            )) {
            result.error = "chain solution contains a non-finite or negative value";
            return result;
        }

        const double speed_squared_scale = params_.objective.velocity_scale
            * params_.objective.velocity_scale;
        std::vector<double> optimized_squared(nodes.size());
        std::transform(
            optimized_result.value.begin(), optimized_result.value.end(),
            optimized_squared.begin(),
            [speed_squared_scale](const double value) {
                return std::max(value * speed_squared_scale, 0.0);
            }
        );
        // 端点是精确边界条件，不把链求解器的浮点残差发布为非零起停速度。
        optimized_squared.front() = initial_speed_squared;
        optimized_squared.back() = 0.0;
        PathSpeedProfile optimized = make_profile(nodes, optimized_squared);
        if (!validate_profile_contract(
                geometry, optimized, initial_velocity, contract_error
            )) {
            result.error = "optimized solution violates the profile contract: "
                + contract_error;
            return result;
        }
        selected = std::move(optimized);
        result.diagnostics.selection = Diagnostics::Selection::OPTIMAL;
    }

    result.diagnostics.result_total_time = selected.total_time();
    const double scale_squared = params_.objective.velocity_scale
        * params_.objective.velocity_scale;
    for (size_t i = 0; i + 1 < nodes.size(); ++i) {
        const double z0 = std::pow(selected.eval_arc_length(nodes[i]).velocity, 2);
        const double z1 = std::pow(selected.eval_arc_length(nodes[i + 1]).velocity, 2);
        result.diagnostics.speed_reward_cost -= params_.objective.global_speed_reward
            * (nodes[i + 1] - nodes[i]) * 0.5 * (z0 + z1) / scale_squared;
    }
    for (const TraversalWindowNode& window : windows) {
        const double velocity = selected.eval_arc_length(nodes[window.node_index]).velocity;
        const double z = velocity * velocity / scale_squared;
        const double lower = window.target.min * window.target.min / scale_squared;
        const double upper = window.target.max * window.target.max / scale_squared;
        const double violation = z < lower ? lower - z : (z > upper ? z - upper : 0.0);
        result.diagnostics.traversal_window_cost += params_.objective.traversal_window
            * window.integration_weight * violation * violation;
    }
    for (const LateralAccelerationNode& constraint : lateral_constraints) {
        const double velocity = selected.eval_arc_length(
            nodes[constraint.node_index]
        ).velocity;
        const double acceleration = constraint.curvature * velocity * velocity;
        const double acceleration_upper = constraint.curvature
            * constraint.speed_squared_upper;
        const double violation = std::max(acceleration - acceleration_upper, 0.0);
        result.diagnostics.lateral_acceleration_cost +=
            params_.objective.lateral_acceleration
            * constraint.integration_weight * violation * violation;
    }
    result.diagnostics.step_violations = step_violations(step_segments, selected);
    result.profile = std::move(selected);
    result.success = true;
    return result;
}

} // namespace nav_executor
