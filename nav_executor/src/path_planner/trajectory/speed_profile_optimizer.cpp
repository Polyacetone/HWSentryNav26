#include <nav_executor/path_planner/trajectory/speed_profile_optimizer.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>

#include <Eigen/SparseCore>

namespace nav_executor {

namespace {

constexpr double EPS = 1e-9;
constexpr double INF = std::numeric_limits<double>::infinity();

struct NodeLimit {
    double speed_squared_upper = 0.0;
    double acceleration = 0.0;
};

struct SoftWindowNode {
    size_t node_index = 0;
    size_t segment_index = 0;
    double integration_weight = 0.0;
    TraversalVelocityWindow target;
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
        effective->command_envelope.velocity.max = std::min(
            effective->command_envelope.velocity.max,
            candidate.command_envelope.velocity.max
        );
        effective->command_envelope.angular_velocity.min = std::max(
            effective->command_envelope.angular_velocity.min,
            candidate.command_envelope.angular_velocity.min
        );
        effective->command_envelope.angular_velocity.max = std::min(
            effective->command_envelope.angular_velocity.max,
            candidate.command_envelope.angular_velocity.max
        );
        effective->command_dynamics.velocity_rate_max = std::min(
            effective->command_dynamics.velocity_rate_max,
            candidate.command_dynamics.velocity_rate_max
        );
        effective->command_dynamics.lateral_acceleration_max = std::min(
            effective->command_dynamics.lateral_acceleration_max,
            candidate.command_dynamics.lateral_acceleration_max
        );
    }
    return effective.value_or(params.normal_profile);
}

double local_speed_squared_upper(
    const SpeedProfileOptimizer::Params& params,
    const std::vector<StepPlanSegment>& segments,
    const TrajSample& sample,
    const double progress
) {
    const CapabilityProfile capability = capability_at(params, segments, progress);
    double upper = std::min(
        params.trajectory_velocity_max * params.trajectory_velocity_max,
        capability.command_envelope.velocity.max * capability.command_envelope.velocity.max
    );
    const double curvature = sample.kappa;
    const double absolute_curvature = std::abs(curvature);
    if (absolute_curvature > EPS) {
        const double angular_velocity_allowed = curvature > 0.0
            ? std::min(
                params.trajectory_angular_velocity_max,
                capability.command_envelope.angular_velocity.max
            )
            : std::min(
                params.trajectory_angular_velocity_max,
                -capability.command_envelope.angular_velocity.min
            );
        upper = std::min(
            upper,
            angular_velocity_allowed * angular_velocity_allowed
                / (curvature * curvature)
        );
        upper = std::min(
            upper,
            std::min(
                params.trajectory_lateral_acceleration_max,
                capability.command_dynamics.lateral_acceleration_max
            ) / absolute_curvature
        );
    }
    return std::max(upper, 0.0);
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
    std::vector<NodeLimit> limits(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        limits[i].speed_squared_upper = local_speed_squared_upper(
            params, segments, geometry.eval_arc_length(nodes[i]), nodes[i]
        );
        const CapabilityProfile capability = capability_at(params, segments, nodes[i]);
        limits[i].acceleration = std::min(
            params.trajectory_acceleration_max,
            capability.command_dynamics.velocity_rate_max
        );
    }
    // 对每个区间做更密采样，将几何峰值转换为保守的端点上限。
    for (size_t i = 0; i + 1 < nodes.size(); ++i) {
        const double distance = nodes[i + 1] - nodes[i];
        const int samples = std::max(
            1, static_cast<int>(std::ceil(distance / params.validation.sample_spacing))
        );
        double interval_upper = INF;
        double interval_acceleration = INF;
        for (int sample_index = 0; sample_index <= samples; ++sample_index) {
            const double progress = std::lerp(
                nodes[i], nodes[i + 1],
                static_cast<double>(sample_index) / static_cast<double>(samples)
            );
            interval_upper = std::min(
                interval_upper,
                local_speed_squared_upper(
                    params, segments, geometry.eval_arc_length(progress), progress
                )
            );
            interval_acceleration = std::min(
                interval_acceleration,
                std::min(
                    params.trajectory_acceleration_max,
                    capability_at(params, segments, progress).command_dynamics.velocity_rate_max
                )
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
    const std::vector<NodeLimit>& limits,
    const std::vector<double>& speed_squared
) {
    std::vector<SpeedProfileState> states(nodes.size());
    double time = 0.0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const double velocity = std::sqrt(std::max(speed_squared[i], 0.0));
        double acceleration = 0.0;
        if (i + 1 < nodes.size()) {
            acceleration = (speed_squared[i + 1] - speed_squared[i])
                / (2.0 * (nodes[i + 1] - nodes[i]));
        } else if (i > 0) {
            acceleration = states[i - 1].acceleration;
        }
        states[i] = {
            .arc_length = nodes[i],
            .time = time,
            .velocity = velocity,
            .acceleration = acceleration,
            .velocity_upper = std::sqrt(std::max(limits[i].speed_squared_upper, 0.0)),
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

std::vector<SoftWindowNode> collect_soft_windows(
    const std::vector<double>& nodes,
    const std::vector<StepPlanSegment>& segments
) {
    std::vector<SoftWindowNode> windows;
    for (size_t segment_index = 0; segment_index < segments.size(); ++segment_index) {
        const StepPlanSegment& segment = segments[segment_index];
        for (size_t node_index = 0; node_index < nodes.size(); ++node_index) {
            if (nodes[node_index] + EPS < segment.commit_arc_length
                || nodes[node_index] - EPS > segment.step_exit_arc_length) {
                continue;
            }
            const double left = node_index > 0
                ? 0.5 * (nodes[node_index] - nodes[node_index - 1]) : 0.0;
            const double right = node_index + 1 < nodes.size()
                ? 0.5 * (nodes[node_index + 1] - nodes[node_index]) : 0.0;
            windows.push_back({
                .node_index = node_index,
                .segment_index = segment_index,
                .integration_weight = left + right,
                .target = segment.traversal_constraint.velocity_window,
            });
        }
    }
    return windows;
}

SparseQpProblem build_qp(
    const SpeedProfileOptimizer::Params& params,
    const std::vector<double>& nodes,
    const std::vector<NodeLimit>& limits,
    const std::vector<SoftWindowNode>& windows,
    const double initial_speed_squared
) {
    const int node_count = static_cast<int>(nodes.size());
    const int variable_count = node_count + 2 * static_cast<int>(windows.size());
    const double speed_squared_scale = params.objective.velocity_scale
        * params.objective.velocity_scale;

    std::vector<Eigen::Triplet<double>> quadratic_triplets;
    Eigen::VectorXd linear = Eigen::VectorXd::Zero(variable_count);
    for (int i = 0; i < node_count; ++i) {
        const double left = i > 0 ? 0.5 * (nodes[static_cast<size_t>(i)] - nodes[static_cast<size_t>(i - 1)]) : 0.0;
        const double right = i + 1 < node_count ? 0.5 * (nodes[static_cast<size_t>(i + 1)] - nodes[static_cast<size_t>(i)]) : 0.0;
        linear(i) = -params.objective.global_speed_reward * (left + right);
    }
    for (size_t window_index = 0; window_index < windows.size(); ++window_index) {
        const double diagonal = 2.0 * params.objective.traversal_window
            * windows[window_index].integration_weight;
        quadratic_triplets.emplace_back(
            node_count + 2 * static_cast<int>(window_index),
            node_count + 2 * static_cast<int>(window_index), diagonal
        );
        quadratic_triplets.emplace_back(
            node_count + 2 * static_cast<int>(window_index) + 1,
            node_count + 2 * static_cast<int>(window_index) + 1, diagonal
        );
    }
    Eigen::SparseMatrix<double> quadratic(variable_count, variable_count);
    quadratic.setFromTriplets(quadratic_triplets.begin(), quadratic_triplets.end());

    const int row_count = variable_count + (node_count - 1)
        + 2 * static_cast<int>(windows.size());
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(row_count, -INF);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(row_count, INF);
    std::vector<Eigen::Triplet<double>> constraint_triplets;
    int row = 0;
    for (int i = 0; i < node_count; ++i, ++row) {
        constraint_triplets.emplace_back(row, i, 1.0);
        lower(row) = 0.0;
        upper(row) = limits[static_cast<size_t>(i)].speed_squared_upper
            / speed_squared_scale;
    }
    lower(0) = initial_speed_squared / speed_squared_scale;
    upper(0) = lower(0);
    lower(node_count - 1) = 0.0;
    upper(node_count - 1) = 0.0;
    for (int i = node_count; i < variable_count; ++i, ++row) {
        constraint_triplets.emplace_back(row, i, 1.0);
        lower(row) = 0.0;
    }
    for (int i = 0; i + 1 < node_count; ++i, ++row) {
        const double distance = nodes[static_cast<size_t>(i + 1)]
            - nodes[static_cast<size_t>(i)];
        const double scale = speed_squared_scale
            / (2.0 * params.trajectory_acceleration_max * distance);
        constraint_triplets.emplace_back(row, i, -scale);
        constraint_triplets.emplace_back(row, i + 1, scale);
        const double normalized_limit = limits[static_cast<size_t>(i)].acceleration
            / params.trajectory_acceleration_max;
        lower(row) = -normalized_limit;
        upper(row) = normalized_limit;
    }
    for (size_t window_index = 0; window_index < windows.size(); ++window_index) {
        const SoftWindowNode& window = windows[window_index];
        const int lower_slack = node_count + 2 * static_cast<int>(window_index);
        const int upper_slack = lower_slack + 1;
        constraint_triplets.emplace_back(row, static_cast<int>(window.node_index), 1.0);
        constraint_triplets.emplace_back(row, lower_slack, 1.0);
        lower(row) = window.target.min * window.target.min / speed_squared_scale;
        ++row;
        constraint_triplets.emplace_back(row, static_cast<int>(window.node_index), 1.0);
        constraint_triplets.emplace_back(row, upper_slack, -1.0);
        upper(row) = window.target.max * window.target.max / speed_squared_scale;
        ++row;
    }
    Eigen::SparseMatrix<double> constraints(row_count, variable_count);
    constraints.setFromTriplets(constraint_triplets.begin(), constraint_triplets.end());
    return {
        .quadratic = std::move(quadratic),
        .linear = std::move(linear),
        .constraint_matrix = std::move(constraints),
        .lower = std::move(lower),
        .upper = std::move(upper),
    };
}

bool validate_profile(
    const SpeedProfileOptimizer::Params& params,
    const MincoTrajectory& geometry,
    const std::vector<StepPlanSegment>& segments,
    const PathSpeedProfile& profile,
    const double initial_speed_squared,
    std::string& error
) {
    if (profile.empty() || !std::isfinite(profile.total_time())
        || profile.total_time() <= 0.0) {
        error = "speed profile has invalid or zero traversal time";
        return false;
    }
    const auto& states = profile.states();
    if (std::abs(states.front().velocity - std::sqrt(std::max(initial_speed_squared, 0.0)))
            > params.validation.velocity_tolerance
        || std::abs(states.back().velocity) > params.validation.velocity_tolerance) {
        error = "speed profile violates an endpoint velocity constraint";
        return false;
    }
    for (size_t i = 0; i < states.size(); ++i) {
        if (!std::isfinite(states[i].arc_length) || !std::isfinite(states[i].time)
            || !std::isfinite(states[i].velocity) || !std::isfinite(states[i].acceleration)
            || states[i].velocity < 0.0
            || (i > 0 && (states[i].arc_length <= states[i - 1].arc_length
                || states[i].time <= states[i - 1].time))) {
            error = "speed profile contains non-finite or non-monotone states";
            return false;
        }
    }

    const double total_length = geometry.total_arc_length();
    const int samples = std::max(
        1, static_cast<int>(std::ceil(total_length / params.validation.sample_spacing))
    );
    for (int i = 0; i <= samples; ++i) {
        const double progress = total_length * static_cast<double>(i)
            / static_cast<double>(samples);
        const SpeedProfileState state = profile.eval_arc_length(progress);
        const TrajSample geometry_state = geometry.eval_arc_length(progress);
        const CapabilityProfile capability = capability_at(params, segments, progress);
        const double local_upper = std::sqrt(local_speed_squared_upper(
            params, segments, geometry_state, progress
        ));
        const double unavoidable_initial_upper = std::sqrt(std::max(
            initial_speed_squared - 2.0 * params.trajectory_acceleration_max * progress,
            0.0
        ));
        const double allowed_velocity = std::max(local_upper, unavoidable_initial_upper);
        if (state.velocity > allowed_velocity + params.validation.velocity_tolerance) {
            error = "speed profile violates a local velocity envelope at s="
                + std::to_string(progress);
            return false;
        }
        const double acceleration_limit = std::min(
            params.trajectory_acceleration_max,
            capability.command_dynamics.velocity_rate_max
        );
        if (std::abs(state.acceleration)
            > acceleration_limit + params.validation.acceleration_tolerance) {
            error = "speed profile violates tangential acceleration at s="
                + std::to_string(progress);
            return false;
        }
        const double angular_velocity = geometry_state.kappa * state.velocity;
        const double angular_limit = angular_velocity >= 0.0
            ? std::min(params.trajectory_angular_velocity_max, capability.command_envelope.angular_velocity.max)
            : std::min(params.trajectory_angular_velocity_max, -capability.command_envelope.angular_velocity.min);
        if (std::abs(angular_velocity)
            > angular_limit + params.validation.angular_velocity_tolerance) {
            error = "speed profile violates angular velocity at s=" + std::to_string(progress);
            return false;
        }
        const double lateral_acceleration = std::abs(
            geometry_state.kappa * state.velocity * state.velocity
        );
        const double lateral_limit = std::min(
            params.trajectory_lateral_acceleration_max,
            capability.command_dynamics.lateral_acceleration_max
        );
        if (lateral_acceleration
            > lateral_limit + params.validation.lateral_acceleration_tolerance) {
            error = "speed profile violates lateral acceleration at s="
                + std::to_string(progress);
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
                violation.hard_velocity_upper = state.velocity_upper;
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
    const TrajSample start = geometry.eval_arc_length(0.0);
    Eigen::Vector2d tangent(std::cos(start.theta), std::sin(start.theta));
    if (start.dp_dtau.norm() > EPS) tangent = start.dp_dtau.normalized();
    const double initial_velocity = std::max(0.0, current_velocity_map.dot(tangent));
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
    const PathSpeedProfile seed_profile = make_profile(nodes, limits, *seed_squared);
    std::string validation_error;
    if (!validate_profile(
            params_, geometry, step_segments, seed_profile,
            initial_speed_squared, validation_error
        )) {
        result.error = "reachable seed rejected: " + validation_error;
        return result;
    }

    const std::vector<SoftWindowNode> windows = collect_soft_windows(nodes, step_segments);
    const SparseQpProblem qp = build_qp(
        params_, nodes, limits, windows, initial_speed_squared
    );
    const double speed_squared_scale = params_.objective.velocity_scale
        * params_.objective.velocity_scale;
    Eigen::VectorXd initial = Eigen::VectorXd::Zero(qp.linear.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        initial(static_cast<Eigen::Index>(i)) = (*seed_squared)[i] / speed_squared_scale;
    }
    for (size_t i = 0; i < windows.size(); ++i) {
        const double z = (*seed_squared)[windows[i].node_index] / speed_squared_scale;
        initial(static_cast<Eigen::Index>(nodes.size() + 2 * i)) = std::max(
            windows[i].target.min * windows[i].target.min / speed_squared_scale - z,
            0.0
        );
        initial(static_cast<Eigen::Index>(nodes.size() + 2 * i + 1)) = std::max(
            z - windows[i].target.max * windows[i].target.max / speed_squared_scale,
            0.0
        );
    }

    const AdmmQpSolver qp_solver(params_.solver);
    const AdmmQpSolver::Result qp_result = qp_solver.solve(qp, initial);
    result.diagnostics.node_count = static_cast<int>(nodes.size());
    result.diagnostics.variable_count = static_cast<int>(qp.linear.size());
    result.diagnostics.constraint_count = static_cast<int>(qp.lower.size());
    result.diagnostics.soft_window_node_count = static_cast<int>(windows.size());
    result.diagnostics.iterations = qp_result.iterations;
    result.diagnostics.rho_updates = qp_result.rho_updates;
    result.diagnostics.seed_total_time = seed_profile.total_time();
    result.diagnostics.primal_residual = qp_result.primal_residual;
    result.diagnostics.dual_residual = qp_result.dual_residual;
    result.diagnostics.max_constraint_violation = qp_result.max_constraint_violation;
    result.diagnostics.primal_tolerance = qp_result.primal_tolerance;
    result.diagnostics.dual_tolerance = qp_result.dual_tolerance;
    result.diagnostics.constraint_tolerance = qp_result.constraint_tolerance;
    result.diagnostics.final_rho = qp_result.final_rho;
    result.diagnostics.factorization_ms = qp_result.factorization_ms;
    result.diagnostics.iteration_ms = qp_result.iteration_ms;
    result.diagnostics.solver_status = qp_result.status;
    result.diagnostics.polish_status = qp_result.polish_status;
    result.diagnostics.solver_error = qp_result.error;

    PathSpeedProfile selected = seed_profile;
    const bool candidate_status = qp_result.status == AdmmQpSolver::Status::SOLVED
        || qp_result.status == AdmmQpSolver::Status::MAX_ITERATIONS;
    if (!candidate_status) {
        result.diagnostics.candidate_rejection = qp_result.error.empty()
            ? "QP solver did not produce an admissible candidate status"
            : qp_result.error;
    } else if (qp_result.solution.size() != qp.linear.size()) {
        result.diagnostics.candidate_rejection = "QP solution dimension does not match the problem";
    } else if (!qp_result.solution.allFinite()) {
        result.diagnostics.candidate_rejection = "QP solution contains non-finite values";
    } else {
        std::vector<double> optimized_squared(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            optimized_squared[i] = std::max(
                qp_result.solution(static_cast<Eigen::Index>(i)) * speed_squared_scale,
                0.0
            );
        }
        PathSpeedProfile optimized = make_profile(nodes, limits, optimized_squared);
        if (validate_profile(
                params_, geometry, step_segments, optimized,
                initial_speed_squared, validation_error
            )) {
            selected = std::move(optimized);
            result.diagnostics.selection = qp_result.status == AdmmQpSolver::Status::SOLVED
                ? Diagnostics::Selection::QP_SOLVED
                : Diagnostics::Selection::QP_MAX_ITERATIONS_VALIDATED;
        } else {
            result.diagnostics.candidate_rejection = validation_error;
        }
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
    for (const SoftWindowNode& window : windows) {
        const double velocity = selected.eval_arc_length(
            nodes[window.node_index]
        ).velocity;
        const double z = velocity * velocity / scale_squared;
        const double lower = window.target.min * window.target.min / scale_squared;
        const double upper = window.target.max * window.target.max / scale_squared;
        const double violation = z < lower ? lower - z : (z > upper ? z - upper : 0.0);
        result.diagnostics.traversal_window_cost += params_.objective.traversal_window
            * window.integration_weight * violation * violation;
    }
    result.diagnostics.step_violations = step_violations(step_segments, selected);
    result.profile = std::move(selected);
    result.success = true;
    return result;
}

} // namespace nav_executor
