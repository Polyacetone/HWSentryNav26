#include <nav_executor/task_manager/route_monitor.hpp>
#include <nav_executor/path_executor/monitoring/recovery_helpers.hpp>

#include <algorithm>
#include <cmath>
#include <optional>

#include <rclcpp/logging.hpp>

namespace nav_executor {

const char* replan_reason_str(const ReplanReason reason) {
    switch (reason) {
        case ReplanReason::NONE: return "NONE";
        case ReplanReason::ROUTE_TRACKING_LOST: return "ROUTE_TRACKING_LOST";
        case ReplanReason::PROJECTION_OUT_OF_MAP: return "PROJECTION_OUT_OF_MAP";
        case ReplanReason::PROJECTION_COST_EXCEEDED: return "PROJECTION_COST_EXCEEDED";
        case ReplanReason::STEP_BLOCKED_CURRENT: return "STEP_BLOCKED_CURRENT";
        case ReplanReason::STEP_BLOCKED_PREDICTED: return "STEP_BLOCKED_PREDICTED";
        case ReplanReason::MPC_LETHAL: return "MPC_LETHAL";
        case ReplanReason::EXECUTOR_REPLAN_EVENT: return "EXECUTOR_REPLAN_EVENT";
        case ReplanReason::PERFORMANCE_DEGRADED: return "PERFORMANCE_DEGRADED";
        case ReplanReason::PERFORMANCE_RECOVERED: return "PERFORMANCE_RECOVERED";
        case ReplanReason::HIGH_PRIORITY_SPIN_PREEMPTION: return "HIGH_PRIORITY_SPIN_PREEMPTION";
    }
    return "NONE";
}

namespace {

// 投影保护
std::optional<ReplanReason> check_projection_guard(const RouteMonitorInput& in, rclcpp::Logger logger) {
    const Eigen::Vector2d pos_map = in.chassis_pos_map;
    if (in.route.status != RouteTrackingStatus::TRACKED || in.route.path != in.active_path) {
        RCLCPP_WARN(logger, "RouteMonitor: route tracking lost (tracking_error=%.2f m)", in.route.tracking_error);
        return ReplanReason::ROUTE_TRACKING_LOST;
    }
    const Eigen::Vector2d proj_map = in.route.reference_position;

    if (!in.obstacles || !in.obstacles->hard_route_cost
        || in.proj_guard.cost_max < 0.0 || in.proj_guard.cost_max >= 255.0) {
        return std::nullopt;
    }

    const auto max_cost = recovery_helpers::max_cost_along_segment(
        *in.obstacles->hard_route_cost, pos_map, proj_map, in.proj_guard.cost_samples
    );
    if (!max_cost) {
        RCLCPP_WARN(logger, "RouteMonitor: projection segment out of hard route cost bounds");
        return ReplanReason::PROJECTION_OUT_OF_MAP;
    }
    if (*max_cost > in.proj_guard.cost_max) {
        RCLCPP_WARN(logger, "RouteMonitor: projection segment cost too high (max_cost=%.1f > %.1f)", *max_cost, in.proj_guard.cost_max);
        return ReplanReason::PROJECTION_COST_EXCEEDED;
    }
    return std::nullopt;
}

// 台阶阻塞
struct BlockSampleStats {
    int sample_count = 0;
    int blocked_sample_count = 0;
};

struct ArcInterval {
    double begin;
    double end;
};

std::vector<ArcInterval> build_step_access_intervals(
    const RouteMonitorInput& in,
    const MincoTrajectory& path
) {
    const double route_begin = std::clamp(
        in.route.arc_length, 0.0, path.total_arc_length()
    );
    const double route_end = std::min(
        path.total_arc_length(), route_begin + in.step_block.lookahead_distance
    );

    std::vector<ArcInterval> intervals;
    intervals.reserve(in.active_path->step_segments.size());
    for (const StepPlanSegment& segment : in.active_path->step_segments) {
        const double begin = std::max(route_begin, segment.prepare_arc_length);
        const double end = std::min(route_end, segment.release_arc_length);
        if (end <= begin) continue;
        intervals.push_back({begin, end});
    }
    std::sort(intervals.begin(), intervals.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.begin < rhs.begin;
    });

    std::vector<ArcInterval> merged;
    merged.reserve(intervals.size());
    for (const ArcInterval& interval : intervals) {
        if (merged.empty() || interval.begin > merged.back().end) {
            merged.push_back(interval);
        } else {
            merged.back().end = std::max(merged.back().end, interval.end);
        }
    }
    return merged;
}

// terrain_dynamic_timeline[i] 是 t = i * prediction_dt 时刻的帧，索引 0 为当前帧。
// 取时间上最近的一帧；超出预测时域的样本没有判定依据，返回空。
std::optional<size_t> prediction_frame_index(
    const size_t frame_count,
    const double prediction_dt,
    const double arrival_time
) {
    const long long nearest = std::llround(arrival_time / prediction_dt);
    if (nearest >= static_cast<long long>(frame_count)) return std::nullopt;
    return static_cast<size_t>(std::max<long long>(0, nearest));
}

std::optional<BlockSampleStats> sample_block_stats(
    const RouteMonitorInput& in,
    const MincoTrajectory& path,
    const bool using_prediction
) {
    const auto intervals = build_step_access_intervals(in, path);
    if (intervals.empty()) return BlockSampleStats {};

    const PathSpeedProfile* speed_profile = using_prediction
        ? &in.active_path->speed_profile : nullptr;
    const double route_progress = std::clamp(
        in.route.arc_length, 0.0, path.total_arc_length()
    );
    const double route_time = speed_profile
        ? speed_profile->eval_arc_length(route_progress).time : 0.0;

    BlockSampleStats stats;
    for (const ArcInterval& interval : intervals) {
        const double interval_length = interval.end - interval.begin;
        const int sample_count = std::max(
            1,
            static_cast<int>(std::ceil(
                interval_length / in.step_block.sample_resolution
            ))
        );
        const double sample_length = interval_length / static_cast<double>(sample_count);

        for (int index = 0; index < sample_count; ++index) {
            const double sample_progress = interval.begin
                + (static_cast<double>(index) + 0.5) * sample_length;
            size_t frame_index = 0;
            if (using_prediction) {
                const double arrival_time = std::max(
                    0.0,
                    speed_profile->eval_arc_length(sample_progress).time - route_time
                );
                const auto predicted_index = prediction_frame_index(
                    in.obstacles->terrain_dynamic_timeline.size(),
                    in.obstacles->prediction_dt,
                    arrival_time
                );
                if (!predicted_index) continue;
                frame_index = *predicted_index;
            }

            const CostMap::ConstPtr& cost_map =
                in.obstacles->terrain_dynamic_timeline[frame_index];
            if (!cost_map) return std::nullopt;

            const Eigen::Vector2d pos = path.eval_arc_length(sample_progress).p;
            const auto cost = cost_map->sample_map(pos);
            if (!cost) return std::nullopt;

            stats.sample_count++;
            if (cost->value >= static_cast<double>(in.obstacles->occupied_threshold)) {
                stats.blocked_sample_count++;
            }
        }
    }
    return stats;
}

std::optional<ReplanReason> check_step_blocked(
    const RouteMonitorInput& in,
    const MincoTrajectory& path,
    rclcpp::Logger logger
) {
    const auto& p = in.step_block;
    if (!p.enable) return std::nullopt;

    if (!in.obstacles) return std::nullopt;
    if (in.obstacles->terrain_dynamic_timeline.empty()) return std::nullopt;
    const bool using_predicted = in.obstacles->terrain_dynamic_timeline.size() > 1
        && in.obstacles->prediction_dt > 0.0
        && !in.active_path->speed_profile.empty();

    const auto stats = sample_block_stats(in, path, using_predicted);
    if (!stats || stats->sample_count == 0 || stats->blocked_sample_count == 0) {
        return std::nullopt;
    }

    RCLCPP_WARN(
        logger,
        "RouteMonitor: %s terrain access ahead within %.2f m "
        "(blocked_samples=%d/%d)",
        using_predicted ? "predicted blocked" : "blocked",
        p.lookahead_distance,
        stats->blocked_sample_count,
        stats->sample_count
    );
    return using_predicted
        ? ReplanReason::STEP_BLOCKED_PREDICTED
        : ReplanReason::STEP_BLOCKED_CURRENT;
}

std::optional<ReplanReason> check_performance(const RouteMonitorInput& in, rclcpp::Logger logger) {
    const AnnotatedPath& path = *in.active_path;
    if (!path.planning_performance.high_performance && in.current_performance.high_performance) {
        RCLCPP_INFO(logger, "RouteMonitor: performance recovered; upgrading low-performance route");
        return ReplanReason::PERFORMANCE_RECOVERED;
    }
    if (!path.planning_performance.high_performance || in.current_performance.high_performance) return std::nullopt;

    const double lookahead_progress = std::min(
        path.trajectory.total_arc_length(),
        in.route.arc_length + in.performance.lookahead_distance
    );
    for (const StepPlanSegment& segment : path.step_segments) {
        if (!segment.requires_high_performance) continue;
        if (segment.prepare_arc_length <= in.route.arc_length
            || segment.prepare_arc_length > lookahead_progress) continue;
        RCLCPP_WARN(logger, "RouteMonitor: high-performance crossing ahead is no longer available");
        return ReplanReason::PERFORMANCE_DEGRADED;
    }
    return std::nullopt;
}

} // anonymous namespace

RouteMonitorReport run_route_monitor(const RouteMonitorInput& input, rclcpp::Logger logger) {
    RouteMonitorReport report;

    if (!input.active_path) return report;
    const MincoTrajectory& path = input.active_path->trajectory;

    // MPC_LETHAL 优先（path invalid，已在运动层减速）
    if (input.mpc_lethal) {
        report.needs_replan = true;
        report.reason = ReplanReason::MPC_LETHAL;
        return report;
    }

    if (const auto reason = check_projection_guard(input, logger)) {
        report.needs_replan = true;
        report.reason = *reason;
        return report;
    }

    if (const auto reason = check_step_blocked(input, path, logger)) {
        report.needs_replan = true;
        report.reason = *reason;
        return report;
    }

    if (const auto performance_reason = check_performance(input, logger)) {
        report.needs_replan = true;
        report.reason = *performance_reason;
        return report;
    }

    return report;
}

} // namespace nav_executor
