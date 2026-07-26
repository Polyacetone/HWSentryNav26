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
        case ReplanReason::PROJECTION_GUARD: return "PROJECTION_GUARD";
        case ReplanReason::STEP_BLOCKED: return "STEP_BLOCKED";
        case ReplanReason::MPC_LETHAL: return "MPC_LETHAL";
        case ReplanReason::EXECUTOR_REPLAN_EVENT: return "EXECUTOR_REPLAN_EVENT";
        case ReplanReason::PERFORMANCE_DEGRADED: return "PERFORMANCE_DEGRADED";
        case ReplanReason::PERFORMANCE_RECOVERED: return "PERFORMANCE_RECOVERED";
    }
    return "NONE";
}

namespace {

// 投影保护
bool check_projection_guard(const RouteMonitorInput& in, rclcpp::Logger logger) {
    const Eigen::Vector2d pos_map = in.chassis_pos_map;
    if (in.route.status != RouteTrackingStatus::TRACKED || in.route.path != in.active_path) {
        RCLCPP_WARN(logger, "RouteMonitor: route tracking lost (tracking_error=%.2f m)", in.route.tracking_error);
        return true;
    }
    const Eigen::Vector2d proj_map = in.route.projected_position;

    if (!in.masked_global_cost_map || in.proj_guard.cost_max < 0.0 || in.proj_guard.cost_max >= 255.0) {
        return false;
    }

    const auto max_cost = recovery_helpers::max_cost_along_segment(
        *in.masked_global_cost_map, pos_map, proj_map, in.proj_guard.cost_samples
    );
    if (!max_cost) {
        RCLCPP_WARN(logger, "RouteMonitor: projection segment out of masked_global_cost_map bounds");
        return true;
    }
    if (*max_cost > in.proj_guard.cost_max) {
        RCLCPP_WARN(logger, "RouteMonitor: projection segment cost too high (max_cost=%.1f > %.1f)", *max_cost, in.proj_guard.cost_max);
        return true;
    }
    return false;
}

// 台阶阻塞
struct BlockSampleStats {
    int step_sample_count = 0;
    int blocked_step_sample_count = 0;
};

std::optional<BlockSampleStats> sample_block_stats(const RouteMonitorInput& in, const MincoTrajectory& path) {
    const auto& p = in.step_block;
    const double lookahead_distance = std::max(0.0, p.lookahead_distance);
    const bool using_predicted = !in.per_step_dynamic_cost_maps.empty();
    const double map_limited_resolution = in.base_direction_map
        ? std::min(p.sample_resolution, in.base_direction_map->resolution * 0.5)
        : p.sample_resolution;
    const double resolution = map_limited_resolution;

    BlockSampleStats stats;
    const int samples = using_predicted
        ? static_cast<int>(in.per_step_dynamic_cost_maps.size())
        : std::max(1, static_cast<int>(std::ceil(lookahead_distance / resolution)) + 1);
    if (samples <= 0) return stats;

    double previous_progress = std::clamp(
        in.route.arc_length, 0.0, path.total_arc_length()
    );
    for (int i = 0; i < samples; ++i) {
        const double t = samples == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(samples - 1);
        const double progress = std::min(
            path.total_arc_length(), in.route.arc_length + lookahead_distance * t
        );

        if (using_predicted) {
            const CostMap* const cost_map =
                in.per_step_dynamic_cost_maps[static_cast<size_t>(i)];
            if (!cost_map) return std::nullopt;
            for (const StepPlanSegment& segment : in.active_path->step_segments) {
                const double overlap_begin = std::max(
                    previous_progress, segment.step_enter_arc_length
                );
                const double overlap_end = std::min(
                    progress, segment.step_exit_arc_length
                );
                if (overlap_begin > overlap_end) continue;

                const double field_sample_progress = 0.5 * (overlap_begin + overlap_end);
                const Eigen::Vector2d pos = path.eval_arc_length(field_sample_progress).p;
                const Eigen::Vector2d cost_grid = cost_map->map_coord_to_grid(pos);
                if (!cost_map->is_valid_coord(cost_grid)) return std::nullopt;
                const bool blocked_dynamic =
                    cost_map->interpolate(cost_grid) >= p.obstacle_cost_threshold;
                stats.step_sample_count++;
                if (blocked_dynamic) stats.blocked_step_sample_count++;
            }
            previous_progress = progress;
            continue;
        }

        previous_progress = progress;
        if (!in.base_direction_map) return std::nullopt;
        const Eigen::Vector2d pos = path.eval_arc_length(progress).p;
        const Eigen::Vector2d dir_grid =
            in.base_direction_map->map_coord_to_grid(pos);
        if (!in.base_direction_map->is_valid_coord(dir_grid)) return std::nullopt;
        if (!in.base_direction_map->is_terrain_body_at(dir_grid)) continue;

        bool blocked_dynamic = false;
        if (in.current_dynamic_cost_map) {
            const Eigen::Vector2d cost_grid = in.current_dynamic_cost_map->map_coord_to_grid(pos);
            if (!in.current_dynamic_cost_map->is_valid_coord(cost_grid)) return std::nullopt;
            blocked_dynamic = in.current_dynamic_cost_map->interpolate(cost_grid) >= p.obstacle_cost_threshold;
        }

        stats.step_sample_count++;
        if (blocked_dynamic) stats.blocked_step_sample_count++;
    }

    return stats;
}

bool check_step_blocked(const RouteMonitorInput& in, const MincoTrajectory& path, rclcpp::Logger logger) {
    const auto& p = in.step_block;
    if (!p.enable) return false;
    if (in.per_step_dynamic_cost_maps.empty() && !in.base_direction_map) return false;

    const auto stats = sample_block_stats(in, path);
    if (!stats || stats->step_sample_count == 0) return false;

    if (in.per_step_dynamic_cost_maps.empty()) {
        if (stats->blocked_step_sample_count > 0) {
            RCLCPP_WARN(
                logger, "RouteMonitor: blocked step ahead within %.2f m (blocked_step_samples=%d/%d)",
                p.lookahead_distance, stats->blocked_step_sample_count, stats->step_sample_count
            );
            return true;
        }
        return false;
    }

    const double blocked_ratio = static_cast<double>(stats->blocked_step_sample_count) / static_cast<double>(stats->step_sample_count);
    if (blocked_ratio >= p.predicted_obstacle_ratio_threshold) {
        RCLCPP_WARN(
            logger, "RouteMonitor: predicted blocked step ahead within %.2f m (ratio=%.2f >= %.2f)",
            p.lookahead_distance, blocked_ratio, p.predicted_obstacle_ratio_threshold
        );
        return true;
    }
    return false;
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

    if (check_projection_guard(input, logger)) {
        report.needs_replan = true;
        report.reason = ReplanReason::PROJECTION_GUARD;
        return report;
    }

    if (check_step_blocked(input, path, logger)) {
        report.needs_replan = true;
        report.reason = ReplanReason::STEP_BLOCKED;
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
