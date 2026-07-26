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

// 沿当前路径向前的空间采样栅格，两种采样模式共用。
struct BlockSampleGrid {
    int count = 1;
    double base_progress = 0.0;
    double span = 0.0;
    double total_arc_length = 0.0;

    [[nodiscard]] double progress_at(const int index) const {
        const double fraction = count <= 1
            ? 0.0
            : static_cast<double>(index) / static_cast<double>(count - 1);
        return std::min(total_arc_length, base_progress + span * fraction);
    }
};

BlockSampleGrid build_block_sample_grid(
    const RouteMonitorInput& in,
    const MincoTrajectory& path,
    const double resolution
) {
    BlockSampleGrid grid;
    grid.total_arc_length = path.total_arc_length();
    grid.base_progress = std::clamp(in.route.arc_length, 0.0, grid.total_arc_length);
    grid.span = std::max(0.0, in.step_block.lookahead_distance);
    grid.count = resolution > 0.0
        ? std::max(1, static_cast<int>(std::ceil(grid.span / resolution)) + 1)
        : 1;
    return grid;
}

// per_step_dynamic_cost_maps[i] 是 t = (i+1) * prediction_dt 时刻的预测帧。取时间上
// 最近的一帧；超出预测时域的样本没有判定依据，返回空。
std::optional<size_t> prediction_frame_index(
    const size_t frame_count,
    const double prediction_dt,
    const double arrival_time
) {
    const long long nearest = std::llround(arrival_time / prediction_dt) - 1;
    if (nearest >= static_cast<long long>(frame_count)) return std::nullopt;
    return static_cast<size_t>(std::max<long long>(0, nearest));
}

// 台阶样本按空间步进枚举，读取哪一预测帧由计划到达时刻决定，因此采样间距与预测
// 步长互不耦合。超出预测时域的台阶不计入分母。
std::optional<BlockSampleStats> sample_predicted_block_stats(
    const RouteMonitorInput& in,
    const MincoTrajectory& path
) {
    // 台阶区间由 step_segments 的弧长边界给出，与格点无关，故采样间距不受地图分辨率约束。
    const BlockSampleGrid grid = build_block_sample_grid(
        in, path, in.step_block.sample_resolution
    );
    const PathSpeedProfile& speed_profile = in.active_path->speed_profile;
    const double route_time = speed_profile.eval_arc_length(grid.base_progress).time;

    BlockSampleStats stats;
    double previous_progress = grid.base_progress;
    for (int i = 0; i < grid.count; ++i) {
        const double progress = grid.progress_at(i);
        for (const StepPlanSegment& segment : in.active_path->step_segments) {
            const double overlap_begin = std::max(
                previous_progress, segment.step_enter_arc_length
            );
            const double overlap_end = std::min(progress, segment.step_exit_arc_length);
            if (overlap_begin > overlap_end) continue;

            const double sample_progress = 0.5 * (overlap_begin + overlap_end);
            const double arrival_time = std::max(
                0.0, speed_profile.eval_arc_length(sample_progress).time - route_time
            );
            const auto frame_index = prediction_frame_index(
                in.per_step_dynamic_cost_maps.size(), in.prediction_dt, arrival_time
            );
            if (!frame_index) continue;
            const CostMap* const cost_map = in.per_step_dynamic_cost_maps[*frame_index];
            if (!cost_map) return std::nullopt;

            const Eigen::Vector2d pos = path.eval_arc_length(sample_progress).p;
            const Eigen::Vector2d cost_grid = cost_map->map_coord_to_grid(pos);
            if (!cost_map->is_valid_coord(cost_grid)) return std::nullopt;

            stats.step_sample_count++;
            if (cost_map->interpolate(cost_grid) >= in.step_block.obstacle_cost_threshold) {
                stats.blocked_step_sample_count++;
            }
        }
        previous_progress = progress;
    }
    return stats;
}

// 无预测序列时退化为当前帧判定：样本取方向场的台阶物理本体格。
// 前置条件：base_direction_map 非空。
std::optional<BlockSampleStats> sample_current_block_stats(
    const RouteMonitorInput& in,
    const MincoTrajectory& path
) {
    const DirectionMap& direction_map = *in.base_direction_map;
    // 本体判定读取原始格点，采样间距必须细于半个格宽，否则会漏过整格台阶。
    const BlockSampleGrid grid = build_block_sample_grid(
        in, path, std::min(in.step_block.sample_resolution, direction_map.resolution * 0.5)
    );

    BlockSampleStats stats;
    for (int i = 0; i < grid.count; ++i) {
        const Eigen::Vector2d pos = path.eval_arc_length(grid.progress_at(i)).p;
        const Eigen::Vector2d dir_grid = direction_map.map_coord_to_grid(pos);
        if (!direction_map.is_valid_coord(dir_grid)) return std::nullopt;
        if (!direction_map.is_terrain_body_at(dir_grid)) continue;

        bool blocked_dynamic = false;
        if (in.current_dynamic_cost_map) {
            const Eigen::Vector2d cost_grid = in.current_dynamic_cost_map->map_coord_to_grid(pos);
            if (!in.current_dynamic_cost_map->is_valid_coord(cost_grid)) return std::nullopt;
            blocked_dynamic = in.current_dynamic_cost_map->interpolate(cost_grid)
                >= in.step_block.obstacle_cost_threshold;
        }

        stats.step_sample_count++;
        if (blocked_dynamic) stats.blocked_step_sample_count++;
    }
    return stats;
}

bool check_step_blocked(const RouteMonitorInput& in, const MincoTrajectory& path, rclcpp::Logger logger) {
    const auto& p = in.step_block;
    if (!p.enable) return false;

    // 预测模式需要完整的预测帧序列与计划速度剖面：帧索引由计划到达时刻推出。
    const bool using_predicted = !in.per_step_dynamic_cost_maps.empty()
        && in.prediction_dt > 0.0
        && !in.active_path->speed_profile.empty();
    if (!using_predicted && !in.base_direction_map) return false;

    const auto stats = using_predicted
        ? sample_predicted_block_stats(in, path)
        : sample_current_block_stats(in, path);
    if (!stats || stats->step_sample_count == 0) return false;

    if (!using_predicted) {
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
