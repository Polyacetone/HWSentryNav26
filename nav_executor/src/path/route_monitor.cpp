#include <nav_executor/path/route_monitor.hpp>
#include <nav_executor/executor/recovery_helpers.hpp>

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
    }
    return "NONE";
}

namespace {

// ── projection guard（迁自旧 MainController::check_follow_projection_guard）──
bool check_projection_guard(const RouteMonitorInput& in, const SplinePath& path, rclcpp::Logger logger) {
    const Eigen::Vector2d pos_map = in.chassis_pos_map;
    const Eigen::Vector2d proj_map = path.position(in.current_u);
    const double proj_dist = (proj_map - pos_map).norm();

    if (in.proj_guard.dist_max > 0.0 && proj_dist > in.proj_guard.dist_max) {
        RCLCPP_WARN(logger, "RouteMonitor: projection too far (dist=%.2f m > %.2f m)", proj_dist, in.proj_guard.dist_max);
        return true;
    }

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

// ── step blocked（迁自旧 StepController::check_block_replan）──
struct BlockSampleStats {
    int sample_count = 0;
    int step_sample_count = 0;
    int blocked_step_sample_count = 0;
};

std::optional<BlockSampleStats> sample_block_stats(const RouteMonitorInput& in, const SplinePath& path) {
    const auto& p = in.step_block;
    const double lookahead_distance = std::max(0.0, p.lookahead_distance);
    const double resolution = std::max(1e-3, p.sample_resolution);
    const bool using_predicted = !in.per_step_dynamic_cost_maps.empty();
    const DirectionMap& direction_map = *in.masked_direction_map;

    BlockSampleStats stats;
    const int samples = using_predicted
        ? static_cast<int>(in.per_step_dynamic_cost_maps.size())
        : std::max(1, static_cast<int>(std::ceil(lookahead_distance / resolution)) + 1);
    if (samples <= 0) return stats;

    auto advance_path_u = [&](const double distance) {
        double u = std::clamp(in.current_u, 0.0, 1.0);
        double travelled = 0.0;
        while (u < 1.0 && travelled < distance) {
            const Eigen::Vector2d d1 = path.tangent(u);
            const double speed = d1.norm();
            if (speed < 1e-12) {
                u = std::min(1.0, u + 1e-3);
                continue;
            }
            const double du = resolution / speed;
            const double next_u = std::min(1.0, u + du);
            travelled += (path.position(next_u) - path.position(u)).norm();
            u = next_u;
        }
        return u;
    };

    for (int i = 0; i < samples; ++i) {
        const double t = samples == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(samples - 1);
        const double u = advance_path_u(lookahead_distance * t);

        const Eigen::Vector2d pos = path.position(u);
        const Eigen::Vector2d dir_grid = direction_map.map_coord_to_grid(pos);
        if (!direction_map.is_valid_coord(dir_grid)) return std::nullopt;

        const double step_norm = direction_map.interpolate(dir_grid).norm();
        const bool on_step = step_norm >= p.step_norm_threshold;

        bool blocked_dynamic = false;
        if (using_predicted) {
            const CostMap* const cost_map = in.per_step_dynamic_cost_maps[static_cast<size_t>(i)];
            if (!cost_map) return std::nullopt;
            const Eigen::Vector2d cost_grid = cost_map->map_coord_to_grid(pos);
            if (!cost_map->is_valid_coord(cost_grid)) return std::nullopt;
            blocked_dynamic = cost_map->interpolate(cost_grid) >= p.obstacle_cost_threshold;
        } else if (in.current_dynamic_cost_map) {
            const Eigen::Vector2d cost_grid = in.current_dynamic_cost_map->map_coord_to_grid(pos);
            if (!in.current_dynamic_cost_map->is_valid_coord(cost_grid)) return std::nullopt;
            blocked_dynamic = in.current_dynamic_cost_map->interpolate(cost_grid) >= p.obstacle_cost_threshold;
        }

        stats.sample_count++;
        if (!on_step) continue;
        stats.step_sample_count++;
        if (blocked_dynamic) stats.blocked_step_sample_count++;
    }

    return stats;
}

bool check_step_blocked(const RouteMonitorInput& in, const SplinePath& path, rclcpp::Logger logger) {
    const auto& p = in.step_block;
    if (!p.enable || !in.masked_direction_map) return false;

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

} // anonymous namespace

RouteMonitorReport run_route_monitor(const RouteMonitorInput& input, rclcpp::Logger logger) {
    RouteMonitorReport report;
    report.current_u = input.current_u;

    if (!input.active_path) return report;
    const SplinePath& path = input.active_path->spline;

    // MPC_LETHAL 优先（path invalid，已在运动层减速）
    if (input.mpc_lethal) {
        report.needs_replan = true;
        report.reason = ReplanReason::MPC_LETHAL;
        return report;
    }

    if (check_projection_guard(input, path, logger)) {
        report.needs_replan = true;
        report.reason = ReplanReason::PROJECTION_GUARD;
        return report;
    }

    if (check_step_blocked(input, path, logger)) {
        report.needs_replan = true;
        report.reason = ReplanReason::STEP_BLOCKED;
        return report;
    }

    return report;
}

} // namespace nav_executor
