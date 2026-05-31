#include <path_follower/safety_monitor.hpp>
#include <path_follower/utils.hpp>
#include <rclcpp/logging.hpp>

namespace path_follower {

SafetyMonitor::SafetyMonitor(const FsmParams& fsm_params, rclcpp::Logger logger)
    : fsm_params_(fsm_params), logger_(logger) {}

// ═══════════════════════ 卡住检测 ═══════════════════════

bool SafetyMonitor::check_stuck(
    const Eigen::Vector2d& chassis_pos,
    const double last_cmd_vel,
    const std::chrono::steady_clock::time_point stamp
) {
    const auto& p = fsm_params_.stuck;
    if (std::abs(last_cmd_vel) < p.cmd_vel_threshold) {
        stuck_active_ = false;
        return false;
    }

    if (!stuck_active_) {
        stuck_active_ = true;
        stuck_start_time_ = stamp;
        stuck_start_pos_ = chassis_pos;
        return false;
    }

    const double dt = std::chrono::duration<double>(stamp - stuck_start_time_).count();
    const double disp = (chassis_pos - stuck_start_pos_).norm();
    if (disp > p.max_displacement) {
        stuck_start_time_ = stamp;
        stuck_start_pos_ = chassis_pos;
        return false;
    }

    return dt >= p.timeout;
}

void SafetyMonitor::reset_stuck() {
    stuck_active_ = false;
}

// ═══════════════════════ 危险判断 ═══════════════════════

bool SafetyMonitor::compute_is_hazard(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const Eigen::Vector2d& pos
) const {
    const auto& p = fsm_params_.recovery;

    const auto sample = recovery_helpers::sample_fields(
        cost_map,
        direction_map,
        pos
    );
    if (!sample) return false;

    return (sample->cost >= p.hazard_cost_threshold) ||
        (sample->step_norm >= p.hazard_step_norm_threshold);
}

// ═══════════════════════ 恢复安全判断 ═══════════════════════

bool SafetyMonitor::update_recovery_safe_flag(
    const CostMap& final_cost_map,
    const DirectionMap& direction_map,
    const Eigen::Vector2d& pos,
    const std::chrono::steady_clock::time_point stamp
) {
    const auto& p = fsm_params_.recovery;
    if (!recovery_goal_map_) {
        recovery_safe_since_ = std::nullopt;
        return false;
    }

    const auto sample = recovery_helpers::sample_fields(
        final_cost_map,
        direction_map,
        pos
    );
    if (!sample) {
        recovery_safe_since_ = std::nullopt;
        return false;
    }

    const bool safe_now = recovery_helpers::is_safe_goal(p, *sample);
    if (!safe_now) {
        recovery_safe_since_ = std::nullopt;
        return false;
    }

    if (!recovery_safe_since_) {
        recovery_safe_since_ = stamp;
    }
    return std::chrono::duration<double>(stamp - *recovery_safe_since_).count() >= p.safe_hold_time;
}

void SafetyMonitor::update_recovery_goal_if_needed(
    const Eigen::Vector3d& chassis_pose,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const DirectionMap* const base_dir_map,
    const std::chrono::steady_clock::time_point stamp
) {
    const auto& p = fsm_params_.recovery;

    const bool need_new = (!recovery_goal_map_) || (!recovery_goal_set_time_) ||
        (std::chrono::duration<double>(stamp - *recovery_goal_set_time_).count() >= p.goal_timeout);

    if (!need_new) return;

    recovery_goal_map_ = recovery_helpers::find_goal(
        p, cost_map, direction_map, chassis_pose,
        base_dir_map
    );
    recovery_goal_set_time_ = stamp;

    if (!recovery_goal_map_) {
        RCLCPP_ERROR(logger_, "HAZARD_RECOVERY failed to find a recovery goal");
        return;
    }

    const auto s = recovery_helpers::sample_fields(cost_map, direction_map, *recovery_goal_map_);
    if (!s) {
        RCLCPP_WARN(logger_, "HAZARD_RECOVERY new goal=(%.2f, %.2f) (field sample invalid)",
            recovery_goal_map_->x(), recovery_goal_map_->y());
        return;
    }

    if (recovery_helpers::is_safe_goal(p, *s)) {
        RCLCPP_WARN(logger_, "HAZARD_RECOVERY new goal=(%.2f, %.2f) (SAFE cost=%.1f step=%.3f)",
            recovery_goal_map_->x(), recovery_goal_map_->y(), s->cost, s->step_norm);
    } else {
        RCLCPP_WARN(logger_, "HAZARD_RECOVERY new goal=(%.2f, %.2f) (UNSAFE cost=%.1f step=%.3f)",
            recovery_goal_map_->x(), recovery_goal_map_->y(), s->cost, s->step_norm);
    }
}

void SafetyMonitor::reset_recovery() {
    recovery_goal_map_ = std::nullopt;
    recovery_goal_set_time_ = std::nullopt;
    recovery_safe_since_ = std::nullopt;
}

} // namespace path_follower
