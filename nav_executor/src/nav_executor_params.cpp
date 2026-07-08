#include <nav_executor/nav_executor_node.hpp>

#include <string>

namespace nav_executor {

// ═══════════════════════ 地形配置 ════════════════════════════

CapabilityProfile NavExecutorNode::load_capability_profile(const std::string& prefix) {
    return CapabilityProfile {
        .command_bounds = {
            .vel_max = declare_parameter<double>(prefix + ".command_bounds.vel_max"),
            .vel_min = declare_parameter<double>(prefix + ".command_bounds.vel_min"),
            .omega_max = declare_parameter<double>(prefix + ".command_bounds.omega_max"),
            .omega_min = declare_parameter<double>(prefix + ".command_bounds.omega_min"),
        },
        .motion_constraints = {
            .acc_max = declare_parameter<double>(prefix + ".motion_constraints.acc_max"),
            .alpha_max = declare_parameter<double>(prefix + ".motion_constraints.alpha_max"),
            .a_lat_max = declare_parameter<double>(prefix + ".motion_constraints.a_lat_max"),
        },
    };
}

void NavExecutorNode::load_terrain_config() {
    using enum CapabilityLevel;
    terrain_profiles_.capability_profiles[static_cast<size_t>(LOW)] = load_capability_profile("terrain_profiles.capability_profiles.low");
    terrain_profiles_.capability_profiles[static_cast<size_t>(MEDIUM)] = load_capability_profile("terrain_profiles.capability_profiles.medium");
    terrain_profiles_.capability_profiles[static_cast<size_t>(HIGH)] = load_capability_profile("terrain_profiles.capability_profiles.high");

    // directional_labels — 有方向语义的标签 (SLOPE..STEP_HIGH)
    struct DirEntry { const char* name; uint8_t label; };
    const DirEntry dir_entries[] = {
        {"slope", 2}, {"step_l1", 3}, {"step_l2", 4}, {"fly_slope", 5}, {"step_high", 6},
    };
    for (const auto& [name, label] : dir_entries) {
        for (const auto& dir : {"up", "down"}) {
            const auto prefix = std::string("terrain_profiles.directional_labels.") + name + "." + dir;
            const size_t idx = label - 2;
            auto& rule = (dir == std::string("up"))
                ? terrain_profiles_.directional_labels[idx].up
                : terrain_profiles_.directional_labels[idx].down;
            rule.chassis_mode = static_cast<uint8_t>(declare_parameter<int>(prefix + ".chassis_mode"));
            rule.capability = capability_level_from_string(declare_parameter<std::string>(prefix + ".capability"));
            rule.speed.min = declare_parameter<double>(prefix + ".speed.min");
            rule.speed.max = declare_parameter<double>(prefix + ".speed.max");
        }
    }

    // 通行性规则（planner 可行性判定）
    const auto load_rule = [&](const std::string& prefix, uint8_t label) {
        terrain_rules_[label] = {
            declare_parameter<bool>(prefix + ".forward"),
            declare_parameter<bool>(prefix + ".backward")
        };
    };
    load_rule("traversability.terrain_rules.slope", static_cast<uint8_t>(TerrainType::SLOPE));
    load_rule("traversability.terrain_rules.step_l1", static_cast<uint8_t>(TerrainType::STEP_L1));
    load_rule("traversability.terrain_rules.step_l2", static_cast<uint8_t>(TerrainType::STEP_L2));
    load_rule("traversability.terrain_rules.fly_slope", static_cast<uint8_t>(TerrainType::FLY_SLOPE));
    load_rule("traversability.terrain_rules.step_high", static_cast<uint8_t>(TerrainType::STEP_HIGH));
}

ProfileBlendParams NavExecutorNode::load_blend_params() {
    return {
        .v_step = declare_parameter<double>("terrain_profiles.profile_blend.v_step"),
        .w_step = declare_parameter<double>("terrain_profiles.profile_blend.w_step"),
        .acc_step = declare_parameter<double>("terrain_profiles.profile_blend.acc_step"),
        .alpha_step = declare_parameter<double>("terrain_profiles.profile_blend.alpha_step"),
        .a_lat_step = declare_parameter<double>("terrain_profiles.profile_blend.a_lat_step"),
    };
}

// ═══════════════════════ FSM 参数 ════════════════════════════

FsmParams NavExecutorNode::load_fsm_params() {
    FsmParams fsm;
    fsm.transition = {
        .follow_to_spin_vel_max = declare_parameter<double>("state_machine.follow_to_spin_vel_max"),
        .spin_to_follow_omega_max = declare_parameter<double>("state_machine.spin_to_follow_omega_max"),
        .to_idle_vel_max = declare_parameter<double>("state_machine.to_idle_vel_max"),
        .to_idle_omega_max = declare_parameter<double>("state_machine.to_idle_omega_max"),
        .stopping_timeout = declare_parameter<double>("state_machine.stopping_timeout"),
    };
    fsm.recovery = {
        .hazard_cost_threshold = declare_parameter<double>("recovery.hazard.cost_threshold"),
        .hazard_step_norm_threshold = declare_parameter<double>("recovery.hazard.step_norm_threshold"),
        .safe_cost_threshold = declare_parameter<double>("recovery.safe.cost_threshold"),
        .safe_step_norm_threshold = declare_parameter<double>("recovery.safe.step_norm_threshold"),
        .recovery_cost_threshold = declare_parameter<double>("recovery.search.recovery_cost_threshold"),
        .radius_min = declare_parameter<double>("recovery.search.radius_min"),
        .radius_max = declare_parameter<double>("recovery.search.radius_max"),
        .radius_samples = static_cast<int>(declare_parameter<int>("recovery.search.radius_samples")),
        .angle_samples = static_cast<int>(declare_parameter<int>("recovery.search.angle_samples")),
        .path_integral_resolution = declare_parameter<double>("recovery.search.path_integral_resolution"),
        .path_integral_cost_weight = declare_parameter<double>("recovery.search.path_integral_cost_weight"),
        .path_integral_step_weight = declare_parameter<double>("recovery.search.path_integral_step_weight"),
        .step_ascent_penalty_weight = declare_parameter<double>("recovery.search.step_ascent_penalty_weight"),
        .step_ascent_penalty_norm_threshold = declare_parameter<double>("recovery.search.step_ascent_penalty_norm_threshold"),
        .step_ascent_penalty_dot_threshold = declare_parameter<double>("recovery.search.step_ascent_penalty_dot_threshold"),
        .safe_hold_time = declare_parameter<double>("recovery.exit.safe_hold_time"),
        .goal_timeout = declare_parameter<double>("recovery.search.goal_timeout"),
    };
    fsm.stuck = {
        .cmd_vel_threshold = declare_parameter<double>("recovery.stuck.cmd_vel_threshold"),
        .timeout = declare_parameter<double>("recovery.stuck.timeout"),
        .max_displacement = declare_parameter<double>("recovery.stuck.max_displacement"),
        .reverse_speed = declare_parameter<double>("recovery.stuck.reverse_speed"),
        .reverse_displacement = declare_parameter<double>("recovery.stuck.reverse_displacement"),
        .reverse_timeout = declare_parameter<double>("recovery.stuck.reverse_timeout"),
    };
    return fsm;
}

// ═══════════════════════ PathExecutor 参数 ═══════════════════

PathExecutorParams NavExecutorNode::load_executor_params() {
    PathExecutorParams p;
    p.stop_threshold_dist = declare_parameter<double>("misc.stop_threshold_dist");
    p.stop_threshold_u = declare_parameter<double>("misc.stop_threshold_u");
    p.follow_no_progress_guard = {
        .landmark_spacing = declare_parameter<double>("no_progress_guard.follow.landmark_spacing"),
        .timeout = declare_parameter<double>("no_progress_guard.follow.timeout"),
    };
    p.stepping_no_progress_guard = {
        .landmark_spacing = declare_parameter<double>("no_progress_guard.stepping.landmark_spacing"),
        .timeout = declare_parameter<double>("no_progress_guard.stepping.timeout"),
    };
    p.step_dist_offset = declare_parameter<double>("step.step_dist_offset");
    return p;
}

// ═══════════════════════ Planner 参数 ════════════════════════

PlannerConfig NavExecutorNode::load_planner_config() {
    PlannerConfig c;
    c.occupied_threshold = static_cast<int>(declare_parameter<int>("traversability.occupied_threshold"));
    c.on_step_threshold = declare_parameter<double>("traversability.on_step_threshold");
    c.start_prediction_enable = declare_parameter<bool>("start_prediction.enable");
    c.start_prediction_max_accel = declare_parameter<double>("start_prediction.max_accel");
    c.start_prediction_planning_delay = declare_parameter<double>("start_prediction.planning_delay");
    c.start_prediction_min_speed = declare_parameter<double>("start_prediction.min_speed");
    c.start_prediction_collision_check_step = declare_parameter<double>("start_prediction.collision_check_step");
    c.nudge_max_distance = declare_parameter<double>("nudge.max_distance");
    c.goal_reached_distance = declare_parameter<double>("planner.goal_reached_distance");
    c.skip_distance = declare_parameter<double>("planner.skip_distance");
    c.step_detection = {
        .detect_norm_threshold = declare_parameter<double>("step.detection.detect_norm_threshold"),
        .detect_dot_threshold = declare_parameter<double>("step.detection.detect_dot_threshold"),
        .path_sample_resolution = declare_parameter<double>("step.detection.path_sample_resolution"),
        .prepare_distance = declare_parameter<double>("step.detection.prepare_distance"),
        .active_distance = declare_parameter<double>("step.detection.active_distance"),
        .release_distance = declare_parameter<double>("step.detection.release_distance"),
    };
    c.enable_debug = enable_debug_;
    return c;
}

BSplineOptimizer::Params NavExecutorNode::load_optimizer_params() {
    return {
        .step_norm_threshold = declare_parameter<double>("path_optimizer.step_norm_threshold"),
        .step_norm_transition = declare_parameter<double>("path_optimizer.step_norm_transition"),
        .step_detection_samples_per_meter = declare_parameter<double>("path_optimizer.step_detection_samples_per_meter"),
        .warmup = {
            .obstacle_weight = declare_parameter<double>("path_optimizer.warmup.obstacle_weight"),
            .direction_weight = declare_parameter<double>("path_optimizer.warmup.direction_weight"),
            .step_weight = declare_parameter<double>("path_optimizer.warmup.step_weight"),
            .start_end_weight = declare_parameter<double>("path_optimizer.warmup.start_end_weight"),
            .smoothness_weight = declare_parameter<double>("path_optimizer.warmup.smoothness_weight"),
            .samples_per_meter = declare_parameter<double>("path_optimizer.warmup.samples_per_meter"),
            .max_iterations = static_cast<int>(declare_parameter<int>("path_optimizer.warmup.max_iterations")),
            .max_curvature = declare_parameter<double>("path_optimizer.warmup.max_curvature"),
            .length_penalty_weight = declare_parameter<double>("path_optimizer.warmup.length_penalty_weight"),
            .curvature = {
                .base_weight = declare_parameter<double>("path_optimizer.warmup.curvature.base_weight"),
                .base_beta = declare_parameter<double>("path_optimizer.warmup.curvature.base_beta"),
                .limit_weight = declare_parameter<double>("path_optimizer.warmup.curvature.limit_weight"),
                .limit_beta = declare_parameter<double>("path_optimizer.warmup.curvature.limit_beta"),
                .min_speed_epsilon = declare_parameter<double>("path_optimizer.warmup.curvature.min_speed_epsilon"),
                .speed_gate_threshold = declare_parameter<double>("path_optimizer.warmup.curvature.speed_gate_threshold"),
            },
        },
        .main = {
            .obstacle_weight = declare_parameter<double>("path_optimizer.main.obstacle_weight"),
            .direction_weight = declare_parameter<double>("path_optimizer.main.direction_weight"),
            .step_weight = declare_parameter<double>("path_optimizer.main.step_weight"),
            .start_end_weight = declare_parameter<double>("path_optimizer.main.start_end_weight"),
            .smoothness_weight = declare_parameter<double>("path_optimizer.main.smoothness_weight"),
            .samples_per_meter = declare_parameter<double>("path_optimizer.main.samples_per_meter"),
            .max_iterations = static_cast<int>(declare_parameter<int>("path_optimizer.main.max_iterations")),
            .max_refinement_iterations = static_cast<int>(declare_parameter<int>("path_optimizer.main.max_refinement_iterations")),
            .near_max_curvature = declare_parameter<double>("path_optimizer.main.near_max_curvature"),
            .far_max_curvature = declare_parameter<double>("path_optimizer.main.far_max_curvature"),
            .step_extension_distance = declare_parameter<double>("path_optimizer.main.step_extension_distance"),
            .step_transition_distance = declare_parameter<double>("path_optimizer.main.step_transition_distance"),
            .interval_iou_threshold = declare_parameter<double>("path_optimizer.main.interval_iou_threshold"),
            .length_penalty_weight = declare_parameter<double>("path_optimizer.main.length_penalty_weight"),
            .curvature = {
                .base_weight = declare_parameter<double>("path_optimizer.main.curvature.base_weight"),
                .base_beta = declare_parameter<double>("path_optimizer.main.curvature.base_beta"),
                .limit_weight = declare_parameter<double>("path_optimizer.main.curvature.limit_weight"),
                .limit_beta = declare_parameter<double>("path_optimizer.main.curvature.limit_beta"),
                .min_speed_epsilon = declare_parameter<double>("path_optimizer.main.curvature.min_speed_epsilon"),
                .speed_gate_threshold = declare_parameter<double>("path_optimizer.main.curvature.speed_gate_threshold"),
            },
        },
    };
}

// ═══════════════════════ TaskExecutor 参数 ═══════════════════

TaskExecutorParams NavExecutorNode::load_task_params() {
    return {
        .goal_equivalence_distance = declare_parameter<double>("task.goal_equivalence_distance"),
        .plan_cooldown = declare_parameter<double>("task.plan_cooldown"),
    };
}

} // namespace nav_executor
