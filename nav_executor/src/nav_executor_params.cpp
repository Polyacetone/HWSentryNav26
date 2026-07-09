#include <nav_executor/nav_executor_node.hpp>

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

// ═══════════════════════ MPC 参数 ═════════════════════════════

MPCParams NavExecutorNode::load_mpc_params() {
    using enum CapabilityLevel;
    MPCParams mpc_params = {
        .follow = {
            .start_command = {
                .vel_cmd_act_gap_max = declare_parameter<double>("mpc.follow.start_command.vel_cmd_act_gap_max"),
                .omega_cmd_act_gap_max = declare_parameter<double>("mpc.follow.start_command.omega_cmd_act_gap_max")
            },
            .normal_profile = {
                .command_bounds = {
                    .vel_max = declare_parameter<double>("mpc.follow.command_bounds.vel_max"),
                    .vel_min = declare_parameter<double>("mpc.follow.command_bounds.vel_min"),
                    .omega_max = declare_parameter<double>("mpc.follow.command_bounds.omega_max"),
                    .omega_min = declare_parameter<double>("mpc.follow.command_bounds.omega_min"),
                },
                .motion_constraints = {
                    .acc_max = declare_parameter<double>("mpc.follow.motion_constraints.acc_max"),
                    .alpha_max = declare_parameter<double>("mpc.follow.motion_constraints.alpha_max"),
                    .a_lat_max = declare_parameter<double>("mpc.follow.motion_constraints.a_lat_max"),
                },
            },
            .capability_profiles = {
                terrain_profiles_.capability_profiles[static_cast<size_t>(LOW)],
                terrain_profiles_.capability_profiles[static_cast<size_t>(MEDIUM)],
                terrain_profiles_.capability_profiles[static_cast<size_t>(HIGH)],
            },
            .tracking_weights = {
                .q_y = declare_parameter<double>("mpc.follow.tracking_weights.q_y"),
                .q_theta = declare_parameter<double>("mpc.follow.tracking_weights.q_theta"),
                .q_u = declare_parameter<double>("mpc.follow.tracking_weights.q_u")
            },
            .command_weights = {
                .r_v = declare_parameter<double>("mpc.follow.command_weights.r_v"),
                .r_omega = declare_parameter<double>("mpc.follow.command_weights.r_omega"),
                .r_dv = declare_parameter<double>("mpc.follow.command_weights.r_dv"),
                .r_domega = declare_parameter<double>("mpc.follow.command_weights.r_domega")
            },
            .motion_constraint_weights = {
                .acc_limit = declare_parameter<double>("mpc.follow.motion_constraint_weights.acc_limit"),
                .alpha_limit = declare_parameter<double>("mpc.follow.motion_constraint_weights.alpha_limit"),
                .lat_acc = declare_parameter<double>("mpc.follow.motion_constraint_weights.lat_acc")
            },
            .terrain_limits = {
                .step_reachability_guide_acc = declare_parameter<double>("mpc.follow.terrain_limits.step_reachability_guide_acc")
            },
            .terrain_weights = {
                .step_vel_weight = declare_parameter<double>("mpc.follow.terrain_weights.step_vel_weight"),
                .step_reachability_lo = declare_parameter<double>("mpc.follow.terrain_weights.step_reachability_lo"),
                .step_reachability_hi = declare_parameter<double>("mpc.follow.terrain_weights.step_reachability_hi"),
                .direction = declare_parameter<double>("mpc.follow.terrain_weights.direction")
            },
            .environment_weights = {
                .obstacle = declare_parameter<double>("mpc.follow.environment_weights.obstacle")
            },
            .terminal_weights = {
                .q_v_final = declare_parameter<double>("mpc.follow.terminal_weights.q_v_final"),
                .a_brake = declare_parameter<double>("mpc.follow.terminal_weights.a_brake"),
                .slow_down_target_vel = declare_parameter<double>("mpc.follow.terminal_weights.slow_down_target_vel")
            },
            .projection = {
                .proj_num_samples = static_cast<int>(declare_parameter<int>("mpc.follow.projection.num_samples")),
                .proj_search_window = declare_parameter<double>("mpc.follow.projection.search_window"),
                .local_search_lazy_distance = declare_parameter<double>("mpc.follow.projection.local_search_lazy_distance")
            },
            .mppi = {
                .enable = declare_parameter<bool>("mpc.follow.mppi.enable"),
                .num_threads = static_cast<int>(declare_parameter<int>("mpc.follow.mppi.num_threads")),
                .batch_size = static_cast<int>(declare_parameter<int>("mpc.follow.mppi.batch_size")),
                .iteration_count = static_cast<int>(declare_parameter<int>("mpc.follow.mppi.iteration_count")),
                .temperature = declare_parameter<double>("mpc.follow.mppi.temperature"),
                .gamma = declare_parameter<double>("mpc.follow.mppi.gamma"),
                .sampling_std = {
                    .velocity = declare_parameter<double>("mpc.follow.mppi.sampling_std.velocity"),
                    .omega = declare_parameter<double>("mpc.follow.mppi.sampling_std.omega")
                },
                .noise_smoothing = {
                    .window = static_cast<int>(declare_parameter<int>("mpc.follow.mppi.noise_smoothing.window")),
                    .passes = static_cast<int>(declare_parameter<int>("mpc.follow.mppi.noise_smoothing.passes"))
                },
                .include_nominal_trajectory = declare_parameter<bool>("mpc.follow.mppi.include_nominal_trajectory"),
                .fallback_to_best_sample = declare_parameter<bool>("mpc.follow.mppi.fallback_to_best_sample")
            },
            .rollout_safety = {
                .enable_lethal_obstacle_check = declare_parameter<bool>("mpc.follow.rollout_safety.enable_lethal_obstacle_check"),
                .lethal_obstacle_threshold = declare_parameter<double>("mpc.follow.rollout_safety.lethal_obstacle_threshold"),
                .fddp_lethal_consecutive_threshold = static_cast<int>(declare_parameter<int>("mpc.follow.rollout_safety.fddp_lethal_consecutive_threshold"))
            },
            .max_iters = static_cast<int>(declare_parameter<int>("mpc.follow.max_iters"))
        },
        .stop = {
            .command_bounds = {
                .vel_max = declare_parameter<double>("mpc.stop.command_bounds.vel_max"),
                .vel_min = declare_parameter<double>("mpc.stop.command_bounds.vel_min"),
                .omega_max = declare_parameter<double>("mpc.stop.command_bounds.omega_max"),
                .omega_min = declare_parameter<double>("mpc.stop.command_bounds.omega_min")
            },
            .start_command = {
                .vel_cmd_act_gap_max = declare_parameter<double>("mpc.stop.start_command.vel_cmd_act_gap_max"),
                .omega_cmd_act_gap_max = declare_parameter<double>("mpc.stop.start_command.omega_cmd_act_gap_max")
            },
            .motion_constraints = {
                .acc_max = declare_parameter<double>("mpc.stop.motion_constraints.acc_max"),
                .alpha_max = declare_parameter<double>("mpc.stop.motion_constraints.alpha_max"),
                .a_lat_max = declare_parameter<double>("mpc.stop.motion_constraints.a_lat_max")
            },
            .command_weights = {
                .r_v = declare_parameter<double>("mpc.stop.command_weights.r_v"),
                .r_omega = declare_parameter<double>("mpc.stop.command_weights.r_omega"),
                .r_dv = declare_parameter<double>("mpc.stop.command_weights.r_dv"),
                .r_domega = declare_parameter<double>("mpc.stop.command_weights.r_domega")
            },
            .motion_constraint_weights = {
                .acc_limit = declare_parameter<double>("mpc.stop.motion_constraint_weights.acc_limit"),
                .alpha_limit = declare_parameter<double>("mpc.stop.motion_constraint_weights.alpha_limit"),
                .lat_acc = declare_parameter<double>("mpc.stop.motion_constraint_weights.lat_acc")
            },
            .environment_weights = {
                .obstacle = declare_parameter<double>("mpc.stop.environment_weights.obstacle")
            },
            .terminal_weights = {
                .obstacle_terminal = declare_parameter<double>("mpc.stop.terminal_weights.obstacle_terminal")
            },
            .max_iters = static_cast<int>(declare_parameter<int>("mpc.stop.max_iters"))
        },
        .hold = {
            .command_bounds = {
                .vel_max = declare_parameter<double>("mpc.hold.command_bounds.vel_max"),
                .vel_min = declare_parameter<double>("mpc.hold.command_bounds.vel_min"),
                .omega_max = declare_parameter<double>("mpc.hold.command_bounds.omega_max"),
                .omega_min = declare_parameter<double>("mpc.hold.command_bounds.omega_min")
            },
            .start_command = {
                .vel_cmd_act_gap_max = declare_parameter<double>("mpc.hold.start_command.vel_cmd_act_gap_max"),
                .omega_cmd_act_gap_max = declare_parameter<double>("mpc.hold.start_command.omega_cmd_act_gap_max")
            },
            .motion_constraints = {
                .acc_max = declare_parameter<double>("mpc.hold.motion_constraints.acc_max"),
                .alpha_max = declare_parameter<double>("mpc.hold.motion_constraints.alpha_max"),
                .a_lat_max = declare_parameter<double>("mpc.hold.motion_constraints.a_lat_max")
            },
            .goal_weights = {
                .q_goal_xy = declare_parameter<double>("mpc.hold.goal_weights.q_goal_xy"),
                .q_goal_theta = declare_parameter<double>("mpc.hold.goal_weights.q_goal_theta"),
                .goal_deadzone = declare_parameter<double>("mpc.hold.goal_weights.goal_deadzone")
            },
            .command_weights = {
                .r_v = declare_parameter<double>("mpc.hold.command_weights.r_v"),
                .r_omega = declare_parameter<double>("mpc.hold.command_weights.r_omega"),
                .r_dv = declare_parameter<double>("mpc.hold.command_weights.r_dv"),
                .r_domega = declare_parameter<double>("mpc.hold.command_weights.r_domega")
            },
            .motion_constraint_weights = {
                .acc_limit = declare_parameter<double>("mpc.hold.motion_constraint_weights.acc_limit"),
                .alpha_limit = declare_parameter<double>("mpc.hold.motion_constraint_weights.alpha_limit"),
                .lat_acc = declare_parameter<double>("mpc.hold.motion_constraint_weights.lat_acc")
            },
            .environment_weights = {
                .obstacle = declare_parameter<double>("mpc.hold.environment_weights.obstacle")
            },
            .terminal_weights = {
                .q_goal_xy_terminal = declare_parameter<double>("mpc.hold.terminal_weights.q_goal_xy_terminal"),
                .obstacle_terminal = declare_parameter<double>("mpc.hold.terminal_weights.obstacle_terminal")
            },
            .max_iters = static_cast<int>(declare_parameter<int>("mpc.hold.max_iters"))
        },
        .energy = {
            .enable = declare_parameter<bool>("mpc.energy.enable"),
            .threshold = declare_parameter<double>("mpc.energy.threshold"),
            .weight = declare_parameter<double>("mpc.energy.weight")
        },
        .kinematic_model = {
            .z_ref = declare_parameter<double>("kinematic_model.z_ref"),
            .z_scale = declare_parameter<double>("kinematic_model.z_scale"),
            .rho_clip = declare_parameter<double>("kinematic_model.rho_clip"),
            .sgn_eps = declare_parameter<double>("kinematic_model.sgn_eps"),
            .ca00 = declare_parameter<double>("kinematic_model.ca00"),
            .ca01 = declare_parameter<double>("kinematic_model.ca01"),
            .ca10 = declare_parameter<double>("kinematic_model.ca10"),
            .ca11 = declare_parameter<double>("kinematic_model.ca11"),
            .cb0 = declare_parameter<double>("kinematic_model.cb0"),
            .cb1 = declare_parameter<double>("kinematic_model.cb1"),
            .dca00 = declare_parameter<double>("kinematic_model.dca00"),
            .dca01 = declare_parameter<double>("kinematic_model.dca01"),
            .dca10 = declare_parameter<double>("kinematic_model.dca10"),
            .dca11 = declare_parameter<double>("kinematic_model.dca11"),
            .dcb0 = declare_parameter<double>("kinematic_model.dcb0"),
            .dcb1 = declare_parameter<double>("kinematic_model.dcb1"),
            .gxh = declare_parameter<double>("kinematic_model.gxh"),
            .gv = declare_parameter<double>("kinematic_model.gv"),
            .cf1 = declare_parameter<double>("kinematic_model.cf1"),
            .cf2 = declare_parameter<double>("kinematic_model.cf2"),
            .w_lam0 = declare_parameter<double>("kinematic_model.w_lam0"),
            .w_k0 = declare_parameter<double>("kinematic_model.w_k0"),
            .w_cf0 = declare_parameter<double>("kinematic_model.w_cf0"),
            .w_lam1 = declare_parameter<double>("kinematic_model.w_lam1"),
            .w_k1 = declare_parameter<double>("kinematic_model.w_k1"),
            .w_cf1 = declare_parameter<double>("kinematic_model.w_cf1"),
            .xh0_bias = declare_parameter<double>("kinematic_model.xh0_bias"),
            .xh0_psi = declare_parameter<double>("kinematic_model.xh0_psi"),
            .xh0_v = declare_parameter<double>("kinematic_model.xh0_v"),
            .psi_bias = declare_parameter<double>("kinematic_model.psi_bias"),
            .psi_gain = declare_parameter<double>("kinematic_model.psi_gain"),
            .psi_v = declare_parameter<double>("kinematic_model.psi_v"),
            .obs_lv = declare_parameter<double>("kinematic_model.obs_lv"),
            .obs_lpsi = declare_parameter<double>("kinematic_model.obs_lpsi")
        },
        .power_model = {
            .coeffs = [this]() {
                std::array<double, PWR_N> coeffs {};
                for (int i = 0; i < PWR_N; ++i) {
                    coeffs[static_cast<size_t>(i)] = declare_parameter<double>("power_model.c" + std::to_string(i));
                }
                return coeffs;
            }()
        }
    };
    return mpc_params;
}

} // namespace nav_executor
