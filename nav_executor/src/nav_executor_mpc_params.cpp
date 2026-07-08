#include <nav_executor/nav_executor_node.hpp>

namespace nav_executor {

// MPC 参数加载（迁自旧 path_follower_node.cpp）。
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
