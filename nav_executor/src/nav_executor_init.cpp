#include <nav_executor/nav_executor_node.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>

#include <unordered_map>

namespace nav_executor {

NavExecutorNode::NavExecutorNode(const rclcpp::NodeOptions& options) : Node("nav_executor", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    enable_debug_ = declare_parameter<bool>("node.debug.enable");
    if (enable_debug_) {
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        debug_mpc_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("node.debug.mpc_path_pub_topic"), 1);
        debug_rough_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("node.debug.rough_path_pub_topic"), 1);
        debug_warmup_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("node.debug.warmup_path_pub_topic"), 1);
        debug_v_pred_pub_ = create_publisher<std_msgs::msg::Float64>(declare_parameter<std::string>("node.debug.v_pred_pub_topic"), 1);
        debug_w_pred_pub_ = create_publisher<std_msgs::msg::Float64>(declare_parameter<std::string>("node.debug.w_pred_pub_topic"), 1);
        debug_final_cost_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(declare_parameter<std::string>("node.debug.final_cost_map_pub_topic"), 1);
        debug_global_search_rollouts_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(declare_parameter<std::string>("node.debug.global_search_rollouts_pub_topic"), 1);
    }

    load_terrain_config();

    const MPCParams mpc_params = load_mpc_params();
    auto mpc_solver = std::make_shared<MPCSolver>(mpc_params, get_logger().get_child("mpc"));

    executor_ = std::make_unique<PathExecutor>(
        load_executor_params(), load_fsm_params(), mpc_solver,
        mpc_params.follow.normal_profile, mpc_params.follow.capability_profiles,
        load_blend_params(), get_logger()
    );

    StepRoutingMaskParams step_params;
    step_params.path_align_dot_threshold = declare_parameter<double>("path_planner.step_mask.path_align_dot_threshold");
    step_params.full_effect_radius = declare_parameter<double>("path_planner.step_mask.full_effect_radius");
    step_params.cutoff_radius = declare_parameter<double>("path_planner.step_mask.cutoff_radius");
    step_params.length_num_samples = static_cast<int>(declare_parameter<int>("path_planner.step_mask.length_num_samples"));
    step_routing_mask_ = std::make_shared<StepRoutingMask>(step_params);

    auto a_star = std::make_shared<AStarPlanner>(
        declare_parameter<double>("path_planner.planner.a_star.step_alignment_weight"),
        declare_parameter<double>("path_planner.planner.a_star.obstacle_weight"),
        declare_parameter<double>("path_planner.planner.a_star.step_proximity_weight"),
        declare_parameter<double>("path_planner.planner.a_star.step_mode_dot_threshold"),
        static_cast<int>(declare_parameter<int>("path_planner.planner.a_star.downsampled_waypoint_max_interval")),
        static_cast<int>(declare_parameter<int>("path_planner.planner.a_star.feasible_threshold"))
    );
    auto optimizer = std::make_shared<BSplineOptimizer>(load_optimizer_params());
    planner_ = std::make_unique<PathPlanner>(
        load_planner_config(), a_star, optimizer, step_routing_mask_, get_logger()
    );
    planner_->start();

    task_ = std::make_unique<TaskManager>(load_task_params(), planner_.get(), get_logger());

    proj_guard_params_ = {
        .dist_max = declare_parameter<double>("task_manager.route_monitor.proj_guard.dist_max"),
        .cost_max = declare_parameter<double>("task_manager.route_monitor.proj_guard.cost_max"),
        .cost_samples = static_cast<int>(declare_parameter<int>("task_manager.route_monitor.proj_guard.cost_samples"))
    };
    step_block_params_ = {
        .enable = declare_parameter<bool>("task_manager.route_monitor.step_block.enable"),
        .lookahead_distance = declare_parameter<double>("task_manager.route_monitor.step_block.lookahead_distance"),
        .sample_resolution = declare_parameter<double>("task_manager.route_monitor.step_block.sample_resolution"),
        .step_norm_threshold = declare_parameter<double>("task_manager.route_monitor.step_block.step_norm_threshold"),
        .obstacle_cost_threshold = declare_parameter<double>("task_manager.route_monitor.step_block.obstacle_cost_threshold"),
        .predicted_obstacle_ratio_threshold = declare_parameter<double>("task_manager.route_monitor.step_block.predicted_obstacle_ratio_threshold")
    };
    performance_replan_params_ = {
        .lookahead_distance = declare_parameter<double>("task_manager.route_monitor.performance.lookahead_distance")
    };

    remaining_energy_filter_alpha_ = declare_parameter<double>("node.remaining_energy_filter_alpha");
    prediction_horizon_seconds_ = declare_parameter<double>("node.prediction_horizon_seconds");
    prediction_weight_decay_ = declare_parameter<double>("node.prediction_weight_decay");

    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("node.topics.global_cost_map_sub"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            global_cost_map_ = std::make_shared<CostMap>(*msg);
            RCLCPP_INFO(get_logger(), "Received global cost map: (%d,%d) res=%.2f",
                global_cost_map_->width, global_cost_map_->height, global_cost_map_->resolution);
            try_init_step_mask();
            global_cost_map_sub_.reset();
        }
    );

    global_direction_map_sub_ = create_subscription<sensor_msgs::msg::Image>(
        declare_parameter<std::string>("node.topics.global_direction_map_sub"), 1,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            if (!global_cost_map_) {
                RCLCPP_WARN(get_logger(), "Received direction map before cost map; ignoring");
                return;
            }
            const cv::Mat img = cv_bridge::toCvShare(msg, "8UC3")->image;
            global_direction_map_ = std::make_shared<DirectionMap>(
                img, global_cost_map_->resolution, global_cost_map_->origin_x, global_cost_map_->origin_y
            );
            if (global_direction_map_->width != global_cost_map_->width || global_direction_map_->height != global_cost_map_->height) {
                RCLCPP_FATAL(get_logger(), "Direction map size mismatch with cost map!");
                throw std::runtime_error("Direction map size mismatch");
            }
            RCLCPP_INFO(get_logger(), "Received global direction map");
            try_init_step_mask();
            global_direction_map_sub_.reset();
        }
    );

    local_cost_maps_sub_ = create_subscription<interfaces::msg::CostMaps>(
        declare_parameter<std::string>("node.topics.local_cost_maps_sub"), 1,
        [this](const interfaces::msg::CostMaps::SharedPtr msg) { local_cost_maps_callback(msg); }
    );

    goal_sub_ = create_subscription<interfaces::msg::NavGoal>(
        declare_parameter<std::string>("node.topics.goal_sub"), 1,
        [this](const interfaces::msg::NavGoal::SharedPtr msg) {
            Goal g;
            g.position_map = Eigen::Vector2d(msg->x, msg->y);
            g.fixed = msg->fixed;
            pending_goal_ = g;
        }
    );

    chassis_status_sub_ = create_subscription<interfaces::msg::ChassisStatus>(
        declare_parameter<std::string>("node.topics.chassis_status_sub"), 1,
        [this](const interfaces::msg::ChassisStatus::SharedPtr msg) { chassis_status_callback(msg); }
    );

    comp_stage_sub_ = create_subscription<interfaces::msg::CompStage>(
        declare_parameter<std::string>("node.topics.comp_stage_sub"), 1,
        [this](const interfaces::msg::CompStage::SharedPtr msg) { comp_stage_ = msg->game_progress; }
    );

    spin_cmd_sub_ = create_subscription<interfaces::msg::SpinCmd>(
        declare_parameter<std::string>("node.topics.spin_cmd_sub"), 1,
        [this](const interfaces::msg::SpinCmd::SharedPtr msg) { spin_cmd_callback(msg); }
    );

    global_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("node.topics.global_path_pub"), 1);
    chassis_cmd_pub_ = create_publisher<interfaces::msg::ChassisCmd>(declare_parameter<std::string>("node.topics.chassis_cmd_pub"), 1);
    state_pub_ = create_publisher<interfaces::msg::NavExecutorState>(declare_parameter<std::string>("node.topics.state_pub"), 1);

    control_timer_ = create_wall_timer(std::chrono::duration<double>(MPC_DT), [this]() { control_tick(); });
}

NavExecutorNode::~NavExecutorNode() {
    if (planner_) planner_->stop();
}

void NavExecutorNode::try_init_step_mask() {
    if (step_mask_ready_ || !global_cost_map_ || !global_direction_map_) return;
    try {
        step_routing_mask_->initialize(*global_cost_map_, global_direction_map_);
        step_mask_ready_ = true;
        RCLCPP_INFO(get_logger(), "StepRoutingMask initialized");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to initialize StepRoutingMask: %s", e.what());
    }
}

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

    terrain_profiles_.high_performance_buffercap_threshold = declare_parameter<double>("terrain_profiles.high_performance.buffercap_threshold");
    terrain_profiles_.high_performance_supercap_threshold = declare_parameter<double>("terrain_profiles.high_performance.supercap_threshold");
    terrain_profiles_.high_performance_rfr_pwr_limit_threshold = declare_parameter<double>("terrain_profiles.high_performance.rfr_pwr_limit_threshold");

    // directional_labels — 有方向语义的标签 (SLOPE..STEP_HIGH)
    struct DirEntry { const char* name; uint8_t label; };
    const DirEntry dir_entries[] = {
        {"slope", 2}, {"step_l1", 3}, {"step_l2", 4}, {"fly_slope", 5}, {"step_high", 6},
    };
    for (const auto& [entry_name, label] : dir_entries) {
        const size_t idx = label - 2;
        auto& dir_modes = terrain_profiles_.directional_labels[idx];

        // modes 段在 YAML 中是 entry 级别的共享字典，同一个 mode_name 只 declare 一次
        std::unordered_map<std::string, TerrainStepRule> mode_map;

        for (const auto& dir : {"up", "down"}) {
            const auto prefix = std::string("terrain_profiles.directional_labels.") + entry_name + "." + dir;
            const auto names = declare_parameter<std::vector<std::string>>(prefix);

            auto& target = (dir == std::string("up")) ? dir_modes.up : dir_modes.down;
            target.reserve(names.size());

            for (const std::string& mode_name : names) {
                // mode 参数只 declare 一次，缓存到 mode_map 后复用
                if (mode_map.find(mode_name) == mode_map.end()) {
                    const std::string mode_prefix = std::string("terrain_profiles.directional_labels.") + entry_name + ".modes." + mode_name;
                    mode_map[mode_name] = TerrainStepRule{
                        .name = mode_name,
                        .chassis_mode = static_cast<uint8_t>(declare_parameter<int>(mode_prefix + ".chassis_mode")),
                        .capability = capability_level_from_string(declare_parameter<std::string>(mode_prefix + ".capability")),
                        .speed = {.min = declare_parameter<double>(mode_prefix + ".speed.min"), .max = declare_parameter<double>(mode_prefix + ".speed.max")},
                        .requires_high_performance = declare_parameter<bool>(mode_prefix + ".requires_high_perf"),
                        .run_up = declare_parameter<double>(mode_prefix + ".run_up"),
                    };
                }
                target.push_back(mode_map[mode_name]);
            }
        }
    }
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
        .follow_to_spin_vel_max = declare_parameter<double>("path_executor.state_machine.follow_to_spin_vel_max"),
        .spin_to_follow_omega_max = declare_parameter<double>("path_executor.state_machine.spin_to_follow_omega_max"),
        .to_idle_vel_max = declare_parameter<double>("path_executor.state_machine.to_idle_vel_max"),
        .to_idle_omega_max = declare_parameter<double>("path_executor.state_machine.to_idle_omega_max"),
        .stopping_timeout = declare_parameter<double>("path_executor.state_machine.stopping_timeout"),
    };
    fsm.recovery = {
        .hazard_cost_threshold = declare_parameter<double>("path_executor.recovery.hazard.cost_threshold"),
        .hazard_step_norm_threshold = declare_parameter<double>("path_executor.recovery.hazard.step_norm_threshold"),
        .safe_cost_threshold = declare_parameter<double>("path_executor.recovery.safe.cost_threshold"),
        .safe_step_norm_threshold = declare_parameter<double>("path_executor.recovery.safe.step_norm_threshold"),
        .recovery_cost_threshold = declare_parameter<double>("path_executor.recovery.search.recovery_cost_threshold"),
        .radius_min = declare_parameter<double>("path_executor.recovery.search.radius_min"),
        .radius_max = declare_parameter<double>("path_executor.recovery.search.radius_max"),
        .radius_samples = static_cast<int>(declare_parameter<int>("path_executor.recovery.search.radius_samples")),
        .angle_samples = static_cast<int>(declare_parameter<int>("path_executor.recovery.search.angle_samples")),
        .path_integral_resolution = declare_parameter<double>("path_executor.recovery.search.path_integral_resolution"),
        .path_integral_cost_weight = declare_parameter<double>("path_executor.recovery.search.path_integral_cost_weight"),
        .path_integral_step_weight = declare_parameter<double>("path_executor.recovery.search.path_integral_step_weight"),
        .step_ascent_penalty_weight = declare_parameter<double>("path_executor.recovery.search.step_ascent_penalty_weight"),
        .step_ascent_penalty_norm_threshold = declare_parameter<double>("path_executor.recovery.search.step_ascent_penalty_norm_threshold"),
        .step_ascent_penalty_dot_threshold = declare_parameter<double>("path_executor.recovery.search.step_ascent_penalty_dot_threshold"),
        .safe_hold_time = declare_parameter<double>("path_executor.recovery.exit.safe_hold_time"),
        .goal_timeout = declare_parameter<double>("path_executor.recovery.search.goal_timeout"),
    };
    fsm.stuck = {
        .cmd_vel_threshold = declare_parameter<double>("path_executor.recovery.stuck.cmd_vel_threshold"),
        .timeout = declare_parameter<double>("path_executor.recovery.stuck.timeout"),
        .max_displacement = declare_parameter<double>("path_executor.recovery.stuck.max_displacement"),
        .reverse_speed = declare_parameter<double>("path_executor.recovery.stuck.reverse_speed"),
        .reverse_displacement = declare_parameter<double>("path_executor.recovery.stuck.reverse_displacement"),
        .reverse_timeout = declare_parameter<double>("path_executor.recovery.stuck.reverse_timeout"),
    };
    return fsm;
}

// ═══════════════════════ PathExecutor 参数 ═══════════════════

PathExecutorParams NavExecutorNode::load_executor_params() {
    PathExecutorParams p;
    p.stop_threshold_dist = declare_parameter<double>("path_executor.misc.stop_threshold_dist");
    p.stop_threshold_u = declare_parameter<double>("path_executor.misc.stop_threshold_u");
    p.step_dist_offset = declare_parameter<double>("path_executor.misc.step_dist_offset");
    p.follow_no_progress_guard = {
        .landmark_spacing = declare_parameter<double>("path_executor.no_progress_guard.follow.landmark_spacing"),
        .timeout = declare_parameter<double>("path_executor.no_progress_guard.follow.timeout"),
    };
    p.stepping_no_progress_guard = {
        .landmark_spacing = declare_parameter<double>("path_executor.no_progress_guard.stepping.landmark_spacing"),
        .timeout = declare_parameter<double>("path_executor.no_progress_guard.stepping.timeout"),
    };
    return p;
}

// ═══════════════════════ Planner 参数 ════════════════════════

PlannerConfig NavExecutorNode::load_planner_config() {
    PlannerConfig c;
    c.occupied_threshold = static_cast<int>(declare_parameter<int>("path_planner.traversability.occupied_threshold"));
    c.on_step_threshold = declare_parameter<double>("path_planner.traversability.on_step_threshold");
    c.start_prediction_enable = declare_parameter<bool>("path_planner.start_prediction.enable");
    c.start_prediction_max_accel = declare_parameter<double>("path_planner.start_prediction.max_accel");
    c.start_prediction_planning_delay = declare_parameter<double>("path_planner.start_prediction.planning_delay");
    c.start_prediction_min_speed = declare_parameter<double>("path_planner.start_prediction.min_speed");
    c.start_prediction_collision_check_step = declare_parameter<double>("path_planner.start_prediction.collision_check_step");
    c.nudge_max_distance = declare_parameter<double>("path_planner.nudge.max_distance");
    c.goal_reached_distance = declare_parameter<double>("path_planner.planner.goal_reached_distance");
    c.skip_distance = declare_parameter<double>("path_planner.planner.skip_distance");
    c.step_detection = {
        .detect_norm_threshold = declare_parameter<double>("path_planner.step.detection.detect_norm_threshold"),
        .detect_dot_threshold = declare_parameter<double>("path_planner.step.detection.detect_dot_threshold"),
        .path_sample_resolution = declare_parameter<double>("path_planner.step.detection.path_sample_resolution"),
        .prepare_distance = declare_parameter<double>("path_planner.step.detection.prepare_distance"),
        .active_distance = declare_parameter<double>("path_planner.step.detection.active_distance"),
        .release_distance = declare_parameter<double>("path_planner.step.detection.release_distance"),
        .gate_transition_distance = declare_parameter<double>("path_planner.step.detection.gate_transition_distance"),
    };
    c.enable_debug = enable_debug_;
    return c;
}

BSplineOptimizer::Params NavExecutorNode::load_optimizer_params() {
    return {
        .step_norm_threshold = declare_parameter<double>("path_planner.path_optimizer.step_norm_threshold"),
        .step_norm_transition = declare_parameter<double>("path_planner.path_optimizer.step_norm_transition"),
        .step_detection_samples_per_meter = declare_parameter<double>("path_planner.path_optimizer.step_detection_samples_per_meter"),
        .warmup = {
            .obstacle_weight = declare_parameter<double>("path_planner.path_optimizer.warmup.obstacle_weight"),
            .direction_weight = declare_parameter<double>("path_planner.path_optimizer.warmup.direction_weight"),
            .step_weight = declare_parameter<double>("path_planner.path_optimizer.warmup.step_weight"),
            .start_end_weight = declare_parameter<double>("path_planner.path_optimizer.warmup.start_end_weight"),
            .smoothness_weight = declare_parameter<double>("path_planner.path_optimizer.warmup.smoothness_weight"),
            .samples_per_meter = declare_parameter<double>("path_planner.path_optimizer.warmup.samples_per_meter"),
            .max_iterations = static_cast<int>(declare_parameter<int>("path_planner.path_optimizer.warmup.max_iterations")),
            .max_curvature = declare_parameter<double>("path_planner.path_optimizer.warmup.max_curvature"),
            .length_penalty_weight = declare_parameter<double>("path_planner.path_optimizer.warmup.length_penalty_weight"),
            .curvature = {
                .base_weight = declare_parameter<double>("path_planner.path_optimizer.warmup.curvature.base_weight"),
                .base_beta = declare_parameter<double>("path_planner.path_optimizer.warmup.curvature.base_beta"),
                .limit_weight = declare_parameter<double>("path_planner.path_optimizer.warmup.curvature.limit_weight"),
                .limit_beta = declare_parameter<double>("path_planner.path_optimizer.warmup.curvature.limit_beta"),
                .min_speed_epsilon = declare_parameter<double>("path_planner.path_optimizer.warmup.curvature.min_speed_epsilon"),
                .speed_gate_threshold = declare_parameter<double>("path_planner.path_optimizer.warmup.curvature.speed_gate_threshold"),
            },
        },
        .main = {
            .obstacle_weight = declare_parameter<double>("path_planner.path_optimizer.main.obstacle_weight"),
            .direction_weight = declare_parameter<double>("path_planner.path_optimizer.main.direction_weight"),
            .step_weight = declare_parameter<double>("path_planner.path_optimizer.main.step_weight"),
            .start_end_weight = declare_parameter<double>("path_planner.path_optimizer.main.start_end_weight"),
            .smoothness_weight = declare_parameter<double>("path_planner.path_optimizer.main.smoothness_weight"),
            .samples_per_meter = declare_parameter<double>("path_planner.path_optimizer.main.samples_per_meter"),
            .max_iterations = static_cast<int>(declare_parameter<int>("path_planner.path_optimizer.main.max_iterations")),
            .max_refinement_iterations = static_cast<int>(declare_parameter<int>("path_planner.path_optimizer.main.max_refinement_iterations")),
            .near_max_curvature = declare_parameter<double>("path_planner.path_optimizer.main.near_max_curvature"),
            .far_max_curvature = declare_parameter<double>("path_planner.path_optimizer.main.far_max_curvature"),
            .step_extension_distance = declare_parameter<double>("path_planner.path_optimizer.main.step_extension_distance"),
            .step_transition_distance = declare_parameter<double>("path_planner.path_optimizer.main.step_transition_distance"),
            .interval_iou_threshold = declare_parameter<double>("path_planner.path_optimizer.main.interval_iou_threshold"),
            .length_penalty_weight = declare_parameter<double>("path_planner.path_optimizer.main.length_penalty_weight"),
            .curvature = {
                .base_weight = declare_parameter<double>("path_planner.path_optimizer.main.curvature.base_weight"),
                .base_beta = declare_parameter<double>("path_planner.path_optimizer.main.curvature.base_beta"),
                .limit_weight = declare_parameter<double>("path_planner.path_optimizer.main.curvature.limit_weight"),
                .limit_beta = declare_parameter<double>("path_planner.path_optimizer.main.curvature.limit_beta"),
                .min_speed_epsilon = declare_parameter<double>("path_planner.path_optimizer.main.curvature.min_speed_epsilon"),
                .speed_gate_threshold = declare_parameter<double>("path_planner.path_optimizer.main.curvature.speed_gate_threshold"),
            },
        },
    };
}

// ═══════════════════════ TaskManager 参数 ═══════════════════

TaskManagerParams NavExecutorNode::load_task_params() {
    return {
        .goal_equivalence_distance = declare_parameter<double>("task_manager.goal_equivalence_distance"),
        .plan_cooldown = declare_parameter<double>("task_manager.plan_cooldown"),
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
                .q_u = declare_parameter<double>("mpc.follow.tracking_weights.q_u"),
                .y_tube = declare_parameter<double>("mpc.follow.tracking_weights.y_tube"),
                .q_term_prog = declare_parameter<double>("mpc.follow.tracking_weights.q_term_prog"),
                .q_term_lateral = declare_parameter<double>("mpc.follow.tracking_weights.q_term_lateral")
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
            .global_search = {
                .enable = declare_parameter<bool>("mpc.follow.global_search.enable"),
                .num_threads = static_cast<int>(declare_parameter<int>("mpc.follow.global_search.num_threads")),
                .batch_size = static_cast<int>(declare_parameter<int>("mpc.follow.global_search.batch_size")),
                .iteration_count = static_cast<int>(declare_parameter<int>("mpc.follow.global_search.iteration_count")),
                .elite_fraction = declare_parameter<double>("mpc.follow.global_search.elite_fraction"),
                .sampling_std = {
                    .velocity = declare_parameter<double>("mpc.follow.global_search.sampling_std.velocity"),
                    .omega = declare_parameter<double>("mpc.follow.global_search.sampling_std.omega")
                },
                .noise_smoothing = {
                    .window = static_cast<int>(declare_parameter<int>("mpc.follow.global_search.noise_smoothing.window")),
                    .passes = static_cast<int>(declare_parameter<int>("mpc.follow.global_search.noise_smoothing.passes"))
                },
                .include_nominal_trajectory = declare_parameter<bool>("mpc.follow.global_search.include_nominal_trajectory"),
                .candidate_count = static_cast<int>(declare_parameter<int>("mpc.follow.global_search.candidate_count")),
                .min_period_ms = static_cast<int>(declare_parameter<int>("mpc.follow.global_search.min_period_ms")),
                .max_seed_age_ticks = static_cast<int>(declare_parameter<int>("mpc.follow.global_search.max_seed_age_ticks")),
                .improvement_margin = declare_parameter<double>("mpc.follow.global_search.improvement_margin"),
                .hysteresis_count = static_cast<int>(declare_parameter<int>("mpc.follow.global_search.hysteresis_count")),
                .refinement_iterations = static_cast<int>(declare_parameter<int>("mpc.follow.global_search.refinement_iterations"))
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
