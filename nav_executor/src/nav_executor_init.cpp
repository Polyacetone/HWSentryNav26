#include <nav_executor/nav_executor_node.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace nav_executor {

namespace {

void require_parameter(const bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

bool positive_finite(const double value) {
    return std::isfinite(value) && value > 0.0;
}

bool nonnegative_finite(const double value) {
    return std::isfinite(value) && value >= 0.0;
}

} // anonymous namespace

NavExecutorNode::NavExecutorNode(const rclcpp::NodeOptions& options) : Node("nav_executor", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    enable_debug_ = declare_parameter<bool>("node.debug.enable");
    if (enable_debug_) {
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        debug_mpc_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("node.debug.mpc_path_pub_topic"), 1);
        debug_minco_trajectory_pub_ = create_publisher<visualization_msgs::msg::Marker>(
            declare_parameter<std::string>("node.debug.minco_trajectory_pub_topic"), 1
        );
        debug_rough_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("node.debug.rough_path_pub_topic"), 1);
        debug_warmup_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("node.debug.warmup_path_pub_topic"), 1);
        debug_diag_pub_ = create_publisher<interfaces::msg::NavExecutorDiag>(
            declare_parameter<std::string>("node.debug.diag_pub_topic"), 1
        );
        debug_final_cost_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(declare_parameter<std::string>("node.debug.final_cost_map_pub_topic"), 1);
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

    const PlannerConfig planner_config = load_planner_config(
        mpc_params.follow.normal_profile.command_envelope.velocity
    );
    minco_debug_velocity_min_ = 0.0;
    minco_debug_velocity_max_ = planner_config.minco.trajectory_limits.velocity_max;
    planner_ = std::make_unique<PathPlanner>(planner_config, step_routing_mask_, get_logger());
    planner_->start();

    task_ = std::make_unique<TaskManager>(load_task_params(), planner_.get(), get_logger());

    route_tracker_params_ = {
        .initial_search_distance = declare_parameter<double>("task_manager.route_tracker.initial_search_distance"),
        .max_tracking_error = declare_parameter<double>("task_manager.route_tracker.max_tracking_error"),
        .prediction_time_limit = declare_parameter<double>("task_manager.route_tracker.prediction_time_limit"),
        .path_speed_filter_alpha = declare_parameter<double>("task_manager.route_tracker.path_speed_filter_alpha"),
        .projection = {
            .prediction_weight = declare_parameter<double>("task_manager.route_tracker.projection_prediction_weight"),
            .search_distance_backward = declare_parameter<double>("task_manager.route_tracker.search_distance_backward"),
            .search_distance_forward = declare_parameter<double>("task_manager.route_tracker.search_distance_forward"),
        },
    };
    require_parameter(nonnegative_finite(route_tracker_params_.initial_search_distance), "route_tracker.initial_search_distance must be finite and non-negative");
    require_parameter(positive_finite(route_tracker_params_.max_tracking_error), "route_tracker.max_tracking_error must be finite and positive");
    require_parameter(positive_finite(route_tracker_params_.prediction_time_limit), "route_tracker.prediction_time_limit must be finite and positive");
    require_parameter(route_tracker_params_.path_speed_filter_alpha > 0.0
        && route_tracker_params_.path_speed_filter_alpha <= 1.0,
        "route_tracker.path_speed_filter_alpha must be in (0, 1]");
    require_parameter(positive_finite(route_tracker_params_.projection.prediction_weight)
        && nonnegative_finite(route_tracker_params_.projection.search_distance_backward)
        && positive_finite(route_tracker_params_.projection.search_distance_forward),
        "route_tracker projection parameters are invalid");
    route_tracker_ = std::make_unique<RouteTracker>(route_tracker_params_);
    proj_guard_params_ = {
        .cost_max = declare_parameter<double>("task_manager.route_monitor.proj_guard.cost_max"),
        .cost_samples = static_cast<int>(declare_parameter<int>("task_manager.route_monitor.proj_guard.cost_samples"))
    };
    step_block_params_ = {
        .enable = declare_parameter<bool>("task_manager.route_monitor.step_block.enable"),
        .lookahead_distance = declare_parameter<double>("task_manager.route_monitor.step_block.lookahead_distance"),
        .sample_resolution = declare_parameter<double>("task_manager.route_monitor.step_block.sample_resolution"),
        .obstacle_cost_threshold = declare_parameter<double>("task_manager.route_monitor.step_block.obstacle_cost_threshold"),
        .predicted_obstacle_ratio_threshold = declare_parameter<double>("task_manager.route_monitor.step_block.predicted_obstacle_ratio_threshold")
    };
    require_parameter(
        nonnegative_finite(step_block_params_.lookahead_distance)
        && positive_finite(step_block_params_.sample_resolution)
        && step_block_params_.obstacle_cost_threshold >= 0.0
        && step_block_params_.obstacle_cost_threshold <= 255.0
        && step_block_params_.predicted_obstacle_ratio_threshold >= 0.0
        && step_block_params_.predicted_obstacle_ratio_threshold <= 1.0,
        "route_monitor step_block parameters are invalid"
    );
    performance_replan_params_ = {
        .lookahead_distance = declare_parameter<double>("task_manager.route_monitor.performance.lookahead_distance")
    };

    remaining_energy_filter_alpha_ = declare_parameter<double>("node.remaining_energy_filter_alpha");
    path_publish_sample_resolution_ = declare_parameter<double>("node.path_publish_sample_resolution");
    dynamic_prediction_horizon_seconds_ = declare_parameter<double>("path_planner.dynamic_prediction.horizon_seconds");
    dynamic_prediction_weight_decay_ = declare_parameter<double>("path_planner.dynamic_prediction.weight_decay");
    require_parameter(positive_finite(path_publish_sample_resolution_), "path publish sample_resolution must be finite and positive");
    require_parameter(nonnegative_finite(dynamic_prediction_horizon_seconds_), "dynamic prediction horizon_seconds must be finite and non-negative");
    require_parameter(dynamic_prediction_weight_decay_ >= 0.0 && dynamic_prediction_weight_decay_ <= 1.0, "dynamic prediction weight_decay must be in [0, 1]");

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
        .command_envelope = {
            .velocity = {
                .min = declare_parameter<double>(prefix + ".command_envelope.velocity.min"),
                .max = declare_parameter<double>(prefix + ".command_envelope.velocity.max"),
            },
            .angular_velocity = {
                .min = declare_parameter<double>(prefix + ".command_envelope.angular_velocity.min"),
                .max = declare_parameter<double>(prefix + ".command_envelope.angular_velocity.max"),
            },
        },
        .command_dynamics = {
            .velocity_rate_max = declare_parameter<double>(prefix + ".command_dynamics.velocity_rate_max"),
            .angular_velocity_rate_max = declare_parameter<double>(prefix + ".command_dynamics.angular_velocity_rate_max"),
            .lateral_acceleration_max = declare_parameter<double>(prefix + ".command_dynamics.lateral_acceleration_max"),
        },
    };
}

void NavExecutorNode::load_terrain_config() {
    using enum CapabilityLevel;
    traversal_configuration_.capability_profiles[static_cast<size_t>(LOW)] = load_capability_profile("capability_management.profiles.low");
    traversal_configuration_.capability_profiles[static_cast<size_t>(MEDIUM)] = load_capability_profile("capability_management.profiles.medium");
    traversal_configuration_.capability_profiles[static_cast<size_t>(HIGH)] = load_capability_profile("capability_management.profiles.high");
    bidirectional_profile_ = load_capability_profile(
        "capability_management.profiles.bidirectional"
    );

    traversal_configuration_.high_performance_buffercap_threshold = declare_parameter<double>("terrain_traversal.high_performance_available.buffercap_threshold");
    traversal_configuration_.high_performance_supercap_threshold = declare_parameter<double>("terrain_traversal.high_performance_available.supercap_threshold");
    traversal_configuration_.high_performance_rfr_pwr_limit_threshold = declare_parameter<double>("terrain_traversal.high_performance_available.rfr_pwr_limit_threshold");

    // directional_labels — 有方向语义的标签 (SLOPE..STEP_HIGH)
    struct DirEntry { const char* name; uint8_t label; };
    const DirEntry dir_entries[] = {
        {"slope", 2}, {"step_l1", 3}, {"step_l2", 4}, {"fly_slope", 5}, {"step_high", 6},
    };
    for (const auto& [entry_name, label] : dir_entries) {
        const size_t idx = label - 2;
        auto& dir_modes = traversal_configuration_.directional_labels[idx];

        // modes 段在 YAML 中是 entry 级别的共享字典，同一个 mode_name 只 declare 一次
        std::unordered_map<std::string, TraversalMode> mode_map;

        for (const auto& dir : {"up", "down"}) {
            const auto prefix = std::string("terrain_traversal.directional_labels.") + entry_name + "." + dir;
            const auto names = declare_parameter<std::vector<std::string>>(prefix, std::vector<std::string>{});

            auto& target = (dir == std::string("up")) ? dir_modes.up : dir_modes.down;
            target.reserve(names.size());

            for (const std::string& mode_name : names) {
                // sentinel "disabled" —— 在该方向屏蔽此地形（无需在 modes 中定义）
                if (mode_name == "disabled") { continue; }

                // mode 参数只 declare 一次，缓存到 mode_map 后复用
                if (mode_map.find(mode_name) == mode_map.end()) {
                    const std::string mode_prefix = std::string("terrain_traversal.directional_labels.") + entry_name + ".modes." + mode_name;
                    mode_map[mode_name] = TraversalMode{
                        .name = mode_name,
                        .chassis_mode = static_cast<uint8_t>(declare_parameter<int>(mode_prefix + ".chassis_mode")),
                        .capability = capability_level_from_string(declare_parameter<std::string>(mode_prefix + ".capability")),
                        .velocity_window = {
                            .min = declare_parameter<double>(mode_prefix + ".velocity_target.min"),
                            .max = declare_parameter<double>(mode_prefix + ".velocity_target.max"),
                        },
                        .requires_high_performance = declare_parameter<bool>(mode_prefix + ".requires_high_perf"),
                        .run_up = declare_parameter<double>(mode_prefix + ".run_up"),
                    };
                }
                target.push_back(mode_map[mode_name]);
            }
        }
    }

    const auto valid_capability = [](const CapabilityProfile& profile) {
        const auto& command = profile.command_envelope;
        const auto& dynamics = profile.command_dynamics;
        return std::isfinite(command.velocity.min)
            && std::isfinite(command.velocity.max)
            && command.velocity.min < command.velocity.max
            && std::isfinite(command.angular_velocity.min)
            && std::isfinite(command.angular_velocity.max)
            && command.angular_velocity.min < command.angular_velocity.max
            && positive_finite(dynamics.velocity_rate_max)
            && positive_finite(dynamics.angular_velocity_rate_max)
            && positive_finite(dynamics.lateral_acceleration_max);
    };
    for (const CapabilityProfile& profile : traversal_configuration_.capability_profiles) {
        require_parameter(
            valid_capability(profile) && profile.command_envelope.velocity.min > 0.0,
            "forward capability profile is invalid"
        );
    }
    require_parameter(
        valid_capability(bidirectional_profile_)
            && bidirectional_profile_.command_envelope.velocity.min < 0.0
            && bidirectional_profile_.command_envelope.velocity.max > 0.0,
        "bidirectional capability profile is invalid"
    );
    for (const DirectionalTraversalModes& label : traversal_configuration_.directional_labels) {
        for (const auto* modes : {&label.up, &label.down}) {
            for (const TraversalMode& mode : *modes) {
                require_parameter(
                    nonnegative_finite(mode.velocity_window.min)
                        && std::isfinite(mode.velocity_window.max)
                        && mode.velocity_window.min <= mode.velocity_window.max
                        && nonnegative_finite(mode.run_up),
                    "terrain traversal mode has an invalid velocity target or run-up"
                );
            }
        }
    }
}

ProfileBlendParams NavExecutorNode::load_blend_params() {
    const ProfileBlendParams params {
        .v_step = declare_parameter<double>("capability_management.transition_per_tick.velocity"),
        .w_step = declare_parameter<double>("capability_management.transition_per_tick.angular_velocity"),
        .acc_step = declare_parameter<double>("capability_management.transition_per_tick.velocity_rate"),
        .alpha_step = declare_parameter<double>("capability_management.transition_per_tick.angular_velocity_rate"),
        .a_lat_step = declare_parameter<double>("capability_management.transition_per_tick.lateral_acceleration"),
    };
    require_parameter(
        positive_finite(params.v_step) && positive_finite(params.w_step)
            && positive_finite(params.acc_step) && positive_finite(params.alpha_step)
            && positive_finite(params.a_lat_step),
        "capability transition_per_tick values must be finite and positive"
    );
    return params;
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
    p.stop_threshold_remaining_distance = declare_parameter<double>("path_executor.misc.stop_threshold_remaining_distance");
    p.step_dist_offset = declare_parameter<double>("path_executor.misc.step_dist_offset");
    p.command_history_timeout = declare_parameter<double>("path_executor.misc.command_history_timeout");
    p.follow_no_progress_guard = {
        .arc_length_landmark_spacing = declare_parameter<double>(
            "path_executor.no_progress_guard.follow.arc_length_landmark_spacing"
        ),
        .timeout = declare_parameter<double>("path_executor.no_progress_guard.follow.timeout"),
    };
    p.stepping_no_progress_guard = {
        .arc_length_landmark_spacing = declare_parameter<double>(
            "path_executor.no_progress_guard.stepping.arc_length_landmark_spacing"
        ),
        .timeout = declare_parameter<double>("path_executor.no_progress_guard.stepping.timeout"),
    };
    require_parameter(
        positive_finite(p.follow_no_progress_guard.arc_length_landmark_spacing),
        "follow arc_length_landmark_spacing must be finite and positive"
    );
    require_parameter(
        positive_finite(p.stepping_no_progress_guard.arc_length_landmark_spacing),
        "stepping arc_length_landmark_spacing must be finite and positive"
    );
    require_parameter(
        positive_finite(p.follow_no_progress_guard.timeout)
        && positive_finite(p.stepping_no_progress_guard.timeout),
        "no-progress timeouts must be finite and positive"
    );
    require_parameter(
        std::isfinite(p.command_history_timeout)
        && p.command_history_timeout >= MPC_DT,
        "command_history_timeout must be finite and at least one MPC period"
    );
    return p;
}

// ═══════════════════════ Planner 参数 ════════════════════════

PlannerConfig NavExecutorNode::load_planner_config(
    const SignedVelocityBounds& forward_velocity_bounds
) {
    PlannerConfig c;
    c.occupied_threshold = static_cast<int>(declare_parameter<int>("path_planner.traversability.occupied_threshold"));
    c.on_step_threshold = declare_parameter<double>("path_planner.traversability.on_step_threshold");
    c.nudge_max_distance = declare_parameter<double>("path_planner.nudge.max_distance");
    c.goal_reached_distance = declare_parameter<double>("path_planner.planner.goal_reached_distance");
    c.seed_resample_distance = declare_parameter<double>("path_planner.planner.seed_resample_distance");
    c.forward_velocity_bounds = forward_velocity_bounds;

    c.dijkstra = {
        .obstacle_weight = declare_parameter<double>("path_planner.dijkstra.obstacle_weight"),
        .feasible_threshold = static_cast<int>(declare_parameter<int>("path_planner.dijkstra.feasible_threshold")),
    };
    c.kinodynamic = {
        .state_limits = {
            .speed_max = declare_parameter<double>("path_planner.kinodynamic.state_limits.speed_max"),
            .angular_velocity_max = declare_parameter<double>("path_planner.kinodynamic.state_limits.angular_velocity_max"),
            .acceleration_max = declare_parameter<double>("path_planner.kinodynamic.state_limits.acceleration_max"),
            .lateral_acceleration_max = declare_parameter<double>("path_planner.kinodynamic.state_limits.lateral_acceleration_max"),
        },
        .tangential_accel_samples = static_cast<int>(declare_parameter<int>("path_planner.kinodynamic.tangential_accel_samples")),
        .normal_accel_samples = static_cast<int>(declare_parameter<int>("path_planner.kinodynamic.normal_accel_samples")),
        .primitive_duration = declare_parameter<double>("path_planner.kinodynamic.primitive_duration"),
        .collision_substeps = static_cast<int>(declare_parameter<int>("path_planner.kinodynamic.collision_substeps")),
        .dedup_xy = declare_parameter<double>("path_planner.kinodynamic.dedup_xy"),
        .dedup_theta = declare_parameter<double>("path_planner.kinodynamic.dedup_theta"),
        .dedup_speed = declare_parameter<double>("path_planner.kinodynamic.dedup_speed"),
        .heuristic_weight = declare_parameter<double>("path_planner.kinodynamic.heuristic_weight"),
        .goal_tolerance = declare_parameter<double>("path_planner.kinodynamic.goal_tolerance"),
        .max_expansions = static_cast<int>(declare_parameter<int>("path_planner.kinodynamic.max_expansions")),
    };
    c.minco = {
        .weights = {
            .energy = declare_parameter<double>("path_planner.minco.penalty_weights.energy"),
            .time = declare_parameter<double>("path_planner.minco.penalty_weights.time"),
            .obstacle = declare_parameter<double>("path_planner.minco.penalty_weights.obstacle"),
            .trajectory_velocity = declare_parameter<double>("path_planner.minco.penalty_weights.trajectory_velocity"),
            .lateral_acc = declare_parameter<double>("path_planner.minco.penalty_weights.lateral_acceleration"),
            .omega = declare_parameter<double>("path_planner.minco.penalty_weights.angular_velocity"),
            .accel = declare_parameter<double>("path_planner.minco.penalty_weights.acceleration"),
            .traversal_alignment = declare_parameter<double>("path_planner.minco.penalty_weights.traversal_alignment"),
            .traversal_velocity_target = declare_parameter<double>("path_planner.minco.penalty_weights.traversal_velocity_target"),
            .prohibited_traversal = declare_parameter<double>("path_planner.minco.penalty_weights.prohibited_traversal"),
            .runup_accel = declare_parameter<double>("path_planner.minco.penalty_weights.runup_acceleration"),
            .runup_omega = declare_parameter<double>("path_planner.minco.penalty_weights.runup_angular_velocity"),
        },
        .trajectory_limits = {
            .velocity_max = declare_parameter<double>("path_planner.minco.trajectory_limits.velocity_max"),
            .angular_velocity_max = declare_parameter<double>("path_planner.minco.trajectory_limits.angular_velocity_max"),
            .acceleration_max = declare_parameter<double>("path_planner.minco.trajectory_limits.acceleration_max"),
            .lateral_acceleration_max = declare_parameter<double>("path_planner.minco.trajectory_limits.lateral_acceleration_max"),
        },
        .terrain_gate = {
            .norm_lo = declare_parameter<double>("path_planner.minco.terrain_gate.norm_lo"),
            .norm_hi = declare_parameter<double>("path_planner.minco.terrain_gate.norm_hi"),
            .motion_speed_scale = declare_parameter<double>(
                "path_planner.minco.terrain_gate.motion_speed_scale"
            ),
        },
        .samples_per_segment = static_cast<int>(declare_parameter<int>("path_planner.minco.samples_per_segment")),
        .max_iterations = static_cast<int>(declare_parameter<int>("path_planner.minco.max_iterations")),
        .optimizer = {
            .position_scale = declare_parameter<double>("path_planner.minco.optimizer.position_scale"),
            .physical_time_scale = declare_parameter<double>("path_planner.minco.optimizer.physical_time_scale"),
            .max_virtual_time_scale = declare_parameter<double>("path_planner.minco.optimizer.max_virtual_time_scale"),
            .max_function_evaluations = static_cast<int>(declare_parameter<int>("path_planner.minco.optimizer.max_function_evaluations")),
            .history_size = static_cast<int>(declare_parameter<int>("path_planner.minco.optimizer.history_size")),
            .gradient_tolerance = declare_parameter<double>("path_planner.minco.optimizer.gradient_tolerance"),
            .scaled_step_tolerance = declare_parameter<double>("path_planner.minco.optimizer.scaled_step_tolerance"),
            .trust_region = {
                .initial_radius = declare_parameter<double>("path_planner.minco.optimizer.trust_region.initial_radius"),
                .min_radius = declare_parameter<double>("path_planner.minco.optimizer.trust_region.min_radius"),
                .max_radius = declare_parameter<double>("path_planner.minco.optimizer.trust_region.max_radius"),
                .acceptance_ratio = declare_parameter<double>("path_planner.minco.optimizer.trust_region.acceptance_ratio"),
                .shrink_ratio = declare_parameter<double>("path_planner.minco.optimizer.trust_region.shrink_ratio"),
                .expansion_ratio = declare_parameter<double>("path_planner.minco.optimizer.trust_region.expansion_ratio"),
                .shrink_factor = declare_parameter<double>("path_planner.minco.optimizer.trust_region.shrink_factor"),
                .expansion_factor = declare_parameter<double>("path_planner.minco.optimizer.trust_region.expansion_factor"),
                .max_consecutive_rejections = static_cast<int>(declare_parameter<int>("path_planner.minco.optimizer.trust_region.max_consecutive_rejections")),
                .history_reset_after_rejections = static_cast<int>(declare_parameter<int>("path_planner.minco.optimizer.trust_region.history_reset_after_rejections")),
            },
            .curvature_relative_threshold = declare_parameter<double>("path_planner.minco.optimizer.curvature_relative_threshold"),
            .history_acceptance_ratio = declare_parameter<double>("path_planner.minco.optimizer.history_acceptance_ratio"),
        },
        .min_segment_time = declare_parameter<double>("path_planner.minco.min_segment_time"),
        .runup_body_norm_lo = declare_parameter<double>("path_planner.minco.runup.body_norm_lo"),
        .runup_body_norm_hi = declare_parameter<double>("path_planner.minco.runup.body_norm_hi"),
        .runup_saturation_length = declare_parameter<double>("path_planner.minco.runup.saturation_length"),
        .runup_transition_distance = declare_parameter<double>("path_planner.minco.runup.transition_distance"),
        .debug_check_gradient = enable_debug_,
        .debug_diagnostics = enable_debug_,
    };

    c.step_detection = {
        .detect_dot_threshold = declare_parameter<double>("path_planner.step.detection.detect_dot_threshold"),
        .path_sample_resolution = declare_parameter<double>("path_planner.step.detection.path_sample_resolution"),
        .profile_prepare_distance = declare_parameter<double>("path_planner.step.execution.profile_prepare_distance"),
        .chassis_activation_distance = declare_parameter<double>("path_planner.step.execution.chassis_activation_distance"),
        .fsm_release_distance = declare_parameter<double>("path_planner.step.execution.fsm_release_distance"),
        .gate_transition_distance = declare_parameter<double>("path_planner.step.mpc_constraints.gate_transition_distance"),
    };
    c.trajectory_validation = {
        .samples_per_segment = static_cast<int>(declare_parameter<int>("path_planner.minco.output_validation.samples_per_segment")),
        .velocity_tolerance = declare_parameter<double>("path_planner.minco.output_validation.trajectory_velocity_tolerance"),
        .omega_tolerance = declare_parameter<double>("path_planner.minco.output_validation.angular_velocity_tolerance"),
        .acceleration_tolerance = declare_parameter<double>("path_planner.minco.output_validation.acceleration_tolerance"),
        .lateral_acceleration_tolerance = declare_parameter<double>("path_planner.minco.output_validation.lateral_acceleration_tolerance"),
        .traversal_velocity_target_tolerance = declare_parameter<double>("path_planner.minco.output_validation.traversal_velocity_target_tolerance"),
        .traversal_angle_tolerance = declare_parameter<double>("path_planner.minco.output_validation.traversal_angle_tolerance"),
        .self_intersection_flatness_tolerance = declare_parameter<double>(
            "path_planner.minco.output_validation.self_intersection_flatness_tolerance"
        ),
        .self_intersection_max_edge_length = declare_parameter<double>(
            "path_planner.minco.output_validation.self_intersection_max_edge_length"
        ),
    };
    require_parameter(c.occupied_threshold >= 0 && c.occupied_threshold <= 255, "path_planner occupied_threshold must be in [0, 255]");
    require_parameter(
        c.on_step_threshold > 0.0 && c.on_step_threshold <= 1.0,
        "path_planner on_step_threshold must be finite and in (0, 1]"
    );
    require_parameter(positive_finite(c.seed_resample_distance), "seed_resample_distance must be finite and positive");
    require_parameter(
        positive_finite(c.kinodynamic.state_limits.speed_max)
        && positive_finite(c.kinodynamic.state_limits.angular_velocity_max)
        && positive_finite(c.kinodynamic.state_limits.acceleration_max)
        && positive_finite(c.kinodynamic.state_limits.lateral_acceleration_max)
        && positive_finite(c.kinodynamic.primitive_duration)
        && positive_finite(c.kinodynamic.dedup_xy)
        && positive_finite(c.kinodynamic.dedup_theta)
        && positive_finite(c.kinodynamic.dedup_speed),
        "kinodynamic limits, duration, and dedup resolutions must be finite and positive"
    );
    require_parameter(
        c.kinodynamic.tangential_accel_samples > 0
        && c.kinodynamic.normal_accel_samples > 0
        && c.kinodynamic.collision_substeps > 0 && c.kinodynamic.max_expansions > 0,
        "kinodynamic sample counts and max_expansions must be positive"
    );
    require_parameter(
        positive_finite(c.minco.trajectory_limits.velocity_max),
        "MINCO velocity_max must be finite and positive"
    );
    const auto& minco_weights = c.minco.weights;
    require_parameter(
        nonnegative_finite(minco_weights.energy)
        && nonnegative_finite(minco_weights.time)
        && nonnegative_finite(minco_weights.obstacle)
        && nonnegative_finite(minco_weights.trajectory_velocity)
        && nonnegative_finite(minco_weights.lateral_acc)
        && nonnegative_finite(minco_weights.omega)
        && nonnegative_finite(minco_weights.accel)
        && nonnegative_finite(minco_weights.traversal_alignment)
        && nonnegative_finite(minco_weights.traversal_velocity_target)
        && nonnegative_finite(minco_weights.prohibited_traversal)
        && nonnegative_finite(minco_weights.runup_accel)
        && nonnegative_finite(minco_weights.runup_omega),
        "MINCO penalty weights must be finite and non-negative"
    );
    require_parameter(
        positive_finite(c.minco.trajectory_limits.angular_velocity_max)
        && positive_finite(c.minco.trajectory_limits.acceleration_max)
        && positive_finite(c.minco.trajectory_limits.lateral_acceleration_max)
        && positive_finite(c.minco.min_segment_time),
        "MINCO limits and min_segment_time must be finite and positive"
    );
    require_parameter(
        c.minco.samples_per_segment > 0 && c.minco.max_iterations > 0,
        "MINCO sample and iteration counts must be positive"
    );
    const auto& minco_optimizer = c.minco.optimizer;
    const auto& trust_region = minco_optimizer.trust_region;
    require_parameter(
        positive_finite(minco_optimizer.position_scale)
        && positive_finite(minco_optimizer.physical_time_scale)
        && positive_finite(minco_optimizer.max_virtual_time_scale)
        && minco_optimizer.max_virtual_time_scale >= minco_optimizer.physical_time_scale,
        "MINCO optimizer variable scales must be finite, positive, and ordered"
    );
    require_parameter(
        minco_optimizer.max_function_evaluations > 0
        && minco_optimizer.history_size > 0
        && positive_finite(minco_optimizer.gradient_tolerance)
        && positive_finite(minco_optimizer.scaled_step_tolerance),
        "MINCO optimizer limits and convergence tolerances are invalid"
    );
    require_parameter(
        positive_finite(trust_region.min_radius)
        && positive_finite(trust_region.initial_radius)
        && positive_finite(trust_region.max_radius)
        && trust_region.min_radius <= trust_region.initial_radius
        && trust_region.initial_radius <= trust_region.max_radius,
        "MINCO optimizer trust-region radii are invalid"
    );
    require_parameter(
        std::isfinite(trust_region.acceptance_ratio)
        && std::isfinite(trust_region.shrink_ratio)
        && std::isfinite(trust_region.expansion_ratio)
        && trust_region.acceptance_ratio >= 0.0
        && trust_region.acceptance_ratio < trust_region.shrink_ratio
        && trust_region.shrink_ratio < trust_region.expansion_ratio
        && trust_region.expansion_ratio < 1.0,
        "MINCO optimizer trust-region reduction ratios are invalid"
    );
    require_parameter(
        positive_finite(trust_region.shrink_factor)
        && trust_region.shrink_factor < 1.0
        && std::isfinite(trust_region.expansion_factor)
        && trust_region.expansion_factor > 1.0
        && trust_region.max_consecutive_rejections > 0
        && trust_region.history_reset_after_rejections > 0
        && trust_region.history_reset_after_rejections
            < trust_region.max_consecutive_rejections,
        "MINCO optimizer trust-region factors or rejection limits are invalid"
    );
    require_parameter(
        positive_finite(minco_optimizer.curvature_relative_threshold)
        && minco_optimizer.curvature_relative_threshold < 1.0
        && std::isfinite(minco_optimizer.history_acceptance_ratio)
        && minco_optimizer.history_acceptance_ratio >= trust_region.acceptance_ratio
        && minco_optimizer.history_acceptance_ratio < 1.0,
        "MINCO optimizer curvature parameters are invalid"
    );
    require_parameter(
        c.minco.runup_body_norm_lo >= 0.9
        && c.minco.runup_body_norm_hi > c.minco.runup_body_norm_lo
        && c.minco.runup_body_norm_hi <= 1.0
        && positive_finite(c.minco.runup_saturation_length)
        && nonnegative_finite(c.minco.runup_transition_distance),
        "MINCO runup body gate or distance parameters are invalid"
    );
    require_parameter(
        c.minco.terrain_gate.norm_lo >= 0.0
        && c.minco.terrain_gate.norm_hi > c.minco.terrain_gate.norm_lo
        && c.minco.terrain_gate.norm_hi <= 1.0
        && positive_finite(c.minco.terrain_gate.motion_speed_scale),
        "MINCO terrain_gate or motion_speed_scale is invalid"
    );
    require_parameter(
        c.trajectory_validation.samples_per_segment > 0,
        "trajectory validation samples_per_segment must be positive"
    );
    require_parameter(
        c.step_detection.detect_dot_threshold > 0.0
        && c.step_detection.detect_dot_threshold <= 1.0
        && positive_finite(c.step_detection.path_sample_resolution)
        && nonnegative_finite(c.step_detection.profile_prepare_distance)
        && nonnegative_finite(c.step_detection.chassis_activation_distance)
        && nonnegative_finite(c.step_detection.fsm_release_distance)
        && nonnegative_finite(c.step_detection.gate_transition_distance),
        "step detection/execution parameters are invalid"
    );
    require_parameter(
        nonnegative_finite(c.trajectory_validation.velocity_tolerance)
        && nonnegative_finite(c.trajectory_validation.omega_tolerance)
        && nonnegative_finite(c.trajectory_validation.acceleration_tolerance)
        && nonnegative_finite(c.trajectory_validation.lateral_acceleration_tolerance)
        && nonnegative_finite(c.trajectory_validation.traversal_velocity_target_tolerance)
        && nonnegative_finite(c.trajectory_validation.traversal_angle_tolerance)
        && positive_finite(c.trajectory_validation.self_intersection_flatness_tolerance)
        && positive_finite(c.trajectory_validation.self_intersection_max_edge_length),
        "trajectory validation tolerances must be finite and non-negative"
    );

    require_parameter(
        positive_finite(c.forward_velocity_bounds.min)
            && c.forward_velocity_bounds.min < c.forward_velocity_bounds.max
            && c.forward_velocity_bounds.max <= c.kinodynamic.state_limits.speed_max
            && c.forward_velocity_bounds.max <= c.minco.trajectory_limits.velocity_max,
        "FOLLOW forward capability is incompatible with planner velocity limits"
    );
    const auto overlaps = [](const double speed_max, const TraversalVelocityWindow& window) {
        return window.min <= std::min(speed_max, window.max);
    };
    for (const DirectionalTraversalModes& label : traversal_configuration_.directional_labels) {
        for (const auto* modes : {&label.up, &label.down}) {
            for (const TraversalMode& mode : *modes) {
                if (!overlaps(c.kinodynamic.state_limits.speed_max, mode.velocity_window)) {
                    throw std::invalid_argument(
                        "kinodynamic state velocity limits cannot satisfy traversal mode \""
                        + mode.name + "\""
                    );
                }
                if (!overlaps(c.minco.trajectory_limits.velocity_max, mode.velocity_window)) {
                    throw std::invalid_argument(
                        "MINCO trajectory velocity limits cannot represent traversal target \""
                        + mode.name + "\""
                    );
                }
            }
        }
    }
    c.enable_debug = enable_debug_;
    return c;
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

    // 台阶 capability 仍只使用 low/medium/high；MPC 可额外选择独立双向 profile。
    const auto resolve_profile = [&](const std::string& param_key) -> CapabilityProfile {
        const std::string name = declare_parameter<std::string>(param_key);
        if (name == "bidirectional") return bidirectional_profile_;
        const auto level = capability_level_from_string(name);
        return traversal_configuration_.capability_profiles[static_cast<size_t>(level)];
    };

    MPCParams mpc_params = {
        .follow = {
            .start_command = {
                .vel_cmd_act_gap_max = declare_parameter<double>("mpc.follow.start_command.vel_cmd_act_gap_max"),
                .omega_cmd_act_gap_max = declare_parameter<double>("mpc.follow.start_command.omega_cmd_act_gap_max")
            },
            .normal_profile = resolve_profile("mpc.follow.capability"),
            .capability_profiles = {
                traversal_configuration_.capability_profiles[static_cast<size_t>(LOW)],
                traversal_configuration_.capability_profiles[static_cast<size_t>(MEDIUM)],
                traversal_configuration_.capability_profiles[static_cast<size_t>(HIGH)],
            },
            .tracking_weights = {
                .contour = declare_parameter<double>("mpc.follow.tracking_weights.contour"),
                .lag = declare_parameter<double>("mpc.follow.tracking_weights.lag"),
                .heading = declare_parameter<double>("mpc.follow.tracking_weights.heading"),
                .velocity = declare_parameter<double>("mpc.follow.tracking_weights.velocity"),
                .angular_velocity = declare_parameter<double>("mpc.follow.tracking_weights.angular_velocity"),
                .tangent_blend_speed_scale = declare_parameter<double>("mpc.follow.tracking_weights.tangent_blend_speed_scale"),
            },
            .command_weights = {
                .r_v = declare_parameter<double>("mpc.follow.command_weights.r_v"),
                .r_omega = declare_parameter<double>("mpc.follow.command_weights.r_omega"),
                .r_dv = declare_parameter<double>("mpc.follow.command_weights.r_dv"),
                .r_domega = declare_parameter<double>("mpc.follow.command_weights.r_domega"),
                .r_jerk_v = declare_parameter<double>("mpc.follow.command_weights.r_jerk_v"),
                .r_jerk_omega = declare_parameter<double>("mpc.follow.command_weights.r_jerk_omega")
            },
            .command_dynamics_weights = {
                .lateral_acceleration = declare_parameter<double>("mpc.follow.command_dynamics_weights.lateral_acceleration")
            },
            .traversal_target_weights = {
                .velocity = declare_parameter<double>("mpc.follow.traversal_target_weights.velocity"),
                .direction = declare_parameter<double>("mpc.follow.traversal_target_weights.direction"),
                .angular_velocity_command = declare_parameter<double>("mpc.follow.traversal_target_weights.angular_velocity_command"),
                .angular_velocity_predicted = declare_parameter<double>("mpc.follow.traversal_target_weights.angular_velocity_predicted"),
                .velocity_command_smoothness = declare_parameter<double>("mpc.follow.traversal_target_weights.velocity_command_smoothness"),
                .velocity_predicted_smoothness = declare_parameter<double>("mpc.follow.traversal_target_weights.velocity_predicted_smoothness"),
            },
            .environment_weights = {
                .obstacle = declare_parameter<double>("mpc.follow.environment_weights.obstacle")
            },
            .rollout_safety = {
                .enable_lethal_obstacle_check = declare_parameter<bool>("mpc.follow.rollout_safety.enable_lethal_obstacle_check"),
                .lethal_obstacle_threshold = declare_parameter<double>("mpc.follow.rollout_safety.lethal_obstacle_threshold"),
                .fddp_lethal_consecutive_threshold = static_cast<int>(declare_parameter<int>("mpc.follow.rollout_safety.fddp_lethal_consecutive_threshold"))
            },
            .ancillary_feedback = {
                .enable = declare_parameter<bool>("mpc.follow.ancillary_feedback.enable"),
                .velocity_error_gain = declare_parameter<double>("mpc.follow.ancillary_feedback.velocity_error_gain"),
                .command_error_gain = declare_parameter<double>("mpc.follow.ancillary_feedback.command_error_gain"),
                .velocity_error_reanchor_threshold = declare_parameter<double>("mpc.follow.ancillary_feedback.velocity_error_reanchor_threshold"),
                .velocity_command_margin = declare_parameter<double>("mpc.follow.ancillary_feedback.velocity_command_margin"),
                .velocity_command_rate_margin = declare_parameter<double>("mpc.follow.ancillary_feedback.velocity_command_rate_margin")
            },
            .progress = {
                .progress_reward = declare_parameter<double>("mpc.follow.progress.progress_reward"),
                .speed_tracking_weight = declare_parameter<double>("mpc.follow.progress.speed_tracking_weight"),
                .speed_smoothness_weight = declare_parameter<double>("mpc.follow.progress.speed_smoothness_weight"),
                .overshoot_weight = declare_parameter<double>("mpc.follow.progress.overshoot_weight"),
            },
            .terminal_weights = {
                .position = declare_parameter<double>("mpc.follow.terminal_weights.position"),
                .heading = declare_parameter<double>("mpc.follow.terminal_weights.heading"),
                .velocity = declare_parameter<double>("mpc.follow.terminal_weights.velocity"),
                .angular_velocity = declare_parameter<double>("mpc.follow.terminal_weights.angular_velocity"),
                .remaining_progress = declare_parameter<double>("mpc.follow.terminal_weights.remaining_progress"),
                .overshoot = declare_parameter<double>("mpc.follow.terminal_weights.overshoot"),
            },
            .max_iters = static_cast<int>(declare_parameter<int>("mpc.follow.max_iters"))
        },
        .stop = {
            .profile = resolve_profile("mpc.stop.capability"),
            .start_command = {
                .vel_cmd_act_gap_max = declare_parameter<double>("mpc.stop.start_command.vel_cmd_act_gap_max"),
                .omega_cmd_act_gap_max = declare_parameter<double>("mpc.stop.start_command.omega_cmd_act_gap_max")
            },
            .command_weights = {
                .r_v = declare_parameter<double>("mpc.stop.command_weights.r_v"),
                .r_omega = declare_parameter<double>("mpc.stop.command_weights.r_omega"),
                .r_dv = declare_parameter<double>("mpc.stop.command_weights.r_dv"),
                .r_domega = declare_parameter<double>("mpc.stop.command_weights.r_domega"),
                .r_jerk_v = declare_parameter<double>("mpc.stop.command_weights.r_jerk_v"),
                .r_jerk_omega = declare_parameter<double>("mpc.stop.command_weights.r_jerk_omega")
            },
            .command_dynamics_weights = {
                .lateral_acceleration = declare_parameter<double>("mpc.stop.command_dynamics_weights.lateral_acceleration")
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
            .profile = resolve_profile("mpc.hold.capability"),
            .start_command = {
                .vel_cmd_act_gap_max = declare_parameter<double>("mpc.hold.start_command.vel_cmd_act_gap_max"),
                .omega_cmd_act_gap_max = declare_parameter<double>("mpc.hold.start_command.omega_cmd_act_gap_max")
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
                .r_domega = declare_parameter<double>("mpc.hold.command_weights.r_domega"),
                .r_jerk_v = declare_parameter<double>("mpc.hold.command_weights.r_jerk_v"),
                .r_jerk_omega = declare_parameter<double>("mpc.hold.command_weights.r_jerk_omega")
            },
            .command_dynamics_weights = {
                .lateral_acceleration = declare_parameter<double>("mpc.hold.command_dynamics_weights.lateral_acceleration")
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
            .psi_bias = declare_parameter<double>("kinematic_model.psi_bias"),
            .psi_gain = declare_parameter<double>("kinematic_model.psi_gain"),
            .psi_v = declare_parameter<double>("kinematic_model.psi_v"),
            .obs_lv = declare_parameter<double>("kinematic_model.obs_lv"),
            .obs_v_correction_clip = declare_parameter<double>("kinematic_model.obs_v_correction_clip"),
            .obs_v_reset_threshold = declare_parameter<double>("kinematic_model.obs_v_reset_threshold")
        }
    };
    const auto& progress = mpc_params.follow.progress;
    require_parameter(
        nonnegative_finite(progress.progress_reward)
        && nonnegative_finite(progress.speed_tracking_weight)
        && nonnegative_finite(progress.speed_smoothness_weight)
        && positive_finite(progress.overshoot_weight),
        "mpc.follow progress weights are invalid"
    );
    const auto& tracking = mpc_params.follow.tracking_weights;
    require_parameter(
        nonnegative_finite(tracking.contour)
        && nonnegative_finite(tracking.lag)
        && nonnegative_finite(tracking.heading)
        && nonnegative_finite(tracking.velocity)
        && nonnegative_finite(tracking.angular_velocity)
        && positive_finite(tracking.tangent_blend_speed_scale),
        "mpc.follow tracking weights are invalid"
    );
    const auto& terminal = mpc_params.follow.terminal_weights;
    require_parameter(
        nonnegative_finite(terminal.position)
        && nonnegative_finite(terminal.heading)
        && nonnegative_finite(terminal.velocity)
        && nonnegative_finite(terminal.angular_velocity)
        && nonnegative_finite(terminal.remaining_progress)
        && positive_finite(terminal.overshoot),
        "mpc.follow terminal weights are invalid"
    );
    const auto& feedback = mpc_params.follow.ancillary_feedback;
    require_parameter(
        nonnegative_finite(feedback.velocity_error_gain)
        && nonnegative_finite(feedback.command_error_gain)
        && nonnegative_finite(feedback.velocity_error_reanchor_threshold)
        && nonnegative_finite(feedback.velocity_command_margin)
        && nonnegative_finite(feedback.velocity_command_rate_margin),
        "mpc.follow ancillary feedback parameters are invalid"
    );
    const auto valid_command_cost = [](const auto& command, const MPCCommandDynamicsWeights& dynamics) {
        return nonnegative_finite(command.r_v)
            && nonnegative_finite(command.r_omega)
            && nonnegative_finite(command.r_dv)
            && nonnegative_finite(command.r_domega)
            && nonnegative_finite(command.r_jerk_v)
            && nonnegative_finite(command.r_jerk_omega)
            && nonnegative_finite(dynamics.lateral_acceleration);
    };
    require_parameter(
        valid_command_cost(
            mpc_params.follow.command_weights,
            mpc_params.follow.command_dynamics_weights
        )
        && valid_command_cost(
            mpc_params.stop.command_weights,
            mpc_params.stop.command_dynamics_weights
        )
        && valid_command_cost(
            mpc_params.hold.command_weights,
            mpc_params.hold.command_dynamics_weights
        ),
        "mpc command weights are invalid"
    );

    const auto valid_capability = [](const CapabilityProfile& profile) {
        const auto& command = profile.command_envelope;
        const auto& dynamics = profile.command_dynamics;
        return std::isfinite(command.velocity.min)
            && std::isfinite(command.velocity.max)
            && command.velocity.min < command.velocity.max
            && std::isfinite(command.angular_velocity.min)
            && std::isfinite(command.angular_velocity.max)
            && command.angular_velocity.min < command.angular_velocity.max
            && positive_finite(dynamics.velocity_rate_max)
            && positive_finite(dynamics.angular_velocity_rate_max)
            && positive_finite(dynamics.lateral_acceleration_max);
    };
    require_parameter(
        valid_capability(mpc_params.follow.normal_profile)
            && mpc_params.follow.normal_profile.command_envelope.velocity.min > 0.0,
        "mpc.follow capability profile is invalid"
    );
    if (feedback.enable) {
        require_parameter(
            feedback.velocity_error_gain > 0.0
            && feedback.command_error_gain > 0.0
            && feedback.velocity_error_reanchor_threshold > 0.0
            && feedback.velocity_command_margin > 0.0
            && feedback.velocity_command_rate_margin > 0.0,
            "mpc.follow ancillary feedback requires positive gains and margins"
        );
        const double worst_velocity_correction = feedback.velocity_error_gain
            * feedback.velocity_error_reanchor_threshold;
        require_parameter(
            feedback.command_error_gain * feedback.velocity_command_margin
                > worst_velocity_correction
            && feedback.velocity_command_rate_margin >= worst_velocity_correction,
            "mpc.follow ancillary feedback lacks authority over its error tube"
        );
        for (const CapabilityProfile& profile : mpc_params.follow.capability_profiles) {
            require_parameter(
                2.0 * feedback.velocity_command_margin
                    < profile.command_envelope.velocity.max - profile.command_envelope.velocity.min
                && feedback.velocity_command_rate_margin < profile.command_dynamics.velocity_rate_max,
                "mpc.follow ancillary feedback margins exceed a capability profile"
            );
        }
    }
    require_parameter(
        valid_capability(mpc_params.stop.profile)
            && mpc_params.stop.profile.command_envelope.velocity.min < 0.0
            && mpc_params.stop.profile.command_envelope.velocity.max > 0.0,
        "mpc.stop capability profile is invalid"
    );
    require_parameter(
        valid_capability(mpc_params.hold.profile)
            && mpc_params.hold.profile.command_envelope.velocity.min < 0.0
            && mpc_params.hold.profile.command_envelope.velocity.max > 0.0,
        "mpc.hold capability profile is invalid"
    );
    const auto& model = mpc_params.kinematic_model;
    require_parameter(
        std::isfinite(model.z_ref)
        && positive_finite(model.z_scale)
        && positive_finite(model.rho_clip)
        && positive_finite(model.sgn_eps)
        && std::isfinite(model.ca00)
        && std::isfinite(model.ca01)
        && std::isfinite(model.ca10)
        && std::isfinite(model.ca11)
        && std::isfinite(model.cb0)
        && std::isfinite(model.cb1)
        && std::isfinite(model.dca00)
        && std::isfinite(model.dca01)
        && std::isfinite(model.dca10)
        && std::isfinite(model.dca11)
        && std::isfinite(model.dcb0)
        && std::isfinite(model.dcb1)
        && std::isfinite(model.gxh)
        && std::isfinite(model.gv)
        && std::isfinite(model.cf1)
        && std::isfinite(model.cf2)
        && std::isfinite(model.w_lam0)
        && std::isfinite(model.w_k0)
        && std::isfinite(model.w_cf0)
        && std::isfinite(model.w_lam1)
        && std::isfinite(model.w_k1)
        && std::isfinite(model.w_cf1)
        && std::isfinite(model.psi_bias)
        && std::isfinite(model.psi_gain)
        && std::isfinite(model.psi_v)
        && std::isfinite(model.obs_lv)
        && positive_finite(model.obs_v_correction_clip)
        && positive_finite(model.obs_v_reset_threshold)
        && model.obs_v_correction_clip < model.obs_v_reset_threshold,
        "kinematic model or observer parameters are invalid"
    );
    return mpc_params;
}

} // namespace nav_executor
