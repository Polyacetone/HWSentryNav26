#include <nav_executor/nav_executor_node.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
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
        debug_velocity_color_min_ = declare_parameter<double>(
            "node.debug.velocity_color.min"
        );
        debug_velocity_color_max_ = declare_parameter<double>(
            "node.debug.velocity_color.max"
        );
        require_parameter(
            std::isfinite(debug_velocity_color_min_)
                && std::isfinite(debug_velocity_color_max_)
                && debug_velocity_color_min_ < debug_velocity_color_max_,
            "node debug velocity color range must be finite and ordered"
        );
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        debug_mpc_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("node.debug.mpc_path_pub_topic"), 1);
        debug_minco_trajectory_pub_ = create_publisher<visualization_msgs::msg::Marker>(
            declare_parameter<std::string>("node.debug.minco_trajectory_pub_topic"), 1
        );
        debug_spatial_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("node.debug.spatial_path_pub_topic"), 1);
        debug_smoothed_spatial_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("node.debug.smoothed_spatial_path_pub_topic"), 1);
        debug_kino_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("node.debug.kino_path_pub_topic"), 1);
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

    const RouteTerrainMaskParams route_terrain_mask_params {
        .min_alignment_cosine = declare_parameter<double>(
            "path_executor.route_terrain_mask.min_alignment_cosine"
        ),
        .full_effect_radius = declare_parameter<double>(
            "path_executor.route_terrain_mask.full_effect_radius"
        ),
        .cutoff_radius = declare_parameter<double>(
            "path_executor.route_terrain_mask.cutoff_radius"
        ),
    };
    require_parameter(
        std::isfinite(route_terrain_mask_params.min_alignment_cosine)
            && route_terrain_mask_params.min_alignment_cosine >= 0.0
            && route_terrain_mask_params.min_alignment_cosine < 1.0
            && nonnegative_finite(route_terrain_mask_params.full_effect_radius)
            && positive_finite(route_terrain_mask_params.cutoff_radius)
            && route_terrain_mask_params.full_effect_radius
                < route_terrain_mask_params.cutoff_radius,
        "route terrain mask alignment and radii are invalid"
    );
    route_terrain_mask_ = std::make_shared<RouteTerrainMask>(
        route_terrain_mask_params
    );

    const PlannerConfig planner_config = load_planner_config(
        mpc_params.follow.normal_profile,
        mpc_params.follow.capability_profiles
    );
    obstacle_occupied_threshold_ = planner_config.occupied_cost_threshold;
    planner_ = std::make_unique<PathPlanner>(
        planner_config, route_terrain_mask_, get_logger().get_child("planner")
    );
    planner_->start();

    task_ = std::make_unique<TaskManager>(
        load_task_params(), planner_.get(), get_logger().get_child("task")
    );

    route_tracker_params_ = {
        .initial_search_distance = declare_parameter<double>("task_manager.route_tracker.initial_search_distance"),
        .max_tracking_error = declare_parameter<double>("task_manager.route_tracker.max_tracking_error"),
        .prediction_time_limit = declare_parameter<double>("task_manager.route_tracker.prediction_time_limit"),
        .hypothesis_spacing = declare_parameter<double>("task_manager.route_tracker.hypothesis_spacing"),
        .max_hypotheses = static_cast<int>(declare_parameter<int>("task_manager.route_tracker.max_hypotheses")),
        .hypothesis_prune_ratio = declare_parameter<double>("task_manager.route_tracker.hypothesis_prune_ratio"),
        .position_sigma = declare_parameter<double>("task_manager.route_tracker.position_sigma"),
        .velocity_sigma = declare_parameter<double>("task_manager.route_tracker.velocity_sigma"),
        .progress_sigma = declare_parameter<double>("task_manager.route_tracker.progress_sigma"),
        .profile_speed_sigma = declare_parameter<double>("task_manager.route_tracker.profile_speed_sigma"),
        .speed_dynamics_sigma = declare_parameter<double>("task_manager.route_tracker.speed_dynamics_sigma"),
        .max_path_speed = std::max({
            mpc_params.follow.normal_profile.command_envelope.velocity.max,
            mpc_params.follow.capability_profiles[0].command_envelope.velocity.max,
            mpc_params.follow.capability_profiles[1].command_envelope.velocity.max,
            mpc_params.follow.capability_profiles[2].command_envelope.velocity.max,
        }),
    };
    require_parameter(
        nonnegative_finite(route_tracker_params_.initial_search_distance)
        && positive_finite(route_tracker_params_.max_tracking_error)
        && positive_finite(route_tracker_params_.prediction_time_limit),
        "route_tracker search or prediction parameters are invalid"
    );
    require_parameter(
        positive_finite(route_tracker_params_.hypothesis_spacing)
        && route_tracker_params_.max_hypotheses > 0
        && positive_finite(route_tracker_params_.hypothesis_prune_ratio),
        "route_tracker hypothesis parameters are invalid"
    );
    require_parameter(
        positive_finite(route_tracker_params_.position_sigma)
        && positive_finite(route_tracker_params_.velocity_sigma)
        && positive_finite(route_tracker_params_.progress_sigma)
        && positive_finite(route_tracker_params_.profile_speed_sigma)
        && positive_finite(route_tracker_params_.speed_dynamics_sigma)
        && positive_finite(route_tracker_params_.max_path_speed),
        "route_tracker estimator scales and speed limit must be finite and positive"
    );
    route_tracker_ = std::make_unique<RouteTracker>(route_tracker_params_);
    proj_guard_params_ = {
        .cost_max = declare_parameter<double>("task_manager.route_monitor.proj_guard.cost_max"),
        .cost_samples = static_cast<int>(declare_parameter<int>("task_manager.route_monitor.proj_guard.cost_samples"))
    };
    step_block_params_ = {
        .enable = declare_parameter<bool>("task_manager.route_monitor.step_block.enable"),
        .lookahead_distance = declare_parameter<double>("task_manager.route_monitor.step_block.lookahead_distance"),
        .sample_resolution = declare_parameter<double>("task_manager.route_monitor.step_block.sample_resolution"),
        .predicted_obstacle_ratio_threshold = declare_parameter<double>("task_manager.route_monitor.step_block.predicted_obstacle_ratio_threshold")
    };
    require_parameter(
        nonnegative_finite(step_block_params_.lookahead_distance)
        && positive_finite(step_block_params_.sample_resolution)
        && step_block_params_.predicted_obstacle_ratio_threshold >= 0.0
        && step_block_params_.predicted_obstacle_ratio_threshold <= 1.0,
        "route_monitor step_block parameters are invalid"
    );
    performance_replan_params_ = {
        .lookahead_distance = declare_parameter<double>("task_manager.route_monitor.performance.lookahead_distance")
    };

    remaining_energy_filter_alpha_ = declare_parameter<double>("node.remaining_energy_filter_alpha");
    const double chassis_status_timeout_seconds = declare_parameter<double>("node.chassis_status_timeout_seconds");
    require_parameter(
        positive_finite(chassis_status_timeout_seconds),
        "chassis_status_timeout_seconds must be finite and positive"
    );
    chassis_status_timeout_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(chassis_status_timeout_seconds)
    );
    path_publish_sample_resolution_ = declare_parameter<double>("node.path_publish_sample_resolution");
    dynamic_prediction_horizon_seconds_ = declare_parameter<double>("path_planner.dynamic_prediction.horizon_seconds");
    require_parameter(positive_finite(path_publish_sample_resolution_), "path publish sample_resolution must be finite and positive");
    require_parameter(nonnegative_finite(dynamic_prediction_horizon_seconds_), "dynamic prediction horizon_seconds must be finite and non-negative");

    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("node.topics.global_cost_map_sub"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            try {
                global_cost_map_ = std::make_shared<CostMap>(*msg);
            } catch (const std::exception& error) {
                RCLCPP_ERROR(
                    get_logger(), "Rejected invalid global cost map geometry: %s",
                    error.what()
                );
                return;
            }
            RCLCPP_INFO(get_logger(), "Received global cost map: (%d,%d) res=%.2f",
                global_cost_map_->geometry.width(), global_cost_map_->geometry.height(),
                global_cost_map_->geometry.resolution());
            try_init_route_terrain_mask();
            refresh_planner_obstacles();
            global_cost_map_sub_.reset();
        }
    );

    global_direction_map_sub_ = create_subscription<sensor_msgs::msg::Image>(
        declare_parameter<std::string>("node.topics.global_direction_map_sub"), 1,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            if (!global_cost_map_) {
                record_input_rejection(interfaces::msg::NavExecutorDiag::INPUT_REJECTION_DIRECTION_MAP_BEFORE_GLOBAL_MAP);
                RCLCPP_WARN(get_logger(), "Received direction map before cost map; ignoring");
                return;
            }
            const cv::Mat img = cv_bridge::toCvShare(msg, "8UC3")->image;
            if (img.cols != global_cost_map_->geometry.width()
                || img.rows != global_cost_map_->geometry.height()) {
                record_input_rejection(
                    interfaces::msg::NavExecutorDiag::INPUT_REJECTION_DIRECTION_MAP_SIZE_MISMATCH
                );
                RCLCPP_ERROR(get_logger(), "Direction map size mismatch with cost map");
                return;
            }
            global_direction_map_ = std::make_shared<DirectionMap>(
                img, global_cost_map_->geometry
            );
            RCLCPP_INFO(get_logger(), "Received global direction map");
            try_init_route_terrain_mask();
            refresh_planner_obstacles();
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

    idle_chassis_mode_override_sub_ = create_subscription<interfaces::msg::IdleChassisModeOverride>(
        declare_parameter<std::string>("node.topics.idle_chassis_mode_override_sub"), 1,
        [this](const interfaces::msg::IdleChassisModeOverride::SharedPtr msg) {
            const uint8_t mode = msg->mode_override;
            if (mode != 0 && (mode < 200 || mode > 207)) {
                record_input_rejection(interfaces::msg::NavExecutorDiag::INPUT_REJECTION_IDLE_MODE_INVALID);
                RCLCPP_ERROR(
                    get_logger(),
                    "Ignoring invalid IDLE chassis mode override: %u (expected 0 or 200-207)",
                    static_cast<unsigned int>(mode)
                );
                return;
            }
            idle_chassis_mode_override_ = mode;
        }
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

void NavExecutorNode::try_init_route_terrain_mask() {
    if (route_terrain_mask_ready_ || !global_cost_map_ || !global_direction_map_) return;
    try {
        route_terrain_mask_->initialize(*global_cost_map_, global_direction_map_);
        route_terrain_mask_ready_ = true;
        RCLCPP_INFO(get_logger(), "RouteTerrainMask initialized");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to initialize RouteTerrainMask: %s", e.what());
    }
}

void NavExecutorNode::refresh_planner_obstacles() {
    try {
        planner_obstacles_ = build_planner_obstacle_view(
            ObstacleLayers {
                .global_static = global_cost_map_,
                .dynamic_current = current_cost_map_,
                .dynamic_predictions = prediction_maps_,
                .base_direction = global_direction_map_,
            },
            prediction_dt_, dynamic_prediction_horizon_seconds_
        );
    } catch (const std::exception& error) {
        planner_obstacles_ = {};
        RCLCPP_ERROR(get_logger(), "Failed to build planner obstacle view: %s", error.what());
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
            valid_capability(profile) && profile.command_envelope.velocity.min == 0.0,
            "forward capability profile must have a zero velocity lower bound"
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
    fsm.prepare_spin = {
        .command_velocity_max = declare_parameter<double>("path_executor.state_machine.prepare_spin.command_velocity_max"),
        .measured_velocity_max = declare_parameter<double>("path_executor.state_machine.prepare_spin.measured_velocity_max"),
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
    require_parameter(
        positive_finite(fsm.prepare_spin.command_velocity_max)
        && positive_finite(fsm.prepare_spin.measured_velocity_max),
        "PREPARE_SPIN command and measured velocity thresholds must be finite and positive"
    );
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
    const CapabilityProfile& normal_profile,
    const std::array<CapabilityProfile, 3>& step_profiles
) {
    PlannerConfig c;
    c.occupied_cost_threshold = static_cast<int>(declare_parameter<int>(
        "path_planner.endpoint_handling.occupied_cost_threshold"
    ));
    c.endpoint_direction_norm_max = declare_parameter<double>(
        "path_planner.endpoint_handling.direction_norm_max"
    );
    c.endpoint_nudge_max_distance = declare_parameter<double>(
        "path_planner.endpoint_handling.nudge_max_distance"
    );
    c.endpoint_goal_reached_distance = declare_parameter<double>(
        "path_planner.endpoint_handling.goal_reached_distance"
    );
    c.minco_seed = {
        .max_point_spacing = declare_parameter<double>(
            "path_planner.minco.seed.max_point_spacing"
        ),
        .max_heading_change = declare_parameter<double>(
            "path_planner.minco.seed.max_heading_change"
        ),
    };
    c.directional_terrain.min_alignment_cosine = declare_parameter<double>(
        "path_planner.directional_terrain.min_alignment_cosine"
    );
    const KinoAStar::Params::StartYawRelaxationParams start_yaw_relaxation {
        .root_count = static_cast<int>(declare_parameter<int>(
            "path_planner.kino_a_star.start_yaw_relaxation.root_count"
        )),
        .root_bias_seconds = declare_parameter<double>(
            "path_planner.kino_a_star.start_yaw_relaxation.root_bias_seconds"
        ),
        .yaw_bias_seconds_per_rad = declare_parameter<double>(
            "path_planner.kino_a_star.start_yaw_relaxation.yaw_bias_seconds_per_rad"
        ),
        .max_discarded_velocity = declare_parameter<double>(
            "path_planner.kino_a_star.start_yaw_relaxation.max_discarded_velocity"
        ),
    };
    const ShapingDynamicsLimits shaping_dynamics {
        .velocity_max = declare_parameter<double>("path_planner.search_dynamics.velocity_max"),
        .tangential_acceleration_max = declare_parameter<double>("path_planner.search_dynamics.tangential_acceleration_max"),
        .angular_velocity_max = declare_parameter<double>("path_planner.search_dynamics.angular_velocity_max"),
        .angular_acceleration_max = declare_parameter<double>("path_planner.search_dynamics.angular_acceleration_max"),
        .lateral_acceleration_max = declare_parameter<double>("path_planner.search_dynamics.lateral_acceleration_max"),
    };
    const GeometryLimits geometry_limits {
        .curvature_max = declare_parameter<double>("path_planner.geometry.curvature_max"),
        .curvature_rate_max = declare_parameter<double>("path_planner.geometry.curvature_rate_max"),
        .tangent_regularization = declare_parameter<double>(
            "path_planner.geometry.tangent_regularization"
        ),
    };
    c.motion_primitives.xy_resolution = declare_parameter<double>(
        "path_planner.kino_a_star.lattice.xy_resolution"
    );
    c.motion_primitives.heading_bins = static_cast<int>(declare_parameter<int>(
        "path_planner.kino_a_star.lattice.heading_bins"
    ));
    const std::vector<double> curvature_magnitudes = declare_parameter<std::vector<double>>(
        "path_planner.kino_a_star.motion_primitives.curvature_magnitudes"
    );
    const std::vector<double> band_lengths = declare_parameter<std::vector<double>>(
        "path_planner.kino_a_star.motion_primitives.band_lengths"
    );
    require_parameter(
        curvature_magnitudes.size() == c.motion_primitives.curvature_magnitudes.size()
            && band_lengths.size() == c.motion_primitives.band_lengths.size(),
        "motion primitive curvature_magnitudes or band_lengths has an invalid size"
    );
    std::ranges::copy(
        curvature_magnitudes, c.motion_primitives.curvature_magnitudes.begin()
    );
    std::ranges::copy(band_lengths, c.motion_primitives.band_lengths.begin());
    c.motion_primitives.straight_length = declare_parameter<double>(
        "path_planner.kino_a_star.motion_primitives.straight_length"
    );
    c.motion_primitives.curvature_max = geometry_limits.curvature_max;
    c.spatial_a_star = {
        .obstacle_weight = declare_parameter<double>(
            "path_planner.spatial_a_star.obstacle_weight"
        ),
        .max_expansions = static_cast<int>(declare_parameter<int>(
            "path_planner.spatial_a_star.max_expansions"
        )),
    };
    const double runup_transition_distance = declare_parameter<double>(
        "path_planner.directional_terrain.runup_transition_distance"
    );
    c.reference_path = {
        .resample_spacing = declare_parameter<double>(
            "path_planner.reference_path.resample_spacing"
        ),
        .tangent_lookahead = declare_parameter<double>(
            "path_planner.reference_path.tangent_lookahead"
        ),
        .runup_transition_distance = runup_transition_distance,
    };
    c.guide_field = {
        .corridor_width = declare_parameter<double>(
            "path_planner.guide_field.corridor_width"
        ),
        .start_bulb_radius = declare_parameter<double>(
            "path_planner.guide_field.start_bulb_radius"
        ),
    };
    c.kino_a_star = {
        .start_yaw_relaxation = start_yaw_relaxation,
        .dynamics = shaping_dynamics,
        .speed_bin_count = static_cast<int>(declare_parameter<int>(
            "path_planner.kino_a_star.lattice.speed_bin_count"
        )),
        .collision_check_resolution = declare_parameter<double>("path_planner.kino_a_star.collision_check_resolution"),
        .goal_connection_max_distance = declare_parameter<double>("path_planner.kino_a_star.goal_connection_max_distance"),
        .guidance_weight = declare_parameter<double>("path_planner.kino_a_star.guidance_weight"),
        .deviation_weight = declare_parameter<double>("path_planner.kino_a_star.deviation_weight"),
        .heading_weight = declare_parameter<double>("path_planner.kino_a_star.heading_weight"),
        .speed_weight = declare_parameter<double>("path_planner.kino_a_star.speed_weight"),
        .approach_alignment_weight = declare_parameter<double>("path_planner.kino_a_star.approach_alignment_weight"),
        .approach_window_weight = declare_parameter<double>("path_planner.kino_a_star.approach_window_weight"),
        .max_expansions = static_cast<int>(declare_parameter<int>("path_planner.kino_a_star.max_expansions")),
    };
    c.minco = {
        .weights = {
            .energy = declare_parameter<double>("path_planner.minco.penalty_weights.energy"),
            .time = declare_parameter<double>("path_planner.minco.penalty_weights.time"),
            .obstacle = declare_parameter<double>("path_planner.minco.penalty_weights.obstacle"),
            .curvature = declare_parameter<double>("path_planner.minco.penalty_weights.curvature"),
            .curvature_rate = declare_parameter<double>("path_planner.minco.penalty_weights.curvature_rate"),
            .directed_regularity = declare_parameter<double>("path_planner.minco.penalty_weights.directed_regularity"),
            .traversal_velocity_window = declare_parameter<double>("path_planner.minco.penalty_weights.traversal_velocity_window"),
            .traversal_alignment = declare_parameter<double>("path_planner.minco.penalty_weights.traversal_alignment"),
            .prohibited_traversal = declare_parameter<double>("path_planner.minco.penalty_weights.prohibited_traversal"),
            .runup_curvature = declare_parameter<double>("path_planner.minco.penalty_weights.runup_curvature"),
        },
        .geometry = geometry_limits,
        .directed_cosine_min = declare_parameter<double>("path_planner.minco.directed_cosine_min"),
        .terrain_gate = {
            .norm_lo = declare_parameter<double>("path_planner.minco.terrain_gate.norm_lo"),
            .norm_hi = declare_parameter<double>("path_planner.minco.terrain_gate.norm_hi"),
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
            .cost_window_relative_tolerance = declare_parameter<double>("path_planner.minco.optimizer.cost_window_relative_tolerance"),
            .cost_window_size = static_cast<int>(declare_parameter<int>("path_planner.minco.optimizer.cost_window_size")),
            .cost_plateau_gradient_tolerance = declare_parameter<double>("path_planner.minco.optimizer.cost_plateau_gradient_tolerance"),
            .scaled_step_tolerance = declare_parameter<double>("path_planner.minco.optimizer.scaled_step_tolerance"),
            .step_control = {
                .initial_step_cap = declare_parameter<double>("path_planner.minco.optimizer.step_control.initial_step_cap"),
                .min_step_cap = declare_parameter<double>("path_planner.minco.optimizer.step_control.min_step_cap"),
                .max_step_cap = declare_parameter<double>("path_planner.minco.optimizer.step_control.max_step_cap"),
                .expansion_min_model_ratio = declare_parameter<double>("path_planner.minco.optimizer.step_control.expansion_min_model_ratio"),
                .backtrack_factor = declare_parameter<double>("path_planner.minco.optimizer.step_control.backtrack_factor"),
                .expansion_factor = declare_parameter<double>("path_planner.minco.optimizer.step_control.expansion_factor"),
                .max_rejections_per_iteration = static_cast<int>(declare_parameter<int>("path_planner.minco.optimizer.step_control.max_rejections_per_iteration")),
                .recovery_after_rejections = static_cast<int>(declare_parameter<int>("path_planner.minco.optimizer.step_control.recovery_after_rejections")),
            },
            .curvature_cosine_threshold = declare_parameter<double>("path_planner.minco.optimizer.curvature_cosine_threshold"),
            .history_update_min_model_ratio = declare_parameter<double>("path_planner.minco.optimizer.history_update_min_model_ratio"),
        },
        .min_segment_time = declare_parameter<double>("path_planner.minco.min_segment_time"),
        .runup_body_norm_lo = declare_parameter<double>("path_planner.minco.runup.body_norm_lo"),
        .runup_body_norm_hi = declare_parameter<double>("path_planner.minco.runup.body_norm_hi"),
        .runup_saturation_length = declare_parameter<double>("path_planner.minco.runup.saturation_length"),
        .runup_transition_distance = runup_transition_distance,
        .debug_diagnostics = enable_debug_,
    };

    c.traversal_annotation = {
        .sample_spacing = declare_parameter<double>(
            "path_planner.traversal_annotation.sample_spacing"
        ),
    };
    c.step_execution_timing = {
        .profile_prepare_distance = declare_parameter<double>(
            "path_executor.traversal_execution.profile_prepare_distance"
        ),
        .chassis_activation_distance = declare_parameter<double>(
            "path_executor.traversal_execution.chassis_activation_distance"
        ),
        .fsm_release_distance = declare_parameter<double>(
            "path_executor.traversal_execution.fsm_release_distance"
        ),
    };
    c.traversal_constraint_gate = {
        .gate_transition_distance = declare_parameter<double>(
            "mpc.follow.traversal_constraints.gate_transition_distance"
        ),
    };
    c.trajectory_validation = {
        .samples_per_segment = static_cast<int>(declare_parameter<int>("path_planner.trajectory_validation.samples_per_segment")),
        .alignment_warning_angle = declare_parameter<double>("path_planner.trajectory_validation.alignment_warning_angle"),
    };
    c.speed_profile = {
        .discretization = {
            .max_spacing = declare_parameter<double>("path_planner.speed_profile.discretization.max_spacing"),
            .traversal_max_spacing = declare_parameter<double>("path_planner.speed_profile.discretization.traversal_max_spacing"),
            .curvature_refine_threshold = declare_parameter<double>("path_planner.speed_profile.discretization.curvature_refine_threshold"),
            .envelope_sample_spacing = declare_parameter<double>("path_planner.speed_profile.discretization.envelope_sample_spacing"),
        },
        .objective = {
            .traversal_window = declare_parameter<double>("path_planner.speed_profile.objective.traversal_window"),
            .lateral_acceleration = declare_parameter<double>("path_planner.speed_profile.objective.lateral_acceleration"),
            .global_speed_reward = declare_parameter<double>("path_planner.speed_profile.objective.global_speed_reward"),
            .velocity_scale = declare_parameter<double>("path_planner.speed_profile.objective.velocity_scale"),
        },
        .stationary_velocity_threshold = declare_parameter<double>(
            "path_planner.speed_profile.stationary_velocity_threshold"
        ),
        .geometry = geometry_limits,
        .normal_profile = normal_profile,
        .step_profiles = step_profiles,
    };
    require_parameter(
        c.occupied_cost_threshold > 0 && c.occupied_cost_threshold <= 255,
        "planner occupied cost threshold must be in (0, 255]"
    );
    require_parameter(
        positive_finite(c.endpoint_direction_norm_max)
            && c.endpoint_direction_norm_max <= 1.0
            && nonnegative_finite(c.endpoint_nudge_max_distance)
            && nonnegative_finite(c.endpoint_goal_reached_distance),
        "planner endpoint handling parameters are invalid"
    );
    require_parameter(
        positive_finite(c.minco_seed.max_point_spacing)
            && positive_finite(c.minco_seed.max_heading_change)
            && c.minco_seed.max_heading_change <= std::numbers::pi,
        "MINCO seed spacing or heading threshold is invalid"
    );
    require_parameter(
        std::isfinite(c.directional_terrain.min_alignment_cosine)
            && c.directional_terrain.min_alignment_cosine > 0.0
            && c.directional_terrain.min_alignment_cosine < 1.0,
        "directional terrain minimum alignment cosine must be in (0, 1)"
    );
    require_parameter(
        c.kino_a_star.start_yaw_relaxation.root_count >= 1
            && c.kino_a_star.start_yaw_relaxation.root_count
                <= c.motion_primitives.heading_bins
            && nonnegative_finite(
                c.kino_a_star.start_yaw_relaxation.root_bias_seconds
            )
            && nonnegative_finite(
                c.kino_a_star.start_yaw_relaxation.yaw_bias_seconds_per_rad
            )
            && nonnegative_finite(
                c.kino_a_star.start_yaw_relaxation.max_discarded_velocity
            ),
        "start yaw relaxation parameters are invalid"
    );
    require_parameter(
        positive_finite(shaping_dynamics.velocity_max)
        && positive_finite(shaping_dynamics.tangential_acceleration_max)
        && positive_finite(shaping_dynamics.angular_velocity_max)
        && positive_finite(shaping_dynamics.angular_acceleration_max)
        && positive_finite(shaping_dynamics.lateral_acceleration_max)
        && positive_finite(c.motion_primitives.xy_resolution)
        && positive_finite(geometry_limits.curvature_max)
        && positive_finite(geometry_limits.curvature_rate_max)
        && positive_finite(geometry_limits.tangent_regularization)
        && positive_finite(c.motion_primitives.straight_length)
        && nonnegative_finite(c.spatial_a_star.obstacle_weight)
        && positive_finite(c.reference_path.resample_spacing)
        && positive_finite(c.reference_path.tangent_lookahead)
        && positive_finite(c.guide_field.corridor_width)
        && positive_finite(c.guide_field.start_bulb_radius)
        && c.guide_field.start_bulb_radius >= c.guide_field.corridor_width
        && positive_finite(c.kino_a_star.collision_check_resolution)
        && positive_finite(c.kino_a_star.goal_connection_max_distance)
        && nonnegative_finite(c.kino_a_star.guidance_weight)
        && nonnegative_finite(c.kino_a_star.deviation_weight)
        && nonnegative_finite(c.kino_a_star.heading_weight)
        && nonnegative_finite(c.kino_a_star.speed_weight)
        && nonnegative_finite(c.kino_a_star.approach_alignment_weight)
        && nonnegative_finite(c.kino_a_star.approach_window_weight)
        && nonnegative_finite(c.reference_path.runup_transition_distance),
        "spatial/reference/corridor geometry, dynamics, and guidance weights must be finite and valid"
    );
    require_parameter(
        std::ranges::all_of(
            c.motion_primitives.curvature_magnitudes,
            [&](const double value) {
                return positive_finite(value)
                    && value <= c.motion_primitives.curvature_max;
            }
        ) && std::ranges::all_of(
            c.motion_primitives.band_lengths, positive_finite
        ),
        "motion primitive bands must be finite, positive, and within curvature_max"
    );
    require_parameter(
        c.motion_primitives.heading_bins > 0
        && c.spatial_a_star.max_expansions > 0
        && c.kino_a_star.speed_bin_count >= 2
        && c.kino_a_star.max_expansions > 0,
        "lattice pose/speed bins and expansion limit must be valid"
    );
    require_parameter(
        std::isfinite(c.minco.directed_cosine_min)
        && c.minco.directed_cosine_min > 0.0
        && c.minco.directed_cosine_min < 1.0,
        "MINCO directed_cosine_min must be in (0, 1)"
    );
    const auto& minco_weights = c.minco.weights;
    require_parameter(
        nonnegative_finite(minco_weights.energy)
        && nonnegative_finite(minco_weights.time)
        && nonnegative_finite(minco_weights.obstacle)
        && nonnegative_finite(minco_weights.curvature)
        && nonnegative_finite(minco_weights.curvature_rate)
        && nonnegative_finite(minco_weights.directed_regularity)
        && nonnegative_finite(minco_weights.traversal_velocity_window)
        && nonnegative_finite(minco_weights.traversal_alignment)
        && nonnegative_finite(minco_weights.prohibited_traversal)
        && nonnegative_finite(minco_weights.runup_curvature),
        "MINCO penalty weights must be finite and non-negative"
    );
    require_parameter(
        positive_finite(c.minco.min_segment_time),
        "MINCO min_segment_time must be finite and positive"
    );
    require_parameter(
        c.minco.samples_per_segment > 0 && c.minco.max_iterations > 0,
        "MINCO sample and iteration counts must be positive"
    );
    const auto& minco_optimizer = c.minco.optimizer;
    const auto& step_control = minco_optimizer.step_control;
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
        && positive_finite(minco_optimizer.cost_window_relative_tolerance)
        && minco_optimizer.cost_window_size > 0
        && positive_finite(minco_optimizer.cost_plateau_gradient_tolerance)
        && minco_optimizer.cost_plateau_gradient_tolerance
            >= minco_optimizer.gradient_tolerance
        && positive_finite(minco_optimizer.scaled_step_tolerance),
        "MINCO optimizer limits and convergence tolerances are invalid"
    );
    require_parameter(
        positive_finite(step_control.min_step_cap)
        && positive_finite(step_control.initial_step_cap)
        && positive_finite(step_control.max_step_cap)
        && step_control.min_step_cap <= step_control.initial_step_cap
        && step_control.initial_step_cap <= step_control.max_step_cap,
        "MINCO optimizer step caps are invalid"
    );
    require_parameter(
        std::isfinite(step_control.expansion_min_model_ratio)
        && step_control.expansion_min_model_ratio > 0.0
        && step_control.expansion_min_model_ratio < 1.0
        && positive_finite(step_control.backtrack_factor)
        && step_control.backtrack_factor < 1.0
        && std::isfinite(step_control.expansion_factor)
        && step_control.expansion_factor > 1.0
        && step_control.max_rejections_per_iteration > 0
        && step_control.recovery_after_rejections > 0
        && step_control.recovery_after_rejections
            < step_control.max_rejections_per_iteration,
        "MINCO optimizer step-control factors or rejection limits are invalid"
    );
    require_parameter(
        positive_finite(minco_optimizer.curvature_cosine_threshold)
        && minco_optimizer.curvature_cosine_threshold < 1.0
        && std::isfinite(minco_optimizer.history_update_min_model_ratio)
        && minco_optimizer.history_update_min_model_ratio >= 0.0
        && minco_optimizer.history_update_min_model_ratio < 1.0,
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
        && c.minco.terrain_gate.norm_hi <= 1.0,
        "MINCO terrain_gate thresholds are invalid"
    );
    require_parameter(
        positive_finite(c.traversal_annotation.sample_spacing)
            && nonnegative_finite(c.step_execution_timing.profile_prepare_distance)
            && nonnegative_finite(c.step_execution_timing.chassis_activation_distance)
            && c.step_execution_timing.profile_prepare_distance
                >= c.step_execution_timing.chassis_activation_distance
            && nonnegative_finite(c.step_execution_timing.fsm_release_distance)
            && nonnegative_finite(
                c.traversal_constraint_gate.gate_transition_distance
            ),
        "traversal annotation, execution timing, or constraint gate is invalid"
    );
    require_parameter(
        c.trajectory_validation.samples_per_segment > 0
        && nonnegative_finite(c.trajectory_validation.alignment_warning_angle)
        && c.trajectory_validation.alignment_warning_angle
            <= std::numbers::pi,
        "environment validation parameters are invalid"
    );
    const auto& speed_profile = c.speed_profile;
    require_parameter(
        positive_finite(speed_profile.discretization.max_spacing)
        && positive_finite(speed_profile.discretization.traversal_max_spacing)
        && speed_profile.discretization.traversal_max_spacing
            <= speed_profile.discretization.max_spacing
        && positive_finite(speed_profile.discretization.curvature_refine_threshold)
        && positive_finite(speed_profile.discretization.envelope_sample_spacing)
        && nonnegative_finite(speed_profile.objective.traversal_window)
        && nonnegative_finite(speed_profile.objective.lateral_acceleration)
        && nonnegative_finite(speed_profile.objective.global_speed_reward)
        && positive_finite(speed_profile.objective.velocity_scale)
        && nonnegative_finite(speed_profile.stationary_velocity_threshold),
        "speed profile discretization, objective, or stationary threshold is invalid"
    );
    const auto overlaps = [](const double speed_max, const TraversalVelocityWindow& window) {
        return window.min <= std::min(speed_max, window.max);
    };
    for (const DirectionalTraversalModes& label : traversal_configuration_.directional_labels) {
        for (const auto* modes : {&label.up, &label.down}) {
            for (const TraversalMode& mode : *modes) {
                if (!overlaps(shaping_dynamics.velocity_max, mode.velocity_window)) {
                    throw std::invalid_argument(
                        "shaping velocity limit cannot represent traversal witness target \""
                        + mode.name + "\""
                    );
                }
                const CapabilityProfile& profile = step_profiles[
                    static_cast<size_t>(mode.capability)
                ];
                if (!overlaps(profile.command_envelope.velocity.max, mode.velocity_window)) {
                    throw std::invalid_argument(
                        "traversal capability cannot represent velocity target \""
                        + mode.name + "\""
                    );
                }
            }
        }
    }
    c.enable_diagnostics = enable_debug_;
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
                .lethal_replan_consecutive_threshold = static_cast<int>(declare_parameter<int>("mpc.follow.rollout_safety.lethal_replan_consecutive_threshold"))
            },
            .reference_seed = {
                .lookahead_time = declare_parameter<double>("mpc.follow.reference_seed.lookahead_time"),
                .lookahead_distance_min = declare_parameter<double>("mpc.follow.reference_seed.lookahead_distance_min"),
                .lookahead_distance_max = declare_parameter<double>("mpc.follow.reference_seed.lookahead_distance_max")
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
        && nonnegative_finite(progress.speed_smoothness_weight),
        "mpc.follow progress weights are invalid"
    );
    const auto& tracking = mpc_params.follow.tracking_weights;
    require_parameter(
        nonnegative_finite(tracking.contour)
        && nonnegative_finite(tracking.lag)
        && nonnegative_finite(tracking.heading)
        && nonnegative_finite(tracking.velocity)
        && nonnegative_finite(tracking.angular_velocity),
        "mpc.follow tracking weights are invalid"
    );
    const auto& rollout_safety = mpc_params.follow.rollout_safety;
    require_parameter(
        rollout_safety.lethal_obstacle_threshold > 0.0
        && rollout_safety.lethal_obstacle_threshold <= 255.0
        && rollout_safety.lethal_replan_consecutive_threshold > 0,
        "mpc.follow rollout safety parameters are invalid"
    );
    const auto& reference_seed = mpc_params.follow.reference_seed;
    require_parameter(
        positive_finite(reference_seed.lookahead_time)
        && positive_finite(reference_seed.lookahead_distance_min)
        && reference_seed.lookahead_distance_min <= reference_seed.lookahead_distance_max
        && std::isfinite(reference_seed.lookahead_distance_max),
        "mpc.follow reference seed parameters are invalid"
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
            && mpc_params.follow.normal_profile.command_envelope.velocity.min == 0.0,
        "mpc.follow capability profile must have a zero velocity lower bound"
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
