#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float64.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <interfaces/msg/chassis_status.hpp>
#include <interfaces/msg/spin_cmd.hpp>
#include <interfaces/msg/cost_maps.hpp>
#include <interfaces/msg/chassis_cmd.hpp>
#include <interfaces/msg/comp_stage.hpp>
#include <interfaces/msg/global_path.hpp>
#include <interfaces/msg/follower_state.hpp>

#include <uniform_bspline/uniform_bspline.hpp>
#include <common_utils/convert.hpp>
#include <path_follower/nav_map.hpp>
#include <path_follower/main_controller.hpp>
#include <path_follower/mpc_solver.hpp>
#include <path_follower/step_routing_mask.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

class PathFollowerNode : public rclcpp::Node {
public:
    explicit PathFollowerNode(const rclcpp::NodeOptions& options);

private:
    // ─── ROS 回调 ───
    void control_points_callback(const interfaces::msg::GlobalPath::SharedPtr msg);
    void chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg);
    void local_cost_maps_callback(const interfaces::msg::CostMaps::SharedPtr msg);
    void spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg);
    void control_timer_callback();

    // ─── 工具函数 ───
    bool get_chassis_pose(Eigen::Vector3d& chassis_pose) const;
    std::optional<double> get_map_to_imu_world_yaw() const;
    std::optional<double> get_chassis_theta_imu_world() const;
    void publish_chassis_cmd(const ControlOutput& output);
    nav_msgs::msg::Path path_to_nav_msg(const std::vector<Eigen::Vector2d>& points) const;
    void publish_mppi_rollouts(const std::vector<std::vector<Eigen::Vector2d>>& rollouts);
    bool should_use_prediction_maps() const;
    bool is_step_routing_context_locked() const;
    void refresh_deferred_step_layers();

    // ─── 台阶掩码层更新 ───
    void update_step_layers();
    void update_masked_cost_maps();

    // ─── ROS 通信 ───
    rclcpp::Subscription<interfaces::msg::GlobalPath>::SharedPtr control_points_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_sub_;
    rclcpp::Subscription<interfaces::msg::CostMaps>::SharedPtr local_cost_maps_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr global_direction_map_sub_;
    rclcpp::Subscription<interfaces::msg::SpinCmd>::SharedPtr spin_cmd_sub_;
    rclcpp::Subscription<interfaces::msg::ChassisStatus>::SharedPtr chassis_status_sub_;
    rclcpp::Subscription<interfaces::msg::CompStage>::SharedPtr comp_stage_sub_;
    rclcpp::Publisher<interfaces::msg::ChassisCmd>::SharedPtr chassis_cmd_pub_;
    rclcpp::Publisher<interfaces::msg::FollowerState>::SharedPtr follower_state_pub_;
    rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr replan_trigger_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_predicted_path_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr debug_v_pred_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr debug_w_pred_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr debug_final_cost_map_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_mppi_rollouts_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // ─── 参数 ───
    bool enable_debug_;
    double remaining_energy_filter_alpha_ = 1.0;  // 一阶惯性滤波系数
    double prediction_dt_ = 0.2;                  // 预测步长 (s)

    // ─── 核心组件 ───
    std::unique_ptr<MainController> nav_controller_;

    // ─── 台阶障碍物/方向场掩码模块 ───
    std::unique_ptr<StepRoutingMask> step_routing_mask_;
    CostMap::ConstPtr step_cost_layer_;
    DirectionMap::ConstPtr masked_direction_map_;

    // ─── 缓存数据 ───
    CostMap::ConstPtr global_cost_map_;
    CostMap::ConstPtr current_cost_map_;                       // 当前帧动态代价图（来自 cost_maps[0]）
    std::vector<CostMap::ConstPtr> prediction_maps_;           // 预测代价地图（cost_maps[1..N]，可为空）
    std::vector<CostMap::ConstPtr> per_step_final_cost_maps_;  // 逐步融合后的最终代价地图
    CostMap::ConstPtr masked_global_cost_map_, final_cost_map_;
    DirectionMap::ConstPtr global_direction_map_;
    std::optional<SplineD> global_path_;
    bool path_updated_ = false;
    bool step_layer_update_deferred_ = false;
    bool fixed_goal_ = false;
    Eigen::Vector2d fixed_goal_pos_ = Eigen::Vector2d::Zero();
    ChassisMotionState chassis_state_ {};
    uint8_t chassis_leg_mode_ = 4;
    uint8_t comp_stage_ = 4;
    double rfr_pwr_limit_ = 90.0;
    double remaining_energy_filtered_ = 1400.0;
    int64_t local_cost_maps_stamp_ns_ = 0;
    enum class SpinState { STOP, SPIN_SLOW, SPIN_FAST } spin_state_ = SpinState::STOP;
    bool spin_high_priority_ = false;
};

// ═══════════════════════ 构造函数 ════════════════════════════

PathFollowerNode::PathFollowerNode(const rclcpp::NodeOptions& options) : Node("path_follower", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    enable_debug_ = declare_parameter<bool>("debug.enable");
    if (enable_debug_) {
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        RCLCPP_DEBUG(get_logger(), "Debug mode enabled");
        debug_predicted_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("debug.predicted_path_pub_topic"), 1);
        debug_v_pred_pub_ = create_publisher<std_msgs::msg::Float64>(declare_parameter<std::string>("debug.v_pred_pub_topic"), 1);
        debug_w_pred_pub_ = create_publisher<std_msgs::msg::Float64>(declare_parameter<std::string>("debug.w_pred_pub_topic"), 1);
        debug_final_cost_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(declare_parameter<std::string>("debug.final_cost_map_pub_topic"), 1);
        debug_mppi_rollouts_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(declare_parameter<std::string>("debug.mppi_rollouts_pub_topic"), 1);
    }

    const auto load_follow_mode_profile = [this](const std::string& name) {
        return MPCFollowModeProfile {
            .command_bounds = {
                .vel_max = declare_parameter<double>("mpc.follow.mode_profiles." + name + ".command_bounds.vel_max"),
                .vel_min = declare_parameter<double>("mpc.follow.mode_profiles." + name + ".command_bounds.vel_min"),
                .omega_max = declare_parameter<double>("mpc.follow.mode_profiles." + name + ".command_bounds.omega_max"),
                .omega_min = declare_parameter<double>("mpc.follow.mode_profiles." + name + ".command_bounds.omega_min"),
            },
            .motion_constraints = {
                .acc_max = declare_parameter<double>("mpc.follow.mode_profiles." + name + ".motion_constraints.acc_max"),
                .alpha_max = declare_parameter<double>("mpc.follow.mode_profiles." + name + ".motion_constraints.alpha_max"),
                .a_lat_max = declare_parameter<double>("mpc.follow.mode_profiles." + name + ".motion_constraints.a_lat_max"),
            },
        };
    };

    const auto step_speed_levels = declare_parameter<std::vector<double>>("mpc.follow.terrain_limits.step_speed_levels");
    if (step_speed_levels.size() != 4) {
        throw std::runtime_error("mpc.follow.terrain_limits.step_speed_levels must contain exactly 4 values");
    }

    // ─── MPC 参数加载 ───
    MPCParams mpc_params = {
        .follow = {
            .start_command = {
                .vel_cmd_act_gap_max = declare_parameter<double>("mpc.follow.start_command.vel_cmd_act_gap_max"),
                .omega_cmd_act_gap_max = declare_parameter<double>("mpc.follow.start_command.omega_cmd_act_gap_max")
            },
            .mode_profiles = {
                .normal = load_follow_mode_profile("normal"),
                .up = {
                    .jump = load_follow_mode_profile("up.jump"),
                    .short_leg = load_follow_mode_profile("up.short_leg"),
                    .long_leg = load_follow_mode_profile("up.long_leg"),
                },
                .down = {
                    .jump = load_follow_mode_profile("down.jump"),
                    .short_leg = load_follow_mode_profile("down.short_leg"),
                },
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
                .step_speed_levels = {step_speed_levels[0], step_speed_levels[1], step_speed_levels[2], step_speed_levels[3]},
                .step_vel_deadzone = declare_parameter<double>("mpc.follow.terrain_limits.step_vel_deadzone")
            },
            .terrain_weights = {
                .step_vel_weight = declare_parameter<double>("mpc.follow.terrain_weights.step_vel_weight"),
                .direction = declare_parameter<double>("mpc.follow.terrain_weights.direction")
            },
            .environment_weights = {
                .obstacle = declare_parameter<double>("mpc.follow.environment_weights.obstacle")
            },
            .terminal_weights = {
                .q_v_final = declare_parameter<double>("mpc.follow.terminal_weights.q_v_final")
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
            .base_max_iters = static_cast<int>(declare_parameter<int>("mpc.follow.base_max_iters")),
            .refine_max_iters = static_cast<int>(declare_parameter<int>("mpc.follow.refine_max_iters"))
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
                .q_v = declare_parameter<double>("mpc.stop.command_weights.q_v"),
                .q_omega = declare_parameter<double>("mpc.stop.command_weights.q_omega"),
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

    auto mpc_controller = std::make_shared<MPCSolver>(mpc_params);

    // ─── FSM 参数 ───
    FsmParams fsm_params;
    fsm_params.transition = {
        .follow_to_spin_vel_max = declare_parameter<double>("state_machine.follow_to_spin_vel_max"),
        .spin_to_follow_omega_max = declare_parameter<double>("state_machine.spin_to_follow_omega_max"),
        .to_idle_vel_max = declare_parameter<double>("state_machine.to_idle_vel_max"),
        .to_idle_omega_max = declare_parameter<double>("state_machine.to_idle_omega_max"),
        .stopping_timeout = declare_parameter<double>("state_machine.stopping_timeout"),
        .wait_replan_timeout = declare_parameter<double>("state_machine.wait_replan_timeout")
    };
    fsm_params.recovery = {
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
        .goal_timeout = declare_parameter<double>("recovery.search.goal_timeout")
    };
    fsm_params.stuck = {
        .cmd_vel_threshold = declare_parameter<double>("recovery.stuck.cmd_vel_threshold"),
        .timeout = declare_parameter<double>("recovery.stuck.timeout"),
        .max_displacement = declare_parameter<double>("recovery.stuck.max_displacement"),
        .reverse_speed = declare_parameter<double>("recovery.stuck.reverse_speed"),
        .reverse_displacement = declare_parameter<double>("recovery.stuck.reverse_displacement"),
        .reverse_timeout = declare_parameter<double>("recovery.stuck.reverse_timeout")
    };

    // ─── MainController 参数 ───
    NavigationParams nav_params;
    nav_params.stop_threshold_dist = declare_parameter<double>("misc.stop_threshold_dist");
    nav_params.stop_threshold_u = declare_parameter<double>("misc.stop_threshold_u");
    nav_params.follow_proj_guard = {
        .dist_max = declare_parameter<double>("follow_proj_guard.dist_max"),
        .cost_max = declare_parameter<double>("follow_proj_guard.cost_max"),
        .cost_samples = static_cast<int>(declare_parameter<int>("follow_proj_guard.cost_samples"))
    };
    nav_params.step_detection = {
        .detect_norm_threshold = declare_parameter<double>("step.detection.detect_norm_threshold"),
        .detect_dot_threshold = declare_parameter<double>("step.detection.detect_dot_threshold"),
        .path_sample_resolution = declare_parameter<double>("step.detection.path_sample_resolution"),
        .target_match_distance = declare_parameter<double>("step.detection.target_match_distance"),
        .rollout_match_distance = declare_parameter<double>("step.detection.rollout_match_distance"),
        .latch_threshold = static_cast<int>(declare_parameter<int>("step.detection.latch_threshold")),
        .release_threshold = static_cast<int>(declare_parameter<int>("step.detection.release_threshold")),
        .lookahead_distance = declare_parameter<double>("step.detection.lookahead_distance"),
        .exit_advance_distance = declare_parameter<double>("step.detection.exit_advance_distance")
    };
    nav_params.no_progress_guard = {
        .landmark_spacing = declare_parameter<double>("follow_no_progress_guard.landmark_spacing"),
        .timeout = declare_parameter<double>("follow_no_progress_guard.timeout")
    };
    nav_params.step_block_replan = {
        .enable = declare_parameter<bool>("follow_step_block_replan.enable"),
        .lookahead_distance = declare_parameter<double>("follow_step_block_replan.lookahead_distance"),
        .sample_resolution = declare_parameter<double>("follow_step_block_replan.sample_resolution"),
        .step_norm_threshold = declare_parameter<double>("follow_step_block_replan.step_norm_threshold"),
        .obstacle_cost_threshold = declare_parameter<double>("follow_step_block_replan.obstacle_cost_threshold"),
        .predicted_obstacle_ratio_threshold = declare_parameter<double>("follow_step_block_replan.predicted_obstacle_ratio_threshold")
    };
    nav_params.latch_ttl = declare_parameter<double>("step.latch_ttl");
    nav_params.step_dist_offset = declare_parameter<double>("step.step_dist_offset");

    // ─── 创建 MainController ───
    nav_controller_ = std::make_unique<MainController>(nav_params, fsm_params, mpc_controller, get_logger());

    // ─── 读取台阶路径掩码相关参数 ───
    StepRoutingMaskParams step_params;
    step_params.path_align_dot_threshold = declare_parameter<double>("step_mask.path_align_dot_threshold");
    step_params.full_effect_radius = declare_parameter<double>("step_mask.full_effect_radius");
    step_params.cutoff_radius = declare_parameter<double>("step_mask.cutoff_radius");
    step_params.length_num_samples = static_cast<int>(declare_parameter<int>("step_mask.length_num_samples"));
    step_routing_mask_ = std::make_unique<StepRoutingMask>(step_params);

    // ─── 功率限制滤波参数 ───
    remaining_energy_filter_alpha_ = declare_parameter<double>("misc.remaining_energy_filter_alpha");

    // ─── 订阅 / 发布 ───
    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("global_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            global_cost_map_ = std::make_shared<CostMap>(*msg);
            RCLCPP_INFO(
                get_logger(), "Received global cost map: size=(%d,%d), resolution=%.2f",
                global_cost_map_->width, global_cost_map_->height, global_cost_map_->resolution
            );
            // 如果方向场已到达，则初始化台阶掩码模块并生成输出层
            update_step_layers();
            global_cost_map_sub_.reset();
        }
    );

    global_direction_map_sub_ = create_subscription<sensor_msgs::msg::Image>(
        declare_parameter<std::string>("global_direction_map_sub_topic"), 1,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            if (!global_cost_map_) {
                RCLCPP_WARN(get_logger(), "Received direction map but global cost map is not ready yet!");
                return;
            }
            cv::Mat img = cv_bridge::toCvShare(msg, "8UC3")->image;
            global_direction_map_ = std::make_shared<DirectionMap>(img, global_cost_map_->resolution, global_cost_map_->origin_x, global_cost_map_->origin_y);
            if (global_direction_map_->width != global_cost_map_->width || global_direction_map_->height != global_cost_map_->height) {
                RCLCPP_FATAL(
                    get_logger(), "Direction map size (%d,%d) does not match cost map (%d,%d)!",
                    global_direction_map_->width, global_direction_map_->height,
                    global_cost_map_->width, global_cost_map_->height
                );
                throw std::runtime_error("Direction map size does not match cost map size");
            }
            RCLCPP_INFO(get_logger(), "Received global direction map");
            // 初始化台阶掩码模块并生成输出层
            update_step_layers();
            global_direction_map_sub_.reset();
        }
    );

    local_cost_maps_sub_ = create_subscription<interfaces::msg::CostMaps>(
        declare_parameter<std::string>("local_cost_maps_sub_topic"), 1,
        [this](const interfaces::msg::CostMaps::SharedPtr msg) { local_cost_maps_callback(msg); }
    );

    control_points_sub_ = create_subscription<interfaces::msg::GlobalPath>(
        declare_parameter<std::string>("control_points_sub_topic"), 1,
        [this](const interfaces::msg::GlobalPath::SharedPtr msg) { control_points_callback(msg); }
    );

    chassis_status_sub_ = create_subscription<interfaces::msg::ChassisStatus>(
        declare_parameter<std::string>("chassis_status_sub_topic"), 1,
        [this](const interfaces::msg::ChassisStatus::SharedPtr msg) { chassis_status_callback(msg); }
    );
    
    comp_stage_sub_ = create_subscription<interfaces::msg::CompStage>(
        declare_parameter<std::string>("comp_stage_sub_topic"), 1,
        [this](const interfaces::msg::CompStage::SharedPtr msg) { comp_stage_ = msg->game_progress; }
    );

    spin_cmd_sub_ = create_subscription<interfaces::msg::SpinCmd>(
        declare_parameter<std::string>("spin_cmd_sub_topic"), 1,
        [this](const interfaces::msg::SpinCmd::SharedPtr msg) { spin_cmd_callback(msg); }
    );

    chassis_cmd_pub_ = create_publisher<interfaces::msg::ChassisCmd>(declare_parameter<std::string>("chassis_cmd_pub_topic"), 1);
    follower_state_pub_ = create_publisher<interfaces::msg::FollowerState>(declare_parameter<std::string>("follower_state_pub_topic"), 1);
    replan_trigger_pub_ = create_publisher<std_msgs::msg::Empty>(declare_parameter<std::string>("replan_trigger_pub_topic"), 1);
    control_timer_ = create_wall_timer(std::chrono::duration<double>(MPC_DT), [this]() { control_timer_callback(); });
}

// ═══════════════════════ ROS 回调 ════════════════════════════

void PathFollowerNode::control_points_callback(const interfaces::msg::GlobalPath::SharedPtr msg) {
    if (msg->x.size() < 3) {
        path_updated_ = true;
        if (!msg->x.empty()) {
            RCLCPP_WARN(get_logger(), "Received insufficient control points (%zu), need at least 3!", msg->x.size());
        }
        global_path_ = std::nullopt;
        // 空路径取消 fixed 目标（除非当前已在 FIXED 模式中且目标是 fixed 的空路径，保持 fixed_goal_）
        if (!msg->fixed) {
            fixed_goal_ = false;
        }
        update_step_layers();
        return;
    }
    if (msg->x.size() != msg->y.size()) {
        RCLCPP_ERROR(get_logger(), "GlobalPath x/y size mismatch (%zu vs %zu)!", msg->x.size(), msg->y.size());
        return;
    }
    std::vector<Eigen::Vector2d> cpts;
    cpts.reserve(msg->x.size());
    for (size_t i = 0; i < msg->x.size(); ++i) {
        cpts.emplace_back(static_cast<double>(msg->x[i]), static_cast<double>(msg->y[i]));
    }
    global_path_ = SplineD(cpts);
    global_path_->setExtrapolate(true);
    path_updated_ = true;

    // 更新 fixed 目标信息
    fixed_goal_ = msg->fixed;
    if (fixed_goal_) {
        // fixed 目标位置为路径末端
        fixed_goal_pos_ = global_path_->evaluate(1.0);
        RCLCPP_INFO(get_logger(), "Received fixed goal path, target: (%.2f, %.2f)", fixed_goal_pos_.x(), fixed_goal_pos_.y());
    }

    // 新路径到来，基于方向场擦除对应台阶
    update_step_layers();
}

void PathFollowerNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    chassis_state_.velocity = msg->velocity;
    chassis_state_.omega = msg->omega;
    chassis_state_.leg_h = msg->leg_h;
    chassis_state_.leg_psi = msg->leg_psi;
    chassis_leg_mode_ = msg->leg_mode;
    rfr_pwr_limit_ = static_cast<double>(msg->rfr_pwr_limit);
    remaining_energy_filtered_ = remaining_energy_filter_alpha_ * static_cast<double>(msg->remaining_energy_supercap) + (1.0 - remaining_energy_filter_alpha_) * remaining_energy_filtered_;
}

void PathFollowerNode::spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg) {
    switch (msg->spin_mode) {
        case 0: spin_state_ = SpinState::STOP; break;
        case 1: spin_state_ = SpinState::SPIN_SLOW; break;
        case 2: spin_state_ = SpinState::SPIN_FAST; break;
        default: RCLCPP_ERROR(get_logger(), "Received invalid spin state: %d", msg->spin_mode); return;
    }
    spin_high_priority_ = msg->high_priority;
}

void PathFollowerNode::local_cost_maps_callback(const interfaces::msg::CostMaps::SharedPtr msg) {
    if (!global_cost_map_ || msg->maps.empty()) return;
    const int w = global_cost_map_->width;
    const int h = global_cost_map_->height;
    const auto total = static_cast<size_t>(w * h);

    if (msg->maps[0].data.size() != total) return;
    local_cost_maps_stamp_ns_ = rclcpp::Time(msg->maps[0].header.stamp).nanoseconds();
    prediction_dt_ = msg->prediction_dt;

    // maps[0] = 当前帧动态代价图
    {
        std::vector<uint8_t> data(total);
        for (size_t j = 0; j < total; j++) {
            data[j] = static_cast<uint8_t>(msg->maps[0].data[j]);
        }
        current_cost_map_ = std::make_shared<CostMap>(
            w, h, global_cost_map_->resolution,
            global_cost_map_->origin_x, global_cost_map_->origin_y,
            data
        );
    }

    // maps[1..N] = 预测代价地图
    prediction_maps_.clear();
    for (size_t i = 1; i < msg->maps.size(); i++) {
        const auto& map = msg->maps[i];
        if (map.data.size() != total) continue;
        std::vector<uint8_t> data(total);
        for (size_t j = 0; j < total; j++) {
            data[j] = static_cast<uint8_t>(map.data[j]);
        }
        prediction_maps_.push_back(std::make_shared<CostMap>(
            w, h, global_cost_map_->resolution,
            global_cost_map_->origin_x, global_cost_map_->origin_y,
            data
        ));
    }

    update_masked_cost_maps();
}

// ═══════════════════ 台阶掩码层更新 ════════════════════════════════

bool PathFollowerNode::should_use_prediction_maps() const {
    return !prediction_maps_.empty();
}

bool PathFollowerNode::is_step_routing_context_locked() const {
    return nav_controller_ && nav_controller_->fsm_state() == FsmState::STEPPING;
}

void PathFollowerNode::refresh_deferred_step_layers() {
    if (!step_layer_update_deferred_ || is_step_routing_context_locked()) {
        return;
    }

    update_step_layers();
}

void PathFollowerNode::update_step_layers() {
    if (!global_cost_map_ || !global_direction_map_ || !step_routing_mask_) return;

    if (!step_routing_mask_->ready()) {
        try {
            step_routing_mask_->initialize(*global_cost_map_, global_direction_map_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize StepRoutingMask: %s", e.what());
            return;
        }
    }

    if (is_step_routing_context_locked()) {
        step_layer_update_deferred_ = true;
        return;
    }

    step_routing_mask_->update(global_path_);
    step_cost_layer_ = step_routing_mask_->step_cost_layer();
    masked_direction_map_ = step_routing_mask_->masked_direction_map();
    step_layer_update_deferred_ = false;
    update_masked_cost_maps();
}

void PathFollowerNode::update_masked_cost_maps() {
    if (!global_cost_map_) return;

    try {
        if (step_cost_layer_) {
            masked_global_cost_map_ = std::make_shared<CostMap>(global_cost_map_->merge(*step_cost_layer_));
        } else {
            masked_global_cost_map_ = global_cost_map_;
        }

        per_step_final_cost_maps_.clear();
        if (should_use_prediction_maps()) {
            for (const auto& pred_map : prediction_maps_) {
                per_step_final_cost_maps_.push_back(
                    std::make_shared<CostMap>(masked_global_cost_map_->merge(*pred_map))
                );
            }
        }

        if (current_cost_map_) {
            final_cost_map_ = std::make_shared<CostMap>(masked_global_cost_map_->merge(*current_cost_map_));
        } else {
            final_cost_map_ = masked_global_cost_map_;
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to merge cost maps: %s", e.what());
    }
}

// ═══════════════════ 控制主循环 ══════════════════════════════

void PathFollowerNode::control_timer_callback() {
    refresh_deferred_step_layers();

    if (!global_cost_map_ || !final_cost_map_ || !masked_direction_map_) return;

    if (enable_debug_) {
        nav_msgs::msg::OccupancyGrid grid_msg;
        grid_msg.header.stamp = now();
        grid_msg.header.frame_id = "map";
        grid_msg.info.width = static_cast<uint32_t>(final_cost_map_->width);
        grid_msg.info.height = static_cast<uint32_t>(final_cost_map_->height);
        grid_msg.info.resolution = static_cast<float>(final_cost_map_->resolution);
        grid_msg.info.origin.position.x = final_cost_map_->origin_x;
        grid_msg.info.origin.position.y = final_cost_map_->origin_y;
        grid_msg.data.resize(final_cost_map_->data.size());
        for (size_t idx = 0; idx < final_cost_map_->data.size(); idx++) {
            grid_msg.data[idx] = static_cast<int8_t>(final_cost_map_->data[idx]);
        }
        debug_final_cost_map_pub_->publish(grid_msg);
    }

    Eigen::Vector3d chassis_pose_map;
    if (!get_chassis_pose(chassis_pose_map)) return;

    // 构建逐步代价地图指针数组
    std::vector<const CostMap*> per_step_ptrs;
    for (const auto& m : per_step_final_cost_maps_) {
        per_step_ptrs.push_back(m.get());
    }

    // 组装控制输入
    const bool has_prediction = should_use_prediction_maps();
    const ControlInput input = {
        .global_path = global_path_,
        .path_updated = path_updated_,
        .fixed_goal = fixed_goal_,
        .fixed_goal_pos = fixed_goal_pos_,
        .chassis_pose_map = chassis_pose_map,
        .chassis_state = chassis_state_,
        .remaining_energy = remaining_energy_filtered_,
        .rfr_pwr_limit = rfr_pwr_limit_,
        .chassis_leg_mode = chassis_leg_mode_,
        .comp_stage = comp_stage_,
        .spin_requested = (spin_state_ != SpinState::STOP),
        .spin_high_priority = spin_high_priority_,
        .spin_slow = (spin_state_ == SpinState::SPIN_SLOW),
        .spin_fast = (spin_state_ == SpinState::SPIN_FAST),
        .final_cost_map = final_cost_map_.get(),
        .masked_global_cost_map = masked_global_cost_map_.get(),
        .masked_direction_map = masked_direction_map_.get(),
        .base_direction_map = global_direction_map_.get(),
        .current_dynamic_cost_map = current_cost_map_.get(),
        .per_step_cost_maps = std::move(per_step_ptrs),
        .per_step_dynamic_cost_maps = [&]() {
            std::vector<const CostMap*> maps;
            if (has_prediction) {
                maps.reserve(prediction_maps_.size());
                for (const auto& m : prediction_maps_) maps.push_back(m.get());
            }
            return maps;
        }(),
        .prediction_dt = prediction_dt_,
        .stamp = std::chrono::steady_clock::now()
    };

    // 调用控制逻辑层
    const ControlOutput output = nav_controller_->update(input);
    path_updated_ = false;

    interfaces::msg::FollowerState state_msg;
    state_msg.state = static_cast<uint8_t>(output.fsm_state);
    follower_state_pub_->publish(state_msg);

    if (output.request_replan) {
        replan_trigger_pub_->publish(std_msgs::msg::Empty {});
    }

    if (enable_debug_ && output.mppi_rollouts) {
        publish_mppi_rollouts(*output.mppi_rollouts);
    }

    // 处理路径消费请求：清空 spline 路径，并据此重建台阶擦除地图。
    // 若当前已进入 FIXED，只消费路径，不取消 fixed 目标标记。
    if (output.consume_global_path) {
        global_path_ = std::nullopt;
        // 仅在语义上真正取消目标时才清除 fixed 标志；重规划等待期间保留目标。
        if (output.fsm_state != FsmState::FIXED && !output.keep_goal_on_path_consume) {
            fixed_goal_ = false;
        }
        update_step_layers();
    }

    refresh_deferred_step_layers();

    // 发布指令
    if (output.valid) {
        publish_chassis_cmd(output);
        if (enable_debug_) {
            if (output.predicted_path_map) {
                debug_predicted_path_pub_->publish(path_to_nav_msg(*output.predicted_path_map));
            }
            if (output.predicted_v && output.predicted_w) {
                std_msgs::msg::Float64 v_msg, w_msg;
                v_msg.data = (*output.predicted_v)[0];
                w_msg.data = (*output.predicted_w)[0];
                debug_v_pred_pub_->publish(v_msg);
                debug_w_pred_pub_->publish(w_msg);
            }
        }
    }
}

// ═══════════════════ 工具函数 ════════════════════════════════

void PathFollowerNode::publish_chassis_cmd(const ControlOutput& output) {
    interfaces::msg::ChassisCmd msg;
    msg.velocity = static_cast<float>(output.velocity);
    msg.omega = static_cast<float>(output.omega);
    msg.mode = static_cast<uint8_t>(output.mode);
    msg.step_dist = output.step_dist_cm;
    chassis_cmd_pub_->publish(msg);
}

bool PathFollowerNode::get_chassis_pose(Eigen::Vector3d& chassis_pose) const {
    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = tf_buffer_->lookupTransform("map", "chassis_link", tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(get_logger(), "Could not transform chassis_link to map: %s", ex.what());
        return false;
    }
    chassis_pose.head<2>() = utils::convert_to<Eigen::Vector3d>(tf.transform.translation).head<2>();
    const Eigen::Quaterniond q = utils::convert_to<Eigen::Quaterniond>(tf.transform.rotation);
    const Eigen::Vector2d x_axis = (q * Eigen::Vector3d::UnitX()).head<2>();
    if (x_axis.norm() < 1e-6) {
        RCLCPP_ERROR(get_logger(), "Invalid chassis_link orientation");
        return false;
    }
    chassis_pose.z() = atan2(x_axis.y(), x_axis.x());
    return true;
}

nav_msgs::msg::Path PathFollowerNode::path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const {
    nav_msgs::msg::Path msg;
    msg.header.stamp = now();
    msg.header.frame_id = "map";
    for (const auto& p : path) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = msg.header;
        ps.pose.position.x = p.x();
        ps.pose.position.y = p.y();
        ps.pose.position.z = 0.0;
        msg.poses.push_back(ps);
    }
    return msg;
}

void PathFollowerNode::publish_mppi_rollouts(const std::vector<std::vector<Eigen::Vector2d>>& rollouts) {
    if (!debug_mppi_rollouts_pub_) return;

    visualization_msgs::msg::MarkerArray markers;
    const auto stamp = now();

    constexpr float hue_start = 0.0f;     // red
    constexpr float hue_end = 300.0f;   // magenta
    constexpr float sat = 1.0f;
    constexpr float val = 1.0f;

    for (size_t i = 0; i < rollouts.size(); ++i) {
        const float t = (rollouts.size() <= 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(rollouts.size() - 1);
        const float h = hue_start + t * (hue_end - hue_start);

        const float c = val * sat;
        const float hp = h / 60.0f;
        const float x = c * (1.0f - std::abs(std::fmod(hp, 2.0f) - 1.0f));
        const float m = val - c;

        float r, g, b;
        const int sextant = static_cast<int>(hp) % 6;
        switch (sextant) {
            case 0: r = c; g = x; b = 0; break;
            case 1: r = x; g = c; b = 0; break;
            case 2: r = 0; g = c; b = x; break;
            case 3: r = 0; g = x; b = c; break;
            case 4: r = x; g = 0; b = c; break;
            case 5: r = c; g = 0; b = x; break;
            default: r = 0; g = 0; b = 0; break;
        }

        visualization_msgs::msg::Marker marker;
        marker.header.stamp = stamp;
        marker.header.frame_id = "map";
        marker.ns = "mppi_rollouts";
        marker.id = static_cast<int>(i);
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.06;
        marker.color.a = 0.65f;
        marker.color.r = r + m;
        marker.color.g = g + m;
        marker.color.b = b + m;

        marker.points.reserve(rollouts[i].size());
        for (const auto& pt : rollouts[i]) {
            geometry_msgs::msg::Point p;
            p.x = pt.x();
            p.y = pt.y();
            p.z = 0.0;
            marker.points.push_back(p);
        }

        markers.markers.push_back(std::move(marker));
    }

    debug_mppi_rollouts_pub_->publish(markers);
}

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(path_follower::PathFollowerNode)
