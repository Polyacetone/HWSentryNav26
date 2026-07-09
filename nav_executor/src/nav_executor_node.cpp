#include <nav_executor/nav_executor_node.hpp>
#include <nav_executor/planner/step_annotator.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <common_utils/convert.hpp>

namespace nav_executor {

// ═══════════════════════ 构造函数 ════════════════════════════

NavExecutorNode::NavExecutorNode(const rclcpp::NodeOptions& options) : Node("nav_executor", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    enable_debug_ = declare_parameter<bool>("debug.enable");
    if (enable_debug_) {
        get_logger().set_level(rclcpp::Logger::Level::Debug);
        debug_predicted_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("debug.predicted_path_pub_topic"), 1);
        debug_optimized_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("debug.optimized_path_pub_topic"), 1);
        debug_rough_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("debug.rough_path_pub_topic"), 1);
        debug_warmup_path_pub_ = create_publisher<nav_msgs::msg::Path>(declare_parameter<std::string>("debug.warmup_path_pub_topic"), 1);
        debug_v_pred_pub_ = create_publisher<std_msgs::msg::Float64>(declare_parameter<std::string>("debug.v_pred_pub_topic"), 1);
        debug_w_pred_pub_ = create_publisher<std_msgs::msg::Float64>(declare_parameter<std::string>("debug.w_pred_pub_topic"), 1);
        debug_final_cost_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(declare_parameter<std::string>("debug.final_cost_map_pub_topic"), 1);
        debug_mppi_rollouts_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(declare_parameter<std::string>("debug.mppi_rollouts_pub_topic"), 1);
    }

    // ─── 只读配置 ───
    load_terrain_config();

    // ─── MPC solver ───
    const MPCParams mpc_params = load_mpc_params();
    auto mpc_solver = std::make_shared<MPCSolver>(mpc_params);

    // ─── PathExecutor ───
    executor_ = std::make_unique<PathExecutor>(
        load_executor_params(), load_fsm_params(), mpc_solver,
        mpc_params.follow.normal_profile, mpc_params.follow.capability_profiles,
        load_blend_params(), get_logger()
    );

    // ─── StepRoutingMask（共享给 planner worker）───
    StepRoutingMaskParams step_params;
    step_params.path_align_dot_threshold = declare_parameter<double>("step_mask.path_align_dot_threshold");
    step_params.full_effect_radius = declare_parameter<double>("step_mask.full_effect_radius");
    step_params.cutoff_radius = declare_parameter<double>("step_mask.cutoff_radius");
    step_params.length_num_samples = static_cast<int>(declare_parameter<int>("step_mask.length_num_samples"));
    step_routing_mask_ = std::make_shared<StepRoutingMask>(step_params);

    // ─── PathPlanner worker ───
    auto a_star = std::make_shared<AStarPlanner>(
        declare_parameter<double>("planner.a_star.step_alignment_weight"),
        declare_parameter<double>("planner.a_star.obstacle_weight"),
        declare_parameter<double>("planner.a_star.step_proximity_weight"),
        declare_parameter<double>("planner.a_star.step_mode_dot_threshold"),
        static_cast<int>(declare_parameter<int>("planner.a_star.downsampled_waypoint_max_interval")),
        static_cast<int>(declare_parameter<int>("planner.a_star.feasible_threshold"))
    );
    auto optimizer = std::make_shared<BSplineOptimizer>(load_optimizer_params());
    planner_ = std::make_unique<PathPlanner>(
        load_planner_config(), a_star, optimizer, step_routing_mask_, get_logger()
    );
    planner_->start();

    // ─── TaskExecutor ───
    task_ = std::make_unique<TaskExecutor>(load_task_params(), planner_.get(), get_logger());

    // ─── RouteMonitor 参数 ───
    proj_guard_params_ = {
        .dist_max = declare_parameter<double>("route_monitor.proj_guard.dist_max"),
        .cost_max = declare_parameter<double>("route_monitor.proj_guard.cost_max"),
        .cost_samples = static_cast<int>(declare_parameter<int>("route_monitor.proj_guard.cost_samples"))
    };
    step_block_params_ = {
        .enable = declare_parameter<bool>("route_monitor.step_block.enable"),
        .lookahead_distance = declare_parameter<double>("route_monitor.step_block.lookahead_distance"),
        .sample_resolution = declare_parameter<double>("route_monitor.step_block.sample_resolution"),
        .step_norm_threshold = declare_parameter<double>("route_monitor.step_block.step_norm_threshold"),
        .obstacle_cost_threshold = declare_parameter<double>("route_monitor.step_block.obstacle_cost_threshold"),
        .predicted_obstacle_ratio_threshold = declare_parameter<double>("route_monitor.step_block.predicted_obstacle_ratio_threshold")
    };

    prediction_horizon_seconds_ = declare_parameter<double>("prediction.horizon_seconds");
    prediction_weight_decay_ = declare_parameter<double>("prediction.weight_decay");
    remaining_energy_filter_alpha_ = declare_parameter<double>("misc.remaining_energy_filter_alpha");

    // ─── 订阅 / 发布 ───
    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("global_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            global_cost_map_ = std::make_shared<CostMap>(*msg);
            RCLCPP_INFO(get_logger(), "Received global cost map: (%d,%d) res=%.2f",
                global_cost_map_->width, global_cost_map_->height, global_cost_map_->resolution);
            try_init_step_mask();
            global_cost_map_sub_.reset();
        }
    );

    global_direction_map_sub_ = create_subscription<sensor_msgs::msg::Image>(
        declare_parameter<std::string>("global_direction_map_sub_topic"), 1,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            if (!global_cost_map_) {
                RCLCPP_WARN(get_logger(), "Received direction map before cost map; ignoring");
                return;
            }
            const cv::Mat img = cv_bridge::toCvShare(msg, "8UC3")->image;
            global_direction_map_ = std::make_shared<DirectionMap>(
                img, global_cost_map_->resolution, global_cost_map_->origin_x, global_cost_map_->origin_y,
                terrain_profiles_, terrain_rules_
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
        declare_parameter<std::string>("local_cost_maps_sub_topic"), 1,
        [this](const interfaces::msg::CostMaps::SharedPtr msg) { local_cost_maps_callback(msg); }
    );

    goal_sub_ = create_subscription<interfaces::msg::NavGoal>(
        declare_parameter<std::string>("goal_sub_topic"), 1,
        [this](const interfaces::msg::NavGoal::SharedPtr msg) {
            Goal g;
            g.position_map = Eigen::Vector2d(msg->x, msg->y);
            g.fixed = msg->fixed;
            pending_goal_ = g; // 只缓存，不做状态转移
        }
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
    state_pub_ = create_publisher<interfaces::msg::NavExecutorState>(declare_parameter<std::string>("state_pub_topic"), 1);

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

// ═══════════════════════ ROS 回调 ════════════════════════════

void NavExecutorNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    chassis_state_.velocity = msg->velocity;
    chassis_state_.omega = msg->omega;
    chassis_state_.leg_h = msg->leg_h;
    chassis_state_.leg_psi = msg->leg_psi;
    chassis_leg_mode_ = msg->leg_mode;
    rfr_pwr_limit_ = static_cast<double>(msg->rfr_pwr_limit);
    remaining_energy_filtered_ = remaining_energy_filter_alpha_ * static_cast<double>(msg->remaining_energy_supercap)
        + (1.0 - remaining_energy_filter_alpha_) * remaining_energy_filtered_;
}

void NavExecutorNode::spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg) {
    switch (msg->spin_mode) {
        case 0: spin_state_ = SpinState::STOP; break;
        case 1: spin_state_ = SpinState::SPIN_SLOW; break;
        case 2: spin_state_ = SpinState::SPIN_FAST; break;
        default: RCLCPP_ERROR(get_logger(), "Invalid spin_mode: %d", msg->spin_mode); return;
    }
    spin_high_priority_ = msg->high_priority;
}

void NavExecutorNode::local_cost_maps_callback(const interfaces::msg::CostMaps::SharedPtr msg) {
    if (!global_cost_map_ || msg->maps.empty()) return;
    const int w = global_cost_map_->width;
    const int h = global_cost_map_->height;
    const auto total = static_cast<size_t>(w * h);
    if (msg->maps[0].data.size() != total) return;
    prediction_dt_ = msg->prediction_dt;

    const auto to_cost_map = [&](const nav_msgs::msg::OccupancyGrid& grid) {
        std::vector<uint8_t> data(total);
        for (size_t j = 0; j < total; j++) data[j] = static_cast<uint8_t>(grid.data[j]);
        return std::make_shared<CostMap>(w, h, global_cost_map_->resolution, global_cost_map_->origin_x, global_cost_map_->origin_y, data);
    };

    current_cost_map_ = to_cost_map(msg->maps[0]);

    prediction_maps_.clear();
    for (size_t i = 1; i < msg->maps.size(); i++) {
        if (msg->maps[i].data.size() != total) continue;
        prediction_maps_.push_back(to_cost_map(msg->maps[i]));
    }

    // planner 用：global + 时域融合动态
    CostMap::ConstPtr fused_dynamic;
    if (prediction_horizon_seconds_ <= 0.0 || msg->maps.size() <= 1 || msg->prediction_dt <= 0.0) {
        fused_dynamic = current_cost_map_;
    } else {
        const size_t n = std::min(msg->maps.size(), static_cast<size_t>(std::ceil(prediction_horizon_seconds_ / msg->prediction_dt)) + 1);
        const double inv_denom = n > 1 ? 1.0 / static_cast<double>(n - 1) : 0.0;
        std::vector<double> frame_weights(n);
        double total_weight = 0.0;
        for (size_t i = 0; i < n; i++) {
            frame_weights[i] = std::max(0.0, 1.0 - prediction_weight_decay_ * static_cast<double>(i) * inv_denom);
            total_weight += frame_weights[i];
        }
        if (total_weight <= 0.0) {
            fused_dynamic = current_cost_map_;
        } else {
            std::vector<double> accum(total, 0.0);
            for (size_t i = 0; i < n; i++) {
                const double weight = frame_weights[i];
                if (weight <= 0.0) continue;
                const auto& frame = msg->maps[i];
                for (size_t j = 0; j < total; j++) accum[j] += static_cast<double>(frame.data[j]) * weight;
            }
            std::vector<uint8_t> result(total);
            for (size_t j = 0; j < total; j++) {
                const uint32_t u = static_cast<uint32_t>(accum[j] / total_weight + 0.5);
                result[j] = u > 255u ? 255u : static_cast<uint8_t>(u);
            }
            fused_dynamic = std::make_shared<CostMap>(w, h, global_cost_map_->resolution, global_cost_map_->origin_x, global_cost_map_->origin_y, result);
        }
    }

    try {
        merged_prediction_cost_map_ = std::make_shared<CostMap>(global_cost_map_->merge(*fused_dynamic));
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to merge planner cost map: %s", e.what());
        merged_prediction_cost_map_ = global_cost_map_;
    }
}

// ═══════════════════════ 主控制循环 ════════════════════════════

void NavExecutorNode::control_tick() {
    if (!global_cost_map_ || !global_direction_map_ || !step_mask_ready_) return;

    Eigen::Vector3d chassis_pose_map;
    if (!get_chassis_pose(chassis_pose_map)) return;

    const auto stamp = std::chrono::steady_clock::now();
    const bool preemptible = executor_->preemptible();
    const AnnotatedPath* active_path = task_->active_path();

    // ── step 1：合并代价地图 ──
    // active_path 携带针对自身样条的 step_cost_layer / masked_direction_map。
    const CostMap* step_cost_layer = active_path ? active_path->step_cost_layer.get() : nullptr;
    const DirectionMap* masked_direction_map = active_path && active_path->masked_direction_map
        ? active_path->masked_direction_map.get() : global_direction_map_.get();

    CostMap::ConstPtr masked_global = step_cost_layer
        ? std::make_shared<CostMap>(global_cost_map_->merge(*step_cost_layer))
        : global_cost_map_;
    CostMap::ConstPtr final_cost_map = current_cost_map_
        ? std::make_shared<CostMap>(masked_global->merge(*current_cost_map_))
        : masked_global;

    std::vector<CostMap::ConstPtr> per_step_owned;
    std::vector<const CostMap*> per_step_ptrs;
    std::vector<const CostMap*> per_step_dynamic_ptrs;
    if (!prediction_maps_.empty()) {
        per_step_owned.reserve(prediction_maps_.size());
        per_step_ptrs.reserve(prediction_maps_.size());
        per_step_dynamic_ptrs.reserve(prediction_maps_.size());
        for (const auto& pred : prediction_maps_) {
            per_step_owned.push_back(std::make_shared<CostMap>(masked_global->merge(*pred)));
            per_step_ptrs.push_back(per_step_owned.back().get());
            per_step_dynamic_ptrs.push_back(pred.get());
        }
    }

    // ── step 2：处理新 goal 输入 ──
    task_->ingest_goal(pending_goal_, preemptible);
    pending_goal_.reset();

    // ── step 3：消费 executor_replan_event（上周期底层 one-shot）──
    // 注：executor 事件与本周期底层输出解耦，故在 step 8 后消费下一轮；
    // 我们在 tick 末尾把事件传入 task_（见下方 step 8/9）。

    // ── step 4：轮询 planner 结果 ──
    task_->poll_planner_result(preemptible);

    // ── step 5：统一规划调度 ──
    if (merged_prediction_cost_map_) {
        TaskExecutor::PlanRequestSnapshot snapshot;
        snapshot.current_pos_map = chassis_pose_map.head<2>();
        snapshot.current_yaw = chassis_pose_map.z();
        snapshot.current_velocity = chassis_state_.velocity;
        snapshot.global_cost_map = global_cost_map_;
        snapshot.merged_cost_map = merged_prediction_cost_map_;
        snapshot.direction_map = global_direction_map_;
        task_->maybe_submit_plan(preemptible, snapshot, stamp);
    }

    // ── step 6 / 7：RouteMonitor（仅 FOLLOW && has_path）──
    active_path = task_->active_path();
    if (active_path && executor_->motion_state() == MotionState::FOLLOW) {
        RouteMonitorInput rm;
        rm.active_path = active_path;
        rm.current_u = prev_route_u_;
        rm.chassis_pos_map = chassis_pose_map.head<2>();
        rm.masked_global_cost_map = masked_global.get();
        rm.current_dynamic_cost_map = current_cost_map_.get();
        rm.per_step_dynamic_cost_maps = per_step_dynamic_ptrs;
        rm.masked_direction_map = masked_direction_map;
        rm.proj_guard = proj_guard_params_;
        rm.step_block = step_block_params_;
        // lethal 仅当针对当前这条 path 时才有效（新 path 已替换则忽略陈旧标志）
        rm.mpc_lethal = prev_mpc_lethal_ && (prev_lethal_path_ == active_path);

        const RouteMonitorReport report = run_route_monitor(rm, get_logger());
        if (report.needs_replan) {
            task_->on_route_invalid(report.reason);
        }
    }

    // ── step 8：调用 PathExecutor ──
    active_path = task_->active_path();
    ExecutorInput ein;
    ein.active_path = active_path;
    ein.hold_goal = task_->hold_goal();
    ein.spin_requested = (spin_state_ != SpinState::STOP);
    ein.spin_high_priority = spin_high_priority_;
    ein.spin_fast = (spin_state_ == SpinState::SPIN_FAST);
    ein.chassis_pose_map = chassis_pose_map;
    ein.chassis_state = chassis_state_;
    ein.remaining_energy = remaining_energy_filtered_;
    ein.rfr_pwr_limit = rfr_pwr_limit_;
    ein.chassis_leg_mode = chassis_leg_mode_;
    ein.comp_stage = comp_stage_;
    ein.final_cost_map = final_cost_map.get();
    ein.masked_global_cost_map = masked_global.get();
    ein.masked_direction_map = masked_direction_map;
    ein.base_direction_map = global_direction_map_.get();
    ein.current_dynamic_cost_map = current_cost_map_.get();
    ein.per_step_cost_maps = std::move(per_step_ptrs);
    ein.per_step_dynamic_cost_maps = std::move(per_step_dynamic_ptrs);
    ein.prediction_dt = prediction_dt_;
    ein.stamp = stamp;

    const ExecutorOutput out = executor_->update(ein);

    // 缓存本周期底层输出供下一周期 RouteMonitor 使用
    prev_mpc_lethal_ = out.mpc_lethal;
    prev_lethal_path_ = out.mpc_lethal ? active_path : nullptr;
    prev_route_u_ = out.current_u;

    // ── step 9：消费 goal_reached ──
    task_->ingest_goal_reached(out.goal_reached);

    // ── step 3（顺延）：消费 executor_replan_event ──
    // one-shot 事件在底层产出后于本周期末消费，等效于下一控制周期 step 3。
    task_->ingest_executor_replan_event(out.executor_replan_event);

    // ── step 10：发布底盘命令与诊断 ──
    if (out.valid) {
        interfaces::msg::ChassisCmd cmd;
        cmd.velocity = static_cast<float>(out.velocity);
        cmd.omega = static_cast<float>(out.omega);
        cmd.mode = out.mode;
        cmd.step_dist = out.step_dist_cm;
        chassis_cmd_pub_->publish(cmd);

        if (enable_debug_) {
            if (out.predicted_path_map) debug_predicted_path_pub_->publish(path_to_nav_msg(*out.predicted_path_map));
            if (out.predicted_v && out.predicted_w && !out.predicted_v->empty() && !out.predicted_w->empty()) {
                std_msgs::msg::Float64 v_msg, w_msg;
                v_msg.data = (*out.predicted_v)[0];
                w_msg.data = (*out.predicted_w)[0];
                debug_v_pred_pub_->publish(v_msg);
                debug_w_pred_pub_->publish(w_msg);
            }
            if (out.mppi_rollouts) publish_mppi_rollouts(*out.mppi_rollouts);
        }
    }

    publish_diagnostics(task_->diagnostics(), out.motion_state);

    if (enable_debug_ && debug_final_cost_map_pub_) {
        nav_msgs::msg::OccupancyGrid grid_msg;
        grid_msg.header.stamp = now();
        grid_msg.header.frame_id = "map";
        grid_msg.info.width = static_cast<uint32_t>(final_cost_map->width);
        grid_msg.info.height = static_cast<uint32_t>(final_cost_map->height);
        grid_msg.info.resolution = static_cast<float>(final_cost_map->resolution);
        grid_msg.info.origin.position.x = final_cost_map->origin_x;
        grid_msg.info.origin.position.y = final_cost_map->origin_y;
        grid_msg.data.resize(final_cost_map->data.size());
        for (size_t idx = 0; idx < final_cost_map->data.size(); idx++) {
            grid_msg.data[idx] = static_cast<int8_t>(final_cost_map->data[idx]);
        }
        debug_final_cost_map_pub_->publish(grid_msg);
    }
}

// ═══════════════════════ 工具 ════════════════════════════════

bool NavExecutorNode::get_chassis_pose(Eigen::Vector3d& chassis_pose) const {
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
    chassis_pose.z() = std::atan2(x_axis.y(), x_axis.x());
    return true;
}

void NavExecutorNode::publish_diagnostics(const TaskDiagnostics& diag, const MotionState motion_state) {
    interfaces::msg::NavExecutorState msg;
    msg.motion_state = static_cast<uint8_t>(motion_state);
    msg.has_goal = diag.has_goal;
    msg.has_path = diag.has_path;
    msg.has_hold_goal = diag.has_hold_goal;
    msg.planner_state = static_cast<uint8_t>(diag.planner_state);
    msg.last_replan_reason = static_cast<uint8_t>(diag.last_replan_reason);
    state_pub_->publish(msg);

    if (enable_debug_) {
        if (!diag.debug_rough_path.empty()) debug_rough_path_pub_->publish(path_to_nav_msg(diag.debug_rough_path));
        if (!diag.debug_warmup_path.empty()) debug_warmup_path_pub_->publish(path_to_nav_msg(diag.debug_warmup_path));
        if (!diag.debug_optimized_path.empty()) debug_optimized_path_pub_->publish(path_to_nav_msg(diag.debug_optimized_path));
    }
}

nav_msgs::msg::Path NavExecutorNode::path_to_nav_msg(const std::vector<Eigen::Vector2d>& path) const {
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

void NavExecutorNode::publish_mppi_rollouts(const std::vector<std::vector<Eigen::Vector2d>>& rollouts) {
    if (!debug_mppi_rollouts_pub_) return;

    visualization_msgs::msg::MarkerArray markers;
    const auto stamp = now();
    constexpr float hue_start = 0.0f, hue_end = 300.0f, sat = 1.0f, val = 1.0f;

    for (size_t i = 0; i < rollouts.size(); ++i) {
        const float t = (rollouts.size() <= 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(rollouts.size() - 1);
        const float h = hue_start + t * (hue_end - hue_start);
        const float c = val * sat;
        const float hp = h / 60.0f;
        const float x = c * (1.0f - std::abs(std::fmod(hp, 2.0f) - 1.0f));
        const float m = val - c;
        float r, g, b;
        switch (static_cast<int>(hp) % 6) {
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

} // namespace nav_executor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nav_executor::NavExecutorNode)
