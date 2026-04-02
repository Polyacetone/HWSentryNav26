#include <chrono>
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <interfaces/msg/chassis_status.hpp>
#include <interfaces/msg/spin_cmd.hpp>
#include <interfaces/msg/comp_stage.hpp>
#include <interfaces/msg/global_path.hpp>
#include <interfaces/msg/follower_state.hpp>
#include <interfaces/msg/local_plan.hpp>

#include <common_utils/convert.hpp>
#include <local_planner/nav_map.hpp>
#include <local_planner/utils.hpp>
#include <local_planner/step_routing_mask.hpp>
#include <local_planner/global_path_manager.hpp>
#include <local_planner/local_planner_state_machine.hpp>

namespace local_planner {

namespace {

/// V1: 从全局 B-spline 上裁剪一段局部路径（control_points）
/// 返回局部参考路径的采样点坐标。
std::vector<Eigen::Vector2d> trim_local_path(
    const SplineD& global_path,
    const Eigen::Vector2d& robot_pos,
    double u_hint,
    double lookahead_distance,
    int proj_num_samples,
    double proj_search_window,
    double proj_lazy_dist,
    int output_num_points,
    double& u_out
) {
    const double u0 = project_to_spline_u(
        global_path, robot_pos, u_hint,
        proj_num_samples, proj_search_window, proj_lazy_dist
    );
    u_out = u0;

    // 估计 lookahead 对应的 u 范围
    // 简单方法：通过弧长估计
    const double total_arc = [&]() {
        double acc = 0.0;
        constexpr int N = 50;
        Eigen::Vector2d prev = global_path.evaluate(0.0);
        for (int i = 1; i <= N; i++) {
            const double t = static_cast<double>(i) / N;
            const Eigen::Vector2d cur = global_path.evaluate(t);
            acc += (cur - prev).norm();
            prev = cur;
        }
        return std::max(acc, 1e-6);
    }();

    const double du = lookahead_distance / total_arc;
    const double u_end = std::min(u0 + du, 1.0);

    std::vector<Eigen::Vector2d> points;
    points.reserve(static_cast<size_t>(output_num_points));
    for (int i = 0; i < output_num_points; i++) {
        const double t = (output_num_points <= 1) ? u0 :
            u0 + (u_end - u0) * static_cast<double>(i) / static_cast<double>(output_num_points - 1);
        points.push_back(global_path.evaluate(t));
    }
    return points;
}

} // anonymous namespace

class LocalPlannerNode : public rclcpp::Node {
public:
    explicit LocalPlannerNode(const rclcpp::NodeOptions& options);

private:
    // ─── ROS 回调 ───
    void global_path_callback(const interfaces::msg::GlobalPath::SharedPtr msg);
    void chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg);
    void spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg);
    void control_timer_callback();

    // ─── 工具 ───
    bool get_chassis_pose(Eigen::Vector3d& pose) const;
    void update_step_layers();
    void update_merged_cost_maps();
    bool check_stuck();

    // ─── ROS 通信 ───
    rclcpp::Subscription<interfaces::msg::GlobalPath>::SharedPtr global_path_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_cost_map_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr local_cost_map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr global_direction_map_sub_;
    rclcpp::Subscription<interfaces::msg::SpinCmd>::SharedPtr spin_cmd_sub_;
    rclcpp::Subscription<interfaces::msg::ChassisStatus>::SharedPtr chassis_status_sub_;
    rclcpp::Subscription<interfaces::msg::CompStage>::SharedPtr comp_stage_sub_;

    rclcpp::Publisher<interfaces::msg::LocalPlan>::SharedPtr local_plan_pub_;
    rclcpp::Publisher<interfaces::msg::FollowerState>::SharedPtr planner_state_pub_;

    rclcpp::TimerBase::SharedPtr control_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // ─── 核心组件 ───
    GlobalPathManager path_manager_;
    std::unique_ptr<LocalPlannerStateMachine> fsm_;
    std::unique_ptr<StepRoutingMask> step_routing_mask_;

    // ─── 台阶掩码缓存 ───
    CostMap::ConstPtr step_cost_layer_;
    DirectionMap::ConstPtr masked_direction_map_;

    // ─── 地图缓存 ───
    CostMap::ConstPtr global_cost_map_, local_cost_map_;
    CostMap::ConstPtr final_cost_map_;
    DirectionMap::ConstPtr global_direction_map_;

    // ─── 底盘状态 ───
    Eigen::Vector2d chassis_status_ = Eigen::Vector2d::Zero();
    uint8_t chassis_leg_mode_ = 4;
    uint8_t comp_stage_ = 4;

    // ─── 小陀螺请求 ───
    enum class SpinState { STOP, SLOW, FAST } spin_state_ = SpinState::STOP;
    bool spin_high_priority_ = false;

    // ─── 路径投影状态 ───
    double last_reference_u_ = 0.0;

    // ─── 台阶前瞻检测 ───
    struct StepAheadParams {
        double detect_norm_threshold;
        double detect_dot_threshold;
        int on_count_threshold;
        int off_count_threshold;
    } step_ahead_params_{};
    int step_up_on_ = 0, step_up_off_ = 0;
    int step_down_on_ = 0, step_down_off_ = 0;
    bool step_up_flag_ = false, step_down_flag_ = false;

    // ─── stuck 检测 ───
    StuckParams stuck_params_{};
    bool stuck_active_ = false;
    std::chrono::steady_clock::time_point stuck_start_time_;
    Eigen::Vector2d stuck_start_pos_ = Eigen::Vector2d::Zero();
    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero();

    // ─── 到达判定参数 ───
    double stop_threshold_dist_ = 0.5;
    double stop_threshold_u_ = 0.99;

    // ─── 局部路径裁剪参数 ───
    double lookahead_distance_ = 3.0;
    int output_num_points_ = 30;
    int proj_num_samples_ = 50;
    double proj_search_window_ = 0.1;
    double proj_lazy_dist_ = 0.1;

    double control_dt_ = 0.05;
};

// ═══════════════════════ 构造函数 ════════════════════════════

LocalPlannerNode::LocalPlannerNode(const rclcpp::NodeOptions& options) : Node("local_planner", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // ─── FSM 参数 ───
    LocalPlannerFsmParams fsm_params;
    fsm_params.transition = {
        .follow_to_spin_vel_max = declare_parameter<double>("state_machine.follow_to_spin_vel_max"),
        .spin_to_follow_omega_max = declare_parameter<double>("state_machine.spin_to_follow_omega_max"),
        .to_idle_vel_max = declare_parameter<double>("state_machine.to_idle_vel_max"),
        .to_idle_omega_max = declare_parameter<double>("state_machine.to_idle_omega_max"),
        .stopping_timeout = declare_parameter<double>("state_machine.stopping_timeout"),
    };

    stuck_params_ = {
        .enable = declare_parameter<bool>("recovery.stuck.enable"),
        .cmd_vel_threshold = declare_parameter<double>("recovery.stuck.cmd_vel_threshold"),
        .timeout = declare_parameter<double>("recovery.stuck.timeout"),
        .max_displacement = declare_parameter<double>("recovery.stuck.max_displacement"),
        .reverse_speed = declare_parameter<double>("recovery.stuck.reverse_speed"),
        .reverse_duration = declare_parameter<double>("recovery.stuck.reverse_duration"),
    };
    fsm_params.stuck = stuck_params_;

    fsm_ = std::make_unique<LocalPlannerStateMachine>(fsm_params, get_logger());

    // ─── 台阶掩码参数 ───
    StepRoutingMaskParams step_mask_params;
    step_mask_params.path_align_dot_threshold = declare_parameter<double>("step_mask.path_align_dot_threshold");
    step_mask_params.full_effect_radius = declare_parameter<double>("step_mask.full_effect_radius");
    step_mask_params.cutoff_radius = declare_parameter<double>("step_mask.cutoff_radius");
    step_mask_params.length_num_samples = static_cast<int>(declare_parameter<int>("step_mask.length_num_samples"));
    step_routing_mask_ = std::make_unique<StepRoutingMask>(step_mask_params);

    // ─── 台阶前瞻参数 ───
    step_ahead_params_.detect_norm_threshold = declare_parameter<double>("step_ahead_flag.detect_norm_threshold");
    step_ahead_params_.detect_dot_threshold = declare_parameter<double>("step_ahead_flag.detect_dot_threshold");
    step_ahead_params_.on_count_threshold = static_cast<int>(declare_parameter<int>("step_ahead_flag.on_count_threshold"));
    step_ahead_params_.off_count_threshold = static_cast<int>(declare_parameter<int>("step_ahead_flag.off_count_threshold"));

    // ─── 到达判定参数 ───
    stop_threshold_dist_ = declare_parameter<double>("misc.stop_threshold_dist");
    stop_threshold_u_ = declare_parameter<double>("misc.stop_threshold_u");
    control_dt_ = declare_parameter<double>("misc.control_dt");

    // ─── 局部路径裁剪参数 ───
    lookahead_distance_ = declare_parameter<double>("local_path.lookahead_distance");
    output_num_points_ = static_cast<int>(declare_parameter<int>("local_path.num_points"));
    proj_num_samples_ = static_cast<int>(declare_parameter<int>("local_path.proj_num_samples"));
    proj_search_window_ = declare_parameter<double>("local_path.proj_search_window");
    proj_lazy_dist_ = declare_parameter<double>("local_path.proj_lazy_dist");

    // ─── 订阅 ───
    global_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("global_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            global_cost_map_ = std::make_shared<CostMap>(*msg);
            RCLCPP_INFO(get_logger(), "Received global cost map: size=(%d,%d)", global_cost_map_->width, global_cost_map_->height);
            update_step_layers();
            global_cost_map_sub_.reset();
        }
    );

    global_direction_map_sub_ = create_subscription<sensor_msgs::msg::Image>(
        declare_parameter<std::string>("global_direction_map_sub_topic"), 1,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            if (!global_cost_map_) {
                RCLCPP_WARN(get_logger(), "Received direction map but global cost map is not ready!");
                return;
            }
            cv::Mat img = cv_bridge::toCvShare(msg, "8UC2")->image;
            global_direction_map_ = std::make_shared<DirectionMap>(img, global_cost_map_->resolution, global_cost_map_->origin_x, global_cost_map_->origin_y);
            RCLCPP_INFO(get_logger(), "Received global direction map");
            update_step_layers();
            global_direction_map_sub_.reset();
        }
    );

    local_cost_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("local_cost_map_sub_topic"), 1,
        [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
            if (!global_cost_map_) return;
            local_cost_map_ = std::make_shared<CostMap>(*msg);
            update_merged_cost_maps();
        }
    );

    global_path_sub_ = create_subscription<interfaces::msg::GlobalPath>(
        declare_parameter<std::string>("global_path_sub_topic"), 1,
        [this](const interfaces::msg::GlobalPath::SharedPtr msg) { global_path_callback(msg); }
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

    // ─── 发布 ───
    local_plan_pub_ = create_publisher<interfaces::msg::LocalPlan>(
        declare_parameter<std::string>("local_plan_pub_topic"), 1
    );
    planner_state_pub_ = create_publisher<interfaces::msg::FollowerState>(
        declare_parameter<std::string>("planner_state_pub_topic"), 1
    );

    control_timer_ = create_wall_timer(
        std::chrono::duration<double>(control_dt_),
        [this]() { control_timer_callback(); }
    );
}

// ═══════════════════ 回调 ════════════════════════════════════

void LocalPlannerNode::global_path_callback(const interfaces::msg::GlobalPath::SharedPtr msg) {
    std::vector<Eigen::Vector2d> cpts;
    cpts.reserve(msg->x.size());
    for (size_t i = 0; i < msg->x.size(); ++i) {
        cpts.emplace_back(static_cast<double>(msg->x[i]), static_cast<double>(msg->y[i]));
    }
    path_manager_.set_path(cpts, msg->fixed);
    update_step_layers();
}

void LocalPlannerNode::chassis_status_callback(const interfaces::msg::ChassisStatus::SharedPtr msg) {
    chassis_status_.x() = msg->velocity;
    chassis_status_.y() = msg->omega;
    chassis_leg_mode_ = msg->leg_mode;
}

void LocalPlannerNode::spin_cmd_callback(const interfaces::msg::SpinCmd::SharedPtr msg) {
    switch (msg->spin_mode) {
        case 0: spin_state_ = SpinState::STOP; break;
        case 1: spin_state_ = SpinState::SLOW; break;
        case 2: spin_state_ = SpinState::FAST; break;
        default: RCLCPP_ERROR(get_logger(), "Invalid spin_mode: %d", msg->spin_mode); return;
    }
    spin_high_priority_ = msg->high_priority;
}

// ═══════════════════ 台阶掩码层 ═════════════════════════════

void LocalPlannerNode::update_step_layers() {
    if (!global_cost_map_ || !global_direction_map_ || !step_routing_mask_) return;
    if (!step_routing_mask_->ready()) {
        try {
            step_routing_mask_->initialize(*global_cost_map_, global_direction_map_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "StepRoutingMask init failed: %s", e.what());
            return;
        }
    }
    step_routing_mask_->update(path_manager_.global_path());
    step_cost_layer_ = step_routing_mask_->step_cost_layer();
    masked_direction_map_ = step_routing_mask_->masked_direction_map();
    update_merged_cost_maps();
}

void LocalPlannerNode::update_merged_cost_maps() {
    if (!global_cost_map_) return;
    try {
        if (step_cost_layer_) {
            auto masked_global = std::make_shared<CostMap>(global_cost_map_->merge(*step_cost_layer_));
            if (local_cost_map_) {
                final_cost_map_ = std::make_shared<CostMap>(masked_global->merge(*local_cost_map_));
            } else {
                final_cost_map_ = masked_global;
            }
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Cost map merge failed: %s", e.what());
    }
}

// ═══════════════════ stuck 检测 ══════════════════════════════

bool LocalPlannerNode::check_stuck() {
    if (!stuck_params_.enable) return false;
    if (std::abs(last_cmd_.x()) < stuck_params_.cmd_vel_threshold) {
        stuck_active_ = false;
        return false;
    }

    Eigen::Vector3d pose;
    if (!get_chassis_pose(pose)) return false;
    const Eigen::Vector2d pos = pose.head<2>();

    if (!stuck_active_) {
        stuck_active_ = true;
        stuck_start_time_ = std::chrono::steady_clock::now();
        stuck_start_pos_ = pos;
        return false;
    }

    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - stuck_start_time_).count();
    const double disp = (pos - stuck_start_pos_).norm();
    if (disp > stuck_params_.max_displacement) {
        stuck_start_time_ = std::chrono::steady_clock::now();
        stuck_start_pos_ = pos;
        return false;
    }
    return dt >= stuck_params_.timeout;
}

// ═══════════════════ 台阶前瞻检测 ═══════════════════════════

namespace {

struct StepFlags {
    bool step_up = false;
    bool step_down = false;
};

StepFlags detect_step_flags_on_path(
    const std::vector<Eigen::Vector2d>& path_points,
    const DirectionMap& dir_map,
    double norm_threshold,
    double dot_threshold,
    int& up_on, int& up_off, int& down_on, int& down_off,
    bool& up_flag, bool& down_flag,
    int on_threshold, int off_threshold
) {
    bool raw_up = false, raw_down = false;

    for (size_t i = 0; i + 1 < path_points.size(); i++) {
        const Eigen::Vector2d& p = path_points[i];
        const Eigen::Vector2d g = dir_map.map_coord_to_grid(p);
        if (!dir_map.is_valid_coord(Eigen::Vector2i(
            static_cast<int>(std::floor(g.x())),
            static_cast<int>(std::floor(g.y()))
        ))) continue;

        const Eigen::Vector2d dir = dir_map.interpolate(g);
        if (dir.norm() < norm_threshold) continue;

        // 使用路径切向方向代替 heading
        const Eigen::Vector2d tangent = (path_points[i + 1] - path_points[i]).normalized();
        const double dot_val = dir.normalized().dot(tangent);
        if (dot_val > dot_threshold) raw_up = true;
        if (dot_val < -dot_threshold) raw_down = true;
    }

    // 防抖
    if (raw_up) { up_on++; up_off = 0; if (up_on >= on_threshold) up_flag = true; }
    else        { up_off++; up_on = 0;  if (up_off >= off_threshold) up_flag = false; }

    if (raw_down) { down_on++; down_off = 0; if (down_on >= on_threshold) down_flag = true; }
    else          { down_off++; down_on = 0;  if (down_off >= off_threshold) down_flag = false; }

    return { up_flag, down_flag };
}

} // anonymous namespace

// ═══════════════════ 主定时回调 ══════════════════════════════

void LocalPlannerNode::control_timer_callback() {
    if (!global_cost_map_ || !final_cost_map_ || !masked_direction_map_) return;

    Eigen::Vector3d chassis_pose;
    if (!get_chassis_pose(chassis_pose)) return;

    const auto now_tp = std::chrono::steady_clock::now();
    const bool has_path = path_manager_.has_path();
    const bool has_new_path = path_manager_.path_updated();

    // 到达判定
    bool reach_goal = false;
    if (has_path) {
        const auto& gp = *path_manager_.global_path();
        const double dist_to_end = (chassis_pose.head<2>() - gp.evaluate(1.0)).norm();
        reach_goal = (dist_to_end < stop_threshold_dist_) || (last_reference_u_ > stop_threshold_u_);
    }

    // ─── 组装 FSM 输入 ───
    LocalPlannerFsmInput fsm_in;
    fsm_in.has_path = has_path;
    fsm_in.has_new_path = has_new_path;
    fsm_in.fixed_goal_flag = path_manager_.fixed_goal();
    fsm_in.reach_goal = reach_goal;
    fsm_in.spin_requested = (spin_state_ != SpinState::STOP);
    fsm_in.spin_high_priority = spin_high_priority_;
    fsm_in.is_stuck = check_stuck();
    fsm_in.velocity = last_cmd_.x();
    fsm_in.omega = last_cmd_.y();
    fsm_in.stamp = now_tp;

    // ─── FSM 更新 ───
    const auto fsm_out = fsm_->update(fsm_in);
    const PlannerState state = fsm_out.state;

    // 处理路径消费
    if (fsm_out.consume_global_path) {
        const bool keep_fixed = (state == PlannerState::HOLD_FIXED);
        path_manager_.consume(keep_fixed);
        last_reference_u_ = 0.0;
        stuck_active_ = false;
        // 重置台阶检测防抖
        step_up_on_ = step_up_off_ = step_down_on_ = step_down_off_ = 0;
        step_up_flag_ = step_down_flag_ = false;
        update_step_layers();
    }

    // path_updated 标记在本周期结束后重置
    path_manager_.reset_update_flag();

    // ─── 构建 LocalPlan 消息 ───
    interfaces::msg::LocalPlan plan_msg;
    plan_msg.consume_global_path = fsm_out.consume_global_path;
    plan_msg.step_up_ahead = false;
    plan_msg.step_down_ahead = false;
    plan_msg.slow_spin = false;
    plan_msg.fast_spin = false;

    switch (state) {
    case PlannerState::IDLE: {
        plan_msg.mode = interfaces::msg::LocalPlan::MODE_INVALID;
        last_cmd_ = Eigen::Vector2d::Zero();
        break;
    }

    case PlannerState::TRACK: {
        if (!path_manager_.has_path()) {
            plan_msg.mode = interfaces::msg::LocalPlan::MODE_STOP;
            break;
        }

        // V1: 从全局路径裁剪局部路径
        double u_out = last_reference_u_;
        auto local_points = trim_local_path(
            *path_manager_.global_path(),
            chassis_pose.head<2>(),
            last_reference_u_,
            lookahead_distance_,
            proj_num_samples_, proj_search_window_, proj_lazy_dist_,
            output_num_points_,
            u_out
        );
        last_reference_u_ = u_out;

        // 台阶前瞻检测
        if (masked_direction_map_ && !local_points.empty()) {
            auto sf = detect_step_flags_on_path(
                local_points, *masked_direction_map_,
                step_ahead_params_.detect_norm_threshold,
                step_ahead_params_.detect_dot_threshold,
                step_up_on_, step_up_off_, step_down_on_, step_down_off_,
                step_up_flag_, step_down_flag_,
                step_ahead_params_.on_count_threshold,
                step_ahead_params_.off_count_threshold
            );
            plan_msg.step_up_ahead = sf.step_up;
            plan_msg.step_down_ahead = sf.step_down;
        }

        plan_msg.mode = interfaces::msg::LocalPlan::MODE_TRACK;
        plan_msg.x.reserve(local_points.size());
        plan_msg.y.reserve(local_points.size());
        for (const auto& pt : local_points) {
            plan_msg.x.push_back(static_cast<float>(pt.x()));
            plan_msg.y.push_back(static_cast<float>(pt.y()));
        }
        // 估算 last_cmd 用于 stuck 检测和 STOPPING 退出
        // V1 走 MPC 时由下游设定，这里先标记非零让 stuck 正常工作
        last_cmd_.x() = 0.5; // 名义速度标记（实际速度由 mpc_controller 执行）
        last_cmd_.y() = 0.0;
        break;
    }

    case PlannerState::SPIN: {
        plan_msg.mode = interfaces::msg::LocalPlan::MODE_SPIN;
        plan_msg.slow_spin = (spin_state_ == SpinState::SLOW);
        plan_msg.fast_spin = (spin_state_ == SpinState::FAST);
        last_cmd_ = Eigen::Vector2d::Zero();
        break;
    }

    case PlannerState::STOP_TRANSITION: {
        plan_msg.mode = interfaces::msg::LocalPlan::MODE_STOP;
        // last_cmd 会由 mpc_controller 的 stop() 减小，这里不更新
        break;
    }

    case PlannerState::HOLD_FIXED: {
        plan_msg.mode = interfaces::msg::LocalPlan::MODE_HOLD_FIXED;
        plan_msg.fixed_x = static_cast<float>(path_manager_.fixed_goal_pos().x());
        plan_msg.fixed_y = static_cast<float>(path_manager_.fixed_goal_pos().y());
        last_cmd_.x() = 0.1; // 名义标记
        last_cmd_.y() = 0.0;
        break;
    }

    case PlannerState::REVERSE: {
        plan_msg.mode = interfaces::msg::LocalPlan::MODE_REVERSE;
        last_cmd_.x() = -stuck_params_.reverse_speed;
        last_cmd_.y() = 0.0;
        break;
    }
    } // switch

    // ─── 发布 ───
    local_plan_pub_->publish(plan_msg);

    interfaces::msg::FollowerState state_msg;
    state_msg.state = static_cast<uint8_t>(state);
    planner_state_pub_->publish(state_msg);
}

// ═══════════════════ 工具函数 ════════════════════════════════

bool LocalPlannerNode::get_chassis_pose(Eigen::Vector3d& chassis_pose) const {
    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = tf_buffer_->lookupTransform("map", "chassis_link", tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(get_logger(), "TF lookup failed: %s", ex.what());
        return false;
    }
    chassis_pose.head<2>() = utils::convert_to<Eigen::Vector3d>(tf.transform.translation).head<2>();
    const Eigen::Quaterniond q = utils::convert_to<Eigen::Quaterniond>(tf.transform.rotation);
    const Eigen::Vector2d x_axis = (q * Eigen::Vector3d::UnitX()).head<2>();
    if (x_axis.norm() < 1e-6) return false;
    chassis_pose.z() = std::atan2(x_axis.y(), x_axis.x());
    return true;
}

} // namespace local_planner

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(local_planner::LocalPlannerNode)
