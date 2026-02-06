#include <path_follower/control_fsm.hpp>
#include <rclcpp/logging.hpp>

#include <boost/statechart/custom_reaction.hpp>
#include <boost/statechart/event.hpp>
#include <boost/statechart/state.hpp>
#include <boost/statechart/state_machine.hpp>

namespace path_follower {
namespace {

namespace sc = boost::statechart;

// ═══════════════════════════ 辅助类型 ════════════════════════

// 正常模式目的态（Idle / Follow / Spin）
enum class NormalDest { IDLE, FOLLOW, SPIN };

// 根据外部输入计算"应当处于的正常模式"
inline NormalDest compute_desired(const FsmInput& in) {
    const bool can_spin = in.spin_high_priority || (!in.has_path);
    const bool should_spin = in.spin_requested && can_spin;
    if (should_spin) return NormalDest::SPIN;
    if (in.has_path) return NormalDest::FOLLOW;
    return NormalDest::IDLE;
}

// ═════════════════════ 卡住检测器 ═════════════════════

struct StuckMonitor {
    bool active = false;
    rclcpp::Time start_time{0, 0, RCL_ROS_TIME};
    Eigen::Vector2d start_pos = Eigen::Vector2d::Zero();

    void reset() {
        active = false;
    }

    bool update(const StuckParams& p, double last_cmd_vel, const rclcpp::Time& now, const Eigen::Vector2d& pos) {
        if (!p.enable) return false;
        if (std::abs(last_cmd_vel) < p.cmd_vel_threshold) {
            reset();
            return false;
        }
        if (!active) {
            active = true;
            start_time = now;
            start_pos = pos;
            return false;
        }
        const double dt = (now - start_time).seconds();
        const double disp = (pos - start_pos).norm();
        if (disp > p.max_displacement) {
            start_time = now;
            start_pos = pos;
            return false;
        }
        return dt >= p.timeout;
    }
};

// ═════════════════════ 危险恢复目标搜索逻辑 ═════════════════════

struct HazardLogic {
    std::optional<Eigen::Vector2d> goal_map;
    rclcpp::Time goal_set_time{0, 0, RCL_ROS_TIME};
    std::optional<rclcpp::Time> safe_since;

    void reset() {
        goal_map = std::nullopt;
        safe_since = std::nullopt;
    }

    static bool is_pose_hazard(
        const RecoveryParams& p,
        const CostMap& cost_map,
        const DirectionMap& dir_map,
        const Eigen::Vector2d& pos
    ) {
        const Eigen::Vector2d gc = cost_map.map_coord_to_grid(pos);
        const double cost = cost_map.interpolate(gc);
        const Eigen::Vector2d gd = dir_map.map_coord_to_grid(pos);
        const double step_norm = dir_map.interpolate(gd).norm();
        return (cost >= p.hazard_cost_threshold) || (step_norm >= p.hazard_step_norm_threshold);
    }

    struct FieldSample {
        double cost = 0.0;
        double step_norm = 0.0;
    };

    static std::optional<FieldSample> sample_fields(
        const CostMap& cost_map, const DirectionMap& dir_map, const Eigen::Vector2d& pos
    ) {
        const Eigen::Vector2d gc = cost_map.map_coord_to_grid(pos);
        if (!cost_map.is_valid_coord(gc)) return std::nullopt;
        const Eigen::Vector2d gd = dir_map.map_coord_to_grid(pos);
        if (!dir_map.is_valid_coord(gd)) return std::nullopt;

        FieldSample s;
        s.cost = cost_map.interpolate(gc);
        s.step_norm = dir_map.interpolate(gd).norm();
        return s;
    }

    static bool is_safe_goal(const RecoveryParams& p, const FieldSample& s) {
        return (s.cost < p.safe_cost_threshold) && (s.step_norm < p.safe_step_norm_threshold);
    }

    static double potential_cost(const FieldSample& s) {
        const double cost01 = std::clamp(s.cost / 255.0, 0.0, 1.0);
        return cost01 + s.step_norm;
    }

    static std::optional<Eigen::Vector2d> potential_gradient(
        const CostMap& cost_map,
        const DirectionMap& dir_map,
        const Eigen::Vector2d& pos,
        double eps
    ) {
        if (!(eps > 1e-6)) return std::nullopt;

        const Eigen::Vector2d dx(eps, 0.0);
        const Eigen::Vector2d dy(0.0, eps);

        const auto fx1 = sample_fields(cost_map, dir_map, pos + dx);
        const auto fx0 = sample_fields(cost_map, dir_map, pos - dx);
        const auto fy1 = sample_fields(cost_map, dir_map, pos + dy);
        const auto fy0 = sample_fields(cost_map, dir_map, pos - dy);
        if (!fx1 || !fx0 || !fy1 || !fy0) return std::nullopt;

        const double dfdx = (potential_cost(*fx1) - potential_cost(*fx0)) / (2.0 * eps);
        const double dfdy = (potential_cost(*fy1) - potential_cost(*fy0)) / (2.0 * eps);
        return Eigen::Vector2d(dfdx, dfdy);
    }

    struct PathScore {
        double score = std::numeric_limits<double>::infinity();
        bool end_safe = false;
        FieldSample end_sample;
    };

    static std::optional<PathScore> score_candidate_by_path_integral(
        const RecoveryParams& p,
        const CostMap& cost_map,
        const DirectionMap& dir_map,
        const Eigen::Vector2d& origin,
        const Eigen::Vector2d& goal
    ) {
        double acc = 0.0;
        std::optional<FieldSample> end_s;
        for (int i = 0; i <= p.circ_radius_samples; i++) {
            const double t = static_cast<double>(i) / static_cast<double>(p.circ_radius_samples);
            const Eigen::Vector2d pos = origin + (goal - origin) * t;
            const auto s = sample_fields(cost_map, dir_map, pos);
            if (!s) return std::nullopt;
            acc += potential_cost(*s);
            if (i == p.circ_radius_samples) end_s = s;
        }

        PathScore out;
        out.score = acc;
        out.end_sample = *end_s;
        out.end_safe = is_safe_goal(p, *end_s);
        return out;
    }

    std::optional<Eigen::Vector2d> find_goal(
        const RecoveryParams& p,
        const CostMap& cost_map,
        const DirectionMap& dir_map,
        const Eigen::Vector3d& chassis_pose
    ) const {
        const Eigen::Vector2d origin = chassis_pose.head<2>();

        // 固定半径环上取点，路径积分评分
        std::optional<Eigen::Vector2d> best_pt;
        std::optional<PathScore> best_sc;

        for (int i = 0; i < p.circ_angle_samples; i++) {
            const double a = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(p.circ_angle_samples);
            const Eigen::Vector2d pt = origin + Eigen::Vector2d(std::cos(a), std::sin(a)) * p.circ_radius;
            const auto sc = score_candidate_by_path_integral(p, cost_map, dir_map, origin, pt);
            if (!sc) continue;
            if (!best_sc || sc->score < best_sc->score) {
                best_sc = *sc;
                best_pt = pt;
            }
        }

        return best_pt;
    }

    /// 更新恢复目标 + 退出判定（不调用 MPC）
    struct StepResult {
        std::optional<Eigen::Vector2d> goal;
        bool finished = false;
    };

    StepResult step(
        const RecoveryParams& p,
        const rclcpp::Time& now,
        const Eigen::Vector3d& chassis_pose_map,
        const CostMap& cost_map,
        const DirectionMap& dir_map,
        rclcpp::Logger& logger
    ) {
        if (!p.enable) return {std::nullopt, true};

        // 需要新目标？
        const bool need_new = (!goal_map) || ((now - goal_set_time).seconds() >= p.goal_timeout);
        if (need_new) {
            goal_map = find_goal(p, cost_map, dir_map, chassis_pose_map);
            goal_set_time = now;
            if (goal_map) {
                const auto s = sample_fields(cost_map, dir_map, *goal_map);
                if (s) {
                    if (is_safe_goal(p, *s)) {
                        RCLCPP_WARN(logger, "HAZARD_RECOVERY new goal=(%.2f, %.2f) (SAFE cost=%.1f step=%.3f)", goal_map->x(), goal_map->y(), s->cost, s->step_norm);
                    } else {
                        RCLCPP_WARN(logger, "HAZARD_RECOVERY new goal=(%.2f, %.2f) (UNSAFE cost=%.1f step=%.3f)", goal_map->x(), goal_map->y(), s->cost, s->step_norm);
                    }
                } else {
                    RCLCPP_WARN(logger, "HAZARD_RECOVERY new goal=(%.2f, %.2f) (field sample invalid)", goal_map->x(), goal_map->y());
                }
            } else {
                RCLCPP_ERROR(logger, "HAZARD_RECOVERY failed to find a safe goal");
            }
        }

        if (!goal_map) {
            return {std::nullopt, false};
        }

        // 退出判定：到达目标且位于安全区一段时间
        const double dist = (chassis_pose_map.head<2>() - *goal_map).norm();
        const Eigen::Vector2d gc = cost_map.map_coord_to_grid(chassis_pose_map.head<2>());
        const double cost_now = cost_map.interpolate(gc);
        const Eigen::Vector2d gd = dir_map.map_coord_to_grid(chassis_pose_map.head<2>());
        const double step_now = dir_map.interpolate(gd).norm();
        const bool safe = (cost_now < p.safe_cost_threshold) && (step_now < p.safe_step_norm_threshold);

        if (safe && dist <= p.goal_reached_dist) {
            if (!safe_since) safe_since = now;
            if ((now - *safe_since).seconds() >= p.safe_hold_time) {
                RCLCPP_INFO(logger, "Exit HAZARD_RECOVERY (safe for %.2f s)", (now - *safe_since).seconds());
                return {goal_map, true};
            }
        } else {
            safe_since = std::nullopt;
        }

        return {goal_map, false};
    }
};

// ═══════════════════ boost::statechart 状态声明 ══════════════

struct EvUpdate final : sc::event<EvUpdate> {
    explicit EvUpdate(const FsmInput& in) : input(in) {}
    const FsmInput& input;
};

struct StIdle;
struct StFollow;
struct StSpin;
struct StStopping;
struct StHazardRecovery;
struct StStuckReverse;

// ═══════════════════════ 状态机本体 ═════════════════════════

struct Machine final : sc::state_machine<Machine, StIdle> {
    Machine(const FsmParams& p, rclcpp::Logger lg) : params(p), logger(lg) {}

    // ──── 参数 ────
    FsmParams params;
    rclcpp::Logger logger;

    // ──── 当前状态 ────
    FsmState active_state = FsmState::IDLE;

    // ──── 输出缓冲 ────
    FsmOutput output;

    // ──── Stopping 目的态 ────
    NormalDest stop_dest = NormalDest::IDLE;

    // ──── Hazard Recovery 恢复后目标态 ────
    FsmState hazard_resume_state = FsmState::IDLE;

    // ──── 恢复目标搜索逻辑 ────
    HazardLogic hazard_logic;

    // ──── 卡住检测 ────
    StuckMonitor stuck_monitor;
    double last_cmd_velocity = 0.0;
    double last_cmd_omega = 0.0;
    rclcpp::Time last_cmd_time{0, 0, RCL_ROS_TIME};

    // ──── 倒车 ────
    rclcpp::Time reverse_start_time{0, 0, RCL_ROS_TIME};

    void clear_output() {
        output = {};
        output.state = active_state;
    }

    bool check_stuck(const FsmInput& in) {
        if (!params.stuck.enable) return false;
        if (last_cmd_time.nanoseconds() == 0) return false;
        return stuck_monitor.update(params.stuck, last_cmd_velocity, in.stamp, in.chassis_pose_map.head<2>());
    }
};

// ═══════════════════════ 各状态实现 ═════════════════════════

// ─────────────── IDLE ───────────────
struct StIdle final : sc::state<StIdle, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StIdle(my_context ctx) : sc::state<StIdle, Machine>(ctx) {
        context<Machine>().active_state = FsmState::IDLE;
        context<Machine>().stuck_monitor.reset();
        RCLCPP_INFO(context<Machine>().logger, "FSM -> IDLE");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        if (m.params.recovery.enable && in.merged_cost_map && in.global_direction_map) {
            if (HazardLogic::is_pose_hazard(m.params.recovery, *in.merged_cost_map, *in.global_direction_map, in.chassis_pose_map.head<2>())) {
                m.hazard_resume_state = FsmState::IDLE;
                return transit<StHazardRecovery>();
            }
        }

        const auto desired = compute_desired(in);
        if (desired == NormalDest::FOLLOW) return transit<StFollow>();
        if (desired == NormalDest::SPIN) return transit<StSpin>();
        return discard_event();
    }
};

// ─────────────── FOLLOW ───────────────
struct StFollow final : sc::state<StFollow, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StFollow(my_context ctx) : sc::state<StFollow, Machine>(ctx) {
        context<Machine>().active_state = FsmState::FOLLOW;
        context<Machine>().stuck_monitor.reset();
        RCLCPP_INFO(context<Machine>().logger, "FSM -> FOLLOW");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        if (m.check_stuck(in)) {
            return transit<StStuckReverse>();
        }

        const auto desired = compute_desired(in);
        if (desired == NormalDest::FOLLOW) return discard_event();

        m.stop_dest = desired;
        return transit<StStopping>();
    }
};

// ─────────────── SPIN ───────────────
struct StSpin final : sc::state<StSpin, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StSpin(my_context ctx) : sc::state<StSpin, Machine>(ctx) {
        context<Machine>().active_state = FsmState::SPIN;
        context<Machine>().stuck_monitor.reset();
        RCLCPP_INFO(context<Machine>().logger, "FSM -> SPIN");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        if (m.params.recovery.enable && in.merged_cost_map && in.global_direction_map) {
            if (HazardLogic::is_pose_hazard(m.params.recovery, *in.merged_cost_map, *in.global_direction_map, in.chassis_pose_map.head<2>())) {
                m.hazard_resume_state = FsmState::SPIN;
                return transit<StHazardRecovery>();
            }
        }

        const auto desired = compute_desired(in);
        if (desired == NormalDest::SPIN) return discard_event();

        m.stop_dest = desired;
        return transit<StStopping>();
    }
};

// ─────────────── STOPPING ───────────────
struct StStopping final : sc::state<StStopping, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StStopping(my_context ctx) : sc::state<StStopping, Machine>(ctx) {
        context<Machine>().active_state = FsmState::STOPPING;
        RCLCPP_INFO(
            context<Machine>().logger, "FSM -> STOPPING (dest=%s)",
            context<Machine>().stop_dest == NormalDest::IDLE ? "IDLE" :
            context<Machine>().stop_dest == NormalDest::FOLLOW ? "FOLLOW" : "SPIN"
        );
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        if (m.check_stuck(in)) {
            return transit<StStuckReverse>();
        }

        m.stop_dest = compute_desired(in);

        const double v = in.velocity;
        const double w = in.omega;
        const auto& tp = m.params.transition;

        switch (m.stop_dest) {
            case NormalDest::SPIN: {
                if (std::abs(v) < tp.follow_to_spin_vel_max) return transit<StSpin>();
                break;
            }
            case NormalDest::FOLLOW: {
                if (std::abs(w) < tp.spin_to_follow_omega_max) return transit<StFollow>();
                break;
            }
            case NormalDest::IDLE: {
                if (std::abs(v) < tp.to_idle_vel_max && std::abs(w) < tp.to_idle_omega_max) return transit<StIdle>();
                break;
            }
        }

        return discard_event();
    }
};

// ─────────────── HAZARD_RECOVERY ───────────────
struct StHazardRecovery final : sc::state<StHazardRecovery, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StHazardRecovery(my_context ctx) : sc::state<StHazardRecovery, Machine>(ctx) {
        auto& m = context<Machine>();
        m.active_state = FsmState::HAZARD_RECOVERY;
        m.hazard_logic.reset();
        m.stuck_monitor.reset();
        RCLCPP_WARN(m.logger, "FSM -> HAZARD_RECOVERY (resume=%s)", m.hazard_resume_state == FsmState::SPIN ? "SPIN" : "IDLE");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        if (m.check_stuck(in)) {
            return transit<StStuckReverse>();
        }

        if (!in.merged_cost_map || !in.global_direction_map) {
            return discard_event();
        }

        // 只做目标搜索和退出判定，不调用 MPC
        auto result = m.hazard_logic.step(
            m.params.recovery,
            in.stamp,
            in.chassis_pose_map,
            *in.merged_cost_map,
            *in.global_direction_map,
            m.logger
        );

        // 输出恢复目标点，供 NavigationController 调用 MPC
        m.output.recovery_goal_map = result.goal;

        if (result.finished) {
            m.output.recovery_finished = true;
            if (m.hazard_resume_state == FsmState::SPIN) {
                return transit<StSpin>();
            }
            return transit<StIdle>();
        }

        return discard_event();
    }
};

// ─────────────── STUCK_REVERSE ───────────────
struct StStuckReverse final : sc::state<StStuckReverse, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StStuckReverse(my_context ctx) : sc::state<StStuckReverse, Machine>(ctx) {
        auto& m = context<Machine>();
        m.active_state = FsmState::STUCK_REVERSE;
        m.reverse_start_time = m.last_cmd_time;
        m.stuck_monitor.reset();
        RCLCPP_WARN(m.logger, "FSM -> STUCK_REVERSE");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        if (!m.params.stuck.enable || !in.chassis_theta_imu_world) {
            m.output.clear_global_path = true;
            return transit<StIdle>();
        }

        const double elapsed = (in.stamp - m.reverse_start_time).seconds();
        if (elapsed >= m.params.stuck.reverse_duration) {
            RCLCPP_INFO(m.logger, "STUCK_REVERSE done (%.2f s) -> clearing path, entering IDLE", elapsed);
            m.output.clear_global_path = true;
            return transit<StIdle>();
        }

        // 输出倒车指令（简单运动学，不需要 MPC）
        FsmOutput::ReverseCmd cmd;
        cmd.velocity = -m.params.stuck.reverse_speed;
        cmd.theta_imu_world = *in.chassis_theta_imu_world;
        m.output.reverse_cmd = cmd;

        return discard_event();
    }
};

}  // anonymous namespace

// ═══════════════════════ Impl 桥接 ═══════════════════════════

struct ControlFsm::Impl {
    Impl(const FsmParams& params, rclcpp::Logger logger) : machine(params, logger) {
        machine.initiate();
    }

    Machine machine;
};

// ═══════════════════════ 公开接口 ════════════════════════════

ControlFsm::ControlFsm(const FsmParams& params, rclcpp::Logger logger) : impl_(std::make_unique<Impl>(params, logger)) {}

ControlFsm::~ControlFsm() = default;
ControlFsm::ControlFsm(ControlFsm&&) noexcept = default;
ControlFsm& ControlFsm::operator=(ControlFsm&&) noexcept = default;

FsmOutput ControlFsm::update(const FsmInput& input) {
    impl_->machine.clear_output();
    impl_->machine.process_event(EvUpdate(input));
    impl_->machine.output.state = impl_->machine.active_state;
    return impl_->machine.output;
}

FsmState ControlFsm::state() const {
    return impl_->machine.active_state;
}

void ControlFsm::on_chassis_cmd_published(double velocity, double omega, const rclcpp::Time& stamp) {
    impl_->machine.last_cmd_velocity = velocity;
    impl_->machine.last_cmd_omega = omega;
    impl_->machine.last_cmd_time = stamp;
}

}