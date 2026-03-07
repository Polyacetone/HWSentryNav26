#include <path_follower/state_machine.hpp>
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

inline FsmState desired_to_state(const NormalDest d) {
    switch (d) {
        case NormalDest::IDLE: return FsmState::IDLE;
        case NormalDest::FOLLOW: return FsmState::FOLLOW;
        case NormalDest::SPIN: return FsmState::SPIN;
    }
    return FsmState::IDLE;
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

// ═════════════════════ 危险恢复逻辑（FSM 仅负责进入/退出） ═════════════════════

struct HazardLogic {
    std::optional<rclcpp::Time> safe_since;

    void reset() { safe_since = std::nullopt; }

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

    bool should_exit(
        const RecoveryParams& p,
        const rclcpp::Time& now,
        const Eigen::Vector3d& chassis_pose_map,
        const CostMap& cost_map,
        const DirectionMap& dir_map,
        const std::optional<Eigen::Vector2d>& recovery_goal_map,
        rclcpp::Logger& logger
    ) {
        if (!p.enable) return true;
        if (!recovery_goal_map) {
            safe_since = std::nullopt;
            return false;
        }

        const Eigen::Vector2d pos = chassis_pose_map.head<2>();

        const Eigen::Vector2d gc = cost_map.map_coord_to_grid(pos);
        const double cost_now = cost_map.interpolate(gc);
        const Eigen::Vector2d gd = dir_map.map_coord_to_grid(pos);
        const double step_now = dir_map.interpolate(gd).norm();
        const bool safe = (cost_now < p.safe_cost_threshold) && (step_now < p.safe_step_norm_threshold);

        if (safe) {
            if (!safe_since) safe_since = now;
            if ((now - *safe_since).seconds() >= p.safe_hold_time) {
                RCLCPP_INFO(logger, "Exit HAZARD_RECOVERY (safe for %.2f s)", (now - *safe_since).seconds());
                return true;
            }
        } else {
            safe_since = std::nullopt;
        }

        return false;
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
struct StDead;
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

    // ──── DEAD 恢复后目标态（尽量恢复到进入前的状态） ────
    FsmState dead_resume_state = FsmState::IDLE;

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

        if (in.chassis_dead) {
            m.dead_resume_state = FsmState::IDLE;
            m.stuck_monitor.reset();
            return transit<StDead>();
        }

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

        if (in.chassis_dead) {
            m.dead_resume_state = FsmState::FOLLOW;
            m.stuck_monitor.reset();
            return transit<StDead>();
        }

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

        if (in.chassis_dead) {
            m.dead_resume_state = FsmState::SPIN;
            m.stuck_monitor.reset();
            return transit<StDead>();
        }

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

        if (in.chassis_dead) {
            // STOPPING 属于过渡态：失效恢复后直接回到当下 desired 的正常态即可
            m.dead_resume_state = desired_to_state(compute_desired(in));
            m.stuck_monitor.reset();
            return transit<StDead>();
        }

        if (m.check_stuck(in)) {
            return transit<StStuckReverse>();
        }

        m.stop_dest = compute_desired(in);

        // 使用上一周期的指令速度（而非实际速度）判断过渡条件，以保证指令信号的连续性
        const double v = m.last_cmd_velocity;
        const double w = m.last_cmd_omega;
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

// ─────────────── DEAD ───────────────
struct StDead final : sc::state<StDead, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StDead(my_context ctx) : sc::state<StDead, Machine>(ctx) {
        auto& m = context<Machine>();
        m.active_state = FsmState::DEAD;
        m.stuck_monitor.reset();
        RCLCPP_WARN(m.logger, "FSM -> DEAD");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        if (in.chassis_dead) {
            // 底盘仍处于失效模式：保持 DEAD
            return discard_event();
        }

        // 底盘恢复：优先恢复到进入 DEAD 前的可恢复态；否则回到当下 desired 正常态
        const auto desired = compute_desired(in);
        const FsmState desired_state = desired_to_state(desired);

        auto resume = m.dead_resume_state;

        // 不恢复这些“动作型/过渡型”状态，避免恢复瞬间发生倒车/过渡指令
        if (resume == FsmState::STOPPING || resume == FsmState::STUCK_REVERSE || resume == FsmState::DEAD) {
            resume = desired_state;
        }

        // 若恢复态与当前输入不兼容，则退化为 desired
        if (resume == FsmState::FOLLOW && !in.has_path) {
            resume = desired_state;
        }
        if (resume == FsmState::SPIN) {
            const bool can_spin = in.spin_high_priority || (!in.has_path);
            const bool should_spin = in.spin_requested && can_spin;
            if (!should_spin) resume = desired_state;
        }
        if (resume == FsmState::HAZARD_RECOVERY && !m.params.recovery.enable) {
            resume = desired_state;
        }

        switch (resume) {
            case FsmState::HAZARD_RECOVERY: return transit<StHazardRecovery>();
            case FsmState::FOLLOW: return transit<StFollow>();
            case FsmState::SPIN: return transit<StSpin>();
            case FsmState::IDLE: return transit<StIdle>();
            default: return transit<StIdle>();
        }
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

        if (in.chassis_dead) {
            m.dead_resume_state = FsmState::HAZARD_RECOVERY;
            m.stuck_monitor.reset();
            return transit<StDead>();
        }

        if (m.check_stuck(in)) {
            return transit<StStuckReverse>();
        }

        if (!in.merged_cost_map || !in.global_direction_map) {
            return discard_event();
        }

        // FSM 只负责退出判定；恢复目标点由 MainController 维护
        const bool finished = m.hazard_logic.should_exit(
            m.params.recovery,
            in.stamp,
            in.chassis_pose_map,
            *in.merged_cost_map,
            *in.global_direction_map,
            in.recovery_goal_map,
            m.logger
        );

        if (finished) {
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

        if (in.chassis_dead) {
            // 避免恢复瞬间继续倒车：恢复后回到当下 desired 正常态
            m.dead_resume_state = desired_to_state(compute_desired(in));
            m.stuck_monitor.reset();
            return transit<StDead>();
        }

        if (!m.params.stuck.enable) {
            m.output.clear_global_path = true;
            return transit<StIdle>();
        }

        const double elapsed = (in.stamp - m.reverse_start_time).seconds();
        if (elapsed >= m.params.stuck.reverse_duration) {
            RCLCPP_INFO(m.logger, "STUCK_REVERSE done (%.2f s) -> clearing path, entering IDLE", elapsed);
            m.output.clear_global_path = true;
            return transit<StIdle>();
        }

        return discard_event();
    }
};

} // anonymous namespace

// ═══════════════════════ Impl 桥接 ═══════════════════════════

struct StateMachine::Impl {
    Impl(const FsmParams& params, rclcpp::Logger logger) : machine(params, logger) {
        machine.initiate();
    }

    Machine machine;
};

// ═══════════════════════ 公开接口 ════════════════════════════

StateMachine::StateMachine(const FsmParams& params, rclcpp::Logger logger) : impl_(std::make_unique<Impl>(params, logger)) {}

StateMachine::~StateMachine() = default;
StateMachine::StateMachine(StateMachine&&) noexcept = default;
StateMachine& StateMachine::operator=(StateMachine&&) noexcept = default;

FsmOutput StateMachine::update(const FsmInput& input) {
    impl_->machine.clear_output();
    impl_->machine.process_event(EvUpdate(input));
    impl_->machine.output.state = impl_->machine.active_state;
    return impl_->machine.output;
}

FsmState StateMachine::state() const {
    return impl_->machine.active_state;
}

void StateMachine::on_chassis_cmd_published(double velocity, double omega, const rclcpp::Time& stamp) {
    impl_->machine.last_cmd_velocity = velocity;
    impl_->machine.last_cmd_omega = omega;
    impl_->machine.last_cmd_time = stamp;
}

}