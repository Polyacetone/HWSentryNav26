#include <path_follower/state_machine.hpp>

#include <boost/statechart/custom_reaction.hpp>
#include <boost/statechart/event.hpp>
#include <boost/statechart/state.hpp>
#include <boost/statechart/state_machine.hpp>
#include <Eigen/Core>
#include <rclcpp/logging.hpp>

namespace path_follower {
namespace {

namespace sc = boost::statechart;

// Stopping 的目标态（Hub 路由出口）
enum class DestState { IDLE, FIXED, SPIN, FOLLOW };

struct EvUpdate final : sc::event<EvUpdate> {
    explicit EvUpdate(const FsmInput& in) : input(in) {}
    const FsmInput& input;
};

struct StIdle;
struct StFixed;
struct StFollow;
struct StSpin;
struct StStopping;
struct StStuckReverse;
struct StHazardRecovery;
struct StWaitReplan;
struct StStepping;

struct Machine final : sc::state_machine<Machine, StIdle> {
    Machine(const FsmParams& p, rclcpp::Logger lg) : params(p), logger(lg) {}

    FsmParams params;
    rclcpp::Logger logger;

    FsmState active_state = FsmState::IDLE;
    FsmOutput output;

    DestState stopping_dest = DestState::IDLE;
    std::chrono::steady_clock::time_point stopping_start_time;
    bool replan_after_recovery = false;

    // 以下为各状态构造参数槽（由源状态写入、目标状态构造函数消费）
    std::chrono::steady_clock::time_point pending_reverse_start_time;
    Eigen::Vector2d pending_reverse_start_pos = Eigen::Vector2d::Zero();
    std::chrono::steady_clock::time_point pending_wait_replan_start_time;

    void clear_output() {
        output = {};
        output.state = active_state;
    }

    bool stopping_ready(const FsmInput& in) const {
        const auto& t = params.transition;
        switch (stopping_dest) {
            case DestState::IDLE:
            case DestState::FIXED:
                return std::abs(in.velocity) < t.to_idle_vel_max &&
                    std::abs(in.omega) < t.to_idle_omega_max;
            case DestState::SPIN:
                // Follow -> Spin: primarily gate by |v| to avoid spinning while sliding.
                return std::abs(in.velocity) < t.follow_to_spin_vel_max &&
                    std::abs(in.omega) < t.to_idle_omega_max;
            case DestState::FOLLOW:
                // Spin -> Follow: primarily gate by |omega| to avoid leaving spin while still rotating.
                return std::abs(in.velocity) < t.to_idle_vel_max &&
                    std::abs(in.omega) < t.spin_to_follow_omega_max;
        }
        return false;
    }
};

struct StIdle final : sc::state<StIdle, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StIdle(my_context ctx) : sc::state<StIdle, Machine>(ctx) {
        auto& m = context<Machine>();
        m.active_state = FsmState::IDLE;
        RCLCPP_INFO(m.logger, "FSM -> IDLE");
    }

    sc::result react(const EvUpdate& ev) {
        const auto& in = ev.input;

        if (in.is_hazard) {
            return transit<StHazardRecovery>();
        }
        if (in.has_new_path) {
            return transit<StFollow>();
        }
        if (in.spin_requested) {
            return transit<StSpin>();
        }

        return discard_event();
    }
};

struct StFixed final : sc::state<StFixed, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StFixed(my_context ctx) : sc::state<StFixed, Machine>(ctx) {
        auto& m = context<Machine>();
        m.active_state = FsmState::FIXED;
        RCLCPP_INFO(m.logger, "FSM -> FIXED");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        if (in.is_stuck) {
            m.replan_after_recovery = true;
            m.pending_reverse_start_time = in.stamp;
            m.pending_reverse_start_pos = in.chassis_pos_map;
            return transit<StStuckReverse>();
        }
        if (in.spin_requested && in.spin_high_priority) {
            m.stopping_dest = DestState::SPIN;
            m.stopping_start_time = in.stamp;
            return transit<StStopping>();
        }
        if (in.has_new_path) {
            return transit<StFollow>();
        }

        return discard_event();
    }
};

struct StFollow final : sc::state<StFollow, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StFollow(my_context ctx) : sc::state<StFollow, Machine>(ctx) {
        auto& m = context<Machine>();
        m.active_state = FsmState::FOLLOW;
        RCLCPP_INFO(m.logger, "FSM -> FOLLOW");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        if (in.step_active) {
            return transit<StStepping>();
        }
        if (in.replan_requested) {
            m.pending_wait_replan_start_time = in.stamp;
            return transit<StWaitReplan>();
        }

        // 路径被取消/清空：FOLLOW 不应“卡死”在无路径状态。
        // 走 STOPPING 做平滑减速，并根据外部请求选择落点。
        if (!in.has_path) {
            const bool should_spin = in.spin_requested && (in.spin_high_priority || (!in.fixed_goal_flag));
            if (should_spin) {
                m.stopping_dest = DestState::SPIN;
            } else if (in.fixed_goal_flag) {
                m.stopping_dest = DestState::FIXED;
            } else {
                m.stopping_dest = DestState::IDLE;
            }
            m.stopping_start_time = in.stamp;
            return transit<StStopping>();
        }

        if (in.is_stuck) {
            m.replan_after_recovery = true;
            m.pending_reverse_start_time = in.stamp;
            m.pending_reverse_start_pos = in.chassis_pos_map;
            return transit<StStuckReverse>();
        }
        if (in.spin_requested && in.spin_high_priority) {
            m.stopping_dest = DestState::SPIN;
            m.stopping_start_time = in.stamp;
            return transit<StStopping>();
        }
        if (in.reach_goal) {
            if (in.fixed_goal_flag) {
                m.stopping_dest = DestState::FIXED;
            } else {
                m.stopping_dest = DestState::IDLE;
            }
            m.stopping_start_time = in.stamp;
            return transit<StStopping>();
        }

        return discard_event();
    }
};

struct StStepping final : sc::state<StStepping, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StStepping(my_context ctx) : sc::state<StStepping, Machine>(ctx) {
        auto& m = context<Machine>();
        m.active_state = FsmState::STEPPING;
        RCLCPP_INFO(m.logger, "FSM -> STEPPING");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        // latch_ttl 超时直接进 STUCK_REVERSE，无需经过 FOLLOW 确认
        if (in.step_ttl_expired) {
            m.replan_after_recovery = false;
            m.pending_reverse_start_time = in.stamp;
            m.pending_reverse_start_pos = in.chassis_pos_map;
            return transit<StStuckReverse>();
        }

        // 台阶中水平推进可能真卡住，stuck 先于 step_active 检查。
        if (in.is_stuck) {
            m.replan_after_recovery = false;
            m.pending_reverse_start_time = in.stamp;
            m.pending_reverse_start_pos = in.chassis_pos_map;
            return transit<StStuckReverse>();
        }

        if (in.step_active) {
            return discard_event();
        }

        // 如果 stepping 完成时要求重规划（例如 stepping 期间收到了外部路径更新），
        // 直接进入 WAIT_REPLAN 等待新路径，避免使用陈旧路径。
        if (in.replan_requested) {
            m.pending_wait_replan_start_time = in.stamp;
            return transit<StWaitReplan>();
        }

        return transit<StFollow>();
    }
};

struct StSpin final : sc::state<StSpin, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StSpin(my_context ctx) : sc::state<StSpin, Machine>(ctx) {
        auto& m = context<Machine>();
        m.active_state = FsmState::SPIN;
        RCLCPP_INFO(m.logger, "FSM -> SPIN");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        const bool keep_spinning = in.spin_requested && (in.spin_high_priority || (!in.has_path && !in.fixed_goal_flag));

        // SPIN 不检测 stuck：小陀螺模式是下位机伺服，且本来线速度就 ≈ 0，check_stuck 不会触发。
        if (in.is_hazard) {
            return transit<StHazardRecovery>();
        }
        if (!keep_spinning) {
            if (in.has_path) {
                m.stopping_dest = DestState::FOLLOW;
            } else if (in.fixed_goal_flag) {
                m.stopping_dest = DestState::FIXED;
            } else {
                m.stopping_dest = DestState::IDLE;
            }
            m.stopping_start_time = in.stamp;
            return transit<StStopping>();
        }

        return discard_event();
    }
};

struct StStopping final : sc::state<StStopping, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StStopping(my_context ctx) : sc::state<StStopping, Machine>(ctx) {
        auto& m = context<Machine>();
        m.active_state = FsmState::STOPPING;
        RCLCPP_INFO(m.logger, "FSM -> STOPPING");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        // STOPPING 不检测 stuck：减速中指令速度未归零但位移小是正常现象，检 stuck 会导致每次常规停车误报。
        if (m.stopping_dest != DestState::FOLLOW && in.has_new_path) {
            return transit<StFollow>();
        }

        const bool timeout = std::chrono::duration<double>(in.stamp - m.stopping_start_time).count() > m.params.transition.stopping_timeout;
        if (m.stopping_ready(in) || timeout) {
            switch (m.stopping_dest) {
                case DestState::IDLE:
                    m.output.consume_global_path = true;
                    return transit<StIdle>();
                case DestState::FIXED:
                    m.output.consume_global_path = true;
                    return transit<StFixed>();
                case DestState::SPIN: return transit<StSpin>();
                case DestState::FOLLOW: return transit<StFollow>();
            }
        }

        return discard_event();
    }
};

struct StStuckReverse final : sc::state<StStuckReverse, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StStuckReverse(my_context ctx) : sc::state<StStuckReverse, Machine>(ctx) {
        auto& m = context<Machine>();
        m.active_state = FsmState::STUCK_REVERSE;
        mature_accumulated_ = 0.0;
        last_mature_stamp_ = m.pending_reverse_start_time;
        entry_pos_ = m.pending_reverse_start_pos;
        RCLCPP_WARN(m.logger, "FSM -> STUCK_REVERSE");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        if (in.command_blocked) {
            mature_accumulated_ += std::chrono::duration<double>(in.stamp - last_mature_stamp_).count();
            last_mature_stamp_ = in.stamp;
            return discard_event();
        }

        const double mature_elapsed = mature_accumulated_
            + std::chrono::duration<double>(in.stamp - last_mature_stamp_).count();
        const double displacement = (in.chassis_pos_map - entry_pos_).norm();

        // 主退出条件：里程计检测到足够位移
        if (displacement >= m.params.stuck.reverse_displacement) {
            RCLCPP_WARN(
                m.logger,
                "STUCK_REVERSE: displaced %.2f m >= %.2f m, exiting to HAZARD_RECOVERY",
                displacement, m.params.stuck.reverse_displacement
            );
            return transit<StHazardRecovery>();
        }

        // 安全网：MATURE 下累计超时仍未达到位移，打印 ERROR 表示严重异常
        if (mature_elapsed >= m.params.stuck.reverse_timeout) {
            RCLCPP_ERROR(
                m.logger,
                "STUCK_REVERSE TIMEOUT: mature elapsed %.1f s >= %.1f s but displacement only %.2f m < %.2f m — robot may be physically stuck",
                mature_elapsed, m.params.stuck.reverse_timeout,
                displacement, m.params.stuck.reverse_displacement
            );
            return transit<StHazardRecovery>();
        }

        return discard_event();
    }

private:
    double mature_accumulated_ = 0.0;
    std::chrono::steady_clock::time_point last_mature_stamp_;
    Eigen::Vector2d entry_pos_ = Eigen::Vector2d::Zero();
};

struct StHazardRecovery final : sc::state<StHazardRecovery, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StHazardRecovery(my_context ctx) : sc::state<StHazardRecovery, Machine>(ctx) {
        auto& m = context<Machine>();
        m.active_state = FsmState::HAZARD_RECOVERY;
        RCLCPP_WARN(m.logger, "FSM -> HAZARD_RECOVERY");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        if (in.is_stuck) {
            m.replan_after_recovery = false;
            m.pending_reverse_start_time = in.stamp;
            m.pending_reverse_start_pos = in.chassis_pos_map;
            return transit<StStuckReverse>();
        }
        if (in.is_recovery_safe) {
            m.output.recovery_finished = true;

            if (m.replan_after_recovery) {
                m.replan_after_recovery = false;
m.pending_wait_replan_start_time = in.stamp;
            return transit<StWaitReplan>();
            }

            const bool should_spin = in.spin_requested && (in.spin_high_priority || (!in.has_path && !in.fixed_goal_flag));

            if (should_spin) {
                return transit<StSpin>();
            }

            if (in.has_path) {
                return transit<StFollow>();
            }

            if (in.fixed_goal_flag) {
                return transit<StFixed>();
            }

            m.output.consume_global_path = true;
            return transit<StIdle>();
        }

        return discard_event();
    }
};

struct StWaitReplan final : sc::state<StWaitReplan, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;

    explicit StWaitReplan(my_context ctx) : sc::state<StWaitReplan, Machine>(ctx) {
        auto& m = context<Machine>();
        m.active_state = FsmState::WAIT_REPLAN;
        start_time_ = m.pending_wait_replan_start_time;
        m.output.consume_global_path = true;
        m.output.request_replan = true;
        RCLCPP_INFO(m.logger, "FSM -> WAIT_REPLAN");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        if (in.has_new_path) {
            return transit<StFollow>();
        }

        if (in.replan_failed) {
            m.output.consume_global_path = true;
            RCLCPP_WARN(m.logger, "WAIT_REPLAN: replan failed (empty path)");

            const bool should_spin = in.spin_requested && (in.spin_high_priority || (!in.fixed_goal_flag));
            if (should_spin) {
                return transit<StSpin>();
            }
            return transit<StIdle>();
        }

        const bool timeout = std::chrono::duration<double>(in.stamp - start_time_).count() > m.params.transition.wait_replan_timeout;
        if (timeout) {
            m.output.consume_global_path = true;
            RCLCPP_WARN(m.logger, "WAIT_REPLAN timed out after %.2f s without a valid path", m.params.transition.wait_replan_timeout);

            const bool should_spin = in.spin_requested && (in.spin_high_priority || (!in.fixed_goal_flag));
            if (should_spin) {
                return transit<StSpin>();
            }
            return transit<StIdle>();
        }

        return discard_event();
    }

private:
    std::chrono::steady_clock::time_point start_time_;
};

} // anonymous namespace

struct StateMachine::Impl {
    Impl(const FsmParams& params, rclcpp::Logger logger) : machine(params, logger) {
        machine.initiate();
    }

    Machine machine;
};

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

} // namespace path_follower
