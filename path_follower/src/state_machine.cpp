#include <path_follower/state_machine.hpp>

#include <boost/statechart/custom_reaction.hpp>
#include <boost/statechart/event.hpp>
#include <boost/statechart/state.hpp>
#include <boost/statechart/state_machine.hpp>
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

struct Machine final : sc::state_machine<Machine, StIdle> {
    Machine(const FsmParams& p, rclcpp::Logger lg) : params(p), logger(lg) {}

    FsmParams params;
    rclcpp::Logger logger;

    FsmState active_state = FsmState::IDLE;
    FsmOutput output;

    DestState stopping_dest = DestState::IDLE;
    rclcpp::Time stopping_start_time{0, 0, RCL_ROS_TIME};
    rclcpp::Time reverse_start_time{0, 0, RCL_ROS_TIME};

    void clear_output() {
        output = {};
        output.state = active_state;
    }

    bool stopping_ready(const FsmInput& in) const {
        return std::abs(in.velocity) < params.transition.to_idle_vel_max &&
            std::abs(in.omega) < params.transition.to_idle_omega_max;
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
            m.reverse_start_time = in.stamp;
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

        if (in.is_stuck) {
            m.reverse_start_time = in.stamp;
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

        if (in.is_hazard) {
            return transit<StHazardRecovery>();
        }
        if (in.is_stuck) {
            m.reverse_start_time = in.stamp;
            return transit<StStuckReverse>();
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

        if (in.is_stuck) {
            m.reverse_start_time = in.stamp;
            return transit<StStuckReverse>();
        }

        if (m.stopping_dest != DestState::FOLLOW && in.has_new_path) {
            return transit<StFollow>();
        }

        const bool timeout = (in.stamp - m.stopping_start_time).seconds() > m.params.transition.stopping_timeout;
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
        RCLCPP_WARN(m.logger, "FSM -> STUCK_REVERSE");
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        const auto& in = ev.input;

        const double elapsed = (in.stamp - m.reverse_start_time).seconds();
        if (elapsed > m.params.stuck.reverse_duration) {
            // 脱困链条固定为：倒车 -> HazardRecovery
            m.output.consume_global_path = true;
            return transit<StHazardRecovery>();
        }

        return discard_event();
    }
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
            m.reverse_start_time = in.stamp;
            return transit<StStuckReverse>();
        }
        if (in.is_recovery_safe) {
            m.output.recovery_finished = true;

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