#include <path_follower/control_fsm.hpp>

#include <boost/statechart/custom_reaction.hpp>
#include <boost/statechart/event.hpp>
#include <boost/statechart/state.hpp>
#include <boost/statechart/state_machine.hpp>

namespace path_follower {
namespace {

namespace sc = boost::statechart;

struct EvUpdate final : sc::event<EvUpdate> {
    explicit EvUpdate(ControlFsm::Inputs inputs_in) : inputs(inputs_in) {}
    ControlFsm::Inputs inputs;
};

struct StIdle;
struct StFollow;
struct StSpin;
struct StStopping;

struct Machine final : sc::state_machine<Machine, StIdle> {
    explicit Machine(ControlFsm::Params params_in) : params(params_in) {
        active_state = ControlFsm::State::IDLE;
        stop_dest = ControlFsm::Destination::IDLE;
    }

    static ControlFsm::Destination compute_desired(const ControlFsm::Inputs& in) {
        const bool can_spin = in.spin_high_priority || (!in.has_path);
        const bool should_spin = in.spin_requested && can_spin;
        if (should_spin) return ControlFsm::Destination::SPIN;
        if (in.has_path) return ControlFsm::Destination::FOLLOW;
        return ControlFsm::Destination::IDLE;
    }

    ControlFsm::Params params;
    ControlFsm::Inputs last_inputs{};
    ControlFsm::State active_state{ControlFsm::State::IDLE};
    ControlFsm::Destination stop_dest{ControlFsm::Destination::IDLE};
};

struct StIdle final : sc::state<StIdle, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;
    explicit StIdle(my_context ctx) : sc::state<StIdle, Machine>(ctx) {
        context<Machine>().active_state = ControlFsm::State::IDLE;
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        m.last_inputs = ev.inputs;
        const auto desired = Machine::compute_desired(ev.inputs);
        if (desired == ControlFsm::Destination::FOLLOW) return transit<StFollow>();
        if (desired == ControlFsm::Destination::SPIN) return transit<StSpin>();
        return discard_event();
    }
};

struct StFollow final : sc::state<StFollow, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;
    explicit StFollow(my_context ctx) : sc::state<StFollow, Machine>(ctx) {
        context<Machine>().active_state = ControlFsm::State::FOLLOW;
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        m.last_inputs = ev.inputs;
        const auto desired = Machine::compute_desired(ev.inputs);
        if (desired == ControlFsm::Destination::FOLLOW) return discard_event();

        // FOLLOW -> (SPIN/IDLE) 必须经过 STOPPING
        m.stop_dest = desired;
        return transit<StStopping>();
    }
};

struct StSpin final : sc::state<StSpin, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;
    explicit StSpin(my_context ctx) : sc::state<StSpin, Machine>(ctx) {
        context<Machine>().active_state = ControlFsm::State::SPIN;
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        m.last_inputs = ev.inputs;
        const auto desired = Machine::compute_desired(ev.inputs);
        if (desired == ControlFsm::Destination::SPIN) return discard_event();

        // SPIN -> (FOLLOW/IDLE) 必须经过 STOPPING
        m.stop_dest = desired;
        return transit<StStopping>();
    }
};

struct StStopping final : sc::state<StStopping, Machine> {
    using reactions = sc::custom_reaction<EvUpdate>;
    explicit StStopping(my_context ctx) : sc::state<StStopping, Machine>(ctx) {
        context<Machine>().active_state = ControlFsm::State::STOPPING;
    }

    sc::result react(const EvUpdate& ev) {
        auto& m = context<Machine>();
        m.last_inputs = ev.inputs;

        // STOPPING 过程中允许根据最新输入更新目的态
        m.stop_dest = Machine::compute_desired(ev.inputs);

        const double v = ev.inputs.velocity;
        const double w = ev.inputs.omega;

        switch (m.stop_dest) {
            case ControlFsm::Destination::SPIN: {
                if (std::abs(v) < m.params.follow_to_spin_vel_max) {
                    return transit<StSpin>();
                }
                break;
            }
            case ControlFsm::Destination::FOLLOW: {
                if (std::abs(w) < m.params.spin_to_follow_omega_max) {
                    return transit<StFollow>();
                }
                break;
            }
            case ControlFsm::Destination::IDLE: {
                if (std::abs(v) < m.params.to_idle_vel_max && std::abs(w) < m.params.to_idle_omega_max) {
                    return transit<StIdle>();
                }
                break;
            }
        }

        return discard_event();
    }
};

}

struct ControlFsm::Impl {
    explicit Impl(const Params& p) : machine(p) {
        machine.initiate();
    }

    Machine machine;
};

ControlFsm::ControlFsm(const Params& params) : impl_(std::make_unique<Impl>(params)) {}

ControlFsm::~ControlFsm() = default;

ControlFsm::ControlFsm(ControlFsm&&) noexcept = default;
ControlFsm& ControlFsm::operator=(ControlFsm&&) noexcept = default;

void ControlFsm::update(const Inputs& inputs) {
    impl_->machine.process_event(EvUpdate(inputs));
}

ControlFsm::State ControlFsm::state() const {
    return impl_->machine.active_state;
}

ControlFsm::Destination ControlFsm::destination() const {
    if (impl_->machine.active_state == State::STOPPING) {
        return impl_->machine.stop_dest;
    }
    return Machine::compute_desired(impl_->machine.last_inputs);
}

}