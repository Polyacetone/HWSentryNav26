#pragma once

#include <memory>

namespace path_follower {

class ControlFsm {
public:
    struct Params {
        double follow_to_spin_vel_max;
        double spin_to_follow_omega_max;
        double to_idle_vel_max;
        double to_idle_omega_max;
    };

    struct Inputs {
        bool has_path;
        bool spin_requested;
        bool spin_high_priority;
        double velocity;
        double omega;
    };

    enum class State {
        IDLE,
        FOLLOW,
        SPIN,
        STOPPING,
    };

    enum class Destination {
        IDLE,
        FOLLOW,
        SPIN,
    };

    explicit ControlFsm(const Params& params);
    ~ControlFsm();

    ControlFsm(ControlFsm&&) noexcept;
    ControlFsm& operator=(ControlFsm&&) noexcept;

    ControlFsm(const ControlFsm&) = delete;
    ControlFsm& operator=(const ControlFsm&) = delete;

    void update(const Inputs& inputs);

    [[nodiscard]] State state() const;
    [[nodiscard]] Destination destination() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}