#pragma once

#include <cstdint>
#include <utility>

namespace path_follower {

namespace chassis_mode {
inline constexpr uint8_t NORMAL = 0;
inline constexpr uint8_t SPIN_SLOW = 1;
inline constexpr uint8_t SPIN_FAST = 2;
} // namespace chassis_mode

inline bool is_step_mode(const uint8_t mode) {
    return mode != chassis_mode::NORMAL
        && mode != chassis_mode::SPIN_SLOW
        && mode != chassis_mode::SPIN_FAST;
}

enum class StepDirection : uint8_t {
    UP = 0,
    DOWN = 1,
};

enum class ChassisControlState : uint8_t {
    STOPPED = 0,
    BLOCKED = 1,
    NORMAL = 2,
};

enum class LegMode : uint8_t {
    DEAD = 0,
    RECOVERY = 1,
    FLIGHT = 2,
    JUMP = 3,
    MATURE = 4,
    STEP = 5,
    ABNORMAL = 6,
};

constexpr uint8_t COMP_STAGE_MATCH = 4u;

inline ChassisControlState classify_chassis_control_state(const uint8_t leg_mode, const uint8_t comp_stage) {
    if (comp_stage != COMP_STAGE_MATCH) {
        return ChassisControlState::STOPPED;
    }

    switch (static_cast<LegMode>(leg_mode)) {
        case LegMode::DEAD:
        case LegMode::RECOVERY:
        case LegMode::ABNORMAL:
            return ChassisControlState::STOPPED;
        case LegMode::STEP:
        case LegMode::FLIGHT:
        case LegMode::JUMP:
            return ChassisControlState::BLOCKED;
        case LegMode::MATURE:
            return ChassisControlState::NORMAL;
    }

    std::unreachable();
}

} // namespace path_follower
