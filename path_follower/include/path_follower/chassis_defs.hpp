#pragma once

#include <cstdint>
#include <utility>

namespace path_follower {

// ═══════════════════════ 底盘模式 ═════════════════════════════

enum class ChassisMode : uint8_t {
    NORMAL = 0,
    SPIN_SLOW = 1,
    SPIN_FAST = 2,
    STEP_UP_LEG_LONG = 3,
    STEP_UP_LEG_SHORT = 4,
    STEP_UP_JUMP = 5,
    STEP_DOWN_LEG_SHORT = 6,
    STEP_DOWN_JUMP = 7,
};

inline bool is_step_mode(const ChassisMode mode) {
    switch (mode) {
        case ChassisMode::STEP_UP_LEG_SHORT:
        case ChassisMode::STEP_UP_JUMP:
        case ChassisMode::STEP_UP_LEG_LONG:
        case ChassisMode::STEP_DOWN_LEG_SHORT:
        case ChassisMode::STEP_DOWN_JUMP:
            return true;
        default:
            return false;
    }
}

inline const char* mode_label(const ChassisMode m) {
    switch (m) {
        case ChassisMode::STEP_UP_LEG_SHORT: return "UP_LEG_SHORT";
        case ChassisMode::STEP_UP_JUMP: return "UP_JUMP";
        case ChassisMode::STEP_UP_LEG_LONG: return "UP_LEG_LONG";
        case ChassisMode::STEP_DOWN_LEG_SHORT: return "DOWN_LEG_SHORT";
        case ChassisMode::STEP_DOWN_JUMP: return "DOWN_JUMP";
        case ChassisMode::NORMAL: return "NORMAL";
        default: return "?";
    }
}

// ═══════════════════════ 底盘控制状态 ═════════════════════════

enum class ChassisControlState : uint8_t {
    STOPPED,
    BLOCKED,
    NORMAL,
};

enum class LegMode : uint8_t {
    DEAD = 0,
    RECOVERY = 1,
    FLIGHT = 2,
    JUMP = 3,
    MATURE = 4,
    STEP = 5,
    ABNORMAL = 6
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
        case LegMode::STEP:
            return ChassisControlState::STOPPED;
        case LegMode::FLIGHT:
        case LegMode::JUMP:
            return ChassisControlState::BLOCKED;
        case LegMode::MATURE:
            return ChassisControlState::NORMAL;
    }

    std::unreachable();
}

} // namespace path_follower
