#pragma once

#include <cstdint>

namespace path_follower {

enum class ChassisMode : uint8_t {
    NORMAL = 0,
    SPIN_SLOW = 1,
    SPIN_FAST = 2,
    STEP_UP_LEG_LONG = 3,    // 长伸腿上台阶
    STEP_UP_LEG_SHORT = 4,   // 短伸腿上台阶（上坡）
    STEP_UP_JUMP = 5,        // 跳跃上台阶
    STEP_DOWN_LEG_SHORT = 6, // 短伸腿下台阶
    STEP_DOWN_JUMP = 7,      // 跳跃下台阶
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

} // namespace path_follower
