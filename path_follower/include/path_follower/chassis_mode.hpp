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

} // namespace path_follower
