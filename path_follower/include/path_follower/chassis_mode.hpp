#pragma once

#include <cstdint>

namespace path_follower {

enum class ChassisMode : uint8_t {
    NORMAL = 0,
    SPIN_SLOW = 1,
    SPIN_FAST = 2,
    STEP_UP_LEG = 3,
    STEP_UP_JUMP = 4,
    STEP_DOWN_LEG = 5,
    STEP_DOWN_JUMP = 6,
};

} // namespace path_follower
