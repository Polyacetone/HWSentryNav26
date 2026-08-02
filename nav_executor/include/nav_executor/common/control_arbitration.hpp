#pragma once

#include <cstdint>

namespace nav_executor {

// 正常运动请求的无状态仲裁结果。安全恢复和物理交接仍由 PathExecutor FSM
// 执行；该结果表示物理条件允许时应由谁取得底盘控制权。
enum class ControlOwner : uint8_t {
    IDLE = 0,
    NAVIGATION = 1,
    LOW_PRIORITY_SPIN = 2,
    HIGH_PRIORITY_SPIN = 3,
};

enum class NavigationAccess : uint8_t {
    AVAILABLE = 0,     // 可替换执行产物、提交规划并接纳结果。
    LOCK_CURRENT = 1,  // 保留当前执行产物，但冻结替换和规划。
    REVOKED = 2,       // 当前执行产物失效，并冻结规划。
};

struct ControlArbitrationInput {
    bool navigation_ready = false;
    bool spin_requested = false;
    bool spin_high_priority = false;
};

[[nodiscard]] constexpr ControlOwner arbitrate_control(
    const ControlArbitrationInput& input
) {
    if (input.spin_requested && input.spin_high_priority) {
        return ControlOwner::HIGH_PRIORITY_SPIN;
    }
    if (input.navigation_ready) {
        return ControlOwner::NAVIGATION;
    }
    if (input.spin_requested) {
        return ControlOwner::LOW_PRIORITY_SPIN;
    }
    return ControlOwner::IDLE;
}

[[nodiscard]] constexpr bool is_spin_owner(const ControlOwner owner) {
    return owner == ControlOwner::LOW_PRIORITY_SPIN
        || owner == ControlOwner::HIGH_PRIORITY_SPIN;
}

[[nodiscard]] constexpr NavigationAccess arbitrate_navigation_access(
    const ControlOwner owner,
    const bool path_replacement_locked,
    const bool planning_suspended
) {
    // COMMITTED 保护正在执行的物理过程，不能被任何普通请求撤销。
    if (path_replacement_locked) {
        return NavigationAccess::LOCK_CURRENT;
    }
    // 安全状态仍拥有实际底盘控制权，但高优先级请求可以先撤销导航执行产物。
    if (owner == ControlOwner::HIGH_PRIORITY_SPIN) {
        return NavigationAccess::REVOKED;
    }
    if (planning_suspended) {
        return NavigationAccess::LOCK_CURRENT;
    }
    return NavigationAccess::AVAILABLE;
}

} // namespace nav_executor
