#pragma once
#include <cmath>

#include "internal/no_discard.hpp"

namespace ubs {
namespace internal {
template <typename T>
UBS_NO_DISCARD static constexpr T sqr(T a) noexcept {
    return a * a;
}
} // namespace internal

/**
 * @brief Calculates @f$ a^n @f$ where n is an integer at compile time.
 * @param[in] a The base value.
 * @param[in] n The exponent.
 * @return Return @f$ a^n @f$.
 */
template <typename T>
UBS_NO_DISCARD static constexpr T power(T a, int n) noexcept {
    return n == 0 ? 1 : internal::sqr(ubs::power(a, n / 2)) * (n % 2 == 0 ? 1 : a);
}

} // namespace ubs
