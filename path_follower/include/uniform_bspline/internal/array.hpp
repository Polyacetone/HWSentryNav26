#pragma once

#include "no_discard.hpp"

namespace ubs {
/** @brief Internal namespace */
namespace internal {

#if __cplusplus >= 201703L

template <typename T, int N>
using Array = std::array<T, N>;

#else

/**
 * @brief A minimal implementation of a constexpr aware std::array.
 *
 * The non-const operator[] method is only supported for C++17 and above.
 *
 * @tparam T_ The type.
 * @tparam N_ The number of elements.
 */
template <typename T_, int N_>
struct Array {
    T_ data[N_]; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)

    UBS_NO_DISCARD constexpr const T_& operator[](int n) const {
        return data[n];
    }

    UBS_NO_DISCARD constexpr T_& operator[](int n) {
        return data[n];
    }

    UBS_NO_DISCARD constexpr int size() const {
        return N_;
    }

    UBS_NO_DISCARD constexpr T_* begin() {
        return data;
    }

    UBS_NO_DISCARD constexpr const T_* begin() const {
        return data;
    }

    UBS_NO_DISCARD constexpr T_* end() {
        return data + N_; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    UBS_NO_DISCARD constexpr const T_* end() const {
        return data + N_; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    UBS_NO_DISCARD constexpr const T_* cbegin() const {
        return data;
    }

    UBS_NO_DISCARD constexpr const T_* cend() const {
        return data + N_; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }
};
#endif

} // namespace internal
} // namespace ubs
