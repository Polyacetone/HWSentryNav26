#pragma once
#include <cmath>

#include "internal/no_discard.hpp"

namespace ubs {

/**
 * @brief Trait used for the B-spline value type.
 * @tparam T_ The value type.
 */
template <typename T_>
struct ValueTypeTrait {
    /**
     * @brief Cast the value type to int.
     * @return The casted value.
     */
    UBS_NO_DISCARD static int toInt(const T_& val) {
        return static_cast<int>(val);
    }

    /**
     * @brief Calculate the power @f$ val^{ex} @f$.
     * @param[in] val The base.
     * @param[in] ex The exponent.
     * @return @f$ val^{ex} @f$
     */
    template <typename T>
    UBS_NO_DISCARD static T_ pow(const T_& val, T ex) {
        return std::pow(val, ex);
    }
};

} // namespace ubs
