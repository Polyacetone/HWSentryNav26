#pragma once
#include <memory>

#include "internal/no_discard.hpp"

namespace ubs {

/**
 * @brief This class provides access to the input / output type used in a uniform B-spline.
 *
 * The default one is a single dimension value input / output.
 *
 * \sa FixedSizeContainerTypeTrait
 * @tparam T_ The value type.
 */
template <typename T_>
struct FixedSizeContainerTypeTraitDefault {
    static constexpr int Size = 1;
    static constexpr bool IsContinuous = true;

    using Type = T_;
    using ValueType = T_;
    using Allocator = std::allocator<Type>;

    UBS_NO_DISCARD static Type zero() {
        return Type(0.0);
    }

    UBS_NO_DISCARD static Type ones() {
        return Type(1.0);
    }

    UBS_NO_DISCARD static ValueType& get(Type& vec, int idx) {
        assert(idx == 0);
        (void)idx;

        return vec;
    }

    UBS_NO_DISCARD static const ValueType& get(const Type& vec, int idx) {
        assert(idx == 0);
        (void)idx;

        return vec;
    }

    UBS_NO_DISCARD static const Type* data(const Type& vec) {
        return &vec;
    }

    UBS_NO_DISCARD static Type* data(Type& vec) {
        return &vec;
    }

    UBS_NO_DISCARD static Type evalSmoothness(ValueType factor, const Type& p1, const Type& p2) {
        return factor * p1 * p2;
    }
};

} // namespace ubs
