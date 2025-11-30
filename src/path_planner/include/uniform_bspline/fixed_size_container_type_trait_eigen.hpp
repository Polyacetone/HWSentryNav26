#pragma once
#include <Eigen/Core>

#include "fixed_size_container_type_trait.hpp"
#include "internal/no_discard.hpp"

namespace ubs {

/**
 * @brief Uniform B-spline output trait for multi-dimensional Eigen vectors.
 * \sa FixedSizeContainerTypeTrait
 * @tparam T_ The value type.
 * @tparam N_ The number of output dimensions.
 */
template <typename T_, int N_>
struct FixedSizeContainerTypeTrait<Eigen::Matrix<T_, N_, 1>> {
    static_assert(N_ > 0, "Only fixed size output dimensions supported.");

    using Type = Eigen::Matrix<T_, N_, 1>;
    using ValueType = T_;
    using Allocator = Eigen::aligned_allocator<Type>;

    static constexpr int Size = N_;
    static constexpr bool IsContinuous = true;

    UBS_NO_DISCARD static auto zero() {
        return Eigen::Matrix<T_, N_, 1>::Zero();
    }

    UBS_NO_DISCARD static auto ones() {
        return Eigen::Matrix<T_, N_, 1>::Ones();
    }

    UBS_NO_DISCARD static ValueType& get(Type& vec, int idx) {
        return vec[idx];
    }

    UBS_NO_DISCARD static const ValueType& get(const Type& vec, int idx) {
        return vec[idx];
    }

    UBS_NO_DISCARD static const ValueType* data(const Type& vec) {
        return vec.data();
    }

    UBS_NO_DISCARD static ValueType* data(Type& vec) {
        return vec.data();
    }

    UBS_NO_DISCARD static Type evalSmoothness(ValueType factor, const Type& p1, const Type& p2) {
        return factor * p1.array() * p2.array();
    }
};

} // namespace ubs
