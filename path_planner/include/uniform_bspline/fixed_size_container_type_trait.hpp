#pragma once
#include <Eigen/Core>

#include "fixed_size_container_type_trait_default.hpp"
#include "utilities.hpp"

namespace ubs {

/**
 * @brief The fixed size container type trait.
 *
 * This trait is used by the UniformBSpline class for the input and output type. The size of those containers needs to
 * be known at compile time.
 *
 * @tparam T_ The container type.
 */
template <typename T_>
struct FixedSizeContainerTypeTrait : public FixedSizeContainerTypeTraitDefault<T_> {
#ifdef DOXYGEN
    /** @brief The output dimensions. */
    static constexpr int Size;

    /** @brief Indicates, if the container is continuous. If this is true, the data function needs to be implemented. */
    static constexpr bool IsContinuous;

    /** @brief The output type. */
    using Type;

    /** @brief The underlying value type. */
    using ValueType;

    /** @brief The allocator used to allocate the output type. */
    using Allocator;

    /** @brief The zero element. */
    static Type zero();

    /** @brief The one element. */
    static Type ones();

    /**
     * @brief Element access of the output type.
     * @param[in] vec The value.
     * @param[in] idx The index.
     * @return The value at the specified index.
     */
    static ValueType& get(Type& vec, int idx);

    /** @copydoc get(Type& vec, int idx) */
    static const ValueType& get(const Type& vec, int idx);

    /**
     * @note This function needs only to be implemented if IsContinuous = true.
     * @param[in] vec The value.
     * @return A pointer to the first element.
     */
    static const Type* data(const Type& vec);

    /** @copydoc data(const Type& vec) */
    static Type* data(Type& vec);

    /**
     * @brief This function is used during smoothness evaluation.
     *
     * This function has to compute the following:
     * @f[
     * s \left( \mathbf{p_1} \circ \mathbf{p_2} \right)
     * @f]
     * where @f$ \circ @f$ is the hadamard (element-wise) product of the two vectors @f$ \mathbf{p_1} @f$ and @f$
     * \mathbf{p_2} @f$.
     *
     * @param[in] factor The scalar multiplication factor.
     * @param[in] p1 The first point.
     * @param[in] p2 The second point.
     * @return @f$ s \left( \mathbf{p_1} \circ \mathbf{p_2} \right) @f$
     */
    static Type evalSmoothness(ValueType factor, const Type& p1, const Type& p2);
#endif
};

} // namespace ubs
