#pragma once
#include <array>
#include <vector>

#include <Eigen/Core>
#include <boost/multi_array.hpp>

namespace ubs {

/**
 * @brief The uniform B-spline control points trait.
 *
 * This class is used to access the control points container. In order to use a container as a control points type in
 * uniform B-spline one have to write a control points trait for that type.
 *
 * @tparam OutputType_ The control points element type.
 */
template <typename OutputType_>
struct ControlPointsTrait {
    static_assert(sizeof(OutputType_) == 0,
                  "No control points trait found for the specified type. Either the include of the specified control "
                  "points type is missing or a control points type for the specified type needs to be added.");

// Interface declaration only for documentation.
#ifdef DOXYGEN
    /** @brief The container used to store the control points. */
    using Container;

    /**
     * @brief Resize the container to the specified shape.
     * @param[in,out] c The control points container.
     * @param[in] shape The new shape.
     */
    template <std::size_t Dims>
    static void resize(Container& c, const std::array<int, Dims>& shape);

    /**
     * @param[in] c The control points container.
     * @return The total number of control points.
     */
    static int getNumElements(const Container& c);

    /**
     * @param[in] c The control points container.
     * @return The pointer to the first control point.
     */
    static const OutputType_* data(const Container& c);

    /** @copydoc data(const Container& c) */
    static OutputType_* data(Container& c);

    /**
     * The stride is the number of elements one needs to add to a pointer to get to the next one in the specified
     * dimension.
     * @param[in] c The control points container.
     * @param[in] dim The dimension.
     * @return The stride of the container in the specified dimension.
     */
    static int getStride(const Container& c, int dim);

    /**
     * @param[in] c The control points container.
     * @param[in] dim The dimension.
     * @return The number of control points in the specified dimension.
     */
    static int getSize(const Container& c, int dim);
#endif
};

} // namespace ubs
