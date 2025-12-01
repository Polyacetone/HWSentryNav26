#pragma once
#include <algorithm>
#include <array>
#include <stdexcept>

#include "control_points_trait.hpp"
#include "control_points_trait_eigen.hpp"
#include "control_points_trait_multi_array.hpp"
#include "control_points_trait_std_container.hpp"
#include "fixed_size_container_type_trait.hpp"
#include "fixed_size_container_type_trait_eigen.hpp"
#include "internal/control_points_for_each.hpp"
#include "internal/gtest_friend.hpp"
#include "internal/no_discard.hpp"

namespace ubs {

/**
 * @brief Control points container class.
 *
 * This class provides a container for control points. It can be used to manipulate, resize, etc. elements of
 * the control points in a generic way (independent of the control points storage type). It uses the control points
 * traits to access the control points storage.
 *
 * In addition to the control points, this class also handles the lower and upper bounds associated with the control
 * points.
 *
 * @note For the template parameter @sa UniformBSpline.
 * @tparam ValueType_ The value type.
 * @tparam Degree_ The degree of the uniform B-spline.
 * @tparam InputType_ The input type.
 * @tparam OutputType_ The output type.
 * @tparam ControlPointsType_ The control points type.
 */
template <typename ValueType_, int Degree_, typename InputType_, typename OutputType_, typename ControlPointsType_>
class ControlPointsContainer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /** @brief The spline degree. */
    static constexpr int Degree = Degree_;
    /** @brief The spline order. */
    static constexpr int Order = Degree_ + 1;
    /** @brief The number of input dimensions. */
    static constexpr int InputDims = FixedSizeContainerTypeTrait<InputType_>::Size;
    /** @brief The number of output dimensions. */
    static constexpr int OutputDims = FixedSizeContainerTypeTrait<OutputType_>::Size;

    static_assert(InputDims > 0, "The number of input dimensions must be positive.");
    static_assert(OutputDims > 0, "The number of output dimensions must be positive.");

    /** @brief The value type. */
    using ValueType = ValueType_;

    /** @brief The input type. */
    using InputType = InputType_;
    /** @brief The output type. */
    using OutputType = OutputType_;

    /** @brief The control points type. */
    using ControlPointsType = ControlPointsType_;

    /**
     * @brief Default constructor.
     *
     * The lowest number of possible control points is chosen and initialized to zero. The lower bound is set to zero
     * and the upper bound is set to one.
     */
    ControlPointsContainer()
            : ControlPointsContainer(FixedSizeContainerTypeTrait<InputType>::zero(),
                                     FixedSizeContainerTypeTrait<InputType>::ones()) {
    }

    /**
     * @brief Initialization with control points.
     *
     * The lower bound is set to zero and the upper bound is set to one.
     * @param[in] controlPoints The control points. The size in each dimension must be at least order.
     */
    explicit ControlPointsContainer(const ControlPointsType& controlPoints)
            : ControlPointsContainer(FixedSizeContainerTypeTrait<InputType>::zero(),
                                     FixedSizeContainerTypeTrait<InputType>::ones(),
                                     controlPoints) {
    }

    /** @copydoc ControlPointsContainer(const ControlPointsType& controlPoints) */
    explicit ControlPointsContainer(ControlPointsType&& controlPoints)
            : ControlPointsContainer(FixedSizeContainerTypeTrait<InputType>::zero(),
                                     FixedSizeContainerTypeTrait<InputType>::ones(),
                                     std::move(controlPoints)) {
    }

    /**
     * @brief Initialization with lower and upper bound.
     *
     * The lowest number of possible control points is chosen and initialized to zero.
     * @param[in] lowerBound The lower bound.
     * @param[in] upperBound The upper bound.
     */
    ControlPointsContainer(InputType lowerBound, InputType upperBound)
            : lowerBound_(std::move(lowerBound)), upperBound_(std::move(upperBound)) {
        // Initialize control points.
        std::array<int, InputDims> shape{};
        std::fill(shape.begin(), shape.end(), Order);
        ControlPointsTrait<ControlPointsType>::resize(controlPoints_, shape);
        std::fill_n(data(), getNumElements(), FixedSizeContainerTypeTrait<OutputType>::zero());

        updatedShape();
    }

    /**
     * @brief Initialize with lower bound, upper bound and control points.
     * @param[in] lowerBound The lower bound.
     * @param[in] upperBound The upper bound.
     * @param[in] controlPoints The control points.
     */
    ControlPointsContainer(InputType lowerBound, InputType upperBound, const ControlPointsType& controlPoints)
            : lowerBound_(std::move(lowerBound)), upperBound_(std::move(upperBound)),
              controlPoints_(std::move(controlPoints)) {
        updatedShape();
    }

    /** @copydoc ControlPointsContainer(const InputType& lowerBound, const InputType& upperBound, const
     * ControlPointsType& controlPoints) */
    ControlPointsContainer(InputType lowerBound, InputType upperBound, ControlPointsType&& controlPoints)
            : lowerBound_(std::move(lowerBound)), upperBound_(std::move(upperBound)),
              controlPoints_(std::move(controlPoints)) {
        updatedShape();
    }

    /**
     * @brief Set new control points.
     * @param[in] controlPoints The new control points.
     */
    void set(const ControlPointsType& controlPoints) {
        controlPoints_ = controlPoints;
        updatedShape();
    }

    /** @copydoc set(const ControlPointsType& controlPoints) */
    void set(ControlPointsType&& controlPoints) {
        controlPoints_ = std::move(controlPoints);
        updatedShape();
    }

    /**
     * @brief Set new bounds.
     * @note lowerBound < upperBound for each dimensions.
     * @param[in] lowerBound The lower range of the input type.
     * @param[in] upperBound The upper range of the input type.
     */
    void setBounds(const InputType& lowerBound, const InputType& upperBound) {
        lowerBound_ = lowerBound;
        upperBound_ = upperBound;
        updatedShape();
    }

    /**
     * @return The lower bound in each dimension.
     */
    UBS_NO_DISCARD const InputType& getLowerBound() const {
        return lowerBound_;
    }

    /**
     * @return The upper bound in each dimension.
     */
    UBS_NO_DISCARD const InputType& getUpperBound() const {
        return upperBound_;
    }

    /**
     * @brief Return the lower bound for the specified dimension.
     * @param[in] dim The dimension.
     * @return The lower bound.
     */
    UBS_NO_DISCARD const ValueType& getLowerBound(int dim) const {
        return FixedSizeContainerTypeTrait<InputType>::get(lowerBound_, dim);
    }

    /**
     * @brief Return the upper bound for the specified dimension.
     * @param[in] dim The dimension.
     * @return The upper bound.
     */
    UBS_NO_DISCARD const ValueType& getUpperBound(int dim) const {
        return FixedSizeContainerTypeTrait<InputType>::get(upperBound_, dim);
    }

    /**
     * @brief Resize the control points.
     * @param[in] dimensions The new shape.
     */
    void resize(const std::array<int, InputDims>& dimensions) {
        ControlPointsTrait<ControlPointsType>::resize(controlPoints_, dimensions);
        updatedShape();
    }

    /**
     * @return The total number of control points.
     */
    UBS_NO_DISCARD int getNumElements() const {
        return ControlPointsTrait<ControlPointsType>::getNumElements(controlPoints_);
    }

    /**
     * @return The data pointer.
     */
    UBS_NO_DISCARD OutputType* data() {
        return ControlPointsTrait<ControlPointsType>::data(controlPoints_);
    }

    /**
     * @copydoc data()
     */
    UBS_NO_DISCARD const OutputType* data() const {
        return ControlPointsTrait<ControlPointsType>::data(controlPoints_);
    }

    /**
     * @parain[in] flatIdx The index into the control points array.
     * @return The element at the specified flat index.
     */
    UBS_NO_DISCARD OutputType& at(int flatIdx) {
        assert(isIdxValid(flatIdx));
        return data()[flatIdx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    /**
     * @copydoc at(int flatIdx)
     */
    UBS_NO_DISCARD const OutputType& at(int flatIdx) const {
        assert(isIdxValid(flatIdx));
        return data()[flatIdx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    /**
     * @brief Return the non-const internally stored control points container.
     *
     * @note Do not change the shape of the control points, as bad things can happen.
     * @return The control points.
     */
    UBS_NO_DISCARD ControlPointsType& get() {
        return controlPoints_;
    }

    /**
     * @brief Return the const internally stored control points container.
     * @return The control points.
     */
    UBS_NO_DISCARD const ControlPointsType& get() const {
        return controlPoints_;
    }

    /**
     * @param[in] dim The dimension.
     * @return The number of control points in the specified dimension.
     */
    UBS_NO_DISCARD int getSize(int dim) const {
        return ControlPointsTrait<ControlPointsType>::getSize(controlPoints_, dim);
    }

    /**
     * @brief Return the scale.
     *
     * The scale is dependent on the number of control points and the lower and upper bound.
     * @return The scales in each dimension.
     */
    UBS_NO_DISCARD const InputType& getScales() const {
        return scales_;
    }

    /**
     * @param[in] dim The dimension.
     * @return The scale in the specified dimension.
     */
    UBS_NO_DISCARD const ValueType& getScale(int dim) const {
        assert(dim >= 0 && dim < (int)scales_.size());
        return scales_[dim];
    }

    /**
     * @return The strides in each dimension.
     */
    UBS_NO_DISCARD const std::array<int, InputDims>& getStrides() const {
        return strides_;
    }

    /**
     * @param[in] dim The dimension.
     * @return The stride for the specified dimension in element size.
     */
    UBS_NO_DISCARD int getStride(int dim) const {
        return ControlPointsTrait<ControlPointsType>::getStride(controlPoints_, dim);
    }

    /**
     * @return An array containing the shape (number of control points) in each dimension.
     */
    UBS_NO_DISCARD std::array<int, InputDims> getShape() const {
        std::array<int, InputDims> shape{};

        for (int dim = 0; dim < InputDims; ++dim) {
            shape[dim] = getSize(dim);
        }

        return shape;
    }

    /**
     * @brief Iterates over each control point and calls the callback function.
     * @param[in] func The callback function called for each control point. The function must have the following
     *                 signature: void(const OutputType&).
     */
    template <typename Func>
    void forEach(Func func) const {
        forEach(0, getShape(), func);
    }

    /** @copydoc forEach(Func func) const */
    template <typename Func>
    void forEach(Func func) {
        forEach(0, getShape(), func);
    }

    /**
     * @brief Iterates of the control points starting at the given start index. The number of control points iterated
     *        can be specified by the sizes parameter.
     * @param[in] startIdx The start index, where iteration starts.
     * @param[in] sizes The number of control points in each dimension, which should be iterated.
     * @param[in] func The callback function called for each control point. The function must have the following
     *                 signature: void(const OutputType&).
     */
    template <typename Func>
    void forEach(int startIdx, const std::array<int, InputDims>& sizes, Func func) const {
        internal::ForEachBSplineControlPoint<InputDims, 0>::apply(*this, startIdx, sizes, func);
    }

    /** @copydoc forEach(int startIdx, const std::array<int, InputDims>& sizes, Func func) const */
    template <typename Func>
    void forEach(int startIdx, const std::array<int, InputDims>& sizes, Func func) {
        internal::ForEachBSplineControlPoint<InputDims, 0>::apply(*this, startIdx, sizes, func);
    }

    /**
     * @brief Transform applied the given function to each control points and stores the result in another control
     *        points container.
     * @note The shape of the out control points container must be the same.
     * @param[out] outControlPoints The output control points container, to which the transformed data is written to.
     * @param[in] func The callback function called for each control point. The function must have the following
     *                 signature: OtherOutputType(const OutputType&).
     */
    template <typename OutControlPoints, typename Func>
    void transform(OutControlPoints& outControlPoints, Func func) const {
        // Spline must have the same number input dimension and the same shape, otherwise its not known how to do a
        // transform.
        static_assert(OutControlPoints::InputDims == InputDims);
        assert(outControlPoints.getShape() == getShape());

        internal::TransformBSplineControlPoint<InputDims, 0>::apply(*this, outControlPoints, 0, 0, getShape(), func);
    }

private:
    FRIEND_TEST(ControlPointsContainer, IsIdxValid);

    /**
     * @brief Recompute the scale factors based on the control points and the input scales and updates the stride
     *        array.
     * @note This has to be done whenever the bounds or shape of the control points changes.
     */
    void updatedShape() {
        using FixedSizeContainerTypeTrait = FixedSizeContainerTypeTrait<InputType>;

        for (int i = 0; i < InputDims; ++i) {
            const int size = getSize(i);
            if (size < Order) {
                throw std::invalid_argument("Number of control points must be at least the order of the spline.");
            }

            const ValueType& lowerBound = FixedSizeContainerTypeTrait::get(lowerBound_, i);
            const ValueType& upperBound = FixedSizeContainerTypeTrait::get(upperBound_, i);
            if (upperBound < lowerBound) {
                throw std::invalid_argument("Upper bound is smaller than lower bound.");
            }

            scales_[i] = static_cast<ValueType>(size - Degree);
            if (lowerBound < upperBound) {
                scales_[i] /= upperBound - lowerBound;
            }

            strides_[i] = getStride(i);
        }
    }

    /**
     * @brief Check, if index is a valid flat index into the control points container.
     * @param[in] flatIdx The index to check.
     * @return True, if it is valid, otherwise false.
     */
    UBS_NO_DISCARD bool isIdxValid(int flatIdx) const {
        for (int i = 0; i < InputDims; ++i) {
            const int pos = flatIdx / getStride(i);
            if (pos < 0 || pos >= getSize(i)) {
                return false;
            }

            flatIdx -= pos * getStride(i);
        }

        return true;
    }

    /** @brief The variable lowerBound. */
    InputType lowerBound_{FixedSizeContainerTypeTrait<InputType>::zero()};

    /** @brief The variable upperBound. */
    InputType upperBound_{FixedSizeContainerTypeTrait<InputType>::ones()};

    /** @brief The control points. */
    ControlPointsType controlPoints_{};

    /** @brief The strides. */
    std::array<int, InputDims> strides_{};

    /** @brief The scale factors in each spline dimension. */
    std::array<ValueType, InputDims> scales_{};
};

} // namespace ubs

#include "internal/control_points_container_impl.hpp"
