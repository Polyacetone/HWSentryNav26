#pragma once

#include <array>

#include <uniform_bspline/uniform_bspline.hpp>

#include "fixed_size_container_type_trait_ceres.hpp"
#include "value_type_trait_ceres.hpp"

namespace ubs {

/**
 * @brief The uniform B-spline generator.
 *
 * A uniform B-spline generator is used to generate splines in a ceres cost function. The generated spline can be used
 * also in autodiff.
 *
 * @tparam UniformBSpline_ A spline templated on the value type.
 */
template <template <typename ValueType> class UniformBSpline_>
class UniformBSplineCeresGenerator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /** @brief The spline type. */
    template <typename T>
    using UniformBSplineType = UniformBSpline_<T>;

    /** @brief The number of input dimension */
    static constexpr int InputDims = UniformBSplineType<double>::InputDims;
    /** @brief The input type if not during optimization */
    using InputType = typename UniformBSplineType<double>::InputType;

    /**
     * @brief The constructor.
     * @param[in] lowerBound The lower bound.
     * @param[in] upperBound The upper bound.
     * @param[in] shape The number of control points needed in each input dimension.
     */
    UniformBSplineCeresGenerator(InputType lowerBound, InputType upperBound, const std::array<int, InputDims>& shape)
            : lowerBound_{std::move(lowerBound)}, upperBound_{std::move(upperBound)}, shape_{shape} {
    }

    /**
     * @brief Generates a spline using the specified control points.
     * @param[in] controlPointsRaw The control points from ceres.
     * @param[in] If true, the generated spline allows extrapolation, otherwise not.
     * @return The spline.
     */
    template <typename T>
    UBS_NO_DISCARD UniformBSplineType<T> generate(const T* const* controlPointsRaw, bool extrapolate = false) const {
        using OptSpline = UniformBSplineType<T>;
        using OptInputType = typename OptSpline::InputType;
        using OptOutputType = typename OptSpline::OutputType;
        using OptControlPointsType = typename OptSpline::ControlPointsType;
        using OptControlPointsContainerType = typename OptSpline::ControlPointsContainerType;
        using OptInputContainerTrait = FixedSizeContainerTypeTrait<OptInputType>;
        using OptOutputContainerTrait = FixedSizeContainerTypeTrait<OptOutputType>;

        using InputContainerTrait = FixedSizeContainerTypeTrait<InputType>;

        // Convert lower bound.
        OptInputType lowerBound{};
        for (int i = 0; i < OptInputContainerTrait::Size; ++i) {
            OptInputContainerTrait::get(lowerBound, i) = T(InputContainerTrait::get(lowerBound_, i));
        }

        // Convert upper bound.
        OptInputType upperBound{};
        for (int i = 0; i < OptInputContainerTrait::Size; ++i) {
            OptInputContainerTrait::get(upperBound, i) = T(InputContainerTrait::get(upperBound_, i));
        }

        OptControlPointsType controlPoints{};
        ControlPointsTrait<OptControlPointsType>::resize(controlPoints, shape_);
        OptControlPointsContainerType controlPointContainer(lowerBound, upperBound, std::move(controlPoints));

        // Copy control points.
        int curControlPoints = 0;
        controlPointContainer.forEach([&](OptOutputType& v) {
            for (int i = 0; i < OptOutputContainerTrait::Size; ++i) {
                OptOutputContainerTrait::get(v, i) = controlPointsRaw[curControlPoints][i];
            }
            ++curControlPoints;
        });

        // Create spline.
        OptSpline spline(std::move(controlPointContainer));
        spline.setExtrapolate(extrapolate);
        return spline;
    }

private:
    /** @brief The lower bound. */
    InputType lowerBound_{};
    /** @brief The upper bound. */
    InputType upperBound_{};
    /** @brief The shape (number of control points needed to generate a spline. */
    std::array<int, InputDims> shape_{};
};

} // namespace ubs
