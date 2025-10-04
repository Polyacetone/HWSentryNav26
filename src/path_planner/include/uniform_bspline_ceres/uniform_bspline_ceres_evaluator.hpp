#pragma once

#include <array>
#include <tuple>
#include <type_traits>

#include <uniform_bspline/utilities.hpp>

namespace ubs {

/**
 * @brief Helper class to evaluates a spline during optimization.
 *
 * The evaluator is usually passed to a ceres cost function. It supports autodiff and numeric diff cost
 * functions and dynamic autodiff and dynamic numeric diff cost functions.
 *
 * @tparam Spline_ The spline type.
 */
template <typename Spline_>
class UniformBSplineCeresEvaluator {
public:
    /** @brief The spline type. */
    using Spline = Spline_;

    /** @brief The degree of the spline. */
    static constexpr int Degree = Spline::Degree;
    /** @brief The order of the spline. */
    static constexpr int Order = Spline::Order;

    /** @brief The input dimensions of the spline. */
    static constexpr int InputDims = Spline::InputDims;
    /** @brief The output dimensions of the spline. */
    static constexpr int OutputDims = Spline::OutputDims;
    /** @brief The number of control points needed to evaluate the spline. */
    static constexpr int ControlPointsSupport = ubs::power(Order, InputDims);

    // As ceres uses double, the value type of the underlying spline must be double.
    static_assert(std::is_same<double, typename Spline::ValueType>::value, "Specified value type is not supported.");

    /**
     * @brief Constructor.
     * @param[in] basisVals The basis values.
     */
    explicit UniformBSplineCeresEvaluator(const std::array<double, ControlPointsSupport>& basisVals)
            : basisVals_(basisVals) {
    }

    /**
     * @brief Evaluates the uniform B-spline using the specified control points.
     * @param[in] controlPoints The control points.
     * @param[out] result The result.
     */
    template <typename T>
    void evaluate(T const* const* controlPoints, T* result) const {
        // Clear result.
        for (int j = 0; j < OutputDims; ++j) {
            result[j] = T(0.0);
        }

        // Calculate uniform B spline value.
        for (int i = 0; i < ControlPointsSupport; ++i) {
            for (int j = 0; j < OutputDims; ++j) {
                result[j] += controlPoints[i][j] * T(basisVals_[i]);
            }
        }
    }

    /**
     * @copybrief evaluate(T const* const* controlPoints, T* result) const
     *
     * This function takes the control point pointers as separate arguments. The parameter pointers must be kept in
     * order and the last element must be the result pointer (and non-cost).
     *
     * @param[in] controlPoint The first control point.
     * @param[in,out] ts The other control points. The last pointer is the result pointer.
     */
    template <typename T, typename... Ts>
    void evaluate(const T* controlPoint, Ts*... ts) const {
        static_assert(ControlPointsSupport == int(sizeof...(ts)),
                      "Invalid number of control points specified for evaluation");

        std::array<const T*, ControlPointsSupport + 1> cPs{{controlPoint, ts...}};

        // The last element of the control points is the place to store the output.
        T* result = std::get<ControlPointsSupport>(std::make_tuple(controlPoint, ts...));

        evaluate(cPs.data(), result);
    }

    /** @brief The basis values used to calculate the spline value. */
    std::array<double, ControlPointsSupport> basisVals_{};
};
} // namespace ubs

#include "internal/uniform_bspline_ceres_evaluator_impl.hpp"
