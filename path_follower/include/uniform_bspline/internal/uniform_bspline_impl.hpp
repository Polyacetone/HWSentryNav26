#pragma once

/**
 * @file
 * Definitions used for constexpr variables. This is only need for code pre C++17.
 */
#if not defined(__cplusplus) or __cplusplus < 201703L

namespace ubs {

template <typename T_, int Degree_, typename InputType_, typename OutputType_, typename ControlPointsType_>
constexpr int UniformBSpline<T_, Degree_, InputType_, OutputType_, ControlPointsType_>::Degree;

template <typename T_, int Degree_, typename InputType_, typename OutputType_, typename ControlPointsType_>
constexpr int UniformBSpline<T_, Degree_, InputType_, OutputType_, ControlPointsType_>::Order;

template <typename T_, int Degree_, typename InputType_, typename OutputType_, typename ControlPointsType_>
constexpr int UniformBSpline<T_, Degree_, InputType_, OutputType_, ControlPointsType_>::InputDims;

template <typename T_, int Degree_, typename InputType_, typename OutputType_, typename ControlPointsType_>
constexpr int UniformBSpline<T_, Degree_, InputType_, OutputType_, ControlPointsType_>::OutputDims;

} // namespace ubs

#endif
