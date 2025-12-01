#pragma once

/**
 * @file
 * Definitions used for constexpr variables. This is only need for code pre C++17.
 */
#if not defined(__cplusplus) or __cplusplus < 201703L

namespace ubs {
template <typename ValueType_, int Degree_, typename InputType_, typename OutputType_, typename ControlPointsType_>
constexpr int ControlPointsContainer<ValueType_, Degree_, InputType_, OutputType_, ControlPointsType_>::Degree;

template <typename ValueType_, int Degree_, typename InputType_, typename OutputType_, typename ControlPointsType_>
constexpr int ControlPointsContainer<ValueType_, Degree_, InputType_, OutputType_, ControlPointsType_>::Order;

template <typename ValueType_, int Degree_, typename InputType_, typename OutputType_, typename ControlPointsType_>
constexpr int ControlPointsContainer<ValueType_, Degree_, InputType_, OutputType_, ControlPointsType_>::InputDims;

template <typename ValueType_, int Degree_, typename InputType_, typename OutputType_, typename ControlPointsType_>
constexpr int ControlPointsContainer<ValueType_, Degree_, InputType_, OutputType_, ControlPointsType_>::OutputDims;
} // namespace ubs

#endif
