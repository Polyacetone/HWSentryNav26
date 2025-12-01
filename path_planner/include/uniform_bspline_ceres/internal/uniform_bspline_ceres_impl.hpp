#pragma once

/**
 * @file
 * Definitions used for constexpr variables. This is only need for code pre C++17.
 */
#if not defined(__cplusplus) or __cplusplus < 201703L

namespace ubs {

template <typename Spline_>
constexpr int UniformBSplineCeres<Spline_>::Degree;

template <typename Spline_>
constexpr int UniformBSplineCeres<Spline_>::Order;

template <typename Spline_>
constexpr int UniformBSplineCeres<Spline_>::InputDims;

template <typename Spline_>
constexpr int UniformBSplineCeres<Spline_>::OutputDims;

template <typename Spline_>
constexpr int UniformBSplineCeres<Spline_>::ControlPointsSupport;

} // namespace ubs

#endif
