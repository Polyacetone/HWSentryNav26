#pragma once
#include <utility>

#include <ceres/autodiff_cost_function.h>

/**
 * @file
 * Used to determine the Ceres auto differential smoothing type.
 *
 * The number of residuals is the spline order times the output dimensions and the parameter dimensions are the output
 * dimensions with the control points support.
 */

namespace ubs {
namespace internal {

template <int CurVal_, int Order_, typename Spline_, typename CostFunctor_, typename Seq_>
struct AutoDiffSmoothingImpl;

template <int CurVal_, int Order_, typename Spline_, typename CostFunctor_, int... Ns_>
struct AutoDiffSmoothingImpl<CurVal_, Order_, Spline_, CostFunctor_, std::integer_sequence<int, Ns_...>> {
    using Type = typename AutoDiffSmoothingImpl<CurVal_ + 1,
                                                Order_,
                                                Spline_,
                                                CostFunctor_,
                                                std::integer_sequence<int, Ns_..., Spline_::OutputDims>>::Type;
};

template <int Order_, typename Spline_, typename CostFunctor_, int... Ns_>
struct AutoDiffSmoothingImpl<Order_, Order_, Spline_, CostFunctor_, std::integer_sequence<int, Ns_...>> {
    using Type = ceres::AutoDiffCostFunction<CostFunctor_, Order_ * Spline_::OutputDims, Ns_...>;
};

template <typename Spline, typename CostFunctor>
using AutoDiffSmoothingType =
    typename AutoDiffSmoothingImpl<0, Spline::Order, Spline, CostFunctor, std::integer_sequence<int>>::Type;

} // namespace internal
} // namespace ubs