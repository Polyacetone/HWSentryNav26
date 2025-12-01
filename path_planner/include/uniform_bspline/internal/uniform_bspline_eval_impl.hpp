#pragma once

/**
 * @file
 * Definitions used for constexpr variables. This is only need for code pre C++17.
 */
#if not defined(__cplusplus) or __cplusplus < 201703L

namespace ubs {
namespace internal {

template <int InputDims_, int TotalDerivative_, int NumPartialDerivatives_, int PartialDerivativeIdx_>
constexpr Array<int, InputDims_>
    ComputeSmoothness<InputDims_, TotalDerivative_, NumPartialDerivatives_, PartialDerivativeIdx_>::PartialDerivatives_;

template <int InputDims_, int TotalDerivative_, int NumPartialDerivatives_, int PartialDerivativeIdx_>
constexpr double ComputeSmoothness<InputDims_, TotalDerivative_, NumPartialDerivatives_, PartialDerivativeIdx_>::
    PartialDerivativesMultiplicity_;

} // namespace internal
} // namespace ubs

#endif
