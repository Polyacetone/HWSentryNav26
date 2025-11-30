#pragma once

namespace ubs {
namespace internal {

/**
 * @brief Iterates through the input dimensional control points grid.
 * @tparam InputDims_ The input dimension.
 * @tparam CurDim_ The current dimension.
 */
template <int InputDims_, int CurDim_>
struct ForEachBSplineControlPoint {
    template <typename ControlPointsContainer, typename EvalFunc>
    static void apply(ControlPointsContainer&& controlPoints,
                      int idx,
                      const std::array<int, InputDims_>& sizes,
                      EvalFunc evalFunc) {
        const int stride = controlPoints.getStride(CurDim_);

        for (int i = 0; i < sizes[CurDim_]; ++i, idx += stride) {
            ForEachBSplineControlPoint<InputDims_, CurDim_ + 1>::apply(
                std::forward<ControlPointsContainer>(controlPoints), idx, sizes, evalFunc);
        }
    }
};

/**
 * @brief 'Loop' end, if the current dimension matches the input dimensions.
 */
template <int InputDims_>
struct ForEachBSplineControlPoint<InputDims_, InputDims_> {
    template <typename ControlPointsContainer, typename EvalFunc>
    static void apply(ControlPointsContainer&& controlPoints,
                      int idx,
                      const std::array<int, InputDims_>& /*sizes*/,
                      EvalFunc evalFunc) {
        evalFunc(controlPoints.at(idx));
    }
};

/**
 * @brief Iterates through the input dimensional control points grid.
 * @tparam InputDims_ The input dimension.
 * @tparam CurDim_ The current dimension.
 */
template <int InputDims_, int CurDim_>
struct TransformBSplineControlPoint {
    template <typename ControlPointContainerIn, typename ControlPointContainerOut, typename EvalFunc>
    static void apply(const ControlPointContainerIn& controlPointsIn,
                      ControlPointContainerOut& controlPointsOut,
                      int idxIn,
                      int idxOut,
                      const std::array<int, InputDims_>& sizes,
                      EvalFunc evalFunc) {
        const int strideIn = controlPointsIn.getStride(CurDim_);
        const int strideOut = controlPointsOut.getStride(CurDim_);

        for (int i = 0; i < sizes[CurDim_]; ++i, idxIn += strideIn, idxOut += strideOut) {
            TransformBSplineControlPoint<InputDims_, CurDim_ + 1>::apply(
                controlPointsIn, controlPointsOut, idxIn, idxOut, sizes, evalFunc);
        }
    }
};

/**
 * @brief 'Loop' end, if the current dimension matches the input dimensions.
 */
template <int InputDims_>
struct TransformBSplineControlPoint<InputDims_, InputDims_> {
    template <typename ControlPointContainerIn, typename ControlPointContainerOut, typename EvalFunc>
    static void apply(const ControlPointContainerIn& controlPointsIn,
                      ControlPointContainerOut& controlPointsOut,
                      int idxIn,
                      int idxOut,
                      const std::array<int, InputDims_>& /*sizes*/,
                      EvalFunc evalFunc) {
        controlPointsOut.at(idxOut) = evalFunc(controlPointsIn.at(idxIn));
    }
};

} // namespace internal
} // namespace ubs
