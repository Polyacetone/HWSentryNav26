#pragma once
#include <Eigen/Core>

#include "smoothness_matrix.hpp"
#include "total_derivative.hpp"
#include "../fixed_size_container_type_trait.hpp"
#include "../value_type_trait.hpp"

namespace ubs {
/** @brief Internal namespace */
namespace internal {

/**
 * @brief Implementation to evaluate a spline at the specified position.
 *
 * This is implemented using template meta programming (TMP) as one needs to have one sum per dimension. A spline
 * can be evaluated using the following formula.
 * @f[
 * \mathbf{f}(\mathbf{t}) = \sum_{i_1} N_{i_1} \sum_{i_2} N_{i_2} \cdots \sum_{i_M} N_{i_M}
 * \mathbf{r}(\mathbf{t})
 * @f]
 *
 * @note As this class uses the internal of the spline, it must be a friend of the UniformBSpline class.
 * @tparam InputDims_ The total number of input dimensions.
 * @tparam CurDim_ The current dimension.
 */
template <typename Spline_, int InputDims_, int CurDim_>
struct EvaluateBSpline {
    /**
     * @brief Evaluates the spline at the specified position.
     *
     * @param[in] spline The spline.
     * @param[in] strides The strides of the control points.
     * @param[in] idx The current flattened index in the control points.
     * @param[in] basisVal The current basis value.
     * @param[in] pos The position.
     * @param[in] basisFunc The basis function.
     * @param[in] evalFunc The evaluation function.
     */
    template <typename T, typename EvalFunc>
    static void apply(const std::array<typename Spline_::BasisType, Spline_::InputDims>& fullBasis,
                      const std::array<int, Spline_::InputDims>& strides,
                      int idx,
                      T basisVal,
                      EvalFunc evalFunc) {
        const auto basis = fullBasis[CurDim_];
        for (int i = 0; i < Spline_::Order; ++i, idx += strides[CurDim_]) {
            const T newBasisVal = basisVal * basis[i];
            EvaluateBSpline<Spline_, InputDims_, CurDim_ + 1>::apply(fullBasis, strides, idx, newBasisVal, evalFunc);
        }
    }
};

/**
 * @brief 'Recursion' end, if the current dimension matches the input dimensions.
 * @tparam InputDims_ The input dimensions.
 */
template <typename Spline_, int InputDims_>
struct EvaluateBSpline<Spline_, InputDims_, InputDims_> {
    /**
     * @brief Call the evaluate function.
     */
    template <typename T, typename EvalFunc>
    static void apply(const std::array<typename Spline_::BasisType, Spline_::InputDims>& /*fullBasis*/,
                      const std::array<int, Spline_::InputDims>& /*strides*/,
                      int idx,
                      T basisVal,
                      EvalFunc evalFunc) {
        evalFunc(idx, basisVal);
    }
};

/**
 * @brief Implementation to evaluate the spline smoothness.
 * @note As this class uses the internal of the spline, it must be a friend of the UniformBSpline class.
 * @tparam TotalDerivative_ The order of the total derivative.
 * @tparam PartialDerivativeIdx_ The current partial derivative index.
 * @tparam InputDims_ The total number of input dimensions.
 * @tparam CurDim_ The current dimension.
 */
template <int TotalDerivative_, int PartialDerivativeIdx_, int InputDims_, int CurDim_>
class EvaluateBSplineSmoothness {
private:
    static constexpr auto FullPartialDerivatives_ =
        TotalDerivative<InputDims_, TotalDerivative_>::getPartialDerivatives();

    static constexpr int PartialDerivative_ =
        FullPartialDerivatives_[PartialDerivativeIdx_].partialDerivatives[CurDim_];

public:
    /**
     * @brief Evaluates the spline smoothness.
     * @param[in] spline The spline.
     * @param[in] idx1 The first index.
     * @param[in] idx2 The seconds index.
     * @param[in] strides The strides of the control points.
     * @param[in] factor The multiplication factor of the smoothness matrix.
     * @param[out] res The resulting smoothness.
     */
    template <typename Spline>
    static void apply(const Spline& spline,
                      int idx1,
                      int idx2,
                      const std::array<int, Spline::InputDims>& strides,
                      typename Spline::ValueType factor,
                      typename Spline::OutputType& res) {

        const auto smoothnessMatrix = UniformBSplineSmoothnessBasis<Spline::Order, PartialDerivative_>::matrix()
                                          .template cast<typename Spline::ValueType>()
                                          .eval();

        for (int p = 0; p < spline.getNumControlPoints(CurDim_) - Spline::Degree;
             ++p, idx1 += strides[CurDim_], idx2 += strides[CurDim_]) {
            int newIdx1 = idx1;
            for (int i = 0; i < Spline::Order; ++i, newIdx1 += strides[CurDim_]) {
                int newIdx2 = idx2;
                for (int j = 0; j < Spline::Order; ++j, newIdx2 += strides[CurDim_]) {
                    auto newFactor = factor * smoothnessMatrix(i, j);
                    EvaluateBSplineSmoothness<TotalDerivative_, PartialDerivativeIdx_, InputDims_, CurDim_ + 1>::apply(
                        spline, newIdx1, newIdx2, strides, newFactor, res);
                }
            }
        }
    }
};

/**
 * @brief 'Recursion' end, if the current dimension matches the input dimensions.
 * @tparam TotalDerivative_ The order of the total derivative.
 * @tparam PartialDerivativeIdx_ The current partial derivative index.
 * @tparam InputDims_ The input dimensions.
 */
template <int TotalDerivative_, int PartialDerivativeIdx_, int InputDims_>
class EvaluateBSplineSmoothness<TotalDerivative_, PartialDerivativeIdx_, InputDims_, InputDims_> {
public:
    template <typename Spline>
    static void apply(const Spline& spline,
                      int idx1,
                      int idx2,
                      const std::array<int, Spline::InputDims>& /*strides*/,
                      typename Spline::ValueType factor,
                      typename Spline::OutputType& res) {
        assert(idx1 < int(spline.controlPoints_.getNumElements()));
        assert(idx2 < int(spline.controlPoints_.getNumElements()));

        res += FixedSizeContainerTypeTrait<typename Spline::OutputType>::evalSmoothness(
            factor, spline.controlPoints_.at(idx1), spline.controlPoints_.at(idx2));
    }
};

/**
 * @brief Compute the smoothness with respect to the total derivative.
 *
 * To calculate the smoothness of the total derivative, the problem can be split up by calculating the smoothness for
 * each partial derivative, multiply that value by the multiplicity of that derivative and sum them up.
 *
 * @tparam InputDims_ The number of input dimensions.
 * @tparam TotalDerivative_ The order of the total derivative.
 * @tparam NumPartialDerivatives_ The total number of distinct partial derivatives.
 * @tparam PartialDerivativeIdx_ The current number of the partial derivative to compute the smoothness.
 */
template <int InputDims_, int TotalDerivative_, int NumPartialDerivatives_, int PartialDerivativeIdx_>
struct ComputeSmoothness {
private:
    static constexpr auto FullPartialDerivatives_ =
        TotalDerivative<InputDims_, TotalDerivative_>::getPartialDerivatives();
    static constexpr Array<int, InputDims_> PartialDerivatives_ =
        FullPartialDerivatives_[PartialDerivativeIdx_].partialDerivatives;
    static constexpr double PartialDerivativesMultiplicity_ =
        FullPartialDerivatives_[PartialDerivativeIdx_].multiplicity;

public:
    /**
     * @brief Compute the smoothness value of the partial derivatives needed to calculate the total derivative
     *        smoothness value
     * @param[in] spline The spline.
     * @param[in] strides The strides of the control points.
     * @param[out] res The smoothness value for each dimension.
     */
    template <typename Spline>
    static void apply(const Spline& spline,
                      const std::array<int, InputDims_>& strides,
                      typename Spline::OutputType& res) {
        static_assert(InputDims_ == Spline::InputDims, "Input dimension missmatch.");
        using T = typename Spline::ValueType;
        using OutputType = typename Spline::OutputType;

        // Calculate the scaling factor. The scaling factor depends on the multiplicity of the partial derivative,
        // the control points scale and the partial derivative.
        T scale = T(PartialDerivativesMultiplicity_);

        for (int dim = 0; dim < InputDims_; ++dim) {
            using ValueTypeTrait = ValueTypeTrait<T>;
            scale *= ValueTypeTrait::pow(spline.getScale(dim), 2 * PartialDerivatives_[dim] - 1);
        }

        // Calculate the smoothness value.
        OutputType smoothness = FixedSizeContainerTypeTrait<OutputType>::zero();
        EvaluateBSplineSmoothness<TotalDerivative_, PartialDerivativeIdx_, InputDims_, 0>::apply(
            spline, 0, 0, strides, T(1.0), smoothness);
        res += scale * smoothness;

        // Compute smoothness value for the next partial derivative.
        ComputeSmoothness<InputDims_, TotalDerivative_, NumPartialDerivatives_, PartialDerivativeIdx_ + 1>::apply(
            spline, strides, res);
    }
};

/**
 * @brief 'Loop' end, if the current dimension matches the input dimensions.
 * @tparam InputDims_ The input dimensions.
 */
template <int InputDims_, int TotalDerivative_, int NumPartialDerivatives_>
struct ComputeSmoothness<InputDims_, TotalDerivative_, NumPartialDerivatives_, NumPartialDerivatives_> {
    template <typename Spline>
    static void apply(const Spline& /*spline*/,
                      const std::array<int, InputDims_>& /*strides*/,
                      const typename Spline::OutputType& /*res*/) {
        // No more partial derivatives left. Nothing more to do.
    }
};

} // namespace internal
} // namespace ubs

#include "uniform_bspline_eval_impl.hpp"
