#pragma once
#include <array>
#include <cassert>
#include <type_traits>

#include <Eigen/Core>

#include "control_points_container.hpp"
#include "fixed_size_container_type_trait.hpp"
#include "fixed_size_container_type_trait_eigen.hpp"
#include "multi_array.hpp"
#include "value_type_trait.hpp"
#include "internal/base_matrix.hpp"
#include "internal/gtest_friend.hpp"
#include "internal/no_discard.hpp"
#include "internal/uniform_bspline_eval.hpp"

namespace ubs {

/**
 * @brief A uniform B-spline.
 *
 * An implementation of a uniform B-spline from @f$ \mathbb{R}^n \rightarrow \mathbb{R}^m @f$. Uniform means, that the
 * knot vector of the B-spline is equally distributed. Using such a knot vector makes the computation much more
 * efficient, as the basis can be precomputed.
 *
 * For an introduction to B-splines see <https://en.wikipedia.org/wiki/B-spline>.
 *
 * The input and output dimensions of the spline is defined by the input and output type.
 *
 * @tparam ValueType_ The value type.
 * @tparam Degree_ The spline degree.
 * @tparam InputType_ The input type. A FixedSizeContainerTypeTrait must be available for that type.
 * @tparam OutputType_ The output type. A FixedSizeContainerTypeTrait must be available for that type.
 * @tparam ControlPointsType_ The control points type. A ControlPointsTrait must be available for that
 *                            type.
 */
template <typename ValueType_,
          int Degree_,
          typename InputType_,
          typename OutputType_,
          typename ControlPointsType_ = ubs::MultiArray<OutputType_,
                                                        FixedSizeContainerTypeTrait<InputType_>::Size,
                                                        typename FixedSizeContainerTypeTrait<OutputType_>::Allocator>>
class UniformBSpline {
    // Friends for classes, which need to be partial specialized. (These classes cannot be functions as in class
    // partial specialization is not allowed.)
    template <typename Spline_, int InputDims_, int CurDim_>
    friend struct internal::EvaluateBSpline;
    template <int TotalDerivative_, int PartialDerivativeIdx_, int InputDims_, int CurDim_>
    friend class internal::EvaluateBSplineSmoothness;

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

    // If one would like to use a higher degree spline, the basis matrices needs to be generated. The code generation
    // is written in a Mathematica script located at scripts/generate_bspline_basis_cpp_files.
    static_assert(Degree >= 1 && Degree <= 5, "Unsupported degree specified.");

    /** @brief The value type. */
    using ValueType = ValueType_;

    /** @brief The input type. */
    using InputType = InputType_;
    /** @brief The output type. */
    using OutputType = OutputType_;

    /** @brief The B-spline basis type. */
    using BasisType = Eigen::Matrix<ValueType, Order, 1>;
    /** @brief The control points type. */
    using ControlPointsType = ControlPointsType_;
    /** @brief The control points container type. */
    using ControlPointsContainerType =
        ControlPointsContainer<ValueType, Degree, InputType, OutputType, ControlPointsType>;

    /**
     * @brief Default constructor.
     *
     * Creates a uniform B-spline for which all control points are initialized with zero.
     * The number of control points in each dimension is the lowest number possible, which is Order.
     * The lower bound is set to zero and the upper bound is set to one in each dimension.
     */
    UniformBSpline() = default;

    /**
     * @brief Constructs a uniform B-spline using the specified control points.
     *
     * The number of control points for each dimension must be at least the order of the spline.
     * The lower bound is set to zero and the upper bound is set to one in each dimension.
     *
     * @param[in] controlPoints The control points.
     */
    explicit UniformBSpline(const ControlPointsType& controlPoints) : controlPoints_(controlPoints) {
    }

    /**
     * @brief Constructs a uniform B-spline using the specified lower bound and upper bound.
     *
     * All the control points are zero initialized. The number of control points in each dimension is the lowest
     * number possible, which is Order.
     *
     * @param[in] lowerBound The lower bound.
     * @param[in] upperBound The upper bound.
     */
    UniformBSpline(const InputType& lowerBound, const InputType& upperBound) : controlPoints_(lowerBound, upperBound) {
    }

    /**
     * @brief Constructs a uniform B-spline using the specified control points, lower bound and upper bound.
     *
     * @note The number of control points for each dimension must be at least the order of the spline.
     * @param[in] controlPoints The control points.
     * @param[in] lowerBound The lower bound.
     * @param[in] upperBound The upper bound.
     */
    UniformBSpline(const InputType& lowerBound, const InputType& upperBound, const ControlPointsType& controlPoints)
            : controlPoints_(lowerBound, upperBound, controlPoints) {
    }

    /**
     * @brief Construct a new Uniform B Spline object using a control points container.
     *
     * @param[in] controlPointsContainer The control points container.
     */
    explicit UniformBSpline(const ControlPointsContainerType& controlPointsContainer)
            : controlPoints_(controlPointsContainer) {
    }

    /** @copydoc UniformBSpline(const ControlPointsContainerType& controlPointsContainer) */
    explicit UniformBSpline(ControlPointsContainerType&& controlPointsContainer)
            : controlPoints_(std::move(controlPointsContainer)) {
    }

    /**
     * @brief Set new control points.
     * @param[in] controlPoints The new control points.
     */
    void setControlPoints(const ControlPointsType& controlPoints) {
        controlPoints_.set(controlPoints);
    }

    /** @copydoc setControlPoints() */
    void setControlPoints(ControlPointsType&& controlPoints) {
        controlPoints_.set(std::move(controlPoints));
    }

    /**
     * @brief Set new bounds.
     * @note lowerBound < upperBound for each dimensions.
     * @param[in] lowerBound The lower range of the input type.
     * @param[in] upperBound The upper range of the input type.
     */
    void setBounds(const InputType& lowerBound, const InputType& upperBound) {
        controlPoints_.setBounds(lowerBound, upperBound);
    }

    /**
     * @brief Checks, whether the specified position is between lower and upper bound.
     * @param[in] pos The position.
     * @return True, if this point can be used to evaluate the B-spline, otherwise false.
     */
    UBS_NO_DISCARD bool inRange(const InputType& pos) const {
        using ContainerTrait = FixedSizeContainerTypeTrait<InputType>;
        for (int dim = 0; dim < InputDims; ++dim) {
            const auto& val = ContainerTrait::get(pos, dim);
            if (val < getLowerBound(dim) || val > getUpperBound(dim)) {
                return false;
            }
        }

        return true;
    }

    /**
     * @return The lower bound in each dimension.
     */
    UBS_NO_DISCARD const InputType& getLowerBound() const {
        return controlPoints_.getLowerBound();
    }

    /**
     * @return The upper bound in each dimension.
     */
    UBS_NO_DISCARD const InputType& getUpperBound() const {
        return controlPoints_.getUpperBound();
    }

    /**
     * @brief Return the lower bound for the specified dimension.
     * @param[in] dim The dimension.
     * @return The lower bound.
     */
    UBS_NO_DISCARD ValueType getLowerBound(int dim) const {
        return controlPoints_.getLowerBound(dim);
    }

    /**
     * @brief Return the upper bound for the specified dimension.
     * @param[in] dim The dimension.
     * @return The upper bound.
     */
    UBS_NO_DISCARD ValueType getUpperBound(int dim) const {
        return controlPoints_.getUpperBound(dim);
    }

    /**
     * @brief Returns the number of control points of the specified dimension.
     * @param[in] dim The dimension.
     * @return The number of control points.
     */
    UBS_NO_DISCARD int getNumControlPoints(int dim) const {
        return int(controlPoints_.getSize(dim));
    }

    /**
     * @return The constant control points.
     */
    UBS_NO_DISCARD const ControlPointsType& getControlPoints() const {
        return controlPoints_.get();
    }

    /**
     * @brief Returns the control points.
     * @return The control points.
     */
    UBS_NO_DISCARD const ControlPointsContainerType& getControlPointsContainer() const {
        return controlPoints_;
    }

    /** @copydoc getControlPointsContainer() const */
    UBS_NO_DISCARD ControlPointsContainerType& getControlPointsContainer() {
        return controlPoints_;
    }

    /**
     * @brief Returns the scale of the B-spline.
     *
     * The scales are depending on the number of control points and the lower and upper bounds.
     *
     * @param[in] dim The dimension.
     * @return The scale in the specified dimension.
     */
    UBS_NO_DISCARD ValueType getScale(int dim) const {
        return controlPoints_.getScale(dim);
    }

    /**
     * @brief Set extrapolation.
     *
     * If set to true, the spline can be queried outside of the bounds. As an extrapolation the closest spline segment
     * is used.
     *
     * @param[in] v True, if extrapolating should be enabled, otherwise false.
     */
    void setExtrapolate(bool v) {
        extrapolate_ = v;
    }

    /**
     * @sa setExtrapolate(bool v)
     * @return True, if extrapolation is enabled, otherwise false.
     */
    UBS_NO_DISCARD bool isExtrapolating() const {
        return extrapolate_;
    }

    /**
     * @brief Determine the span and value in the specified dimension at the specified value.
     *
     * The span is the start position, which needs to be used together with the basis to evaluate the spline at val.
     * The span value is the value which is needed to get the basis function at the correct position. This means,
     * one can calculate the spline value as follows (1D -> 1D):
     *
     * @code
     * double pos = 0.1;
     * const std::pair<int, double> spanVal = getSpanAndValue(0, pos);
     * auto basis = basisFunctions(spanVal.second);
     *
     * double ret = 0.0;
     * int idx = 0;
     * for (int i = spanVal.first; i < spanVal.first + Order; ++i, ++idx) {
     *      ret += basis[idx] * controlPoint[i];
     * }
     * @endcode
     *
     * @param[in] dim The dimension.
     * @param[in] val The position (0 <= x <= 1).
     * @return A pair, where first is the span index and second is the span value.
     */
    UBS_NO_DISCARD std::pair<int, ValueType> getSpanIndexAndValue(int dim, ValueType val) const {
        assert(controlPoints_.getScale(dim) > ValueType(0.0));

        if (!extrapolate_) {
            assert(val >= getLowerBound(dim) - ValueType(1e-5) && val <= getUpperBound(dim) + ValueType(1e-5));
        }

        const ValueType baseX = (val - controlPoints_.getLowerBound(dim)) * controlPoints_.getScale(dim);

        // Use trait to convert to int. For 'normal' types this is just a static cast, but e.g. for a ceres::Jet type
        // the value part is used and cast to int.
        const int xii = ValueTypeTrait<ValueType>::toInt(baseX);
        const auto xi = std::max(0, std::min(xii, getNumControlPoints(dim) - Order));
        const ValueType xv = baseX - ValueType(xi);

        return std::make_pair(xi, xv);
    }

    /**
     * @brief Return the start index and values.
     *
     * These information can be used to evaluate the B-spline and its derivatives.
     * The returned data is only valid as long as the shape of the control points and the lower and upper bound is not
     * changed.
     *
     * @sa getSpanIndexAndValue(int dim, ValueType val) const
     * @param[in] pos The position to evaluate the start index and values in each dimension.
     * @return std::pair<int, InputType>
     */
    UBS_NO_DISCARD std::pair<int, InputType> getStartIndexAndValues(const InputType& pos) const {
        std::pair<int, InputType> data{};

        const auto& strides = controlPoints_.getStrides();
        for (int dim = 0; dim < InputDims; ++dim) {
            const ValueType val = FixedSizeContainerTypeTrait<InputType>::get(pos, dim);
            const auto sv = getSpanIndexAndValue(dim, val);

            data.first += strides[dim] * sv.first;
            FixedSizeContainerTypeTrait<InputType>::get(data.second, dim) = sv.second;
        }

        return data;
    }

    /**
     * @sa getSpanIndexAndValue
     * @return The span index.
     */
    UBS_NO_DISCARD int getSpanIndex(int dim, ValueType point) const {
        return getSpanIndexAndValue(dim, point).first;
    }

    /**
     * @brief Evaluates the span indices.
     * @sa getSpanIndex(int dim, ValueType point) const
     * @param[in] pos The position to evaluate the spline indices.
     * @return The span indices in each dimension.
     */
    UBS_NO_DISCARD std::array<int, InputDims> getSpanIndices(const InputType& pos) const {
        std::array<int, InputDims> indices{};
        for (int dim = 0; dim < InputDims; ++dim) {
            indices[dim] = getSpanIndex(dim, FixedSizeContainerTypeTrait<InputType>::get(pos, dim));
        }
        return indices;
    }

    /**
     * @sa getSpanIndexAndValue
     * @return The span value.
     */
    UBS_NO_DISCARD ValueType getSpanValue(int dim, ValueType point) const {
        return getSpanIndexAndValue(dim, point).second;
    }

    /**
     * @brief Determine the basis at the span value.
     * @sa getSpanIndexAndValue
     * @param[in] spanValue The span value.
     * @return The basis.
     */
    UBS_NO_DISCARD BasisType basisFunctions(ValueType spanValue) const {
        BasisType basis{basis_.col(0)};
        for (int i = 1; i < Order; ++i) {
            basis = spanValue * basis + basis_.col(i);
        }
        return basis;
    }

    /**
     * @brief Determine the basis derivative at the span value.
     * @sa getSpanIndexAndValue
     * @param[in] dim The dimension.
     * @param[in] derivative The derivative.
     * @param[in] spanValue The span value.
     * @return The derivative basis.
     */
    UBS_NO_DISCARD BasisType basisFunctionDerivatives(int dim, int derivative, ValueType spanValue) const {
        if (derivative > Degree) {
            return BasisType::Zero();
        }
        if (derivative == 0) {
            return basisFunctions(spanValue);
        }

        using ValueTypeTrait = ValueTypeTrait<ValueType>;
        const ValueType scale{ValueTypeTrait::pow(controlPoints_.getScale(dim), derivative)};

        BasisType basis = basis_.col(0) * (derivativeFactors_(0, derivative - 1) * scale);
        for (int i = 1; i < Order - derivative; ++i) {
            basis = spanValue * basis + basis_.col(i) * (derivativeFactors_(i, derivative - 1) * scale);
        }
        return basis;
    }

    /**
     * @brief Evaluates the spline at the specified position.
     * @param[in] pos The position. Each value must be between lower bound and upper bound.
     * @return The spline value.
     */
    UBS_NO_DISCARD OutputType evaluate(const InputType& pos) const {
        return evaluate(pos, [this](int /*dim*/, ValueType pos) { return basisFunctions(pos); });
    }

    /**
     * @brief Evaluates the derivative of the spline at the specified position.
     * @param[in] pos The position. Each value must be between lower bound and upper bound.
     * @param[in] d The derivative in the first dimension.
     * @param[in] derivatives The derivative in the second, third, ... dimension.
     * @return The derivative value.
     */
    template <typename... Ts>
    UBS_NO_DISCARD OutputType derivative(const InputType& pos, int d, Ts... derivatives) const {
        static_assert(1 + sizeof...(Ts) == InputDims, "Invalid number of inputs specified.");
        const std::array<int, InputDims> derivs{d, derivatives...};
        return derivative(pos, derivs);
    }

    /**
     * @brief Evaluates the derivative of the spline at the specified position.
     * @param[in] pos The position. Each value must be between lower bound and upper bound.
     * @param[in] derivatives The partial derivatives.
     * @return The derivative value.
     */
    UBS_NO_DISCARD OutputType derivative(const InputType& pos, const std::array<int, InputDims>& derivatives) const {
        return evaluate(pos, [this, &derivatives](int dim, ValueType pos) {
            const int derivative = derivatives[dim];
            if (derivative == 0) {
                return basisFunctions(pos);
            }

            return basisFunctionDerivatives(dim, derivative, pos);
        });
    }

    /**
     * @brief Evaluates the smoothness value.
     *
     * The smoothness value is defined as:
     * @f[
     * s_i = \int_0^1 \lVert f_i^{(n)}(\mathbf{x}) \rVert^2 \mathbf{dx}
     * @f]
     *
     * @tparam TotalDerivative The total derivative.
     * @return The smoothness value in each dimension.
     */
    template <int TotalDerivative>
    UBS_NO_DISCARD OutputType smoothness() const {
        static_assert(TotalDerivative >= 0 && TotalDerivative <= 3, "Unsupported derivative specified.");
        constexpr int MaxTotal = internal::TotalDerivative<InputDims, TotalDerivative>::NumPartialDerivatives;

        OutputType res = FixedSizeContainerTypeTrait<OutputType>::zero();
        internal::ComputeSmoothness<InputDims, TotalDerivative, MaxTotal, 0>::apply(
            *this, controlPoints_.getStrides(), res);
        return res;
    }

    /**
     * @brief Evaluates the B-spline by calculating the basis and passes it to the evalFunction.
     *
     * This function iterates over the sums, which are needed to compute a B-spline. It calculates the basis value and
     * for each calculated basis value it calls the eval function. The signature must be void(int idx, ValueType
     * basisVal), where idx is the control point index and basisVal value which need to be multiplied with the control
     * point.
     *
     * This function is mainly used if one would like to get the basis values and control points indices and defer the
     * evaluation. This is usually the case during optimization.
     *
     * @param[in] pos The position. Each value must be between lower bound and upper bound.
     * @param[in] basisFunction The basis function.
     * @param[in] evalFunction The evaluation function.
     */
    template <typename BasisFunction, typename EvalFunction>
    void evaluate(const InputType& pos, BasisFunction basisFunction, EvalFunction evalFunction) const {
        const std::pair<int, InputType> startIndexValues = getStartIndexAndValues(pos);
        evaluate(startIndexValues.first, startIndexValues.second, basisFunction, evalFunction);
    }

    /**
     * @copybrief evaluate(const InputType& pos, BasisFunction basisFunction, EvalFunction evalFunction) const
     *
     * For more information see
     * evaluate(const InputType& pos, BasisFunction basisFunction, EvalFunction evalFunction) const.
     *
     * @param[in] startIdx The start index returned by getStartIndexAndValues().
     * @param[in] values The values returned by getStartIndexAndValues().
     * @param[in] basisFunction The basis function.
     * @param[in] evalFunction The evaluation function.
     */
    template <typename BasisFunction, typename EvalFunction>
    void evaluate(int startIdx, const InputType& values, BasisFunction basisFunction, EvalFunction evalFunction) const {
        const auto& strides = controlPoints_.getStrides();

        std::array<BasisType, InputDims> fullBasis{};
        for (int dim = 0; dim < InputDims; ++dim) {
            fullBasis[dim] = basisFunction(dim, FixedSizeContainerTypeTrait<InputType>::get(values, dim));
        }

        using SplineType = UniformBSpline<ValueType_, Degree_, InputType_, OutputType_, ControlPointsType>;
        internal::EvaluateBSpline<SplineType, InputDims, 0>::apply(
            fullBasis, strides, startIdx, ValueType(1.0), evalFunction);
    }

    /**
     * @brief Casts this spline to the spline specified in the template parameter SplineOut.
     *
     * The cast function copies the control points, the lower and upper bounds. The value type of the source and
     * destination types must be convertible.
     *
     * @tparam SplineOut The spline to which this spline is casted to.
     * @return The casted spline.
     */
    template <typename SplineOut,
              typename = std::enable_if_t<
                  std::is_same<UniformBSpline<ValueType_, Degree_, InputType_, OutputType_, ControlPointsType>,
                               SplineOut>::value>>
    UBS_NO_DISCARD const SplineOut& cast() {
        return *this;
    }

    /** \copydoc const SplineOut& cast() */
    template <typename SplineOut,
              typename = std::enable_if_t<
                  !std::is_same<UniformBSpline<ValueType_, Degree_, InputType_, OutputType_, ControlPointsType>,
                                SplineOut>::value>>
    UBS_NO_DISCARD SplineOut cast() const {
        using T = typename SplineOut::ValueType;
        using OptSpline = SplineOut;
        using OptInputType = typename OptSpline::InputType;
        using OptOutputType = typename OptSpline::OutputType;
        using OptControlPointsType = typename OptSpline::ControlPointsType;
        using OptControlPointsContainerType = typename OptSpline::ControlPointsContainerType;
        using OptInputContainerTrait = ubs::FixedSizeContainerTypeTrait<OptInputType>;
        using OptOutputContainerTrait = ubs::FixedSizeContainerTypeTrait<OptOutputType>;

        // Convert bounds.
        OptInputType lowerBound{};
        OptInputType upperBound{};
        for (int i = 0; i < OptInputContainerTrait::Size; ++i) {
            OptInputContainerTrait::get(lowerBound, i) = T(getLowerBound(i));
            OptInputContainerTrait::get(upperBound, i) = T(getUpperBound(i));
        }

        OptControlPointsType controlPoints{};
        ubs::ControlPointsTrait<OptControlPointsType>::resize(controlPoints, controlPoints_.getShape());
        OptControlPointsContainerType controlPointsContainer(lowerBound, upperBound, std::move(controlPoints));

        // Copy control points.
        controlPoints_.transform(controlPointsContainer, [](const auto& p) {
            OptOutputType v{};
            for (int i = 0; i < OptOutputContainerTrait::Size; ++i) {
                OptOutputContainerTrait::get(v, i) = T(ubs::FixedSizeContainerTypeTrait<OutputType>::get(p, i));
            }
            return v;
        });

        SplineOut splineOut(std::move(controlPointsContainer));
        splineOut.setExtrapolate(extrapolate_);
        return splineOut;
    }

private:
    // Used for unit test of derivative factors.
    FRIEND_TEST(UniformBSpline, DerivativeFactors);

    /**
     * @brief Evaluate the B-spline given a basis function.
     *
     * The evaluation of the function itself and the derivatives is similar, the basis function selects, which basis
     * is used.
     *
     * @param[in] pos The position, at which the B-spline will be evaluated.
     * @param[in] basisFunction The basis function. The signature must be @code BasisType(int dim, ValueType pos)
     * @endcode
     * @return The evaluated basis value.
     */
    template <typename BasisFunction>
    UBS_NO_DISCARD OutputType evaluate(const InputType& pos, BasisFunction basisFunction) const {
        OutputType res = FixedSizeContainerTypeTrait<OutputType>::zero();
        evaluate(pos, basisFunction, [this, &res](int idx, ValueType basisVal) {
            res += controlPoints_.at(idx) * basisVal;
        });
        return res;
    }

    /**
     * @brief Determines the derivative factors.
     *
     * The derivative factors are the factor from deriving the vector
     * @f[
     * \mathbf{x}(t) = \begin{pmatrix} \vdots  \\ t^2 \\ t \\ 1 \end{pmatrix}
     * @f]
     * n times.
     * As an example, the derivative factors for order 4 are:
     * @f[
     * \begin{blockarray}{cccc}
     * x^{(1)} & x^{(2)} & x^{(3)} & x^{(4)} \\
     * \begin{block}{(cccc)}
     * 4 & 12 & 24 & 24 \\
     * 3 &  6 &  6 &  0 \\
     * 2 &  2 &  0 &  0 \\
     * 1 &  0 &  0 &  0 \\
     * \end{block}
     * \end{blockarray}
     * @f]
     *
     * @return The derivative factors.
     */
    UBS_NO_DISCARD Eigen::Matrix<ValueType, Degree, Degree> getDerivativeFactors() const {
        Eigen::Matrix<int, Degree, Degree> derivativeFactors = Eigen::Matrix<int, Degree, Degree>::Zero();

        for (int i = 0; i < Degree; ++i) {
            derivativeFactors(i, 0) = Order - 1 - i;
        }

        for (int c = 1; c < Degree; ++c) {
            for (int r = 0; r < Degree - c; ++r) {
                derivativeFactors(r, c) = derivativeFactors(r, c - 1) * (Degree - r - c);
            }
        }

        return derivativeFactors.template cast<ValueType>();
    }

    /** @brief The control points. */
    ControlPointsContainerType controlPoints_;

    /** @brief The uniform B-spline basis. */
    Eigen::Matrix<ValueType, Order, Order> basis_{
        internal::UniformBSplineBasis<Order>::matrix().template cast<ValueType>()};
    /** @brief The derivative factors. */
    Eigen::Matrix<ValueType, Degree, Degree> derivativeFactors_{getDerivativeFactors()};
    /** @brief Flag, if spline extrapolation is enabled. If true, the spline can be evaluated outside of the bounds. */
    bool extrapolate_{false};
};
} // namespace ubs

#include "internal/uniform_bspline_impl.hpp"

namespace ubs {

/**
 * @brief Using declaration for a spline @f$ \mathbb{R} \rightarrow \mathbb{R} @f$.
 * @tparam ValueType_ The value type.
 * @tparam Degree_ The spline degree.
 */
template <typename ValueType, int Degree>
using UniformBSpline11 =
    UniformBSpline<ValueType,
                   Degree,
                   ValueType,
                   ValueType,
                   std::vector<ValueType, typename FixedSizeContainerTypeTrait<ValueType>::Allocator>>;

/**
 * @copybrief UniformBSpline11
 *
 * The value type is float.
 * @tparam Degree_ The spline degree.
 */
template <int Degree>
using UniformBSpline11f = UniformBSpline11<float, Degree>;

/**
 * @copybrief UniformBSpline11
 *
 * The value type is double.
 * @tparam Degree_ The spline degree.
 */
template <int Degree>
using UniformBSpline11d = UniformBSpline11<double, Degree>;

/**
 * @brief Using declaration for an Eigen spline @f$ \mathbb{R}^n \rightarrow \mathbb{R}^m @f$.
 * @tparam ValueType_ The value type.
 * @tparam Degree_ The spline degree.
 * @tparam InputDims_ The input dimensions.
 * @tparam OutputDims_ The output dimensions.
 */
template <typename ValueType, int Degree, int InputDims, int OutputDims>
using EigenUniformBSpline =
    UniformBSpline<ValueType, Degree, Eigen::Matrix<ValueType, InputDims, 1>, Eigen::Matrix<ValueType, OutputDims, 1>>;

} // namespace ubs
