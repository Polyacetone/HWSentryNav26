#pragma once
#include <ceres/autodiff_cost_function.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <uniform_bspline/uniform_bspline.hpp>
#include <uniform_bspline/internal/total_derivative.hpp>

#include "fixed_size_container_type_trait_ceres.hpp"
#include "uniform_bspline_ceres_evaluator.hpp"
#include "uniform_bspline_ceres_generator.hpp"
#include "value_type_trait_ceres.hpp"
#include "internal/autodiff_smoothing_type.hpp"
#include "internal/smoothness_cost_functor.hpp"

namespace ubs {

/**
 * @brief This class provides the functionality of optimizing the control points of a uniform B-spline with ceres.
 *
 * This class helps to create cost functions, which needs to evaluate the spline value or its derivative during
 * optimization. The position, at which the spline is evaluated cannot be changed during optimization and must be
 * known during ceres problem creation.
 *
 * @tparam Spline The uniform B-spline.
 */
template <typename Spline_>
class UniformBSplineCeres {
public:
    /** @brief The spline value type. */
    using ValueType = typename Spline_::ValueType;

    /** @brief The spline input type .*/
    using InputType = typename Spline_::InputType;
    /** @brief The spline output type. */
    using OutputType = typename Spline_::OutputType;

    // As ceres uses double, the value type of the underlying spline must be double.
    static_assert(std::is_same<double, ValueType>::value, "Specified value type is not supported.");
    static_assert(std::is_same<double, typename FixedSizeContainerTypeTrait<OutputType>::ValueType>::value,
                  "Specified output value type is not supported.");
    static_assert(FixedSizeContainerTypeTrait<OutputType>::IsContinuous, "The output value type must be continuous.");

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

    /** @brief Point data used to represent a point evaluation of the spline. */
    struct PointData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /**
         * @brief The start index into the control points storage.
         *
         * The index is the same as returned by a call to getStartIndexAndValues().
         */
        int startIdx{};

        /**
         * @brief The values used to specify the local coordinate in a segment.
         *
         * The values are the same as returned by a call to getStartIndexAndValues().
         */
        InputType values{};
    };

    /** @brief Range data used to represent a spline range to create a spline generator. */
    struct RangeData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /** @brief The start index into the control points storage. */
        int startIdx{};

        /** @brief The lower bound used for the generated spline. */
        InputType lowerBound{};

        /**  @brief The lower bound used for the generated spline. */
        InputType upperBound{};

        /** @brief The number of control needed in each dimension to form the spline. */
        std::array<int, InputDims> shape{};
    };

    /**
     * @brief The constructor.
     *
     * The optimization is directly done on the storage of the control points of the specified spline. This means, the
     * the spline should be initialized and after optimization, the spline is automatically updated by ceres.
     *
     * @note The spline is held as a reference, so the spline must outlive this class.
     * @param[in] spline The spline to optimize.
     */
    explicit UniformBSplineCeres(Spline& spline) : spline_(spline) {
    }

    /**
     * @brief Deleted copy constructor.
     *
     * This is deleted because it can easily result in a dangling reference to the spline object. As this object is
     * really cheap to create, it should be created where it is needed.
     */
    UniformBSplineCeres(const UniformBSplineCeres&) = delete;
    UniformBSplineCeres(UniformBSplineCeres&&) = delete;

    /**
     * @brief Deleted assignment operator.
     * @copydetails UniformBSplineCeres(const UniformBSplineCeres&)
     */
    UniformBSplineCeres& operator=(const UniformBSplineCeres&) = delete;
    UniformBSplineCeres& operator=(UniformBSplineCeres&&) = delete;

    ~UniformBSplineCeres() = default;

    /**
     * @return The number of parameter pointers needed to evaluate a spline during optimization at a single point.
     */
    UBS_NO_DISCARD static constexpr int getNumPointParameterPointers() {
        return ControlPointsSupport;
    }

    /**
     * @brief Determines the number of parameter pointers needed for the spline generator.
     * @param[in] data The range data used to get the generator.
     * @return The number of parameter pointers needed to generate a spline.
     */
    UBS_NO_DISCARD int getNumRangeParameterPointers(const RangeData& data) const {
        return std::accumulate(data.shape.begin(), data.shape.end(), 1, std::multiplies<>());
    }

    /**
     * @brief Determines the point data needed to evaluate the spline at a specific position.
     * @param[in] pos The position one wants to evaluate the spline.
     * @return The point data.
     */
    UBS_NO_DISCARD PointData getPointData(const InputType& pos) const {
        const auto& v = spline_.getStartIndexAndValues(pos);
        PointData data{};
        data.startIdx = v.first;
        data.values = v.second;
        return data;
    }

    /**
     * @brief Determines the range data needed to generate a spline for the specified range indices.
     * @param[in] lowerBoundSpanIndices The lower bound span indices.
     * @param[in] upperBoundSpanIndices The upper bound span indices.
     * @return The range data.
     */
    UBS_NO_DISCARD RangeData getIndicesRangeData(const std::array<int, InputDims>& lowerBoundSpanIndices,
                                                 const std::array<int, InputDims>& upperBoundSpanIndices) const {
        using ContainerType = FixedSizeContainerTypeTrait<InputType>;

        RangeData data{};
        data.startIdx = 0;
        for (int dim = 0; dim < InputDims; ++dim) {
            auto lowerIdx = lowerBoundSpanIndices[dim];
            auto upperIdx = upperBoundSpanIndices[dim];

            assert(lowerIdx >= 0 && lowerIdx <= spline_.getNumControlPoints(dim) - Order);
            assert(upperIdx >= 0 && upperIdx <= spline_.getNumControlPoints(dim) - Order);

            data.shape[dim] = upperIdx - lowerIdx + Order;

            // Expand lower and upper bound.
            const auto& scale = spline_.getScale(dim);
            ContainerType::get(data.lowerBound, dim) = ValueType(lowerIdx) / scale + spline_.getLowerBound(dim);
            ContainerType::get(data.upperBound, dim) = ValueType(upperIdx + 1) / scale + spline_.getLowerBound(dim);

            data.startIdx += lowerIdx * spline_.getControlPointsContainer().getStride(dim);
        }

        return data;
    }

    /**
     * @brief Determines the range data needed to generate a spline for the specified range.
     * @param[in] lowerBound The lower bound for the spline which will be generator.
     * @param[in] upperBound The upper bound for the spline which will be generator.
     * @return The range data.
     */
    UBS_NO_DISCARD RangeData getRangeData(const InputType& lowerBound, const InputType& upperBound) const {
        return getIndicesRangeData(spline_.getSpanIndices(lowerBound), spline_.getSpanIndices(upperBound));
    }

    /**
     * @copybrief getRangeData(const InputType& lowerBound, const InputType& upperBound) const
     *
     * This function determines the range data based on a center position (c) and range (r). The resulting lower and
     * upper bounds are [c - r, c + r]. If clip is set to true, the calculated range is automatically clipped to the
     * lower and upper bound of the original spline.
     *
     * @param[in] pos The center position.
     * @param[in] range The range.
     * @param[in] clip True, if range should be clipped to the bounds of the original spline, other false.
     * @return The range data.
     */
    UBS_NO_DISCARD RangeData getRangeData(const InputType& pos, const InputType& range, bool clip) const {
        using ContainerType = FixedSizeContainerTypeTrait<InputType>;

        // Determin lower and upper bound.
        InputType lowerBound{};
        InputType upperBound{};
        for (int dim = 0; dim < InputDims; ++dim) {
            const auto posVal = ContainerType::get(pos, dim);
            const auto rangeVal = ContainerType::get(range, dim);
            auto& lowerBoundVal = ContainerType::get(lowerBound, dim);
            auto& upperBoundVal = ContainerType::get(upperBound, dim);

            lowerBoundVal = posVal - rangeVal;
            upperBoundVal = posVal + rangeVal;

            if (clip) {
                lowerBoundVal = std::max(lowerBoundVal, spline_.getLowerBound(dim));
                upperBoundVal = std::min(upperBoundVal, spline_.getUpperBound(dim));
            }
        }

        return getRangeData(lowerBound, upperBound);
    }

    /**
     * @brief Fills the parameter pointers for a point evaluation.
     *
     * The determined parameter pointers are written from the start of the begin iterator. The distance between begin
     * and end must be at least getNumPointParameterPointers().
     *
     * @param[in] data The point data.
     * @param[out] begin The begin iterator of the parameter points.
     * @param[out] end The end iterator.
     */
    template <typename Iter>
    void fillParameterPointers(const PointData& data, Iter begin, Iter end) {
        using ContainerType = FixedSizeContainerTypeTrait<OutputType>;
        (void)end;

        spline_.getControlPointsContainer().forEach(data.startIdx, getShape(), [&](OutputType& val) {
            assert(begin != end);
            *begin = ContainerType::data(val);
            ++begin;
        });
    }

    /**
     * @brief Fills the parameter pointers for a range spline generator.
     *
     * The determined parameter pointers are written from the start of the begin iterator. The distance between begin
     * and end must be at least getNumRangeParameterPointers().
     *
     * @param[in] data The range data.
     * @param[out] begin The begin iterator of the parameter points.
     * @param[out] end The end iterator.
     */
    template <typename Iter>
    void fillParameterPointers(const RangeData& data, Iter begin, Iter end) {
        using ContainerType = FixedSizeContainerTypeTrait<OutputType>;

        (void)end;

        spline_.getControlPointsContainer().forEach(data.startIdx, data.shape, [&](OutputType& val) {
            assert(begin != end);
            *begin = ContainerType::data(val);
            ++begin;
        });
    }

    /**
     * @brief Determines the evaluator to compute the spline value in a ceres cost function.
     *
     * This function calculates the basis value and the control point parameter pointers used to calculate the uniform
     * B-spline at the specified position stored in the point data. The uniform B-spline evaluator is returned.
     *
     * @sa UniformBSplineCeresEvaluator
     * @param[in] data The point data used to determine where the spline should be evaluated.
     * @return The evaluator to be used in a ceres cost function to compute the spline value.
     */
    UBS_NO_DISCARD UniformBSplineCeresEvaluator<Spline> getEvaluator(const PointData& data) const {
        return evaluate(data, [this](int /*dim*/, ValueType pos) { return spline_.basisFunctions(pos); });
    }

    /**
     * @brief Determines the evaluator to compute the spline derivatives in a ceres cost function.
     * @sa UniformBSplineCeresEvaluator
     * @sa evaluate(const PointData& data)
     * @param[in] data The point data used to determine where the spline should be evaluated.
     * @param[in] derivatives The orders of the derivative in each input dimension.
     * @return The evaluator to be used in a ceres cost function to compute the spline value.
     */
    UBS_NO_DISCARD UniformBSplineCeresEvaluator<Spline> getEvaluator(
        const PointData& data, const std::array<int, InputDims>& derivatives) const {
        return evaluate(data, [this, &derivatives](int dim, ValueType pos) {
            const int derivative = derivatives[dim];
            if (derivative == 0) {
                return spline_.basisFunctions(pos);
            }

            return spline_.basisFunctionDerivatives(dim, derivative, pos);
        });
    }

    /**
     * @brief Determines the generator to get a spline in a ceres cost function.
     *
     * This function return a spline generator, which can be used to create a spline based on control points, which are
     * optimized. The range of the spline is specified by the range data. As the spline generator does not know the
     * value type beforehand, a template template parameter must be specified (due to autodiff).
     *
     * @tparam SplineGenerator The spline generator type.
     * @param[in] data The range data.
     * @return The spline generator to be used in a ceres cost function to generate a spline.
     */
    template <template <typename ValueType> class SplineGenerator>
    UBS_NO_DISCARD UniformBSplineCeresGenerator<SplineGenerator> getGenerator(const RangeData& data) const {
        static_assert(std::is_same_v<typename SplineGenerator<double>::InputType, InputType>,
                      "Input type of spline generator and used spline must be the same.");
        return UniformBSplineCeresGenerator<SplineGenerator>(data.lowerBound, data.upperBound, data.shape);
    }

    /**
     * @copybrief addSmoothnessResidualsGrid()
     *
     * Adds smoothness residuals the the specified problem, where the weight in each input dimension is the same.
     *
     * @sa addSmoothnessResiduals(ceres::Problem&,const std::array<double, InputDims>&,ceres::LossFunction*) const
     * @tparam Derivative The smoothness derivative.
     * @param[out] problem The ceres problem, to which the smoothness residuals will be added.
     * @param[in] weight The weight of the smoothing residuals. The same weight for each input dimension is used.
     * @param[in] lossFunction The loss function, which will be used in adding the smoothness residuals to the problem.
     */
    template <int Derivative>
    void addSmoothnessResidualsGrid(ceres::Problem& problem,
                                    double weight = 1.0,
                                    ceres::LossFunction* lossFunction = nullptr) {
        assert(weight >= 0.0);
        std::array<double, InputDims> weights{};
        for (auto& w : weights) {
            w = weight;
        }

        addSmoothnessResidualsGrid<Derivative>(problem, weights, lossFunction);
    }

    /**
     * @brief Adds smoothness grid residuals to the specified problem.
     *
     * This function adds residuals to the specified problem to evaluate a one-dimensional smoothness term. The
     * one-dimensional smoothness term is mathematically expressed as:
     * @f[
     * \sum_i \int_0^1 \lVert f_i^{(n)}(x) \rVert^2 dx
     * @f]
     * In the one dimensional case this can be transformed to
     * @f[
     * \sum_i \lVert s \mathbf{B} \mathbf{P}_{i:(i+o)} \rVert^2
     * @f]
     * where @f$ \mathbf{B} @f$ can be precomputed and only depends on the degree and the smoothness derivative. The
     * scale factor @f$ s @f$ depends on the number of control points @f$ \mathbf{P} @f$ are the optimization
     * parameters. This means, calculating the exact integral is a matrix matrix multiplication during optimization for
     * each control point.
     *
     * As there is no such form (or at least not found by me) for higher dimensional B-splines, the two dimensional case
     * is implemented using the one dimensional case. Consider the following example, where we have five control points
     * in x direction and three in y direction.
     *
     * ```
     *   1 2 3 4 5
     * 1 ---------
     *   | | | | |
     * 2 ---------
     *   | | | | |
     * 3 ---------
     * ```
     *
     * One can think of building a 1D spline for each horizontal and vertical line in the grid and adding the smoothness
     * residual for each of the 1D spline.
     *
     * @tparam Derivative The smoothness derivative.
     * @param[out] problem The ceres problem, to which the smoothness residuals will be added.
     * @param[in] weights The weights of the smoothing residuals in each dimension.
     * @param[in] lossFunction The loss function, which will be used in adding the smoothness residuals to the problem.
     */
    template <int Derivative>
    void addSmoothnessResidualsGrid(ceres::Problem& problem,
                                    const std::array<double, InputDims>& weights,
                                    ceres::LossFunction* lossFunction = nullptr) {
        auto& controlPoints = spline_.getControlPointsContainer();
        const std::array<int, InputDims>& strides = controlPoints.getStrides();

        const int numElements = controlPoints.getNumElements();
        const int totalStrides = std::accumulate(strides.begin(), strides.end(), 1, std::multiplies<>());

        for (int curDim = 0; curDim < InputDims; ++curDim) {
            // Add smoothness integral for the current dimension.
            const int size = controlPoints.getSize(curDim);
            const int stride = strides[curDim];

            const int numIterations = numElements / size;
            const int idxStride = totalStrides / stride;

            assert(weights[curDim] >= 0.0);
            std::unique_ptr<ceres::CostFunction> costFunction =
                getSmoothingCostFunction1D<Derivative>(weights[curDim], curDim);

            // Release pointer here because ceres owns the pointer.
            auto* costFunctionPtr = costFunction.release();

            for (int i = 0, idx = 0; i < numIterations; ++i, idx += idxStride) {
                addSmoothnessResiduals1d(problem, idx, size, stride, costFunctionPtr, lossFunction);
            }
        }
    }

    /**
     * @brief Adds smoothness residuals to the specified problem.
     *
     * This function adds smoothness residuals to the specified problem. Currently the input dimensions must be less
     * than or equal to two. For higher dimensional splines use addSmoothnessResidualsGrid(). The output dimension
     * can be arbitrary.
     *
     * @note For a more detailed description about the smoothness of a B-spline see the <code>uniform_bspline</code>
     *       package.
     * @tparam TotalDerivative The smoothness total derivative.
     * @param[out] problem The ceres problem, to which the smoothness residuals will be added.
     * @param[in] weight The weight of the smoothing residuals.
     * @param[in] lossFunction The loss function, which will be used in adding the smoothness residuals to the problem.
     */
    template <int TotalDerivative>
    void addSmoothnessResiduals(ceres::Problem& problem,
                                double weight = 1.0,
                                ceres::LossFunction* lossFunction = nullptr) {
        addSmoothnessResiduals<TotalDerivative>(
            problem, weight, lossFunction, std::integral_constant<int, InputDims>());
    }

private:
    /** @brief A one dimensional smoothness cost function depends on Order control points. */
    using SmoothnessCostFunctor1D_ = internal::SmoothnessCostFunctor<OutputDims, Order>;
    /** @brief A two dimensional smoothness cost function depends on Order * Order control points. */
    using SmoothnessCostFunctor2D_ = internal::SmoothnessCostFunctor<OutputDims, Order * Order>;

    /** @brief Get the shape (size in each dimension) to evaluate the spline at a single point. */
    UBS_NO_DISCARD static constexpr std::array<int, InputDims> getShape() {
        std::array<int, InputDims> shape{};
        for (int i = 0; i < InputDims; ++i) {
            shape[i] = Order;
        }
        return shape;
    }

    /**
     * @brief Add smoothness residuals to the problem.
     * @tparam TotalDerivative The smoothness total derivative.
     * @tparam N The input dimension of the B-spline.
     */
    template <int TotalDerivative, int N>
    void addSmoothnessResiduals(ceres::Problem& /*problem*/,
                                double /*weight*/,
                                ceres::LossFunction* /*lossFunction*/,
                                std::integral_constant<int, N> /*inputDim*/) {
        static_assert(N == 1 || N == 2, "Unsupported dimension.");
    }

    /**
     * @brief Add smoothness residuals to the problem for a B-spline with InputDims = 1.
     * @tparam TotalDerivative The smoothness total derivative.
     * @param[out] problem The ceres problem, to which the smoothness residuals will be added.
     * @param[in] weight The weight of the smoothing residuals.
     * @param[in] lossFunction The loss function, which will be used in adding the smoothness residuals to the problem.
     */
    template <int TotalDerivative>
    void addSmoothnessResiduals(ceres::Problem& problem,
                                double weight,
                                ceres::LossFunction* lossFunction,
                                std::integral_constant<int, 1> /*dim*/) {
        auto& controlPoints = spline_.getControlPointsContainer();
        const int size = controlPoints.getSize(0);
        const int stride = controlPoints.getStride(0);
        auto costFunction = getSmoothingCostFunction1D<TotalDerivative>(weight, 0);

        // Release cost function here because ceres owns it.
        addSmoothnessResiduals1d(problem, 0, size, stride, costFunction.release(), lossFunction);
    }

    /**
     * @brief Add smoothness residuals to the problem for a B-spline with InputDims = 2.
     * @tparam TotalDerivative The smoothness total derivative.
     * @param[out] problem The ceres problem, to which the smoothness residuals will be added.
     * @param[in] weight The weight of the smoothing residuals.
     * @param[in] lossFunction The loss function, which will be used in adding the smoothness residuals to the problem.
     */
    template <int TotalDerivative>
    void addSmoothnessResiduals(ceres::Problem& problem,
                                double weight,
                                ceres::LossFunction* lossFunction,
                                std::integral_constant<int, 2> /*dim*/) {
        auto& controlPoints = spline_.getControlPointsContainer();
        const int size1 = controlPoints.getSize(0);
        const int stride1 = controlPoints.getStride(0);
        const int size2 = controlPoints.getSize(1);
        const int stride2 = controlPoints.getStride(1);

        constexpr int NumPartialDerivatives = internal::TotalDerivative<2, TotalDerivative>::NumPartialDerivatives;
        addPartialDerivativeSmoothnessResiduals2d<TotalDerivative, NumPartialDerivatives, 0>(
            problem, size1, stride1, size2, stride2, lossFunction, weight, std::false_type());
    }

    /**
     * @brief Add the partial derivatives residual cost functor the the ceres problem.
     *
     * @tparam TotalDerivative The smoothness total derivative.
     * @tparam NumPartialDerivatives The total number of partial derivatives needed to calculate the smoothness value
     *         of total derivative of the B-spline.
     * @tparam PartialDerivativeIdx The current partial derivative index.
     * @param[out] problem The ceres problem, to which the smoothness residuals will be added.
     * @param[in] size1 The total number of control points in the first dimension.
     * @param[in] stride1 The stride to get to the next index in the first dimension.
     * @param[in] size2 The total number of control points in the second dimension.
     * @param[in] stride2 The stride to get to the next index in the second dimension.
     * @param[in] lossFunction The loss function, which will be used in adding the smoothness residuals to the problem.
     * @param[in] weight The weight of the smoothing residuals.
     */
    template <int TotalDerivative, int NumPartialDerivatives, int PartialDerivativeIdx>
    void addPartialDerivativeSmoothnessResiduals2d(ceres::Problem& problem,
                                                   int size1,
                                                   int stride1,
                                                   int size2,
                                                   int stride2,
                                                   ceres::LossFunction* lossFunction,
                                                   double weight,
                                                   std::false_type /*lastSmoothnessResidual*/) {
        // Extract multiplicity and degree of partial derivative for first and seconds dimension.
        constexpr auto FullPartialDerivatives =
            internal::TotalDerivative<2, TotalDerivative>::getPartialDerivatives()[PartialDerivativeIdx];
        constexpr auto PartialDerivativesMultiplicity = double(FullPartialDerivatives.multiplicity);

        static_assert(FullPartialDerivatives.partialDerivatives.size() == 2, "Internal error.");
        constexpr int PartialDerivative1 = FullPartialDerivatives.partialDerivatives[0];
        constexpr int PartialDerivative2 = FullPartialDerivatives.partialDerivatives[1];

        // Determine cost function and add smoothness residual.
        std::unique_ptr<ceres::CostFunction> costFunction =
            getSmoothingCostFunction2D<PartialDerivative1, PartialDerivative2>(weight * PartialDerivativesMultiplicity);

        // Release unique_ptr here because ceres owns it.
        addSmoothnessResiduals2d(problem, 0, size1, stride1, size2, stride2, costFunction.release(), lossFunction);

        // Add next smoothness partial derivative to the problem.
        addPartialDerivativeSmoothnessResiduals2d<TotalDerivative, NumPartialDerivatives, PartialDerivativeIdx + 1>(
            problem,
            size1,
            stride1,
            size2,
            stride2,
            lossFunction,
            weight,
            std::integral_constant<bool, NumPartialDerivatives == PartialDerivativeIdx + 1>());
    }

    /** @brief Recursion end. */
    template <int TotalDerivative, int NumPartialDerivatives, int PartialDerivativeIdx>
    void addPartialDerivativeSmoothnessResiduals2d(ceres::Problem& /*problem*/,
                                                   int /*cols*/,
                                                   int /*strideCols*/,
                                                   int /*rows*/,
                                                   int /*strideRows*/,
                                                   ceres::LossFunction* /*lossFunction*/,
                                                   double /*weight*/,
                                                   std::true_type /*lastSmoothnessResidual*/) {
        // No more partial derivatives left. Nothing more to do.
    }

    /**
     * @brief Determine the scale of the smoothness matrix when used in a (non-)linear least squares problem.
     * @tparam Derivative The smoothness derivative.
     * @param[in] dim The input dimension.
     * @return The smoothness matrix scale.
     */
    template <int Derivative>
    UBS_NO_DISCARD double getSmoothnessMatrixScale(int dim) const {
        const double scale = spline_.getScale(dim);
        return std::pow(scale, double(Derivative) - 1.0 / 2.0);
    }

    /**
     * @brief Creates a 1D smoothing cost function for the specified dimension.
     *
     * The returned cost function can be used for all smoothing residuals in that direction.
     *
     * @note The caller is responsible of releasing the memory of the returned cost function.
     * @tparam Derivative The smoothness derivative.
     * @param[in] weight The weight of the smoothness.
     * @param[in] dim The input dimension for which the cost function will be used.
     * @return The cost function.
     */
    template <int Derivative>
    UBS_NO_DISCARD std::unique_ptr<ceres::CostFunction> getSmoothingCostFunction1D(double weight, int dim) const {
        using CostFunctionType = internal::AutoDiffSmoothingType<Spline, SmoothnessCostFunctor1D_>;
        const Eigen::Matrix<double, Order, Order>& smoothnessBasisRoot =
            internal::UniformBSplineSmoothnessBasis<Order, Derivative>::matrixRoot();

        const double scale = getSmoothnessMatrixScale<Derivative>(dim);
        return std::make_unique<CostFunctionType>(
            new SmoothnessCostFunctor1D_(scale * std::sqrt(2.0 * weight) * smoothnessBasisRoot));
    }

    /**
     * @brief Creates a 2D smoothing cost function for the specified dimension.
     *
     * The returned cost function can be used for all smoothing residuals in that direction.
     *
     * @note The caller is responsible of releasing the memory of the returned cost function.
     * @tparam PartialDerivative1 The order of the partial derivative with respect to the first variable.
     * @tparam PartialDerivative2 The order of the partial derivative with respect to the second variable.
     * @param[in] weight The weight of the smoothness.
     * @return The cost function.
     */
    template <int PartialDerivative1, int PartialDerivative2>
    UBS_NO_DISCARD std::unique_ptr<ceres::CostFunction> getSmoothingCostFunction2D(double weight) const {
        // Determine the scale.
        const double scale =
            getSmoothnessMatrixScale<PartialDerivative1>(0) * getSmoothnessMatrixScale<PartialDerivative2>(1);

        // The two dimensional cost function can be evaluated in a NLS by using the following formula:
        // \sum_{i,j} || S_1^1/2 P_{i,j} S_2^1/2 ||^2
        // where S is the smoothness matrix of the partial derivatives and P are the control points. This can be
        // reformulated as: \sum_{i,j} || M P_{i,j} ||^2 where M is a (Order*Order) x (Order*Order) sized matrix.
        const Eigen::Matrix<double, Order, Order>& smoothnessBasisRoot1 =
            internal::UniformBSplineSmoothnessBasis<Order, PartialDerivative1>::matrixRoot();
        const Eigen::Matrix<double, Order, Order>& smoothnessBasisRoot2 =
            internal::UniformBSplineSmoothnessBasis<Order, PartialDerivative2>::matrixRoot();

        // Compute 2D smoothness matrix.
        constexpr int OrderS = Order * Order;
        Eigen::Matrix<double, OrderS, OrderS> m;
        for (int i = 0; i < Order; ++i) {
            for (int j = 0; j < Order; ++j) {
                const Eigen::Matrix<double, Order, Order> temp =
                    smoothnessBasisRoot2.col(j) * smoothnessBasisRoot1.row(i);
                m.col(i * Order + j) = Eigen::Map<const Eigen::Matrix<double, OrderS, 1>>(temp.data());
            }
        }

        // Create cost function. Use DynamicAutoDiffCostFunction here because of limitation of number of parameter
        // blocks in AutoDiffCostFunction (this limitation is lifted in a newer version of ceres).
        auto costFunction =
            std::make_unique<ceres::DynamicAutoDiffCostFunction<SmoothnessCostFunctor2D_, OrderS * OutputDims>>(
                new SmoothnessCostFunctor2D_(scale * std::sqrt(2.0 * weight) * m));
        for (int i = 0; i < OrderS; ++i) {
            costFunction->AddParameterBlock(OutputDims);
        }

        costFunction->SetNumResiduals(OrderS * OutputDims);
        return costFunction;
    }

    /**
     * @brief Adds smoothness residuals in one axis to the problem.
     * @param[out] problem The ceres problem, to which the residuals are added.
     * @param[in] startIdx The start index.
     * @param[in] size The total number of control points in that direction.
     * @param[in] stride The stride to get to the next index.
     * @param[in] costFunction The smoothness cost function.
     * @param[in] lossFunction The smoothness loss function.
     */
    void addSmoothnessResiduals1d(ceres::Problem& problem,
                                  int startIdx,
                                  int size,
                                  int stride,
                                  ceres::CostFunction* costFunction,
                                  ceres::LossFunction* lossFunction) {
        auto& controlPoints = spline_.getControlPointsContainer();
        auto data = controlPoints.data();

        std::vector<double*> paramPointers(Order);
        std::vector<double*> allParameterPointers(size);

        // Collect parameter pointers.
        for (int i = 0, idx = startIdx; i < size; ++i, idx += stride) {
            allParameterPointers[i] = FixedSizeContainerTypeTrait<OutputType>::data(*(data + idx));
        }

        // Add parameter pointers.
        for (int i = 0; i < size - Order + 1; ++i) {
            std::copy(
                allParameterPointers.begin() + i, allParameterPointers.begin() + i + Order, paramPointers.begin());

            problem.AddResidualBlock(costFunction, lossFunction, paramPointers);
        }
    }

    /**
     * @brief Adds smoothness residuals in two axis to the problem.
     * @param[out] problem The ceres problem, to which the residuals are added.
     * @param[in] startIdx The start index.
     * @param[in] size1 The total number of control points in the first dimension.
     * @param[in] stride1 The stride to get to the next index in the first dimension.
     * @param[in] size2 The total number of control points in the second dimension.
     * @param[in] stride2 The stride to get to the next index in the second dimension.
     * @param[in] costFunction The smoothness cost function.
     * @param[in] lossFunction The smoothness loss function.
     */
    void addSmoothnessResiduals2d(ceres::Problem& problem,
                                  int startIdx,
                                  int size1,
                                  int stride1,
                                  int size2,
                                  int stride2,
                                  ceres::CostFunction* costFunction,
                                  ceres::LossFunction* lossFunction) {
        auto& controlPoints = spline_.getControlPointsContainer();
        auto data = controlPoints.data();

        // Collect all parameter pointers.
        Eigen::Matrix<double*, Eigen::Dynamic, Eigen::Dynamic> controlPointsPointers(size2, size1);
        for (int c = 0, idxOuter = startIdx; c < size1; ++c, idxOuter += stride1) {
            for (int r = 0, idxInner = idxOuter; r < size2; ++r, idxInner += stride2) {
                controlPointsPointers(r, c) = FixedSizeContainerTypeTrait<OutputType>::data(*(data + idxInner));
            }
        }

        // Add residuals blocks.
        std::vector<double*> paramPointers(Order * Order);
        for (int c = 0; c < size1 - Degree; ++c) {
            for (int r = 0; r < size2 - Degree; ++r) {
                Eigen::Matrix<double*, Order, Order> controlPointsBlock =
                    controlPointsPointers.block<Order, Order>(r, c);
                std::copy(controlPointsBlock.data(), controlPointsBlock.data() + Order * Order, paramPointers.begin());
                problem.AddResidualBlock(costFunction, lossFunction, paramPointers);
            }
        }
    }

    /**
     * @brief Evaluates the basis and creates a ceres cost functor to evaluate the B-spline based on its control points.
     * @param[in] pos The position, at which the B-spline should be evaluated.
     * @param[out] begin Iterator to the begin of parameter pointers.
     * @param[out] end Iterator to the end of parameter pointers.
     * @param[in] basisFunction
     * @return The uniform B-spline evaluator.
     */
    template <typename BasisFunction>
    UBS_NO_DISCARD UniformBSplineCeresEvaluator<Spline> evaluate(const PointData& data,
                                                                 BasisFunction basisFunction) const {
        int counter = 0;
        std::array<double, ControlPointsSupport> basisVals{};

        spline_.evaluate(data.startIdx, data.values, basisFunction, [&, this](int /*idx*/, ValueType basisVal) {
            basisVals[counter] = basisVal;
            ++counter;
        });

        return UniformBSplineCeresEvaluator<Spline>(basisVals);
    }

    /** @brief A reference to a spline. */
    Spline& spline_;
};
} // namespace ubs

#include "internal/uniform_bspline_ceres_impl.hpp"
