#pragma once
#include <type_traits>

#include <Eigen/Core>

namespace ubs {
namespace internal {
/**
 * @brief The smoothness cost functor.
 *
 * This cost functor is used to calculate the smoothness of a spline during optimization. The operation for
 * uniform B-splines boils down to a matrix matrix product, where the first matrix is build by concatenating the
 * control points and the second one is the fixed smoothing matrix.
 *
 * @tparam N_ The number of control points the cost function depends on.
 */
template <int OutputDims_, int N_>
class SmoothnessCostFunctor {
private:
    /* @brief Flag, which indicates if dynamic sized matrices should be used.
     *
     * Due to stack limitation of the number of elements of the matrix use Eigen::Dynamic for big ones.
     */
    static constexpr bool UseDynamicSizedMatrix_ = (N_ * OutputDims_) > 16;
    static constexpr int EigenN_ = UseDynamicSizedMatrix_ ? Eigen::Dynamic : N_;

    /**
     * @brief Helper function to create the control points matrix.
     *
     * This overload is picked if fixed size matrices are used.
     *
     * @tparam T The value type of the matrix.
     * @return The control points matrix with the correct size.
     */
    template <typename T>
    UBS_NO_DISCARD Eigen::Matrix<T, OutputDims_, EigenN_> makeControlPoints(std::false_type /* isDynamic */) const {
        return {};
    }

    /**
     * @brief Helper function to create the control points matrix.
     *
     * This overload is picked if dynamic sized matrices are used.
     *
     * @tparam T The value type of the matrix.
     * @return The control points matrix with the correct size.
     */
    template <typename T>
    UBS_NO_DISCARD Eigen::Matrix<T, OutputDims_, EigenN_> makeControlPoints(std::true_type /* isDynamic */) const {
        return {OutputDims_, N_};
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief The constructor.
     * @param[in] m The smoothness cost factors.
     */
    explicit SmoothnessCostFunctor(Eigen::Matrix<double, N_, N_> m) : m_{std::move(m)} {
    }

    /**
     * @brief Cost function used in DynamicAutoDiffCostFunction.
     * @param[in] controlPointsRaw The control points pointer.
     * @param[out] residualRaw The residual pointer.
     * @return True on success, otherwise false.
     */
    template <typename T>
    bool operator()(T const* const* controlPointsRaw, T* residualRaw) const {
        // Copy the control points to an Eigen matrix.
        auto controlPoints = makeControlPoints<T>(std::integral_constant<bool, UseDynamicSizedMatrix_>());
        for (int i = 0; i < N_; ++i) {
            controlPoints.col(i) = Eigen::Map<const Eigen::Matrix<T, OutputDims_, 1>>(controlPointsRaw[i]);
        }

        // Evaluate smoothness values.
        Eigen::Map<Eigen::Matrix<T, OutputDims_, N_>> residuals(residualRaw);
        residuals = controlPoints * m_.template cast<T>();

        return true;
    }

    /**
     * @brief Cost function used in AutoDiffCostFunction.
     *
     * Enable if is necessary because other the wrong overload would be picked if it is used in a dynamic cost
     * function.
     *
     * @param[in] controlPoint The control points.
     * @param[in,out] ts The other control points and the residual pointer.
     * @return True on success, otherwise false.
     */
    template <typename T, typename... Ts>
    typename std::enable_if<!std::is_pointer<T>::value, bool>::type operator()(const T* controlPoint, Ts*... ts) const {
        static_assert(N_ == sizeof...(Ts), "Invalid number of control points specified.");

        // Build control points array.
        std::array<const T*, sizeof...(Ts) + 1> controlPointsRaw{{controlPoint, ts...}};

        // The last element of the control points is the place to store the output.
        T* residualRaw = std::get<sizeof...(Ts)>(std::make_tuple(controlPoint, ts...));

        return this->operator()(controlPointsRaw.data(), residualRaw);
    }

private:
    /** @brief The smoothness matrix. */
    Eigen::Matrix<double, EigenN_, EigenN_> m_;
};

} // namespace internal
} // namespace ubs
