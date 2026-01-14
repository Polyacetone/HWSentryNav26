#pragma once
#include "array.hpp"
#include "no_discard.hpp"
#include "../utilities.hpp"

namespace ubs {
/** @brief Internal namespace */
namespace internal {

/**
 * @brief Represents a partial derivative of a multivariate function.
 *
 * E.g. in the case of a two dimensional function, the partial derivative
 * @f[
 * \frac{\partial^2 f(x,y)}{\partial x \partial y}
 * @f]
 * would be represented as
 *
 * @tparam Dimensions_ The dimensionality of the function.
 */
template <int Dimensions_>
struct PartialDerivativeData {
    /** @brief The multiplicity of the partial derivative. */
    int multiplicity;

    /** @brief The partial derivatives. */
    Array<int, Dimensions_> partialDerivatives;
};

/**
 * @brief Class to determine the partial derivatives needed to calculate a total derivative.
 *
 * Given the dimension of a multivariate function and the total derivative one would like to calculate, this class
 * determines the partial derivatives and the multiplicity needed to calculate the total derivative. All of that
 * calculation is done at compile time.
 *
 * As an example the second total derivative a two dimensional function (Hessian matrix) would look like:
 * @f[
 * \begin{pmatrix}
 * \frac{\partial^2 f}{\partial x^2}          & \frac{\partial^2 f}{\partial x \partial y} \\
 * \frac{\partial^2 f}{\partial x \partial y} & \frac{\partial^2 f}{\partial y^2}          \\
 * \end{pmatrix}
 * @f]
 *
 * This class would provide all different partial derivatives in an array of PartialDerivativeData:
 * | Idx | multiplicity | dx  | dy  |
 * |----:|-------------:|----:|----:|
 * |  0  |     1        |  2  |  0  |
 * |  1  |     1        |  0  |  2  |
 * |  2  |     2        |  1  |  1  |
 * @tparam Dimensions_ The dimension of the multivariate function.
 * @tparam TotalDerivative_ The total derivative one would like to compute.
 */
template <int Dimensions_, int TotalDerivative_>
class TotalDerivative {
private:
    /** @brief Upper limit of number of partial derivatives.
     *
     * The derivative value span is from 0 to TotalDerivative_. So one need @f$ (n_{TotalDerivative} +
     * 1)^{n_{Dimensions}} @f$ space.
     */
    static constexpr int FullNumPartialDerivatives_ = ubs::power(TotalDerivative_ + 1, Dimensions_);
    /** @brief Array containing all partial derivatives (including ones with zero multiplicity). */
    using FullPartialDerivativeData_ = Array<PartialDerivativeData<Dimensions_>, FullNumPartialDerivatives_>;

    /**
     * @brief Helper function to prepare derivatives.
     * @sa prepareDerivatives()
     * @param[in] partialDerivatives The current partial derivative.
     * @param[out] result Full partial derivative data in which the result will be stored.
     */
    template <int /*CurTotalDerivative*/>
    static constexpr void prepareDerivatives(const Array<int, Dimensions_>& partialDerivatives,
                                             FullPartialDerivativeData_& result,
                                             std::true_type /*finalDerivative*/) {
        // Compute the index of the partial derivatives. The goal here is to create the same index for e.g. dxdy and
        // dydx. To achieve this the index is computed based on how many times in each dimension the derivative is
        // taken. As the derivative is between zero and TotalDerivative_ the index is calculated like for a matrix
        // with dimensionality Dimensions_ and size (TotalDerivative_ + 1) in each dimension.
        int idx = partialDerivatives[0];
        for (int i = 1; i < Dimensions_; ++i) {
            idx = (idx * (TotalDerivative_ + 1)) + partialDerivatives[i];
        }

        ++result[idx].multiplicity;
        result[idx].partialDerivatives = partialDerivatives;
    }

    /**
     * @brief Helper function to prepare derivatives.
     * @sa prepareDerivatives()
     * @param[in] partialDerivatives The current partial derivative.
     * @param[out] result Full partial derivative data in which the result will be stored.
     */
    template <int CurTotalDerivative>
    static constexpr void prepareDerivatives(const Array<int, Dimensions_>& partialDerivatives,
                                             FullPartialDerivativeData_& result,
                                             std::false_type /*finalDerivative*/) {
        for (int i = 0; i < Dimensions_; ++i) {
            auto d = partialDerivatives;
            ++d[i];
            prepareDerivatives<CurTotalDerivative + 1>(
                d, result, std::integral_constant<bool, CurTotalDerivative + 1 == TotalDerivative_>());
        }
    }

    /**
     * @brief Determine the partial derivatives given the total derivative.
     * @return All partial derivative data including those with zero multiplicity.
     */
    UBS_NO_DISCARD static constexpr FullPartialDerivativeData_ prepareDerivatives() {
        Array<int, Dimensions_> derivative{};
        FullPartialDerivativeData_ data{};
        prepareDerivatives<0>(derivative, data, std::integral_constant<bool, (TotalDerivative_ == 0)>());
        return data;
    }

    /**
     * @brief Determine the number of partial derivatives with non-zero multiplicity.
     * @param[in] a The full partial derivatives.
     * @return The number of partial derivatives with non-zero multiplicity.
     */
    UBS_NO_DISCARD static constexpr int countNonZero(const FullPartialDerivativeData_& a) {
        int ret = 0;
        for (int i = 0; i < a.size(); ++i) {
            if (a[i].multiplicity > 0) {
                ++ret;
            }
        }
        return ret;
    }

    /** @brief All partial derivatives including those with zero multiplicity.*/
    static constexpr auto FullPartialDerivatives_ = prepareDerivatives();

public:
    /** @brief Number of different partial derivatives needed to compute the total derivative. */
    static constexpr int NumPartialDerivatives = countNonZero(FullPartialDerivatives_);
    /** @brief Type containing the partial derivative data needed to compute the total derivative. */
    using PartialDerivatives = Array<PartialDerivativeData<Dimensions_>, NumPartialDerivatives>;

    /**
     * @return The partial derivatives needed to compute the total derivative.
     */
    UBS_NO_DISCARD static constexpr PartialDerivatives getPartialDerivatives() {
        PartialDerivatives ret{};

        // Use the full partial derivatives and filter those with zero multiplicity.
        int counter = 0;
        for (const auto& d : FullPartialDerivatives_) {
            if (d.multiplicity > 0) {
                ret[counter] = d;
                ++counter;
            }
        }

        return ret;
    }
};

} // namespace internal
} // namespace ubs
