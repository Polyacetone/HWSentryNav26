#pragma once
#include <array>

#include <Eigen/Core>
#include <boost/multi_array.hpp>

#include "internal/no_discard.hpp"

namespace ubs {

/**
 * @brief  Wrapper class around boost::multi_array to fix assignment operators.
 *
 * This class fixes the assignment operator for boost::multi_array, as the behavior of boost::multi_array assignment
 * operator is, that the array to which will be assigned (lhs), needs to have the right shape. This means that every
 * class holding a boost::multi_array as member needs to have a custom (move) assignment operator.
 *
 * This class implements the (move) assignment operator in a way, that the lhs is resized before calling the
 * boost::multi_array assignment operator. Otherwise it behaves like a boost::multi_array.
 *
 * @tparam ValueType_ The value type.
 * @tparam NumDims_ The number of dimensions.
 * @tparam Allocator_ The allocator type.
 */
template <typename ValueType_, std::size_t NumDims_, typename Allocator_ = std::allocator<ValueType_>>
class MultiArray : public boost::multi_array<ValueType_, NumDims_, Allocator_> {
public:
    // Inherit all constructors from boost::multi_array.
    // NOLINTNEXTLINE(modernize-use-equals-default)
    using boost::multi_array<ValueType_, NumDims_, Allocator_>::multi_array;

    // The copy constructor and move constructor of boost::multi_array do the resize so the default can be used.
    MultiArray(const MultiArray<ValueType_, NumDims_, Allocator_>&) = default;
    MultiArray(MultiArray<ValueType_, NumDims_, Allocator_>&&) noexcept = default;

    ~MultiArray() = default;

    /**
     * @brief The assignment operator which resizes the left-hand side before assignment.
     * @param[in] other The right-hand side.
     * @return A reference to this.
     */
    MultiArray<ValueType_, NumDims_, Allocator_>& operator=(const MultiArray<ValueType_, NumDims_, Allocator_>& other) {
        if (this != &other) {
            MultiArray::resizeInternal(other);
            boost::multi_array<ValueType_, NumDims_, Allocator_>::operator=(other);
        }

        return *this;
    }

    /**
     * @brief The assignment operator which resizes the left-hand side before assignment.
     * @param[in] other The right-hand side.
     * @return A reference to this.
     */
    // NOLINTNEXTLINE(performance-noexcept-move-constructor): This cannot be nothrow as we are doing a copy here.
    MultiArray<ValueType_, NumDims_, Allocator_>& operator=(MultiArray<ValueType_, NumDims_, Allocator_>&& other) {
        if (this != &other) {
            MultiArray::resizeInternal(other);
            boost::multi_array<ValueType_, NumDims_, Allocator_>::operator=(std::move(other));
        }

        return *this;
    }

    /**
     * @param[in] dim The dimension.
     * @return The number of elements in the specified dimension.
     */
    // NOLINTNEXTLINE(readability-identifier-naming): Case style used according to boost::multi_array.
    UBS_NO_DISCARD int get_size(int dim) const {
        assert(dim >= 0 && dim < NumDims_);
        return this->shape()[dim]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    /**
     * @param[in] dim The dimension.
     * @return The stride of the specified dimension in element size.
     */
    // NOLINTNEXTLINE(readability-identifier-naming): Case style used according to boost::multi_array.
    UBS_NO_DISCARD int get_stride(int dim) const {
        assert(dim >= 0 && dim < NumDims_);
        return this->strides()[dim]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    /**
     * @return A flat iterator to the first element of the multi_array.
     */
    // NOLINTNEXTLINE(readability-identifier-naming): Case style used according to boost::multi_array.
    UBS_NO_DISCARD typename boost::multi_array<ValueType_, NumDims_, Allocator_>::element* flat_begin() {
        assert(is_continuous());
        return this->data();
    }

    /**
     * @return A constant flat iterator to the first element of the multi_array.
     */
    // NOLINTNEXTLINE(readability-identifier-naming): Case style used according to boost::multi_array.
    UBS_NO_DISCARD const typename boost::multi_array<ValueType_, NumDims_, Allocator_>::element* flat_begin() const {
        assert(is_continuous());
        return this->data();
    }

    /**
     * @return A constant flat iterator to the first element of the multi_array.
     */
    // NOLINTNEXTLINE(readability-identifier-naming): Case style used according to boost::multi_array.
    UBS_NO_DISCARD const typename boost::multi_array<ValueType_, NumDims_, Allocator_>::element* flat_cbegin() const {
        assert(is_continuous());
        return this->data();
    }

    /**
     * @return A flat iterator to the last element of the multi_array.
     */
    // NOLINTNEXTLINE(readability-identifier-naming): Case style used according to boost::multi_array.
    UBS_NO_DISCARD typename boost::multi_array<ValueType_, NumDims_, Allocator_>::element* flat_end() {
        assert(is_continuous());
        return this->data() + this->num_elements(); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    /**
     * @return A constant flat iterator to the last element of the multi_array.
     */
    // NOLINTNEXTLINE(readability-identifier-naming): Case style used according to boost::multi_array.
    UBS_NO_DISCARD const typename boost::multi_array<ValueType_, NumDims_, Allocator_>::element* flat_end() const {
        assert(is_continuous());
        return this->data() + this->num_elements(); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    /**
     * @return A constant flat iterator to the last element of the multi_array.
     */
    // NOLINTNEXTLINE(readability-identifier-naming): Case style used according to boost::multi_array.
    UBS_NO_DISCARD const typename boost::multi_array<ValueType_, NumDims_, Allocator_>::element* flat_cend() const {
        assert(is_continuous());
        return this->data() + this->num_elements(); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    /**
     * @return True if continuous otherwise false.
     */
    // NOLINTNEXTLINE(readability-identifier-naming): Case style used according to boost::multi_array.
    UBS_NO_DISCARD bool is_continuous() const {
        int curNumElements = 1;
        for (int dimIdx = NumDims_ - 1; dimIdx >= 0; --dimIdx) {
            if (get_stride(dimIdx) != curNumElements) {
                return false;
            }

            curNumElements *= get_size(dimIdx);
        }

        return curNumElements == this->num_elements();
    }

private:
    /**
     * @brief Resize this multi_array to the same shape as another multi_array.
     * @param[in] other The other multi_array.
     */
    template <typename Other>
    void resizeInternal(const Other& other) {
        using Index = typename boost::multi_array<ValueType_, NumDims_, Allocator_>::index;
        std::array<Index, NumDims_> shape{};
        std::copy_n(other.shape(), NumDims_, shape.begin());
        this->resize(shape);
    }
};

template <typename ValueType, std::size_t NumDims>
using EigenAlignedMultiArray = MultiArray<ValueType, NumDims, Eigen::aligned_allocator<ValueType>>;

}; // namespace ubs
