#pragma once
#include "multi_array.hpp"

#include "control_points_trait.hpp"
#include "internal/no_discard.hpp"

namespace ubs {
/** @brief The control points trait for ubs::MultiArray. */
template <typename OutputType_, std::size_t InputDims_, typename Allocator_>
struct ControlPointsTrait<ubs::MultiArray<OutputType_, InputDims_, Allocator_>> {
    using Container = ubs::MultiArray<OutputType_, InputDims_, Allocator_>;

    template <std::size_t Dims>
    static void resize(Container& c, const std::array<int, Dims>& shape) {
        static_assert(Dims == InputDims_, "Shape mismatch.");
        c.resize(shape);
    }

    UBS_NO_DISCARD static int getNumElements(const Container& c) {
        return c.num_elements();
    }

    UBS_NO_DISCARD static OutputType_* data(Container& c) {
        return c.data();
    }

    UBS_NO_DISCARD static const OutputType_* data(const Container& c) {
        return c.data();
    }

    UBS_NO_DISCARD static int getStride(const Container& c, int dim) {
        assert(dim >= 0 && dim < (int)InputDims_);
        return c.get_stride(dim);
    }

    UBS_NO_DISCARD static int getSize(const Container& c, int dim) {
        assert(dim >= 0 && dim < (int)InputDims_);
        return c.get_size(dim);
    }
};
} // namespace ubs
