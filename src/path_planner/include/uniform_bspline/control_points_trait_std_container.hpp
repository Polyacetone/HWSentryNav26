#pragma once
#include <array>
#include <cassert>
#include <vector>

#include "control_points_trait.hpp"
#include "internal/no_discard.hpp"

namespace ubs {

/** @brief The control points trait for one dimensional continuous containers. */
template <typename Container_>
struct ControlPointsTraitStd {
    using Container = Container_;
    using OutputType = typename Container_::value_type;

    template <std::size_t Dims>
    static void resize(Container& c, const std::array<int, Dims>& shape) {
        static_assert(Dims <= 1, "An eigen matrix can only used for up to two dimensions.");

        c.resize(shape[0]);
    }

    UBS_NO_DISCARD static int getNumElements(const Container& c) {
        return int(c.size());
    }

    UBS_NO_DISCARD static OutputType* data(Container& c) {
        return c.data();
    }

    UBS_NO_DISCARD static const OutputType* data(const Container& c) {
        return c.data();
    }

    UBS_NO_DISCARD static int getStride(const Container& /*c*/, int dim) {
        assert(dim == 0);
        (void)dim;

        return 1;
    }

    UBS_NO_DISCARD static int getSize(const Container& c, int dim) {
        assert(dim == 0);
        (void)dim;

        return int(c.size());
    }
};

template <typename OutputType_, typename Allocator_>
struct ControlPointsTrait<std::vector<OutputType_, Allocator_>>
        : ControlPointsTraitStd<std::vector<OutputType_, Allocator_>> {};

template <typename OutputType_, std::size_t N_>
struct ControlPointsTrait<std::array<OutputType_, N_>> : ControlPointsTraitStd<std::array<OutputType_, N_>> {};

} // namespace ubs
