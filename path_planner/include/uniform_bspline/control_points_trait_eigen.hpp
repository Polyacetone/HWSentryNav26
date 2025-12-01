#pragma once
#include <Eigen/Core>

#include "control_points_trait.hpp"
#include "internal/no_discard.hpp"

namespace ubs {

/** @brief The control points trait for Eigen::Matrix. */
template <typename OutputType_, int N_, int M_>
struct ControlPointsTrait<Eigen::Matrix<OutputType_, N_, M_>> {
    using Container = Eigen::Matrix<OutputType_, N_, M_>;

    template <std::size_t Dims>
    static void resize(Container& c, const std::array<int, Dims>& shape) {
        static_assert(Dims <= 2, "An eigen matrix can only used for up to two dimensions.");
        resize(c, shape, std::integral_constant<int, Dims>());
    }

    UBS_NO_DISCARD static int getNumElements(const Container& c) {
        return int(c.size());
    }

    UBS_NO_DISCARD static OutputType_* data(Container& c) {
        return c.data();
    }

    UBS_NO_DISCARD static const OutputType_* data(const Container& c) {
        return c.data();
    }

    UBS_NO_DISCARD static int getStride(const Container& c, int dim) {
        switch (dim) {
        case 0:
            return (M_ == 1) ? 1 : int(c.rows());
        case 1:
            assert(M_ != 1);
            return 1;
        default:
            assert(false);
            return 0;
        }
    }

    UBS_NO_DISCARD static int getSize(const Container& c, int dim) {
        if (M_ == 1) {
            return int(c.size());
        }

        switch (dim) {
        case 0:
            return int(c.cols());
        case 1:
            return int(c.rows());
        default:
            assert(false);
            return 0;
        }
    }

private:
    template <std::size_t Dims>
    static void resize(Container& c, const std::array<int, Dims>& shape, std::integral_constant<int, 1> /*dim*/) {
        c.conservativeResize(shape[0], 1);
    }

    template <std::size_t Dims>
    static void resize(Container& c, const std::array<int, Dims>& shape, std::integral_constant<int, 2> /*dim*/) {
        c.conservativeResize(shape[1], shape[0]);
    }
};
} // namespace ubs
