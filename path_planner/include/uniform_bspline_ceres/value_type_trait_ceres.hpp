#pragma once

#include <ceres/jet.h>
#include <uniform_bspline/value_type_trait.hpp>

namespace ubs {

template <typename T_, int N_>
struct ValueTypeTrait<ceres::Jet<T_, N_>> {
    UBS_NO_DISCARD static int toInt(const ceres::Jet<T_, N_>& val) {
        // To cast a Jet to int, use the value part. As this function is only used to get the segment index, the
        // gradient computation still works.
        return static_cast<int>(val.a);
    }

    template <typename T>
    UBS_NO_DISCARD static ceres::Jet<T_, N_> pow(const ceres::Jet<T_, N_>& val, T ex) {
        return ceres::pow(val, ex);
    }
};

} // namespace ubs
