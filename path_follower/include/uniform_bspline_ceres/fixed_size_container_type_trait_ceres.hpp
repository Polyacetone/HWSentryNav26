#pragma once
#include <ceres/jet.h>
#include <uniform_bspline/fixed_size_container_type_trait.hpp>
#include <uniform_bspline/fixed_size_container_type_trait_default.hpp>

namespace ubs {

template <typename T_, int N_>
struct FixedSizeContainerTypeTrait<ceres::Jet<T_, N_>> : public FixedSizeContainerTypeTraitDefault<ceres::Jet<T_, N_>> {
    // The Jet type needs the Eigen aligned allocator as it contains a fixed size Eigen type.
    using Allocator = Eigen::aligned_allocator<ceres::Jet<T_, N_>>;
};

} // namespace ubs
