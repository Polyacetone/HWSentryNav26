#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include <path_follower/common/nav_map.hpp>
#include <path_follower/safety/recovery_helpers.hpp>

namespace path_follower {

struct StepRoutingMaskParams {
    double path_align_dot_threshold;
    double full_effect_radius;
    double cutoff_radius;
    int length_num_samples;
};

class StepRoutingMask {
public:
    explicit StepRoutingMask(const StepRoutingMaskParams& params);

    void initialize(const CostMap& cost_map, DirectionMap::ConstPtr direction_map);

    void update(const std::optional<SplinePath>& global_path);

    [[nodiscard]] bool ready() const { return static_cast<bool>(base_direction_map_); }

    [[nodiscard]] CostMap::ConstPtr step_cost_layer() const { return step_cost_layer_; }
    [[nodiscard]] DirectionMap::ConstPtr masked_direction_map() const { return masked_direction_map_; }
    [[nodiscard]] DirectionMap::ConstPtr base_direction_map() const { return base_direction_map_; }

private:
    struct KernelCell {
        int dx;
        int dy;
        double alpha;
    };

    [[nodiscard]] double approximate_path_length(const SplinePath& spline) const;
    void build_kernel(double resolution);
    void apply_kernel_at(const Eigen::Vector2i& grid_coord, std::vector<double>& max_alpha) const;

    StepRoutingMaskParams params_;

    DirectionMap::ConstPtr base_direction_map_;
    std::vector<uint8_t> base_step_cost_data_;

    std::vector<KernelCell> kernel_;

    CostMap::ConstPtr step_cost_layer_;
    DirectionMap::ConstPtr masked_direction_map_;
};

} // namespace path_follower
