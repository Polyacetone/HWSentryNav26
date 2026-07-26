#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/environment/nav_map.hpp>
#include <nav_executor/common/trajectory/minco_trajectory.hpp>

namespace nav_executor {

struct StepRoutingMaskParams {
    double path_align_dot_threshold;
    double full_effect_radius;
    double cutoff_radius;
    int length_num_samples;
};

// 初始化后底图保持不变，compute 可在规划线程并发调用。
class StepRoutingMask {
public:
    struct Layers {
        CostMap::ConstPtr step_cost_layer;
        DirectionMap::ConstPtr masked_direction_map;
    };

    explicit StepRoutingMask(const StepRoutingMaskParams& params);

    void initialize(const CostMap& cost_map, DirectionMap::ConstPtr direction_map);

    [[nodiscard]] bool ready() const { return static_cast<bool>(base_direction_map_); }

    // nullopt 表示不保留任何台阶通道。
    [[nodiscard]] Layers compute(const std::optional<MincoTrajectory>& global_path) const;

    [[nodiscard]] DirectionMap::ConstPtr base_direction_map() const { return base_direction_map_; }

private:
    struct KernelCell {
        int dx;
        int dy;
        double alpha;
    };

    [[nodiscard]] double approximate_path_length(const MincoTrajectory& spline) const;
    void build_kernel(double resolution);
    void apply_kernel_at(const Eigen::Vector2i& grid_coord, std::vector<double>& max_alpha) const;

    StepRoutingMaskParams params_;

    DirectionMap::ConstPtr base_direction_map_;
    std::vector<uint8_t> base_step_cost_data_;

    std::vector<KernelCell> kernel_;
};

} // namespace nav_executor
