#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/path_planner/nav_map.hpp>
#include <nav_executor/common/spline_path.hpp>

namespace nav_executor {

struct StepRoutingMaskParams {
    double path_align_dot_threshold;
    double full_effect_radius;
    double cutoff_radius;
    int length_num_samples;
};

// 规划期产出的台阶掩码层。
//
// initialize() 由 nav_executor_node 在全局地图到达后调用一次，之后 base
// 数据不可变。compute() 是 const、无副作用的：对给定样条产出一对新的
// (step_cost_layer, masked_direction_map)，可在规划 worker 线程安全调用。
class StepRoutingMask {
public:
    struct Layers {
        CostMap::ConstPtr step_cost_layer;
        DirectionMap::ConstPtr masked_direction_map;
    };

    explicit StepRoutingMask(const StepRoutingMaskParams& params);

    void initialize(const CostMap& cost_map, DirectionMap::ConstPtr direction_map);

    [[nodiscard]] bool ready() const { return static_cast<bool>(base_direction_map_); }

    // 对给定路径（nullopt = 无路径经过）产出台阶掩码层。无副作用。
    [[nodiscard]] Layers compute(const std::optional<SplinePath>& global_path) const;

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
};

} // namespace nav_executor
