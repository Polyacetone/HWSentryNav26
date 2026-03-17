#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include <path_follower/nav_map.hpp>
#include <path_follower/utils.hpp>

namespace path_follower {

struct StepRoutingMaskParams {
    // 路径切向与台阶方向点积阈值：超过则认为“路径经过台阶”（触发掩码）
    double path_align_dot_threshold;

    // 掩码半径：在 full_effect_radius 内 alpha=1；在 full_effect_radius~cutoff_radius 线性衰减到 0
    double full_effect_radius;
    double cutoff_radius;
    int length_num_samples;
};

/// 将“全局路径经过台阶”的掩码同时作用于：
/// 1) 台阶额外代价层（用于 merge 进代价地图）
/// 2) 台阶方向场（用于 MPC 方向对齐项）
///
/// 语义：
/// - 全局路径经过台阶：保留方向场；台阶额外代价被擦除（不增加额外代价）
/// - 全局路径不经过台阶：抹去方向场；台阶额外代价保留（增加额外代价）
class StepRoutingMask {
public:
    explicit StepRoutingMask(const StepRoutingMaskParams& params);

    /// 设置基础地图（全局方向场 + 全局代价图几何参数）。
    /// direction_map 与 cost_map 需同尺寸/分辨率/原点。
    void initialize(const CostMap& cost_map, DirectionMap::ConstPtr direction_map);

    /// 基于当前全局路径更新输出层。
    void update(const std::optional<SplineD>& global_path);

    [[nodiscard]] bool ready() const { return static_cast<bool>(base_direction_map_); }

    [[nodiscard]] CostMap::ConstPtr step_cost_layer() const { return step_cost_layer_; }
    [[nodiscard]] DirectionMap::ConstPtr masked_direction_map() const { return masked_direction_map_; }

private:
    struct KernelCell {
        int dx;
        int dy;
        double alpha;
    };

    [[nodiscard]] double approximate_path_length(const SplineD& spline) const;
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
