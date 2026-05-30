#pragma once

#include <expected>
#include <memory>
#include <vector>
#include <path_planner/nav_map.hpp>

namespace path_planner {
class BSplineOptimizer {
public:
    using Ptr = std::shared_ptr<BSplineOptimizer>;
    using ConstPtr = std::shared_ptr<const BSplineOptimizer>;

    struct CurvaturePenaltyParams {
        double base_weight;
        double base_beta;
        double limit_weight;
        double limit_beta;
        double min_speed_epsilon;
        double speed_gate_threshold;
    };

    struct WarmupParams {
        double obstacle_weight;
        double direction_weight;
        double step_weight;
        double start_end_weight;
        double smoothness_weight;
        double samples_per_meter;
        int max_iterations;
        double max_curvature;
        double length_penalty_weight;
        CurvaturePenaltyParams curvature;
    };

    struct MainParams {
        double obstacle_weight;
        double direction_weight;
        double step_weight;
        double start_end_weight;
        double smoothness_weight;
        double samples_per_meter;
        int max_iterations;
        int max_refinement_iterations;
        double near_max_curvature;
        double far_max_curvature;
        double step_extension_distance;
        double step_transition_distance;
        double interval_iou_threshold;
        double length_penalty_weight;
        CurvaturePenaltyParams curvature;
    };

    struct Params {
        double step_norm_threshold;
        double step_norm_transition;
        double step_detection_samples_per_meter;
        WarmupParams warmup;
        MainParams main;
    };

    explicit BSplineOptimizer(Params params);

    // 返回优化后的控制点和采样点
    std::expected<std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>>, std::string> optimize(
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const std::vector<Eigen::Vector2d>& init_path,
        const Eigen::Vector2d& start_grid,
        const Eigen::Vector2d& goal_grid
    ) const;

private:
    Params params_;
};
} // namespace path_planner