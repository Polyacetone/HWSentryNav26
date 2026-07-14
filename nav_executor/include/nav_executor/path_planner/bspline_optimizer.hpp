#pragma once

#include <expected>
#include <memory>
#include <vector>
#include <nav_executor/path_planner/nav_map.hpp>

namespace nav_executor {
class BSplineOptimizer {
public:
    using Ptr = std::shared_ptr<BSplineOptimizer>;
    using ConstPtr = std::shared_ptr<const BSplineOptimizer>;

    // 所有 weight 均直接乘到量纲归一化后的 residual 上，再由 Ceres 平方。
    struct StepDetectionParams {
        double norm_threshold;
        double norm_transition;
        double samples_per_meter;
    };

    struct SolverParams {
        double samples_per_meter;
        int max_iterations;
    };

    struct ObjectiveWeights {
        double obstacle;
        double direction;
        double step_traversal;
        double endpoint;
        double smoothness;
        double length;
    };

    struct SmoothPenaltyParams {
        double weight;
        double beta;
    };

    struct CurvaturePenaltyParams {
        SmoothPenaltyParams base;
        SmoothPenaltyParams limit;
        double denominator_regularization_length;
        double tangent_gate_threshold;
    };

    struct TangentRegularizationParams {
        double weight;
        double min_normalized_ratio;
    };

    struct WarmupCurvatureParams {
        double max_curvature;
        CurvaturePenaltyParams penalty;
    };

    struct MainCurvatureParams {
        double near_step_max_curvature;
        double far_from_step_max_curvature;
        double step_extension_distance;
        double step_transition_distance;
        CurvaturePenaltyParams penalty;
    };

    struct RefinementParams {
        int max_iterations;
        double interval_iou_threshold;
    };

    struct WarmupParams {
        SolverParams solver;
        ObjectiveWeights objective_weights;
        TangentRegularizationParams tangent_regularization;
        WarmupCurvatureParams curvature;
    };

    struct MainParams {
        SolverParams solver;
        RefinementParams refinement;
        ObjectiveWeights objective_weights;
        TangentRegularizationParams tangent_regularization;
        MainCurvatureParams curvature;
    };

    struct Params {
        StepDetectionParams step_detection;
        WarmupParams warmup;
        MainParams main;
    };

    explicit BSplineOptimizer(Params params);

    // 返回优化后的控制点、warmup路径、最终采样点
    std::expected<std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>>, std::string> optimize(
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const TerrainTraversalConstraints& terrain_constraints,
        const std::vector<Eigen::Vector2d>& init_path,
        const Eigen::Vector2d& start_grid,
        const Eigen::Vector2d& goal_grid
    ) const;

private:
    Params params_;
};
} // namespace nav_executor
