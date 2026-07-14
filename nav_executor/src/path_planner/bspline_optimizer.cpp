#include <ceres/ceres.h>

#include <numeric>

#include <nav_executor/path_planner/bspline_optimizer.hpp>
#include <uniform_bspline/uniform_bspline.hpp>
#include <uniform_bspline_ceres/uniform_bspline_ceres.hpp>

namespace nav_executor {
using Spline = ubs::UniformBSpline<double, 2, double, Eigen::Vector2d, std::vector<Eigen::Vector2d>>;
}

namespace {
using nav_executor::BSplineOptimizer;
using nav_executor::CostMap;
using nav_executor::DirectionMap;
using nav_executor::Spline;

constexpr double EPS = 1e-9;
constexpr double TANGENT_REGULARIZATION_BETA = 10.0;

ceres::Solver::Options default_solver_options(int max_iterations) {
    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_QR;
    options.use_nonmonotonic_steps = true;
    options.max_num_iterations = std::max(max_iterations, 1);
    return options;
}

inline double smoothstep01(double t) {
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

template <typename T>
T smooth_relu(const T& x, double beta) {
    const double safe_beta = std::max(beta, 1e-6);
    const T bx = T(safe_beta) * x;
    return (ceres::sqrt(bx * bx + T(1.0)) + bx) / T(2.0 * safe_beta);
}

struct PathGridTerrainSample {
    Eigen::Vector2d dir; // 方向场向量（map 帧）
    uint8_t label;       // 地形标签
};

PathGridTerrainSample sample_direction_terrain_from_path_grid(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const Eigen::Vector2d& path_grid
) {
    const Eigen::Vector2d map_coord = cost_map.grid_coord_to_map(path_grid);
    const Eigen::Vector2d dir_grid = direction_map.map_coord_to_grid(map_coord);
    return {direction_map.interpolate(dir_grid), direction_map.terrain_at(dir_grid)};
}

double step_gate(double direction_norm, double threshold, double transition) {
    if (transition <= 0.0) {
        return direction_norm >= threshold ? 1.0 : 0.0;
    }

    return smoothstep01((direction_norm - threshold) / transition);
}

double step_gate_derivative(double direction_norm, double threshold, double transition) {
    if (transition <= 0.0 || direction_norm <= threshold || direction_norm >= threshold + transition) {
        return 0.0;
    }
    const double t = (direction_norm - threshold) / transition;
    return 6.0 * t * (1.0 - t) / transition;
}

double estimate_polyline_length_grid(const std::vector<Eigen::Vector2d>& path) {
    double length = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
        length += (path[i] - path[i - 1]).norm();
    }
    return length;
}

struct SplineSample {
    double u;
    double arc_length_m;
    Eigen::Vector2d pos_grid;
    double step_norm;
    Eigen::Vector2d dir;   // 方向场向量（map 帧），用于判定穿越方向
    Eigen::Vector2d vel_grid; // 样条一阶导（grid 帧切向），用于判定穿越方向
    uint8_t terrain_label; // 地形标签，用于查询 run_up
};

struct SampledSpline {
    std::vector<SplineSample> samples;
    double total_length_m;
};

struct StepInterval {
    double start_m;
    double end_m;
};

struct ProblemSample {
    double u;
    double max_curvature;
    double integration_length_m;
    double integration_parameter;
};

struct StageSolveParams {
    nav_executor::BSplineOptimizer::ObjectiveWeights objective_weights;
    int max_iterations;
    nav_executor::BSplineOptimizer::TangentRegularizationParams tangent_regularization;
    nav_executor::BSplineOptimizer::CurvaturePenaltyParams curvature;
};

SampledSpline sample_spline(
    const Spline& spline,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    double samples_per_meter,
    double length_hint_m
) {
    const double clamped_length_hint = std::max(length_hint_m, cost_map.resolution);
    const double clamped_samples_per_meter = std::max(samples_per_meter, 1e-3);
    const int num_samples = std::max(2, static_cast<int>(std::ceil(clamped_length_hint * clamped_samples_per_meter)) + 1);

    SampledSpline sampled;
    sampled.samples.reserve(static_cast<size_t>(num_samples));

    Eigen::Vector2d prev = spline.evaluate(0.0);
    double arc_length_m = 0.0;
    for (int i = 0; i < num_samples; ++i) {
        const double u = num_samples == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(num_samples - 1);
        const Eigen::Vector2d pos_grid = spline.evaluate(u);
        if (i > 0) {
            arc_length_m += (pos_grid - prev).norm() * cost_map.resolution;
        }

        const PathGridTerrainSample dt = sample_direction_terrain_from_path_grid(cost_map, direction_map, pos_grid);
        sampled.samples.push_back({
            .u = u,
            .arc_length_m = arc_length_m,
            .pos_grid = pos_grid,
            .step_norm = dt.dir.norm(),
            .dir = dt.dir,
            .vel_grid = spline.derivative(u, 1),
            .terrain_label = dt.label,
        });
        prev = pos_grid;
    }

    sampled.total_length_m = arc_length_m;
    return sampled;
}

std::vector<StepInterval> merge_intervals(std::vector<StepInterval> intervals) {
    if (intervals.empty()) {
        return intervals;
    }

    std::sort(intervals.begin(), intervals.end(), [](const StepInterval& lhs, const StepInterval& rhs) {
        return lhs.start_m < rhs.start_m;
    });

    std::vector<StepInterval> merged;
    merged.reserve(intervals.size());
    merged.push_back(intervals.front());
    for (size_t i = 1; i < intervals.size(); ++i) {
        StepInterval& back = merged.back();
        if (intervals[i].start_m <= back.end_m) {
            back.end_m = std::max(back.end_m, intervals[i].end_m);
            continue;
        }

        merged.push_back(intervals[i]);
    }
    return merged;
}

// 查询某台阶段入口处的助跑提前量 run_up：由地形标签 + 穿越方向（切向·方向场符号）选出运行模式。
double lookup_run_up(
    const nav_executor::TerrainTraversalConstraints& terrain_constraints,
    const SplineSample& entry_sample
) {
    const bool is_up = entry_sample.vel_grid.dot(entry_sample.dir) > 0.0;
    const nav_executor::TerrainStepRule* rule = terrain_constraints.selected_mode(entry_sample.terrain_label, is_up);
    return rule ? std::max(rule->run_up, 0.0) : 0.0;
}

// 检测台阶区间并按运动学需求扩展。
// 入口（上游 / 低弧长侧）额外扩展 run_up：把「上位机视角的台阶起点」纳入近曲率区，
// 使助跑段被压直、垂直进入。出口侧仅按 step_extension_distance 对称扩展。
std::vector<StepInterval> detect_step_intervals(
    const SampledSpline& sampled_spline,
    const nav_executor::TerrainTraversalConstraints& terrain_constraints,
    double step_norm_threshold,
    double extension_distance_m
) {
    std::vector<StepInterval> intervals;
    const auto& samples = sampled_spline.samples;
    if (samples.empty()) {
        return intervals;
    }

    const double total_length_m = sampled_spline.total_length_m;
    const double extension = std::max(extension_distance_m, 0.0);
    int start_idx = -1;
    for (size_t i = 0; i < samples.size(); ++i) {
        const bool on_step = samples[i].step_norm >= step_norm_threshold;
        if (on_step && start_idx < 0) {
            start_idx = static_cast<int>(i);
        }

        const bool end_of_segment = start_idx >= 0 && (!on_step || i + 1 == samples.size());
        if (!end_of_segment) {
            continue;
        }

        const size_t end_idx = on_step && i + 1 == samples.size() ? i : i - 1;
        const SplineSample& entry_sample = samples[static_cast<size_t>(start_idx)];
        const double run_up = lookup_run_up(terrain_constraints, entry_sample);
        const double upstream_extension = std::max(extension, run_up);
        const double raw_start_m = entry_sample.arc_length_m;
        const double raw_end_m = samples[end_idx].arc_length_m;
        intervals.push_back({
            .start_m = std::max(0.0, raw_start_m - upstream_extension),
            .end_m = std::min(total_length_m, raw_end_m + extension),
        });
        start_idx = -1;
    }

    return merge_intervals(std::move(intervals));
}

double distance_to_intervals(double s_m, const std::vector<StepInterval>& intervals) {
    double min_distance = std::numeric_limits<double>::infinity();
    for (const StepInterval& interval : intervals) {
        if (s_m >= interval.start_m && s_m <= interval.end_m) {
            return 0.0;
        }
        if (s_m < interval.start_m) {
            min_distance = std::min(min_distance, interval.start_m - s_m);
        } else {
            min_distance = std::min(min_distance, s_m - interval.end_m);
        }
    }
    return std::isfinite(min_distance) ? min_distance : std::numeric_limits<double>::infinity();
}

double curvature_limit_from_distance(
    double distance_to_step_m,
    double near_max_curvature,
    double far_max_curvature,
    double transition_distance_m
) {
    if (!std::isfinite(distance_to_step_m)) {
        return far_max_curvature;
    }

    const double transition = std::max(transition_distance_m, 0.0);
    if (transition <= EPS) {
        return distance_to_step_m <= 0.0 ? near_max_curvature : far_max_curvature;
    }

    const double t = smoothstep01(distance_to_step_m / transition);
    return near_max_curvature + (far_max_curvature - near_max_curvature) * t;
}

std::vector<ProblemSample> build_problem_samples(
    const SampledSpline& sampled_spline,
    const std::vector<StepInterval>& step_intervals,
    double near_max_curvature,
    double far_max_curvature,
    double transition_distance_m
) {
    std::vector<ProblemSample> problem_samples;
    problem_samples.reserve(sampled_spline.samples.size());
    const auto& samples = sampled_spline.samples;
    for (size_t i = 0; i < samples.size(); ++i) {
        const SplineSample& sample = samples[i];
        const double distance_to_step_m = distance_to_intervals(sample.arc_length_m, step_intervals);
        const double prev_arc_length = i == 0 ? sample.arc_length_m : samples[i - 1].arc_length_m;
        const double next_arc_length = i + 1 == samples.size() ? sample.arc_length_m : samples[i + 1].arc_length_m;
        const double prev_u = i == 0 ? sample.u : samples[i - 1].u;
        const double next_u = i + 1 == samples.size() ? sample.u : samples[i + 1].u;
        problem_samples.push_back({
            .u = sample.u,
            .max_curvature = curvature_limit_from_distance(
                distance_to_step_m, near_max_curvature, far_max_curvature, transition_distance_m
            ),
            .integration_length_m = 0.5 * (next_arc_length - prev_arc_length),
            .integration_parameter = 0.5 * (next_u - prev_u),
        });
    }
    return problem_samples;
}

double interval_total_length(const std::vector<StepInterval>& intervals) {
    double total = 0.0;
    for (const StepInterval& interval : intervals) {
        total += std::max(0.0, interval.end_m - interval.start_m);
    }
    return total;
}

double interval_iou(const std::vector<StepInterval>& lhs, const std::vector<StepInterval>& rhs) {
    if (lhs.empty() && rhs.empty()) {
        return 1.0;
    }

    double intersection = 0.0;
    size_t i = 0;
    size_t j = 0;
    while (i < lhs.size() && j < rhs.size()) {
        const double overlap_start = std::max(lhs[i].start_m, rhs[j].start_m);
        const double overlap_end = std::min(lhs[i].end_m, rhs[j].end_m);
        if (overlap_end > overlap_start) {
            intersection += overlap_end - overlap_start;
        }

        if (lhs[i].end_m < rhs[j].end_m) {
            ++i;
        } else {
            ++j;
        }
    }

    std::vector<StepInterval> union_intervals = lhs;
    union_intervals.insert(union_intervals.end(), rhs.begin(), rhs.end());
    const double union_length = interval_total_length(merge_intervals(std::move(union_intervals)));
    if (union_length <= EPS) {
        return 1.0;
    }

    return intersection / union_length;
}

class ObstacleCostFunction : public ceres::CostFunction {
public:
    ObstacleCostFunction(
        const ubs::UniformBSplineCeresEvaluator<Spline>& pos_evaluator,
        const CostMap& cost_map,
        double weight,
        double integration_length_m
    ):
        pos_evaluator_(pos_evaluator),
        cost_map_(cost_map),
        residual_scale_(weight * std::sqrt(integration_length_m) / 255.0) {
        mutable_parameter_block_sizes()->clear();
        for (size_t i = 0; i < pos_evaluator_.ControlPointsSupport; ++i) {
            mutable_parameter_block_sizes()->push_back(2);
        }
        set_num_residuals(1);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Eigen::Vector2d point;
        pos_evaluator_.evaluate(parameters[0], parameters[1], parameters[2], point.data());
        const double cost = cost_map_.interpolate(point);
        residuals[0] = residual_scale_ * cost;
        if (jacobians) {
            const Eigen::Vector2d cost_gradient = cost_map_.gradient(point);
            for (size_t i = 0; i < pos_evaluator_.ControlPointsSupport; ++i) {
                if (!jacobians[i]) {
                    continue;
                }

                const Eigen::Vector2d jac = residual_scale_ * pos_evaluator_.basisVals_[i] * cost_gradient;
                jacobians[i][0] = jac.x();
                jacobians[i][1] = jac.y();
            }
        }
        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> pos_evaluator_;
    const CostMap& cost_map_;
    const double residual_scale_;
};

class DirectionCostFunction : public ceres::CostFunction {
public:
    DirectionCostFunction(
        const ubs::UniformBSplineCeresEvaluator<Spline>& pos_evaluator,
        const ubs::UniformBSplineCeresEvaluator<Spline>& vel_evaluator,
        const DirectionMap& direction_map,
        double weight,
        double integration_length_m,
        double step_norm_threshold,
        double step_norm_transition
    ):
        pos_evaluator_(pos_evaluator),
        vel_evaluator_(vel_evaluator),
        direction_map_(direction_map),
        residual_scale_(weight * std::sqrt(integration_length_m)),
        step_norm_threshold_(step_norm_threshold),
        step_norm_transition_(step_norm_transition) {
        mutable_parameter_block_sizes()->clear();
        for (size_t i = 0; i < pos_evaluator_.ControlPointsSupport; ++i) {
            mutable_parameter_block_sizes()->push_back(2);
        }
        set_num_residuals(1);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Eigen::Vector2d pos, vel;
        pos_evaluator_.evaluate(parameters[0], parameters[1], parameters[2], pos.data());
        vel_evaluator_.evaluate(parameters[0], parameters[1], parameters[2], vel.data());

        // interpolate_direction_from_path_grid 在 map 几何一致时退化为 direction_map.interpolate(pos)
        const auto sample = direction_map_.interpolate_with_gradient(pos);
        const Eigen::Vector2d& dir = sample.value;
        const Eigen::Matrix2d& dir_grad = sample.gradient; // d(dir)/d(pos), 每列为偏导向量

        const double dir_norm = dir.norm();
        const double gate = step_gate(dir_norm, step_norm_threshold_, step_norm_transition_);
        constexpr double epsilon = 1e-12;
        const double speed = std::sqrt(vel.squaredNorm() + epsilon);
        const double safe_dir_norm = std::max(dir_norm, epsilon);
        const double cross = vel.x() * dir.y() - vel.y() * dir.x();
        const double alignment = cross / (speed * safe_dir_norm);
        residuals[0] = residual_scale_ * gate * alignment;

        if (jacobians) {
            // d(gate)/d(norm) — smoothstep 导数（门控过渡区外为 0）
            const double dgate_dnorm = step_gate_derivative(dir_norm, step_norm_threshold_, step_norm_transition_);

            // d(norm)/d(dir) = dir / norm（零向量时为零）
            Eigen::Vector2d dnorm_ddir = Eigen::Vector2d::Zero();
            if (dir_norm > epsilon) dnorm_ddir = dir / dir_norm;

            // d(gate)/d(pos) = dgate_dnorm * (dir/norm)ᵀ * dir_grad
            const Eigen::Vector2d dgate_dpos = dgate_dnorm * dir_grad.transpose() * dnorm_ddir;
            const Eigen::Vector2d ddir_norm_dpos = dir_grad.transpose() * dnorm_ddir;

            const Eigen::Vector2d dcross_dpos = vel.x() * dir_grad.row(1) - vel.y() * dir_grad.row(0);
            const Eigen::Vector2d dcross_dvel(dir.y(), -dir.x());
            const Eigen::Vector2d dspeed_dvel = vel / speed;
            const Eigen::Vector2d dalignment_dpos = dcross_dpos / (speed * safe_dir_norm)
                - cross * ddir_norm_dpos / (speed * safe_dir_norm * safe_dir_norm);
            const Eigen::Vector2d dalignment_dvel = dcross_dvel / (speed * safe_dir_norm)
                - cross * dspeed_dvel / (speed * speed * safe_dir_norm);

            const size_t n = pos_evaluator_.ControlPointsSupport;
            for (size_t i = 0; i < n; ++i) {
                if (!jacobians[i]) continue;

                const double bp = pos_evaluator_.basisVals_[i];
                const double bv = vel_evaluator_.basisVals_[i];

                jacobians[i][0] = residual_scale_ * (
                    bp * (dgate_dpos.x() * alignment + gate * dalignment_dpos.x())
                  + bv * gate * dalignment_dvel.x()
                );
                jacobians[i][1] = residual_scale_ * (
                    bp * (dgate_dpos.y() * alignment + gate * dalignment_dpos.y())
                  + bv * gate * dalignment_dvel.y()
                );
            }
        }

        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> pos_evaluator_;
    const ubs::UniformBSplineCeresEvaluator<Spline> vel_evaluator_;
    const DirectionMap& direction_map_;
    const double residual_scale_;
    const double step_norm_threshold_;
    const double step_norm_transition_;
};

class StepFieldMagnitudeCostFunction : public ceres::CostFunction {
public:
    StepFieldMagnitudeCostFunction(
        const ubs::UniformBSplineCeresEvaluator<Spline>& pos_evaluator,
        const DirectionMap& direction_map,
        double weight,
        double integration_length_m,
        double step_norm_threshold,
        double step_norm_transition
    ):
        pos_evaluator_(pos_evaluator),
        direction_map_(direction_map),
        residual_scale_(weight * std::sqrt(integration_length_m)),
        step_norm_threshold_(step_norm_threshold),
        step_norm_transition_(step_norm_transition) {
        mutable_parameter_block_sizes()->clear();
        for (size_t i = 0; i < pos_evaluator_.ControlPointsSupport; ++i) {
            mutable_parameter_block_sizes()->push_back(2);
        }
        set_num_residuals(1);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Eigen::Vector2d pos;
        pos_evaluator_.evaluate(parameters[0], parameters[1], parameters[2], pos.data());

        const auto sample = direction_map_.interpolate_with_gradient(pos);
        const double dir_norm = sample.value.norm();
        const double gate = step_gate(dir_norm, step_norm_threshold_, step_norm_transition_);
        residuals[0] = residual_scale_ * gate;

        if (jacobians) {
            const double eps = 1e-12;
            const double dgate_dnorm = step_gate_derivative(dir_norm, step_norm_threshold_, step_norm_transition_);
            Eigen::Vector2d dnorm_ddir = Eigen::Vector2d::Zero();
            if (dir_norm > eps) dnorm_ddir = sample.value / dir_norm;
            // d(gate)/d(pos) = dgate_dnorm * (dir/norm)ᵀ * d(dir)/d(pos)
            const Eigen::Vector2d dgate_dpos = dgate_dnorm * sample.gradient.transpose() * dnorm_ddir;

            const size_t n = pos_evaluator_.ControlPointsSupport;
            for (size_t i = 0; i < n; ++i) {
                if (!jacobians[i]) continue;
                const double bp = pos_evaluator_.basisVals_[i];
                jacobians[i][0] = residual_scale_ * bp * dgate_dpos.x();
                jacobians[i][1] = residual_scale_ * bp * dgate_dpos.y();
            }
        }

        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> pos_evaluator_;
    const DirectionMap& direction_map_;
    const double residual_scale_;
    const double step_norm_threshold_;
    const double step_norm_transition_;
};

class StartEndPositionCostFunction {
public:
    StartEndPositionCostFunction(
        const ubs::UniformBSplineCeresEvaluator<Spline>& pos_evaluator,
        const Eigen::Vector2d& target,
        double resolution,
        double weight
    ):
        pos_evaluator_(pos_evaluator),
        target_(target),
        residual_scale_(resolution * weight) {}

    template <typename T>
    bool operator()(T const* const p0, T const* const p1, T const* const p2, T* residuals) const {
        T pos[2];
        pos_evaluator_.evaluate(p0, p1, p2, pos);
        residuals[0] = T(residual_scale_) * (pos[0] - T(target_.x()));
        residuals[1] = T(residual_scale_) * (pos[1] - T(target_.y()));
        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> pos_evaluator_;
    const Eigen::Vector2d target_;
    const double residual_scale_;
};

class LengthPenaltyCostFunction {
public:
    LengthPenaltyCostFunction(
        const ubs::UniformBSplineCeresEvaluator<Spline>& vel_evaluator,
        double resolution,
        double weight,
        double integration_parameter
    ):
        vel_evaluator_(vel_evaluator),
        resolution_(resolution),
        residual_scale_(weight * std::sqrt(integration_parameter)) {}

    template <typename T>
    bool operator()(T const* const p0, T const* const p1, T const* const p2, T* residuals) const {
        T vel[2];
        vel_evaluator_.evaluate(p0, p1, p2, vel);
        // 1/2 r² = 1/2 weight² * du * |dp/du|，即加权弧长积分。
        const T speed_m = T(resolution_) * ceres::sqrt(vel[0] * vel[0] + vel[1] * vel[1] + T(EPS));
        residuals[0] = T(residual_scale_) * ceres::sqrt(speed_m);
        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> vel_evaluator_;
    const double resolution_;
    const double residual_scale_;
};

class TangentRegularityCostFunction {
public:
    TangentRegularityCostFunction(
        const ubs::UniformBSplineCeresEvaluator<Spline>& vel_evaluator,
        double resolution,
        double spline_scale,
        double mean_span_length_m,
        double min_normalized_ratio,
        double weight,
        double integration_parameter
    ):
        vel_evaluator_(vel_evaluator),
        resolution_(resolution),
        spline_scale_(spline_scale),
        mean_span_length_m_(mean_span_length_m),
        min_normalized_ratio_(min_normalized_ratio),
        residual_scale_(weight * std::sqrt(integration_parameter)) {}

    template <typename T>
    bool operator()(T const* const p0, T const* const p1, T const* const p2, T* residuals) const {
        T vel[2];
        vel_evaluator_.evaluate(p0, p1, p2, vel);

        const T local_tangent_length_m = T(resolution_ / spline_scale_)
            * ceres::sqrt(vel[0] * vel[0] + vel[1] * vel[1] + T(EPS));
        const T normalized_tangent = local_tangent_length_m / T(mean_span_length_m_);
        const T relative_deficit = (T(min_normalized_ratio_) - normalized_tangent)
            / T(min_normalized_ratio_);
        residuals[0] = T(residual_scale_) * smooth_relu(relative_deficit, TANGENT_REGULARIZATION_BETA);
        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> vel_evaluator_;
    const double resolution_;
    const double spline_scale_;
    const double mean_span_length_m_;
    const double min_normalized_ratio_;
    const double residual_scale_;
};

class CurvatureCostFunction {
public:
    enum class PenaltyType {
        BASE,
        LIMIT,
    };

    CurvatureCostFunction(
        const ubs::UniformBSplineCeresEvaluator<Spline>& vel_evaluator,
        const ubs::UniformBSplineCeresEvaluator<Spline>& acc_evaluator,
        double resolution,
        double spline_scale,
        double max_curvature,
        double weight,
        double penalty_beta,
        double integration_length_m,
        double denominator_regularization_length,
        double tangent_gate_threshold,
        PenaltyType penalty_type
    ):
        vel_evaluator_(vel_evaluator),
        acc_evaluator_(acc_evaluator),
        resolution_(resolution),
        spline_scale_(spline_scale),
        max_curvature_(max_curvature),
        residual_scale_(weight * std::sqrt(integration_length_m)),
        penalty_beta_(penalty_beta),
        denominator_regularization_length_(denominator_regularization_length),
        tangent_gate_threshold_(tangent_gate_threshold),
        penalty_type_(penalty_type) {
    }

    template <typename T>
    bool operator()(T const* const p0, T const* const p1, T const* const p2, T* residuals) const {
        T vel[2];
        T acc[2];
        vel_evaluator_.evaluate(p0, p1, p2, vel);
        acc_evaluator_.evaluate(p0, p1, p2, acc);

        // 对 u 导数按 knot scale 归一化，使正则和门控不随控制点数量变化。
        const T local_vel_x = vel[0] / T(spline_scale_);
        const T local_vel_y = vel[1] / T(spline_scale_);
        const T local_acc_x = acc[0] / T(spline_scale_ * spline_scale_);
        const T local_acc_y = acc[1] / T(spline_scale_ * spline_scale_);
        const T tangent_length_sq_m = T(resolution_ * resolution_)
            * (local_vel_x * local_vel_x + local_vel_y * local_vel_y);
        const double safe_threshold = std::max(tangent_gate_threshold_, 1e-6);
        const T gate = tangent_length_sq_m / (tangent_length_sq_m + T(safe_threshold * safe_threshold));

        const double denominator_regularization_grid = std::max(
            denominator_regularization_length_ / std::max(resolution_, 1e-6), 1e-6
        );
        const T safe_speed_sq_grid = local_vel_x * local_vel_x + local_vel_y * local_vel_y
            + T(denominator_regularization_grid * denominator_regularization_grid);
        const T safe_speed_grid = ceres::sqrt(safe_speed_sq_grid);
        const T denom = T(std::max(resolution_, 1e-6)) * safe_speed_sq_grid * safe_speed_grid;
        const T cross = local_vel_x * local_acc_y - local_vel_y * local_acc_x;
        const T curvature = cross / denom;
        const T curvature_mag = ceres::sqrt(curvature * curvature + T(EPS));
        const T penalty_argument = penalty_type_ == PenaltyType::BASE
            ? curvature_mag
            : curvature_mag - T(max_curvature_);
        residuals[0] = T(residual_scale_) * gate * smooth_relu(penalty_argument, penalty_beta_);
        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> vel_evaluator_;
    const ubs::UniformBSplineCeresEvaluator<Spline> acc_evaluator_;
    const double resolution_;
    const double spline_scale_;
    const double max_curvature_;
    const double residual_scale_;
    const double penalty_beta_;
    const double denominator_regularization_length_;
    const double tangent_gate_threshold_;
    const PenaltyType penalty_type_;
};

std::expected<void, std::string> solve_stage(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    Spline& spline,
    const std::vector<ProblemSample>& problem_samples,
    const StageSolveParams& params,
    double step_norm_threshold,
    double step_norm_transition,
    const Eigen::Vector2d& start_grid,
    const Eigen::Vector2d& goal_grid
) {
    ubs::UniformBSplineCeres<Spline> spline_ceres(spline);
    ceres::Problem problem;
    std::vector<double*> parameter_pointers(spline_ceres.getNumPointParameterPointers());
    const double spline_scale = spline.getScale(0);
    const double sampled_length_m = std::accumulate(
        problem_samples.begin(),
        problem_samples.end(),
        0.0,
        [](double total, const ProblemSample& sample) {
            return total + sample.integration_length_m;
        }
    );
    const double mean_span_length_m = std::max(sampled_length_m / spline_scale, EPS);

    if (params.objective_weights.smoothness > 0.0) {
        // 该库接收 objective 权重；换算后其 residual 系数仍是配置的 weight。
        const double normalized_smoothness_weight = 0.5
            * params.objective_weights.smoothness * params.objective_weights.smoothness
            * cost_map.resolution * cost_map.resolution
            / std::pow(spline_scale, 3.0);
        spline_ceres.addSmoothnessResiduals<2>(problem, normalized_smoothness_weight);
    }

    for (const ProblemSample& sample : problem_samples) {
        const auto data = spline_ceres.getPointData(sample.u);
        spline_ceres.fillParameterPointers(data, parameter_pointers.begin(), parameter_pointers.end());
        const auto pos_evaluator = spline_ceres.getEvaluator(data);
        const auto vel_evaluator = spline_ceres.getEvaluator(data, {1});
        const auto acc_evaluator = spline_ceres.getEvaluator(data, {2});

        if (params.objective_weights.obstacle > 0.0) {
            problem.AddResidualBlock(
                new ObstacleCostFunction(
                    pos_evaluator, cost_map, params.objective_weights.obstacle, sample.integration_length_m
                ),
                nullptr,
                parameter_pointers
            );
        }

        if (params.objective_weights.direction > 0.0) {
            problem.AddResidualBlock(
                new DirectionCostFunction(
                    pos_evaluator,
                    vel_evaluator,
                    direction_map,
                    params.objective_weights.direction,
                    sample.integration_length_m,
                    step_norm_threshold,
                    step_norm_transition
                ),
                nullptr,
                parameter_pointers
            );
        }

        if (params.objective_weights.step_traversal > 0.0) {
            problem.AddResidualBlock(
                new StepFieldMagnitudeCostFunction(
                    pos_evaluator,
                    direction_map,
                    params.objective_weights.step_traversal,
                    sample.integration_length_m,
                    step_norm_threshold,
                    step_norm_transition
                ),
                nullptr,
                parameter_pointers
            );
        }

        if (params.objective_weights.length > 0.0) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<LengthPenaltyCostFunction, 1, 2, 2, 2>(
                    new LengthPenaltyCostFunction(
                        vel_evaluator,
                        cost_map.resolution,
                        params.objective_weights.length,
                        sample.integration_parameter
                    )
                ),
                nullptr,
                parameter_pointers
            );
        }

        if (params.tangent_regularization.weight > 0.0
            && params.tangent_regularization.min_normalized_ratio > 0.0) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<TangentRegularityCostFunction, 1, 2, 2, 2>(
                    new TangentRegularityCostFunction(
                        vel_evaluator,
                        cost_map.resolution,
                        spline_scale,
                        mean_span_length_m,
                        params.tangent_regularization.min_normalized_ratio,
                        params.tangent_regularization.weight,
                        sample.integration_parameter
                    )
                ),
                nullptr,
                parameter_pointers
            );
        }

        if (params.curvature.base.weight > 0.0) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CurvatureCostFunction, 1, 2, 2, 2>(
                    new CurvatureCostFunction(
                        vel_evaluator,
                        acc_evaluator,
                        cost_map.resolution,
                        spline_scale,
                        sample.max_curvature,
                        params.curvature.base.weight,
                        params.curvature.base.beta,
                        sample.integration_length_m,
                        params.curvature.denominator_regularization_length,
                        params.curvature.tangent_gate_threshold,
                        CurvatureCostFunction::PenaltyType::BASE
                    )
                ),
                nullptr,
                parameter_pointers
            );
        }
        if (params.curvature.limit.weight > 0.0) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CurvatureCostFunction, 1, 2, 2, 2>(
                    new CurvatureCostFunction(
                        vel_evaluator,
                        acc_evaluator,
                        cost_map.resolution,
                        spline_scale,
                        sample.max_curvature,
                        params.curvature.limit.weight,
                        params.curvature.limit.beta,
                        sample.integration_length_m,
                        params.curvature.denominator_regularization_length,
                        params.curvature.tangent_gate_threshold,
                        CurvatureCostFunction::PenaltyType::LIMIT
                    )
                ),
                nullptr,
                parameter_pointers
            );
        }
    }

    if (params.objective_weights.endpoint > 0.0) {
        const auto start_data = spline_ceres.getPointData(0.0);
        const auto goal_data = spline_ceres.getPointData(1.0);
        const auto start_evaluator = spline_ceres.getEvaluator(start_data);
        const auto goal_evaluator = spline_ceres.getEvaluator(goal_data);
        std::vector<double*> start_parameter_pointers(spline_ceres.getNumPointParameterPointers());
        std::vector<double*> goal_parameter_pointers(spline_ceres.getNumPointParameterPointers());
        spline_ceres.fillParameterPointers(start_data, start_parameter_pointers.begin(), start_parameter_pointers.end());
        spline_ceres.fillParameterPointers(goal_data, goal_parameter_pointers.begin(), goal_parameter_pointers.end());

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<StartEndPositionCostFunction, 2, 2, 2, 2>(
                new StartEndPositionCostFunction(
                    start_evaluator, start_grid, cost_map.resolution, params.objective_weights.endpoint
                )
            ),
            nullptr,
            start_parameter_pointers
        );
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<StartEndPositionCostFunction, 2, 2, 2, 2>(
                new StartEndPositionCostFunction(
                    goal_evaluator, goal_grid, cost_map.resolution, params.objective_weights.endpoint
                )
            ),
            nullptr,
            goal_parameter_pointers
        );
    }

    const ceres::Solver::Options options = default_solver_options(params.max_iterations);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
        return std::unexpected(summary.BriefReport());
    }

    return {};
}
} // namespace

namespace nav_executor {
BSplineOptimizer::BSplineOptimizer(BSplineOptimizer::Params params): params_(std::move(params)) {
}

std::expected<std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>>, std::string> BSplineOptimizer::optimize(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const std::vector<Eigen::Vector2d>& init_path,
    const Eigen::Vector2d& start_grid,
    const Eigen::Vector2d& goal_grid
) const {
    if (init_path.size() <= 2) {
        return std::unexpected("Initial path too short for optimization");
    }

    Spline spline(init_path);
    const double init_length_m = estimate_polyline_length_grid(init_path) * cost_map.resolution;
    double length_hint_m = std::max(init_length_m, cost_map.resolution);

    const SampledSpline warmup_sampling = sample_spline(
        spline,
        cost_map,
        direction_map,
        params_.warmup.solver.samples_per_meter,
        length_hint_m
    );
    const std::vector<ProblemSample> warmup_problem_samples = build_problem_samples(
        warmup_sampling,
        {},
        params_.warmup.curvature.max_curvature,
        params_.warmup.curvature.max_curvature,
        0.0
    );

    const StageSolveParams warmup_params = {
        .objective_weights = params_.warmup.objective_weights,
        .max_iterations = params_.warmup.solver.max_iterations,
        .tangent_regularization = params_.warmup.tangent_regularization,
        .curvature = params_.warmup.curvature.penalty,
    };
    if (const auto warmup_result = solve_stage(
            cost_map,
            direction_map,
            spline,
            warmup_problem_samples,
            warmup_params,
            params_.step_detection.norm_threshold,
            params_.step_detection.norm_transition,
            start_grid,
            goal_grid
        );
        !warmup_result) {
        return std::unexpected("Warmup optimization failed: " + warmup_result.error());
    }

    // 采样 warmup 后的路径，用于 debug 对比
    const SampledSpline warmup_path_sampling = sample_spline(
        spline,
        cost_map,
        direction_map,
        params_.warmup.solver.samples_per_meter,
        length_hint_m
    );
    std::vector<Eigen::Vector2d> warmup_path;
    warmup_path.reserve(warmup_path_sampling.samples.size());
    for (const SplineSample& sample : warmup_path_sampling.samples) {
        warmup_path.push_back(sample.pos_grid);
    }

    length_hint_m = std::max(sample_spline(
        spline,
        cost_map,
        direction_map,
        params_.step_detection.samples_per_meter,
        length_hint_m
    ).total_length_m, cost_map.resolution);

    const StageSolveParams main_params = {
        .objective_weights = params_.main.objective_weights,
        .max_iterations = params_.main.solver.max_iterations,
        .tangent_regularization = params_.main.tangent_regularization,
        .curvature = params_.main.curvature.penalty,
    };

    const int main_refinement_iterations = params_.main.refinement.max_iterations;
    for (int iter = 0; iter < main_refinement_iterations; ++iter) {
        const SampledSpline detection_before = sample_spline(
            spline,
            cost_map,
            direction_map,
            params_.step_detection.samples_per_meter,
            length_hint_m
        );
        const std::vector<StepInterval> step_intervals_before = detect_step_intervals(
            detection_before,
            terrain_constraints,
            params_.step_detection.norm_threshold,
            params_.main.curvature.step_extension_distance
        );

        const SampledSpline main_sampling = sample_spline(
            spline,
            cost_map,
            direction_map,
            params_.main.solver.samples_per_meter,
            std::max(detection_before.total_length_m, cost_map.resolution)
        );
        const std::vector<ProblemSample> main_problem_samples = build_problem_samples(
            main_sampling,
            step_intervals_before,
            params_.main.curvature.near_step_max_curvature,
            params_.main.curvature.far_from_step_max_curvature,
            params_.main.curvature.step_transition_distance
        );

        if (const auto main_result = solve_stage(
                cost_map,
                direction_map,
                spline,
                main_problem_samples,
                main_params,
                params_.step_detection.norm_threshold,
                params_.step_detection.norm_transition,
                start_grid,
                goal_grid
            );
            !main_result) {
            return std::unexpected("Main optimization failed: " + main_result.error());
        }

        const SampledSpline detection_after = sample_spline(
            spline,
            cost_map,
            direction_map,
            params_.step_detection.samples_per_meter,
            std::max(main_sampling.total_length_m, cost_map.resolution)
        );
        const std::vector<StepInterval> step_intervals_after = detect_step_intervals(
            detection_after,
            terrain_constraints,
            params_.step_detection.norm_threshold,
            params_.main.curvature.step_extension_distance
        );
        length_hint_m = std::max(detection_after.total_length_m, cost_map.resolution);

        if (interval_iou(step_intervals_before, step_intervals_after)
            >= params_.main.refinement.interval_iou_threshold) {
            break;
        }
    }

    const double output_samples_per_meter = std::max(
        params_.warmup.solver.samples_per_meter,
        params_.main.solver.samples_per_meter
    );
    const SampledSpline final_sampling = sample_spline(
        spline,
        cost_map,
        direction_map,
        output_samples_per_meter,
        length_hint_m
    );

    std::vector<Eigen::Vector2d> sample_points;
    sample_points.reserve(final_sampling.samples.size());
    for (const SplineSample& sample : final_sampling.samples) {
        sample_points.push_back(sample.pos_grid);
    }

    return std::tuple{spline.getControlPoints(), warmup_path, sample_points};
}
} // namespace nav_executor
