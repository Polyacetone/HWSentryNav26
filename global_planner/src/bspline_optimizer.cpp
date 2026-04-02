#include <ceres/ceres.h>
#include <uniform_bspline/uniform_bspline.hpp>
#include <uniform_bspline_ceres/uniform_bspline_ceres.hpp>
#include <global_planner/bspline_optimizer.hpp>
#include <global_planner/nav_map.hpp>

namespace global_planner {
using Spline = ubs::UniformBSpline<double, 2, double, Eigen::Vector2d, std::vector<Eigen::Vector2d>>;
}

namespace {
inline double smoothstep01(double t) {
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

inline double smoothstep(double x, double edge0, double edge1) {
    const double denom = std::max(1e-12, edge1 - edge0);
    const double t = (x - edge0) / denom;
    return smoothstep01(t);
}
} // namespace

// 残差函数类的定义和实现
namespace global_planner {
// 障碍物代价涉及离散的代价地图，需要手动求导
class ObstacleCostFunction : public ceres::CostFunction {
public:
    explicit ObstacleCostFunction(
        const ubs::UniformBSplineCeresEvaluator<Spline>& pos_evaluator,
        const CostMap& cost_map,
        const double weight
    ): pos_evaluator_(pos_evaluator), cost_map_(cost_map), weight_(weight) {
        // 设置参数块，每个控制点是2维向量
        mutable_parameter_block_sizes()->clear();
        for (size_t i = 0; i < pos_evaluator_.ControlPointsSupport; i++) {
            mutable_parameter_block_sizes()->push_back(2);
        }
        set_num_residuals(1);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Eigen::Vector2d point;
        pos_evaluator_.evaluate(parameters[0], parameters[1], parameters[2], point.data());
        const double cost = cost_map_.interpolate(point);
        residuals[0] = weight_ * cost;
        if (jacobians) {
            const Eigen::Vector2d cost_gradient = cost_map_.gradient(point);
            for (size_t i = 0; i < pos_evaluator_.ControlPointsSupport; i++) {
                if (jacobians[i]) {
                    // ∂L/∂P_i = N_i,p(u) * ∂L/∂V
                    Eigen::Vector2d jac = weight_ * pos_evaluator_.basisVals_[i] * cost_gradient;
                    jacobians[i][0] = jac.x();  // 对控制点x分量的导数
                    jacobians[i][1] = jac.y();  // 对控制点y分量的导数
                }
            }
        }
        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> pos_evaluator_;
    const CostMap& cost_map_;
    const double weight_;
};

// 方向代价涉及离散的方向地图，需要手动求导
class DirectionCostFunction: public ceres::CostFunction {
public:
    explicit DirectionCostFunction(
        const ubs::UniformBSplineCeresEvaluator<Spline>& pos_evaluator,
        const ubs::UniformBSplineCeresEvaluator<Spline>& vel_evaluator,
        const DirectionMap& direction_map,
        const double weight,
        const double step_norm_threshold,
        const double step_norm_transition
    ) : pos_evaluator_(pos_evaluator),
        vel_evaluator_(vel_evaluator),
        direction_map_(direction_map),
        weight_(weight),
        step_norm_threshold_(step_norm_threshold),
        step_norm_transition_(step_norm_transition) {
        mutable_parameter_block_sizes()->clear();
        for (size_t i = 0; i < pos_evaluator_.ControlPointsSupport; i++) {
            mutable_parameter_block_sizes()->push_back(2);
        }
        set_num_residuals(1);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Eigen::Vector2d pos, vel;
        pos_evaluator_.evaluate(parameters[0], parameters[1], parameters[2], pos.data());
        vel_evaluator_.evaluate(parameters[0], parameters[1], parameters[2], vel.data());

        const Eigen::Vector2d dir = direction_map_.interpolate(pos);
        const double dir_norm = dir.norm();
        const double gate = smoothstep(dir_norm, step_norm_threshold_, step_norm_threshold_ + step_norm_transition_);
        if (gate <= 1e-6) {
            residuals[0] = 0.0;
            if (jacobians) {
                for (size_t i = 0; i < pos_evaluator_.ControlPointsSupport; i++) {
                    if (jacobians[i]) {
                        jacobians[i][0] = 0.0;
                        jacobians[i][1] = 0.0;
                    }
                }
            }
            return true;
        }

        // 代价：gate(pos) * | vel x dir(pos) |
        // 说明：dir 的模长作为“台阶置信度”，模长小 -> 对齐代价小（避免被吸过去）；
        // 同时 gate 的位置梯度会把路径从台阶区域推开。
        const double cross = vel.x() * dir.y() - vel.y() * dir.x();
        const double abs_cross = std::abs(cross);
        residuals[0] = weight_ * gate * abs_cross;

        if (jacobians) {
            // d gate / d pos：用有限差分近似（direction map 来自离散图）
            constexpr double eps = 0.25;
            const auto gate_at = [&](const Eigen::Vector2d& p) {
                const double n = direction_map_.interpolate(p).norm();
                return smoothstep(n, step_norm_threshold_, step_norm_threshold_ + step_norm_transition_);
            };

            const double gate_xp = gate_at(pos + Eigen::Vector2d(eps, 0.0));
            const double gate_xm = gate_at(pos - Eigen::Vector2d(eps, 0.0));
            const double gate_yp = gate_at(pos + Eigen::Vector2d(0.0, eps));
            const double gate_ym = gate_at(pos - Eigen::Vector2d(0.0, eps));
            const Eigen::Vector2d grad_gate((gate_xp - gate_xm) / (2.0 * eps), (gate_yp - gate_ym) / (2.0 * eps));

            for (size_t i = 0; i < vel_evaluator_.ControlPointsSupport; i++) {
                if (jacobians[i]) {
                    const double basis_pos = pos_evaluator_.basisVals_[i];
                    const double basis_vel = vel_evaluator_.basisVals_[i];
                    const double sign = cross >= 0 ? 1.0 : -1.0;

                    // ∂|cross|/∂P_i (通过 vel)
                    const Eigen::Vector2d d_abs_cross_dPi = sign * basis_vel * Eigen::Vector2d(dir.y(), -dir.x());

                    // ∂gate/∂P_i (通过 pos)
                    const Eigen::Vector2d d_gate_dPi = basis_pos * grad_gate;

                    // 链式法则：d( gate * |cross| ) = |cross| dgate + gate d|cross|
                    const Eigen::Vector2d jac = abs_cross * d_gate_dPi + gate * d_abs_cross_dPi;
                    jacobians[i][0] = weight_ * jac.x();
                    jacobians[i][1] = weight_ * jac.y();
                }
            }
        }
        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> pos_evaluator_;
    const ubs::UniformBSplineCeresEvaluator<Spline> vel_evaluator_;
    const DirectionMap& direction_map_;
    const double weight_;
    const double step_norm_threshold_;
    const double step_norm_transition_;
};

// 台阶惩罚：基于方向场模长（越大越像台阶/边缘），鼓励路径远离台阶
// 方向场来自离散图，使用有限差分近似对位置的梯度，从而能对控制点产生可用梯度
class StepFieldMagnitudeCostFunction : public ceres::CostFunction {
public:
    explicit StepFieldMagnitudeCostFunction(
        const ubs::UniformBSplineCeresEvaluator<Spline>& pos_evaluator,
        const DirectionMap& direction_map,
        const double weight,
        const double step_norm_threshold,
        const double step_norm_transition
    ): pos_evaluator_(pos_evaluator),
       direction_map_(direction_map),
       weight_(weight),
       step_norm_threshold_(step_norm_threshold),
       step_norm_transition_(step_norm_transition) {
        mutable_parameter_block_sizes()->clear();
        for (size_t i = 0; i < pos_evaluator_.ControlPointsSupport; i++) {
            mutable_parameter_block_sizes()->push_back(2);
        }
        set_num_residuals(1);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Eigen::Vector2d pos;
        pos_evaluator_.evaluate(parameters[0], parameters[1], parameters[2], pos.data());

        const double mag = direction_map_.interpolate(pos).norm();
        const double gate = smoothstep(mag, step_norm_threshold_, step_norm_threshold_ + step_norm_transition_);
        residuals[0] = weight_ * gate;

        if (jacobians) {
            constexpr double eps = 0.25; // grid 坐标系下的差分步长
            const auto gate_at = [&](const Eigen::Vector2d& p) {
                const double n = direction_map_.interpolate(p).norm();
                return smoothstep(n, step_norm_threshold_, step_norm_threshold_ + step_norm_transition_);
            };

            const double gate_xp = gate_at(pos + Eigen::Vector2d(eps, 0.0));
            const double gate_xm = gate_at(pos - Eigen::Vector2d(eps, 0.0));
            const double gate_yp = gate_at(pos + Eigen::Vector2d(0.0, eps));
            const double gate_ym = gate_at(pos - Eigen::Vector2d(0.0, eps));

            const Eigen::Vector2d grad((gate_xp - gate_xm) / (2.0 * eps), (gate_yp - gate_ym) / (2.0 * eps));
            for (size_t i = 0; i < pos_evaluator_.ControlPointsSupport; i++) {
                if (jacobians[i]) {
                    Eigen::Vector2d jac = weight_ * pos_evaluator_.basisVals_[i] * grad;
                    jacobians[i][0] = jac.x();
                    jacobians[i][1] = jac.y();
                }
            }
        }
        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> pos_evaluator_;
    const DirectionMap& direction_map_;
    const double weight_;
    const double step_norm_threshold_;
    const double step_norm_transition_;
};

// 起终点保持代价，自动求导
class StartEndPositionCostFunction {
public:
    StartEndPositionCostFunction(
        const ubs::UniformBSplineCeresEvaluator<Spline>& pos_evaluator,
        const Eigen::Vector2d& target,
        const double weight
    ) : pos_evaluator_(pos_evaluator), target_(target), weight_(weight) {}

    template <typename T>
    bool operator()(T const* const p0, T const* const p1, T const* const p2, T* residuals) const {
        T pos[2];
        pos_evaluator_.evaluate(p0, p1, p2, pos);
        residuals[0] = T(weight_) * (pos[0] - T(target_.x()));
        residuals[1] = T(weight_) * (pos[1] - T(target_.y()));
        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> pos_evaluator_;
    const Eigen::Vector2d target_;
    const double weight_;
};

// 速度均匀代价，自动求导
class UniformSpeedCostFunction {
public:
    UniformSpeedCostFunction(
        const ubs::UniformBSplineCeresEvaluator<Spline>& vel_evaluator,
        double target_speed,
        double weight
    ) : vel_evaluator_(vel_evaluator), target_speed_(target_speed), weight_(weight) {}

    template <typename T>
    bool operator()(T const* const p0, T const* const p1, T const* const p2, T* residuals) const {
        T vel[2];
        vel_evaluator_.evaluate(p0, p1, p2, vel);
        T speed = ceres::sqrt(vel[0] * vel[0] + vel[1] * vel[1] + T(1e-8));
        residuals[0] = T(weight_) * (speed - T(target_speed_));
        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> vel_evaluator_;
    const double target_speed_;
    const double weight_;
};
}

// 路径优化器主类实现
namespace global_planner {
BSplineOptimizer::BSplineOptimizer(
    const double smoothness_weight,
    const double length_weight,
    const double obstacle_weight,
    const double direction_weight,
    const double step_weight,
    const double step_norm_threshold,
    const double step_norm_transition,
    const double start_end_weight,
    const double num_samples_per_length,
    const int max_iterations
):
    smoothness_weight_(smoothness_weight),
    uniform_speed_weight_(length_weight),
    obstacle_weight_(obstacle_weight),
    direction_weight_(direction_weight),
    step_weight_(step_weight),
    step_norm_threshold_(step_norm_threshold),
    step_norm_transition_(step_norm_transition),
    start_end_weight_(start_end_weight),
    num_samples_per_length_(num_samples_per_length),
    max_iterations_(max_iterations) {}

std::expected<std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>>, std::string> BSplineOptimizer::optimize(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const std::vector<Eigen::Vector2d>& init_path,
    const Eigen::Vector2d& start_grid,
    const Eigen::Vector2d& goal_grid
) const {
    if (init_path.size() <= 2) {
        return std::unexpected("Initial path too short for optimization");
    }
    Spline spline(init_path);
    ubs::UniformBSplineCeres<Spline> spline_ceres(spline);
    ceres::Problem problem;

    // 添加光滑度代价
    spline_ceres.addSmoothnessResiduals<2>(problem, smoothness_weight_);

    const double path_length = estimate_path_length(init_path);
    const size_t num_samples = static_cast<size_t>(std::ceil(path_length * num_samples_per_length_));
    std::vector<double*> parameter_pointers(spline_ceres.getNumPointParameterPointers());
    for (size_t i = 0; i < num_samples; i++) {
        const double pos_u = double(i) / double(num_samples);
        const auto data = spline_ceres.getPointData(pos_u);
        spline_ceres.fillParameterPointers(data, parameter_pointers.begin(), parameter_pointers.end());
        const ubs::UniformBSplineCeresEvaluator<Spline> pos_evaluator = spline_ceres.getEvaluator(data);
        const ubs::UniformBSplineCeresEvaluator<Spline> vel_evaluator = spline_ceres.getEvaluator(data, {1});

        // 对于每个采样点添加障碍物代价
        problem.AddResidualBlock(
            new ObstacleCostFunction(pos_evaluator, cost_map, obstacle_weight_),
            nullptr,
            parameter_pointers
        );

        // 对于每个采样点添加方向代价
        problem.AddResidualBlock(
            new DirectionCostFunction(pos_evaluator, vel_evaluator, direction_map, direction_weight_, step_norm_threshold_, step_norm_transition_),
            nullptr,
            parameter_pointers
        );

        // 对于每个采样点添加台阶模长惩罚（避免路径被台阶吸引）
        problem.AddResidualBlock(
            new StepFieldMagnitudeCostFunction(pos_evaluator, direction_map, step_weight_, step_norm_threshold_, step_norm_transition_),
            nullptr,
            parameter_pointers
        );

        // 对于每个采样点添加速度均匀代价
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<UniformSpeedCostFunction, 1, 2, 2, 2>(
                new UniformSpeedCostFunction(vel_evaluator, path_length * 0.5, uniform_speed_weight_)
            ),
            nullptr,
            parameter_pointers
        );
    }

    // 保持起终点位置
    const auto start_data = spline_ceres.getPointData(0.0);
    const auto goal_data = spline_ceres.getPointData(1.0);
    const ubs::UniformBSplineCeresEvaluator<Spline> start_evaluator = spline_ceres.getEvaluator(start_data);
    const ubs::UniformBSplineCeresEvaluator<Spline> goal_evaluator = spline_ceres.getEvaluator(goal_data);
    std::vector<double*> start_parameter_pointers(spline_ceres.getNumPointParameterPointers());
    std::vector<double*> goal_parameter_pointers(spline_ceres.getNumPointParameterPointers());
    spline_ceres.fillParameterPointers(start_data, start_parameter_pointers.begin(), start_parameter_pointers.end());
    spline_ceres.fillParameterPointers(goal_data, goal_parameter_pointers.begin(), goal_parameter_pointers.end());
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<StartEndPositionCostFunction, 2, 2, 2, 2>(
            new StartEndPositionCostFunction(start_evaluator, start_grid, start_end_weight_)
        ),
        nullptr, start_parameter_pointers
    );
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<StartEndPositionCostFunction, 2, 2, 2, 2>(
            new StartEndPositionCostFunction(goal_evaluator, goal_grid, start_end_weight_)
        ), 
        nullptr, goal_parameter_pointers
    );

    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_QR;
    options.use_nonmonotonic_steps = true;
    options.max_num_iterations = max_iterations_;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable()) return std::unexpected(summary.BriefReport());

    std::vector<Eigen::Vector2d> sample_points;
    for (size_t i = 0; i < num_samples; i++) {
        sample_points.push_back(spline.evaluate(double(i) / double(num_samples)));
    }

    return std::tuple{spline.getControlPoints(), sample_points};
}

double BSplineOptimizer::estimate_path_length(const std::vector<Eigen::Vector2d>& path) const {
    const size_t len = path.size();
    double length = 0;
    for (size_t i = 1; i < len; i++) {
        length += (path[i] - path[i - 1]).norm();
    }
    return length;
}
}