#include <rclcpp/logging.hpp>
#include <ceres/ceres.h>
#include <uniform_bspline/uniform_bspline.hpp>
#include <uniform_bspline_ceres/uniform_bspline_ceres.hpp>
#include <path_planner/bspline_optimizer.hpp>
#include <path_planner/nav_map.hpp>

namespace path_planner {
using Spline = ubs::UniformBSpline<double, 2, double, Eigen::Vector2d, std::vector<Eigen::Vector2d>>;
}

// 残差函数类的定义和实现
namespace path_planner {
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
        for (int i = 0; i < pos_evaluator_.ControlPointsSupport; i++) {
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
            for (int i = 0; i < pos_evaluator_.ControlPointsSupport; i++) {
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
        const double weight
    ): pos_evaluator_(pos_evaluator), vel_evaluator_(vel_evaluator), direction_map_(direction_map), weight_(weight) {
        mutable_parameter_block_sizes()->clear();
        for (int i = 0; i < pos_evaluator_.ControlPointsSupport; i++) {
            mutable_parameter_block_sizes()->push_back(2);
        }
        set_num_residuals(1);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Eigen::Vector2d pos, vel;
        pos_evaluator_.evaluate(parameters[0], parameters[1], parameters[2], pos.data());
        vel_evaluator_.evaluate(parameters[0], parameters[1], parameters[2], vel.data());

        Eigen::Vector2d dir = direction_map_.interpolate(pos);
        if (dir.norm() < 1e-3) {
            residuals[0] = 0.0;
            if (jacobians) {
                for (int i = 0; i < pos_evaluator_.ControlPointsSupport; i++) {
                    if (jacobians[i]) {
                        jacobians[i][0] = 0.0;
                        jacobians[i][1] = 0.0;
                    }
                }
            }
            return true;
        }
        dir.normalize();

        // 代价定义为速度与期望方向的叉积绝对值
        // error = abs(vel.x * dir.y - vel.y * dir.x)
        double cross = vel.x() * dir.y() - vel.y() * dir.x();
        residuals[0] = weight_ * std::abs(cross);

        if (jacobians) {
            for (int i = 0; i < vel_evaluator_.ControlPointsSupport; i++) {
                if (jacobians[i]) {
                    double basis_vel = vel_evaluator_.basisVals_[i];
                    double sign = cross >= 0 ? 1.0 : -1.0;
                    // ∂error/∂P_i = sign * (dir.y, -dir.x) * N_i,p(u)
                    Eigen::Vector2d jac = sign * basis_vel * Eigen::Vector2d(dir.y(), -dir.x());
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
}

// 路径优化器主类实现
namespace path_planner {
BSplineOptimizer::BSplineOptimizer(
    const double smoothness_weight,
    const double length_weight,
    const double obstacle_weight,
    const double direction_weight,
    const double start_end_weight,
    const double num_samples_per_length,
    const int max_iterations
):
    smoothness_weight_(smoothness_weight),
    length_weight_(length_weight),
    obstacle_weight_(obstacle_weight),
    direction_weight_(direction_weight),
    start_end_weight_(start_end_weight),
    num_samples_per_length_(num_samples_per_length),
    max_iterations_(max_iterations) {}

std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>> BSplineOptimizer::optimize(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const std::vector<Eigen::Vector2d>& init_path,
    const Eigen::Vector2d& start_grid,
    const Eigen::Vector2d& goal_grid
) const {
    if (init_path.size() <= 2) {
        RCLCPP_WARN(rclcpp::get_logger("bspline_optimizer"), "Path too short to optimize!");
        return {init_path, init_path};
    }
    Spline spline(pad_control_points(init_path));
    ubs::UniformBSplineCeres<Spline> spline_ceres(spline);
    auto& control_points = spline.getControlPointsContainer();
    using ContainerT = ubs::FixedSizeContainerTypeTrait<Eigen::Vector2d>;
    ceres::Problem problem;

    // 添加长度代价（最小化路径速度的平方积分）
    spline_ceres.addSmoothnessResiduals<1>(problem, length_weight_);

    // 添加光滑度代价（最小化路径加速度的平方积分）
    spline_ceres.addSmoothnessResiduals<2>(problem, smoothness_weight_);

    const int num_samples = num_samples_per_length_ * estimate_path_length(init_path);
    std::vector<double*> parameter_pointers(spline_ceres.getNumPointParameterPointers());
    for (int i = 0; i < num_samples; i++) {
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
            new DirectionCostFunction(pos_evaluator, vel_evaluator, direction_map, direction_weight_),
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
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.use_nonmonotonic_steps = true;
    options.max_num_iterations = max_iterations_;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::vector<Eigen::Vector2d> sample_points;
    for (int i = 0; i < num_samples; i++) {
        sample_points.push_back(spline.evaluate(double(i) / num_samples));
    }
    return {spline.getControlPoints(), sample_points};
}

std::vector<Eigen::Vector2d> BSplineOptimizer::pad_control_points(const std::vector<Eigen::Vector2d>& path) const {
    std::vector<Eigen::Vector2d> patched;
    patched.push_back(path[0]);
    patched.insert(patched.end(), path.begin(), path.end());
    patched.push_back(path[path.size() - 1]);
    return patched;
}

double BSplineOptimizer::estimate_path_length(const std::vector<Eigen::Vector2d>& path) const {
    const int len = path.size();
    double length = 0;
    for (int i = 1; i < len; i++) {
        length += (path[i] - path[i - 1]).norm();
    }
    return length;
}
}