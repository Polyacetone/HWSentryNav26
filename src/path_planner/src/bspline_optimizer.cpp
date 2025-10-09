#include <rclcpp/logging.hpp>
#include <ceres/ceres.h>
#include <uniform_bspline/uniform_bspline.hpp>
#include <uniform_bspline_ceres/uniform_bspline_ceres.hpp>
#include <path_planner/bspline_optimizer.hpp>
#include <path_planner/costmap_2d.hpp>

namespace path_planner {
using Spline = ubs::UniformBSpline<double, 2, double, Eigen::Vector2d, std::vector<Eigen::Vector2d>>;
}

// 残差函数类的定义和实现
namespace path_planner {
// 障碍物代价涉及离散的代价地图，需要手动求导
class ObstacleCostFunction : public ceres::CostFunction {
public:
    explicit ObstacleCostFunction(
        const ubs::UniformBSplineCeresEvaluator<Spline>& evaluator,
        const Costmap2D& cost_map,
        const double weight = 1.0
    ): evaluator_(evaluator), cost_map_(cost_map), weight_(weight) {
        // 设置参数块，每个控制点是2维向量
        mutable_parameter_block_sizes()->clear();
        for (int i = 0; i < evaluator_.ControlPointsSupport; ++i) {
            mutable_parameter_block_sizes()->push_back(2);
        }
        set_num_residuals(1);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Eigen::Vector2d point;
        evaluator_.evaluate(parameters[0], parameters[1], parameters[2], point.data());
        const double cost = cost_map_.interpolate(point);
        residuals[0] = weight_ * cost;
        if (jacobians != nullptr) {
            const Eigen::Vector2d cost_gradient = cost_map_.gradient(point);
            for (int i = 0; i < evaluator_.ControlPointsSupport; i++) {
                if (jacobians[i] != nullptr) {
                    // ∂L/∂P_i = N_i,p(u) * ∂L/∂V
                    Eigen::Vector2d jac = weight_ * evaluator_.basisVals_[i] * cost_gradient;
                    jacobians[i][0] = jac.x();  // 对控制点x分量的导数
                    jacobians[i][1] = jac.y();  // 对控制点y分量的导数
                }
            }
        }
        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> evaluator_;
    const Costmap2D& cost_map_;
    const double weight_;
};

// 长度代价使用自动求导
class LengthCostFunction {
public:
    explicit LengthCostFunction(const double weight): weight_(weight) {}
    template<typename T> bool operator()(const T* const p0, const T* const p1, T* residual) const {
        const T dx = p1[0] - p0[0];
        const T dy = p1[1] - p0[1];
        const T distance = ceres::sqrt(dx * dx + dy * dy + T(1e-6));
        residual[0] = weight_ * distance;
        return true;
    }

private:
    const double weight_;
};

// 起始点速度方向代价使用自动求导
class StartVelocityCostFunction {
public:
    explicit StartVelocityCostFunction(const Eigen::Vector2d& start_velocity, const double weight):
        start_velocity_(start_velocity), weight_(weight) {}
    template<typename T> bool operator()(const T* const p0, const T* const p1, T* residual) const {
        const T px = p1[0] - p0[0];
        const T py = p1[1] - p0[1];
        const T pnorm = ceres::sqrt(px * px + py * py + T(1e-6));
        const T dot = px * T(start_velocity_.x()) + py * T(start_velocity_.y());
        residual[0] = T(weight_) * (T(start_velocity_.norm()) - dot / pnorm);
        return true;
    }

private:
    const Eigen::Vector2d start_velocity_;
    const double weight_;
};
}

// 路径优化器主类实现
namespace path_planner {
BSplineOptimizer::BSplineOptimizer(
    const double num_samples_per_length,
    const double obstable_weight,
    const double length_weight,
    const double smooth_weight
):
    num_samples_per_length_(num_samples_per_length),
    obstable_weight_(obstable_weight),
    length_weight_(length_weight),
    smooth_weight_(smooth_weight) {}

std::vector<Eigen::Vector2d> BSplineOptimizer::optimize(
    const Costmap2D& costmap,
    const std::vector<Eigen::Vector2d>& path
) const {
    if (path.size() <= 2) {
        RCLCPP_WARN(rclcpp::get_logger("bspline_optimizer"), "Path too short to optimize!");
        return path;
    }
    Spline spline(pad_control_points(path));
    ubs::UniformBSplineCeres<Spline> spline_ceres(spline);

    ceres::Problem problem;
    // 添加光滑度代价
    spline_ceres.addSmoothnessResidualsGrid<2>(problem, smooth_weight_);
    std::vector<double*> parameter_pointers(spline_ceres.getNumPointParameterPointers());
    const int num_samples = num_samples_per_length_ * estimate_path_length(path);
    // 对于每个采样点添加障碍物代价
    for (int i = 0; i < num_samples; i++) {
        const double pos_u = double(i) / double(num_samples);
        const auto data = spline_ceres.getPointData(pos_u);
        spline_ceres.fillParameterPointers(data, parameter_pointers.begin(), parameter_pointers.end());
        const ubs::UniformBSplineCeresEvaluator<Spline> evaluator = spline_ceres.getEvaluator(data);
        problem.AddResidualBlock(
            new ObstacleCostFunction(evaluator, costmap, obstable_weight_),
            nullptr,
            parameter_pointers
        );
    }

    // 对于每个控制点添加长度代价
    auto& control_points = spline.getControlPointsContainer();
    using ContainerT = ubs::FixedSizeContainerTypeTrait<Eigen::Vector2d>;
    parameter_pointers.resize(2);
    for (int i = 1; i < control_points.getNumElements(); i++) {
        parameter_pointers[0] = ContainerT::data(control_points.at(i - 1));
        parameter_pointers[1] = ContainerT::data(control_points.at(i));
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<LengthCostFunction, 1, 2, 2>(
                new LengthCostFunction(length_weight_)
            ),
            nullptr,
            parameter_pointers
        );
    }

    // 保持起终点不变
    problem.SetParameterBlockConstant(ContainerT::data(control_points.at(0)));
    problem.SetParameterBlockConstant(ContainerT::data(control_points.at(1)));
    problem.SetParameterBlockConstant(ContainerT::data(control_points.at(control_points.getNumElements() - 2)));
    problem.SetParameterBlockConstant(ContainerT::data(control_points.at(control_points.getNumElements() - 1)));

    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.use_nonmonotonic_steps = true;
    options.max_solver_time_in_seconds = 0.05;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::vector<Eigen::Vector2d> optimized;
    for (int i = 0; i < num_samples; i++) {
        optimized.push_back(spline.evaluate(double(i) / num_samples));
    }
    return optimized;
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