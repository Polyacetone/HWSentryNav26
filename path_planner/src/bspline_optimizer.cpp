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
        const ESDFMap& esdf_map,
        const double clearance_weight,
        const double collision_weight,
        const double safe_distance_m,
        const double smooth_eps_m
    ): pos_evaluator_(pos_evaluator),
       esdf_map_(esdf_map),
       clearance_weight_(clearance_weight),
       collision_weight_(collision_weight),
       safe_distance_m_(safe_distance_m),
       smooth_eps_m_(smooth_eps_m) {
        // 设置参数块，每个控制点是2维向量
        mutable_parameter_block_sizes()->clear();
        for (int i = 0; i < pos_evaluator_.ControlPointsSupport; i++) {
            mutable_parameter_block_sizes()->push_back(2);
        }
        // r0: clearance hinge, r1: hard collision
        set_num_residuals(2);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Eigen::Vector2d point;
        pos_evaluator_.evaluate(parameters[0], parameters[1], parameters[2], point.data());

        const double d = esdf_map_.interpolate(point);            // signed distance (m)
        const Eigen::Vector2d dd_dxy = esdf_map_.gradient(point); // ∂d/∂(x,y) in grid coords

        // smooth hinge (C1):
        // phi(x) = 0                    , x <= 0
        //        = x^2 / (2*eps)        , 0 < x < eps
        //        = x - eps/2            , x >= eps
        // phi'(x) = 0, x<=0; x/eps, 0<x<eps; 1, x>=eps
        const auto smooth_hinge = [](double x, double eps) -> double {
            if (x <= 0.0) return 0.0;
            if (x < eps) return 0.5 * x * x / eps;
            return x - 0.5 * eps;
        };
        const auto smooth_hinge_grad = [](double x, double eps) -> double {
            if (x <= 0.0) return 0.0;
            if (x < eps) return x / eps;
            return 1.0;
        };

        // 距离上限截断：仅在 d < safe_distance_m_ 的范围内产生排斥
        const double x_clearance = safe_distance_m_ - d;
        const double x_collision = -d;

        residuals[0] = clearance_weight_ * smooth_hinge(x_clearance, smooth_eps_m_);
        residuals[1] = collision_weight_ * smooth_hinge(x_collision, smooth_eps_m_);

        if (jacobians) {
            const double g_clear = smooth_hinge_grad(x_clearance, smooth_eps_m_);
            const double g_coll = smooth_hinge_grad(x_collision, smooth_eps_m_);

            for (int i = 0; i < pos_evaluator_.ControlPointsSupport; i++) {
                if (!jacobians[i]) continue;

                // point = sum_i N_i * P_i     => ∂point/∂P_i = N_i * I
                // d = d(point)                => ∂d/∂P_i = N_i * ∂d/∂point
                // r_clear = w * (d_safe - d)  => ∂r/∂P_i = -w * ∂d/∂P_i
                // r_coll  = w * (-d)          => ∂r/∂P_i = -w * ∂d/∂P_i
                const double basis = pos_evaluator_.basisVals_[i];
                const Eigen::Vector2d dd_dpi = basis * dd_dxy;

                // jacobians[i] layout: [dr0/dx, dr0/dy, dr1/dx, dr1/dy]
                // r0 = w0 * phi(safe - d) => dr0/dd = -w0 * phi'(x_clear)
                // r1 = w1 * phi(-d)       => dr1/dd = -w1 * phi'(x_coll)
                const Eigen::Vector2d j0 = (-clearance_weight_ * g_clear) * dd_dpi;
                const Eigen::Vector2d j1 = (-collision_weight_ * g_coll) * dd_dpi;
                jacobians[i][0] = j0.x();
                jacobians[i][1] = j0.y();
                jacobians[i][2] = j1.x();
                jacobians[i][3] = j1.y();
            }
        }
        return true;
    }

private:
    const ubs::UniformBSplineCeresEvaluator<Spline> pos_evaluator_;
    const ESDFMap& esdf_map_;
    const double clearance_weight_;
    const double collision_weight_;
    const double safe_distance_m_;
    const double smooth_eps_m_;
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
namespace path_planner {
BSplineOptimizer::BSplineOptimizer(
    const double smoothness_weight,
    const double length_weight,
    const double obstacle_weight,
    const double esdf_collision_weight,
    const double esdf_safe_distance_m,
    const double esdf_smooth_eps_m,
    const double direction_weight,
    const double start_end_weight,
    const double num_samples_per_length,
    const int max_iterations
):
    smoothness_weight_(smoothness_weight),
    uniform_speed_weight_(length_weight),
    obstacle_weight_(obstacle_weight),
    esdf_collision_weight_(esdf_collision_weight),
    esdf_safe_distance_m_(esdf_safe_distance_m),
    esdf_smooth_eps_m_(esdf_smooth_eps_m),
    direction_weight_(direction_weight),
    start_end_weight_(start_end_weight),
    num_samples_per_length_(num_samples_per_length),
    max_iterations_(max_iterations) {}

std::expected<std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>>, std::string> BSplineOptimizer::optimize(
    const ESDFMap& esdf_map,
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
    auto& control_points = spline.getControlPointsContainer();
    using ContainerT = ubs::FixedSizeContainerTypeTrait<Eigen::Vector2d>;
    ceres::Problem problem;

    // 添加光滑度代价
    spline_ceres.addSmoothnessResiduals<2>(problem, smoothness_weight_);

    const double path_length = estimate_path_length(init_path);
    const int num_samples = num_samples_per_length_ * path_length;
    std::vector<double*> parameter_pointers(spline_ceres.getNumPointParameterPointers());
    for (int i = 0; i < num_samples; i++) {
        const double pos_u = double(i) / double(num_samples);
        const auto data = spline_ceres.getPointData(pos_u);
        spline_ceres.fillParameterPointers(data, parameter_pointers.begin(), parameter_pointers.end());
        const ubs::UniformBSplineCeresEvaluator<Spline> pos_evaluator = spline_ceres.getEvaluator(data);
        const ubs::UniformBSplineCeresEvaluator<Spline> vel_evaluator = spline_ceres.getEvaluator(data, {1});

        // 对于每个采样点添加障碍物（ESDF）代价
        problem.AddResidualBlock(
            new ObstacleCostFunction(pos_evaluator, esdf_map, obstacle_weight_, esdf_collision_weight_, esdf_safe_distance_m_, esdf_smooth_eps_m_),
            nullptr,
            parameter_pointers
        );

        // 对于每个采样点添加方向代价
        problem.AddResidualBlock(
            new DirectionCostFunction(pos_evaluator, vel_evaluator, direction_map, direction_weight_),
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
    for (int i = 0; i < num_samples; i++) {
        sample_points.push_back(spline.evaluate(double(i) / num_samples));
    }

    return std::tuple{spline.getControlPoints(), sample_points};
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