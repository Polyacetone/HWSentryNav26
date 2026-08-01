#pragma once

#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/trajectory/minco_trajectory.hpp>
#include <nav_executor/path_planner/numerics/banded_system.hpp>

namespace nav_executor {

// 几何边界：切向只定义朝向，曲率和位置均与时标无关。
struct GeometricBoundary {
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    Eigen::Vector2d tangent = Eigen::Vector2d::UnitX();
    double curvature = 0.0;
};

// 五次 MINCO 的系数消元与梯度回传。内部节点保持速度至四阶导连续。
// 矩阵带宽固定，使用带状部分主元 LU 避免随段数立方增长。
class MincoMinJerk {
public:
    static constexpr int DIM = 2;   // x, y（平坦输出）
    static constexpr int NCOEF = 6;

    void reset(int segment_count);

    [[nodiscard]] bool generate(
        const std::vector<double>& times,
        const GeometricBoundary& head,
        const GeometricBoundary& tail,
        const Eigen::Matrix<double, DIM, Eigen::Dynamic>& waypoints,
        GeometryLimits geometry_limits
    );

    [[nodiscard]] int segment_count() const { return segment_count_; }

    [[nodiscard]] MincoTrajectory to_trajectory() const;

    [[nodiscard]] const Eigen::MatrixXd& coefficients() const { return coeffs_; }

    // 通过伴随方程把系数梯度回传到内部路点和时长。
    void propagate_gradient(
        const Eigen::MatrixXd& grad_c,
        const Eigen::VectorXd& grad_t_explicit,
        Eigen::Matrix<double, DIM, Eigen::Dynamic>& grad_q,
        Eigen::VectorXd& grad_t
    ) const;

private:
    // β^{(order)}(t) 的 6 维基向量。
    static Eigen::Matrix<double, NCOEF, 1> basis(double t, int order);

    // M 的带宽（与段数无关）。装配顺序：见 generate 中行布局。
    static constexpr int LOWER_BW = 8;
    static constexpr int UPPER_BW = 2;

    int segment_count_ = 0;
    std::vector<double> times_;
    GeometricBoundary head_;
    GeometricBoundary tail_;
    GeometryLimits geometry_limits_;
    double head_scale_ = 0.0;
    double tail_scale_ = 0.0;
    Eigen::Vector2d head_scale_gradient_ = Eigen::Vector2d::Zero();
    Eigen::Vector2d tail_scale_gradient_ = Eigen::Vector2d::Zero();
    bool factorized_ = false;
    BandedSystem banded_;           // M 的带状 LU（部分主元）
    Eigen::MatrixXd b_;             // 6N×2（右端项 / 求解后为系数）
    Eigen::MatrixXd coeffs_;        // 6N×2
};

} // namespace nav_executor
