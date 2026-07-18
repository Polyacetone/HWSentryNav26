#pragma once

#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/minco_trajectory.hpp>
#include <nav_executor/path_planner/banded_system.hpp>

namespace nav_executor {

// 五次 MINCO 的系数消元与梯度回传。普通节点保持速度至四阶导连续；换向尖点强制两侧速度为零，
// 仅保持加速度和三阶导连续。矩阵带宽固定，使用带状部分主元 LU 避免随段数立方增长。
class MincoMinJerk {
public:
    static constexpr int DIM = 2;   // x, y（平坦输出）
    static constexpr int NCOEF = 6;

    struct BoundaryPVA {
        Eigen::Matrix<double, DIM, 1> pos = Eigen::Matrix<double, DIM, 1>::Zero();
        Eigen::Matrix<double, DIM, 1> vel = Eigen::Matrix<double, DIM, 1>::Zero();
        Eigen::Matrix<double, DIM, 1> acc = Eigen::Matrix<double, DIM, 1>::Zero();
    };

    void reset(int segment_count);

    // cusp_waypoint 为空时，所有内部节点按普通连续节点处理。
    void generate(
        const std::vector<double>& times,
        const BoundaryPVA& head,
        const BoundaryPVA& tail,
        const Eigen::Matrix<double, DIM, Eigen::Dynamic>& waypoints,
        const std::vector<char>& cusp_waypoint = {}
    );

    [[nodiscard]] int segment_count() const { return segment_count_; }

    // 导出为可求值的 MincoTrajectory。gears：每段换向符号 ±1。
    [[nodiscard]] MincoTrajectory to_trajectory(const std::vector<double>& gears) const;

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
    std::vector<char> cusp_;        // 每内部节点是否尖点（size N-1）
    BandedSystem banded_;           // M 的带状 LU（部分主元）
    Eigen::MatrixXd b_;             // 6N×2（右端项 / 求解后为系数）
    Eigen::MatrixXd coeffs_;        // 6N×2
};

} // namespace nav_executor
