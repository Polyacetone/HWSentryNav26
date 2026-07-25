#pragma once

#include <vector>

#include <Eigen/Core>

namespace nav_executor {

// 微分平坦轨迹的求值结果。车身朝向由有向运动切线导出；端点速度为零时，
// eval 使用段内侧速度恢复朝向。切线帧始终按位置曲线定义。
struct TrajSample {
    Eigen::Vector2d p = Eigen::Vector2d::Zero();
    Eigen::Vector2d dp_dtau = Eigen::Vector2d::Zero();
    Eigen::Vector2d ddp_dtau = Eigen::Vector2d::Zero();

    double theta = 0.0;
    double dtheta_dtau = 0.0;

    // 位置曲线切线帧，用于横向误差投影。
    double phi = 0.0;       // atan2(dp_dtau.y, dp_dtau.x)；|dp_dtau|→0 时保持上一有效值意义不定
    double sin_phi = 0.0;
    double cos_phi = 0.0;
    double ds_dtau = 0.0;   // |dp_dtau|，弧长对 τ 的变化率
    double kappa = 0.0;     // 位置曲线曲率
};

// 逐段五次多项式前向轨迹，内部节点保持速度至四阶导连续。
// tau 与绝对时间线性对应，仅用于 MINCO 轨迹求值；跟随层统一使用累计弧长。
class MincoTrajectory {
public:
    static constexpr int DIM = 2;               // x, y（平坦输出）
    static constexpr int DEGREE = 5;            // 五次（min-jerk）
    static constexpr int NCOEF = DEGREE + 1;    // 6

    // 单段系数：行 k = t^k 的系数，列 = (x, y)。段内 p(t)=Σ_k coef.row(k)·t^k，t∈[0,T]。
    using CoefBlock = Eigen::Matrix<double, NCOEF, DIM>;
    using ControlPointBlock = Eigen::Matrix<double, NCOEF, DIM>;

    MincoTrajectory() = default;
    MincoTrajectory(
        std::vector<double> durations,
        std::vector<CoefBlock> coeffs
    );

    [[nodiscard]] bool empty() const { return durations_.empty(); }
    [[nodiscard]] int segment_count() const { return static_cast<int>(durations_.size()); }
    [[nodiscard]] double total_time() const { return total_time_; }
    [[nodiscard]] double total_arc_length() const { return total_arc_length_; }
    [[nodiscard]] double length() const { return total_arc_length_; }
    // 第 boundary_index 个段边界对应的 tau，合法范围 [0, segment_count()]。
    [[nodiscard]] double segment_boundary_tau(int boundary_index) const;

    [[nodiscard]] double arc_length_at_tau(double tau) const;
    [[nodiscard]] double tau_at_arc_length(double arc_length) const;
    [[nodiscard]] double segment_boundary_arc_length(int boundary_index) const;
    [[nodiscard]] int segment_index_at_arc_length(double arc_length) const;
    // |弧长(tau1) − 弧长(tau0)|。
    [[nodiscard]] double arc_length_between(double tau0, double tau1) const;

    [[nodiscard]] Eigen::Vector2d position(double tau) const { return eval(tau).p; }
    [[nodiscard]] Eigen::Vector2d tangent(double tau) const { return eval(tau).dp_dtau; }

    // 按归一化 MINCO 参数求值；超出 [0, 1] 时线性外推。
    [[nodiscard]] TrajSample eval(double tau) const;
    // 按绝对时间 t∈[0,total_time] 求值。
    [[nodiscard]] TrajSample eval_time(double t) const;
    // 按累计弧长 s∈[0,total_arc_length] 求值；弧长是跟随层唯一的路径进度坐标。
    [[nodiscard]] TrajSample eval_arc_length(double arc_length) const;

    [[nodiscard]] double longitudinal_velocity(const TrajSample& s) const;
    [[nodiscard]] double angular_velocity(const TrajSample& s) const;
    [[nodiscard]] double nominal_path_speed(const TrajSample& s) const;
    [[nodiscard]] double heading_rate_per_arc_length(const TrajSample& s) const;
    [[nodiscard]] ControlPointBlock segment_bezier_control_points(int segment_index) const;

    [[nodiscard]] bool operator==(const MincoTrajectory& other) const;
    [[nodiscard]] bool operator!=(const MincoTrajectory& other) const { return !(*this == other); }

private:
    struct Locator {
        int segment = 0;
        double local_t = 0.0;
    };
    [[nodiscard]] Locator locate_time(double t) const;
    [[nodiscard]] TrajSample sample_at(int segment, double local_t) const;

    void compute_arc_length();

    std::vector<double> durations_;         // 每段时长 T_i
    std::vector<double> cumulative_times_;  // 前缀和，size = segment_count+1，首元素 0
    std::vector<CoefBlock> coeffs_;         // 每段系数（2D 平坦输出）
    double total_time_ = 0.0;
    double total_arc_length_ = 0.0;
    std::vector<double> arc_taus_;          // 每段均匀细分的 τ 节点，显式包含所有段边界
    std::vector<double> arc_samples_;       // arc_taus_ 对应的累计弧长
};

} // namespace nav_executor
