#pragma once

#include <vector>

#include <Eigen/Core>

namespace nav_executor {

// 轨迹曲线的求值结果。发布前的数值验收排除非有限值和检测到的内部 cusp。
struct TrajSample {
    Eigen::Vector2d p = Eigen::Vector2d::Zero();
    Eigen::Vector2d dp_dtau = Eigen::Vector2d::Zero();
    Eigen::Vector2d ddp_dtau = Eigen::Vector2d::Zero();

    double theta = 0.0;      // atan2(dp_dtau)：有向切线航向，也是车身航向
    double ds_dtau = 0.0;    // |dp_dtau|，弧长对 τ 的变化率
    double kappa = 0.0;      // 真实几何曲率 det(p',p'')/|p'|³，全系统唯一定义
    double kappa_rate = 0.0; // dκ/ds，与 κ 同源；速度剖面用它约束角加速度
};

// 逐段五次多项式前向轨迹，内部节点保持速度至四阶导连续。构造时的物理时间仅是
// MINCO 动力学塑形见证；执行层不读取它，统一以累计弧长和 PathSpeedProfile 确定时标。
// tau 与该见证时间线性对应，只用于参数化几何求值。
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
    [[nodiscard]] double total_time() const { return total_time_; } // 动力学塑形见证时长
    [[nodiscard]] double total_arc_length() const { return total_arc_length_; }
    [[nodiscard]] double length() const { return total_arc_length_; }
    // 第 boundary_index 个段边界对应的 tau，合法范围 [0, segment_count()]。
    [[nodiscard]] double segment_boundary_tau(int boundary_index) const;

    [[nodiscard]] double segment_duration(int segment_index) const;

    [[nodiscard]] double arc_length_at_tau(double tau) const;
    [[nodiscard]] double tau_at_arc_length(double arc_length) const;
    [[nodiscard]] double segment_boundary_arc_length(int boundary_index) const;

    [[nodiscard]] Eigen::Vector2d position(double tau) const { return eval(tau).p; }
    [[nodiscard]] Eigen::Vector2d tangent(double tau) const { return eval(tau).dp_dtau; }

    // 按归一化 MINCO 参数求值；超出 [0, 1] 时沿端点切线线性外推。
    [[nodiscard]] TrajSample eval(double tau) const;
    // 按 MINCO 动力学见证时间 t∈[0,total_time] 求值，不是执行期参考时间。
    [[nodiscard]] TrajSample eval_time(double t) const;
    // 按累计弧长 s∈[0,total_arc_length] 求值；弧长是跟随层唯一的路径进度坐标。
    [[nodiscard]] TrajSample eval_arc_length(double arc_length) const;

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

    std::vector<double> durations_;         // 每段动力学塑形见证时长 T_i
    std::vector<double> cumulative_times_;  // 前缀和，size = segment_count+1，首元素 0
    std::vector<CoefBlock> coeffs_;         // 每段系数（2D 平坦输出）
    double total_time_ = 0.0;
    double total_arc_length_ = 0.0;
    std::vector<double> arc_taus_;          // 每段均匀细分的 τ 节点，显式包含所有段边界
    std::vector<double> arc_samples_;       // arc_taus_ 对应的累计弧长
};

} // namespace nav_executor
