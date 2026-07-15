#pragma once

#include <vector>

#include <Eigen/Core>

namespace nav_executor {

// ── MINCO 全状态轨迹在某参数 τ / 时刻 t 的求值结果 ──
//
// 关键语义分离（相对旧 SplineEval）：
//   - theta / dtheta_dtau 是**独立**朝向状态，取自轨迹自身，不再是位置切线的 atan2。
//     支持倒车（朝前但位置后退）与原地旋转（位置不动、朝向变）。
//   - phi / sin_phi / cos_phi / ds_dtau / kappa 描述**位置曲线**的切线帧，供横向误差
//     （cross-track）投影使用。横向误差是几何量，其参考帧由 p(τ) 的切线决定，与朝向无关；
//     倒车时切线反向只翻转 ey 符号，|ey| 与惩罚不变。
//   - 原地旋转处 |dp/dτ|→0，切线帧退化（ds_dtau→0），由调用方按阈值保护；此时朝向进度
//     由 dtheta_dtau 接管（见 §3.4 广义进度）。
struct TrajSample {
    // 位置曲线及其对 τ 的导数
    Eigen::Vector2d p = Eigen::Vector2d::Zero();
    Eigen::Vector2d dp_dtau = Eigen::Vector2d::Zero();
    Eigen::Vector2d ddp_dtau = Eigen::Vector2d::Zero();

    // 独立朝向
    double theta = 0.0;
    double dtheta_dtau = 0.0;

    // 位置曲线切线帧（横向误差投影用）
    double phi = 0.0;       // atan2(dp_dtau.y, dp_dtau.x)；|dp_dtau|→0 时保持上一有效值意义不定
    double sin_phi = 0.0;
    double cos_phi = 0.0;
    double ds_dtau = 0.0;   // |dp_dtau|，弧长对 τ 的变化率（原地旋转处→0）
    double kappa = 0.0;     // 位置曲线曲率
};

struct GovernedClockParams {
    double error_scale = 0.5;
    double heading_weight = 1.0;
};

// ── 参数化全状态轨迹载体 ──
//
// 替换 SplinePath 在参考链中的角色。内部是逐段五次多项式（MINCO min-jerk，C² 连续），
// 每段一个时长 T_i；(x, y, θ) 各一维独立多项式。
//
// 归一参数 τ∈[0,1] 与绝对时间线性对应：t = τ · total_time。跟随层把 τ 当作参数化路径
// 参数使用（非强制时间，见落地版 Q1b），故对外主接口按 τ 求值。
//
// 本类只负责“表示 + 求值”，不掺入优化。系数由 MINCO 优化器（[3]）产出，或由
// from_boundary_states 工厂（暖启动 / [4a] 手工桩）构造。
class MincoTrajectory {
public:
    static constexpr int DIM = 3;               // x, y, theta
    static constexpr int DEGREE = 5;            // 五次（min-jerk）
    static constexpr int NCOEF = DEGREE + 1;    // 6

    // 单段系数：行 k = t^k 的系数，列 = (x, y, theta)。段内 s(t)=Σ_k coef.row(k)·t^k，t∈[0,T]。
    using CoefBlock = Eigen::Matrix<double, NCOEF, DIM>;

    // 单段边界全状态：位置/速度/加速度（(x,y,θ) 各一）。速度分量即 d/dt。
    struct BoundaryState {
        Eigen::Matrix<double, DIM, 1> pos = Eigen::Matrix<double, DIM, 1>::Zero();
        Eigen::Matrix<double, DIM, 1> vel = Eigen::Matrix<double, DIM, 1>::Zero();
        Eigen::Matrix<double, DIM, 1> acc = Eigen::Matrix<double, DIM, 1>::Zero();
    };

    MincoTrajectory() = default;
    MincoTrajectory(std::vector<double> durations, std::vector<CoefBlock> coeffs);

    // 由 N+1 个边界全状态 + N 段时长构造逐段五次 Hermite 轨迹。相邻段共享边界状态即保证
    // C² 连续。用于暖启动与 [4a] 手工桩（含折返 / 原地旋转）。
    [[nodiscard]] static MincoTrajectory from_boundary_states(
        const std::vector<BoundaryState>& states,
        const std::vector<double>& durations
    );

    [[nodiscard]] bool empty() const { return durations_.empty(); }
    [[nodiscard]] int segment_count() const { return static_cast<int>(durations_.size()); }
    [[nodiscard]] double total_time() const { return total_time_; }
    // 采样估计的位置曲线总弧长（构造期一次算好；折返段按几何路程累加）。
    [[nodiscard]] double total_arc_length() const { return total_arc_length_; }
    [[nodiscard]] double length() const { return total_arc_length_; }
    // 第 boundary_index 个段边界对应的 tau，合法范围 [0, segment_count()]。
    [[nodiscard]] double segment_boundary_tau(int boundary_index) const;

    // ── 弧长 ↔ τ 映射（供 route_tracker / route_monitor / step 逻辑复用）──
    // 注意：折返轨迹上弧长沿 τ 单调递增（几何路程累加），故映射良定义。
    [[nodiscard]] double arc_length_at_tau(double tau) const;
    [[nodiscard]] double tau_at_arc_length(double arc_length) const;
    // |弧长(tau1) − 弧长(tau0)|。
    [[nodiscard]] double arc_length_between(double tau0, double tau1) const;

    // 便捷访问（τ 参数）。
    [[nodiscard]] Eigen::Vector2d position(double tau) const { return eval(tau).p; }
    [[nodiscard]] Eigen::Vector2d tangent(double tau) const { return eval(tau).dp_dtau; }

    // 主接口：按归一参数 τ∈[0,1] 求值（超界线性外推，与旧 SplinePath 外推语义一致）。
    [[nodiscard]] TrajSample eval(double tau) const;
    // 按绝对时间 t∈[0,total_time] 求值。
    [[nodiscard]] TrajSample eval_time(double t) const;

    // 窗口化投影（替代 SplinePath::project 在 route_tracker 中的角色）：在
    // τ∈[tau_lo, tau_hi] 内找位置曲线上距 pos 最近的点。窗口消歧折返段——位置曲线
    // 会自交，但弧长（∝τ）单调，局部窗口保证选到正确的那一支。返回最近点 τ。
    [[nodiscard]] double project(const Eigen::Vector2d& pos, double tau_lo, double tau_hi) const;

    // 带符号纵向速度 v = ẋcosθ + ẏsinθ 与角速度 ω = θ̇（对时间的真实速度，非对 τ）。
    [[nodiscard]] double longitudinal_velocity(const TrajSample& s) const;
    [[nodiscard]] double angular_velocity(const TrajSample& s) const;

    [[nodiscard]] bool operator==(const MincoTrajectory& other) const;
    [[nodiscard]] bool operator!=(const MincoTrajectory& other) const { return !(*this == other); }

private:
    // τ→(段索引, 段内局部时间)。
    struct Locator {
        int segment = 0;
        double local_t = 0.0;
    };
    [[nodiscard]] Locator locate_time(double t) const;
    [[nodiscard]] TrajSample sample_at(int segment, double local_t) const;

    void compute_arc_length();

    std::vector<double> durations_;         // 每段时长 T_i
    std::vector<double> cumulative_times_;  // 前缀和，size = segment_count+1，首元素 0
    std::vector<CoefBlock> coeffs_;         // 每段系数
    double total_time_ = 0.0;
    double total_arc_length_ = 0.0;
    std::vector<double> arc_samples_;       // 均匀 τ 采样处的累计弧长，size = ARC_SAMPLES+1
};

// 跟踪误差只允许降低名义轨迹时钟速率。该函数是执行期持久化 tau 与 MPC 预测 tau 的共同语义源。
[[nodiscard]] double governed_clock_factor(
    const TrajSample& reference,
    const Eigen::Vector3d& chassis_pose_map,
    const GovernedClockParams& params
);

} // namespace nav_executor
