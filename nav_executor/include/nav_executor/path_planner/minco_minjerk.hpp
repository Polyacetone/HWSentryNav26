#pragma once

#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/minco_trajectory.hpp>
#include <nav_executor/path_planner/banded_system.hpp>

namespace nav_executor {

// ── MINCO min-jerk (s=3, 五次) 核心：M(T) 消元 + 梯度回传（2D 平坦）──
//
// 参考 GCOPTER 的 MINCO 思路自适配（非照搬）。只优化 2D 平坦输出（x, y 独立）：
//   段 i 多项式 p_i(t)=Σ_{k=0..5} c_{i,k} t^k, t∈[0,T_i]，共 N 段。
//   系数 c∈R^{6N} 由 6N 个线性条件唯一确定（每维一列，M 共用）：
//     - 首端 (pos,vel,acc) 固定；
//     - 普通内部节点 k：左段到达路点 q_k、右段从 q_k 出发、vel/acc/jerk/snap 连续（6 条）；
//     - **换向尖点节点** k：左段到达 q_k、右段从 q_k 出发、左段 vel=0、右段 vel=0、
//       acc/jerk 连续（6 条）。v=0 精确满足使换向物理自洽；acc 连续保证平坦 θ 跨尖点连续。
//     - 尾端 (pos,vel,acc) 固定。
//   写成 M(T) c = b(q)。M 只依赖 T + 尖点掩码，b 每维不同。
//
// M(T) 常数带宽 kl=8 / ku=2（尖点节点用更窄的 vel=0 基，带宽不增）。带状部分主元 LU
// 求解，O(N·bw²)。
//
// 梯度回传（GCOPTER Thm 2 的自适配）：代价 G(c, T)，c=M⁻¹b。
//   伴随：解 Mᵀλ = ∂G/∂c（复用 M 的同一 LU 分解做转置回代）
//   ∂G/∂q_k = λ 在 q_k 到达/出发两行的分量（尖点节点同样只这两行依赖 q）
//   ∂G/∂T_i = ∂G/∂T_i|_explicit − λᵀ (∂M/∂T_i) c
class MincoMinJerk {
public:
    static constexpr int DIM = 2;   // x, y（平坦输出）
    static constexpr int NCOEF = 6;

    // 每维一列的边界全状态（pos/vel/acc）。
    struct BoundaryPVA {
        Eigen::Matrix<double, DIM, 1> pos = Eigen::Matrix<double, DIM, 1>::Zero();
        Eigen::Matrix<double, DIM, 1> vel = Eigen::Matrix<double, DIM, 1>::Zero();
        Eigen::Matrix<double, DIM, 1> acc = Eigen::Matrix<double, DIM, 1>::Zero();
    };

    void reset(int segment_count);

    // 由 T（N 段）、首/尾边界 PVA、内部路点 Q（DIM×(N-1)）求解系数并 LU 分解 M。
    // cusp_waypoint：长度 N-1 的掩码，true 表示该内部节点是换向尖点（两侧 v=0）。
    // 空则全部按普通连续节点处理。
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

    // 系数访问：coeffs_ 是 (6N)×2，段 i 的第 k 阶系数在行 6i+k。
    [[nodiscard]] const Eigen::MatrixXd& coefficients() const { return coeffs_; }

    // 梯度回传。输入 ∂G/∂c（(6N)×2）与 ∂G/∂T 的显式部分（N），
    // 输出 ∂G/∂q（DIM×(N-1)）与总 ∂G/∂T（N）。
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
