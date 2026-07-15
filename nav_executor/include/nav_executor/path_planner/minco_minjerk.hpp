#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <nav_executor/common/minco_trajectory.hpp>

namespace nav_executor {

// ── MINCO min-jerk (s=3, 五次) 核心：M(T) 消元 + 梯度回传 ──
//
// 参考 GCOPTER 的 MINCO 思路自适配（非照搬）。对每一维（x, y, θ 独立）：
//   段 i 多项式 p_i(t)=Σ_{k=0..5} c_{i,k} t^k, t∈[0,T_i]，共 N 段。
//   系数 c∈R^{6N} 由 6N 个线性条件唯一确定：
//     - 首端 (pos,vel,acc) 固定；
//     - 每个内部节点 k：左段到达路点 q_k、右段从 q_k 出发、vel/acc/jerk/snap 连续（6 条）；
//     - 尾端 (pos,vel,acc) 固定。
//   写成 M(T) c = b(q)。M 只依赖 T（三维共用），b 每维不同。
//
// 决策记录：问题规模小（N≤~6 → 6N≤36），用稠密 LU 而非 GCOPTER 的带状 PLU——
// O(N³) 但 N 极小，换取实现简洁与低出错面（落地版「smallest complete correct thing」）。
//
// 梯度回传（GCOPTER Thm 2 的自适配）：代价 G(c, T)，c=M⁻¹b。
//   伴随：解 Mᵀλ = ∂G/∂c
//   ∂G/∂q_k = λ 在 q_k 所在行的分量（b 对 q 线性，∂b/∂q_k 是选择子）
//   ∂G/∂T_i = ∂G/∂T_i|_explicit − λᵀ (∂M/∂T_i) c
class MincoMinJerk {
public:
    static constexpr int DIM = 3;
    static constexpr int NCOEF = 6;

    // 每维一列的边界全状态（pos/vel/acc）。
    struct BoundaryPVA {
        Eigen::Matrix<double, DIM, 1> pos = Eigen::Matrix<double, DIM, 1>::Zero();
        Eigen::Matrix<double, DIM, 1> vel = Eigen::Matrix<double, DIM, 1>::Zero();
        Eigen::Matrix<double, DIM, 1> acc = Eigen::Matrix<double, DIM, 1>::Zero();
    };

    void reset(int segment_count);

    // 由 T（N 段）、首/尾边界 PVA、内部路点 Q（DIM×(N-1)）求解系数并 LU 分解 M。
    void generate(
        const std::vector<double>& times,
        const BoundaryPVA& head,
        const BoundaryPVA& tail,
        const Eigen::Matrix<double, DIM, Eigen::Dynamic>& waypoints
    );

    [[nodiscard]] int segment_count() const { return segment_count_; }

    // 导出为可求值的 MincoTrajectory。
    [[nodiscard]] MincoTrajectory to_trajectory() const;

    // 系数访问：coeffs_ 是 (6N)×3，段 i 的第 k 阶系数在行 6i+k。
    [[nodiscard]] const Eigen::MatrixXd& coefficients() const { return coeffs_; }

    // 梯度回传。输入 ∂G/∂c（(6N)×3）与 ∂G/∂T 的显式部分（N），
    // 输出 ∂G/∂q（DIM×(N-1)）与总 ∂G/∂T（N）。
    void propagate_gradient(
        const Eigen::MatrixXd& grad_c,
        const Eigen::VectorXd& grad_t_explicit,
        Eigen::Matrix<double, DIM, Eigen::Dynamic>& grad_q,
        Eigen::VectorXd& grad_t,
        Eigen::Vector3d* grad_tail_pos = nullptr
    ) const;

private:
    // β^{(order)}(t) 的 6 维基向量。
    static Eigen::Matrix<double, NCOEF, 1> basis(double t, int order);

    int segment_count_ = 0;
    std::vector<double> times_;
    Eigen::MatrixXd m_;              // 6N×6N
    Eigen::PartialPivLU<Eigen::MatrixXd> lu_;
    Eigen::MatrixXd b_;             // 6N×3
    Eigen::MatrixXd coeffs_;        // 6N×3
};

} // namespace nav_executor
