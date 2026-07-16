#include <nav_executor/path_planner/minco_minjerk.hpp>

#include <algorithm>

namespace nav_executor {

namespace {
constexpr double MIN_SEG_T = 1e-6;
} // anonymous namespace

Eigen::Matrix<double, MincoMinJerk::NCOEF, 1> MincoMinJerk::basis(const double t, const int order) {
    // β(t) = [1, t, t², t³, t⁴, t⁵]，order 阶导数。
    Eigen::Matrix<double, NCOEF, 1> b = Eigen::Matrix<double, NCOEF, 1>::Zero();
    if (order == 0) {
        double tp = 1.0;
        for (int k = 0; k < NCOEF; ++k) { b(k) = tp; tp *= t; }
    } else if (order == 1) {
        double tp = 1.0;
        for (int k = 1; k < NCOEF; ++k) { b(k) = static_cast<double>(k) * tp; tp *= t; }
    } else if (order == 2) {
        double tp = 1.0;
        for (int k = 2; k < NCOEF; ++k) { b(k) = static_cast<double>(k * (k - 1)) * tp; tp *= t; }
    } else if (order == 3) {
        double tp = 1.0;
        for (int k = 3; k < NCOEF; ++k) { b(k) = static_cast<double>(k * (k - 1) * (k - 2)) * tp; tp *= t; }
    } else if (order == 4) {
        double tp = 1.0;
        for (int k = 4; k < NCOEF; ++k) { b(k) = static_cast<double>(k * (k - 1) * (k - 2) * (k - 3)) * tp; tp *= t; }
    } else if (order == 5) {
        b(5) = 120.0;
    }
    return b;
}

void MincoMinJerk::reset(const int segment_count) {
    const int new_count = std::max(segment_count, 1);
    const int rows = NCOEF * new_count;
    if (new_count != segment_count_) {
        banded_.create(rows, LOWER_BW, UPPER_BW);
        b_ = Eigen::MatrixXd::Zero(rows, DIM);
        coeffs_ = Eigen::MatrixXd::Zero(rows, DIM);
        times_.assign(static_cast<size_t>(new_count), 0.0);
    } else {
        banded_.reset();
        b_.setZero();
    }
    cusp_.assign(static_cast<size_t>(std::max(new_count - 1, 0)), 0);
    segment_count_ = new_count;
}

void MincoMinJerk::generate(
    const std::vector<double>& times,
    const BoundaryPVA& head,
    const BoundaryPVA& tail,
    const Eigen::Matrix<double, DIM, Eigen::Dynamic>& waypoints,
    const std::vector<char>& cusp_waypoint
) {
    const int n = static_cast<int>(times.size());
    reset(n);
    for (int i = 0; i < n; ++i) times_[static_cast<size_t>(i)] = std::max(times[i], MIN_SEG_T);
    for (int i = 0; i < n - 1 && i < static_cast<int>(cusp_waypoint.size()); ++i) {
        cusp_[static_cast<size_t>(i)] = cusp_waypoint[static_cast<size_t>(i)];
    }

    // 把 NCOEF 长基向量写入 M 第 row 行、列偏移 c_off 处的连续 NCOEF 列。
    const auto set_row = [this](
        const int row, const int c_off, const Eigen::Matrix<double, NCOEF, 1>& coef
    ) {
        for (int k = 0; k < NCOEF; ++k) banded_.ref(row, c_off + k) = coef(k);
    };

    // ── 首端 3 条：pos/vel/acc（t=0 处 β(0),β'(0),β''(0)）──
    set_row(0, 0, basis(0.0, 0));
    set_row(1, 0, basis(0.0, 1));
    set_row(2, 0, basis(0.0, 2));
    b_.row(0) = head.pos.transpose();
    b_.row(1) = head.vel.transpose();
    b_.row(2) = head.acc.transpose();

    int row = 3;
    // ── 每个内部节点 i=1..n-1：6 条（q 到达 / q 出发 / vel,acc,jerk,snap 连续）──
    for (int i = 0; i < n - 1; ++i) {
        const double ti = times_[static_cast<size_t>(i)];
        const int c_left = NCOEF * i;        // 段 i 系数列偏移
        const int c_right = NCOEF * (i + 1); // 段 i+1 系数列偏移

        const auto b0 = basis(ti, 0);
        const auto b1 = basis(ti, 1);
        const auto b2 = basis(ti, 2);
        const auto b3 = basis(ti, 3);
        const auto b4 = basis(ti, 4);
        const auto b0_0 = basis(0.0, 0);
        const auto b1_0 = basis(0.0, 1);
        const auto b2_0 = basis(0.0, 2);
        const auto b3_0 = basis(0.0, 3);
        const auto b4_0 = basis(0.0, 4);

        // 左段末端到达路点 q_i
        set_row(row, c_left, b0);
        b_.row(row) = waypoints.col(i).transpose();
        ++row;

        // 右段起点出发路点 q_i
        set_row(row, c_right, b0_0);
        b_.row(row) = waypoints.col(i).transpose();
        ++row;

        if (cusp_[static_cast<size_t>(i)]) {
            // 换向尖点：两侧 v=0（精确），acc/jerk 连续。b 对应行为 0（默认）。
            set_row(row, c_left, b1); ++row;   // 左段末端 vel = 0
            set_row(row, c_right, b1_0); ++row; // 右段起点 vel = 0
            set_row(row, c_left, b2);
            set_row(row, c_right, -b2_0);
            ++row;                              // acc 连续
            set_row(row, c_left, b3);
            set_row(row, c_right, -b3_0);
            ++row;                              // jerk 连续
        } else {
            // 普通内部节点：内部速度由优化自选，保持 vel/acc/jerk/snap 连续。
            set_row(row, c_left, b1);
            set_row(row, c_right, -b1_0);
            ++row;
            set_row(row, c_left, b2);
            set_row(row, c_right, -b2_0);
            ++row;
            set_row(row, c_left, b3);
            set_row(row, c_right, -b3_0);
            ++row;
            set_row(row, c_left, b4);
            set_row(row, c_right, -b4_0);
            ++row;
        }
    }

    // ── 尾端 3 条：pos/vel/acc（末段 t=T_{n-1}）──
    const double tn = times_[static_cast<size_t>(n - 1)];
    const int c_last = NCOEF * (n - 1);
    set_row(row, c_last, basis(tn, 0));
    b_.row(row) = tail.pos.transpose();
    ++row;
    set_row(row, c_last, basis(tn, 1));
    b_.row(row) = tail.vel.transpose();
    ++row;
    set_row(row, c_last, basis(tn, 2));
    b_.row(row) = tail.acc.transpose();
    ++row;

    banded_.factorize_lu();
    coeffs_ = b_;
    banded_.solve(coeffs_);
}

MincoTrajectory MincoMinJerk::to_trajectory(const std::vector<double>& gears) const {
    std::vector<MincoTrajectory::CoefBlock> blocks(static_cast<size_t>(segment_count_));
    for (int i = 0; i < segment_count_; ++i) {
        MincoTrajectory::CoefBlock blk;
        for (int k = 0; k < NCOEF; ++k) {
            for (int d = 0; d < DIM; ++d) {
                blk(k, d) = coeffs_(NCOEF * i + k, d);
            }
        }
        blocks[static_cast<size_t>(i)] = blk;
    }
    return MincoTrajectory(times_, std::move(blocks), gears);
}

void MincoMinJerk::propagate_gradient(
    const Eigen::MatrixXd& grad_c,
    const Eigen::VectorXd& grad_t_explicit,
    Eigen::Matrix<double, DIM, Eigen::Dynamic>& grad_q,
    Eigen::VectorXd& grad_t
) const {
    const int n = segment_count_;

    // 伴随：Mᵀ λ = ∂G/∂c（每维独立，共 2 列）。复用 generate 的 LU 分解做转置回代。
    Eigen::MatrixXd lambda = grad_c; // (6N)×2
    banded_.solve_transpose(lambda);

    // ── ∂G/∂q_k：b 对 q_k 线性，∂b/∂q_k 在「左段到达」「右段出发」两行为单位。
    //    普通节点与尖点节点前两行同为 q 到达 / q 出发。
    grad_q.setZero(DIM, std::max(n - 1, 0));
    for (int i = 0; i < n - 1; ++i) {
        const int node_row0 = 3 + i * NCOEF;
        for (int d = 0; d < DIM; ++d) {
            grad_q(d, i) = lambda(node_row0, d) + lambda(node_row0 + 1, d);
        }
    }

    // ── ∂G/∂T_i = grad_t_explicit_i − Σ_d λ_col_dᵀ (∂M/∂T_i) c_col_d
    //    ∂M/∂T_i 只影响「以 t=T_i 求值」的行（左段末端项），右段 t=0 行不依赖 T_i。
    grad_t = grad_t_explicit;
    for (int i = 0; i < n; ++i) {
        const double ti = times_[static_cast<size_t>(i)];
        const int c_off = NCOEF * i;

        auto accum_row = [&](int r, int order) {
            // ∂/∂T_i of [β(T_i, order)ᵀ c] = β(T_i, order+1)ᵀ c
            const Eigen::Matrix<double, NCOEF, 1> db = basis(ti, order + 1);
            for (int d = 0; d < DIM; ++d) {
                double dM_c = 0.0;
                for (int k = 0; k < NCOEF; ++k) dM_c += db(k) * coeffs_(c_off + k, d);
                grad_t(i) -= lambda(r, d) * dM_c;
            }
        };

        if (i < n - 1) {
            const int node_row0 = 3 + i * NCOEF;
            accum_row(node_row0, 0);         // 左段到达 q_i（β0）
            if (cusp_[static_cast<size_t>(i)]) {
                accum_row(node_row0 + 2, 1); // 左段 vel=0（β1）
                accum_row(node_row0 + 4, 2); // acc 连续左段项（β2）
                accum_row(node_row0 + 5, 3); // jerk 连续左段项（β3）
            } else {
                accum_row(node_row0 + 2, 1); // vel 连续左段项
                accum_row(node_row0 + 3, 2); // acc
                accum_row(node_row0 + 4, 3); // jerk
                accum_row(node_row0 + 5, 4); // snap
            }
        } else {
            const int tail_row0 = 3 + (n - 1) * NCOEF;
            accum_row(tail_row0, 0);     // 末端 pos
            accum_row(tail_row0 + 1, 1); // 末端 vel
            accum_row(tail_row0 + 2, 2); // 末端 acc
        }
    }
}

} // namespace nav_executor
