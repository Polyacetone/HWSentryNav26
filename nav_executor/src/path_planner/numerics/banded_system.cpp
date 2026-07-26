#include <nav_executor/path_planner/numerics/banded_system.hpp>

#include <algorithm>
#include <cmath>

namespace nav_executor {

void BandedSystem::create(const int n, const int lower_bw, const int upper_bw) {
    n_ = std::max(n, 0);
    kl_ = std::max(lower_bw, 0);
    ku_ = std::max(upper_bw, 0);
    super_ = kl_ + ku_;
    // LAPACK 带格式：行数 2·kl + ku + 1（上部 kl 行为主元填充预留）。
    storage_.setZero(2 * kl_ + ku_ + 1, std::max(n_, 1));
    pivots_.assign(static_cast<size_t>(n_), 0);
}

void BandedSystem::reset() {
    storage_.setZero();
}

void BandedSystem::factorize_lu() {
    for (int j = 0; j < n_; ++j) {
        const int km = std::min(kl_, n_ - 1 - j);       // 列 j 的次对角元个数
        const int ju = std::min(n_ - 1, j + kl_ + ku_); // U 填充后受影响的末列

        int pivot_row = j;
        double pivot_mag = std::abs(at(j, j));
        for (int i = j + 1; i <= j + km; ++i) {
            const double mag = std::abs(at(i, j));
            if (mag > pivot_mag) {
                pivot_mag = mag;
                pivot_row = i;
            }
        }
        pivots_[static_cast<size_t>(j)] = pivot_row;

        if (pivot_mag == 0.0) continue;

        // 行交换：跨列 [j, ju] 交换 j 与 pivot_row。
        if (pivot_row != j) {
            for (int jj = j; jj <= ju; ++jj) {
                std::swap(ref(j, jj), ref(pivot_row, jj));
            }
        }

        // 归一化次对角元（L 的列 j）。
        const double diag = at(j, j);
        for (int i = j + 1; i <= j + km; ++i) {
            ref(i, j) /= diag;
        }

        for (int jj = j + 1; jj <= ju; ++jj) {
            const double ujj = at(j, jj);
            if (ujj == 0.0) continue;
            for (int i = j + 1; i <= j + km; ++i) {
                ref(i, jj) -= at(i, j) * ujj;
            }
        }
    }
}

void BandedSystem::solve(Eigen::Ref<Eigen::MatrixXd> b) const {
    for (int j = 0; j < n_; ++j) {
        const int jp = pivots_[static_cast<size_t>(j)];
        if (jp != j) b.row(j).swap(b.row(jp));
        const int i_end = std::min(n_ - 1, j + kl_);
        for (int i = j + 1; i <= i_end; ++i) {
            b.row(i) -= at(i, j) * b.row(j);
        }
    }

    for (int j = n_ - 1; j >= 0; --j) {
        b.row(j) /= at(j, j);
        const int i_begin = std::max(0, j - kl_ - ku_);
        for (int i = i_begin; i < j; ++i) {
            b.row(i) -= at(i, j) * b.row(j);
        }
    }
}

void BandedSystem::solve_transpose(Eigen::Ref<Eigen::MatrixXd> b) const {
    // Aᵀ = Uᵀ Lᵀ P。先前代 Uᵀ（U 上带宽 kl+ku），再回代 Lᵀ（L 下带宽 kl，单位对角），
    // 最后按逆序应用主元置换（P 的转置）。
    const int u_bw = kl_ + ku_;
    for (int j = 0; j < n_; ++j) {
        const int i_begin = std::max(0, j - u_bw);
        for (int i = i_begin; i < j; ++i) {
            b.row(j) -= at(i, j) * b.row(i);
        }
        b.row(j) /= at(j, j);
    }

    // Lᵀ 求解与主元置换须在同一逆序循环内交错：第 j 步的交换会影响第 j−1 步读取的行。
    for (int j = n_ - 1; j >= 0; --j) {
        const int i_end = std::min(n_ - 1, j + kl_);
        for (int i = j + 1; i <= i_end; ++i) {
            b.row(j) -= at(i, j) * b.row(i);
        }
        const int jp = pivots_[static_cast<size_t>(j)];
        if (jp != j) b.row(j).swap(b.row(jp));
    }
}

} // namespace nav_executor
