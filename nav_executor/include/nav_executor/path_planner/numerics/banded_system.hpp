#pragma once

#include <vector>

#include <Eigen/Core>

namespace nav_executor {

// 用带状部分主元 LU 求解 MINCO 矩阵。部分主元不可省略：右段起点方程的对角元可能为零。
class BandedSystem {
public:
    void create(int n, int lower_bw, int upper_bw);

    void reset();

    double& ref(const int i, const int j) { return storage_(super_ + i - j, j); }
    [[nodiscard]] double at(const int i, const int j) const { return storage_(super_ + i - j, j); }

    // 返回 false 表示矩阵在相对主元判据下奇异或近奇异。
    [[nodiscard]] bool factorize_lu();

    void solve(Eigen::Ref<Eigen::MatrixXd> b) const;

    // 伴随方程复用同一分解；主元置换必须在最后逆序应用。
    void solve_transpose(Eigen::Ref<Eigen::MatrixXd> b) const;

    [[nodiscard]] int size() const { return n_; }

private:
    int n_ = 0;
    int kl_ = 0;
    int ku_ = 0;
    int super_ = 0;                 // kl + ku
    Eigen::MatrixXd storage_;       // (2·kl + ku + 1) × n
    std::vector<int> pivots_;       // 每列的主元行索引（部分主元）
};

} // namespace nav_executor
