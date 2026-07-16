#pragma once

#include <vector>

#include <Eigen/Core>

namespace nav_executor {

// ── 带状线性系统：部分主元 LU（LAPACK gbtrf/gbtrs 思路自适配）──
//
// 面向 MINCO 的 M(T) c = b：M 只有常数带宽（下带宽 kl、上带宽 ku 与段数无关），
// 用带状存储 + 部分主元 LU，把稠密 O((6N)³) 降到 O(N·bw²)。
//
// 存储采用 LAPACK 带格式：元素 A(i,j) 存于 storage_(super + i − j, j)，其中
// super = kl + ku（为部分主元产生的填充预留上带宽扩展）。共 2·kl + ku + 1 行。
//
// 决策记录：本类不做无主元消元（GCOPTER 版本依赖其特定方程排序保证主元非零）。
// nav_executor 的 MINCO 方程排序中「右段出发」行对角元为 0，必须部分主元，否则除零。
// 因此保留主元交换，数值行为与稠密 partialPivLu 一致。
class BandedSystem {
public:
    // 分配 n×n、下带宽 lower_bw、上带宽 upper_bw 的带状系统并清零。
    void create(int n, int lower_bw, int upper_bw);

    // 清零带内容（复用已分配存储），用于重复 generate。
    void reset();

    // 带内元素读写（装配期使用；调用方须保证 (i,j) 落在带内）。
    double& ref(const int i, const int j) { return storage_(super_ + i - j, j); }
    [[nodiscard]] double at(const int i, const int j) const { return storage_(super_ + i - j, j); }

    // 原地部分主元 LU 分解。
    void factorize_lu();

    // 就地求解 A X = B（B 为 n×cols，分解后调用；结果覆盖 B）。
    void solve(Eigen::Ref<Eigen::MatrixXd> b) const;

    // 就地求解 Aᵀ X = B，复用 A 的同一 LU 分解（伴随方程 Mᵀλ=g 用）。
    // 由 PA=LU 得 Aᵀ=UᵀLᵀP：先解 Uᵀ，再解 Lᵀ，末尾逆序应用主元置换。
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
