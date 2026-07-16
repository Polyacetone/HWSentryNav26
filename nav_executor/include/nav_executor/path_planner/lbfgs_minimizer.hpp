#pragma once

#include <functional>

#include <Eigen/Core>

namespace nav_executor {

// ── 有限内存 BFGS 无约束最小化器 ──
//
// 两段循环递归近似逆 Hessian，配合 Armijo 回溯线搜索。历史对 (s, y) 中曲率
// sᵀy ≤ 0 时跳过更新以保持正定。
class LbfgsMinimizer {
public:
    struct Options {
        int max_iterations = 200;
        int history_size = 8;        // 保留的 (s, y) 对数
        double grad_tolerance = 1e-5; // ‖g‖_inf 收敛阈（作用于归一化目标）
        double step_tolerance = 1e-12; // 步长区间下界（zoom 收缩判定）
        double armijo_c = 1e-4;       // 充分下降系数 c1（strong Wolfe）
        double wolfe_c = 0.9;         // 曲率条件系数 c2（strong Wolfe，quasi-Newton 取 0.9）
        int max_line_search = 24;     // bracket / zoom 各自的最大评估次数
    };

    enum class Status {
        CONVERGED,        // 梯度达阈
        MAX_ITERATIONS,   // 迭代上限
        LINE_SEARCH_FAILED, // 线搜索无法找到下降步
    };

    struct Result {
        Status status = Status::MAX_ITERATIONS;
        double cost = 0.0;
        int iterations = 0;
        int line_search_iterations = 0;
        double grad_inf_norm = 0.0;
    };

    // 目标：填充 grad，返回代价。x 与 grad 同维。
    using CostFunction = std::function<double(const Eigen::VectorXd& x, Eigen::VectorXd& grad)>;

    explicit LbfgsMinimizer(Options options) : opt_(options) {}

    // 就地最小化，x 既是初值也是输出。
    Result minimize(const CostFunction& cost_fn, Eigen::VectorXd& x) const;

private:
    Options opt_;
};

} // namespace nav_executor
