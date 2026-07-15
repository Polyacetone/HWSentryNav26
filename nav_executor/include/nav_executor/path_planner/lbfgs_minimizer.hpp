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
        double grad_tolerance = 1e-5; // ‖g‖_inf 收敛阈
        double step_tolerance = 1e-9; // 步长下界（线搜索失败判定）
        double armijo_c = 1e-4;       // 充分下降系数
        double init_step = 1.0;       // 首次迭代试探步长
        int max_line_search = 20;
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
