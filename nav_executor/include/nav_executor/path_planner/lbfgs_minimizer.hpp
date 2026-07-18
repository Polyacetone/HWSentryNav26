#pragma once

#include <functional>

#include <Eigen/Core>

namespace nav_executor {

// 有限内存 BFGS。强 Wolfe 搜索失败时回退到已探测到的最佳 Armijo 下降点；
// 曲率非正的历史对会被丢弃，以维持逆 Hessian 近似正定。
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

    using CostFunction = std::function<double(const Eigen::VectorXd& x, Eigen::VectorXd& grad)>;

    explicit LbfgsMinimizer(Options options) : opt_(options) {}

    Result minimize(const CostFunction& cost_fn, Eigen::VectorXd& x) const;

private:
    Options opt_;
};

} // namespace nav_executor
