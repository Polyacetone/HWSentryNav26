#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace nav_executor {

// q 的可分离凸代价与相邻差分约束组成的一维链式问题。最后一个节点固定为 0。
struct ChainProblem {
    struct SoftWindow {
        size_t node_index = 0;
        double lower = 0.0; // 与 q 相同量纲
        double upper = 0.0;
        double weight = 0.0; // 已包含该节点的积分权重
    };

    std::vector<double> linear_reward; // 目标中的 -c_i q_i
    std::vector<double> node_upper;    // 0 <= q_i <= U_i
    std::vector<double> step_limit;    // |q_{i+1} - q_i| <= d_i，长度为 N-1
    std::vector<SoftWindow> soft_windows;
    double initial_value = 0.0;        // 固定 q_0；q_{N-1} 固定为 0
};

class PiecewiseQuadraticChainSolver {
public:
    enum class Status {
        OPTIMAL,
        INFEASIBLE,
        INVALID_PROBLEM,
    };

    struct Result {
        Status status = Status::INVALID_PROBLEM;
        std::vector<double> value;
        double objective = 0.0;
        int max_breakpoints = 0;
        double solve_ms = 0.0;
        std::string error;
    };

    [[nodiscard]] static Result solve(const ChainProblem& problem);
};

} // namespace nav_executor
