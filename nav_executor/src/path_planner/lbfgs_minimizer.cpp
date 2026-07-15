#include <nav_executor/path_planner/lbfgs_minimizer.hpp>

#include <algorithm>
#include <cmath>
#include <deque>

namespace nav_executor {

LbfgsMinimizer::Result LbfgsMinimizer::minimize(const CostFunction& cost_fn, Eigen::VectorXd& x) const {
    const int n = static_cast<int>(x.size());
    Result result;

    Eigen::VectorXd grad(n);
    double cost = cost_fn(x, grad);
    if (!std::isfinite(cost) || !grad.allFinite()) {
        result.status = Status::LINE_SEARCH_FAILED;
        result.cost = cost;
        result.grad_inf_norm = grad.lpNorm<Eigen::Infinity>();
        return result;
    }

    // (s, y) 历史与其曲率标量 rho = 1/(yᵀs)。
    std::deque<Eigen::VectorXd> s_hist;
    std::deque<Eigen::VectorXd> y_hist;
    std::deque<double> rho_hist;

    for (int iter = 0; iter < opt_.max_iterations; ++iter) {
        result.iterations = iter;
        result.cost = cost;
        result.grad_inf_norm = grad.lpNorm<Eigen::Infinity>();

        if (result.grad_inf_norm < opt_.grad_tolerance) {
            result.status = Status::CONVERGED;
            return result;
        }

        // ── 两段循环递归：direction = -H·grad ──
        Eigen::VectorXd q = grad;
        const int m = static_cast<int>(s_hist.size());
        std::vector<double> alpha(static_cast<size_t>(m));
        for (int i = m - 1; i >= 0; --i) {
            const auto idx = static_cast<size_t>(i);
            alpha[idx] = rho_hist[idx] * s_hist[idx].dot(q);
            q -= alpha[idx] * y_hist[idx];
        }
        // 初始逆 Hessian 缩放 gamma = sᵀy / yᵀy（最近一对）。
        double gamma = 1.0;
        if (m > 0) {
            const double yy = y_hist.back().squaredNorm();
            gamma = yy > 1e-18 ? y_hist.back().dot(s_hist.back()) / yy : 1.0;
        }
        q *= gamma;
        for (int i = 0; i < m; ++i) {
            const auto idx = static_cast<size_t>(i);
            const double beta = rho_hist[idx] * y_hist[idx].dot(q);
            q += (alpha[idx] - beta) * s_hist[idx];
        }
        Eigen::VectorXd direction = -q;

        double dir_deriv = grad.dot(direction);
        if (dir_deriv >= 0.0) {
            // 方向非下降（数值退化）→ 重置为最速下降。
            direction = -grad;
            dir_deriv = grad.dot(direction);
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
        }

        // ── Armijo 回溯线搜索 ──
        // MINCO 目标含采样罚、角度 wrap 和 AL 项，并非处处二阶光滑。这里只要求可靠下降；
        // 曲率信息由成功步后的 s/y 对筛选负责。
        double step = (iter == 0) ? std::min(1.0, opt_.init_step / std::max(grad.lpNorm<Eigen::Infinity>(), 1e-12)) : 1.0;
        const Eigen::VectorXd x0 = x;
        const double cost0 = cost;

        Eigen::VectorXd x_try(n);
        Eigen::VectorXd grad_try(n);
        double cost_try = cost0;
        bool ls_ok = false;
        for (int ls = 0; ls < opt_.max_line_search; ++ls) {
            ++result.line_search_iterations;
            x_try = x0 + step * direction;
            cost_try = cost_fn(x_try, grad_try);

            const bool finite = std::isfinite(cost_try) && grad_try.allFinite();
            if (finite && cost_try <= cost0 + opt_.armijo_c * step * dir_deriv) {
                ls_ok = true;
                break;
            }
            step *= 0.5;
            if (step < opt_.step_tolerance) break;
        }

        if (!ls_ok) {
            result.status = Status::LINE_SEARCH_FAILED;
            result.grad_inf_norm = grad.lpNorm<Eigen::Infinity>();
            return result;
        }

        // ── 更新历史 ──
        Eigen::VectorXd s_new = x_try - x0;
        Eigen::VectorXd y_new = grad_try - grad;
        const double sy = s_new.dot(y_new);
        if (sy > 1e-10) { // 曲率正 → 更新，否则跳过保持正定
            if (static_cast<int>(s_hist.size()) >= opt_.history_size) {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            rho_hist.push_back(1.0 / sy);
            s_hist.push_back(std::move(s_new));
            y_hist.push_back(std::move(y_new));
        }

        x = x_try;
        grad = grad_try;
        cost = cost_try;
    }

    result.status = Status::MAX_ITERATIONS;
    result.cost = cost;
    result.iterations = opt_.max_iterations;
    result.grad_inf_norm = grad.lpNorm<Eigen::Infinity>();
    return result;
}

} // namespace nav_executor
