#include <nav_executor/path_planner/lbfgs_minimizer.hpp>

#include <algorithm>
#include <cmath>
#include <deque>

namespace nav_executor {

namespace {

// 一次目标评估的结果快照（归一化后的代价与梯度）。
struct Probe {
    Eigen::VectorXd x;
    Eigen::VectorXd grad;
    double cost = 0.0;
    double dphi = 0.0; // 方向导数 gradᵀ·direction
    bool finite = false;
};

} // anonymous namespace

LbfgsMinimizer::Result LbfgsMinimizer::minimize(const CostFunction& cost_fn, Eigen::VectorXd& x) const {
    const int n = static_cast<int>(x.size());
    Result result;

    Eigen::VectorXd grad(n);
    const double cost_raw = cost_fn(x, grad);
    if (!std::isfinite(cost_raw) || !grad.allFinite()) {
        result.status = Status::LINE_SEARCH_FAILED;
        result.cost = cost_raw;
        result.grad_inf_norm = grad.lpNorm<Eigen::Infinity>();
        return result;
    }

    // ── 目标归一化 ──
    // 按初始梯度无穷范数把整个目标缩放到 O(1)，使 grad_tolerance 成为与量纲无关的相对判据。
    // 常数缩放不改变极小点，也不改变 L-BFGS 的搜索方向（H·g 对目标常数缩放不变），
    // 仅令收敛判据与首步启发在不同问题规模间可迁移。
    const double obj_scale = 1.0 / std::max(grad.lpNorm<Eigen::Infinity>(), 1e-12);
    double cost = cost_raw * obj_scale;
    grad *= obj_scale;

    // 归一化目标评估：返回缩放后代价，填充缩放后梯度。
    const auto eval = [&](const Eigen::VectorXd& xx, Eigen::VectorXd& gg) {
        const double c = cost_fn(xx, gg);
        gg *= obj_scale;
        return c * obj_scale;
    };

    // (s, y) 历史与其曲率标量 rho = 1/(yᵀs)。
    std::deque<Eigen::VectorXd> s_hist;
    std::deque<Eigen::VectorXd> y_hist;
    std::deque<double> rho_hist;

    const double c1 = opt_.armijo_c;
    const double c2 = std::clamp(opt_.wolfe_c, c1 + 1e-6, 1.0 - 1e-6);

    // 对外报告用物理量（除以缩放还原），使收敛判据在归一化目标上、而日志与其它诊断
    // 保持同一量纲可比。
    const double inv_scale = 1.0 / obj_scale;

    for (int iter = 0; iter < opt_.max_iterations; ++iter) {
        const double grad_inf_norm_norm = grad.lpNorm<Eigen::Infinity>();
        result.iterations = iter;
        result.cost = cost * inv_scale;
        result.grad_inf_norm = grad_inf_norm_norm * inv_scale;

        if (grad_inf_norm_norm < opt_.grad_tolerance) {
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

        double dphi0 = grad.dot(direction);
        if (dphi0 >= 0.0) {
            // 方向非下降（数值退化）→ 重置为最速下降。
            direction = -grad;
            dphi0 = grad.dot(direction);
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
        }

        const Eigen::VectorXd x0 = x;
        const double phi0 = cost;
        Eigen::VectorXd grad_try(n);

        // ── 强 Wolfe 线搜索（bracketing + zoom，Nocedal & Wright Alg. 3.5/3.6）──
        // 沿给定 direction 搜索。满足强 Wolfe（充分下降 c1 + 曲率 |φ'|≤c2|φ'0|）则填 accepted；
        // 否则退回探测过的最优 Armijo 下降点（策略 B）。返回是否找到可接受步。
        // 曲率条件是 L-BFGS 生成有效 (s,y) 对、维持超线性收敛的关键。
        Probe accepted;
        const auto run_line_search = [&](const Eigen::VectorXd& dir, const double dphi_0, const double a_init) {
            const double dir_sq = std::max(dir.squaredNorm(), 1e-300);
            Probe best_armijo;
            double best_armijo_cost = phi0;
            bool ok = false;

            const auto probe_at = [&](const double a) {
                Probe p;
                p.x = x0 + a * dir;
                p.cost = eval(p.x, grad_try);
                p.grad = grad_try;
                p.finite = std::isfinite(p.cost) && p.grad.allFinite();
                p.dphi = p.finite ? p.grad.dot(dir) : 0.0;
                if (p.finite && p.cost <= phi0 + c1 * a * dphi_0 && p.cost < best_armijo_cost) {
                    best_armijo = p;
                    best_armijo_cost = p.cost;
                }
                return p;
            };

            const auto zoom = [&](Probe lo, Probe hi) {
                for (int j = 0; j < opt_.max_line_search; ++j) {
                    ++result.line_search_iterations;
                    const double a_lo = (lo.x - x0).dot(dir) / dir_sq;
                    const double a_hi = (hi.x - x0).dot(dir) / dir_sq;
                    if (std::abs(a_hi - a_lo) < opt_.step_tolerance) break;
                    const Probe mid = probe_at(0.5 * (a_lo + a_hi));
                    if (!mid.finite || mid.cost > phi0 + c1 * (0.5 * (a_lo + a_hi)) * dphi_0
                        || mid.cost >= lo.cost) {
                        hi = mid;
                        continue;
                    }
                    if (std::abs(mid.dphi) <= -c2 * dphi_0) {
                        accepted = mid;
                        ok = true;
                        return;
                    }
                    if (mid.dphi * (a_hi - a_lo) >= 0.0) hi = lo;
                    lo = mid;
                }
                if (lo.finite && lo.cost < phi0) { accepted = lo; ok = true; }
            };

            constexpr double A_MAX = 1e8;
            constexpr double EXPAND = 2.5;
            Probe prev;
            prev.x = x0;
            prev.cost = phi0;
            prev.dphi = dphi_0;
            prev.finite = true;
            double a = a_init;
            for (int i = 0; i < opt_.max_line_search && !ok; ++i) {
                ++result.line_search_iterations;
                const Probe cur = probe_at(a);
                const bool armijo_violated = !cur.finite
                    || cur.cost > phi0 + c1 * a * dphi_0
                    || (i > 0 && cur.cost >= prev.cost);
                if (armijo_violated) { zoom(prev, cur); break; }
                if (std::abs(cur.dphi) <= -c2 * dphi_0) { accepted = cur; ok = true; break; }
                if (cur.dphi >= 0.0) { zoom(cur, prev); break; }
                prev = cur;
                a = std::min(a * EXPAND, A_MAX);
                if (a >= A_MAX) {
                    if (cur.finite && cur.cost < phi0) { accepted = cur; ok = true; }
                    break;
                }
            }
            // 策略 B：Wolfe 失败但存在满足 Armijo 的下降点 → 退回它。
            if (!ok && best_armijo.finite && best_armijo_cost < phi0) {
                accepted = best_armijo;
                ok = true;
            }
            return ok;
        };

        const double a_first = (iter == 0)
            ? std::min(1.0, 1.0 / std::max(grad.lpNorm<1>(), 1e-12))
            : 1.0;
        bool ls_ok = run_line_search(direction, dphi0, a_first);

        // 最速下降重启：quasi-Newton 方向被残余不连续（如台阶类型边界的速度窗跳变）顶在墙上、
        // 任意步长都上升时，清空记忆库、沿 −grad 再试一次——最速下降常能沿墙滑走。
        if (!ls_ok && direction != -grad) {
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            direction = -grad;
            dphi0 = grad.dot(direction);
            const double a_sd = std::min(1.0, 1.0 / std::max(grad.lpNorm<1>(), 1e-12));
            ls_ok = run_line_search(direction, dphi0, a_sd);
        }

        if (!ls_ok || !accepted.finite) {
            // 最速下降也无法下降 → 已在（不连续）局部极小，判线搜索失败（optimize 返回当前解）。
            result.status = Status::LINE_SEARCH_FAILED;
            result.cost = cost * inv_scale;
            result.grad_inf_norm = grad.lpNorm<Eigen::Infinity>() * inv_scale;
            return result;
        }

        // ── 更新历史 ──
        Eigen::VectorXd s_new = accepted.x - x0;
        Eigen::VectorXd y_new = accepted.grad - grad;
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

        x = accepted.x;
        grad = accepted.grad;
        cost = accepted.cost;
    }

    result.status = Status::MAX_ITERATIONS;
    result.cost = cost * inv_scale;
    result.iterations = opt_.max_iterations;
    result.grad_inf_norm = grad.lpNorm<Eigen::Infinity>() * inv_scale;
    return result;
}

} // namespace nav_executor
