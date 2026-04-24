#pragma once

/// @file fddp_solver.hpp
/// @brief Feasibility-driven DDP solver with trust-region, filter line-search,
///        and control-limited (box-constrained) backward pass.

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

namespace fddp {

// ═══════════════════════════════════════════════════════════════
//  Compile-time problem dimensions
// ═══════════════════════════════════════════════════════════════

template<typename Problem>
struct Dims;

// ═══════════════════════════════════════════════════════════════
//  Type aliases
// ═══════════════════════════════════════════════════════════════

template<typename P>
using StateVec = Eigen::Matrix<double, Dims<P>::NX, 1>;

template<typename P>
using ControlVec = Eigen::Matrix<double, Dims<P>::NU, 1>;

template<typename P>
using FxMat = Eigen::Matrix<double, Dims<P>::NX, Dims<P>::NX>;

template<typename P>
using FuMat = Eigen::Matrix<double, Dims<P>::NX, Dims<P>::NU>;

// ═══════════════════════════════════════════════════════════════
//  Solver options
// ═══════════════════════════════════════════════════════════════

struct SolverOptions {
    int max_iters = 30;
    double tol_grad = 1e-6;
    double tol_cost = 1e-8;

    double mu_init = 1e-6;
    double mu_min = 1e-9;
    double mu_max = 1e6;
    double mu_factor = 10.0;

    // Line search (Armijo)
    double alpha_min = 1e-4;
    double armijo_c1 = 1e-4;

    // Feasibility-driven gap contraction
    double gap_threshold = 1e-3;

    // Trust-region: max feed-forward norm per step
    double trust_region_radius = 10.0;
    double tr_expand_factor = 1.5;
    double tr_shrink_factor = 0.5;
    double tr_min = 0.1;
    double tr_max = 100.0;

    // Deprecated: kept for config compatibility, ignored in Box-DDP.
    double al_penalty_init = 10.0;
    double al_penalty_max = 1e4;
    double al_penalty_factor = 5.0;
    int al_outer_iters = 3;
};

// ═══════════════════════════════════════════════════════════════
//  Solver result
// ═══════════════════════════════════════════════════════════════

struct SolverResult {
    double cost = 0.0;
    int iters = 0;
    bool converged = false;
};

// ═══════════════════════════════════════════════════════════════
//  Filter entry for (cost, feasibility) multi-criteria acceptance
// ═══════════════════════════════════════════════════════════════

struct FilterEntry {
    double cost;
    double constraint_violation;
};

// ═══════════════════════════════════════════════════════════════
//  FDDP Solver
// ═══════════════════════════════════════════════════════════════

template<typename Problem>
class Solver {
public:
    static constexpr int NX = Dims<Problem>::NX;
    static constexpr int NU = Dims<Problem>::NU;
    static constexpr int N = Dims<Problem>::N;

    using VecX = Eigen::Matrix<double, NX, 1>;
    using VecU = Eigen::Matrix<double, NU, 1>;
    using MatXX = Eigen::Matrix<double, NX, NX>;
    using MatXU = Eigen::Matrix<double, NX, NU>;
    using MatUX = Eigen::Matrix<double, NU, NX>;
    using MatUU = Eigen::Matrix<double, NU, NU>;

    // ─── Trajectory storage ───
    std::array<VecX, N + 1> xs;
    std::array<VecU, N> us;

    SolverResult solve(const Problem& prob, const SolverOptions& opts = {});

private:
    // ─── Backward pass data ───
    struct FeedbackGain {
        MatUX K;
        VecU k;
    };

    std::array<FeedbackGain, N> gains_;

    std::array<VecX, N + 1> Vx_;
    std::array<MatXX, N + 1> Vxx_;

    std::array<MatXX, N> fx_;
    std::array<MatXU, N> fu_;

    std::array<VecX, N + 1> fs_;

    // ─── Reused forward-pass buffers ───
    std::array<VecX, N + 1> xs_try_;
    std::array<VecU, N> us_try_;
    std::array<VecX, N + 1> xs_old_;
    std::array<VecU, N> us_old_;

    double dV1_ = 0.0;
    double dV2_ = 0.0;

    // ─── Filter ───
    std::vector<FilterEntry> filter_;

    // ─── Methods ───
    double rollout_cost(const Problem& prob) const;
    void compute_gaps(const Problem& prob);
    double gap_norm() const;
    bool backward_pass(const Problem& prob, double mu, double tr_radius, const VecU& u_lo, const VecU& u_hi);
    double forward_pass(const Problem& prob, double alpha, const VecU& u_lo, const VecU& u_hi);

    bool filter_accepts(double cost, double cv) const;
    void filter_add(double cost, double cv);

    static VecU clamp_u(const VecU& u, const VecU& lo, const VecU& hi) {
        return u.cwiseMax(lo).cwiseMin(hi);
    }

    static bool solve_spd(const MatUU& h, const VecU& b, VecU& x);
    static bool solve_spd(const MatUU& h, const MatUX& b, MatUX& x);
    static bool solve_box_qp(
        const MatUU& h,
        const VecU& g,
        const MatUX& gx,
        const VecU& du_lo,
        const VecU& du_hi,
        VecU& k,
        MatUX& K
    );
};

// ═══════════════════════════════════════════════════════════════
//  Implementation
// ═══════════════════════════════════════════════════════════════

template<typename P>
double Solver<P>::rollout_cost(const P& prob) const {
    double c = 0.0;
    for (int k = 0; k < N; ++k) {
        c += prob.running_cost(k, xs[k], us[k]);
    }
    c += prob.terminal_cost(xs[N]);
    return c;
}

template<typename P>
void Solver<P>::compute_gaps(const P& prob) {
    fs_[0].setZero();
    for (int k = 1; k <= N; ++k) {
        fs_[k] = prob.dynamics(k - 1, xs[k - 1], us[k - 1]) - xs[k];
    }
}

template<typename P>
double Solver<P>::gap_norm() const {
    double s = 0.0;
    for (int k = 0; k <= N; ++k) {
        s += fs_[k].squaredNorm();
    }
    return std::sqrt(s);
}

template<typename P>
bool Solver<P>::solve_spd(const MatUU& h, const VecU& b, VecU& x) {
    if constexpr (NU == 2) {
        const double a = h(0, 0);
        const double b01 = h(0, 1);
        const double b10 = h(1, 0);
        const double d = h(1, 1);
        const double det = a * d - b01 * b10;
        if (!(det > 1e-14)) {
            return false;
        }

        const double inv00 = d / det;
        const double inv01 = -b01 / det;
        const double inv10 = -b10 / det;
        const double inv11 = a / det;

        x(0) = inv00 * b(0) + inv01 * b(1);
        x(1) = inv10 * b(0) + inv11 * b(1);
        return std::isfinite(x(0)) && std::isfinite(x(1));
    } else {
        Eigen::LLT<MatUU> llt(h);
        if (llt.info() != Eigen::Success) {
            return false;
        }
        x = llt.solve(b);
        return llt.info() == Eigen::Success;
    }
}

template<typename P>
bool Solver<P>::solve_spd(const MatUU& h, const MatUX& b, MatUX& x) {
    if constexpr (NU == 2) {
        const double a = h(0, 0);
        const double b01 = h(0, 1);
        const double b10 = h(1, 0);
        const double d = h(1, 1);
        const double det = a * d - b01 * b10;
        if (!(det > 1e-14)) {
            return false;
        }

        const double inv00 = d / det;
        const double inv01 = -b01 / det;
        const double inv10 = -b10 / det;
        const double inv11 = a / det;

        x.row(0).noalias() = inv00 * b.row(0) + inv01 * b.row(1);
        x.row(1).noalias() = inv10 * b.row(0) + inv11 * b.row(1);
        return x.allFinite();
    } else {
        Eigen::LLT<MatUU> llt(h);
        if (llt.info() != Eigen::Success) {
            return false;
        }
        x = llt.solve(b);
        return llt.info() == Eigen::Success;
    }
}

template<typename P>
bool Solver<P>::solve_box_qp(
    const MatUU& h,
    const VecU& g,
    const MatUX& gx,
    const VecU& du_lo,
    const VecU& du_hi,
    VecU& k,
    MatUX& K
) {
    VecU k_unc;
    MatUX K_unc;
    if (!solve_spd(h, -g, k_unc) || !solve_spd(h, -gx, K_unc)) {
        return false;
    }

    if constexpr (NU == 2) {
        k = k_unc;
        K = K_unc;

        std::array<bool, 2> active {false, false};
        for (int i = 0; i < 2; ++i) {
            if (k(i) < du_lo(i)) {
                k(i) = du_lo(i);
                active[static_cast<size_t>(i)] = true;
            } else if (k(i) > du_hi(i)) {
                k(i) = du_hi(i);
                active[static_cast<size_t>(i)] = true;
            }
        }

        const int active_cnt = (active[0] ? 1 : 0) + (active[1] ? 1 : 0);
        if (active_cnt == 0) {
            return true;
        }
        if (active_cnt == 2) {
            K.setZero();
            return true;
        }

        const int a = active[0] ? 0 : 1;
        const int f = 1 - a;
        K.row(a).setZero();

        const double hff = h(f, f);
        if (!(hff > 1e-12)) {
            return false;
        }

        const double hfa = h(f, a);
        const double kf = -(g(f) + hfa * k(a)) / hff;
        if (kf <= du_lo(f) || kf >= du_hi(f)) {
            k(f) = std::clamp(kf, du_lo(f), du_hi(f));
            K.row(f).setZero();
            return true;
        }

        k(f) = kf;
        K.row(f).noalias() = -gx.row(f) / hff;
        return true;
    } else {
        k = clamp_u(k_unc, du_lo, du_hi);
        K = K_unc;
        for (int i = 0; i < NU; ++i) {
            if (k(i) <= du_lo(i) + 1e-12 || k(i) >= du_hi(i) - 1e-12) {
                K.row(i).setZero();
            }
        }
        return true;
    }
}

template<typename P>
bool Solver<P>::filter_accepts(double cost, double cv) const {
    // Accept if the point dominates at least one dimension vs all filter entries
    for (const auto& f: filter_) {
        if (cost >= f.cost && cv >= f.constraint_violation) {
            return false; // dominated by existing entry
        }
    }
    return true;
}

template<typename P>
void Solver<P>::filter_add(double cost, double cv) {
    // Remove entries dominated by the new point
    std::erase_if(filter_, [&](const FilterEntry& f) { return cost <= f.cost && cv <= f.constraint_violation; });
    filter_.push_back({cost, cv});
}

template<typename P>
bool Solver<P>::backward_pass(const P& prob, double mu, double tr_radius, const VecU& u_lo, const VecU& u_hi) {
    prob.terminal_cost_derivatives(xs[N], Vx_[N], Vxx_[N]);
    Vx_[N] -= Vxx_[N] * fs_[N];

    dV1_ = 0.0;
    dV2_ = 0.0;

    for (int k = N - 1; k >= 0; --k) {
        prob.dynamics_jacobians(k, xs[k], us[k], fx_[k], fu_[k]);

        VecX lx;
        VecU lu;
        MatXX lxx;
        MatUX lux;
        MatUU luu;
        prob.running_cost_derivatives(k, xs[k], us[k], lx, lu, lxx, lux, luu);

        const MatXX& Vxx_next = Vxx_[k + 1];
        const VecX& Vx_next = Vx_[k + 1];

        const MatXX FxTV = fx_[k].transpose() * Vxx_next;

        VecX Qx = lx + fx_[k].transpose() * Vx_next;
        MatXX Qxx = lxx + FxTV * fx_[k];

        VecU Qu = lu;
        for (int i = 0; i < NU; ++i) {
            double acc = 0.0;
            for (int r = 0; r < NX; ++r) {
                const double fri = fu_[k](r, i);
                if (fri != 0.0) {
                    acc += fri * Vx_next(r);
                }
            }
            Qu(i) += acc;
        }

        MatUX Qux = lux;
        for (int i = 0; i < NU; ++i) {
            for (int r = 0; r < NX; ++r) {
                const double fri = fu_[k](r, i);
                if (fri != 0.0) {
                    Qux.row(i).noalias() += fri * (Vxx_next.row(r) * fx_[k]);
                }
            }
        }

        MatUU Quu = luu;
        for (int i = 0; i < NU; ++i) {
            for (int j = 0; j < NU; ++j) {
                double acc = 0.0;
                for (int r = 0; r < NX; ++r) {
                    const double fri = fu_[k](r, i);
                    if (fri == 0.0) {
                        continue;
                    }
                    for (int c = 0; c < NX; ++c) {
                        const double fcj = fu_[k](c, j);
                        if (fcj != 0.0) {
                            acc += fri * Vxx_next(r, c) * fcj;
                        }
                    }
                }
                Quu(i, j) += acc;
            }
        }

        MatUU Quu_reg = (Quu + Quu.transpose()).eval() * 0.5;
        for (int i = 0; i < NU; ++i) {
            Quu_reg(i, i) += mu;
        }

        VecU k_ff;
        MatUX K_fb;
        const VecU du_lo = u_lo - us[k];
        const VecU du_hi = u_hi - us[k];
        if (!solve_box_qp(Quu_reg, Qu, Qux, du_lo, du_hi, k_ff, K_fb)) {
            return false;
        }

        const double k_norm = k_ff.norm();
        if (k_norm > tr_radius && k_norm > 1e-12) {
            k_ff *= tr_radius / k_norm;
        }

        gains_[k].k = k_ff;
        gains_[k].K = K_fb;

        dV1_ += k_ff.dot(Qu);
        dV2_ += 0.5 * k_ff.dot(Quu * k_ff);

        Vx_[k] = Qx + K_fb.transpose() * Quu * k_ff + K_fb.transpose() * Qu + Qux.transpose() * k_ff;
        Vxx_[k] = Qxx + K_fb.transpose() * Quu * K_fb + K_fb.transpose() * Qux + Qux.transpose() * K_fb;
        Vxx_[k] = (Vxx_[k] + Vxx_[k].transpose()).eval() * 0.5;

        Vx_[k] -= Vxx_[k] * fs_[k];
    }

    return true;
}

template<typename P>
double Solver<P>::forward_pass(const P& prob, double alpha, const VecU& u_lo, const VecU& u_hi) {
    xs_try_[0] = xs[0];

    for (int k = 0; k < N; ++k) {
        const VecX dx = xs_try_[k] - xs[k];
        us_try_[k] = clamp_u(us[k] + alpha * gains_[k].k + gains_[k].K * dx, u_lo, u_hi);
        xs_try_[k + 1] = prob.dynamics(k, xs_try_[k], us_try_[k]) - (1.0 - alpha) * fs_[k + 1];
    }

    double cost = 0.0;
    for (int k = 0; k < N; ++k) {
        cost += prob.running_cost(k, xs_try_[k], us_try_[k]);
    }
    cost += prob.terminal_cost(xs_try_[N]);

    xs = xs_try_;
    us = us_try_;
    return cost;
}

template<typename P>
SolverResult Solver<P>::solve(const P& prob, const SolverOptions& opts) {
    const VecU u_lo = prob.u_lower();
    const VecU u_hi = prob.u_upper();

    for (int k = 0; k < N; ++k) {
        us[k] = clamp_u(us[k], u_lo, u_hi);
    }

    filter_.clear();
    filter_.reserve(static_cast<size_t>(std::max(opts.max_iters, 8)) + 8U);

    double cost = rollout_cost(prob);
    double tr_radius = opts.trust_region_radius;
    double mu = opts.mu_init;

    SolverResult result;
    result.cost = cost;

    for (int iter = 0; iter < opts.max_iters; ++iter) {
        result.iters = iter + 1;

        compute_gaps(prob);
        const double gnorm = gap_norm();
        const bool use_fddp = (gnorm > opts.gap_threshold);

        bool bp_ok = false;
        for (int retry = 0; retry < 20; ++retry) {
            bp_ok = backward_pass(prob, mu, tr_radius, u_lo, u_hi);
            if (bp_ok) {
                break;
            }
            mu = std::min(mu * opts.mu_factor, opts.mu_max);
        }
        if (!bp_ok) {
            break;
        }

        double grad_max = 0.0;
        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < NU; ++i) {
                grad_max = std::max(grad_max, std::abs(gains_[k].k(i)));
            }
        }
        if (grad_max < opts.tol_grad && gnorm < opts.gap_threshold) {
            result.converged = true;
            break;
        }

        xs_old_ = xs;
        us_old_ = us;
        const double cost_old = cost;

        bool accepted = false;
        double alpha = 1.0;
        while (alpha >= opts.alpha_min) {
            xs = xs_old_;
            us = us_old_;

            const double cost_try = forward_pass(prob, alpha, u_lo, u_hi);

            compute_gaps(prob);
            const double cv_try = gap_norm();

            const double dV_expected = alpha * dV1_ + 0.5 * alpha * alpha * dV2_;
            const double expected_reduction = -dV_expected;
            const double cost_reduction = cost_old - cost_try;

            const bool armijo_ok =
                (expected_reduction > 0.0 && (cost_reduction / expected_reduction) >= opts.armijo_c1);
            const bool filter_ok = filter_accepts(cost_try, cv_try);

            if (armijo_ok || (use_fddp && cost_try < cost_old) || filter_ok) {
                cost = cost_try;
                accepted = true;
                filter_add(cost_try, cv_try);
                tr_radius = std::min(tr_radius * opts.tr_expand_factor, opts.tr_max);
                break;
            }

            alpha *= 0.5;
        }

        if (!accepted) {
            xs = xs_old_;
            us = us_old_;
            mu = std::min(mu * opts.mu_factor, opts.mu_max);
            tr_radius = std::max(tr_radius * opts.tr_shrink_factor, opts.tr_min);
            continue;
        }

        mu = std::max(mu / opts.mu_factor, opts.mu_min);
        result.cost = cost;

        const double denom = std::max(std::abs(cost_old), 1.0);
        if (std::abs(cost_old - cost) < opts.tol_cost * denom && gnorm < opts.gap_threshold) {
            result.converged = true;
            break;
        }
    }

    for (int k = 0; k < N; ++k) {
        us[k] = clamp_u(us[k], u_lo, u_hi);
    }

    result.cost = cost;
    return result;
}

} // namespace fddp