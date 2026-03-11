#pragma once

/// @file fddp_solver.hpp
/// @brief Enhanced FDDP solver with Trust-Region, Filter line-search,
///        and Augmented Lagrangian box constraints.
///
/// Reference: Mastalli et al., "Feasibility-Driven DDP" (RAL 2020).

#include <array>
#include <cmath>
#include <algorithm>
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

    // Augmented Lagrangian for box constraints
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

    double dV1_ = 0.0;
    double dV2_ = 0.0;

    // ─── Augmented Lagrangian multipliers & penalty ───
    std::array<VecU, N> lambda_lo_;
    std::array<VecU, N> lambda_hi_;
    double al_penalty_ = 10.0;

    // ─── Filter ───
    std::vector<FilterEntry> filter_;

    // ─── Methods ───
    double rollout_cost(const Problem& prob) const;
    double rollout_augmented_cost(const Problem& prob, const VecU& u_lo, const VecU& u_hi) const;
    void compute_gaps(const Problem& prob);
    double gap_norm() const;
    bool backward_pass(const Problem& prob, double mu, double tr_radius, const VecU& u_lo, const VecU& u_hi);
    double forward_pass(const Problem& prob, double alpha, const VecU& u_lo, const VecU& u_hi);

    // AL penalty/gradient additions for a single control
    void
    al_cost_derivatives(int k, const VecU& u, const VecU& u_lo, const VecU& u_hi, VecU& lu_al, MatUU& luu_al) const;

    bool filter_accepts(double cost, double cv) const;
    void filter_add(double cost, double cv);

    static VecU clamp_u(const VecU& u, const VecU& lo, const VecU& hi) {
        return u.cwiseMax(lo).cwiseMin(hi);
    }
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
double Solver<P>::rollout_augmented_cost(const P& prob, const VecU& u_lo, const VecU& u_hi) const {
    double c = rollout_cost(prob);
    // Add AL penalty terms
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NU; ++i) {
            const double viol_lo = u_lo(i) - us[k](i);
            const double viol_hi = us[k](i) - u_hi(i);
            if (viol_lo > 0.0) {
                c += lambda_lo_[k](i) * viol_lo + 0.5 * al_penalty_ * viol_lo * viol_lo;
            }
            if (viol_hi > 0.0) {
                c += lambda_hi_[k](i) * viol_hi + 0.5 * al_penalty_ * viol_hi * viol_hi;
            }
        }
    }
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
void Solver<P>::al_cost_derivatives(
    int k,
    const VecU& u,
    const VecU& u_lo,
    const VecU& u_hi,
    VecU& lu_al,
    MatUU& luu_al
) const {
    lu_al.setZero();
    luu_al.setZero();
    for (int i = 0; i < NU; ++i) {
        const double viol_lo = u_lo(i) - u(i);
        const double viol_hi = u(i) - u_hi(i);
        if (viol_lo > 0.0) {
            lu_al(i) += -(lambda_lo_[k](i) + al_penalty_ * viol_lo);
            luu_al(i, i) += al_penalty_;
        }
        if (viol_hi > 0.0) {
            lu_al(i) += lambda_hi_[k](i) + al_penalty_ * viol_hi;
            luu_al(i, i) += al_penalty_;
        }
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
    // Terminal
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

        // Add AL penalty derivatives
        VecU lu_al;
        MatUU luu_al;
        al_cost_derivatives(k, us[k], u_lo, u_hi, lu_al, luu_al);
        lu += lu_al;
        luu += luu_al;

        const MatXX& Vxx_next = Vxx_[k + 1];
        const VecX& Vx_next = Vx_[k + 1];

        const MatXX FxTV = fx_[k].transpose() * Vxx_next;

        VecX Qx = lx + fx_[k].transpose() * Vx_next;
        VecU Qu = lu + fu_[k].transpose() * Vx_next;
        MatXX Qxx = lxx + FxTV * fx_[k];
        MatUX Qux = lux + fu_[k].transpose() * Vxx_next * fx_[k];
        MatUU Quu = luu + fu_[k].transpose() * Vxx_next * fu_[k];

        // Regularization
        MatUU Quu_reg = Quu;
        for (int i = 0; i < NU; ++i) {
            Quu_reg(i, i) += mu;
        }
        Quu_reg = (Quu_reg + Quu_reg.transpose()).eval() * 0.5;

        Eigen::LLT<MatUU> llt(Quu_reg);
        if (llt.info() != Eigen::Success) {
            return false;
        }

        VecU k_ff = -llt.solve(Qu);
        MatUX K_fb = -llt.solve(Qux);

        // Trust-region: scale down feed-forward if too large
        const double k_norm = k_ff.norm();
        if (k_norm > tr_radius) {
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
    std::array<VecX, N + 1> xs_try;
    std::array<VecU, N> us_try;

    xs_try[0] = xs[0];

    for (int k = 0; k < N; ++k) {
        const VecX dx = xs_try[k] - xs[k];
        us_try[k] = us[k] + alpha * gains_[k].k + gains_[k].K * dx;
        us_try[k] = clamp_u(us_try[k], u_lo, u_hi);
        xs_try[k + 1] = prob.dynamics(k, xs_try[k], us_try[k]) - (1.0 - alpha) * fs_[k + 1];
    }

    double cost = 0.0;
    for (int k = 0; k < N; ++k) {
        cost += prob.running_cost(k, xs_try[k], us_try[k]);
    }
    cost += prob.terminal_cost(xs_try[N]);

    // Add AL penalty to cost for acceptance criterion
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NU; ++i) {
            const double viol_lo = u_lo(i) - us_try[k](i);
            const double viol_hi = us_try[k](i) - u_hi(i);
            if (viol_lo > 0.0) {
                cost += lambda_lo_[k](i) * viol_lo + 0.5 * al_penalty_ * viol_lo * viol_lo;
            }
            if (viol_hi > 0.0) {
                cost += lambda_hi_[k](i) * viol_hi + 0.5 * al_penalty_ * viol_hi * viol_hi;
            }
        }
    }

    xs = xs_try;
    us = us_try;
    return cost;
}

template<typename P>
SolverResult Solver<P>::solve(const P& prob, const SolverOptions& opts) {
    const VecU u_lo = prob.u_lower();
    const VecU u_hi = prob.u_upper();

    // Initialize AL multipliers and penalty
    al_penalty_ = opts.al_penalty_init;
    for (int k = 0; k < N; ++k) {
        lambda_lo_[k].setZero();
        lambda_hi_[k].setZero();
        // Clamp initial controls to keep them near feasible
        us[k] = clamp_u(us[k], u_lo, u_hi);
    }

    // Initialize filter
    filter_.clear();

    double cost = rollout_augmented_cost(prob, u_lo, u_hi);
    double tr_radius = opts.trust_region_radius;
    double mu = opts.mu_init;
    SolverResult result;
    result.cost = cost;

    int total_iters = 0;

    for (int al_iter = 0; al_iter < opts.al_outer_iters; ++al_iter) {
        const int inner_iters = (al_iter == 0) ? opts.max_iters : std::max(opts.max_iters / 2, 5);

        for (int iter = 0; iter < inner_iters; ++iter) {
            ++total_iters;

            compute_gaps(prob);
            const double gnorm = gap_norm();
            const bool use_fddp = (gnorm > opts.gap_threshold);

            // Backward pass
            bool bp_ok = false;
            for (int retry = 0; retry < 20; ++retry) {
                bp_ok = backward_pass(prob, mu, tr_radius, u_lo, u_hi);
                if (bp_ok) break;
                mu = std::min(mu * opts.mu_factor, opts.mu_max);
            }
            if (!bp_ok) {
                result.iters = total_iters;
                return result;
            }

            // Gradient convergence check
            {
                double grad_max = 0.0;
                for (int k = 0; k < N; ++k) {
                    for (int i = 0; i < NU; ++i) {
                        grad_max = std::max(grad_max, std::abs(gains_[k].k(i)));
                    }
                }
                if (grad_max < opts.tol_grad && gnorm < opts.gap_threshold) {
                    result.cost = cost;
                    result.iters = total_iters;
                    result.converged = true;
                    // Do final clamp + dual update before returning
                    goto al_update;
                }
            }

            // Forward pass with Filter-based line search
            {
                const auto xs_old = xs;
                const auto us_old = us;
                const double cost_old = cost;

                // Compute constraint violation of current trajectory
                double cv_old = gnorm;
                for (int k = 0; k < N; ++k) {
                    for (int i = 0; i < NU; ++i) {
                        cv_old += std::max(0.0, u_lo(i) - us[k](i));
                        cv_old += std::max(0.0, us[k](i) - u_hi(i));
                    }
                }

                bool accepted = false;
                double alpha = 1.0;
                while (alpha >= opts.alpha_min) {
                    xs = xs_old;
                    us = us_old;

                    const double cost_try = forward_pass(prob, alpha, u_lo, u_hi);

                    // Compute new constraint violation
                    compute_gaps(prob);
                    double cv_try = gap_norm();
                    for (int k = 0; k < N; ++k) {
                        for (int i = 0; i < NU; ++i) {
                            cv_try += std::max(0.0, u_lo(i) - us[k](i));
                            cv_try += std::max(0.0, us[k](i) - u_hi(i));
                        }
                    }

                    // Filter criterion: accept if not dominated by any filter entry
                    // AND (Armijo condition OR significant feasibility improvement)
                    const double dV_expected = alpha * dV1_ + 0.5 * alpha * alpha * dV2_;
                    const double cost_reduction = cost_old - cost_try;
                    const double expected_reduction = -dV_expected;

                    bool armijo_ok =
                        (expected_reduction > 0.0 && cost_reduction / expected_reduction >= opts.armijo_c1);

                    bool filter_ok = filter_accepts(cost_try, cv_try);

                    // Accept if: (1) Armijo holds, or (2) FDDP gap reduction, or
                    // (3) passes filter (cost or feasibility improved enough)
                    if (armijo_ok || (use_fddp && cost_try < cost_old) || filter_ok) {
                        cost = cost_try;
                        accepted = true;
                        filter_add(cost_try, cv_try);

                        // Trust-region expansion on good step
                        tr_radius = std::min(tr_radius * opts.tr_expand_factor, opts.tr_max);
                        break;
                    }

                    alpha *= 0.5;
                }

                if (!accepted) {
                    xs = xs_old;
                    us = us_old;
                    mu = std::min(mu * opts.mu_factor, opts.mu_max);
                    tr_radius = std::max(tr_radius * opts.tr_shrink_factor, opts.tr_min);
                } else {
                    mu = std::max(mu / opts.mu_factor, opts.mu_min);

                    if (std::abs(cost_old - cost) < opts.tol_cost * std::abs(cost_old) && gnorm < opts.gap_threshold) {
                        result.cost = cost;
                        result.iters = total_iters;
                        result.converged = true;
                        goto al_update;
                    }
                }
            }

            result.cost = cost;
            result.iters = total_iters;
        }

    al_update:
        // Augmented Lagrangian outer loop: update multipliers & penalty
        bool all_feasible = true;
        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < NU; ++i) {
                const double viol_lo = u_lo(i) - us[k](i);
                const double viol_hi = us[k](i) - u_hi(i);
                if (viol_lo > 0.0) {
                    lambda_lo_[k](i) = std::max(0.0, lambda_lo_[k](i) + al_penalty_ * viol_lo);
                    all_feasible = false;
                } else {
                    lambda_lo_[k](i) = 0.0;
                }
                if (viol_hi > 0.0) {
                    lambda_hi_[k](i) = std::max(0.0, lambda_hi_[k](i) + al_penalty_ * viol_hi);
                    all_feasible = false;
                } else {
                    lambda_hi_[k](i) = 0.0;
                }
            }
        }

        if (all_feasible && result.converged) {
            // Final clamp for numerical precision
            for (int k = 0; k < N; ++k) {
                us[k] = clamp_u(us[k], u_lo, u_hi);
            }
            return result;
        }

        // Increase penalty for next outer iteration
        al_penalty_ = std::min(al_penalty_ * opts.al_penalty_factor, opts.al_penalty_max);

        // Recompute cost with updated multipliers
        cost = rollout_augmented_cost(prob, u_lo, u_hi);
        filter_.clear();
    }

    // Final clamp
    for (int k = 0; k < N; ++k) {
        us[k] = clamp_u(us[k], u_lo, u_hi);
    }

    return result;
}

} // namespace fddp