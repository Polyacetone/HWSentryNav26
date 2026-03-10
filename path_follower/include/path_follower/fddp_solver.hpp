#pragma once

/// @file fddp_solver.hpp
/// @brief Feasibility-Driven Differential Dynamic Programming (FDDP) solver.
///
/// Generic, header-only FDDP with box constraints on controls.
/// The user supplies a concrete Problem that provides dynamics and costs.
///
/// Reference: Mastalli et al., "Feasibility-Driven DDP" (RAL 2020).

#include <array>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

namespace fddp {

// ═══════════════════════════════════════════════════════════════
//  Compile-time problem dimensions
// ═══════════════════════════════════════════════════════════════

/// Users must specialize this for their problem type.
///
///   template<> struct Dims<MyProblem> {
///       static constexpr int NX = ...;
///       static constexpr int NU = ...;
///       static constexpr int N  = ...;  // horizon length
///   };
template <typename Problem>
struct Dims;

// ═══════════════════════════════════════════════════════════════
//  Type aliases
// ═══════════════════════════════════════════════════════════════

template <typename P>
using StateVec = Eigen::Matrix<double, Dims<P>::NX, 1>;

template <typename P>
using ControlVec = Eigen::Matrix<double, Dims<P>::NU, 1>;

template <typename P>
using FxMat = Eigen::Matrix<double, Dims<P>::NX, Dims<P>::NX>;

template <typename P>
using FuMat = Eigen::Matrix<double, Dims<P>::NX, Dims<P>::NU>;

// ═══════════════════════════════════════════════════════════════
//  Problem concept (duck-typed)
// ═══════════════════════════════════════════════════════════════
//
//  A valid Problem must provide:
//
//    // Discrete dynamics: x_{k+1} = f(x_k, u_k)
//    StateVec<P> dynamics(int k, const StateVec<P>& x, const ControlVec<P>& u) const;
//
//    // Dynamics Jacobians: fx = ∂f/∂x, fu = ∂f/∂u
//    void dynamics_jacobians(int k, const StateVec<P>& x, const ControlVec<P>& u,
//                            FxMat<P>& fx, FuMat<P>& fu) const;
//
//    // Running cost: l(k, x, u)
//    double running_cost(int k, const StateVec<P>& x, const ControlVec<P>& u) const;
//
//    // Running cost derivatives (Gauss-Newton / exact hybrid)
//    void running_cost_derivatives(int k, const StateVec<P>& x, const ControlVec<P>& u,
//                                  StateVec<P>& lx, ControlVec<P>& lu,
//                                  FxMat<P>& lxx, Eigen::Matrix<double,NU,NX>& lux,
//                                  Eigen::Matrix<double,NU,NU>& luu) const;
//
//    // Terminal cost: lf(x_N)
//    double terminal_cost(const StateVec<P>& x) const;
//
//    // Terminal cost derivatives
//    void terminal_cost_derivatives(const StateVec<P>& x, StateVec<P>& lfx,
//                                   FxMat<P>& lfxx) const;
//
//    // Control bounds
//    ControlVec<P> u_lower() const;
//    ControlVec<P> u_upper() const;

// ═══════════════════════════════════════════════════════════════
//  Solver options
// ═══════════════════════════════════════════════════════════════

struct SolverOptions {
    int    max_iters        = 30;
    double tol_grad         = 1e-6;
    double tol_cost         = 1e-8;

    double mu_init          = 1e-6;   // initial regularization
    double mu_min           = 1e-9;
    double mu_max           = 1e6;
    double mu_factor        = 10.0;

    // Line search (Armijo)
    double alpha_min        = 1e-4;
    double armijo_c1        = 1e-4;

    // Feasibility-driven gap contraction
    double gap_threshold    = 1e-3;   // switch to standard DDP when ||gaps|| < this
};

// ═══════════════════════════════════════════════════════════════
//  Solver result
// ═══════════════════════════════════════════════════════════════

struct SolverResult {
    double cost      = 0.0;
    int    iters     = 0;
    bool   converged = false;
};

// ═══════════════════════════════════════════════════════════════
//  FDDP Solver
// ═══════════════════════════════════════════════════════════════

template <typename Problem>
class Solver {
public:
    static constexpr int NX = Dims<Problem>::NX;
    static constexpr int NU = Dims<Problem>::NU;
    static constexpr int N  = Dims<Problem>::N;

    using VecX = Eigen::Matrix<double, NX, 1>;
    using VecU = Eigen::Matrix<double, NU, 1>;
    using MatXX = Eigen::Matrix<double, NX, NX>;
    using MatXU = Eigen::Matrix<double, NX, NU>;
    using MatUX = Eigen::Matrix<double, NU, NX>;
    using MatUU = Eigen::Matrix<double, NU, NU>;

    // ─── Trajectory storage ───
    std::array<VecX, N + 1> xs;   // states  [0..N]
    std::array<VecU, N>     us;   // controls [0..N-1]

    /// Solve the OCP. xs[0] must be set to the initial state before calling.
    /// us should be warm-started (or zero-initialized).
    SolverResult solve(const Problem& prob, const SolverOptions& opts = {});

private:
    // ─── Backward pass data ───
    struct FeedbackGain {
        MatUX K;
        VecU  k;
    };

    std::array<FeedbackGain, N> gains_;

    // Value function
    std::array<VecX, N + 1>  Vx_;
    std::array<MatXX, N + 1> Vxx_;

    // Dynamics cache
    std::array<MatXX, N> fx_;
    std::array<MatXU, N> fu_;

    // Feasibility gaps: fs_[k] = f(x_bar[k-1], u_bar[k-1]) - x_bar[k] for k>=1, fs_[0]=0
    std::array<VecX, N + 1> fs_;

    // Expected cost reduction from backward pass
    double dV1_ = 0.0;  // linear term
    double dV2_ = 0.0;  // quadratic term

    // ─── Methods ───
    double rollout_cost(const Problem& prob) const;
    void compute_gaps(const Problem& prob);
    double gap_norm() const;
    bool backward_pass(const Problem& prob, double mu);
    double forward_pass(const Problem& prob, double alpha);

    /// Clamp u element-wise to [lo, hi]
    static VecU clamp_u(const VecU& u, const VecU& lo, const VecU& hi) {
        return u.cwiseMax(lo).cwiseMin(hi);
    }
};

// ═══════════════════════════════════════════════════════════════
//  Implementation
// ═══════════════════════════════════════════════════════════════

template <typename P>
double Solver<P>::rollout_cost(const P& prob) const {
    double c = 0.0;
    for (int k = 0; k < N; ++k) {
        c += prob.running_cost(k, xs[k], us[k]);
    }
    c += prob.terminal_cost(xs[N]);
    return c;
}

template <typename P>
void Solver<P>::compute_gaps(const P& prob) {
    fs_[0].setZero();
    for (int k = 1; k <= N; ++k) {
        fs_[k] = prob.dynamics(k - 1, xs[k - 1], us[k - 1]) - xs[k];
    }
}

template <typename P>
double Solver<P>::gap_norm() const {
    double s = 0.0;
    for (int k = 0; k <= N; ++k) {
        s += fs_[k].squaredNorm();
    }
    return std::sqrt(s);
}

template <typename P>
bool Solver<P>::backward_pass(const P& prob, double mu) {
    // Terminal
    prob.terminal_cost_derivatives(xs[N], Vx_[N], Vxx_[N]);

    // Add gap correction to terminal value gradient (FDDP)
    Vx_[N] -= Vxx_[N] * fs_[N];

    dV1_ = 0.0;
    dV2_ = 0.0;

    const VecU u_lo = prob.u_lower();
    const VecU u_hi = prob.u_upper();

    for (int k = N - 1; k >= 0; --k) {
        // Dynamics Jacobians
        prob.dynamics_jacobians(k, xs[k], us[k], fx_[k], fu_[k]);

        // Cost derivatives
        VecX lx;
        VecU lu;
        MatXX lxx;
        MatUX lux;
        MatUU luu;
        prob.running_cost_derivatives(k, xs[k], us[k], lx, lu, lxx, lux, luu);

        // Q-function derivatives
        const MatXX& Vxx_next = Vxx_[k + 1];
        const VecX&  Vx_next  = Vx_[k + 1];

        const MatXX FxTV = fx_[k].transpose() * Vxx_next;

        VecX  Qx  = lx  + fx_[k].transpose() * Vx_next;
        VecU  Qu  = lu  + fu_[k].transpose() * Vx_next;
        MatXX Qxx = lxx + FxTV * fx_[k];
        MatUX Qux = lux + fu_[k].transpose() * Vxx_next * fx_[k];
        MatUU Quu = luu + fu_[k].transpose() * Vxx_next * fu_[k];

        // Regularization (on Quu only)
        MatUU Quu_reg = Quu;
        for (int i = 0; i < NU; ++i) {
            Quu_reg(i, i) += mu;
        }

        // Symmetrize
        Quu_reg = (Quu_reg + Quu_reg.transpose()).eval() * 0.5;

        // Check positive-definiteness via Cholesky
        Eigen::LLT<MatUU> llt(Quu_reg);
        if (llt.info() != Eigen::Success) {
            return false;  // signal to increase regularization
        }

        // ─── Box-constrained backward pass ───
        // Compute the free (unconstrained) feed-forward: k_free = -Quu_reg^{-1} Qu
        // Then project: for each dimension, if the control is at a bound and the
        // unconstrained direction pushes further into the bound, clamp to zero
        // and remove that dimension from the gain computation.
        VecU k_ff = -llt.solve(Qu);

        // Determine active set at current iterate
        // A dimension i is "clamped" if u[i] is at bound and k_ff pushes further out
        Eigen::Array<bool, NU, 1> clamped;
        for (int i = 0; i < NU; ++i) {
            const double ui = us[k](i);
            const bool at_lo = (ui <= u_lo(i) + 1e-10);
            const bool at_hi = (ui >= u_hi(i) - 1e-10);
            clamped(i) = (at_lo && k_ff(i) < 0.0) || (at_hi && k_ff(i) > 0.0);
        }

        // Zero out clamped dimensions in feed-forward and gain
        MatUX K_fb;
        const int n_free = static_cast<int>((clamped == false).count());
        if (n_free == NU) {
            // All free — standard DDP
            K_fb = -llt.solve(Qux);
        } else if (n_free == 0) {
            // All clamped
            k_ff.setZero();
            K_fb.setZero();
        } else {
            // Partial clamping: solve reduced system
            // We solve the free subproblem and set clamped dims to zero
            k_ff = VecU::Zero();
            K_fb = MatUX::Zero();

            // Build index map for free dimensions
            Eigen::VectorXi free_idx(n_free);
            int cnt = 0;
            for (int i = 0; i < NU; ++i) {
                if (!clamped(i)) {
                    free_idx(cnt++) = i;
                }
            }

            // Extract free sub-matrices
            Eigen::MatrixXd Quu_free(n_free, n_free);
            Eigen::VectorXd Qu_free(n_free);
            Eigen::MatrixXd Qux_free(n_free, NX);

            for (int a = 0; a < n_free; ++a) {
                Qu_free(a) = Qu(free_idx(a));
                Qux_free.row(a) = Qux.row(free_idx(a));
                for (int b = 0; b < n_free; ++b) {
                    Quu_free(a, b) = Quu_reg(free_idx(a), free_idx(b));
                }
            }

            Eigen::LLT<Eigen::MatrixXd> llt_free(Quu_free);
            if (llt_free.info() != Eigen::Success) {
                return false;
            }

            Eigen::VectorXd k_free = -llt_free.solve(Qu_free);
            Eigen::MatrixXd K_free = -llt_free.solve(Qux_free);

            for (int a = 0; a < n_free; ++a) {
                k_ff(free_idx(a)) = k_free(a);
                K_fb.row(free_idx(a)) = K_free.row(a);
            }
        }

        gains_[k].k = k_ff;
        gains_[k].K = K_fb;

        // Expected improvement
        dV1_ += k_ff.dot(Qu);
        dV2_ += 0.5 * k_ff.dot(Quu * k_ff);

        // Value function update
        Vx_[k]  = Qx  + K_fb.transpose() * Quu * k_ff + K_fb.transpose() * Qu + Qux.transpose() * k_ff;
        Vxx_[k] = Qxx + K_fb.transpose() * Quu * K_fb + K_fb.transpose() * Qux + Qux.transpose() * K_fb;
        Vxx_[k] = (Vxx_[k] + Vxx_[k].transpose()).eval() * 0.5;

        // FDDP gap correction for previous node
        Vx_[k] -= Vxx_[k] * fs_[k];
    }

    return true;
}

template <typename P>
double Solver<P>::forward_pass(const P& prob, double alpha) {
    const VecU u_lo = prob.u_lower();
    const VecU u_hi = prob.u_upper();

    // Trial trajectory
    std::array<VecX, N + 1> xs_try;
    std::array<VecU, N>     us_try;

    xs_try[0] = xs[0];  // initial state is fixed

    for (int k = 0; k < N; ++k) {
        const VecX dx = xs_try[k] - xs[k];
        us_try[k] = clamp_u(us[k] + alpha * gains_[k].k + gains_[k].K * dx, u_lo, u_hi);
        xs_try[k + 1] = prob.dynamics(k, xs_try[k], us_try[k]) - (1.0 - alpha) * fs_[k + 1];
    }

    // Compute cost
    double cost = 0.0;
    for (int k = 0; k < N; ++k) {
        cost += prob.running_cost(k, xs_try[k], us_try[k]);
    }
    cost += prob.terminal_cost(xs_try[N]);

    // Accept
    xs = xs_try;
    us = us_try;

    return cost;
}

template <typename P>
SolverResult Solver<P>::solve(const P& prob, const SolverOptions& opts) {
    // Keep caller-provided primal trajectory; only clamp controls.
    {
        const VecU u_lo = prob.u_lower();
        const VecU u_hi = prob.u_upper();
        for (int k = 0; k < N; ++k) {
            us[k] = clamp_u(us[k], u_lo, u_hi);
        }
    }
    double cost = rollout_cost(prob);

    double mu = opts.mu_init;
    SolverResult result;
    result.cost = cost;

    for (int iter = 0; iter < opts.max_iters; ++iter) {
        // Compute feasibility gaps
        compute_gaps(prob);
        const double gnorm = gap_norm();
        const bool use_fddp = (gnorm > opts.gap_threshold);

        // Backward pass (retry with increasing mu on failure)
        bool bp_ok = false;
        for (int retry = 0; retry < 20; ++retry) {
            bp_ok = backward_pass(prob, mu);
            if (bp_ok) break;
            mu = std::min(mu * opts.mu_factor, opts.mu_max);
        }
        if (!bp_ok) {
            result.iters = iter + 1;
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
                result.iters = iter + 1;
                result.converged = true;
                return result;
            }
        }

        // Forward pass with Armijo line search
        // Save current trajectory
        const auto xs_old = xs;
        const auto us_old = us;
        const double cost_old = cost;

        bool accepted = false;
        double alpha = 1.0;
        while (alpha >= opts.alpha_min) {
            // Restore before trying
            xs = xs_old;
            us = us_old;

            const double cost_try = forward_pass(prob, alpha);

            // Expected improvement
            double dV_expected;
            if (use_fddp) {
                // FDDP criterion: expected improvement includes gap terms
                // Use simpler criterion: accept if cost decreases
                dV_expected = alpha * dV1_ + 0.5 * alpha * alpha * dV2_;
            } else {
                dV_expected = alpha * dV1_ + 0.5 * alpha * alpha * dV2_;
            }

            // Armijo condition
            const double cost_reduction = cost_old - cost_try;
            const double expected_reduction = -dV_expected;

            if (expected_reduction > 0.0 && cost_reduction / expected_reduction >= opts.armijo_c1) {
                cost = cost_try;
                accepted = true;
                break;
            }

            // For FDDP with large gaps, accept any improvement
            if (use_fddp && cost_try < cost_old) {
                cost = cost_try;
                accepted = true;
                break;
            }

            alpha *= 0.5;
        }

        if (!accepted) {
            // Reject: restore old trajectory, increase regularization
            xs = xs_old;
            us = us_old;
            mu = std::min(mu * opts.mu_factor, opts.mu_max);
        } else {
            // Decrease regularization
            mu = std::max(mu / opts.mu_factor, opts.mu_min);

            // Cost convergence check
            if (std::abs(cost_old - cost) < opts.tol_cost * std::abs(cost_old) && gnorm < opts.gap_threshold) {
                result.cost = cost;
                result.iters = iter + 1;
                result.converged = true;
                return result;
            }
        }

        result.cost = cost;
        result.iters = iter + 1;
    }

    return result;
}

}  // namespace fddp