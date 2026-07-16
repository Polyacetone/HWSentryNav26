#pragma once

/// @file fddp_solver.hpp
/// @brief Feasibility-driven DDP solver with trust-region, filter line-search,
///        and control-limited (box-constrained) backward pass.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/Dense>

namespace fddp {

template<typename Problem>
struct Dims;

template<typename P>
using StateVec = Eigen::Matrix<double, Dims<P>::NX, 1>;

template<typename P>
using ControlVec = Eigen::Matrix<double, Dims<P>::NU, 1>;

template<typename P>
using FxMat = Eigen::Matrix<double, Dims<P>::NX, Dims<P>::NX>;

template<typename P>
using FuMat = Eigen::Matrix<double, Dims<P>::NX, Dims<P>::NU>;

struct SolverOptions {
    int max_iters = 30;
    double tol_grad = 1e-6;
    double tol_cost = 1e-8;

    double mu_init = 1e-6;
    double mu_min = 1e-9;
    double mu_max = 1e6;
    double mu_factor = 10.0;

    double alpha_min = 1e-4;
    double armijo_c1 = 1e-4;

    double gap_threshold = 1e-3;

    double trust_region_radius = 10.0;
    double tr_expand_factor = 1.5;
    double tr_shrink_factor = 0.5;
    double tr_min = 0.1;
    double tr_max = 100.0;
};

struct SolverResult {
    double cost = 0.0;
    int iters = 0;
    bool converged = false;
};

struct FilterEntry {
    double cost;
    double constraint_violation;
};

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

    std::array<VecX, N + 1> xs;
    std::array<VecU, N> us;

    SolverResult solve(const Problem& prob, const SolverOptions& opts = {});

private:
    struct ForwardPassResult {
        double cost = 0.0;
        double max_control_update = 0.0;
        double max_state_deviation = 0.0;
        bool within_trust_region = true;
    };

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
    std::array<VecX, N + 1> fs_old_;

    std::array<VecX, N + 1> xs_try_;
    std::array<VecU, N> us_try_;
    std::array<VecX, N + 1> xs_old_;
    std::array<VecU, N> us_old_;

    double dV1_ = 0.0;
    double dV2_ = 0.0;
    double fs_old_norm_ = 0.0;
    VecU tr_radii_;

    std::vector<FilterEntry> filter_;

    double rollout_cost(const Problem& prob) const;
    void compute_gaps(const Problem& prob);
    double gap_norm() const;
    bool backward_pass(const Problem& prob, double mu, const VecU& tr_lo, const VecU& tr_hi, const VecU& u_lo, const VecU& u_hi);
    ForwardPassResult forward_pass(const Problem& prob, double alpha, double state_tr, const VecU& tr_lo, const VecU& tr_hi, const VecU& u_lo, const VecU& u_hi);

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
        constexpr double BOX_DZ = 1e-10;
        if (kf <= du_lo(f) + BOX_DZ || kf >= du_hi(f) - BOX_DZ) {
            k(f) = std::clamp(kf, du_lo(f), du_hi(f));
            K.row(f).setZero();
            return true;
        }

        k(f) = kf;
        K.row(f).noalias() = -gx.row(f) / hff;
        return true;
    } else {
        // 枚举每个控制量的 free/lower/upper 状态。MPCC 的 NU=3，仅 27 种组合；
        // 相比逐元素截断，这会正确计入激活边界与自由变量之间的 Hessian 耦合。
        constexpr int ACTIVE_SET_COUNT = [] {
            int count = 1;
            for (int i = 0; i < NU; ++i) count *= 3;
            return count;
        }();
        double best_objective = std::numeric_limits<double>::infinity();
        bool found = false;
        constexpr double FEASIBILITY_EPS = 1e-10;

        for (int encoded = 0; encoded < ACTIVE_SET_COUNT; ++encoded) {
            int code = encoded;
            std::array<int, NU> status {};
            std::array<int, NU> free_indices {};
            int free_count = 0;
            VecU candidate = VecU::Zero();
            MatUX candidate_K = MatUX::Zero();

            for (int i = 0; i < NU; ++i) {
                status[static_cast<size_t>(i)] = code % 3 - 1;
                code /= 3;
                if (status[static_cast<size_t>(i)] < 0) {
                    candidate(i) = du_lo(i);
                } else if (status[static_cast<size_t>(i)] > 0) {
                    candidate(i) = du_hi(i);
                } else {
                    free_indices[static_cast<size_t>(free_count++)] = i;
                }
            }

            if (free_count > 0) {
                Eigen::MatrixXd h_free(free_count, free_count);
                Eigen::VectorXd rhs(free_count);
                Eigen::MatrixXd rhs_feedback(free_count, NX);
                for (int r = 0; r < free_count; ++r) {
                    const int row = free_indices[static_cast<size_t>(r)];
                    rhs(r) = -g(row);
                    rhs_feedback.row(r) = -gx.row(row);
                    for (int active = 0; active < NU; ++active) {
                        if (status[static_cast<size_t>(active)] != 0) {
                            rhs(r) -= h(row, active) * candidate(active);
                        }
                    }
                    for (int c = 0; c < free_count; ++c) {
                        h_free(r, c) = h(row, free_indices[static_cast<size_t>(c)]);
                    }
                }

                Eigen::LLT<Eigen::MatrixXd> llt(h_free);
                if (llt.info() != Eigen::Success) continue;
                const Eigen::VectorXd free_solution = llt.solve(rhs);
                const Eigen::MatrixXd free_feedback = llt.solve(rhs_feedback);
                if (llt.info() != Eigen::Success
                    || !free_solution.allFinite() || !free_feedback.allFinite()) {
                    continue;
                }
                for (int i = 0; i < free_count; ++i) {
                    const int index = free_indices[static_cast<size_t>(i)];
                    candidate(index) = free_solution(i);
                    candidate_K.row(index) = free_feedback.row(i);
                }
            }

            bool feasible = true;
            for (int i = 0; i < NU; ++i) {
                if (candidate(i) < du_lo(i) - FEASIBILITY_EPS
                    || candidate(i) > du_hi(i) + FEASIBILITY_EPS) {
                    feasible = false;
                    break;
                }
            }
            if (!feasible) continue;

            const double objective = 0.5 * candidate.dot(h * candidate) + g.dot(candidate);
            if (objective < best_objective) {
                best_objective = objective;
                k = candidate;
                K = candidate_K;
                found = true;
            }
        }
        return found;
    }
}

template<typename P>
bool Solver<P>::filter_accepts(double cost, double cv) const {
    for (const auto& f: filter_) {
        if (cost >= f.cost && cv >= f.constraint_violation) {
            return false;
        }
    }
    return true;
}

template<typename P>
void Solver<P>::filter_add(double cost, double cv) {
    std::erase_if(filter_, [&](const FilterEntry& f) { return cost <= f.cost && cv <= f.constraint_violation; });
    filter_.push_back({cost, cv});
}

template<typename P>
bool Solver<P>::backward_pass(const P& prob, double mu, const VecU& tr_lo, const VecU& tr_hi, const VecU& u_lo, const VecU& u_hi) {
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
        const VecU du_lo = (u_lo - us[k]).cwiseMax(tr_lo);
        const VecU du_hi = (u_hi - us[k]).cwiseMin(tr_hi);
        if (!solve_box_qp(Quu_reg, Qu, Qux, du_lo, du_hi, k_ff, K_fb)) {
            return false;
        }

        gains_[k].k = k_ff;
        gains_[k].K = K_fb;

        dV1_ += k_ff.dot(Qu);
        dV2_ += k_ff.dot(Quu_reg * k_ff);

        Vx_[k] = Qx + K_fb.transpose() * Quu * k_ff + K_fb.transpose() * Qu + Qux.transpose() * k_ff;
        Vxx_[k] = Qxx + K_fb.transpose() * Quu * K_fb + K_fb.transpose() * Qux + Qux.transpose() * K_fb;
        Vxx_[k] = (Vxx_[k] + Vxx_[k].transpose()).eval() * 0.5;

        Vx_[k] -= Vxx_[k] * fs_[k];
    }

    return true;
}

template<typename P>
typename Solver<P>::ForwardPassResult Solver<P>::forward_pass(
    const P& prob,
    double alpha,
    double state_tr,
    const VecU& tr_lo,
    const VecU& tr_hi,
    const VecU& u_lo,
    const VecU& u_hi
) {
    ForwardPassResult result;
    xs_try_[0] = xs[0];

    for (int k = 0; k < N; ++k) {
        const VecX dx = xs_try_[k] - xs[k];
        result.max_state_deviation = std::max(result.max_state_deviation, dx.norm());
        if (result.max_state_deviation > state_tr) {
            result.within_trust_region = false;
            return result;
        }

        us_try_[k] = clamp_u(us[k] + alpha * gains_[k].k + gains_[k].K * dx, u_lo, u_hi);
        const VecU du = us_try_[k] - us[k];
        for (int i = 0; i < NU; ++i) {
            if (du(i) < tr_lo(i) - 1e-12 || du(i) > tr_hi(i) + 1e-12) {
                result.within_trust_region = false;
                return result;
            }
        }
        result.max_control_update = std::max(result.max_control_update, du.norm());

        xs_try_[k + 1] = prob.dynamics(k, xs_try_[k], us_try_[k]) - (1.0 - alpha) * fs_[k + 1];
        if (!xs_try_[k + 1].allFinite()) {
            result.within_trust_region = false;
            return result;
        }
        result.max_state_deviation = std::max(result.max_state_deviation, (xs_try_[k + 1] - xs[k + 1]).norm());
        if (result.max_state_deviation > state_tr) {
            result.within_trust_region = false;
            return result;
        }
    }

    result.cost = 0.0;
    for (int k = 0; k < N; ++k) {
        result.cost += prob.running_cost(k, xs_try_[k], us_try_[k]);
    }
    result.cost += prob.terminal_cost(xs_try_[N]);

    xs = xs_try_;
    us = us_try_;
    return result;
}

template<typename P>
SolverResult Solver<P>::solve(const P& prob, const SolverOptions& opts) {
    const VecU u_lo = prob.u_lower();
    const VecU u_hi = prob.u_upper();

    for (int k = 0; k < N; ++k) {
        us[k] = clamp_u(us[k], u_lo, u_hi);
    }

    const VecU ctrl_span = (u_hi - u_lo).cwiseMax(1.0);
    const double mean_span = ctrl_span.mean();
    const double norm_tr = std::max(opts.trust_region_radius, 1e-6) / std::max(mean_span, 1.0);
    tr_radii_ = norm_tr * ctrl_span;

    filter_.clear();
    filter_.reserve(static_cast<size_t>(std::max(opts.max_iters, 8)) + 8U);

    double cost = rollout_cost(prob);
    compute_gaps(prob);
    filter_add(cost, gap_norm());

    double tr_radius = tr_radii_.maxCoeff();
    double mu = opts.mu_init;

    SolverResult result;
    result.cost = cost;

    for (int iter = 0; iter < opts.max_iters; ++iter) {
        result.iters = iter + 1;

        double gnorm = gap_norm();
        const bool use_fddp = (gnorm > opts.gap_threshold);

        const VecU tr_vec_lo = -tr_radii_;
        const VecU tr_vec_hi = tr_radii_;

        bool bp_ok = false;
        for (int retry = 0; retry < 20; ++retry) {
            bp_ok = backward_pass(prob, mu, tr_vec_lo, tr_vec_hi, u_lo, u_hi);
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
                if (us[k](i) > u_lo(i) + 1e-8 && us[k](i) < u_hi(i) - 1e-8) {
                    grad_max = std::max(grad_max, std::abs(gains_[k].k(i)));
                }
            }
        }
        if (grad_max < opts.tol_grad && gnorm < opts.gap_threshold) {
            result.converged = true;
            break;
        }

        xs_old_ = xs;
        us_old_ = us;
        const double cost_old = cost;

        fs_old_ = fs_;
        fs_old_norm_ = gap_norm();

        bool accepted = false;
        double alpha = 1.0;
        while (alpha >= opts.alpha_min) {
            xs = xs_old_;
            us = us_old_;

            const ForwardPassResult fp = forward_pass(
                prob, alpha, tr_radius, tr_vec_lo, tr_vec_hi, u_lo, u_hi
            );
            if (!fp.within_trust_region) {
                alpha *= 0.5;
                continue;
            }

            const double cost_try = fp.cost;
            const double cv_try = (1.0 - alpha) * fs_old_norm_;

            const double dV_expected = alpha * dV1_ + 0.5 * alpha * alpha * dV2_;
            const double expected_reduction = -dV_expected;
            const double cost_reduction = cost_old - cost_try;

            const bool armijo_ok =
                (expected_reduction > 0.0 && (cost_reduction / expected_reduction) >= opts.armijo_c1);
            const bool filter_ok = filter_accepts(cost_try, cv_try);

            if (armijo_ok || (use_fddp && cost_try < cost_old) || filter_ok) {
                const double rho = expected_reduction > 0.0
                    ? cost_reduction / expected_reduction
                    : -1.0;
                if (rho > 0.75) {
                    mu = std::max(mu / opts.mu_factor, opts.mu_min);
                } else if (rho < 0.25) {
                    mu = std::min(mu * opts.mu_factor, opts.mu_max);
                }

                cost = cost_try;
                gnorm = cv_try;
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
            fs_ = fs_old_;
            mu = std::min(mu * opts.mu_factor, opts.mu_max);
            tr_radius = std::max(tr_radius * opts.tr_shrink_factor, opts.tr_min);
            continue;
        }

        fs_[0].setZero();
        for (int k = 1; k <= N; ++k) {
            fs_[k] = (1.0 - alpha) * fs_old_[k];
        }

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
