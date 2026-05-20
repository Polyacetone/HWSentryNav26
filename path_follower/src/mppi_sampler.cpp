#include <path_follower/mppi_sampler.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace path_follower {
namespace {

constexpr double SIGMA_EPS = 1e-6;
constexpr double SPLINE_EPS = 1e-9;

double wrap_pi(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

struct SampledSequence {
    std::array<StateVec, MPC_HORIZON + 1> xs {};
    std::array<ControlVec, MPC_HORIZON> us {};
    std::array<ControlVec, MPC_HORIZON> delta_u {};
    double cost = std::numeric_limits<double>::infinity();
    bool valid = false;
    std::optional<RolloutLethalObstacleInfo> lethal_obstacle;
};

ControlVec regularization_sigma_vector(const MPCFollowMPPIParams& params) {
    return ControlVec(
        std::max(params.regularization_std.velocity, SIGMA_EPS),
        std::max(params.regularization_std.omega, SIGMA_EPS)
    );
}

uint64_t make_seed(uint64_t base_seed, int iteration, int sample_index) {
    constexpr uint64_t K0 = 0x9E3779B97F4A7C15ULL;
    constexpr uint64_t K1 = 0xBF58476D1CE4E5B9ULL;
    return base_seed ^ K0 ^ (static_cast<uint64_t>(iteration + 1) * K1) ^ static_cast<uint64_t>(sample_index + 1);
}

double project_to_control_point_path_u(
    const std::vector<Eigen::Vector2d>& cps,
    const Eigen::Vector2d& pos,
    double u_hint,
    const MPCFollowProjection& projection,
    double u_min = PATH_U_EXTRAP_MIN,
    double u_max = PATH_U_EXTRAP_MAX
) {
    auto eval_pos = [&](double u) {
        Eigen::Vector2d p;
        eval_quadratic_bspline2_extrapolated(cps, u, &p, nullptr, nullptr);
        return p;
    };

    const auto search = [&](double a, double b, int n) {
        double best_u = a;
        double best_d2 = std::numeric_limits<double>::infinity();
        for (int i = 0; i <= n; ++i) {
            const double ratio = (n == 0) ? 0.0 : (static_cast<double>(i) / static_cast<double>(n));
            const double u = a + (b - a) * ratio;
            const double d2 = (eval_pos(u) - pos).squaredNorm();
            if (d2 < best_d2) {
                best_d2 = d2;
                best_u = u;
            }
        }
        return best_u;
    };

    const int num_samples = std::max(1, projection.proj_num_samples);
    double u_best = search(
        std::clamp(u_hint - projection.proj_search_window, 0.0, 1.0),
        std::clamp(u_hint + projection.proj_search_window, 0.0, 1.0),
        num_samples
    );

    if ((eval_pos(u_best) - pos).norm() > projection.local_search_lazy_distance) {
        u_best = search(0.0, 1.0, num_samples);
        u_best = search(
            std::clamp(u_best - projection.proj_search_window, 0.0, 1.0),
            std::clamp(u_best + projection.proj_search_window, 0.0, 1.0),
            num_samples
        );
    }

    double best_d2 = (eval_pos(u_best) - pos).squaredNorm();
    const auto check_endpoint_extension = [&](double u_edge) {
        Eigen::Vector2d p_edge;
        Eigen::Vector2d d1_edge;
        eval_quadratic_bspline2_extrapolated(cps, u_edge, &p_edge, &d1_edge, nullptr);
        const double tangent_norm2 = d1_edge.squaredNorm();
        if (tangent_norm2 <= SPLINE_EPS) {
            return;
        }

        double u_ext = u_edge;
        if (u_edge <= 0.0) {
            u_ext = (pos - p_edge).dot(d1_edge) / tangent_norm2;
            if (u_ext >= 0.0) {
                return;
            }
        } else {
            u_ext = 1.0 + (pos - p_edge).dot(d1_edge) / tangent_norm2;
            if (u_ext <= 1.0) {
                return;
            }
        }

        const double d2 = (p_edge + d1_edge * (u_ext - u_edge) - pos).squaredNorm();
        if (d2 < best_d2) {
            best_d2 = d2;
            u_best = u_ext;
        }
    };

    check_endpoint_extension(0.0);
    check_endpoint_extension(1.0);
    return clamp_path_u_extrapolated(u_best, u_min, u_max);
}

double advance_u_progress_on_path(double u_cur, const StateVec& x, const std::vector<Eigen::Vector2d>& ref_cps) {
    const double uc = clamp_path_u_extrapolated(u_cur);

    Eigen::Vector2d pr;
    Eigen::Vector2d d1;
    Eigen::Vector2d d2;
    eval_quadratic_bspline2_extrapolated(ref_cps, uc, &pr, &d1, &d2);

    const double dsdu = std::sqrt(d1.squaredNorm() + 0.01) + 1e-6;
    const double kappa = quadratic_bspline_curvature(d1, d2);
    const double theta_path = std::atan2(d1.y(), d1.x());
    const double sin_r = std::sin(theta_path);
    const double cos_r = std::cos(theta_path);

    const double ex = x(ix::X) - pr.x();
    const double ey_w = x(ix::Y) - pr.y();
    const double ey = -ex * sin_r + ey_w * cos_r;
    const double etheta = wrap_pi(x(ix::THETA) - theta_path);

    const double num = x(ix::V) * std::cos(etheta);
    const double denom_raw = 1.0 - kappa * ey;
    constexpr double DENOM_EPS = 0.1;
    const double denom_mag = std::sqrt(denom_raw * denom_raw + DENOM_EPS * DENOM_EPS);
    const double denom = std::copysign(denom_mag, denom_raw);
    const double dsdt = num / denom;
    return clamp_path_u_extrapolated(uc + MPC_DT * dsdt / dsdu);
}

double control_regularization_cost(
    const std::array<ControlVec, MPC_HORIZON>& delta_u,
    const ControlVec& inv_variance,
    double gamma
) {
    double cost = 0.0;
    for (int k = 0; k < MPC_HORIZON; ++k) {
        const ControlVec scaled_delta = delta_u[static_cast<size_t>(k)].cwiseProduct(inv_variance);
    cost += 0.5 * gamma * delta_u[static_cast<size_t>(k)].dot(scaled_delta);
    }
    return cost;
}

std::vector<Eigen::Vector2d> build_control_point_normals(const std::vector<Eigen::Vector2d>& ref_cps) {
    std::vector<Eigen::Vector2d> normals(ref_cps.size(), Eigen::Vector2d::UnitY());
    if (ref_cps.empty()) {
        return normals;
    }

    for (size_t i = 0; i < ref_cps.size(); ++i) {
        Eigen::Vector2d tangent = Eigen::Vector2d::Zero();
        if (i > 0) {
            tangent += ref_cps[i] - ref_cps[i - 1];
        }
        if (i + 1 < ref_cps.size()) {
            tangent += ref_cps[i + 1] - ref_cps[i];
        }

        if (tangent.norm() <= SPLINE_EPS) {
            if (i > 0) {
                normals[i] = normals[i - 1];
            }
            continue;
        }

        tangent.normalize();
        normals[i] = Eigen::Vector2d(-tangent.y(), tangent.x());
    }

    return normals;
}

std::vector<Eigen::Vector2d> deform_reference_path(
    const std::vector<Eigen::Vector2d>& ref_cps,
    const std::vector<Eigen::Vector2d>& normals,
    double offset_std,
    std::mt19937_64& rng
) {
    std::vector<Eigen::Vector2d> perturbed = ref_cps;
    std::normal_distribution<double> offset_dist(0.0, std::max(offset_std, SIGMA_EPS));
    for (size_t i = 0; i < perturbed.size(); ++i) {
        perturbed[i] += offset_dist(rng) * normals[i];
    }
    return perturbed;
}

SampledSequence rollout_control_sequence(
    const FollowProblem& problem,
    const StateVec& x0,
    const std::array<ControlVec, MPC_HORIZON>& nominal_controls,
    const std::array<ControlVec, MPC_HORIZON>& controls,
    const ControlVec& u_lo,
    const ControlVec& u_hi,
    const ControlVec& inv_variance,
    double gamma
) {
    SampledSequence sample;
    sample.xs[0] = x0;
    if (const auto lethal = problem.detect_lethal_obstacle(0, sample.xs[0])) {
        sample.lethal_obstacle = lethal;
        return sample;
    }

    double running_cost = 0.0;
    for (int k = 0; k < MPC_HORIZON; ++k) {
        const size_t idx = static_cast<size_t>(k);
        sample.us[idx] = controls[idx].cwiseMax(u_lo).cwiseMin(u_hi);
        sample.delta_u[idx] = sample.us[idx] - nominal_controls[idx];
        running_cost += problem.running_cost(k, sample.xs[idx], sample.us[idx]);
        sample.xs[idx + 1] = problem.dynamics(k, sample.xs[idx], sample.us[idx]);
        if (!sample.xs[idx + 1].allFinite()) {
            return sample;
        }
        if (const auto lethal = problem.detect_lethal_obstacle(k + 1, sample.xs[idx + 1])) {
            sample.lethal_obstacle = lethal;
            return sample;
        }
    }

    running_cost += problem.terminal_cost(sample.xs[MPC_HORIZON]);
    running_cost += control_regularization_cost(sample.delta_u, inv_variance, gamma);
    if (!std::isfinite(running_cost)) {
        return sample;
    }

    sample.cost = running_cost;
    sample.valid = true;
    return sample;
}

SampledSequence rollout_generated_sequence(
    const FollowProblem& problem,
    const StateVec& x0,
    const std::array<ControlVec, MPC_HORIZON>& nominal_controls,
    const std::vector<Eigen::Vector2d>& ref_cps,
    const ControlVec& u_lo,
    const ControlVec& u_hi,
    const ControlVec& inv_variance,
    const MPCFollowMPPIParams& params
) {
    SampledSequence sample;
    sample.xs[0] = x0;
    sample.xs[0](ix::PATH_U) = project_to_control_point_path_u(
        ref_cps,
        Eigen::Vector2d(x0(ix::X), x0(ix::Y)),
        std::clamp(x0(ix::PATH_U), 0.0, 1.0),
        problem.params().follow.projection
    );

    if (const auto lethal = problem.detect_lethal_obstacle(0, sample.xs[0])) {
        sample.lethal_obstacle = lethal;
        return sample;
    }

    const double heading_k = params.geometry_sampling.heading_feedback_gain;

    double running_cost = 0.0;
    for (int k = 0; k < MPC_HORIZON; ++k) {
        const size_t idx = static_cast<size_t>(k);
        const double path_u = clamp_path_u_extrapolated(sample.xs[idx](ix::PATH_U));

        Eigen::Vector2d path_pos;
        Eigen::Vector2d d1;
        Eigen::Vector2d d2;
        eval_quadratic_bspline2_extrapolated(ref_cps, path_u, &path_pos, &d1, &d2);
        const double theta_path = std::atan2(d1.y(), d1.x());
        const double curvature = quadratic_bspline_curvature(d1, d2);

        const double v_desired = std::clamp(nominal_controls[idx](0), u_lo(0), u_hi(0));
        const double omega_desired = curvature * v_desired + heading_k * wrap_pi(theta_path - sample.xs[idx](ix::THETA));

        sample.us[idx] = ControlVec(v_desired, omega_desired).cwiseMax(u_lo).cwiseMin(u_hi);
        sample.delta_u[idx] = sample.us[idx] - nominal_controls[idx];
        running_cost += problem.running_cost(k, sample.xs[idx], sample.us[idx]);

        sample.xs[idx + 1] = problem.dynamics(k, sample.xs[idx], sample.us[idx]);
        sample.xs[idx + 1](ix::PATH_U) = advance_u_progress_on_path(sample.xs[idx](ix::PATH_U), sample.xs[idx], ref_cps);
        if (!sample.xs[idx + 1].allFinite()) {
            return sample;
        }
        if (const auto lethal = problem.detect_lethal_obstacle(k + 1, sample.xs[idx + 1])) {
            sample.lethal_obstacle = lethal;
            return sample;
        }
    }

    running_cost += problem.terminal_cost(sample.xs[MPC_HORIZON]);
    running_cost += control_regularization_cost(sample.delta_u, inv_variance, params.gamma);
    if (!std::isfinite(running_cost)) {
        return sample;
    }

    sample.cost = running_cost;
    sample.valid = true;
    return sample;
}

} // namespace

MPPIFollowSamplingResult MPPIFollowSampler::optimize(
    const FollowProblem& problem,
    const StateVec& x0,
    const std::array<ControlVec, MPC_HORIZON>& nominal_controls
) const {
    MPPIFollowSamplingResult invalid_result;
    if (!params_.enable || params_.batch_size <= 0) {
        return invalid_result;
    }

    const auto& base_ref_cps = problem.ref_control_points();
    if (base_ref_cps.size() < 3) {
        return invalid_result;
    }

    const ControlVec u_lo = problem.u_lower();
    const ControlVec u_hi = problem.u_upper();
    const ControlVec sigma = regularization_sigma_vector(params_);
    const ControlVec inv_variance(
        1.0 / (sigma(0) * sigma(0)),
        1.0 / (sigma(1) * sigma(1))
    );

    std::array<ControlVec, MPC_HORIZON> nominal = nominal_controls;
    for (int k = 0; k < MPC_HORIZON; ++k) {
        nominal[static_cast<size_t>(k)] = nominal[static_cast<size_t>(k)].cwiseMax(u_lo).cwiseMin(u_hi);
    }

    SampledSequence best_sequence = rollout_control_sequence(
        problem, x0, nominal, nominal, u_lo, u_hi, inv_variance, params_.gamma
    );
    std::vector<Eigen::Vector2d> best_path;
    if (best_sequence.valid) {
        best_path.reserve(MPC_HORIZON + 1);
        for (int k = 0; k <= MPC_HORIZON; ++k) {
            best_path.emplace_back(best_sequence.xs[static_cast<size_t>(k)](ix::X), best_sequence.xs[static_cast<size_t>(k)](ix::Y));
        }
    }

    const std::vector<Eigen::Vector2d> cp_normals = build_control_point_normals(base_ref_cps);
    std::vector<SampledSequence> samples(static_cast<size_t>(params_.batch_size));

    std::random_device rd;
    const uint64_t base_seed = (static_cast<uint64_t>(rd()) << 32U) ^ static_cast<uint64_t>(rd());

#pragma omp parallel for if(params_.batch_size >= 32) schedule(static) num_threads(params_.num_threads)
    for (int sample_index = 0; sample_index < params_.batch_size; ++sample_index) {
        std::mt19937_64 rng(make_seed(base_seed, 0, sample_index));
        const std::vector<Eigen::Vector2d> sampled_ref_cps = deform_reference_path(
            base_ref_cps, cp_normals, params_.geometry_sampling.lateral_offset_std, rng
        );
        const FollowProblem sampled_problem = problem.with_reference_path(sampled_ref_cps);

        samples[static_cast<size_t>(sample_index)] = rollout_generated_sequence(
            sampled_problem, x0, nominal, sampled_ref_cps,
            u_lo, u_hi, inv_variance, params_
        );
    }

    for (int sample_index = 0; sample_index < params_.batch_size; ++sample_index) {
        const auto& sample = samples[static_cast<size_t>(sample_index)];
        if (!sample.valid) continue;
        if (sample.cost < best_sequence.cost) {
            best_sequence = sample;
            best_path.clear();
            best_path.reserve(MPC_HORIZON + 1);
            for (int k = 0; k <= MPC_HORIZON; ++k) {
                best_path.emplace_back(best_sequence.xs[static_cast<size_t>(k)](ix::X), best_sequence.xs[static_cast<size_t>(k)](ix::Y));
            }
        }
    }

    if (!best_sequence.valid) {
        return invalid_result;
    }

    MPPIFollowSamplingResult result;
    result.xs = best_sequence.xs;
    result.us = best_sequence.us;
    result.cost = best_sequence.cost;
    result.valid = true;
    result.lethal_obstacle = best_sequence.lethal_obstacle;
    result.rollout_path = std::move(best_path);
    return result;
}

} // namespace path_follower
