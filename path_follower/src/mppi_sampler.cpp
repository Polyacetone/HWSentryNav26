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
constexpr double WEIGHT_EPS = 1e-12;

struct SampledSequence {
    std::array<StateVec, MPC_HORIZON + 1> xs {};
    std::array<ControlVec, MPC_HORIZON> us {};
    std::array<ControlVec, MPC_HORIZON> effective_noise {};
    double cost = std::numeric_limits<double>::infinity();
    bool valid = false;
};

ControlVec sigma_vector(const MPCFollowMPPIParams& params) {
    return ControlVec(
        std::max(params.sampling_std.velocity, SIGMA_EPS),
        std::max(params.sampling_std.omega, SIGMA_EPS)
    );
}

template<size_t N>
void smooth_noise_sequence(std::array<ControlVec, N>& noise, const MPPINoiseSmoothing& smoothing) {
    const int window = std::max(1, smoothing.window);
    const int radius = window / 2;
    const int passes = std::max(0, smoothing.passes);
    if (window <= 1 || passes == 0) {
        return;
    }

    std::array<ControlVec, N> tmp {};
    for (int pass = 0; pass < passes; ++pass) {
        for (size_t k = 0; k < N; ++k) {
            ControlVec accum = ControlVec::Zero();
            int count = 0;
            const int center = static_cast<int>(k);
            const int begin = std::max(0, center - radius);
            const int end = std::min(static_cast<int>(N) - 1, center + radius);
            for (int idx = begin; idx <= end; ++idx) {
                accum += noise[static_cast<size_t>(idx)];
                ++count;
            }
            tmp[k] = accum / static_cast<double>(count);
        }
        noise = tmp;
    }
}

double control_regularization_cost(
    const std::array<ControlVec, MPC_HORIZON>& nominal_controls,
    const std::array<ControlVec, MPC_HORIZON>& effective_noise,
    const ControlVec& inv_variance,
    double gamma
) {
    double cost = 0.0;
    for (int k = 0; k < MPC_HORIZON; ++k) {
        const ControlVec scaled_noise = effective_noise[static_cast<size_t>(k)].cwiseProduct(inv_variance);
        cost += gamma * (
            0.5 * effective_noise[static_cast<size_t>(k)].dot(scaled_noise)
            + nominal_controls[static_cast<size_t>(k)].dot(scaled_noise)
        );
    }
    return cost;
}

SampledSequence rollout_sequence(
    const FollowProblem& problem,
    const StateVec& x0,
    const std::array<ControlVec, MPC_HORIZON>& nominal_controls,
    const std::array<ControlVec, MPC_HORIZON>& raw_noise,
    const ControlVec& u_lo,
    const ControlVec& u_hi,
    const ControlVec& inv_variance,
    double gamma
) {
    SampledSequence sample;
    sample.xs[0] = x0;

    double running_cost = 0.0;
    for (int k = 0; k < MPC_HORIZON; ++k) {
        const size_t idx = static_cast<size_t>(k);
        sample.us[idx] = (nominal_controls[idx] + raw_noise[idx]).cwiseMax(u_lo).cwiseMin(u_hi);
        sample.effective_noise[idx] = sample.us[idx] - nominal_controls[idx];
        running_cost += problem.running_cost(k, sample.xs[idx], sample.us[idx]);
        sample.xs[idx + 1] = problem.dynamics(k, sample.xs[idx], sample.us[idx]);
        if (!sample.xs[idx + 1].allFinite()) {
            return sample;
        }
    }

    running_cost += problem.terminal_cost(sample.xs[MPC_HORIZON]);
    running_cost += control_regularization_cost(nominal_controls, sample.effective_noise, inv_variance, gamma);

    if (!std::isfinite(running_cost)) {
        return sample;
    }

    sample.cost = running_cost;
    sample.valid = true;
    return sample;
}

MPPIFollowSamplingResult to_result(const SampledSequence& sample) {
    MPPIFollowSamplingResult result;
    result.xs = sample.xs;
    result.us = sample.us;
    result.cost = sample.cost;
    result.valid = sample.valid;
    return result;
}

uint64_t make_seed(uint64_t base_seed, int iteration, int sample_index) {
    constexpr uint64_t K0 = 0x9E3779B97F4A7C15ULL;
    constexpr uint64_t K1 = 0xBF58476D1CE4E5B9ULL;
    return base_seed ^ K0 ^ (static_cast<uint64_t>(iteration + 1) * K1) ^ static_cast<uint64_t>(sample_index + 1);
}

}

MPPIFollowSamplingResult MPPIFollowSampler::optimize(
    const FollowProblem& problem,
    const StateVec& x0,
    const std::array<ControlVec, MPC_HORIZON>& nominal_controls
) const {
    MPPIFollowSamplingResult invalid_result;
    if (!params_.enable || params_.batch_size <= 0 || params_.iteration_count <= 0 || params_.temperature <= 0.0) {
        return invalid_result;
    }

    const ControlVec u_lo = problem.u_lower();
    const ControlVec u_hi = problem.u_upper();
    const ControlVec sigma = sigma_vector(params_);
    const ControlVec inv_variance(
        1.0 / (sigma(0) * sigma(0)),
        1.0 / (sigma(1) * sigma(1))
    );

    std::array<ControlVec, MPC_HORIZON> nominal = nominal_controls;
    for (int k = 0; k < MPC_HORIZON; ++k) {
        nominal[static_cast<size_t>(k)] = nominal[static_cast<size_t>(k)].cwiseMax(u_lo).cwiseMin(u_hi);
    }

    const std::array<ControlVec, MPC_HORIZON> zero_noise = [] {
        std::array<ControlVec, MPC_HORIZON> noise {};
        for (auto& u : noise) {
            u.setZero();
        }
        return noise;
    }();

    SampledSequence best_sequence = rollout_sequence(
        problem, x0, nominal, zero_noise, u_lo, u_hi, inv_variance, params_.gamma
    );

    std::vector<SampledSequence> samples(static_cast<size_t>(params_.batch_size));
    std::random_device rd;
    const uint64_t base_seed = (static_cast<uint64_t>(rd()) << 32U) ^ static_cast<uint64_t>(rd());

    std::vector<std::vector<Eigen::Vector2d>> iteration_best_paths;

    for (int iter = 0; iter < params_.iteration_count; ++iter) {
#pragma omp parallel for if(params_.batch_size >= 32) schedule(static) num_threads(params_.num_threads)
        for (int sample_index = 0; sample_index < params_.batch_size; ++sample_index) {
            std::mt19937_64 rng(make_seed(base_seed, iter, sample_index));
            std::normal_distribution<double> vel_dist(0.0, sigma(0));
            std::normal_distribution<double> omega_dist(0.0, sigma(1));

            std::array<ControlVec, MPC_HORIZON> noise {};
            const bool use_nominal = params_.include_nominal_trajectory && sample_index == 0;
            for (int k = 0; k < MPC_HORIZON; ++k) {
                if (use_nominal) {
                    noise[static_cast<size_t>(k)].setZero();
                    continue;
                }
                noise[static_cast<size_t>(k)] = ControlVec(vel_dist(rng), omega_dist(rng));
            }

            smooth_noise_sequence(noise, params_.noise_smoothing);
            samples[static_cast<size_t>(sample_index)] = rollout_sequence(
                problem, x0, nominal, noise, u_lo, u_hi, inv_variance, params_.gamma
            );
        }

        double min_cost = std::numeric_limits<double>::infinity();
        int best_batch_index = -1;
        for (int sample_index = 0; sample_index < params_.batch_size; ++sample_index) {
            const auto& sample = samples[static_cast<size_t>(sample_index)];
            if (!sample.valid) {
                continue;
            }
            if (sample.cost < min_cost) {
                min_cost = sample.cost;
                best_batch_index = sample_index;
            }
        }

        if (best_batch_index < 0) {
            break;
        }

        const auto& best_batch_sample = samples[static_cast<size_t>(best_batch_index)];
        if (!best_sequence.valid || best_batch_sample.cost < best_sequence.cost) {
            best_sequence = best_batch_sample;
        }

        {
            std::vector<Eigen::Vector2d> path;
            path.reserve(MPC_HORIZON + 1);
            for (int k = 0; k <= MPC_HORIZON; ++k) {
                const auto& s = best_batch_sample.xs[static_cast<size_t>(k)];
                path.emplace_back(s(ix::X), s(ix::Y));
            }
            iteration_best_paths.push_back(std::move(path));
        }

        double weight_sum = 0.0;
        std::array<ControlVec, MPC_HORIZON> weighted_noise {};
        for (auto& u : weighted_noise) {
            u.setZero();
        }

        for (int sample_index = 0; sample_index < params_.batch_size; ++sample_index) {
            const auto& sample = samples[static_cast<size_t>(sample_index)];
            if (!sample.valid) {
                continue;
            }
            const double weight = std::exp(-(sample.cost - min_cost) / params_.temperature);
            weight_sum += weight;
            for (int k = 0; k < MPC_HORIZON; ++k) {
                weighted_noise[static_cast<size_t>(k)] += weight * sample.effective_noise[static_cast<size_t>(k)];
            }
        }

        if (weight_sum <= WEIGHT_EPS) {
            break;
        }

        for (int k = 0; k < MPC_HORIZON; ++k) {
            nominal[static_cast<size_t>(k)] += weighted_noise[static_cast<size_t>(k)] / weight_sum;
            nominal[static_cast<size_t>(k)] = nominal[static_cast<size_t>(k)].cwiseMax(u_lo).cwiseMin(u_hi);
        }

        const auto updated_nominal = rollout_sequence(
            problem, x0, nominal, zero_noise, u_lo, u_hi, inv_variance, params_.gamma
        );
        if (updated_nominal.valid && (!best_sequence.valid || updated_nominal.cost < best_sequence.cost)) {
            best_sequence = updated_nominal;
        }
    }

    if (!best_sequence.valid) {
        return invalid_result;
    }

    MPPIFollowSamplingResult result;
    if (!params_.fallback_to_best_sample) {
        const auto final_nominal = rollout_sequence(
            problem, x0, nominal, zero_noise, u_lo, u_hi, inv_variance, params_.gamma
        );
        if (final_nominal.valid) {
            result = to_result(final_nominal);
        } else {
            result = to_result(best_sequence);
        }
    } else {
        result = to_result(best_sequence);
    }

    result.rollout_paths = std::move(iteration_best_paths);
    return result;
}

}
