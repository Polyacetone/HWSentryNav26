#include <path_follower/mppi_sampler.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <Eigen/Core>
#include <limits>
#include <random>
#include <vector>

namespace path_follower {
namespace {

constexpr double SIGMA_EPS = 1e-6;
constexpr double WEIGHT_EPS = 1e-12;

using NoiseBatch = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct SampledSequence {
    std::array<StateVec, MPC_HORIZON + 1> xs {};
    std::array<ControlVec, MPC_HORIZON> us {};
    std::array<ControlVec, MPC_HORIZON> effective_noise {};
    double cost = std::numeric_limits<double>::infinity();
    bool valid = false;
};

struct FIRKernel {
    std::vector<double> weights;
    int radius = 0;

    [[nodiscard]] bool enabled() const {
        return !weights.empty() && radius > 0;
    }
};

ControlVec sigma_vector(const MPCFollowMPPIParams& params) {
    return ControlVec(
        std::max(params.sampling_std.velocity, SIGMA_EPS),
        std::max(params.sampling_std.omega, SIGMA_EPS)
    );
}

int effective_smoothing_window(const MPPINoiseSmoothing& smoothing) {
    return 2 * std::max(0, smoothing.window / 2) + 1;
}

FIRKernel build_fir_kernel(const MPPINoiseSmoothing& smoothing) {
    const int window = effective_smoothing_window(smoothing);
    const int passes = std::max(0, smoothing.passes);
    if (window <= 1 || passes == 0) {
        return {};
    }

    std::vector<double> kernel(1, 1.0);
    const std::vector<double> box_kernel(static_cast<size_t>(window), 1.0 / static_cast<double>(window));

    for (int pass = 0; pass < passes; ++pass) {
        std::vector<double> next(kernel.size() + box_kernel.size() - 1, 0.0);
        for (size_t i = 0; i < kernel.size(); ++i) {
            for (size_t j = 0; j < box_kernel.size(); ++j) {
                next[i + j] += kernel[i] * box_kernel[j];
            }
        }
        kernel = std::move(next);
    }

    return FIRKernel {
        .weights = std::move(kernel),
        .radius = static_cast<int>(kernel.size() / 2),
    };
}

void smooth_noise_batch_row(const NoiseBatch& raw_batch, NoiseBatch& smoothed_batch, int row, const FIRKernel& kernel) {
    if (!kernel.enabled()) {
        smoothed_batch.row(row) = raw_batch.row(row);
        return;
    }

    constexpr int horizon = MPC_HORIZON;
    for (int k = 0; k < horizon; ++k) {
        double accum = 0.0;
        double weight_sum = 0.0;
        for (int offset = -kernel.radius; offset <= kernel.radius; ++offset) {
            const int src = k + offset;
            if (src < 0 || src >= horizon) {
                continue;
            }
            const double weight = kernel.weights[static_cast<size_t>(offset + kernel.radius)];
            accum += weight * raw_batch(row, src);
            weight_sum += weight;
        }
        smoothed_batch(row, k) = (weight_sum > 0.0) ? (accum / weight_sum) : raw_batch(row, k);
    }
}

template<typename NoiseGetter>
void rollout_sequence(
    const FollowProblem& problem,
    const StateVec& x0,
    const std::array<ControlVec, MPC_HORIZON>& nominal_controls,
    NoiseGetter&& noise_getter,
    const ControlVec& u_lo,
    const ControlVec& u_hi,
    SampledSequence& sample
) {
    sample.cost = std::numeric_limits<double>::infinity();
    sample.valid = false;
    sample.xs[0] = x0;

    double cost = 0.0;
    for (int k = 0; k < MPC_HORIZON; ++k) {
        const size_t idx = static_cast<size_t>(k);
        sample.us[idx] = (nominal_controls[idx] + noise_getter(k)).cwiseMax(u_lo).cwiseMin(u_hi);
        sample.effective_noise[idx] = sample.us[idx] - nominal_controls[idx];
        cost += problem.running_cost_value_only(k, sample.xs[idx], sample.us[idx]);
        sample.xs[idx + 1] = problem.dynamics(k, sample.xs[idx], sample.us[idx]);
        if (!sample.xs[idx + 1].allFinite()) {
            return;
        }
        if (problem.detect_lethal_obstacle(k + 1, sample.xs[idx + 1]).has_value()) {
            return;
        }
    }

    cost += problem.terminal_cost(sample.xs[MPC_HORIZON]);

    if (!std::isfinite(cost)) {
        return;
    }

    sample.cost = cost;
    sample.valid = true;
}

void rollout_nominal_sequence(
    const FollowProblem& problem,
    const StateVec& x0,
    const std::array<ControlVec, MPC_HORIZON>& nominal_controls,
    const ControlVec& u_lo,
    const ControlVec& u_hi,
    SampledSequence& sample
) {
    rollout_sequence(
        problem,
        x0,
        nominal_controls,
        [](int) { return ControlVec::Zero(); },
        u_lo,
        u_hi,
        sample
    );
}

void rollout_sequence_from_batches(
    const FollowProblem& problem,
    const StateVec& x0,
    const std::array<ControlVec, MPC_HORIZON>& nominal_controls,
    const NoiseBatch& vel_noise_batch,
    const NoiseBatch& omega_noise_batch,
    int sample_index,
    const ControlVec& u_lo,
    const ControlVec& u_hi,
    SampledSequence& sample
) {
    rollout_sequence(
        problem,
        x0,
        nominal_controls,
        [&](int k) {
            return ControlVec(vel_noise_batch(sample_index, k), omega_noise_batch(sample_index, k));
        },
        u_lo,
        u_hi,
        sample
    );
}

MPPIFollowSamplingResult to_result(const SampledSequence& sample) {
    MPPIFollowSamplingResult result;
    result.xs = sample.xs;
    result.us = sample.us;
    result.cost = sample.cost;
    result.valid = sample.valid;
    return result;
}

std::mt19937_64& thread_local_rng(uint64_t base_seed) {
    thread_local std::mt19937_64 rng;
    thread_local bool initialized = false;
    if (!initialized) {
        const uint64_t seed = base_seed ^ 0x9E3779B97F4A7C15ULL
            ^ (static_cast<uint64_t>(omp_get_thread_num() + 1) * 0xBF58476D1CE4E5B9ULL);
        rng.seed(seed);
        initialized = true;
    }
    return rng;
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
    const FIRKernel fir_kernel = build_fir_kernel(params_.noise_smoothing);

    std::array<ControlVec, MPC_HORIZON> nominal = nominal_controls;
    for (int k = 0; k < MPC_HORIZON; ++k) {
        nominal[static_cast<size_t>(k)] = nominal[static_cast<size_t>(k)].cwiseMax(u_lo).cwiseMin(u_hi);
    }

    SampledSequence best_sequence;
    rollout_nominal_sequence(problem, x0, nominal, u_lo, u_hi, best_sequence);

    std::vector<SampledSequence> samples(static_cast<size_t>(params_.batch_size));
    NoiseBatch raw_vel_noise(params_.batch_size, MPC_HORIZON);
    NoiseBatch raw_omega_noise(params_.batch_size, MPC_HORIZON);
    NoiseBatch smoothed_vel_noise(params_.batch_size, MPC_HORIZON);
    NoiseBatch smoothed_omega_noise(params_.batch_size, MPC_HORIZON);
    std::random_device rd;
    const uint64_t base_seed = (static_cast<uint64_t>(rd()) << 32U) ^ static_cast<uint64_t>(rd());

    std::vector<std::vector<Eigen::Vector2d>> iteration_best_paths;
    iteration_best_paths.reserve(static_cast<size_t>(params_.iteration_count));

        for (int iter = 0; iter < params_.iteration_count; ++iter) {
#pragma omp parallel if(params_.batch_size >= 32) num_threads(params_.num_threads)
        {
            auto& rng = thread_local_rng(base_seed);
            std::normal_distribution<double> vel_dist(0.0, sigma(0));
            std::normal_distribution<double> omega_dist(0.0, sigma(1));

#pragma omp for schedule(static)
            for (int sample_index = 0; sample_index < params_.batch_size; ++sample_index) {
                const bool use_nominal = params_.include_nominal_trajectory && sample_index == 0;
                if (use_nominal) {
                    raw_vel_noise.row(sample_index).setZero();
                    raw_omega_noise.row(sample_index).setZero();
                    continue;
                }
                for (int k = 0; k < MPC_HORIZON; ++k) {
                    raw_vel_noise(sample_index, k) = vel_dist(rng);
                    raw_omega_noise(sample_index, k) = omega_dist(rng);
                }
            }

#pragma omp for schedule(static)
            for (int sample_index = 0; sample_index < params_.batch_size; ++sample_index) {
                smooth_noise_batch_row(raw_vel_noise, smoothed_vel_noise, sample_index, fir_kernel);
                smooth_noise_batch_row(raw_omega_noise, smoothed_omega_noise, sample_index, fir_kernel);
            }

#pragma omp for schedule(static)
            for (int sample_index = 0; sample_index < params_.batch_size; ++sample_index) {
                rollout_sequence_from_batches(
                    problem,
                    x0,
                    nominal,
                    smoothed_vel_noise,
                    smoothed_omega_noise,
                    sample_index,
                    u_lo,
                    u_hi,
                    samples[static_cast<size_t>(sample_index)]
                );
            }
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

        iteration_best_paths.emplace_back();
        auto& best_path = iteration_best_paths.back();
        best_path.resize(MPC_HORIZON + 1);
        for (int k = 0; k <= MPC_HORIZON; ++k) {
            const auto& s = best_batch_sample.xs[static_cast<size_t>(k)];
            best_path[static_cast<size_t>(k)] = Eigen::Vector2d(s(ix::X), s(ix::Y));
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
    }

    if (!best_sequence.valid) {
        return invalid_result;
    }

    MPPIFollowSamplingResult result;
    if (!params_.fallback_to_best_sample) {
        SampledSequence final_nominal;
        rollout_nominal_sequence(problem, x0, nominal, u_lo, u_hi, final_nominal);
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
