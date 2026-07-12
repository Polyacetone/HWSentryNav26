#include <nav_executor/path_executor/solver/cem_optimizer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <Eigen/Core>
#include <limits>
#include <random>
#include <vector>

namespace nav_executor {
namespace {

constexpr double SIGMA_EPS = 1e-6;
struct SampledSequence {
    std::array<StateVec, MPC_HORIZON + 1> xs {};
    std::array<ControlVec, MPC_HORIZON> us {};
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

ControlVec sigma_vector(const GlobalSearchParams& params) {
    return ControlVec(
        std::max(params.sampling_std.velocity, SIGMA_EPS),
        std::max(params.sampling_std.omega, SIGMA_EPS)
    );
}

int effective_smoothing_window(const GlobalSearchNoiseSmoothing& smoothing) {
    return 2 * std::max(0, smoothing.window / 2) + 1;
}

FIRKernel build_fir_kernel(const GlobalSearchNoiseSmoothing& smoothing) {
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

/// smooth_noise_1d: apply FIR kernel to a 1-D horizon-length array (no barrier, fully local)
void smooth_noise_1d(
    const std::array<double, MPC_HORIZON>& raw,
    std::array<double, MPC_HORIZON>& smoothed,
    const FIRKernel& kernel
) {
    if (!kernel.enabled()) {
        smoothed = raw;
        return;
    }
    for (int k = 0; k < MPC_HORIZON; ++k) {
        double accum = 0.0;
        double weight_sum = 0.0;
        for (int offset = -kernel.radius; offset <= kernel.radius; ++offset) {
            const int src = k + offset;
            if (src < 0 || src >= MPC_HORIZON) continue;
            const double w = kernel.weights[static_cast<size_t>(offset + kernel.radius)];
            accum += w * raw[static_cast<size_t>(src)];
            weight_sum += w;
        }
        smoothed[static_cast<size_t>(k)] = (weight_sum > 0.0) ? (accum / weight_sum) : raw[static_cast<size_t>(k)];
    }
}

/// rollout_sequence_v2: rollout with
///   (a) early termination when cumulative cost exceeds cutoff,
///   (b) cached cost bilinear sample from detect_lethal_obstacle → next step's cost.
template<typename NoiseGetter>
void rollout_sequence_v2(
    const FollowProblem& problem,
    const StateVec& x0,
    const std::array<ControlVec, MPC_HORIZON>& nominal_controls,
    NoiseGetter&& noise_getter,
    const ControlVec& u_lo,
    const ControlVec& u_hi,
    SampledSequence& sample,
    double early_cutoff
) {
    sample.cost = std::numeric_limits<double>::infinity();
    sample.valid = false;
    sample.xs[0] = x0;

    double cost = 0.0;
    double cached_cost_value = -1.0;   // invalid — no cached cost at step 0

    for (int k = 0; k < MPC_HORIZON; ++k) {
        const size_t idx = static_cast<size_t>(k);
        sample.us[idx] = (nominal_controls[idx] + noise_getter(k)).cwiseMax(u_lo).cwiseMin(u_hi);

        // Use cached cost value from last step's lethal detection (at same state)
        cost += problem.running_cost_value_only(k, sample.xs[idx], sample.us[idx], &cached_cost_value);

        // Early termination: if cumulative cost already exceeds best known, discard this sample
        if (cost >= early_cutoff) {
            return;
        }

        sample.xs[idx + 1] = problem.dynamics(k, sample.xs[idx], sample.us[idx]);
        if (!sample.xs[idx + 1].allFinite()) {
            return;
        }

        // Lethal check — also outputs cost value for next step's running_cost_value_only
        if (problem.detect_lethal_obstacle(k + 1, sample.xs[idx + 1], &cached_cost_value).has_value()) {
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

void rollout_nominal_sequence_v2(
    const FollowProblem& problem,
    const StateVec& x0,
    const std::array<ControlVec, MPC_HORIZON>& nominal_controls,
    const ControlVec& u_lo,
    const ControlVec& u_hi,
    SampledSequence& sample,
    double early_cutoff
) {
    rollout_sequence_v2(
        problem, x0, nominal_controls,
        [](int) { return ControlVec::Zero(); },
        u_lo, u_hi, sample, early_cutoff
    );
}

CEMOptimizationResult to_result(const SampledSequence& sample) {
    CEMOptimizationResult result;
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

CEMOptimizationResult CEMOptimizer::optimize(
    const FollowProblem& problem,
    const StateVec& x0,
    const std::array<ControlVec, MPC_HORIZON>& nominal_controls
) const {
    CEMOptimizationResult invalid_result;
    if (!params_.enable || params_.batch_size <= 0 || params_.iteration_count <= 0
        || params_.elite_fraction <= 0.0 || params_.elite_fraction > 1.0) {
        return invalid_result;
    }

    const ControlVec u_lo = problem.u_lower();
    const ControlVec u_hi = problem.u_upper();
    const ControlVec sigma = sigma_vector(params_);
    const double sigma_v = sigma(0);
    const double sigma_w = sigma(1);
    const FIRKernel fir_kernel = build_fir_kernel(params_.noise_smoothing);

    std::array<ControlVec, MPC_HORIZON> nominal = nominal_controls;
    for (int k = 0; k < MPC_HORIZON; ++k) {
        nominal[static_cast<size_t>(k)] = nominal[static_cast<size_t>(k)].cwiseMax(u_lo).cwiseMin(u_hi);
    }

    SampledSequence best_sequence;
    rollout_nominal_sequence_v2(
        problem, x0, nominal, u_lo, u_hi, best_sequence,
        std::numeric_limits<double>::infinity()
    );

    std::vector<SampledSequence> samples(static_cast<size_t>(params_.batch_size));
    std::random_device rd;
    const uint64_t base_seed = (static_cast<uint64_t>(rd()) << 32U) ^ static_cast<uint64_t>(rd());

    for (int iter = 0; iter < params_.iteration_count; ++iter) {
        const double early_cutoff = best_sequence.valid
            ? best_sequence.cost
            : std::numeric_limits<double>::infinity();

        // ── Merged parallel loop: noise gen + smoothing + rollout (optimization 4) ──
#pragma omp parallel for schedule(static) \
    if(params_.batch_size >= 32) num_threads(params_.num_threads)
        for (int sample_index = 0; sample_index < params_.batch_size; ++sample_index) {
            auto& rng = thread_local_rng(base_seed);
            std::normal_distribution<double> vel_dist(0.0, sigma_v);
            std::normal_distribution<double> omega_dist(0.0, sigma_w);

            // Thread-local noise arrays — no shared memory, no false sharing
            std::array<double, MPC_HORIZON> raw_vel{};
            std::array<double, MPC_HORIZON> raw_omega{};
            std::array<double, MPC_HORIZON> smoothed_vel{};
            std::array<double, MPC_HORIZON> smoothed_omega{};

            const bool use_nominal = params_.include_nominal_trajectory && sample_index == 0;

            // 1) Noise generation
            if (!use_nominal) {
                for (int k = 0; k < MPC_HORIZON; ++k) {
                    raw_vel[static_cast<size_t>(k)] = vel_dist(rng);
                    raw_omega[static_cast<size_t>(k)] = omega_dist(rng);
                }
            }

            // 2) Noise smoothing (local arrays, zero-copy when disabled)
            if (use_nominal) {
                smoothed_vel.fill(0.0);
                smoothed_omega.fill(0.0);
            } else if (fir_kernel.enabled()) {
                smooth_noise_1d(raw_vel, smoothed_vel, fir_kernel);
                smooth_noise_1d(raw_omega, smoothed_omega, fir_kernel);
            } else {
                smoothed_vel = raw_vel;
                smoothed_omega = raw_omega;
            }

            // 3) Rollout with early termination + cost caching (optimizations 1, 2)
            rollout_sequence_v2(
                problem, x0, nominal,
                [&](int k) {
                    return ControlVec(
                        smoothed_vel[static_cast<size_t>(k)],
                        smoothed_omega[static_cast<size_t>(k)]
                    );
                },
                u_lo, u_hi,
                samples[static_cast<size_t>(sample_index)],
                early_cutoff
            );
        }

        // ── Serial reduction ──
        double min_cost = std::numeric_limits<double>::infinity();
        int best_batch_index = -1;
        for (int sample_index = 0; sample_index < params_.batch_size; ++sample_index) {
            const auto& sample = samples[static_cast<size_t>(sample_index)];
            if (!sample.valid) continue;
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

        std::vector<const SampledSequence*> elites;
        elites.reserve(samples.size());
        for (int sample_index = 0; sample_index < params_.batch_size; ++sample_index) {
            const auto& sample = samples[static_cast<size_t>(sample_index)];
            if (sample.valid) elites.push_back(&sample);
        }
        std::sort(elites.begin(), elites.end(), [](const auto* lhs, const auto* rhs) {
            return lhs->cost < rhs->cost;
        });
        const size_t elite_count = std::max<size_t>(
            1, static_cast<size_t>(std::ceil(params_.elite_fraction * static_cast<double>(elites.size())))
        );
        for (int k = 0; k < MPC_HORIZON; ++k) {
            nominal[static_cast<size_t>(k)].setZero();
            for (size_t elite_index = 0; elite_index < elite_count; ++elite_index) {
                nominal[static_cast<size_t>(k)] += elites[elite_index]->us[static_cast<size_t>(k)];
            }
            nominal[static_cast<size_t>(k)] /= static_cast<double>(elite_count);
            nominal[static_cast<size_t>(k)] = nominal[static_cast<size_t>(k)].cwiseMax(u_lo).cwiseMin(u_hi);
        }
    }

    if (!best_sequence.valid) {
        return invalid_result;
    }

    SampledSequence final_nominal;
    rollout_nominal_sequence_v2(
        problem, x0, nominal, u_lo, u_hi, final_nominal,
        std::numeric_limits<double>::infinity()
    );
    if (final_nominal.valid && final_nominal.cost < best_sequence.cost) {
        best_sequence = std::move(final_nominal);
    }
    CEMOptimizationResult result = to_result(best_sequence);
    return result;
}

}
