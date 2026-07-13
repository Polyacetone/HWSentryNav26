#include <mpc_tuner/optimizer.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

namespace mpc_tuner {
namespace {

double decode_value(const double normalized, const SearchRange& range) {
    const double t = std::clamp(normalized, 0.0, 1.0);
    if (!range.logarithmic) return range.lower + t * (range.upper - range.lower);
    return std::exp(std::log(range.lower) + t * (std::log(range.upper) - std::log(range.lower)));
}

double encode_value(const double value, const SearchRange& range) {
    if (!range.logarithmic) return std::clamp((value - range.lower) / (range.upper - range.lower), 0.0, 1.0);
    return std::clamp((std::log(value) - std::log(range.lower)) / (std::log(range.upper) - std::log(range.lower)), 0.0, 1.0);
}

ParameterCandidate decode_candidate(
    const std::array<double, PARAMETER_COUNT>& normalized,
    const TunerConfig& config
) {
    ParameterCandidate candidate;
    candidate.normalized = normalized;
    for (size_t i = 0; i < PARAMETER_COUNT; ++i) {
        candidate.values[i] = decode_value(normalized[i], config.search_ranges[i]);
    }
    return candidate;
}

std::array<double, PARAMETER_COUNT> base_values(const nav_executor::MPCParams& params) {
    const auto& t = params.follow.tracking_weights;
    const auto& c = params.follow.command_weights;
    return {t.q_y, t.q_theta, t.q_u, t.y_tube, t.q_term_prog, t.q_term_lateral,
            c.r_v, c.r_omega, c.r_dv, c.r_domega};
}

struct EvaluatedCandidate {
    ParameterCandidate candidate;
    Fitness fitness;
    std::vector<EpisodeMetrics> episodes;
};

void write_trial_header(std::ofstream& stream, const TunerConfig& config) {
    stream << "generation,index";
    for (const auto& range : config.search_ranges) stream << ',' << range.name;
    stream << ",failed,progress_deficit,severe_environment,severe_step,fast_arrival,arrival_excess,quality\n";
}

} // namespace

ParameterCandidate baseline_candidate(const nav_executor::MPCParams& params, const TunerConfig& config) {
    const auto values = base_values(params);
    std::array<double, PARAMETER_COUNT> normalized {};
    for (size_t i = 0; i < PARAMETER_COUNT; ++i) normalized[i] = encode_value(values[i], config.search_ranges[i]);
    return decode_candidate(normalized, config);
}

ParameterCandidate run_cem(
    const nav_executor::MPCParams& base_params,
    const TunerConfig& config,
    const EvaluateFunction& evaluate,
    const std::filesystem::path& output_directory,
    const ProgressFunction& report_progress
) {
    std::filesystem::create_directories(output_directory);
    std::ofstream trial_stream(output_directory / "trials.csv");
    std::ofstream episode_stream(output_directory / "episodes.csv");
    if (!trial_stream || !episode_stream) throw std::runtime_error("Failed to open tuner output files");
    write_trial_header(trial_stream, config);
    episode_stream << "generation,index,scenario,seed,reached,solver_failed,time,progress,arrival_v,arrival_w,max_cost,high_cost_integral,lethal_duration,step_speed,step_alignment,severe_step\n";

    const ParameterCandidate baseline = baseline_candidate(base_params, config);
    std::array<double, PARAMETER_COUNT> mean = baseline.normalized;
    std::array<double, PARAMETER_COUNT> stddev;
    stddev.fill(config.study.initial_std);
    std::mt19937_64 rng(config.study.seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    EvaluatedCandidate best;
    best.fitness.failed_scenarios = std::numeric_limits<int>::max();
    const int population_size = std::max(config.study.population_size, 2);
    const int elite_count = std::clamp(
        static_cast<int>(std::ceil(config.study.elite_fraction * population_size)), 1, population_size
    );
    const unsigned int hardware_workers = std::max(std::thread::hardware_concurrency(), 1U);
    const int requested_workers = config.study.parallel_workers == 0
        ? static_cast<int>(hardware_workers) : config.study.parallel_workers;
    const int worker_count = std::clamp(requested_workers, 1, population_size);
    const auto progress_interval = std::chrono::duration<double>(config.study.progress_interval_seconds);

    for (int generation = 0; generation < config.study.generations; ++generation) {
        std::vector<EvaluatedCandidate> population(static_cast<size_t>(population_size));
        for (int index = 0; index < population_size; ++index) {
            std::array<double, PARAMETER_COUNT> normalized;
            if (generation == 0 && index == 0) {
                normalized = baseline.normalized;
            } else {
                for (size_t dim = 0; dim < PARAMETER_COUNT; ++dim) {
                    normalized[dim] = std::clamp(mean[dim] + stddev[dim] * normal(rng), 0.0, 1.0);
                }
            }
            population[static_cast<size_t>(index)].candidate = decode_candidate(normalized, config);
        }

        std::atomic<int> next_index = 0;
        std::atomic<int> completed_candidates = 0;
        std::atomic<int> active_workers = 0;
        std::atomic<bool> stop = false;
        std::exception_ptr evaluation_error;
        std::mutex progress_mutex;
        std::condition_variable progress_cv;
        const auto started_at = std::chrono::steady_clock::now();

        const auto make_progress = [&] {
            return OptimizationProgress {
                .generation = generation + 1,
                .total_generations = config.study.generations,
                .completed_candidates = completed_candidates.load(),
                .population_size = population_size,
                .active_workers = active_workers.load(),
                .elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count(),
            };
        };
        report_progress(make_progress());

        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(worker_count));
        for (int worker = 0; worker < worker_count; ++worker) {
            workers.emplace_back([&] {
                while (!stop.load()) {
                    const int index = next_index.fetch_add(1);
                    if (index >= population_size) break;
                    active_workers.fetch_add(1);
                    try {
                        auto& item = population[static_cast<size_t>(index)];
                        std::tie(item.fitness, item.episodes) = evaluate(item.candidate);
                        completed_candidates.fetch_add(1);
                    } catch (...) {
                        std::lock_guard lock(progress_mutex);
                        if (!evaluation_error) evaluation_error = std::current_exception();
                        stop.store(true);
                    }
                    active_workers.fetch_sub(1);
                    progress_cv.notify_one();
                }
            });
        }

        {
            std::unique_lock lock(progress_mutex);
            while (completed_candidates.load() < population_size && !stop.load()) {
                progress_cv.wait_for(lock, progress_interval, [&] {
                    return completed_candidates.load() == population_size || stop.load();
                });
                if (completed_candidates.load() < population_size && !stop.load()) report_progress(make_progress());
            }
        }
        for (auto& worker : workers) worker.join();
        if (evaluation_error) std::rethrow_exception(evaluation_error);
        report_progress(make_progress());

        std::ranges::sort(population, [](const EvaluatedCandidate& lhs, const EvaluatedCandidate& rhs) {
            return lhs.fitness < rhs.fitness;
        });
        if (population.front().fitness < best.fitness) best = population.front();

        for (int index = 0; index < population_size; ++index) {
            const auto& item = population[static_cast<size_t>(index)];
            trial_stream << generation << ',' << index;
            for (double value : item.candidate.values) trial_stream << ',' << std::setprecision(12) << value;
            const auto& f = item.fitness;
            trial_stream << ',' << f.failed_scenarios << ',' << f.progress_deficit << ',' << f.severe_environment_violations
                << ',' << f.severe_step_violations << ',' << f.fast_arrival_count << ',' << f.arrival_speed_excess
                << ',' << f.soft_quality_cost << '\n';
            for (const auto& e : item.episodes) {
                episode_stream << generation << ',' << index << ',' << e.scenario_name << ',' << e.seed << ','
                    << e.reached << ',' << e.solver_failed << ',' << e.elapsed_time << ',' << e.final_progress << ','
                    << e.arrival_speed << ',' << e.arrival_omega << ',' << e.max_cost << ',' << e.high_cost_integral << ','
                    << e.lethal_cost_duration << ',' << e.step_speed_violation << ',' << e.step_alignment_violation << ','
                    << e.severe_step_violations << '\n';
            }
        }
        trial_stream.flush();
        episode_stream.flush();

        for (size_t dim = 0; dim < PARAMETER_COUNT; ++dim) {
            double elite_mean = 0.0;
            for (int i = 0; i < elite_count; ++i) elite_mean += population[static_cast<size_t>(i)].candidate.normalized[dim];
            elite_mean /= static_cast<double>(elite_count);
            double variance = 0.0;
            for (int i = 0; i < elite_count; ++i) {
                const double delta = population[static_cast<size_t>(i)].candidate.normalized[dim] - elite_mean;
                variance += delta * delta;
            }
            variance /= static_cast<double>(elite_count);
            mean[dim] = 0.25 * mean[dim] + 0.75 * elite_mean;
            stddev[dim] = std::max(config.study.min_std, 0.25 * stddev[dim] + 0.75 * std::sqrt(variance));
        }
    }
    write_parameter_overlay(best.candidate, output_directory / "best_params.yaml");
    return best.candidate;
}

void write_parameter_overlay(const ParameterCandidate& c, const std::filesystem::path& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write " + path.string());
    out << std::setprecision(12)
        << "/**:\n  ros__parameters:\n    mpc:\n      follow:\n        tracking_weights:\n"
        << "          q_y: " << c.values[0] << '\n'
        << "          q_theta: " << c.values[1] << '\n'
        << "          q_u: " << c.values[2] << '\n'
        << "          y_tube: " << c.values[3] << '\n'
        << "          q_term_prog: " << c.values[4] << '\n'
        << "          q_term_lateral: " << c.values[5] << '\n'
        << "        command_weights:\n"
        << "          r_v: " << c.values[6] << '\n'
        << "          r_omega: " << c.values[7] << '\n'
        << "          r_dv: " << c.values[8] << '\n'
        << "          r_domega: " << c.values[9] << '\n';
}

} // namespace mpc_tuner
