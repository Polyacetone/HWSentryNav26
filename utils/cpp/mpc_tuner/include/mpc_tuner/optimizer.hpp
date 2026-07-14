#pragma once

#include <filesystem>
#include <functional>
#include <vector>

#include <mpc_tuner/types.hpp>

namespace mpc_tuner {

using EvaluateFunction = std::function<std::pair<Fitness, std::vector<EpisodeMetrics>>(const ParameterCandidate&)>;

struct OptimizationProgress {
    int generation = 0;
    int total_generations = 0;
    int completed_candidates = 0;
    int population_size = 0;
    int active_workers = 0;
    double elapsed_seconds = 0.0;
};

using ProgressFunction = std::function<void(const OptimizationProgress&)>;

ParameterCandidate run_cem(
    const nav_executor::MPCParams& base_params,
    const TunerConfig& config,
    const EvaluateFunction& evaluate,
    const std::filesystem::path& output_directory,
    const ProgressFunction& report_progress
);

} // namespace mpc_tuner
