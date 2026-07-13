#pragma once

#include <filesystem>
#include <functional>
#include <vector>

#include <mpc_tuner/types.hpp>

namespace mpc_tuner {

using EvaluateFunction = std::function<std::pair<Fitness, std::vector<EpisodeMetrics>>(const ParameterCandidate&)>;

ParameterCandidate baseline_candidate(
    const nav_executor::MPCParams& params,
    const TunerConfig& config
);

ParameterCandidate run_cem(
    const nav_executor::MPCParams& base_params,
    const TunerConfig& config,
    const EvaluateFunction& evaluate,
    const std::filesystem::path& output_directory
);

void write_parameter_overlay(
    const ParameterCandidate& candidate,
    const std::filesystem::path& path
);

} // namespace mpc_tuner
