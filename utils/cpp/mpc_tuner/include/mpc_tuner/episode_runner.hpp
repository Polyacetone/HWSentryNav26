#pragma once

#include <rclcpp/logger.hpp>

#include <mpc_tuner/types.hpp>

namespace mpc_tuner {

nav_executor::MPCParams apply_candidate(
    const nav_executor::MPCParams& base,
    const ParameterCandidate& candidate
);

EpisodeMetrics run_episode(
    const CompiledScenario& scenario,
    uint64_t seed,
    const RuntimeConfig& runtime,
    const EpisodeConfig& config,
    const ParameterCandidate& candidate,
    rclcpp::Logger logger
);

Fitness aggregate_fitness(const std::vector<EpisodeMetrics>& episodes, const EpisodeConfig& config);

} // namespace mpc_tuner
