#pragma once

#include <vector>

#include <mpc_tuner/types.hpp>

namespace mpc_tuner {

// 聚合一组 episode 度量为两层适应度。
//
// 第一层（硬门槛）对各类安全/成败事件计数，并累积连续伴随量，用于字典序比较。
// 第二层（软标量）对时间主导的加权代价按路线取均值，并叠加向基线的软正则：
//     soft_cost = mean_over_routes(J_route) + lambda * ||candidate.normalized - baseline.normalized||^2
// baseline 传入用于正则锚定；lambda 来自 StudyConfig。
Fitness aggregate_fitness(
    const std::vector<EpisodeMetrics>& episodes,
    const EpisodeConfig& config,
    const ParameterCandidate& candidate,
    const ParameterCandidate& baseline,
    double regularization_lambda
);

} // namespace mpc_tuner
