#pragma once

#include <rclcpp/logger.hpp>

#include <mpc_tuner/types.hpp>

namespace mpc_tuner {

// 在给定候选参数下对单条场景 + seed 跑一次闭环仿真，采集原始度量。
// 台阶评价采用 run-up 感知：run-up 段仅在前进时累积违规，真正跨越物理边缘 step_enter_u
// 的瞬间做达速/对齐判定。硬/软划分由 fitness 层完成，此处只记录物理量。
EpisodeMetrics run_episode(
    const CompiledScenario& scenario,
    uint64_t seed,
    const RuntimeConfig& runtime,
    const EpisodeConfig& config,
    const ParameterCandidate& candidate,
    rclcpp::Logger logger
);

} // namespace mpc_tuner
