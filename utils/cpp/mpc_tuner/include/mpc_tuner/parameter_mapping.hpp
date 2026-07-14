#pragma once

#include <array>
#include <filesystem>

#include <mpc_tuner/types.hpp>

namespace mpc_tuner {

// 归一化 [0,1] ↔ 物理量的单值编解码（依据 SearchRange 的对数/线性设定）。
double decode_value(double normalized, const SearchRange& range);
double encode_value(double value, const SearchRange& range);

// 由归一化向量构造完整候选（同时填充物理值）。
ParameterCandidate decode_candidate(
    const std::array<double, PARAMETER_COUNT>& normalized,
    const TunerConfig& config
);

// 基线候选：从 base_params 按描述表读取各参数初值，编码到归一化空间。
// 这是 CEM 的起始均值，也是正则项锚定的中心。
ParameterCandidate baseline_candidate(
    const nav_executor::MPCParams& base_params,
    const TunerConfig& config
);

// 把候选的物理值写入 base 的副本并返回。始终作用在基线副本上，保证共享乘子幂等。
nav_executor::MPCParams apply_candidate(
    const nav_executor::MPCParams& base,
    const ParameterCandidate& candidate
);

// 以 ROS 参数覆盖文件形式导出候选（仅含被调参数，供 mpc.yaml 叠加）。
// 需要 base 以正确结算共享乘子的最终权重值。
void write_parameter_overlay(
    const nav_executor::MPCParams& base,
    const ParameterCandidate& candidate,
    const std::filesystem::path& path
);

} // namespace mpc_tuner
