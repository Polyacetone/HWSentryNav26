#include <mpc_tuner/parameter_mapping.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace mpc_tuner {

double decode_value(const double normalized, const SearchRange& range) {
    const double t = std::clamp(normalized, 0.0, 1.0);
    if (!range.logarithmic) return range.lower + t * (range.upper - range.lower);
    return std::exp(std::log(range.lower) + t * (std::log(range.upper) - std::log(range.lower)));
}

double encode_value(const double value, const SearchRange& range) {
    if (!range.logarithmic) {
        return std::clamp((value - range.lower) / (range.upper - range.lower), 0.0, 1.0);
    }
    return std::clamp(
        (std::log(value) - std::log(range.lower)) / (std::log(range.upper) - std::log(range.lower)),
        0.0, 1.0
    );
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

ParameterCandidate baseline_candidate(const nav_executor::MPCParams& base_params, const TunerConfig& config) {
    std::array<double, PARAMETER_COUNT> normalized {};
    for (size_t i = 0; i < PARAMETER_COUNT; ++i) {
        const double value = PARAMETER_DESCRIPTORS[i].read_baseline(base_params);
        normalized[i] = encode_value(value, config.search_ranges[i]);
    }
    return decode_candidate(normalized, config);
}

nav_executor::MPCParams apply_candidate(
    const nav_executor::MPCParams& base,
    const ParameterCandidate& candidate
) {
    nav_executor::MPCParams out = base;
    for (size_t i = 0; i < PARAMETER_COUNT; ++i) {
        PARAMETER_DESCRIPTORS[i].apply(out, candidate.values[i]);
    }
    // 调参在裸 FDDP follow 求解器上进行：全局搜索挂在墙钟时间上、致命检测会打断 rollout，
    // 二者都无法与仿真时间快进忠实同步，故关闭。相关权重仍作为 rollout 打分代价被间接评估。
    out.follow.global_search.enable = false;
    out.follow.rollout_safety.enable_lethal_obstacle_check = false;
    return out;
}

void write_parameter_overlay(
    const nav_executor::MPCParams& base,
    const ParameterCandidate& candidate,
    const std::filesystem::path& path
) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write " + path.string());

    // 覆盖文件反映“应用候选后的真实权重”：在基线副本上结算，共享乘子的缩放效果已并入具体字段。
    const nav_executor::MPCParams resolved = apply_candidate(base, candidate);
    const auto& f = resolved.follow;

    out << std::setprecision(12)
        << "/**:\n  ros__parameters:\n    mpc:\n      follow:\n"
        << "        tracking_weights:\n"
        << "          q_y: " << f.tracking_weights.q_y << '\n'
        << "          q_theta: " << f.tracking_weights.q_theta << '\n'
        << "          q_u: " << f.tracking_weights.q_u << '\n'
        << "          y_tube: " << f.tracking_weights.y_tube << '\n'
        << "          q_term_prog: " << f.tracking_weights.q_term_prog << '\n'
        << "          q_term_lateral: " << f.tracking_weights.q_term_lateral << '\n'
        << "        command_weights:\n"
        << "          r_v: " << f.command_weights.r_v << '\n'
        << "          r_omega: " << f.command_weights.r_omega << '\n'
        << "          r_dv: " << f.command_weights.r_dv << '\n'
        << "          r_domega: " << f.command_weights.r_domega << '\n'
        << "        environment_weights:\n"
        << "          obstacle: " << f.environment_weights.obstacle << '\n'
        << "        terrain_weights:\n"
        << "          approach:\n"
        << "            reachability_lo: " << f.terrain_weights.step_reachability_lo << '\n'
        << "            reachability_hi: " << f.terrain_weights.step_reachability_hi << '\n'
        << "          internal:\n"
        << "            velocity_window: " << f.terrain_weights.step_vel_weight << '\n'
        << "            direction: " << f.terrain_weights.direction << '\n'
        << "            omega: " << f.terrain_weights.step_omega << '\n'
        << "            velocity_smooth: " << f.terrain_weights.step_dv << '\n'
        << "            omega_smooth: " << f.terrain_weights.step_domega << '\n'
        << "        motion_constraint_weights:\n"
        << "          acc_limit: " << f.motion_constraint_weights.acc_limit << '\n'
        << "          alpha_limit: " << f.motion_constraint_weights.alpha_limit << '\n'
        << "          lat_acc: " << f.motion_constraint_weights.lat_acc << '\n'
        << "        terminal_weights:\n"
        << "          q_v_final: " << f.terminal_weights.q_v_final << '\n';
}

} // namespace mpc_tuner
