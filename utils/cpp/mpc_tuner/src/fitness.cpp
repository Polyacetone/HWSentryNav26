#include <mpc_tuner/fitness.hpp>

#include <algorithm>
#include <cmath>

namespace mpc_tuner {
namespace {

// 软膝盖：x / (x + scale)，把无上界的正量压到 [0,1) 且保留全程梯度（不硬截断）。
double soft_knee(const double value, const double scale) {
    const double s = std::max(scale, 1e-9);
    return value / (value + s);
}

// 线性归一：value / scale，允许 > 1（时间/偏离等尺度项不饱和，CEM 全程可见梯度）。
double normalized(const double value, const double scale) {
    return value / std::max(scale, 1e-9);
}

// 单条 episode 的软标量代价（未含正则、未跨路线聚合）。
double episode_soft_cost(const EpisodeMetrics& m, const EpisodeConfig& config) {
    const SoftScales& scales = config.soft_scales;
    const SoftWeights& weights = config.soft_weights;

    // 时间：每路线参考时间 t_ref = 弧长 / v_ref，使长短路线的提速贡献可比。
    const double t_ref = std::max(m.path_length / scales.reference_speed, 1e-3);
    const double time_term = normalized(m.elapsed_time, t_ref);

    // 到达剩余速度：超过 target 的部分，按 [target, acceptable] 带宽归一。
    const double arrival_excess = std::max(0.0, m.arrival_speed - config.target_arrival_speed);
    const double arrival_term = normalized(arrival_excess, scales.arrival_speed_band);

    // 经过代价：靠近障碍但未致命，用软膝盖压有界。
    const double high_cost_term = soft_knee(m.high_cost_integral, scales.high_cost_integral);

    // 台阶轻微违规：软区正好衔接硬门槛阈值（严重部分已进第一层，此处仅计未达严重的积分）。
    const double step_speed_term = normalized(m.minor_step_speed_violation, config.severe_step_speed_margin);
    const double step_heading_term = normalized(m.minor_step_heading_violation, config.severe_step_heading_error);

    // 横向偏离 RMS：按管廊半宽归一（管廊内 MPC 本就不罚）。
    const double duration = std::max(m.elapsed_time, 1e-3);
    const double cross_track_rms = std::sqrt(m.cross_track_squared_integral / duration);
    const double cross_track_term = normalized(cross_track_rms, scales.cross_track);

    // 平滑度地板：仅惩罚超出物理设计加速度的抖动积分（阈值内零成本，不拖慢）。
    const double smoothness_term = soft_knee(m.smoothness_excess_integral, 1.0);

    return weights.time * time_term
        + weights.high_cost * high_cost_term
        + weights.arrival_speed * arrival_term
        + weights.step_speed_minor * step_speed_term
        + weights.step_heading_minor * step_heading_term
        + weights.cross_track * cross_track_term
        + weights.smoothness * smoothness_term;
}

double regularization_penalty(
    const ParameterCandidate& candidate,
    const ParameterCandidate& baseline,
    const double lambda
) {
    if (lambda <= 0.0) return 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < PARAMETER_COUNT; ++i) {
        const double delta = candidate.normalized[i] - baseline.normalized[i];
        sum_sq += delta * delta;
    }
    return lambda * sum_sq;
}

} // namespace

Fitness aggregate_fitness(
    const std::vector<EpisodeMetrics>& episodes,
    const EpisodeConfig& config,
    const ParameterCandidate& candidate,
    const ParameterCandidate& baseline,
    const double regularization_lambda
) {
    Fitness fitness;
    if (episodes.empty()) {
        // 无可评估场景视为完全失败，落在硬门槛最深处。
        fitness.failed_scenarios = 1;
        fitness.progress_deficit = 1.0;
        fitness.soft_cost = std::numeric_limits<double>::infinity();
        return fitness;
    }

    double soft_sum = 0.0;
    for (const EpisodeMetrics& m : episodes) {
        // ── 第一层：硬门槛 ──
        if (m.hazard_duration > 1e-6) {
            ++fitness.hazard_events;
            fitness.hazard_duration += m.hazard_duration;
        }
        fitness.severe_step_speed_events += m.severe_step_speed_events;
        fitness.severe_step_speed_excess += m.severe_step_speed_excess;
        fitness.severe_step_heading_events += m.severe_step_heading_events;
        fitness.severe_step_heading_excess += m.severe_step_heading_excess;
        if (!m.reached || m.solver_failed) {
            ++fitness.failed_scenarios;
            fitness.progress_deficit += 1.0 - std::clamp(m.final_progress, 0.0, 1.0);
        }

        // ── 第二层：软标量 ──
        soft_sum += episode_soft_cost(m, config);
    }

    const double count = static_cast<double>(episodes.size());
    fitness.progress_deficit /= count;
    fitness.soft_cost = soft_sum / count + regularization_penalty(candidate, baseline, regularization_lambda);
    return fitness;
}

} // namespace mpc_tuner
