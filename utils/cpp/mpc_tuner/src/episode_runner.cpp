#include <mpc_tuner/episode_runner.hpp>

#include <algorithm>
#include <cmath>

#include <nav_executor/path_executor/solver/mpc_solver.hpp>

#include <mpc_tuner/parameter_mapping.hpp>
#include <mpc_tuner/wheel_leg_plant.hpp>

namespace mpc_tuner {
namespace {

constexpr double SUBSTEP_DT = 0.001; // 与 WheelLegPlant::DT 一致，每个 plant 子步的时长

// run-up 感知的台阶评价器。
//
// 语义（对应设计决策 Q1.A + Q2.A）：
//   - run-up 段（约束窗内、尚未跨越物理边缘）只在“前进”时累积轻微违规到软标量；
//     后退助跑（Δu < 0）一律豁免，因此“先退再冲”不再被误判为违规。
//   - 真正前进跨越物理边缘 step_enter_u 的那一刻做严重性判定：速度/航向超出严重阈值
//     记入硬门槛（跨越事件），否则仅由 run-up 段的软积分体现。这天然回答了“两次进入
//     算哪次”——只有真正把 path_u 推过边缘的那次前进才触发跨越判定。
class StepMonitor {
public:
    StepMonitor(const nav_executor::StepConstraintSchedule& schedule, const EpisodeConfig& config, double initial_u)
        : schedule_(schedule), config_(config), previous_u_(initial_u) {}

    void update(const double metric_u, const PlantSample& sample, EpisodeMetrics& metrics) {
        const nav_executor::StepTraversalConstraint* const step = schedule_.constraint_at(metric_u);
        const bool moving_forward = metric_u - previous_u_ > config_.forward_progress_epsilon;

        if (step && moving_forward) {
            accumulate_run_up_violation(*step, sample, metrics);
            judge_crossing(*step, metric_u, sample, metrics);
        }
        previous_u_ = metric_u;
    }

private:
    // 前进段的连续轻微违规积分（软标量）。度量“穿越窗内挣扎了多少”，与跨越瞬间的硬判定正交。
    void accumulate_run_up_violation(
        const nav_executor::StepTraversalConstraint& step,
        const PlantSample& sample,
        EpisodeMetrics& metrics
    ) const {
        const double speed = sample.chassis.velocity;
        const double under = std::max(0.0, step.speed_min - speed);
        const double over = std::max(0.0, speed - step.speed_max);
        metrics.minor_step_speed_violation += (under + over) * SUBSTEP_DT;

        const double angle_error = heading_error(step, sample);
        metrics.minor_step_heading_violation += angle_error * SUBSTEP_DT;
    }

    // 前进跨越物理边缘 step_enter_u 的瞬间：严重超限记入硬门槛伴随量。
    void judge_crossing(
        const nav_executor::StepTraversalConstraint& step,
        const double metric_u,
        const PlantSample& sample,
        EpisodeMetrics& metrics
    ) const {
        const bool crossed_forward = previous_u_ < step.step_enter_u && step.step_enter_u <= metric_u;
        if (!crossed_forward) return;

        const double speed = sample.chassis.velocity;
        const double under = std::max(0.0, step.speed_min - speed);
        const double over = std::max(0.0, speed - step.speed_max);
        const double speed_excess = std::max(under, over);
        if (speed_excess > config_.severe_step_speed_margin) {
            ++metrics.severe_step_speed_events;
            metrics.severe_step_speed_excess += speed_excess;
        }

        const double angle_error = heading_error(step, sample);
        if (angle_error > config_.severe_step_heading_error) {
            ++metrics.severe_step_heading_events;
            metrics.severe_step_heading_excess += angle_error;
        }
    }

    static double heading_error(const nav_executor::StepTraversalConstraint& step, const PlantSample& sample) {
        const Eigen::Vector2d heading(std::cos(sample.pose.z()), std::sin(sample.pose.z()));
        const double alignment = std::clamp(std::abs(heading.dot(step.dir_map)), 0.0, 1.0);
        return std::acos(alignment);
    }

    const nav_executor::StepConstraintSchedule& schedule_;
    const EpisodeConfig& config_;
    double previous_u_;
};

} // namespace

EpisodeMetrics run_episode(
    const CompiledScenario& scenario,
    const uint64_t seed,
    const RuntimeConfig& runtime,
    const EpisodeConfig& config,
    const ParameterCandidate& candidate,
    rclcpp::Logger logger
) {
    EpisodeMetrics metrics;
    metrics.scenario_name = scenario.spec.name;
    metrics.seed = seed;
    metrics.path_length = scenario.path->spline.arc_length(0.0, 1.0, 100);

    const nav_executor::MPCParams params = apply_candidate(runtime.mpc, candidate);
    nav_executor::MPCSolver solver(params, logger.get_child("solver"));
    nav_executor::StepController step_controller(
        runtime.step_dist_offset, params.follow.normal_profile, params.follow.capability_profiles,
        runtime.profile_blend, logger.get_child("step")
    );
    step_controller.set_path(scenario.path);
    WheelLegPlant plant(params.power_model);
    plant.reset(scenario.spec.start_pose, seed);
    solver.reset_warm_start();
    solver.reset_observer();

    double path_u = scenario.path->spline.project_extrapolated(
        scenario.spec.start_pose.head<2>(), 0.0,
        params.follow.projection.proj_num_samples,
        params.follow.projection.proj_search_window,
        params.follow.projection.local_search_lazy_distance
    );
    double metric_u = path_u;
    StepMonitor step_monitor(*scenario.path->step_constraint_schedule, config, metric_u);

    Eigen::Vector2d previous_command = Eigen::Vector2d::Zero();
    const double timeout = scenario.spec.timeout > 0.0 ? scenario.spec.timeout : config.default_timeout;
    const int max_ticks = static_cast<int>(std::ceil(timeout / nav_executor::MPC_DT));

    for (int tick = 0; tick < max_ticks && !metrics.reached; ++tick) {
        const PlantSample current = plant.sample();
        path_u = scenario.path->spline.project_extrapolated(
            current.pose.head<2>(), path_u,
            params.follow.projection.proj_num_samples,
            params.follow.projection.proj_search_window,
            params.follow.projection.local_search_lazy_distance
        );
        step_controller.update_active_segment(path_u);
        step_controller.tick_profile_blend();
        solver.update_observer(current.chassis);
        solver.set_energy_state(current.energy, plant.referee_power_limit());

        const std::vector<const nav_executor::CostMap*> prediction_maps;
        auto result = solver.solve_follow(
            scenario.path->spline,
            current.pose,
            current.chassis,
            *scenario.control_cost_map,
            *scenario.control_cost_map,
            prediction_maps,
            nav_executor::MPC_DT,
            step_controller.current_blended_profile(),
            scenario.path->step_constraint_schedule,
            false
        );
        if (!result) {
            metrics.solver_failed = true;
            break;
        }

        // 平滑度地板：仅累积超出物理设计加速度阈值的相对抖动（阈值内零成本，不拖慢控制器）。
        const Eigen::Vector2d command = result->command;
        const double accel_v = (command.x() - previous_command.x()) / nav_executor::MPC_DT;
        const double accel_w = (command.y() - previous_command.y()) / nav_executor::MPC_DT;
        const double excess_v = std::max(0.0, std::abs(accel_v) / config.soft_scales.accel_floor - 1.0);
        const double excess_w = std::max(0.0, std::abs(accel_w) / config.soft_scales.alpha_floor - 1.0);
        metrics.smoothness_excess_integral += (excess_v + excess_w) * nav_executor::MPC_DT;
        previous_command = command;

        for (const PlantSample& sample : plant.step(command, nav_executor::MPC_DT)) {
            metrics.elapsed_time += SUBSTEP_DT;
            metric_u = scenario.path->spline.project_extrapolated(
                sample.pose.head<2>(), metric_u, 8, 0.03,
                params.follow.projection.local_search_lazy_distance
            );
            metrics.final_progress = std::max(metrics.final_progress, std::clamp(metric_u, 0.0, 1.0));

            const auto spline_eval = scenario.path->spline.eval(metric_u);
            const Eigen::Vector2d error = sample.pose.head<2>() - spline_eval.p;
            const double lateral = -error.x() * spline_eval.sin_r + error.y() * spline_eval.cos_r;
            metrics.cross_track_squared_integral += lateral * lateral * SUBSTEP_DT;

            double cost = 255.0;
            const Eigen::Vector2d grid = scenario.global_cost_map->map_coord_to_grid(sample.pose.head<2>());
            if (scenario.global_cost_map->is_valid_coord(grid)) cost = scenario.global_cost_map->interpolate(grid);
            metrics.max_cost = std::max(metrics.max_cost, cost);
            if (cost > config.high_cost_threshold) {
                metrics.high_cost_integral += (cost - config.high_cost_threshold)
                    / std::max(255.0 - config.high_cost_threshold, 1.0) * SUBSTEP_DT;
            }
            if (cost >= config.lethal_cost_threshold) metrics.hazard_duration += SUBSTEP_DT;

            step_monitor.update(metric_u, sample, metrics);

            if ((sample.pose.head<2>() - scenario.spec.goal).norm() <= config.goal_radius) {
                metrics.reached = true;
                metrics.arrival_speed = std::abs(sample.chassis.velocity);
                metrics.arrival_omega = std::abs(sample.chassis.omega);
                break;
            }
        }
    }
    return metrics;
}

} // namespace mpc_tuner
