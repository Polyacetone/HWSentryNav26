#include <mpc_tuner/episode_runner.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

#include <nav_executor/path_executor/solver/mpc_solver.hpp>

#include <mpc_tuner/wheel_leg_plant.hpp>

namespace mpc_tuner {
namespace {

double wrap_pi(const double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

double clamped_ratio(const double value, const double scale) {
    return std::clamp(value / std::max(scale, 1e-9), 0.0, 1.0);
}

} // namespace

nav_executor::MPCParams apply_candidate(
    const nav_executor::MPCParams& base,
    const ParameterCandidate& candidate
) {
    nav_executor::MPCParams out = base;
    auto& tracking = out.follow.tracking_weights;
    auto& command = out.follow.command_weights;
    tracking.q_y = candidate.values[0];
    tracking.q_theta = candidate.values[1];
    tracking.q_u = candidate.values[2];
    tracking.y_tube = candidate.values[3];
    tracking.q_term_prog = candidate.values[4];
    tracking.q_term_lateral = candidate.values[5];
    command.r_v = candidate.values[6];
    command.r_omega = candidate.values[7];
    command.r_dv = candidate.values[8];
    command.r_domega = candidate.values[9];
    out.follow.global_search.enable = false;
    out.follow.rollout_safety.enable_lethal_obstacle_check = false;
    return out;
}

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
    Eigen::Vector2d previous_command = Eigen::Vector2d::Zero();
    bool severe_step_active = false;
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

        const Eigen::Vector2d command = result->command;
        const Eigen::Vector2d delta = command - previous_command;
        metrics.command_dv_squared_sum += std::pow(delta.x() / nav_executor::MPC_DT, 2);
        metrics.command_domega_squared_sum += std::pow(delta.y() / nav_executor::MPC_DT, 2);
        ++metrics.control_samples;
        previous_command = command;

        for (const PlantSample& sample : plant.step(command, nav_executor::MPC_DT)) {
            metrics.elapsed_time += 0.001;
            metric_u = scenario.path->spline.project_extrapolated(
                sample.pose.head<2>(), metric_u, 8, 0.03,
                params.follow.projection.local_search_lazy_distance
            );
            metrics.final_progress = std::max(metrics.final_progress, std::clamp(metric_u, 0.0, 1.0));
            const auto spline_eval = scenario.path->spline.eval(metric_u);
            const Eigen::Vector2d error = sample.pose.head<2>() - spline_eval.p;
            const double lateral = -error.x() * spline_eval.sin_r + error.y() * spline_eval.cos_r;
            const double heading_error = wrap_pi(sample.pose.z() - spline_eval.thetar);
            metrics.cross_track_squared_integral += lateral * lateral * 0.001;
            metrics.heading_error_squared_integral += heading_error * heading_error * 0.001;

            double cost = 255.0;
            const Eigen::Vector2d grid = scenario.global_cost_map->map_coord_to_grid(sample.pose.head<2>());
            if (scenario.global_cost_map->is_valid_coord(grid)) cost = scenario.global_cost_map->interpolate(grid);
            metrics.max_cost = std::max(metrics.max_cost, cost);
            if (cost > config.high_cost_threshold) {
                metrics.high_cost_integral += (cost - config.high_cost_threshold)
                    / std::max(255.0 - config.high_cost_threshold, 1.0) * 0.001;
            }
            if (cost >= config.lethal_cost_threshold) metrics.lethal_cost_duration += 0.001;

            bool severe_now = false;
            if (const auto* step = scenario.path->step_constraint_schedule->constraint_at(metric_u)) {
                const double speed = sample.chassis.velocity;
                const double under = std::max(0.0, step->speed_min - speed);
                const double over = std::max(0.0, speed - step->speed_max);
                metrics.step_speed_violation += (under + over) * 0.001;
                const Eigen::Vector2d heading(std::cos(sample.pose.z()), std::sin(sample.pose.z()));
                const double alignment = std::clamp(std::abs(heading.dot(step->dir_map)), 0.0, 1.0);
                const double angle_error = std::acos(alignment);
                metrics.step_alignment_violation += (1.0 - alignment) * 0.001;
                severe_now = under > config.severe_step_speed_margin
                    || over > config.severe_step_speed_margin
                    || angle_error > config.severe_step_heading_error;
            }
            if (severe_now && !severe_step_active) ++metrics.severe_step_violations;
            severe_step_active = severe_now;

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

Fitness aggregate_fitness(const std::vector<EpisodeMetrics>& episodes, const EpisodeConfig& config) {
    Fitness fitness;
    if (episodes.empty()) {
        fitness.failed_scenarios = 1;
        return fitness;
    }
    for (const EpisodeMetrics& metrics : episodes) {
        if (!metrics.reached || metrics.solver_failed) {
            ++fitness.failed_scenarios;
            fitness.progress_deficit += 1.0 - std::clamp(metrics.final_progress, 0.0, 1.0);
        }
        if (metrics.lethal_cost_duration > 1e-6) ++fitness.severe_environment_violations;
        fitness.severe_step_violations += metrics.severe_step_violations;
        if (metrics.reached && metrics.arrival_speed > config.acceptable_arrival_speed) ++fitness.fast_arrival_count;
        fitness.arrival_speed_excess += std::max(0.0, metrics.arrival_speed - config.target_arrival_speed);

        const double duration = std::max(metrics.elapsed_time, 0.001);
        const double cross_track_rms = std::sqrt(metrics.cross_track_squared_integral / duration);
        const double heading_rms = std::sqrt(metrics.heading_error_squared_integral / duration);
        const double dv_rms = metrics.control_samples > 0
            ? std::sqrt(metrics.command_dv_squared_sum / metrics.control_samples) : 0.0;
        const double dw_rms = metrics.control_samples > 0
            ? std::sqrt(metrics.command_domega_squared_sum / metrics.control_samples) : 0.0;
        fitness.soft_quality_cost +=
            clamped_ratio(metrics.elapsed_time, 20.0)
            + 2.0 * clamped_ratio(metrics.high_cost_integral, 1.0)
            + 2.0 * clamped_ratio(metrics.step_speed_violation, 1.0)
            + 2.0 * clamped_ratio(metrics.step_alignment_violation, 0.5)
            + 1.5 * clamped_ratio(cross_track_rms, 0.5)
            + clamped_ratio(heading_rms, std::numbers::pi / 4.0)
            + 0.25 * clamped_ratio(dv_rms, 5.0)
            + 0.25 * clamped_ratio(dw_rms, 20.0);
    }
    fitness.progress_deficit /= static_cast<double>(episodes.size());
    fitness.arrival_speed_excess /= static_cast<double>(episodes.size());
    fitness.soft_quality_cost /= static_cast<double>(episodes.size());
    return fitness;
}

} // namespace mpc_tuner
