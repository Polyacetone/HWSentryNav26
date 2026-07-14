#include <mpc_tuner/config_loader.hpp>
#include <mpc_tuner/episode_runner.hpp>
#include <mpc_tuner/fitness.hpp>
#include <mpc_tuner/optimizer.hpp>
#include <mpc_tuner/parameter_mapping.hpp>
#include <mpc_tuner/scene_bundle.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>

namespace {

struct Arguments {
    std::filesystem::path scenes_directory;
    std::filesystem::path output = "mpc_tuning_results";
    std::filesystem::path tuner_config;
    std::filesystem::path nav_config_directory;
    bool smoke = false;
};

Arguments parse_arguments(const std::vector<std::string>& arguments) {
    Arguments out;
    out.scenes_directory = std::filesystem::path(
        ament_index_cpp::get_package_share_directory("mpc_tuner")) / "scenes";
    out.tuner_config = std::filesystem::path(
        ament_index_cpp::get_package_share_directory("mpc_tuner")) / "config/tuner.yaml";
    out.nav_config_directory = std::filesystem::path(
        ament_index_cpp::get_package_share_directory("nav_executor")) / "config";
    for (size_t i = 1; i < arguments.size(); ++i) {
        const std::string& arg = arguments[i];
        const auto take_value = [&](std::filesystem::path& destination) {
            if (i + 1 >= arguments.size()) throw std::runtime_error("Missing value after " + arg);
            destination = arguments[++i];
        };
        if (arg == "--output") take_value(out.output);
        else if (arg == "--tuner-config") take_value(out.tuner_config);
        else if (arg == "--nav-config-dir") take_value(out.nav_config_directory);
        else if (arg == "--smoke") out.smoke = true;
        else if (arg == "--help") {
            std::cout << "Usage: mpc_tuner_node [--output DIR] [--smoke]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return out;
}

std::vector<mpc_tuner::EpisodeMetrics> evaluate_set(
    const std::vector<mpc_tuner::CompiledScenario>& scenarios,
    const std::string& split,
    const mpc_tuner::RuntimeConfig& runtime,
    const mpc_tuner::TunerConfig& config,
    const mpc_tuner::ParameterCandidate& candidate,
    rclcpp::Logger logger
) {
    std::vector<mpc_tuner::EpisodeMetrics> episodes;
    for (const auto& scenario : scenarios) {
        if (scenario.spec.split != split) continue;
        for (const uint64_t seed : scenario.spec.seeds) {
            episodes.push_back(mpc_tuner::run_episode(
                scenario, seed, runtime, config.episode, candidate, logger
            ));
        }
    }
    return episodes;
}

void write_validation(
    const std::filesystem::path& path,
    const std::vector<mpc_tuner::EpisodeMetrics>& episodes
) {
    std::ofstream out(path);
    out << "scenario,seed,reached,solver_failed,time,progress,arrival_v,max_cost,hazard_duration,"
           "severe_step_speed,severe_step_heading\n";
    for (const auto& e : episodes) {
        out << e.scenario_name << ',' << e.seed << ',' << e.reached << ',' << e.solver_failed << ','
            << e.elapsed_time << ',' << e.final_progress << ',' << e.arrival_speed << ',' << e.max_cost << ','
            << e.hazard_duration << ',' << e.severe_step_speed_events << ',' << e.severe_step_heading_events << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    const auto non_ros_arguments = rclcpp::init_and_remove_ros_arguments(argc, argv);
    const auto logger = rclcpp::get_logger("mpc_tuner");
    try {
        const Arguments args = parse_arguments(non_ros_arguments);
        const auto tuner_config = mpc_tuner::load_tuner_config(args.tuner_config);
        const auto runtime = mpc_tuner::load_runtime_config(args.nav_config_directory);
        const auto scenarios = mpc_tuner::load_scene_splits(args.scenes_directory);
        const auto train_count = std::ranges::count_if(scenarios, [](const auto& scenario) {
            return scenario.spec.split == "train";
        });
        const auto validation_count = scenarios.size() - static_cast<size_t>(train_count);
        RCLCPP_INFO(
            logger, "Loaded %zu training and %zu validation routes from %s/train and %s/validation",
            static_cast<size_t>(train_count), validation_count,
            args.scenes_directory.c_str(), args.scenes_directory.c_str()
        );
        const auto baseline = mpc_tuner::baseline_candidate(runtime.mpc, tuner_config);
        const double lambda = tuner_config.study.regularization_lambda;

        const auto evaluate = [&](const mpc_tuner::ParameterCandidate& candidate) {
            auto episodes = evaluate_set(scenarios, "train", runtime, tuner_config, candidate, logger);
            auto fitness = mpc_tuner::aggregate_fitness(
                episodes, tuner_config.episode, candidate, baseline, lambda
            );
            return std::pair {std::move(fitness), std::move(episodes)};
        };

        std::filesystem::create_directories(args.output);
        mpc_tuner::ParameterCandidate best = baseline;
        if (args.smoke) {
            auto [fitness, episodes] = evaluate(baseline);
            write_validation(args.output / "smoke.csv", episodes);
            mpc_tuner::write_parameter_overlay(runtime.mpc, baseline, args.output / "best_params.yaml");
            RCLCPP_INFO(
                logger, "Smoke result: hazard=%d severe_step_speed=%d severe_step_heading=%d failed=%d soft_cost=%.3f",
                fitness.hazard_events, fitness.severe_step_speed_events,
                fitness.severe_step_heading_events, fitness.failed_scenarios, fitness.soft_cost
            );
        } else {
            const auto report_progress = [&](const mpc_tuner::OptimizationProgress& progress) {
                RCLCPP_INFO(
                    logger,
                    "Generation %d/%d: evaluated %d/%d candidates, %d workers active, elapsed %.0f s",
                    progress.generation, progress.total_generations,
                    progress.completed_candidates, progress.population_size,
                    progress.active_workers, progress.elapsed_seconds
                );
            };
            best = mpc_tuner::run_cem(runtime.mpc, tuner_config, evaluate, args.output, report_progress);
        }

        const auto validation = evaluate_set(scenarios, "validation", runtime, tuner_config, best, logger);
        if (!validation.empty()) {
            write_validation(args.output / "validation.csv", validation);
            // 验证集为独立体检，不含正则项（lambda=0），只反映真实质量。
            const auto best_fitness = mpc_tuner::aggregate_fitness(
                validation, tuner_config.episode, best, baseline, 0.0
            );
            RCLCPP_INFO(
                logger, "Validation (best): hazard=%d severe_step_speed=%d severe_step_heading=%d failed=%d soft_cost=%.3f",
                best_fitness.hazard_events, best_fitness.severe_step_speed_events,
                best_fitness.severe_step_heading_events, best_fitness.failed_scenarios, best_fitness.soft_cost
            );

            // Q2.C 验证护栏：train best 相对基线在验证集上是否退化。5 条验证路线太少、单 seed 噪声不小，
            // 因此仅作告警而非自动回退——把最终取舍权留给人工审阅 best_params.yaml。
            const auto baseline_validation = evaluate_set(
                scenarios, "validation", runtime, tuner_config, baseline, logger
            );
            const auto baseline_fitness = mpc_tuner::aggregate_fitness(
                baseline_validation, tuner_config.episode, baseline, baseline, 0.0
            );
            const bool hard_regressed = baseline_fitness < best_fitness;
            if (hard_regressed || best_fitness.soft_cost > baseline_fitness.soft_cost) {
                RCLCPP_WARN(
                    logger,
                    "Validation guardrail: tuned params regress vs baseline "
                    "(soft_cost %.3f -> %.3f, hard_gate_regressed=%d). Review best_params.yaml before adopting.",
                    baseline_fitness.soft_cost, best_fitness.soft_cost, hard_regressed
                );
            }
        }
        rclcpp::shutdown();
        return 0;
    } catch (const std::exception& error) {
        RCLCPP_FATAL(logger, "%s", error.what());
        rclcpp::shutdown();
        return 1;
    }
}
