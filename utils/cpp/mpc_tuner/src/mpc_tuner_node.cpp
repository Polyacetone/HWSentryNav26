#include <mpc_tuner/config_loader.hpp>
#include <mpc_tuner/episode_runner.hpp>
#include <mpc_tuner/optimizer.hpp>
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
    out << "scenario,seed,reached,solver_failed,time,progress,arrival_v,max_cost,lethal_duration,severe_step\n";
    for (const auto& e : episodes) {
        out << e.scenario_name << ',' << e.seed << ',' << e.reached << ',' << e.solver_failed << ','
            << e.elapsed_time << ',' << e.final_progress << ',' << e.arrival_speed << ',' << e.max_cost << ','
            << e.lethal_cost_duration << ',' << e.severe_step_violations << '\n';
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

        const auto evaluate = [&](const mpc_tuner::ParameterCandidate& candidate) {
            auto episodes = evaluate_set(scenarios, "train", runtime, tuner_config, candidate, logger);
            return std::pair {mpc_tuner::aggregate_fitness(episodes, tuner_config.episode), std::move(episodes)};
        };

        std::filesystem::create_directories(args.output);
        mpc_tuner::ParameterCandidate best = baseline;
        if (args.smoke) {
            auto [fitness, episodes] = evaluate(baseline);
            write_validation(args.output / "smoke.csv", episodes);
            mpc_tuner::write_parameter_overlay(baseline, args.output / "best_params.yaml");
            RCLCPP_INFO(
                logger, "Smoke result: failed=%d severe_env=%d severe_step=%d quality=%.3f",
                fitness.failed_scenarios, fitness.severe_environment_violations,
                fitness.severe_step_violations, fitness.soft_quality_cost
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
            const auto fitness = mpc_tuner::aggregate_fitness(validation, tuner_config.episode);
            RCLCPP_INFO(
                logger, "Validation: failed=%d severe_env=%d severe_step=%d fast_arrival=%d quality=%.3f",
                fitness.failed_scenarios, fitness.severe_environment_violations,
                fitness.severe_step_violations, fitness.fast_arrival_count, fitness.soft_quality_cost
            );
        }
        rclcpp::shutdown();
        return 0;
    } catch (const std::exception& error) {
        RCLCPP_FATAL(logger, "%s", error.what());
        rclcpp::shutdown();
        return 1;
    }
}
