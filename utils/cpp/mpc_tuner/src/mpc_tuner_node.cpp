#include <mpc_tuner/config_loader.hpp>
#include <mpc_tuner/episode_runner.hpp>
#include <mpc_tuner/fitness.hpp>
#include <mpc_tuner/optimizer.hpp>
#include <mpc_tuner/parameter_mapping.hpp>
#include <mpc_tuner/scene_bundle.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
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

void write_final_parameters(
    const std::filesystem::path& path,
    const mpc_tuner::ParameterCandidate& baseline,
    const mpc_tuner::ParameterCandidate& best
) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write " + path.string());
    out << "parameter,baseline_value,best_value,baseline_normalized,best_normalized\n";
    out << std::setprecision(12);
    for (size_t i = 0; i < mpc_tuner::PARAMETER_COUNT; ++i) {
        out << mpc_tuner::PARAMETER_DESCRIPTORS[i].name << ',' << baseline.values[i] << ',' << best.values[i] << ','
            << baseline.normalized[i] << ',' << best.normalized[i] << '\n';
    }
}

std::vector<mpc_tuner::ExecutedFeature> features_for_parameter(const std::string_view parameter) {
    using Feature = mpc_tuner::ExecutedFeature;
    if (parameter == "q_y" || parameter == "y_tube") return {Feature::LATERAL_TUBE};
    if (parameter == "q_theta") return {Feature::HEADING};
    if (parameter == "q_u") return {Feature::PROGRESS};
    if (parameter == "q_term_prog") return {Feature::TERMINAL_PROGRESS};
    if (parameter == "q_term_lateral") return {Feature::TERMINAL_LATERAL};
    if (parameter == "r_v") return {Feature::COMMAND_V};
    if (parameter == "r_omega") return {Feature::COMMAND_OMEGA};
    if (parameter == "r_dv") return {Feature::COMMAND_DV};
    if (parameter == "r_domega") return {Feature::COMMAND_DOMEGA};
    if (parameter == "environment_obstacle") return {Feature::OBSTACLE};
    if (parameter == "step_direction") return {Feature::STEP_DIRECTION};
    if (parameter == "step_velocity_window") return {Feature::STEP_VELOCITY};
    if (parameter == "step_reachability_lo") return {Feature::STEP_REACHABILITY_LO};
    if (parameter == "step_reachability_hi") return {Feature::STEP_REACHABILITY_HI};
    if (parameter == "q_v_final") return {Feature::TERMINAL_BRAKE};
    if (parameter == "motion_penalty_scale") {
        return {Feature::ACC_LIMIT, Feature::ALPHA_LIMIT, Feature::LAT_ACCEL};
    }
    if (parameter == "step_smoothness_scale") {
        return {Feature::STEP_OMEGA, Feature::STEP_DV, Feature::STEP_DOMEGA};
    }
    throw std::runtime_error("No feature coverage mapping for parameter " + std::string(parameter));
}

struct CoverageTotals {
    std::array<double, mpc_tuner::EXECUTED_FEATURE_COUNT> raw_sum {};
    std::array<uint64_t, mpc_tuner::EXECUTED_FEATURE_COUNT> active_samples {};
    std::array<uint64_t, mpc_tuner::EXECUTED_FEATURE_COUNT> sample_count {};
};

CoverageTotals aggregate_coverage(const std::vector<mpc_tuner::EpisodeMetrics>& episodes) {
    CoverageTotals totals;
    for (const auto& episode : episodes) {
        for (size_t i = 0; i < mpc_tuner::EXECUTED_FEATURE_COUNT; ++i) {
            totals.raw_sum[i] += episode.feature_coverage.raw_sum[i];
            totals.active_samples[i] += episode.feature_coverage.active_samples[i];
            totals.sample_count[i] += episode.feature_coverage.sample_count[i];
        }
    }
    return totals;
}

void write_coverage_report(
    const std::filesystem::path& feature_path,
    const std::filesystem::path& parameter_path,
    const std::string_view candidate_label,
    const std::string_view split,
    const std::vector<mpc_tuner::EpisodeMetrics>& episodes
) {
    const CoverageTotals totals = aggregate_coverage(episodes);
    std::ofstream feature_out(feature_path, std::ios::app);
    std::ofstream parameter_out(parameter_path, std::ios::app);
    if (!feature_out || !parameter_out) throw std::runtime_error("Failed to write coverage report");
    feature_out << std::setprecision(12);
    parameter_out << std::setprecision(12);

    for (size_t i = 0; i < mpc_tuner::EXECUTED_FEATURE_COUNT; ++i) {
        const double rate = totals.sample_count[i] == 0 ? 0.0
            : static_cast<double>(totals.active_samples[i]) / static_cast<double>(totals.sample_count[i]);
        feature_out << "executed_trajectory," << candidate_label << ',' << split << ',' << mpc_tuner::EXECUTED_FEATURE_NAMES[i] << ','
                    << totals.raw_sum[i] << ',' << totals.active_samples[i] << ',' << totals.sample_count[i] << ','
                    << rate << '\n';
    }
    for (const auto& descriptor : mpc_tuner::PARAMETER_DESCRIPTORS) {
        double raw_sum = 0.0;
        uint64_t active_samples = 0;
        uint64_t sample_count = 0;
        for (const auto feature : features_for_parameter(descriptor.name)) {
            const size_t i = static_cast<size_t>(feature);
            raw_sum += totals.raw_sum[i];
            active_samples += totals.active_samples[i];
            sample_count += totals.sample_count[i];
        }
        const double rate = sample_count == 0 ? 0.0 : static_cast<double>(active_samples) / sample_count;
        parameter_out << "executed_trajectory," << candidate_label << ',' << split << ',' << descriptor.name << ',' << raw_sum << ','
                      << active_samples << ',' << sample_count << ',' << rate << '\n';
    }
}

void initialize_coverage_reports(const std::filesystem::path& output_directory) {
    std::ofstream feature_out(output_directory / "feature_coverage.csv");
    std::ofstream parameter_out(output_directory / "parameter_coverage.csv");
    if (!feature_out || !parameter_out) throw std::runtime_error("Failed to initialize coverage reports");
    feature_out << "coverage_source,candidate,split,feature,raw_sum,active_samples,sample_count,activation_rate\n";
    parameter_out << "coverage_source,candidate,split,parameter,raw_sum,active_samples,sample_count,activation_rate\n";
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
        initialize_coverage_reports(args.output);
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

        write_final_parameters(args.output / "final_parameters.csv", baseline, best);
        const auto best_training = evaluate_set(scenarios, "train", runtime, tuner_config, best, logger);
        write_coverage_report(
            args.output / "feature_coverage.csv", args.output / "parameter_coverage.csv",
            "best", "train", best_training
        );
        const auto baseline_training = evaluate_set(scenarios, "train", runtime, tuner_config, baseline, logger);
        write_coverage_report(
            args.output / "feature_coverage.csv", args.output / "parameter_coverage.csv",
            "baseline", "train", baseline_training
        );

        const auto validation = evaluate_set(scenarios, "validation", runtime, tuner_config, best, logger);
        if (!validation.empty()) {
            write_validation(args.output / "validation.csv", validation);
            write_coverage_report(
                args.output / "feature_coverage.csv", args.output / "parameter_coverage.csv",
                "best", "validation", validation
            );
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
            write_coverage_report(
                args.output / "feature_coverage.csv", args.output / "parameter_coverage.csv",
                "baseline", "validation", baseline_validation
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
