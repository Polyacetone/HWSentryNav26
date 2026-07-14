#include <mpc_tuner/config_loader.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string_view>

#include <yaml-cpp/yaml.h>

namespace mpc_tuner {
namespace {

YAML::Node ros_params(const std::filesystem::path& path) {
    const YAML::Node root = YAML::LoadFile(path.string());
    const YAML::Node params = root["/**"]["ros__parameters"];
    if (!params) throw std::runtime_error("Missing /**/ros__parameters in " + path.string());
    return params;
}

YAML::Node at(const YAML::Node& root, const std::string_view dotted_path) {
    YAML::Node node = root;
    size_t begin = 0;
    while (begin < dotted_path.size()) {
        const size_t end = dotted_path.find('.', begin);
        const std::string key(dotted_path.substr(begin, end == std::string_view::npos ? dotted_path.size() - begin : end - begin));
        const YAML::Node next = node[key];
        if (!next) throw std::runtime_error("Missing YAML key: " + std::string(dotted_path));
        node.reset(next);
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    return node;
}

template<typename T>
T value(const YAML::Node& root, const std::string_view path) {
    return at(root, path).as<T>();
}

nav_executor::CapabilityProfile capability_profile(const YAML::Node& root, const std::string& prefix) {
    return {
        .command_bounds = {
            .vel_max = value<double>(root, prefix + ".command_bounds.vel_max"),
            .vel_min = value<double>(root, prefix + ".command_bounds.vel_min"),
            .omega_max = value<double>(root, prefix + ".command_bounds.omega_max"),
            .omega_min = value<double>(root, prefix + ".command_bounds.omega_min"),
        },
        .motion_constraints = {
            .acc_max = value<double>(root, prefix + ".motion_constraints.acc_max"),
            .alpha_max = value<double>(root, prefix + ".motion_constraints.alpha_max"),
            .a_lat_max = value<double>(root, prefix + ".motion_constraints.a_lat_max"),
        },
    };
}

nav_executor::BSplineOptimizer::CurvaturePenaltyParams curvature_params(
    const YAML::Node& root, const std::string& prefix
) {
    return {
        .base_weight = value<double>(root, prefix + ".base_weight"),
        .base_beta = value<double>(root, prefix + ".base_beta"),
        .limit_weight = value<double>(root, prefix + ".limit_weight"),
        .limit_beta = value<double>(root, prefix + ".limit_beta"),
        .min_speed_epsilon = value<double>(root, prefix + ".min_speed_epsilon"),
        .speed_gate_threshold = value<double>(root, prefix + ".speed_gate_threshold"),
    };
}

void load_terrain_profiles(RuntimeConfig& out, const YAML::Node& root) {
    using enum nav_executor::CapabilityLevel;
    out.terrain_profiles.capability_profiles[static_cast<size_t>(LOW)] = capability_profile(root, "terrain_profiles.capability_profiles.low");
    out.terrain_profiles.capability_profiles[static_cast<size_t>(MEDIUM)] = capability_profile(root, "terrain_profiles.capability_profiles.medium");
    out.terrain_profiles.capability_profiles[static_cast<size_t>(HIGH)] = capability_profile(root, "terrain_profiles.capability_profiles.high");

    out.terrain_profiles.high_performance_buffercap_threshold = value<double>(root, "terrain_profiles.high_performance.buffercap_threshold");
    out.terrain_profiles.high_performance_supercap_threshold = value<double>(root, "terrain_profiles.high_performance.supercap_threshold");
    out.terrain_profiles.high_performance_rfr_pwr_limit_threshold = value<double>(root, "terrain_profiles.high_performance.rfr_pwr_limit_threshold");

    const std::array<std::string, 5> labels {"slope", "step_l1", "step_l2", "fly_slope", "step_high"};
    for (size_t label_index = 0; label_index < labels.size(); ++label_index) {
        const std::string base = "terrain_profiles.directional_labels." + labels[label_index];
        const YAML::Node entry = at(root, base);
        auto load_direction = [&](const char* direction, std::vector<nav_executor::TerrainStepRule>& destination) {
            for (const auto& mode_node : entry[direction]) {
                const std::string name = mode_node.as<std::string>();
                const YAML::Node mode = entry["modes"][name];
                destination.push_back({
                    .name = name,
                    .chassis_mode = mode["chassis_mode"].as<uint8_t>(),
                    .capability = nav_executor::capability_level_from_string(mode["capability"].as<std::string>()),
                    .speed = {.min = mode["speed"]["min"].as<double>(), .max = mode["speed"]["max"].as<double>()},
                    .requires_high_performance = mode["requires_high_perf"].as<bool>(),
                    .run_up = mode["run_up"].as<double>(),
                });
            }
        };
        load_direction("up", out.terrain_profiles.directional_labels[label_index].up);
        load_direction("down", out.terrain_profiles.directional_labels[label_index].down);
    }

    out.profile_blend = {
        .v_step = value<double>(root, "terrain_profiles.profile_blend.v_step"),
        .w_step = value<double>(root, "terrain_profiles.profile_blend.w_step"),
        .acc_step = value<double>(root, "terrain_profiles.profile_blend.acc_step"),
        .alpha_step = value<double>(root, "terrain_profiles.profile_blend.alpha_step"),
        .a_lat_step = value<double>(root, "terrain_profiles.profile_blend.a_lat_step"),
    };
}

void load_follow(RuntimeConfig& out, const YAML::Node& root) {
    auto& p = out.mpc.follow;
    p.start_command = {
        .vel_cmd_act_gap_max = value<double>(root, "mpc.follow.start_command.vel_cmd_act_gap_max"),
        .omega_cmd_act_gap_max = value<double>(root, "mpc.follow.start_command.omega_cmd_act_gap_max"),
    };
    p.normal_profile = capability_profile(root, "mpc.follow");
    p.capability_profiles = out.terrain_profiles.capability_profiles;
    p.tracking_weights = {
        .q_y = value<double>(root, "mpc.follow.tracking_weights.q_y"),
        .q_theta = value<double>(root, "mpc.follow.tracking_weights.q_theta"),
        .q_u = value<double>(root, "mpc.follow.tracking_weights.q_u"),
        .y_tube = value<double>(root, "mpc.follow.tracking_weights.y_tube"),
        .q_term_prog = value<double>(root, "mpc.follow.tracking_weights.q_term_prog"),
        .q_term_lateral = value<double>(root, "mpc.follow.tracking_weights.q_term_lateral"),
    };
    p.command_weights = {
        .r_v = value<double>(root, "mpc.follow.command_weights.r_v"),
        .r_omega = value<double>(root, "mpc.follow.command_weights.r_omega"),
        .r_dv = value<double>(root, "mpc.follow.command_weights.r_dv"),
        .r_domega = value<double>(root, "mpc.follow.command_weights.r_domega"),
    };
    p.motion_constraint_weights = {
        .acc_limit = value<double>(root, "mpc.follow.motion_constraint_weights.acc_limit"),
        .alpha_limit = value<double>(root, "mpc.follow.motion_constraint_weights.alpha_limit"),
        .lat_acc = value<double>(root, "mpc.follow.motion_constraint_weights.lat_acc"),
    };
    p.terrain_limits = {
        .step_reachability_guide_acc = value<double>(root, "mpc.follow.terrain_limits.step_reachability_guide_acc"),
        .step_feasibility_margin_band = value<double>(root, "mpc.follow.terrain_limits.step_feasibility_margin_band"),
    };
    p.terrain_weights = {
        .step_vel_weight = value<double>(root, "mpc.follow.terrain_weights.internal.velocity_window"),
        .step_reachability_lo = value<double>(root, "mpc.follow.terrain_weights.approach.reachability_lo"),
        .step_reachability_hi = value<double>(root, "mpc.follow.terrain_weights.approach.reachability_hi"),
        .direction = value<double>(root, "mpc.follow.terrain_weights.internal.direction"),
        .step_omega = value<double>(root, "mpc.follow.terrain_weights.internal.omega"),
        .step_dv = value<double>(root, "mpc.follow.terrain_weights.internal.velocity_smooth"),
        .step_domega = value<double>(root, "mpc.follow.terrain_weights.internal.omega_smooth"),
    };
    p.environment_weights.obstacle = value<double>(root, "mpc.follow.environment_weights.obstacle");
    p.terminal_weights = {
        .q_v_final = value<double>(root, "mpc.follow.terminal_weights.q_v_final"),
        .a_brake = value<double>(root, "mpc.follow.terminal_weights.a_brake"),
        .slow_down_target_vel = value<double>(root, "mpc.follow.terminal_weights.slow_down_target_vel"),
    };
    p.projection = {
        .proj_num_samples = value<int>(root, "mpc.follow.projection.num_samples"),
        .proj_search_window = value<double>(root, "mpc.follow.projection.search_window"),
        .local_search_lazy_distance = value<double>(root, "mpc.follow.projection.local_search_lazy_distance"),
    };
    p.global_search.enable = false;
    p.rollout_safety.enable_lethal_obstacle_check = false;
    p.rollout_safety.lethal_obstacle_threshold = value<double>(root, "mpc.follow.rollout_safety.lethal_obstacle_threshold");
    p.rollout_safety.fddp_lethal_consecutive_threshold = value<int>(root, "mpc.follow.rollout_safety.fddp_lethal_consecutive_threshold");
    p.max_iters = value<int>(root, "mpc.follow.max_iters");
}

// 读取软标量的尺度与权重；两者均为可选块，缺省时沿用 EpisodeConfig 的内置默认值。
void load_soft_fitness(EpisodeConfig& episode, const YAML::Node& node) {
    if (const YAML::Node scales = node["soft_scales"]) {
        SoftScales& s = episode.soft_scales;
        if (scales["reference_speed"]) s.reference_speed = scales["reference_speed"].as<double>();
        if (scales["arrival_speed_band"]) s.arrival_speed_band = scales["arrival_speed_band"].as<double>();
        if (scales["cross_track"]) s.cross_track = scales["cross_track"].as<double>();
        if (scales["high_cost_integral"]) s.high_cost_integral = scales["high_cost_integral"].as<double>();
        if (scales["accel_floor"]) s.accel_floor = scales["accel_floor"].as<double>();
        if (scales["alpha_floor"]) s.alpha_floor = scales["alpha_floor"].as<double>();
    }
    if (const YAML::Node weights = node["soft_weights"]) {
        SoftWeights& w = episode.soft_weights;
        if (weights["time"]) w.time = weights["time"].as<double>();
        if (weights["high_cost"]) w.high_cost = weights["high_cost"].as<double>();
        if (weights["arrival_speed"]) w.arrival_speed = weights["arrival_speed"].as<double>();
        if (weights["step_speed_minor"]) w.step_speed_minor = weights["step_speed_minor"].as<double>();
        if (weights["step_heading_minor"]) w.step_heading_minor = weights["step_heading_minor"].as<double>();
        if (weights["cross_track"]) w.cross_track = weights["cross_track"].as<double>();
        if (weights["smoothness"]) w.smoothness = weights["smoothness"].as<double>();
    }
}

} // namespace

TunerConfig load_tuner_config(const std::filesystem::path& path) {
    const YAML::Node root = YAML::LoadFile(path.string())["tuner"];
    TunerConfig out;
    const YAML::Node study = root["study"];
    out.study = {
        .seed = study["seed"].as<uint64_t>(),
        .population_size = study["population_size"].as<int>(),
        .elite_fraction = study["elite_fraction"].as<double>(),
        .generations = study["generations"].as<int>(),
        .initial_std = study["initial_std"].as<double>(),
        .min_std = study["min_std"].as<double>(),
        .parallel_workers = study["parallel_workers"].as<int>(),
        .progress_interval_seconds = study["progress_interval_seconds"].as<double>(),
        .regularization_lambda = study["regularization_lambda"].as<double>(),
    };
    if (out.study.population_size < 2) throw std::runtime_error("study.population_size must be at least 2");
    if (out.study.generations < 1) throw std::runtime_error("study.generations must be positive");
    if (out.study.parallel_workers < 0) throw std::runtime_error("study.parallel_workers must not be negative");
    if (!(out.study.progress_interval_seconds > 0.0)) {
        throw std::runtime_error("study.progress_interval_seconds must be positive");
    }
    const YAML::Node episode = root["episode"];
    out.episode.default_timeout = episode["default_timeout"].as<double>();
    out.episode.goal_radius = episode["goal_radius"].as<double>();
    out.episode.target_arrival_speed = episode["target_arrival_speed"].as<double>();
    out.episode.acceptable_arrival_speed = episode["acceptable_arrival_speed"].as<double>();
    out.episode.high_cost_threshold = episode["high_cost_threshold"].as<double>();
    out.episode.lethal_cost_threshold = episode["lethal_cost_threshold"].as<double>();
    out.episode.severe_step_heading_error =
        episode["severe_step_heading_error_deg"].as<double>() * std::numbers::pi / 180.0;
    out.episode.severe_step_speed_margin = episode["severe_step_speed_margin"].as<double>();
    out.episode.forward_progress_epsilon = episode["forward_progress_epsilon"].as<double>();
    load_soft_fitness(out.episode, episode);

    // 搜索空间由参数描述表驱动（单一真相源）：tuner.yaml 的 search_space 提供可选的逐参数区间覆盖，
    // 未列出的参数回退到描述表中的默认区间。这样新增可调参数只需改描述表，配置文件按需覆盖。
    const YAML::Node search_space = root["search_space"];
    for (size_t i = 0; i < PARAMETER_COUNT; ++i) {
        const ParameterDescriptor& desc = PARAMETER_DESCRIPTORS[i];
        double lower = desc.default_lower;
        double upper = desc.default_upper;
        if (const YAML::Node override_node = search_space[std::string(desc.name)]) {
            const auto bounds = override_node.as<std::vector<double>>();
            if (bounds.size() != 2 || !(bounds[0] < bounds[1])) {
                throw std::runtime_error("Invalid search range override for " + std::string(desc.name));
            }
            lower = bounds[0];
            upper = bounds[1];
        }
        out.search_ranges[i] = {.name = desc.name, .lower = lower, .upper = upper, .logarithmic = desc.logarithmic};
    }
    return out;
}

RuntimeConfig load_runtime_config(const std::filesystem::path& directory) {
    RuntimeConfig out {};
    const YAML::Node terrain = ros_params(directory / "terrain_profiles.yaml");
    load_terrain_profiles(out, terrain);

    const YAML::Node mpc = ros_params(directory / "mpc.yaml");
    load_follow(out, mpc);
    out.mpc.energy = {
        .enable = value<bool>(mpc, "mpc.energy.enable"),
        .threshold = value<double>(mpc, "mpc.energy.threshold"),
        .weight = value<double>(mpc, "mpc.energy.weight"),
    };

    const YAML::Node km = ros_params(directory / "kinematic_model.yaml")["kinematic_model"];
    auto& k = out.mpc.kinematic_model;
    k.z_ref = value<double>(km, "z_ref"); k.z_scale = value<double>(km, "z_scale"); k.rho_clip = value<double>(km, "rho_clip"); k.sgn_eps = value<double>(km, "sgn_eps");
    k.ca00 = value<double>(km, "ca00"); k.ca01 = value<double>(km, "ca01"); k.ca10 = value<double>(km, "ca10"); k.ca11 = value<double>(km, "ca11"); k.cb0 = value<double>(km, "cb0"); k.cb1 = value<double>(km, "cb1");
    k.dca00 = value<double>(km, "dca00"); k.dca01 = value<double>(km, "dca01"); k.dca10 = value<double>(km, "dca10"); k.dca11 = value<double>(km, "dca11"); k.dcb0 = value<double>(km, "dcb0"); k.dcb1 = value<double>(km, "dcb1");
    k.gxh = value<double>(km, "gxh"); k.gv = value<double>(km, "gv"); k.cf1 = value<double>(km, "cf1"); k.cf2 = value<double>(km, "cf2");
    k.w_lam0 = value<double>(km, "w_lam0"); k.w_k0 = value<double>(km, "w_k0"); k.w_cf0 = value<double>(km, "w_cf0"); k.w_lam1 = value<double>(km, "w_lam1"); k.w_k1 = value<double>(km, "w_k1"); k.w_cf1 = value<double>(km, "w_cf1");
    k.xh0_bias = value<double>(km, "xh0_bias"); k.xh0_psi = value<double>(km, "xh0_psi"); k.xh0_v = value<double>(km, "xh0_v");
    k.psi_bias = value<double>(km, "psi_bias"); k.psi_gain = value<double>(km, "psi_gain"); k.psi_v = value<double>(km, "psi_v"); k.obs_lv = value<double>(km, "obs_lv"); k.obs_lpsi = value<double>(km, "obs_lpsi");

    const YAML::Node power = ros_params(directory / "power_model.yaml")["power_model"];
    for (int i = 0; i < nav_executor::PWR_N; ++i) out.mpc.power_model.coeffs[static_cast<size_t>(i)] = power["c" + std::to_string(i)].as<double>();

    const YAML::Node planner = ros_params(directory / "path_planner.yaml");
    out.planner = {
        .occupied_threshold = value<int>(planner, "path_planner.traversability.occupied_threshold"),
        .on_step_threshold = value<double>(planner, "path_planner.traversability.on_step_threshold"),
        .start_prediction_enable = value<bool>(planner, "path_planner.start_prediction.enable"),
        .start_prediction_max_accel = value<double>(planner, "path_planner.start_prediction.max_accel"),
        .start_prediction_planning_delay = value<double>(planner, "path_planner.start_prediction.planning_delay"),
        .start_prediction_min_speed = value<double>(planner, "path_planner.start_prediction.min_speed"),
        .start_prediction_collision_check_step = value<double>(planner, "path_planner.start_prediction.collision_check_step"),
        .nudge_max_distance = value<double>(planner, "path_planner.nudge.max_distance"),
        .goal_reached_distance = value<double>(planner, "path_planner.planner.goal_reached_distance"),
        .skip_distance = value<double>(planner, "path_planner.planner.skip_distance"),
        .step_detection = {
            .detect_norm_threshold = value<double>(planner, "path_planner.step.detection.detect_norm_threshold"),
            .detect_dot_threshold = value<double>(planner, "path_planner.step.detection.detect_dot_threshold"),
            .path_sample_resolution = value<double>(planner, "path_planner.step.detection.path_sample_resolution"),
            .profile_prepare_distance = value<double>(planner, "path_planner.step.execution.profile_prepare_distance"),
            .chassis_activation_distance = value<double>(planner, "path_planner.step.execution.chassis_activation_distance"),
            .fsm_release_distance = value<double>(planner, "path_planner.step.execution.fsm_release_distance"),
            .approach_distance = value<double>(planner, "path_planner.step.mpc_constraints.approach_distance"),
            .gate_transition_distance = value<double>(planner, "path_planner.step.mpc_constraints.gate_transition_distance"),
        },
        .enable_debug = false,
    };
    out.step_mask = {
        .path_align_dot_threshold = value<double>(planner, "path_planner.step_mask.path_align_dot_threshold"),
        .full_effect_radius = value<double>(planner, "path_planner.step_mask.full_effect_radius"),
        .cutoff_radius = value<double>(planner, "path_planner.step_mask.cutoff_radius"),
        .length_num_samples = value<int>(planner, "path_planner.step_mask.length_num_samples"),
    };
    out.a_star = {
        .step_alignment_weight = value<double>(planner, "path_planner.planner.a_star.step_alignment_weight"),
        .obstacle_weight = value<double>(planner, "path_planner.planner.a_star.obstacle_weight"),
        .step_proximity_weight = value<double>(planner, "path_planner.planner.a_star.step_proximity_weight"),
        .step_mode_dot_threshold = value<double>(planner, "path_planner.planner.a_star.step_mode_dot_threshold"),
        .downsampled_waypoint_max_interval = value<int>(planner, "path_planner.planner.a_star.downsampled_waypoint_max_interval"),
        .feasible_threshold = value<int>(planner, "path_planner.planner.a_star.feasible_threshold"),
    };
    const std::string po = "path_planner.path_optimizer";
    out.path_optimizer = {
        .step_norm_threshold = value<double>(planner, po + ".step_norm_threshold"),
        .step_norm_transition = value<double>(planner, po + ".step_norm_transition"),
        .step_detection_samples_per_meter = value<double>(planner, po + ".step_detection_samples_per_meter"),
        .warmup = {
            .obstacle_weight = value<double>(planner, po + ".warmup.obstacle_weight"), .direction_weight = value<double>(planner, po + ".warmup.direction_weight"),
            .step_weight = value<double>(planner, po + ".warmup.step_weight"), .start_end_weight = value<double>(planner, po + ".warmup.start_end_weight"),
            .smoothness_weight = value<double>(planner, po + ".warmup.smoothness_weight"), .samples_per_meter = value<double>(planner, po + ".warmup.samples_per_meter"),
            .max_iterations = value<int>(planner, po + ".warmup.max_iterations"), .max_curvature = value<double>(planner, po + ".warmup.max_curvature"),
            .length_penalty_weight = value<double>(planner, po + ".warmup.length_penalty_weight"), .curvature = curvature_params(planner, po + ".warmup.curvature"),
        },
        .main = {
            .obstacle_weight = value<double>(planner, po + ".main.obstacle_weight"), .direction_weight = value<double>(planner, po + ".main.direction_weight"),
            .step_weight = value<double>(planner, po + ".main.step_weight"), .start_end_weight = value<double>(planner, po + ".main.start_end_weight"),
            .smoothness_weight = value<double>(planner, po + ".main.smoothness_weight"), .samples_per_meter = value<double>(planner, po + ".main.samples_per_meter"),
            .max_iterations = value<int>(planner, po + ".main.max_iterations"), .max_refinement_iterations = value<int>(planner, po + ".main.max_refinement_iterations"),
            .near_max_curvature = value<double>(planner, po + ".main.near_max_curvature"), .far_max_curvature = value<double>(planner, po + ".main.far_max_curvature"),
            .step_extension_distance = value<double>(planner, po + ".main.step_extension_distance"), .step_transition_distance = value<double>(planner, po + ".main.step_transition_distance"),
            .interval_iou_threshold = value<double>(planner, po + ".main.interval_iou_threshold"), .length_penalty_weight = value<double>(planner, po + ".main.length_penalty_weight"),
            .curvature = curvature_params(planner, po + ".main.curvature"),
        },
    };
    out.step_dist_offset = value<double>(ros_params(directory / "path_executor.yaml"), "path_executor.misc.step_dist_offset");
    return out;
}

} // namespace mpc_tuner
