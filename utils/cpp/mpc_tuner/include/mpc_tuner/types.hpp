#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/annotated_path.hpp>
#include <nav_executor/path_executor/step_controller.hpp>
#include <nav_executor/path_executor/solver/mpc_types.hpp>
#include <nav_executor/path_planner/bspline_optimizer.hpp>
#include <nav_executor/path_planner/path_planner.hpp>
#include <nav_executor/path_planner/step_routing_mask.hpp>

namespace mpc_tuner {

struct ScenarioSpec {
    std::string name;
    std::string split;
    Eigen::Vector3d start_pose = Eigen::Vector3d::Zero();
    Eigen::Vector2d goal = Eigen::Vector2d::Zero();
    double timeout = 20.0;
    std::vector<uint64_t> seeds;
};

struct MapSnapshot {
    int width = 0;
    int height = 0;
    double resolution = 0.0;
    double origin_x = 0.0;
    double origin_y = 0.0;
    std::vector<uint8_t> global_cost_data;
    std::vector<uint8_t> direction_image_data;
};

struct StoredRoute {
    ScenarioSpec spec;
    std::vector<Eigen::Vector2d> spline_control_points;
    std::vector<nav_executor::StepPlanSegment> step_segments;
    std::vector<uint8_t> control_cost_data;
};

struct SceneBundle {
    static constexpr uint32_t FORMAT_VERSION = 1;

    uint32_t format_version = FORMAT_VERSION;
    std::string split;
    std::string created_at;
    MapSnapshot map;
    std::vector<StoredRoute> routes;
};

struct CompiledScenario {
    ScenarioSpec spec;
    nav_executor::AnnotatedPath::ConstPtr path;
    nav_executor::CostMap::ConstPtr global_cost_map;
    nav_executor::CostMap::ConstPtr control_cost_map;
    nav_executor::DirectionMap::ConstPtr direction_map;
};

struct StudyConfig {
    uint64_t seed = 2026;
    int population_size = 32;
    double elite_fraction = 0.2;
    int generations = 20;
    double initial_std = 0.45;
    double min_std = 0.03;
    int parallel_workers = 0;
    double progress_interval_seconds = 10.0;
};

struct EpisodeConfig {
    double default_timeout = 20.0;
    double goal_radius = 0.5;
    double target_arrival_speed = 0.1;
    double acceptable_arrival_speed = 0.3;
    double high_cost_threshold = 30.0;
    double lethal_cost_threshold = 250.0;
    double severe_step_heading_error = 0.5235987755982988;
    double severe_step_speed_margin = 0.2;
};

struct SearchRange {
    std::string name;
    double lower = 0.0;
    double upper = 0.0;
    bool logarithmic = true;
};

inline constexpr size_t PARAMETER_COUNT = 10;

struct TunerConfig {
    StudyConfig study;
    EpisodeConfig episode;
    std::array<SearchRange, PARAMETER_COUNT> search_ranges;
};

struct RuntimeConfig {
    struct AStarConfig {
        double step_alignment_weight = 0.0;
        double obstacle_weight = 0.0;
        double step_proximity_weight = 0.0;
        double step_mode_dot_threshold = 0.0;
        int downsampled_waypoint_max_interval = 0;
        int feasible_threshold = 0;
    };

    nav_executor::MPCParams mpc;
    nav_executor::TerrainProfiles terrain_profiles;
    nav_executor::ProfileBlendParams profile_blend;
    nav_executor::PlannerConfig planner;
    nav_executor::BSplineOptimizer::Params path_optimizer;
    nav_executor::StepRoutingMaskParams step_mask;
    AStarConfig a_star;
    double step_dist_offset = 0.1;
};

struct ParameterCandidate {
    std::array<double, PARAMETER_COUNT> normalized {};
    std::array<double, PARAMETER_COUNT> values {};
};

struct EpisodeMetrics {
    std::string scenario_name;
    uint64_t seed = 0;
    bool reached = false;
    bool solver_failed = false;
    double elapsed_time = 0.0;
    double final_progress = 0.0;
    double arrival_speed = 0.0;
    double arrival_omega = 0.0;
    double high_cost_integral = 0.0;
    double lethal_cost_duration = 0.0;
    double max_cost = 0.0;
    double step_speed_violation = 0.0;
    double step_alignment_violation = 0.0;
    int severe_step_violations = 0;
    double cross_track_squared_integral = 0.0;
    double heading_error_squared_integral = 0.0;
    double command_dv_squared_sum = 0.0;
    double command_domega_squared_sum = 0.0;
    int control_samples = 0;
};

struct Fitness {
    int failed_scenarios = 0;
    double progress_deficit = 0.0;
    int severe_environment_violations = 0;
    int severe_step_violations = 0;
    int fast_arrival_count = 0;
    double arrival_speed_excess = 0.0;
    double soft_quality_cost = 0.0;

    [[nodiscard]] auto ordering_key() const {
        return std::tuple {
            failed_scenarios,
            progress_deficit,
            severe_environment_violations,
            severe_step_violations,
            fast_arrival_count,
            arrival_speed_excess,
            soft_quality_cost,
        };
    }

    bool operator<(const Fitness& other) const { return ordering_key() < other.ordering_key(); }
};

} // namespace mpc_tuner
