#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/path_planner/search/directed_grid_astar.hpp>
#include <nav_executor/path_planner/search/state_lattice_astar.hpp>

namespace nav_executor {

class LayeredRoutePlanner {
public:
    struct StartYawRelaxationParams {
        double speed_threshold = 0.5;
        double root_penalty = 2.0;
        double yaw_penalty = 0.5;
    };

    struct Params {
        DirectedGridAstar::Params grid_astar;
        StateLatticeAstar::Params state_lattice;
        double lattice_xy_resolution = 0.05;
        int lattice_heading_bins = 80;
        StartYawRelaxationParams start_yaw_relaxation;
        int occupied_threshold = 100;
        double detect_dot_threshold = 0.5;
    };

    struct Diagnostics {
        int global_expansions = 0;
        size_t global_open_peak = 0;
        int lattice_expansions = 0;
        int lattice_labels = 0;
        int lattice_dominated = 0;
        int lattice_transition_checks = 0;
        int lattice_terminal_attempts = 0;
        int rejected_portal_terminals = 0;
        size_t lattice_open_peak = 0;
        size_t lattice_anchor_queue_peak = 0;
        size_t lattice_pending_focal_queue_peak = 0;
        size_t lattice_focal_queue_peak = 0;
        size_t lattice_stale_queue_entries = 0;
        int terrain_expansions = 0;
        int terrain_reachability_expansions = 0;
        size_t terrain_reachability_open_peak = 0;
        size_t passage_count = 0;
        std::vector<size_t> portal_sizes;
        size_t unreachable_portal_transitions = 0;
        std::vector<TerrainRegionId> terrain_regions;
        std::vector<uint8_t> terrain_labels;
        bool selected_relaxed_root = false;
        double selected_root_cost = 0.0;
        double lattice_search_cost = 0.0;
    };

    struct Result {
        SpatialRoute route;
        SpeedSquaredInterval initial_speed;
        std::vector<Eigen::Vector2d> global_raw_path;
        Diagnostics diagnostics;
        bool success = false;
        std::string error;
    };

    LayeredRoutePlanner(
        Params params,
        const MotionPrimitiveLibrary& primitive_library
    ) : params_(params),
        primitive_library_(primitive_library),
        speed_(params.state_lattice.dynamics) {}

    [[nodiscard]] Result search(
        const Eigen::Vector2d& start_map,
        double start_yaw,
        double start_velocity,
        const Eigen::Vector2d& goal_map,
        const CostMap& planning_cost_map,
        const DirectionMap& direction_map,
        const TerrainTraversalConstraints& terrain_constraints
    ) const;

private:
    Params params_;
    const MotionPrimitiveLibrary& primitive_library_;
    SpeedReachability speed_;
};

} // namespace nav_executor
