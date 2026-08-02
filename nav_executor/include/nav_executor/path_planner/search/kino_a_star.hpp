#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/path_planner/search/guide_field.hpp>
#include <nav_executor/path_planner/trajectory/shaping_dynamics.hpp>

namespace nav_executor {

struct LatticeFrame {
    Eigen::Vector2d origin_map = Eigen::Vector2d::Zero();
    double base_heading = 0.0;
    double xy_resolution = 0.05;
    int heading_bins = 80;
};

struct LatticePoseKey {
    int x = 0;
    int y = 0;
    int heading = 0;

    bool operator==(const LatticePoseKey&) const = default;
};

struct LatticeKey {
    int x = 0;
    int y = 0;
    int heading = 0;
    int speed = 0;

    bool operator==(const LatticeKey&) const = default;
};

struct SpatialPose {
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    double heading = 0.0;
};

struct PrimitiveSegment {
    double curvature = 0.0;
    double length = 0.0;
};

struct MotionPrimitive {
    LatticePoseKey endpoint;
    std::vector<PrimitiveSegment> segments;
    double max_abs_curvature = 0.0;
};

struct SpeedWitness {
    std::vector<Eigen::Vector2d> positions;
    std::vector<Eigen::Vector2d> tangents;
    std::vector<double> curvatures;
    std::vector<double> durations;
};

class MotionPrimitiveLibrary {
public:
    struct Params {
        double xy_resolution = 0.05;
        int heading_bins = 80;
        std::array<double, 6> curvature_magnitudes {
            1.0, 4.0,
            0.25, 1.0,
            0.0625, 0.25,
        };
        std::array<double, 3> band_lengths {0.32, 0.63, 1.26};
        double straight_length = 1.26;
        double curvature_max = 4.0;
    };

    explicit MotionPrimitiveLibrary(Params params);

    [[nodiscard]] const std::vector<MotionPrimitive>& for_heading(int heading) const;
    [[nodiscard]] bool valid() const { return error_.empty(); }
    [[nodiscard]] const std::string& error() const { return error_; }
    [[nodiscard]] size_t primitive_count() const { return primitive_count_; }
    [[nodiscard]] double max_position_residual() const { return max_position_residual_; }
    [[nodiscard]] double max_heading_residual() const { return max_heading_residual_; }

private:
    Params params_;
    std::vector<std::vector<MotionPrimitive>> primitives_by_heading_;
    std::string error_;
    size_t primitive_count_ = 0;
    double max_position_residual_ = 0.0;
    double max_heading_residual_ = 0.0;
};

class KinoAStar {
public:
    struct Params {
        struct StartYawRelaxationParams {
            int root_count = 80;
            double root_bias_seconds = 0.0;
            double yaw_bias_seconds_per_rad = 0.0;
            double max_discarded_velocity = 0.0;
        } start_yaw_relaxation;

        ShapingDynamicsLimits dynamics;
        int speed_bin_count = 13;
        double collision_check_resolution = 0.075;
        double goal_connection_max_distance = 1.0;
        double guidance_weight = 1.0;
        double deviation_weight = 1.0;
        double heading_weight = 0.25;
        double speed_weight = 0.5;
        double approach_alignment_weight = 4.0;
        double approach_window_weight = 1.0;
        int max_expansions = 500000;
    };

    struct SearchRoot {
        LatticeKey key;
        double relaxation_bias = 0.0;
        bool relaxed = false;
    };

    struct Diagnostics {
        int expansions = 0;
        int generated_states = 0;
        int improved_states = 0;
        int transition_candidates = 0;
        int terminal_attempts = 0;
        size_t open_peak = 0;
        size_t stale_queue_entries = 0;
        size_t geometry_cache_entries = 0;
        size_t geometry_cache_hits = 0;
        size_t terrain_spans_checked = 0;
        size_t corridor_rejections = 0;
        bool selected_relaxed_root = false;
        double selected_relaxation_bias = 0.0;
        double search_time = 0.0;
    };

    struct Result {
        SpeedWitness witness;
        Diagnostics diagnostics;
        bool success = false;
        std::string error;
    };

    KinoAStar(
        Params params,
        const MotionPrimitiveLibrary& primitive_library
    ) : params_(params), primitive_library_(primitive_library) {}

    [[nodiscard]] Result search(
        const LatticeFrame& frame,
        const std::vector<SearchRoot>& roots,
        const Eigen::Vector2d& goal_map,
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const TerrainTraversalConstraints& terrain_constraints,
        const GuideField& guide_field,
        const ReferencePath& reference_path,
        int occupied_threshold,
        double min_terrain_alignment_cosine
    ) const;

    [[nodiscard]] double speed_of(int speed_bin) const;

    [[nodiscard]] static SpatialPose pose_of(
        const LatticeFrame& frame,
        const LatticePoseKey& key
    );

    [[nodiscard]] static SpatialPose pose_of(
        const LatticeFrame& frame,
        const LatticeKey& key
    );

private:
    Params params_;
    const MotionPrimitiveLibrary& primitive_library_;
};

} // namespace nav_executor
