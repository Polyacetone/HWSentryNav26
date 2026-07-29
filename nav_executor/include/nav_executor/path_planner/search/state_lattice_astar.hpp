#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/path_planner/search/speed_reachability.hpp>
#include <nav_executor/path_planner/search/terrain_topology.hpp>

namespace nav_executor {

struct LatticeFrame {
    Eigen::Vector2d origin_map = Eigen::Vector2d::Zero();
    double base_heading = 0.0;
    double xy_resolution = 0.05;
    int heading_bins = 80;
};

struct LatticeKey {
    int x = 0;
    int y = 0;
    int heading = 0;

    bool operator==(const LatticeKey&) const = default;
};

struct PrimitiveSegment {
    double curvature = 0.0;
    double length = 0.0;
};

struct MotionPrimitive {
    LatticeKey endpoint_delta;
    std::vector<PrimitiveSegment> segments;
    double nominal_curvature = 0.0;
};

struct PointStopTarget {
    Eigen::Vector2d position_map = Eigen::Vector2d::Zero();
};

struct PassingPortalTarget {
    DirectedPortal portal;
};

using SearchTarget = std::variant<PointStopTarget, PassingPortalTarget>;

class MotionPrimitiveLibrary {
public:
    struct Params {
        double xy_resolution = 0.05;
        int heading_bins = 80;
        int nominal_curvature_samples = 11;
        double curvature_max = 4.0;
        double primitive_length = 0.3;
        double endpoint_position_tolerance = 1e-9;
        double endpoint_heading_tolerance = 1e-9;
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

class StateLatticeAstar {
public:
    struct Params {
        ShapingDynamicsLimits dynamics;
        double curvature_max = 4.0;
        double collision_check_resolution = 0.075;
        double goal_connection_max_length = 1.0;
        double goal_tolerance = 0.3;
        double focal_suboptimality = 1.1;
        int max_expansions = 500000;
    };

    struct SearchRoot {
        LatticeKey key;
        SpeedSquaredInterval reachable_speed;
        double initial_cost = 0.0;
        bool relaxed = false;
    };

    struct Result {
        SpatialRoute route;
        SpeedSquaredInterval terminal_speed;
        std::optional<BoundaryTransition> portal_transition;
        bool success = false;
        int expansions = 0;
        int generated_labels = 0;
        int dominated_labels = 0;
        int transition_checks = 0;
        int terminal_attempts = 0;
        int rejected_portal_terminals = 0;
        size_t open_peak = 0;
        size_t anchor_queue_peak = 0;
        size_t pending_focal_queue_peak = 0;
        size_t focal_queue_peak = 0;
        size_t stale_queue_entries = 0;
        bool selected_relaxed_root = false;
        double selected_root_cost = 0.0;
        double selected_search_cost = 0.0;
        std::string error;
    };

    StateLatticeAstar(
        Params params,
        const MotionPrimitiveLibrary& primitive_library
    ) : params_(params), primitive_library_(primitive_library), speed_(params.dynamics) {}

    [[nodiscard]] Result search(
        const LatticeFrame& frame,
        const std::vector<SearchRoot>& roots,
        const SearchTarget& target,
        const CostMap& cost_map,
        const DirectionMap& direction_map,
        const TerrainTraversalConstraints& terrain_constraints,
        int occupied_threshold,
        double detect_dot_threshold,
        const std::vector<Eigen::Vector2d>& reference_path
    ) const;

    [[nodiscard]] static SpatialPose pose_of(
        const LatticeFrame& frame,
        const LatticeKey& key
    );

private:
    Params params_;
    const MotionPrimitiveLibrary& primitive_library_;
    SpeedReachability speed_;
};

} // namespace nav_executor
