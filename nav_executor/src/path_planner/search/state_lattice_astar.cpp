#include <nav_executor/path_planner/search/state_lattice_astar.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <queue>
#include <unordered_map>

#include <nav_executor/path_planner/search/grid_utils.hpp>

namespace nav_executor {

namespace {

constexpr double EPS = 1e-9;

double wrap_angle(const double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

int wrap_heading(const int heading, const int bins) {
    const int wrapped = heading % bins;
    return wrapped < 0 ? wrapped + bins : wrapped;
}

SpatialPose advance_pose(
    const SpatialPose& from,
    const double curvature,
    const double length
) {
    SpatialPose to = from;
    const Eigen::Vector2d tangent(std::cos(from.heading), std::sin(from.heading));
    const Eigen::Vector2d normal(-tangent.y(), tangent.x());
    if (std::abs(curvature) <= EPS) {
        to.position += length * tangent;
    } else {
        const double angle = curvature * length;
        to.position += std::sin(angle) / curvature * tangent
            + (1.0 - std::cos(angle)) / curvature * normal;
        to.heading = wrap_angle(from.heading + angle);
    }
    return to;
}

struct LatticePoseKeyHash {
    size_t operator()(const LatticePoseKey& key) const {
        size_t seed = std::hash<int>{}(key.x);
        seed ^= std::hash<int>{}(key.y) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(key.heading) + 0x9e3779b9U
            + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct LatticeKeyHash {
    size_t operator()(const LatticeKey& key) const {
        size_t seed = LatticePoseKeyHash{}({key.x, key.y, key.heading});
        seed ^= std::hash<int>{}(key.speed) + 0x9e3779b9U
            + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct PrimitiveAction {
    double curvature = 0.0;
    double length = 0.0;
};

std::vector<PrimitiveAction> primitive_actions(
    const MotionPrimitiveLibrary::Params& params
) {
    std::vector<PrimitiveAction> actions;
    actions.reserve(13);
    for (size_t band = 0; band < params.band_lengths.size(); ++band) {
        for (size_t magnitude = 0; magnitude < 2; ++magnitude) {
            const double value = params.curvature_magnitudes[2 * band + magnitude];
            actions.push_back({-value, params.band_lengths[band]});
            actions.push_back({value, params.band_lengths[band]});
        }
    }
    actions.push_back({0.0, params.straight_length});
    return actions;
}

MotionPrimitive quantize_primitive(
    const MotionPrimitiveLibrary::Params& params,
    const double start_heading,
    const PrimitiveAction& action,
    double& position_residual,
    double& heading_residual
) {
    const double heading_resolution = 2.0 * std::numbers::pi
        / static_cast<double>(params.heading_bins);
    const SpatialPose nominal_end = advance_pose(
        {{0.0, 0.0}, start_heading}, action.curvature, action.length
    );
    const int endpoint_x = static_cast<int>(std::llround(
        nominal_end.position.x() / params.xy_resolution
    ));
    const int endpoint_y = static_cast<int>(std::llround(
        nominal_end.position.y() / params.xy_resolution
    ));
    const int unwrapped_heading = static_cast<int>(std::llround(
        nominal_end.heading / heading_resolution
    ));
    const int heading_bin = wrap_heading(unwrapped_heading, params.heading_bins);
    const Eigen::Vector2d canonical_position = params.xy_resolution
        * Eigen::Vector2d(endpoint_x, endpoint_y);
    const double canonical_heading = heading_resolution
        * static_cast<double>(unwrapped_heading);
    position_residual = (nominal_end.position - canonical_position).norm();
    heading_residual = std::abs(wrap_angle(
        nominal_end.heading - canonical_heading
    ));

    MotionPrimitive primitive;
    primitive.endpoint = {endpoint_x, endpoint_y, heading_bin};
    primitive.segments = {{action.curvature, action.length}};
    primitive.max_abs_curvature = std::abs(action.curvature);
    return primitive;
}

struct GeometryEdge {
    SpatialPose from;
    SpatialPose to;
    double length = 0.0;
    double curvature = 0.0;
    double s_begin = 0.0;
    double s_end = 0.0;
};

struct TerrainSpan {
    uint8_t label = 0;
    bool going_up = true;
    double s_begin = 0.0;
    double s_end = 0.0;
};

struct ApproachSpan {
    double s_begin = 0.0;
    double s_end = 0.0;
    double gate = 0.0;
    Eigen::Vector2d direction = Eigen::Vector2d::Zero();
    TraversalVelocityWindow velocity_window;
    Eigen::Vector2d heading = Eigen::Vector2d::Zero();
};

struct GeometryResult {
    bool valid = false;
    double total_length = 0.0;
    double max_abs_curvature = 0.0;
    std::vector<GeometryEdge> edges;
    std::vector<TerrainSpan> terrain_spans;
    std::vector<ApproachSpan> approach_spans;
};

bool append_approach_span(
    GeometryResult& result,
    const GuideField& guide_field,
    const ReferencePath& reference_path,
    const Eigen::Vector2i& cell,
    const SpatialPose& edge_start,
    const double curvature,
    const double edge_length,
    const double fraction_begin,
    const double fraction_end,
    const double path_s_begin
) {
    if (fraction_end - fraction_begin <= 1e-12) return true;
    const GuideFieldCell* guide = guide_field.at_cell(cell);
    if (!guide || guide->nearest_reference_index >= reference_path.points.size()) {
        return false;
    }
    const StepApproach& approach = reference_path.points[
        guide->nearest_reference_index
    ].approach;
    if (approach.gate <= 0.0) return true;
    const double middle_fraction = 0.5 * (fraction_begin + fraction_end);
    const double heading_angle = edge_start.heading
        + curvature * edge_length * middle_fraction;
    result.approach_spans.push_back({
        .s_begin = path_s_begin + edge_length * fraction_begin,
        .s_end = path_s_begin + edge_length * fraction_end,
        .gate = approach.gate,
        .direction = approach.direction,
        .velocity_window = approach.velocity_window,
        .heading = Eigen::Vector2d(std::cos(heading_angle), std::sin(heading_angle)),
    });
    return true;
}

bool append_terrain_span(
    GeometryResult& result,
    const DirectionMap& direction_map,
    const Eigen::Vector2i& cell,
    const SpatialPose& edge_start,
    const double curvature,
    const double edge_length,
    const double fraction_begin,
    const double fraction_end,
    const double path_s_begin,
    const double detect_dot_threshold
) {
    if (!direction_map.is_terrain_body_cell(cell)
        || fraction_end - fraction_begin <= 1e-12) {
        return true;
    }
    const uint8_t label = direction_map.terrain_label_at_cell(cell);
    const Eigen::Vector2d raw_direction = direction_map.raw_direction_at_cell(cell);
    if (label < static_cast<uint8_t>(TerrainType::SLOPE)
        || raw_direction.squaredNorm() <= 1e-12) {
        return false;
    }
    const Eigen::Vector2d direction = raw_direction.normalized();
    std::optional<bool> going_up;
    for (const double fraction : {
            fraction_begin,
            0.5 * (fraction_begin + fraction_end),
            fraction_end,
        }) {
        const double heading = edge_start.heading
            + curvature * edge_length * fraction;
        const double alignment = Eigen::Vector2d(
            std::cos(heading), std::sin(heading)
        ).dot(direction);
        if (std::abs(alignment) <= detect_dot_threshold) return false;
        const bool sample_going_up = alignment > 0.0;
        if (going_up && *going_up != sample_going_up) return false;
        going_up = sample_going_up;
    }
    result.terrain_spans.push_back({
        .label = label,
        .going_up = *going_up,
        .s_begin = path_s_begin + edge_length * fraction_begin,
        .s_end = path_s_begin + edge_length * fraction_end,
    });
    return true;
}

GeometryResult build_geometry(
    const SpatialPose& start,
    const std::vector<PrimitiveSegment>& segments,
    const std::optional<SpatialPose>& canonical_endpoint,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const GuideField& guide_field,
    const ReferencePath& reference_path,
    const int occupied_threshold,
    const double detect_dot_threshold,
    const double collision_resolution,
    size_t& corridor_rejections
) {
    GeometryResult result;
    SpatialPose pose = start;
    double path_s = 0.0;
    for (size_t segment_index = 0; segment_index < segments.size(); ++segment_index) {
        const PrimitiveSegment& segment = segments[segment_index];
        result.max_abs_curvature = std::max(
            result.max_abs_curvature, std::abs(segment.curvature)
        );
        const int substeps = std::max(
            1, static_cast<int>(std::ceil(segment.length / collision_resolution))
        );
        const double substep_length = segment.length / static_cast<double>(substeps);
        for (int substep = 0; substep < substeps; ++substep) {
            const SpatialPose previous = pose;
            pose = advance_pose(previous, segment.curvature, substep_length);
            const bool final_substep = segment_index + 1 == segments.size()
                && substep + 1 == substeps;
            if (canonical_endpoint && final_substep) pose = *canonical_endpoint;

            if (!guide_field.contains_map(previous.position)
                || !guide_field.contains_map(pose.position)) {
                ++corridor_rejections;
                return result;
            }
            const auto start_cell = cost_map.geometry.containing_cell(previous.position);
            const auto end_cost = cost_map.sample_map(pose.position);
            if (!start_cell || !end_cost
                || end_cost->value >= static_cast<double>(occupied_threshold)) {
                return result;
            }
            const std::vector<GridCrossing> crossings = trace_grid_crossings(
                cost_map.geometry, previous.position, pose.position
            );
            Eigen::Vector2i cell = *start_cell;
            double previous_fraction = 0.0;
            for (const GridCrossing& crossing : crossings) {
                if (!guide_field.contains(crossing.from)
                    || !guide_field.contains(crossing.to)) {
                    ++corridor_rejections;
                    return result;
                }
                if (!append_terrain_span(
                        result, direction_map, cell, previous,
                        segment.curvature, substep_length,
                        previous_fraction, crossing.fraction,
                        path_s, detect_dot_threshold
                    ) || !append_approach_span(
                        result, guide_field, reference_path, cell, previous,
                        segment.curvature, substep_length,
                        previous_fraction, crossing.fraction, path_s
                    )
                    || !grid_cell_traversable(
                        cost_map, crossing.to, occupied_threshold
                    )
                    || !grid_edge_avoids_corner_cutting(
                        cost_map, crossing.from, crossing.to, occupied_threshold
                    )) {
                    return result;
                }
                cell = crossing.to;
                previous_fraction = crossing.fraction;
            }
            if (!append_terrain_span(
                    result, direction_map, cell, previous,
                    segment.curvature, substep_length,
                    previous_fraction, 1.0, path_s, detect_dot_threshold
                ) || !append_approach_span(
                    result, guide_field, reference_path, cell, previous,
                    segment.curvature, substep_length,
                    previous_fraction, 1.0, path_s
                )) {
                return result;
            }
            result.edges.push_back({
                .from = previous,
                .to = pose,
                .length = substep_length,
                .curvature = segment.curvature,
                .s_begin = path_s,
                .s_end = path_s + substep_length,
            });
            path_s += substep_length;
        }
    }
    result.total_length = path_s;
    result.valid = !result.edges.empty();
    return result;
}

double dynamic_speed_cap(
    const ShapingDynamicsLimits& limits,
    const double max_abs_curvature
) {
    double cap = limits.velocity_max;
    if (max_abs_curvature <= EPS) return cap;
    cap = std::min(cap, limits.angular_velocity_max / max_abs_curvature);
    cap = std::min(
        cap, std::sqrt(limits.lateral_acceleration_max / max_abs_curvature)
    );
    return std::max(cap, 0.0);
}

double acceleration_cap(
    const ShapingDynamicsLimits& limits,
    const double max_abs_curvature
) {
    double cap = limits.tangential_acceleration_max;
    if (max_abs_curvature > EPS) {
        cap = std::min(cap, limits.angular_acceleration_max / max_abs_curvature);
    }
    return std::max(cap, 0.0);
}

bool speed_transition_feasible(
    const GeometryResult& geometry,
    const double velocity_enter,
    const double velocity_exit,
    const ShapingDynamicsLimits& limits,
    const TerrainTraversalConstraints& terrain_constraints,
    size_t& terrain_spans_checked
) {
    if (!geometry.valid || geometry.total_length <= EPS
        || velocity_enter < 0.0 || velocity_exit < 0.0
        || velocity_enter + velocity_exit <= EPS) {
        return false;
    }
    if (std::max(velocity_enter, velocity_exit)
        > dynamic_speed_cap(limits, geometry.max_abs_curvature) + EPS) {
        return false;
    }
    const double enter_squared = velocity_enter * velocity_enter;
    const double exit_squared = velocity_exit * velocity_exit;
    const double acceleration = (exit_squared - enter_squared)
        / (2.0 * geometry.total_length);
    if (std::abs(acceleration)
        > acceleration_cap(limits, geometry.max_abs_curvature) + EPS) {
        return false;
    }
    const auto speed_squared_at = [&](const double arc_length) {
        return enter_squared + (exit_squared - enter_squared)
            * arc_length / geometry.total_length;
    };
    for (const TerrainSpan& span : geometry.terrain_spans) {
        ++terrain_spans_checked;
        const TraversalMode* mode = terrain_constraints.selected_mode(
            span.label, span.going_up
        );
        if (!mode) return false;
        const double lower = mode->velocity_window.min * mode->velocity_window.min;
        const double upper = mode->velocity_window.max * mode->velocity_window.max;
        const double speed_begin = speed_squared_at(span.s_begin);
        const double speed_end = speed_squared_at(span.s_end);
        if (std::min(speed_begin, speed_end) < lower - EPS
            || std::max(speed_begin, speed_end) > upper + EPS) {
            return false;
        }
    }
    return true;
}

double approach_penalty(
    const GeometryResult& geometry,
    const double velocity_enter,
    const double velocity_exit,
    const double alignment_weight,
    const double window_weight
) {
    if (geometry.approach_spans.empty()
        || (alignment_weight <= 0.0 && window_weight <= 0.0)) {
        return 0.0;
    }
    const double enter_squared = velocity_enter * velocity_enter;
    const double exit_squared = velocity_exit * velocity_exit;
    const auto speed_at = [&](const double arc_length) {
        return std::sqrt(std::max(
            enter_squared + (exit_squared - enter_squared)
                * arc_length / geometry.total_length,
            0.0
        ));
    };
    double penalty = 0.0;
    for (const ApproachSpan& span : geometry.approach_spans) {
        const double length = span.s_end - span.s_begin;
        const double speed = speed_at(0.5 * (span.s_begin + span.s_end));
        if (length <= EPS || speed <= EPS) continue;
        const double alignment = span.heading.x() * span.direction.y()
            - span.heading.y() * span.direction.x();
        const double window_violation = std::max({
            span.velocity_window.min - speed,
            speed - span.velocity_window.max,
            0.0,
        });
        penalty += span.gate
            * (alignment_weight * alignment * alignment
                + window_weight * window_violation * window_violation)
            * length / speed;
    }
    return penalty;
}

struct SearchNode {
    LatticeKey key;
    double g = 0.0;
    double relaxation_bias = 0.0;
    int parent = -1;
    int incoming_primitive = -1;
    bool relaxed_root = false;
    bool expanded = false;
};

struct OpenEntry {
    double f = 0.0;
    double g = 0.0;
    int node = -1;

    bool operator>(const OpenEntry& other) const {
        if (f != other.f) return f > other.f;
        if (g != other.g) return g > other.g;
        return node > other.node;
    }
};

struct GeometryCacheKey {
    LatticePoseKey pose;
    int primitive = -1;

    bool operator==(const GeometryCacheKey&) const = default;
};

struct GeometryCacheKeyHash {
    size_t operator()(const GeometryCacheKey& key) const {
        size_t seed = LatticePoseKeyHash{}(key.pose);
        seed ^= std::hash<int>{}(key.primitive) + 0x9e3779b9U
            + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct TerminalCandidate {
    int parent = -1;
    double total_time = 0.0;
    double exit_velocity = 0.0;
    GeometryResult geometry;
};

} // anonymous namespace

MotionPrimitiveLibrary::MotionPrimitiveLibrary(Params params)
    : params_(std::move(params)),
      primitives_by_heading_(static_cast<size_t>(std::max(params_.heading_bins, 0))) {
    const bool magnitudes_valid = std::ranges::all_of(
        params_.curvature_magnitudes,
        [&](const double value) {
            return std::isfinite(value) && value > 0.0
                && value <= params_.curvature_max;
        }
    );
    const bool lengths_valid = std::ranges::all_of(
        params_.band_lengths,
        [](const double value) { return std::isfinite(value) && value > 0.0; }
    );
    if (params_.xy_resolution <= 0.0 || params_.heading_bins <= 0
        || params_.curvature_max <= 0.0 || params_.straight_length <= 0.0
        || !magnitudes_valid || !lengths_valid) {
        error_ = "invalid motion primitive generation parameters";
        return;
    }

    const std::vector<PrimitiveAction> actions = primitive_actions(params_);
    const double heading_resolution = 2.0 * std::numbers::pi
        / static_cast<double>(params_.heading_bins);
    const double position_bound = std::sqrt(0.5) * params_.xy_resolution
        + 16.0 * std::numeric_limits<double>::epsilon();
    const double heading_bound = 0.5 * heading_resolution
        + 16.0 * std::numeric_limits<double>::epsilon();
    for (int heading_bin = 0; heading_bin < params_.heading_bins; ++heading_bin) {
        auto& output = primitives_by_heading_[static_cast<size_t>(heading_bin)];
        output.reserve(actions.size());
        const double start_heading = heading_resolution
            * static_cast<double>(heading_bin);
        for (const PrimitiveAction& action : actions) {
            double position_residual = 0.0;
            double heading_residual = 0.0;
            MotionPrimitive primitive = quantize_primitive(
                params_, start_heading, action,
                position_residual, heading_residual
            );
            if (position_residual > position_bound
                || heading_residual > heading_bound) {
                error_ = "motion primitive quantization exceeded its lattice bound";
                return;
            }
            max_position_residual_ = std::max(
                max_position_residual_, position_residual
            );
            max_heading_residual_ = std::max(
                max_heading_residual_, heading_residual
            );
            output.push_back(std::move(primitive));
        }
        primitive_count_ += output.size();
    }
}

const std::vector<MotionPrimitive>& MotionPrimitiveLibrary::for_heading(
    const int heading
) const {
    return primitives_by_heading_.at(static_cast<size_t>(
        wrap_heading(heading, params_.heading_bins)
    ));
}

SpatialPose StateLatticeAstar::pose_of(
    const LatticeFrame& frame,
    const LatticePoseKey& key
) {
    const double cosine = std::cos(frame.base_heading);
    const double sine = std::sin(frame.base_heading);
    const Eigen::Matrix2d rotation = (
        Eigen::Matrix2d() << cosine, -sine, sine, cosine
    ).finished();
    return {
        .position = frame.origin_map + rotation * (
            frame.xy_resolution * Eigen::Vector2d(key.x, key.y)
        ),
        .heading = wrap_angle(
            frame.base_heading + 2.0 * std::numbers::pi
                * static_cast<double>(wrap_heading(key.heading, frame.heading_bins))
                / static_cast<double>(frame.heading_bins)
        ),
    };
}

SpatialPose StateLatticeAstar::pose_of(
    const LatticeFrame& frame,
    const LatticeKey& key
) {
    return pose_of(frame, LatticePoseKey {key.x, key.y, key.heading});
}

double StateLatticeAstar::speed_of(const int speed_bin) const {
    if (params_.speed_bin_count <= 1) return 0.0;
    return params_.dynamics.velocity_max
        * static_cast<double>(speed_bin)
        / static_cast<double>(params_.speed_bin_count - 1);
}

StateLatticeAstar::Result StateLatticeAstar::search(
    const LatticeFrame& frame,
    const std::vector<SearchRoot>& roots,
    const Eigen::Vector2d& goal_map,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const GuideField& guide_field,
    const ReferencePath& reference_path,
    const int occupied_threshold,
    const double detect_dot_threshold
) const {
    Result result;
    if (!primitive_library_.valid()) {
        result.error = "motion primitive library is invalid: "
            + primitive_library_.error();
        return result;
    }
    if (roots.empty() || !guide_field.ready() || reference_path.points.empty()
        || frame.heading_bins <= 0
        || frame.xy_resolution <= 0.0 || params_.speed_bin_count <= 1
        || params_.guidance_weight < 0.0 || params_.deviation_weight < 0.0
        || params_.heading_weight < 0.0 || params_.speed_weight < 0.0
        || params_.approach_alignment_weight < 0.0
        || params_.approach_window_weight < 0.0
        || params_.max_expansions <= 0
        || !cost_map.geometry.same_geometry(direction_map.geometry)) {
        result.error = "corridor state-lattice configuration or inputs are invalid";
        return result;
    }

    std::vector<SearchNode> nodes;
    nodes.reserve(16384);
    std::unordered_map<LatticeKey, int, LatticeKeyHash> best;
    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<>> open;
    std::unordered_map<GeometryCacheKey, GeometryResult, GeometryCacheKeyHash>
        geometry_cache;

    const auto guidance_at = [&](const SpatialPose& pose, const int speed_bin) {
        const GuideFieldCell* cell = guide_field.at_map(pose.position);
        if (!cell || cell->nearest_reference_index >= reference_path.points.size()) {
            return std::numeric_limits<double>::infinity();
        }
        const ReferencePoint& reference = reference_path.points[
            cell->nearest_reference_index
        ];
        return reference.time_to_goal
            + params_.deviation_weight
                * static_cast<double>(cell->distance_to_reference)
                / params_.dynamics.velocity_max
            + params_.heading_weight
                * std::abs(wrap_angle(pose.heading - reference.heading))
            + params_.speed_weight
                * std::abs(speed_of(speed_bin) - reference.speed)
                / params_.dynamics.tangential_acceleration_max;
    };
    const auto enqueue = [&](const LatticeKey& raw_key,
                             const double g,
                             const double relaxation_bias,
                             const int parent,
                             const int incoming_primitive,
                             const bool relaxed_root) {
        LatticeKey key = raw_key;
        key.heading = wrap_heading(key.heading, frame.heading_bins);
        if (key.speed < 0 || key.speed >= params_.speed_bin_count
            || !std::isfinite(g) || g < 0.0) {
            return false;
        }
        const double guidance = guidance_at(pose_of(frame, key), key.speed);
        if (!std::isfinite(guidance)) return false;
        const auto incumbent = best.find(key);
        if (incumbent != best.end()
            && nodes[static_cast<size_t>(incumbent->second)].g <= g + EPS) {
            return false;
        }
        if (incumbent != best.end()) ++result.diagnostics.improved_states;
        const int index = static_cast<int>(nodes.size());
        nodes.push_back({
            .key = key,
            .g = g,
            .relaxation_bias = relaxation_bias,
            .parent = parent,
            .incoming_primitive = incoming_primitive,
            .relaxed_root = relaxed_root,
            .expanded = false,
        });
        best[key] = index;
        const double f = g + params_.guidance_weight * guidance + relaxation_bias;
        open.push({f, g, index});
        result.diagnostics.open_peak = std::max(
            result.diagnostics.open_peak, open.size()
        );
        ++result.diagnostics.generated_states;
        return true;
    };

    for (const SearchRoot& root : roots) {
        enqueue(
            root.key, 0.0, root.relaxation_bias,
            -1, -1, root.relaxed
        );
    }
    if (open.empty()) {
        result.error = "state-lattice has no valid root in the selected spatial corridor";
        return result;
    }

    const auto geometry_for = [&](const SearchNode& node, const int primitive_index)
        -> const GeometryResult& {
        const GeometryCacheKey cache_key {
            .pose = {node.key.x, node.key.y, node.key.heading},
            .primitive = primitive_index,
        };
        const auto found = geometry_cache.find(cache_key);
        if (found != geometry_cache.end()) {
            ++result.diagnostics.geometry_cache_hits;
            return found->second;
        }
        const MotionPrimitive& primitive = primitive_library_.for_heading(
            node.key.heading
        ).at(static_cast<size_t>(primitive_index));
        const LatticePoseKey successor_pose {
            .x = node.key.x + primitive.endpoint.x,
            .y = node.key.y + primitive.endpoint.y,
            .heading = primitive.endpoint.heading,
        };
        GeometryResult geometry = build_geometry(
            pose_of(frame, node.key),
            primitive.segments,
            pose_of(frame, successor_pose),
            cost_map,
            direction_map,
            guide_field,
            reference_path,
            occupied_threshold,
            detect_dot_threshold,
            params_.collision_check_resolution,
            result.diagnostics.corridor_rejections
        );
        return geometry_cache.emplace(cache_key, std::move(geometry)).first->second;
    };

    std::optional<TerminalCandidate> terminal;
    while (!open.empty()) {
        const OpenEntry entry = open.top();
        open.pop();
        if (entry.node < 0 || static_cast<size_t>(entry.node) >= nodes.size()) {
            ++result.diagnostics.stale_queue_entries;
            continue;
        }
        SearchNode& selected = nodes[static_cast<size_t>(entry.node)];
        const auto active = best.find(selected.key);
        if (active == best.end() || active->second != entry.node || selected.expanded) {
            ++result.diagnostics.stale_queue_entries;
            continue;
        }
        if (result.diagnostics.expansions >= params_.max_expansions) {
            result.error = "state-lattice expansion limit reached in the selected spatial corridor";
            break;
        }
        selected.expanded = true;
        const SearchNode current = selected;
        const SpatialPose current_pose = pose_of(frame, current.key);
        const double velocity_enter = speed_of(current.key.speed);
        ++result.diagnostics.expansions;

        if ((goal_map - current_pose.position).norm() <= params_.goal_tolerance) {
            ++result.diagnostics.terminal_attempts;
            const Eigen::Vector2d displacement = goal_map - current_pose.position;
            const double connection_length = displacement.norm();
            if (connection_length <= EPS) {
                terminal = TerminalCandidate {
                    .parent = entry.node,
                    .total_time = current.g,
                    .exit_velocity = velocity_enter,
                    .geometry = {},
                };
                break;
            }
            if (connection_length <= params_.goal_connection_max_length) {
                const double connector_heading = std::atan2(
                    displacement.y(), displacement.x()
                );
                const SpatialPose connector_start {
                    .position = current_pose.position,
                    .heading = connector_heading,
                };
                const SpatialPose endpoint {
                    .position = goal_map,
                    .heading = connector_heading,
                };
                GeometryResult geometry = build_geometry(
                    connector_start,
                    {{0.0, connection_length}},
                    endpoint,
                    cost_map,
                    direction_map,
                    guide_field,
                    reference_path,
                    occupied_threshold,
                    detect_dot_threshold,
                    params_.collision_check_resolution,
                    result.diagnostics.corridor_rejections
                );
                if (geometry.valid) {
                    double best_edge_cost = std::numeric_limits<double>::infinity();
                    double best_exit_velocity = 0.0;
                    for (int speed_bin = 0;
                         speed_bin < params_.speed_bin_count; ++speed_bin) {
                        const double velocity_exit = speed_of(speed_bin);
                        if (!speed_transition_feasible(
                                geometry,
                                velocity_enter,
                                velocity_exit,
                                params_.dynamics,
                                terrain_constraints,
                                result.diagnostics.terrain_spans_checked
                            )) {
                            continue;
                        }
                        const double duration = 2.0 * geometry.total_length
                            / (velocity_enter + velocity_exit);
                        const double edge_cost = duration + approach_penalty(
                            geometry, velocity_enter, velocity_exit,
                            params_.approach_alignment_weight,
                            params_.approach_window_weight
                        );
                        if (edge_cost < best_edge_cost) {
                            best_edge_cost = edge_cost;
                            best_exit_velocity = velocity_exit;
                        }
                    }
                    if (std::isfinite(best_edge_cost)) {
                        terminal = TerminalCandidate {
                            .parent = entry.node,
                            .total_time = current.g + best_edge_cost,
                            .exit_velocity = best_exit_velocity,
                            .geometry = std::move(geometry),
                        };
                        break;
                    }
                }
            }
        }

        const auto& primitives = primitive_library_.for_heading(current.key.heading);
        for (size_t primitive_index = 0;
             primitive_index < primitives.size(); ++primitive_index) {
            const MotionPrimitive& primitive = primitives[primitive_index];
            if (velocity_enter > dynamic_speed_cap(
                    params_.dynamics, primitive.max_abs_curvature
                ) + EPS) {
                continue;
            }
            const GeometryResult& geometry = geometry_for(
                current, static_cast<int>(primitive_index)
            );
            if (!geometry.valid) continue;
            for (int speed_bin = 1; speed_bin < params_.speed_bin_count; ++speed_bin) {
                ++result.diagnostics.transition_candidates;
                const double velocity_exit = speed_of(speed_bin);
                if (!speed_transition_feasible(
                        geometry,
                        velocity_enter,
                        velocity_exit,
                        params_.dynamics,
                        terrain_constraints,
                        result.diagnostics.terrain_spans_checked
                    )) {
                    continue;
                }
                const LatticeKey successor {
                    .x = current.key.x + primitive.endpoint.x,
                    .y = current.key.y + primitive.endpoint.y,
                    .heading = primitive.endpoint.heading,
                    .speed = speed_bin,
                };
                const double duration = 2.0 * geometry.total_length
                    / (velocity_enter + velocity_exit);
                const double edge_cost = duration + approach_penalty(
                    geometry, velocity_enter, velocity_exit,
                    params_.approach_alignment_weight,
                    params_.approach_window_weight
                );
                enqueue(
                    successor,
                    current.g + edge_cost,
                    current.relaxation_bias,
                    entry.node,
                    static_cast<int>(primitive_index),
                    false
                );
            }
        }
    }

    result.diagnostics.geometry_cache_entries = geometry_cache.size();
    if (!terminal) {
        if (result.error.empty()) {
            result.error = "state-lattice found no kinodynamic path in the selected spatial corridor";
        }
        return result;
    }

    std::vector<int> chain;
    for (int index = terminal->parent; index >= 0;
         index = nodes[static_cast<size_t>(index)].parent) {
        chain.push_back(index);
    }
    std::reverse(chain.begin(), chain.end());
    if (chain.empty()) {
        result.error = "corridor state-lattice produced an empty parent chain";
        return result;
    }
    const SearchNode& root = nodes[static_cast<size_t>(chain.front())];
    result.diagnostics.selected_relaxed_root = root.relaxed_root;
    result.diagnostics.selected_relaxation_bias = root.relaxation_bias;
    result.diagnostics.search_time = terminal->total_time;

    const SpatialPose root_pose = pose_of(frame, root.key);
    const Eigen::Vector2d root_tangent(
        std::cos(root_pose.heading), std::sin(root_pose.heading)
    );
    result.witness.positions.push_back(root_pose.position);
    result.witness.tangents.push_back(root_tangent);
    result.witness.curvatures.push_back(0.0);
    const auto append_geometry = [&](const GeometryResult& geometry,
                                     const double velocity_start,
                                     const double velocity_end) {
        const double start_squared = velocity_start * velocity_start;
        const double end_squared = velocity_end * velocity_end;
        const auto velocity_at = [&](const double arc_length) {
            return std::sqrt(std::max(
                start_squared + (end_squared - start_squared)
                    * arc_length / geometry.total_length,
                0.0
            ));
        };
        for (const GeometryEdge& edge : geometry.edges) {
            if (result.witness.positions.size() == 1) {
                result.witness.curvatures.front() = edge.curvature;
            }
            const double from_speed = velocity_at(edge.s_begin);
            const double to_speed = velocity_at(edge.s_end);
            const Eigen::Vector2d tangent(
                std::cos(edge.to.heading), std::sin(edge.to.heading)
            );
            result.witness.positions.push_back(edge.to.position);
            result.witness.tangents.push_back(tangent);
            result.witness.curvatures.push_back(edge.curvature);
            result.witness.durations.push_back(
                2.0 * edge.length / (from_speed + to_speed)
            );
        }
    };
    for (size_t i = 1; i < chain.size(); ++i) {
        const SearchNode& parent = nodes[static_cast<size_t>(chain[i - 1])];
        const SearchNode& child = nodes[static_cast<size_t>(chain[i])];
        const GeometryResult& geometry = geometry_for(
            parent, child.incoming_primitive
        );
        append_geometry(
            geometry, speed_of(parent.key.speed), speed_of(child.key.speed)
        );
    }
    const SearchNode& terminal_parent = nodes[static_cast<size_t>(terminal->parent)];
    if (terminal->geometry.valid) {
        append_geometry(
            terminal->geometry,
            speed_of(terminal_parent.key.speed),
            terminal->exit_velocity
        );
    }
    if (result.witness.durations.empty()
        || result.witness.positions.size() != result.witness.tangents.size()
        || result.witness.positions.size() != result.witness.curvatures.size()
        || result.witness.positions.size() != result.witness.durations.size() + 1) {
        result.error = "corridor state-lattice witness reconstruction failed";
        return result;
    }
    result.success = true;
    return result;
}

} // namespace nav_executor
