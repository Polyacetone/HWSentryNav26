#include <nav_executor/path_planner/trajectory/step_annotator.hpp>

#include <algorithm>
#include <cmath>
#include <optional>

#include <rclcpp/logging.hpp>

namespace nav_executor::step_annotator {

namespace {

constexpr double ANGLE_EPSILON = 1e-6;

const TraversalMode* lookup_step_rule(
    const StepDirection direction,
    const Eigen::Vector2d& step_enter_pos_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints
) {
    const auto cell = direction_map.geometry.containing_cell(step_enter_pos_map);
    if (!cell) return nullptr;
    const uint8_t label = direction_map.terrain_label_at_cell(*cell);
    const TraversalMode* rule = terrain_constraints.selected_mode(
        label, direction == StepDirection::UP
    );
    return rule && rule->chassis_mode != 0 ? rule : nullptr;
}

} // anonymous namespace

std::vector<StepPlanSegment> build_step_plan(
    const StepDetectionParams& params,
    const MincoTrajectory& path,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    rclcpp::Logger logger
) {
    const double sample_spacing = std::min(
        params.path_sample_resolution, direction_map.geometry.resolution() * 0.5
    );
    const double total_length = path.total_arc_length();
    const int sample_intervals = std::max(
        1, static_cast<int>(std::ceil(total_length / sample_spacing))
    );

    struct ActiveSegment {
        uint8_t label = 0;
        StepDirection direction = StepDirection::UP;
        double step_enter_arc_length = 0.0;
        double last_body_arc_length = 0.0;
        Eigen::Vector2d step_enter_pos_map = Eigen::Vector2d::Zero();
        Eigen::Vector2d last_body_pos_map = Eigen::Vector2d::Zero();
        Eigen::Vector2d dir_map = Eigen::Vector2d::Zero();
    };
    std::optional<ActiveSegment> active;
    std::vector<StepPlanSegment> plan;

    const auto finalize = [&]() {
        if (!active) return;
        const TraversalMode* rule = lookup_step_rule(
            active->direction,
            active->step_enter_pos_map,
            direction_map,
            terrain_constraints
        );
        if (rule) {
            StepPlanSegment segment;
            segment.commit_arc_length = std::max(
                0.0, active->step_enter_arc_length - rule->run_up
            );
            segment.step_enter_arc_length = active->step_enter_arc_length;
            segment.step_exit_arc_length = active->last_body_arc_length;
            segment.step_enter_pos_map = active->step_enter_pos_map;
            segment.step_exit_pos_map = active->last_body_pos_map;
            segment.dir_map = active->dir_map;
            segment.direction = active->direction;
            segment.chassis_command = {
                .mode = rule->chassis_mode,
                .capability = rule->capability,
            };
            segment.traversal_constraint.velocity_window = rule->velocity_window;
            segment.traversal_constraint.dir_map = active->dir_map.normalized();
            segment.terrain_label = active->label;
            segment.requires_high_performance = rule->requires_high_performance;
            plan.push_back(std::move(segment));
        }
        active.reset();
    };

    for (int i = 0; i <= sample_intervals; ++i) {
        const double arc_length = total_length * static_cast<double>(i)
            / static_cast<double>(sample_intervals);
        const TrajSample sample = path.eval_arc_length(arc_length);
        const auto cell = direction_map.geometry.containing_cell(sample.p);
        if (!cell || !direction_map.is_terrain_body_cell(*cell)) {
            finalize();
            continue;
        }
        const uint8_t label = direction_map.terrain_label_at_cell(*cell);
        const Eigen::Vector2d direction = direction_map.raw_direction_at_cell(*cell);
        if (label < static_cast<uint8_t>(TerrainType::SLOPE)) {
            finalize();
            continue;
        }
        if (active && label == active->label) {
            active->last_body_arc_length = arc_length;
            active->last_body_pos_map = sample.p;
            continue;
        }

        finalize();
        const Eigen::Vector2d tangent = sample.dp_dtau;
        if (direction.norm() < ANGLE_EPSILON || tangent.norm() < ANGLE_EPSILON) continue;
        const double alignment = direction.normalized().dot(tangent.normalized());
        if (std::abs(alignment) <= params.detect_dot_threshold) continue;
        active = ActiveSegment {
            .label = label,
            .direction = alignment > 0.0 ? StepDirection::UP : StepDirection::DOWN,
            .step_enter_arc_length = arc_length,
            .last_body_arc_length = arc_length,
            .step_enter_pos_map = sample.p,
            .last_body_pos_map = sample.p,
            .dir_map = direction,
        };
    }
    finalize();
    if (plan.empty()) return plan;

    for (StepPlanSegment& segment : plan) {
        segment.prepare_arc_length = std::max(
            0.0, segment.commit_arc_length - params.profile_prepare_distance
        );
        segment.active_arc_length = std::max(
            0.0, segment.commit_arc_length - params.chassis_activation_distance
        );
        segment.release_arc_length = std::min(
            total_length, segment.step_exit_arc_length + params.fsm_release_distance
        );
    }

    for (size_t i = 1; i < plan.size(); ++i) {
        StepPlanSegment& previous = plan[i - 1];
        StepPlanSegment& current = plan[i];
        if (current.prepare_arc_length < previous.release_arc_length) {
            current.prepare_arc_length = std::min(
                previous.release_arc_length, current.commit_arc_length
            );
            previous.release_arc_length = std::max(
                previous.step_exit_arc_length, current.prepare_arc_length
            );
        }
        current.active_arc_length = std::clamp(
            current.active_arc_length,
            current.prepare_arc_length,
            current.commit_arc_length
        );
        previous.active_arc_length = std::min(
            previous.active_arc_length, previous.commit_arc_length
        );
        if (current.prepare_arc_length < previous.release_arc_length) {
            RCLCPP_WARN(
                logger,
                "Step segments #%zu and #%zu still overlap after arbitration",
                i - 1, i
            );
        }
    }

    for (StepPlanSegment& segment : plan) {
        segment.commit_arc_length = std::min(
            segment.commit_arc_length, segment.step_enter_arc_length
        );
        segment.prepare_arc_length = std::min(
            segment.prepare_arc_length, segment.commit_arc_length
        );
        segment.active_arc_length = std::clamp(
            segment.active_arc_length,
            segment.prepare_arc_length,
            segment.commit_arc_length
        );
        segment.step_exit_arc_length = std::max(
            segment.step_exit_arc_length, segment.step_enter_arc_length
        );
        segment.release_arc_length = std::max(
            segment.release_arc_length, segment.step_exit_arc_length
        );

        StepTraversalConstraint& constraint = segment.traversal_constraint;
        constraint.commit_arc_length = segment.commit_arc_length;
        constraint.step_enter_arc_length = segment.step_enter_arc_length;
        constraint.exit_arc_length = segment.step_exit_arc_length;
        constraint.gate_start_arc_length = std::max(
            segment.prepare_arc_length,
            segment.commit_arc_length - params.gate_transition_distance
        );
        constraint.gate_end_arc_length = std::min(
            segment.release_arc_length,
            segment.step_exit_arc_length + params.gate_transition_distance
        );
    }
    return plan;
}

} // namespace nav_executor::step_annotator
