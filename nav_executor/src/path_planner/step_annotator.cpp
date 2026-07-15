#include <nav_executor/path_planner/step_annotator.hpp>

#include <algorithm>
#include <optional>

#include <rclcpp/logging.hpp>

namespace nav_executor::step_annotator {

namespace {

constexpr double ANGLE_EPSILON = 1e-6;

double advance_path_u_by_distance(const StepDetectionParams& p, const MincoTrajectory& path, const double start_u, const double distance) {
    const double resolution = std::max(1e-3, p.path_sample_resolution);
    double u = std::clamp(start_u, 0.0, 1.0);
    double travelled = 0.0;

    while (u < 1.0 && travelled < distance) {
        const Eigen::Vector2d d1 = path.tangent(u);
        const double speed = d1.norm();
        if (speed < 1e-12) {
            u = std::min(1.0, u + 1e-3);
            continue;
        }
        const double du = resolution / speed;
        const double next_u = std::min(1.0, u + du);
        travelled += (path.position(next_u) - path.position(u)).norm();
        u = next_u;
    }
    return u;
}

double retreat_path_u_by_distance(const StepDetectionParams& p, const MincoTrajectory& path, const double start_u, const double distance) {
    const double resolution = std::max(1e-3, p.path_sample_resolution);
    double u = std::clamp(start_u, 0.0, 1.0);
    double travelled = 0.0;

    while (u > 0.0 && travelled < distance) {
        const Eigen::Vector2d d1 = path.tangent(u);
        const double speed = d1.norm();
        if (speed < 1e-12) {
            u = std::max(0.0, u - 1e-3);
            continue;
        }
        const double du = resolution / speed;
        const double next_u = std::max(0.0, u - du);
        travelled += (path.position(u) - path.position(next_u)).norm();
        u = next_u;
    }

    if (travelled < distance) {
        const double speed = path.tangent(0.0).norm();
        if (speed > 1e-12) {
            u = -(distance - travelled) / speed;
        }
    }

    return u;
}

const TerrainStepRule* lookup_step_rule(
    const StepDirection direction,
    const Eigen::Vector2d& step_enter_pos_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints
) {
    const Eigen::Vector2d g = direction_map.map_coord_to_grid(step_enter_pos_map);
    const uint8_t label = direction_map.terrain_at(g);
    const TerrainStepRule* rule = terrain_constraints.selected_mode(label, direction == StepDirection::UP);
    if (!rule || rule->chassis_mode == 0) return nullptr;
    return rule;
}

} // anonymous namespace

// 设计选择：基于 terrain label 扫描（而非采样点间 gap 检测）。
// 遍历均匀 u 采样点，跟踪地形标签的变化：
//   - label >= SLOPE 且与上一个采样点不同 → 结束上一段（如有），
//     如果路径在此处的穿越方向有效则开启新段
//   - label < SLOPE → 结束上一段
// 段边界完全由 terrain label 在路径上的连续分布决定，
// 不依赖样条参数化质量或 arc_length 积分。
// 方向过滤（detect_dot_threshold）仅在段入口处做一次，
// 避免每个采样点逐个判断带来的偶发碎片化。
std::vector<StepPlanSegment> build_step_plan(
    const StepDetectionParams& p,
    const MincoTrajectory& path,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    rclcpp::Logger logger
) {
    const double resolution = std::max(1e-3, p.path_sample_resolution);

    // 估算路径长度，确定均匀 u 采样点数
    double estimated_length = 0.0;
    Eigen::Vector2d prev = path.position(0.0);
    constexpr int length_estimate_samples = 100;
    for (int i = 1; i <= length_estimate_samples; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(length_estimate_samples);
        const Eigen::Vector2d pos = path.position(u);
        estimated_length += (pos - prev).norm();
        prev = pos;
    }

    const int sample_count = std::max(2, static_cast<int>(std::ceil(estimated_length / resolution)) + 1);
    const double u_step = 1.0 / static_cast<double>(sample_count - 1);

    struct ActiveSegment {
        int start_index;
        uint8_t label;
        StepDirection direction;
        Eigen::Vector2d step_enter_pos_map;
        Eigen::Vector2d dir_map;
    };
    std::optional<ActiveSegment> active;
    std::vector<StepPlanSegment> plan;

    auto finalize = [&](int end_index) {
        if (!active) return;

        const double step_exit_u = static_cast<double>(end_index - 1) * u_step;
        const Eigen::Vector2d step_exit_pos = path.position(step_exit_u);
        const double step_enter_u = static_cast<double>(active->start_index) * u_step;

        const TerrainStepRule* rule = lookup_step_rule(
            active->direction, active->step_enter_pos_map, direction_map, terrain_constraints
        );
        if (rule) {
            // commit_u：物理边缘上游回退 run_up，作为所有运行时约束的锚点。
            const double commit_u = retreat_path_u_by_distance(p, path, step_enter_u, rule->run_up);

            StepPlanSegment seg;
            seg.commit_u = commit_u;
            seg.step_enter_u = step_enter_u;
            seg.step_exit_u = step_exit_u;
            seg.step_enter_pos_map = active->step_enter_pos_map;
            seg.step_exit_pos_map = step_exit_pos;
            seg.dir_map = active->dir_map;
            seg.direction = active->direction;
            seg.chassis_command = StepChassisCommand {
                .mode = rule->chassis_mode,
                .capability = rule->capability,
            };
            seg.traversal_constraint.speed_min = rule->speed.min;
            seg.traversal_constraint.speed_max = rule->speed.max;
            seg.traversal_constraint.commit_u = commit_u;
            seg.traversal_constraint.step_enter_u = step_enter_u;
            seg.traversal_constraint.exit_u = step_exit_u;
            seg.traversal_constraint.dir_map = active->dir_map.normalized();
            seg.terrain_label = active->label;
            seg.requires_high_performance = rule->requires_high_performance;
            plan.push_back(std::move(seg));
        }
        active.reset();
    };

    for (int i = 0; i < sample_count; ++i) {
        const double u = static_cast<double>(i) * u_step;
        const Eigen::Vector2d pos = path.position(u);
        const Eigen::Vector2d g = direction_map.map_coord_to_grid(pos);

        if (!direction_map.is_valid_coord(g)) {
            finalize(i);
            continue;
        }

        const uint8_t label = direction_map.terrain_at(g);
        const Eigen::Vector2d dir = direction_map.interpolate(g);

        if (label < static_cast<uint8_t>(TerrainType::SLOPE)
            || dir.norm() < p.detect_norm_threshold) {
            finalize(i);
            continue;
        }

        if (active && label == active->label) {
            continue;
        }

        finalize(i);

        const Eigen::Vector2d tangent = path.tangent(u);
        if (dir.norm() < ANGLE_EPSILON) continue;

        const double dot = dir.normalized().dot(tangent.normalized());
        if (std::abs(dot) <= p.detect_dot_threshold) continue;

        const StepDirection direction = dot > 0.0 ? StepDirection::UP : StepDirection::DOWN;

        active = ActiveSegment{
            .start_index = i,
            .label = label,
            .direction = direction,
            .step_enter_pos_map = pos,
            .dir_map = dir,
        };
    }

    finalize(sample_count);

    if (plan.empty()) return plan;

    // ── 计算 prepare / active / release u 边界 ──
    // prepare / active 自 commit_u（而非物理边缘）上游回退：commit 是上位机视角的台阶起点，
    // 能力档 blend 与抬腿指令都以它为基准，从而保证 prepare ≤ active ≤ commit 的余量恒定，
    // 与助跑提前量 run_up 解耦。
    for (StepPlanSegment& segment : plan) {
        segment.prepare_u = retreat_path_u_by_distance(p, path, segment.commit_u, p.profile_prepare_distance);
        segment.active_u = retreat_path_u_by_distance(p, path, segment.commit_u, p.chassis_activation_distance);
        segment.release_u = advance_path_u_by_distance(p, path, segment.step_exit_u, p.fsm_release_distance);
    }

    // ── 段间重叠仲裁 ──
    for (size_t i = 1; i < plan.size(); ++i) {
        StepPlanSegment& prev_seg = plan[i - 1];
        StepPlanSegment& cur = plan[i];

        if (cur.prepare_u >= prev_seg.release_u) {
            if (cur.active_u < cur.prepare_u) cur.active_u = cur.prepare_u;
            if (prev_seg.active_u > prev_seg.commit_u) prev_seg.active_u = prev_seg.commit_u;
            continue;
        }

        cur.prepare_u = std::min(prev_seg.release_u, cur.commit_u);
        if (cur.prepare_u < prev_seg.release_u) {
            prev_seg.release_u = std::max(prev_seg.step_exit_u, cur.prepare_u);
        }

        if (cur.active_u < cur.prepare_u) cur.active_u = cur.prepare_u;
        if (prev_seg.active_u > prev_seg.commit_u) prev_seg.active_u = prev_seg.commit_u;

        if (cur.prepare_u < prev_seg.release_u) {
            RCLCPP_WARN(
                logger,
                "Step segment #%zu (step=[%.3f,%.3f)) and #%zu (step=[%.3f,%.3f)) "
                "overlap after arbitration — terrain labels may be inconsistent",
                i - 1, prev_seg.step_enter_u, prev_seg.step_exit_u,
                i, cur.step_enter_u, cur.step_exit_u
            );
        }
    }

    // ── 终值 clamp + MPC 约束软边界 ──
    // 不变式：prepare ≤ active ≤ commit ≤ step_enter ≤ step_exit ≤ release。
    // 约束窗软门控：在 commit 上游 gate_transition_distance 处 0→1，在 exit 下游同宽 1→0，
    // 使 MPC 首控点跨过 commit 那一帧的速度/航向惩罚连续变化。
    for (StepPlanSegment& segment : plan) {
        segment.commit_u = std::min(segment.commit_u, segment.step_enter_u);
        segment.prepare_u = std::min(segment.prepare_u, segment.commit_u);
        segment.active_u = std::clamp(segment.active_u, segment.prepare_u, segment.commit_u);
        segment.step_exit_u = std::max(segment.step_exit_u, segment.step_enter_u);
        segment.release_u = std::max(segment.release_u, segment.step_exit_u);

        const double gate_start_u = std::max(
            segment.prepare_u,
            retreat_path_u_by_distance(p, path, segment.commit_u, p.gate_transition_distance)
        );
        const double gate_end_u = std::min(
            segment.release_u,
            advance_path_u_by_distance(p, path, segment.step_exit_u, p.gate_transition_distance)
        );

        auto& constraint = segment.traversal_constraint;
        constraint.commit_u = segment.commit_u;
        constraint.step_enter_u = segment.step_enter_u;
        constraint.exit_u = segment.step_exit_u;
        constraint.gate_start_u = gate_start_u;
        constraint.gate_end_u = gate_end_u;
    }

    return plan;
}

} // namespace nav_executor::step_annotator
