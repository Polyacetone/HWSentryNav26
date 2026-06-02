#include <path_follower/step/step_controller.hpp>
#include <rclcpp/logging.hpp>

namespace path_follower {

namespace {

constexpr double ANGLE_EPSILON = 1e-6;
constexpr double U_EPSILON = 1e-6;

} // anonymous namespace

// ═══════════════════════ 构造函数 ════════════════════════════

StepController::StepController(
    const StepDetectionParams& step_detection,
    const StepBlockReplanParams& step_block_replan,
    const double step_dist_offset,
    const CapabilityProfile& normal_profile,
    const std::array<CapabilityProfile, 3>& capability_profiles,
    const ProfileBlendParams& blend_params,
    rclcpp::Logger logger
) : step_detection_(step_detection),
    step_block_replan_(step_block_replan),
    step_dist_offset_(step_dist_offset),
    blend_params_(blend_params),
    normal_profile_(normal_profile),
    capability_profiles_(capability_profiles),
    logger_(logger),
    current_profile_(normal_profile),
    target_profile_(normal_profile) {}

// ═══════════════════════ 路径工具 ═══════════════════════════

double StepController::advance_path_u_by_distance(const SplinePath& path, const double start_u, const double distance) const {
    const double resolution = std::max(1e-3, step_detection_.path_sample_resolution);
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

double StepController::retreat_path_u_by_distance(const SplinePath& path, const double start_u, const double distance) const {
    const double resolution = std::max(1e-3, step_detection_.path_sample_resolution);
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

// ═══════════════════════ 台阶命令构建 ═══════════════════════

std::optional<ActiveStepMode> StepController::build_step_command(
    const StepDirection direction,
    const Eigen::Vector2d& step_enter_pos_map,
    const double step_enter_u,
    const DirectionMap& direction_map
) const {
    const Eigen::Vector2d g = direction_map.map_coord_to_grid(step_enter_pos_map);
    const uint8_t label = direction_map.terrain_at(g);
    const auto& rule = direction_map.rule_for_label(label, direction == StepDirection::UP);

    if (rule.chassis_mode == 0) {
        return std::nullopt;
    }

    return ActiveStepMode {
        .mode = rule.chassis_mode,
        .capability = rule.capability,
        .speed_min = rule.speed.min,
        .speed_max = rule.speed.max,
        .step_entry_u = step_enter_u,
    };
}

// ═══════════════════════ 台阶规划构建 ═══════════════════════
//
// 设计选择：基于 terrain label 扫描（而非采样点间 gap 检测）。
// 遍历均匀 u 采样点，跟踪地形标签的变化：
//   - label >= SLOPE 且与上一个采样点不同 → 结束上一段（如有），
//     如果路径在此处的穿越方向有效则开启新段
//   - label < SLOPE → 结束上一段
// 段边界完全由 terrain label 在路径上的连续分布决定，
// 不依赖样条参数化质量或 arc_length 积分。
// 方向过滤（detect_dot_threshold）仅在段入口处做一次，
// 避免每个采样点逐个判断带来的偶发碎片化。

std::vector<StepController::StepPlanSegment> StepController::build_step_plan(
    const SplinePath& path,
    const DirectionMap& direction_map
) const {
    const double resolution = std::max(1e-3, step_detection_.path_sample_resolution);

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

    // 当前正在构建的段
    struct ActiveSegment {
        int start_index;
        uint8_t label;
        StepDirection direction;
        Eigen::Vector2d step_enter_pos_map;
        Eigen::Vector2d dir_map;
    };
    std::optional<ActiveSegment> active;
    std::vector<StepPlanSegment> plan;

    // 结束当前段、构造 StepPlanSegment 并添加到 plan
    auto finalize = [&](int end_index) {
        if (!active) return;

        const double step_exit_u = static_cast<double>(end_index - 1) * u_step;
        const Eigen::Vector2d step_exit_pos = path.position(step_exit_u);
        const double step_enter_u = static_cast<double>(active->start_index) * u_step;

        auto command = build_step_command(
            active->direction, active->step_enter_pos_map, step_enter_u, direction_map
        );
        if (command) {
            StepPlanSegment seg;
            seg.path_version = path_version_;
            seg.step_enter_u = step_enter_u;
            seg.step_exit_u = step_exit_u;
            seg.step_enter_pos_map = active->step_enter_pos_map;
            seg.step_exit_pos_map = step_exit_pos;
            seg.dir_map = active->dir_map;
            seg.direction = active->direction;
            seg.command = *command;
            seg.terrain_label = active->label;
            plan.push_back(std::move(seg));
        }
        active.reset();
    };

    for (int i = 0; i < sample_count; ++i) {
        const double u = static_cast<double>(i) * u_step;
        const Eigen::Vector2d pos = path.position(u);
        const Eigen::Vector2d g = direction_map.map_coord_to_grid(pos);

        // 网格边界处 → 终止段（如有）
        if (!direction_map.is_valid_coord(g)) {
            finalize(i);
            continue;
        }

        const uint8_t label = direction_map.terrain_at(g);

        // 非台阶地形 → 终止段（如有）
        if (label < static_cast<uint8_t>(TerrainType::SLOPE)) {
            finalize(i);
            continue;
        }

        // ── 在此采样点上是台阶地形 ──

        if (active && label == active->label) {
            // 与当前段 label 相同 → 继续扩展（不检查 gap 或 direction）
            continue;
        }

        // label 变更或无活跃段 → 结束上一段（如有）, 尝试开启新段
        finalize(i);

        // 在段入口确定穿越方向；若 dot 过小说明路径不顺台阶方向，放弃此段
        const Eigen::Vector2d tangent = path.tangent(u);
        const Eigen::Vector2d dir = direction_map.interpolate(g);
        if (dir.norm() < ANGLE_EPSILON) continue;

        const double dot = dir.normalized().dot(tangent.normalized());
        if (std::abs(dot) <= step_detection_.detect_dot_threshold) continue;

        const StepDirection direction = dot > 0.0 ? StepDirection::UP : StepDirection::DOWN;

        active = ActiveSegment{
            .start_index = i,
            .label = label,
            .direction = direction,
            .step_enter_pos_map = pos,
            .dir_map = dir,
        };
    }

    // 结束最后一个段
    finalize(sample_count);

    if (plan.empty()) return plan;

    // ── 计算 prepare / active / release u 边界 ──
    for (size_t i = 0; i < plan.size(); ++i) {
        StepPlanSegment& segment = plan[i];
        segment.prepare_u = retreat_path_u_by_distance(path, segment.step_enter_u, step_detection_.prepare_distance);
        segment.active_u = retreat_path_u_by_distance(path, segment.step_enter_u, step_detection_.active_distance);
        segment.release_u = advance_path_u_by_distance(path, segment.step_exit_u, step_detection_.release_distance);
    }

    // ── 段间重叠仲裁 ──
    for (size_t i = 1; i < plan.size(); ++i) {
        StepPlanSegment& prev = plan[i - 1];
        StepPlanSegment& cur = plan[i];

        if (cur.prepare_u >= prev.release_u) {
            if (cur.active_u < cur.prepare_u) cur.active_u = cur.prepare_u;
            if (prev.active_u > prev.step_enter_u) prev.active_u = prev.step_enter_u;
            continue;
        }

        cur.prepare_u = std::min(prev.release_u, cur.step_enter_u);
        if (cur.prepare_u < prev.release_u) {
            prev.release_u = std::max(prev.step_exit_u, cur.prepare_u);
        }

        if (cur.active_u < cur.prepare_u) cur.active_u = cur.prepare_u;
        if (prev.active_u > prev.step_enter_u) prev.active_u = prev.step_enter_u;

        if (cur.prepare_u < prev.release_u) {
            RCLCPP_WARN(
                logger_,
                "Step segment #%zu (step=[%.3f,%.3f)) and #%zu (step=[%.3f,%.3f)) "
                "overlap after arbitration — terrain labels may be inconsistent",
                i - 1, prev.step_enter_u, prev.step_exit_u,
                i, cur.step_enter_u, cur.step_exit_u
            );
        }
    }

    for (StepPlanSegment& segment : plan) {
        segment.prepare_u = std::min(segment.prepare_u, segment.step_enter_u);
        segment.active_u = std::clamp(segment.active_u, segment.prepare_u, segment.step_enter_u);
        segment.step_exit_u = std::max(segment.step_exit_u, segment.step_enter_u);
        segment.release_u = std::max(segment.release_u, segment.step_exit_u);
        segment.command.step_entry_u = segment.step_enter_u;
        segment.command.prepare_u = segment.prepare_u;
        segment.command.active_u = segment.active_u;
        segment.command.release_u = segment.release_u;
    }

    return plan;
}

void StepController::clear_runtime_state() {
    held_step_segment_index_ = std::nullopt;
    step_locked_path_ = std::nullopt;
    step_locked_fixed_goal_ = false;
    step_locked_fixed_goal_pos_ = Eigen::Vector2d::Zero();
    current_profile_ = normal_profile_;
    target_profile_ = normal_profile_;
}

void StepController::clear_plan() {
    clear_runtime_state();
    step_plan_.clear();
}

void StepController::update_plan_for_path_change(
    const bool has_new_path,
    const std::optional<SplinePath>& path,
    const DirectionMap* const direction_map
) {
    if (!has_new_path) return;

    path_version_++;
    clear_plan();

    if (!path || !direction_map) {
        return;
    }

    step_plan_ = build_step_plan(*path, *direction_map);
    if (!step_plan_.empty()) {
        RCLCPP_DEBUG(logger_, "Built step plan with %zu segments for path_version=%d:", step_plan_.size(), path_version_);
        for (size_t i = 0; i < step_plan_.size(); ++i) {
            const auto& seg = step_plan_[i];
            RCLCPP_DEBUG(
                logger_,
                "  #%zu: label=%hhu dir=%s "
                "prepare=%.3f active=%.3f step=[%.3f,%.3f) release=%.3f "
                "mode=%hhu",
                i, seg.terrain_label,
                seg.direction == StepDirection::UP ? "UP" : "DOWN",
                seg.prepare_u, seg.active_u,
                seg.step_enter_u, seg.step_exit_u, seg.release_u,
                seg.command.mode
            );
        }
    }
}

// ═══════════════════════ 台阶段查询 ═══════════════════════

std::optional<size_t> StepController::find_active_segment_index(const double current_u) const {
    for (size_t i = 0; i < step_plan_.size(); ++i) {
        const StepPlanSegment& segment = step_plan_[i];
        if (current_u + U_EPSILON < segment.prepare_u) {
            break;
        }
        if (current_u < segment.release_u) {
            return i;
        }
    }
    return std::nullopt;
}

const StepController::StepPlanSegment* StepController::active_segment(const double current_u) const {
    if (held_step_segment_index_.has_value()) {
        const auto& segment = step_plan_[*held_step_segment_index_];
        if (current_u < segment.release_u) {
            return &segment;
        }
    }
    const auto index = find_active_segment_index(current_u);
    if (!index) return nullptr;
    return &step_plan_[*index];
}

const StepController::StepPlanSegment* StepController::current_command_segment(const double current_u) const {
    const StepPlanSegment* const segment = active_segment(current_u);
    if (!segment) return nullptr;
    if (!is_step_mode(segment->command.mode)) return nullptr;
    return segment;
}

// ═══════════════════════ 台阶激活/锁存 ═══════════════════════

void StepController::update_active_segment(
    const double current_u,
    const std::optional<SplinePath>& global_path,
    const bool fixed_goal,
    const Eigen::Vector2d& fixed_goal_pos
) {
    // ── 持有段已过期 → 释放，fall through 尝试获取下一段 ──
    if (held_step_segment_index_.has_value()) {
        const auto& segment = step_plan_[*held_step_segment_index_];
        if (current_u >= segment.release_u) {
            RCLCPP_DEBUG(
                logger_,
                "Step segment #%zu released (current_u=%.3f >= release_u=%.3f)",
                *held_step_segment_index_, current_u, segment.release_u
            );
            held_step_segment_index_ = std::nullopt;
            // 暂不清除锁存/profile — fall through 到下面尝试获取下一段
        } else {
            return; // 仍在有效范围内，保持
        }
    }

    // ── 尝试获取下一个有效段 ──
    const auto next_index = find_active_segment_index(current_u);
    if (!next_index) {
        step_locked_path_ = std::nullopt;
        step_locked_fixed_goal_ = false;
        step_locked_fixed_goal_pos_ = Eigen::Vector2d::Zero();
        target_profile_ = normal_profile_;
        return;
    }

    held_step_segment_index_ = next_index;

    {
        const auto& segment = step_plan_[*held_step_segment_index_];
        target_profile_ = capability_profiles_[static_cast<size_t>(segment.command.capability)];

        if (global_path) {
            step_locked_path_ = global_path;
            step_locked_fixed_goal_ = fixed_goal;
            step_locked_fixed_goal_pos_ = fixed_goal_pos;
        }

        RCLCPP_DEBUG(
            logger_,
            "Step segment #%zu acquired: "
            "label=%hhu dir=%s "
            "prepare=%.3f active=%.3f step=[%.3f,%.3f) release=%.3f "
            "mode=%hhu",
            *held_step_segment_index_,
            segment.terrain_label,
            segment.direction == StepDirection::UP ? "UP" : "DOWN",
            segment.prepare_u, segment.active_u,
            segment.step_enter_u, segment.step_exit_u, segment.release_u,
            segment.command.mode
        );
    }
}

// ═══════════════════════ 时间域 profile 融合 ═══════════════

void StepController::tick_profile_blend() {
    auto& cur = current_profile_;
    const auto& tgt = target_profile_;

    // ── command_bounds ──
    {
        auto& c = cur.command_bounds;
        const auto& t = tgt.command_bounds;
        const double v_step = blend_params_.v_step;
        const double w_step = blend_params_.w_step;

        if (t.vel_max >= c.vel_max) {
            c.vel_max = t.vel_max;
        } else {
            c.vel_max = std::max(t.vel_max, c.vel_max - v_step);
        }
        if (t.vel_min >= c.vel_min) {
            c.vel_min = t.vel_min;
        } else {
            c.vel_min = std::max(t.vel_min, c.vel_min - v_step);
        }
        if (t.omega_max >= c.omega_max) {
            c.omega_max = t.omega_max;
        } else {
            c.omega_max = std::max(t.omega_max, c.omega_max - w_step);
        }
        if (t.omega_min >= c.omega_min) {
            c.omega_min = t.omega_min;
        } else {
            c.omega_min = std::max(t.omega_min, c.omega_min - w_step);
        }
    }

    // ── motion_constraints ──
    {
        auto& c = cur.motion_constraints;
        const auto& t = tgt.motion_constraints;

        if (t.acc_max >= c.acc_max) {
            c.acc_max = t.acc_max;
        } else {
            c.acc_max = std::max(t.acc_max, c.acc_max - blend_params_.acc_step);
        }
        if (t.alpha_max >= c.alpha_max) {
            c.alpha_max = t.alpha_max;
        } else {
            c.alpha_max = std::max(t.alpha_max, c.alpha_max - blend_params_.alpha_step);
        }
        if (t.a_lat_max >= c.a_lat_max) {
            c.a_lat_max = t.a_lat_max;
        } else {
            c.a_lat_max = std::max(t.a_lat_max, c.a_lat_max - blend_params_.a_lat_step);
        }
    }
}

// ═══════════════════════ 台阶模式查询 ═══════════════════════

std::optional<ActiveStepMode> StepController::current_active_step_mode(const double current_u) const {
    const StepPlanSegment* const segment = current_command_segment(current_u);
    if (!segment) return std::nullopt;
    return segment->command;
}

bool StepController::is_step_active(const double current_u) const {
    return active_segment(current_u) != nullptr;
}

bool StepController::should_activate_step_mode(const double current_u) const {
    const StepPlanSegment* const segment = current_command_segment(current_u);
    if (!segment) return false;
    return current_u + U_EPSILON >= segment->active_u;
}

uint8_t StepController::compute_step_distance_cm(const SplinePath& path, const double current_u) const {
    const StepPlanSegment* const segment = current_command_segment(current_u);
    if (!segment) {
        return 0;
    }
    if (current_u >= segment->step_enter_u) {
        return 0;
    }

    const double distance = path.arc_length(current_u, segment->step_enter_u);
    const double adjusted_distance = distance + step_dist_offset_;
    const int64_t rounded_cm = std::lround(adjusted_distance * 100.0);
    return static_cast<uint8_t>(std::clamp<int64_t>(rounded_cm, 0, 255));
}

// ═══════════════════════ 台阶阻塞重规划检测 ═══════════════════

/*static*/ std::optional<StepController::FollowStepBlockSampleStats> StepController::sample_block_replan_stats(
    const StepBlockReplanParams& p,
    const SplinePath& path,
    const double start_u,
    const CostMap* const dynamic_cost_map,
    const std::vector<const CostMap*>& dynamic_prediction_maps,
    const DirectionMap& direction_map
) {
    const double lookahead_distance = std::max(0.0, p.lookahead_distance);
    const double resolution = std::max(1e-3, p.sample_resolution);
    const bool using_predicted = !dynamic_prediction_maps.empty();

    FollowStepBlockSampleStats stats;
    const int samples = using_predicted
        ? static_cast<int>(dynamic_prediction_maps.size())
        : std::max(1, static_cast<int>(std::ceil(lookahead_distance / resolution)) + 1);
    if (samples <= 0) return stats;

    auto advance_path_u = [&](const double distance) {
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
    };

    for (int i = 0; i < samples; ++i) {
        const double t = samples == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(samples - 1);
        const double u = advance_path_u(lookahead_distance * t);

        const Eigen::Vector2d pos = path.position(u);
        const Eigen::Vector2d dir_grid = direction_map.map_coord_to_grid(pos);
        if (!direction_map.is_valid_coord(dir_grid)) return std::nullopt;

        const double step_norm = direction_map.interpolate(dir_grid).norm();
        const bool on_step = step_norm >= p.step_norm_threshold;

        bool blocked_dynamic = false;
        if (using_predicted) {
            const CostMap* const cost_map = dynamic_prediction_maps[static_cast<size_t>(i)];
            if (!cost_map) return std::nullopt;
            const Eigen::Vector2d cost_grid = cost_map->map_coord_to_grid(pos);
            if (!cost_map->is_valid_coord(cost_grid)) return std::nullopt;
            blocked_dynamic = cost_map->interpolate(cost_grid) >= p.obstacle_cost_threshold;
        } else if (dynamic_cost_map) {
            const Eigen::Vector2d cost_grid = dynamic_cost_map->map_coord_to_grid(pos);
            if (!dynamic_cost_map->is_valid_coord(cost_grid)) return std::nullopt;
            blocked_dynamic = dynamic_cost_map->interpolate(cost_grid) >= p.obstacle_cost_threshold;
        }

        stats.sample_count++;
        if (!on_step) continue;

        stats.step_sample_count++;
        if (blocked_dynamic) {
            stats.blocked_step_sample_count++;
        }
    }

    return stats;
}

bool StepController::check_block_replan(
    const SplinePath& path,
    const double current_u,
    const DirectionMap* const masked_direction_map,
    const CostMap* const current_dynamic_cost_map,
    const std::vector<const CostMap*>& per_step_dynamic_cost_maps
) const {
    const auto& p = step_block_replan_;
    if (!p.enable || !masked_direction_map) return false;

    const auto stats = sample_block_replan_stats(
        p,
        path,
        current_u,
        current_dynamic_cost_map,
        per_step_dynamic_cost_maps,
        *masked_direction_map
    );
    if (!stats || stats->step_sample_count == 0) {
        return false;
    }

    if (per_step_dynamic_cost_maps.empty()) {
        if (stats->blocked_step_sample_count > 0) {
            RCLCPP_WARN(
                logger_,
                "Follow replan: detected blocked step ahead within %.2f m (blocked_step_samples=%d/%d)",
                p.lookahead_distance,
                stats->blocked_step_sample_count,
                stats->step_sample_count
            );
            return true;
        }
        return false;
    }

    const double blocked_ratio = static_cast<double>(stats->blocked_step_sample_count) / static_cast<double>(stats->step_sample_count);
    if (blocked_ratio >= p.predicted_obstacle_ratio_threshold) {
        RCLCPP_WARN(
            logger_,
            "Follow replan: predicted blocked step ahead within %.2f m (ratio=%.2f >= %.2f)",
            p.lookahead_distance,
            blocked_ratio,
            p.predicted_obstacle_ratio_threshold
        );
        return true;
    }

    return false;
}

} // namespace path_follower
