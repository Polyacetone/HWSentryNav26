#include <nav_executor/path_planner/path_planner.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <queue>

#include <rclcpp/logging.hpp>

namespace nav_executor {

namespace {

double wrap_angle(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

const char* speed_profile_selection_string(
    const SpeedProfileOptimizer::Diagnostics::Selection selection
) {
    using Selection = SpeedProfileOptimizer::Diagnostics::Selection;
    switch (selection) {
        case Selection::OPTIMAL: return "OPTIMAL";
        case Selection::SEED_OPTIMAL_NO_SOFT_CONSTRAINT:
            return "SEED_OPTIMAL_NO_SOFT_CONSTRAINT";
        case Selection::FALLBACK: return "FALLBACK";
    }
    return "UNKNOWN";
}

// 全局动力学搜索见证 → MINCO 平坦边界状态 + 物理见证时长 + 有向切向。
struct MincoSeed {
    std::vector<MincoMinJerk::BoundaryPVA> states; // 2D pos/vel/acc
    std::vector<double> durations;
    std::vector<Eigen::Vector2d> tangents;         // 边界处的 A* 有向单位切向
};

struct EnvironmentValidationReport {
    std::optional<std::string> rejection;
    std::vector<std::string> warnings;
};

// 距离和显著转向点会保留；段时长严格累加 A* 返回的真实逐边时长。
// tangents 记录 A* 的有向性，供 MINCO 有向正则性软罚使用。
MincoSeed build_minco_seed(
    const SpeedWitness& witness,
    const double resample_distance,
    const double directed_speed_min
) {
    MincoSeed seed;
    if (witness.positions.size() < 2
        || witness.tangents.size() != witness.positions.size()
        || witness.velocities.size() != witness.positions.size()
        || witness.durations.size() + 1 != witness.positions.size()
        || !std::isfinite(directed_speed_min)
        || directed_speed_min <= 0.0) return seed;

    std::vector<size_t> selected {0};
    const double distance_threshold = std::max(resample_distance, 0.05);
    constexpr double HEADING_THRESHOLD = 0.5;
    for (size_t i = 1; i + 1 < witness.positions.size(); ++i) {
        const size_t last = selected.back();
        const double distance = (
            witness.positions[i] - witness.positions[last]
        ).norm();
        const double heading = std::atan2(
            witness.tangents[i].y(), witness.tangents[i].x()
        );
        const double last_heading = std::atan2(
            witness.tangents[last].y(), witness.tangents[last].x()
        );
        const double heading_change = std::abs(wrap_angle(heading - last_heading));
        if (distance >= distance_threshold || heading_change >= HEADING_THRESHOLD) {
            selected.push_back(i);
        }
    }
    selected.push_back(witness.positions.size() - 1);

    const size_t n = selected.size();
    seed.states.resize(n);
    seed.tangents.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t selected_index = selected[i];
        seed.states[i].pos = witness.positions[selected_index];
        seed.states[i].vel = witness.velocities[selected_index];
        seed.states[i].acc.setZero();
        const Eigen::Vector2d& tangent = witness.tangents[selected_index];
        if (!tangent.allFinite() || tangent.norm() <= 1e-9) return {};
        seed.tangents[i] = tangent.normalized();
    }
    seed.durations.resize(n - 1);
    for (size_t i = 0; i + 1 < n; ++i) {
        double duration = 0.0;
        for (size_t edge = selected[i]; edge < selected[i + 1]; ++edge) {
            duration += witness.durations[edge];
        }
        seed.durations[i] = std::max(duration, 0.1);
    }

    // MINCO 时标只塑形几何，首尾导数必须保持有向非零；真实起步和停车速度
    // 由后续 PathSpeedProfile 独立确定。
    seed.states.front().vel = seed.tangents.front()
        * std::max(seed.states.front().vel.norm(), directed_speed_min);
    seed.states.back().vel = seed.tangents.back()
        * std::max(seed.states.back().vel.norm(), directed_speed_min);
    seed.states.front().acc.setZero();
    seed.states.back().acc.setZero();

    return seed;
}

// 环境验收：路径与代价地图、方向地形的关系。
EnvironmentValidationReport validate_path_environment(
    const MincoTrajectory& trajectory,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const int occupied_threshold,
    const double step_alignment_threshold,
    const PlannerConfig::EnvironmentValidationParams& validation
) {
    EnvironmentValidationReport report;
    const double total_arc_length = trajectory.total_arc_length();
    double worst_traversal_angle = validation.traversal_angle_tolerance;
    double worst_traversal_angle_arc_length = 0.0;

    // 弧长是跟随层唯一的进度坐标，环境验收使用同一坐标。采样间距同时覆盖
    // 地图分辨率与每段的最小采样密度，避免物理本体缩窄为单格后被跨过。
    const int samples_per_segment = std::max(validation.samples_per_segment, 1);
    const double segment_spacing = total_arc_length
        / static_cast<double>(std::max(trajectory.segment_count(), 1) * samples_per_segment);
    const double spacing = std::min(
        direction_map.geometry.resolution() * 0.5, segment_spacing
    );
    const int intervals = std::max(
        1, static_cast<int>(std::ceil(total_arc_length / spacing))
    );

    for (int i = 0; i <= intervals; ++i) {
        const double arc_length = total_arc_length * static_cast<double>(i)
            / static_cast<double>(intervals);
        const TrajSample sample = trajectory.eval_arc_length(arc_length);
        const std::string at = " at s=" + std::to_string(arc_length);

        const auto cost_sample = cost_map.sample_map(sample.p);
        if (!cost_sample) {
            report.rejection = "trajectory leaves the planning map" + at;
            return report;
        }
        const double cost = cost_sample->value;
        if (cost >= static_cast<double>(occupied_threshold)) {
            report.rejection = "trajectory intersects occupied cost" + at
                + ": cost=" + std::to_string(cost)
                + " >= threshold=" + std::to_string(occupied_threshold);
            return report;
        }

        const auto direction_sample = direction_map.sample_map(sample.p);
        const auto terrain_cell = direction_map.geometry.containing_cell(sample.p);
        if (!direction_sample || !terrain_cell) {
            report.rejection = "trajectory leaves the direction map" + at;
            return report;
        }
        if (!direction_map.is_terrain_body_cell(*terrain_cell)) continue;

        const uint8_t label = direction_map.terrain_label_at_cell(*terrain_cell);
        const Eigen::Vector2d raw_direction = direction_map.raw_direction_at_cell(*terrain_cell);
        if (label < static_cast<uint8_t>(TerrainType::SLOPE)
            || raw_direction.squaredNorm() <= 1e-12) {
            report.rejection = "trajectory entered an invalid directional terrain body" + at;
            return report;
        }
        const Eigen::Vector2d direction = raw_direction.normalized();
        const Eigen::Vector2d heading(std::cos(sample.theta), std::sin(sample.theta));
        const double alignment = heading.dot(direction);
        if (std::abs(alignment) <= step_alignment_threshold) {
            report.rejection = "trajectory is not aligned with directional terrain" + at
                + ": |heading.dot(dir)|=" + std::to_string(std::abs(alignment))
                + " <= threshold=" + std::to_string(step_alignment_threshold);
            return report;
        }
        const bool going_up = alignment >= 0.0;
        if (!terrain_constraints.selected_mode(label, going_up)) {
            report.rejection = "trajectory uses prohibited directional terrain" + at
                + ": label=" + std::to_string(static_cast<int>(label))
                + ", direction=" + (going_up ? "up" : "down");
            return report;
        }
        const double angle = std::acos(std::clamp(std::abs(alignment), 0.0, 1.0));
        if (angle > worst_traversal_angle) {
            worst_traversal_angle = angle;
            worst_traversal_angle_arc_length = arc_length;
        }
    }

    if (worst_traversal_angle > validation.traversal_angle_tolerance) {
        report.warnings.push_back(
            "stair-direction deviation at s="
            + std::to_string(worst_traversal_angle_arc_length)
            + ": angle=" + std::to_string(worst_traversal_angle)
            + " rad, tolerance=" + std::to_string(validation.traversal_angle_tolerance)
            + " rad"
        );
    }
    return report;
}

} // anonymous namespace

PathPlanner::PathPlanner(
    const PlannerConfig& config,
    std::shared_ptr<StepRoutingMask> step_routing_mask,
    rclcpp::Logger logger
) : config_(config),
    primitive_library_(config.motion_primitives),
    step_routing_mask_(std::move(step_routing_mask)),
    logger_(logger) {
    if (!primitive_library_.valid()) {
        RCLCPP_ERROR(
            logger_, "Motion primitive generation failed: %s",
            primitive_library_.error().c_str()
        );
    } else {
        RCLCPP_INFO(
            logger_,
            "Generated %zu quantized motion primitives (max quantization: position=%.3g m, heading=%.3g rad)",
            primitive_library_.primitive_count(),
            primitive_library_.max_position_residual(),
            primitive_library_.max_heading_residual()
        );
    }
}

PathPlanner::~PathPlanner() {
    stop();
}

void PathPlanner::start() {
    std::lock_guard lock(mutex_);
    if (running_) return;
    running_ = true;
    worker_ = std::thread([this] { worker_loop(); });
}

void PathPlanner::stop() {
    {
        std::lock_guard lock(mutex_);
        if (!running_) return;
        running_ = false;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void PathPlanner::submit(const PlanRequest& request) {
    {
        std::lock_guard lock(mutex_);
        pending_request_ = request; // latest-wins
    }
    cv_.notify_one();
}

std::optional<PlanResult> PathPlanner::try_take_result() {
    std::lock_guard lock(mutex_);
    if (!result_) return std::nullopt;
    PlanResult r = std::move(*result_);
    result_.reset();
    return r;
}

bool PathPlanner::busy() const {
    std::lock_guard lock(mutex_);
    return busy_ || pending_request_.has_value();
}

void PathPlanner::worker_loop() {
    while (true) {
        PlanRequest request;
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this] { return !running_ || pending_request_.has_value(); });
            if (!running_) return;
            request = std::move(*pending_request_);
            pending_request_.reset();
            busy_ = true;
        }

        PlanResult result = plan(request);

        {
            std::lock_guard lock(mutex_);
            result_ = std::move(result);
            busy_ = false;
        }
    }
}

// ═══════════════════════ 规划期几何工具 ═══════════════════════

bool PathPlanner::is_map_point_feasible(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const Eigen::Vector2d& map_pt
) const {
    const auto cost = cost_map.sample_map(map_pt);
    const auto direction = direction_map.sample_map(map_pt);
    if (!cost || !direction) return false;
    return cost->value < config_.occupied_threshold
        && direction->value.norm() < config_.on_step_threshold;
}

std::optional<Eigen::Vector2d> PathPlanner::nudge_point_to_free(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const Eigen::Vector2d& map_pt,
    const double max_nudge_distance
) const {
    const GridGeometry& geometry = cost_map.geometry;
    const int width = geometry.width();
    const int height = geometry.height();

    const auto key = [width](const int x, const int y) {
        return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
    };

    if (geometry.contains_map_point(map_pt)
        && is_map_point_feasible(cost_map, direction_map, map_pt)) {
        return map_pt;
    }

    const auto start_cell = geometry.containing_cell(map_pt);
    if (!start_cell) return std::nullopt;
    const int sx = start_cell->x();
    const int sy = start_cell->y();

    std::vector<uint8_t> visited(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
    std::queue<Eigen::Vector2i> q;
    q.push(Eigen::Vector2i(sx, sy));
    visited[key(sx, sy)] = 1;

    static constexpr int dx[] = {0, 1, 0, -1, 1, 1, -1, -1};
    static constexpr int dy[] = {1, 0, -1, 0, 1, -1, 1, -1};

    while (!q.empty()) {
        const auto current = q.front();
        q.pop();

        const Eigen::Vector2d candidate_map = geometry.cell_center(current);
        const double dist = (candidate_map - map_pt).norm();
        if (dist > max_nudge_distance) continue;

        if (is_map_point_feasible(cost_map, direction_map, candidate_map)) {
            return candidate_map;
        }

        for (int i = 0; i < 8; i++) {
            const int nx = current.x() + dx[i];
            const int ny = current.y() + dy[i];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            const double ndist = (geometry.cell_center({nx, ny}) - map_pt).norm();
            if (ndist > max_nudge_distance) continue;
            const size_t nk = key(nx, ny);
            if (visited[nk]) continue;
            visited[nk] = 1;
            q.push(Eigen::Vector2i(nx, ny));
        }
    }

    return std::nullopt;
}

// ═══════════════════════ 规划主流程 ═══════════════════════

PlanResult PathPlanner::plan(const PlanRequest& req) const {
    PlanResult result;
    result.goal_id = req.goal.id;
    result.plan_generation = req.plan_generation;
    result.goal_pos = req.goal.position_map;

    if (!req.global_cost_map || !req.merged_cost_map || !req.direction_map) {
        result.failure_reason = "map snapshot incomplete";
        result.kind = PlanResult::Kind::FAILED;
        return result;
    }

    const Eigen::Vector2d goal_map = req.goal.position_map;
    const bool fixed = req.goal.fixed;

    const auto fail = [&](const std::string& msg) {
        result.failure_reason = msg;
        result.kind = PlanResult::Kind::FAILED;
        return result;
    };

    // 地形合法性只由严格本体上的局部边约束决定。blocked_cost_layer 包含
    // 膨胀 halo，融合后会把本应仅用于软塑形的方向场变成硬障碍。
    const CostMap& global_feasibility_cost = *req.global_cost_map;
    const CostMap& planning_cost_map = *req.merged_cost_map;

    Eigen::Vector2d start_map = req.current_pos_map;
    Eigen::Vector2d goal_plan = goal_map;

    // ── 起点 global 严格检查 ──
    {
        if (!global_feasibility_cost.geometry.contains_map_point(start_map)) {
            return fail("Start is out of bound");
        }
        if (!is_map_point_feasible(global_feasibility_cost, *req.direction_map, start_map)) return fail("Start is not feasible on global map");
    }

    // ── 起点 nudge（merged 上最近 free）──
    {
        const auto nudged = nudge_point_to_free(
            planning_cost_map, *req.direction_map, start_map, config_.nudge_max_distance
        );
        if (!nudged) return fail("Cannot nudge start to a free cell");
        start_map = *nudged;
    }

    // ── 终点 global 严格检查 ──
    {
        if (!global_feasibility_cost.geometry.contains_map_point(goal_map)) {
            RCLCPP_ERROR(
                logger_, "Goal (%.2f, %.2f) is outside the global map",
                goal_map.x(), goal_map.y()
            );
            return fail("Goal is out of bound");
        }
        if (!is_map_point_feasible(
                global_feasibility_cost, *req.direction_map, goal_map
            )) {
            RCLCPP_ERROR(
                logger_, "Goal (%.2f, %.2f) is not feasible on the global map",
                goal_map.x(), goal_map.y()
            );
            return fail("Goal is not feasible on global map");
        }
    }

    // ── 终点 merged 检查 / nudge（取决于 fixed）──
    if (fixed) {
        if (!is_map_point_feasible(planning_cost_map, *req.direction_map, goal_map)) return fail("Fixed goal is occupied by a dynamic obstacle");
    } else {
        const auto nudged = nudge_point_to_free(
            planning_cost_map, *req.direction_map, goal_map, config_.nudge_max_distance
        );
        if (!nudged) return fail("Cannot nudge goal to a free cell");
        goal_plan = *nudged;
    }

    // ── 近距离短路判断：仅在起终点通过完整安全检查后，按最终可执行目标判定 ──
    if ((req.current_pos_map - goal_plan).norm() < config_.goal_reached_distance) {
        result.goal_pos = goal_plan;
        result.kind = fixed
            ? PlanResult::Kind::USE_AS_FIXED_GOAL
            : PlanResult::Kind::COMPLETE_NO_PLAN_NEEDED;
        RCLCPP_DEBUG(
            logger_, "Feasible goal within reached distance (%.2f m): %s",
            config_.goal_reached_distance,
            fixed ? "USE_AS_FIXED_GOAL" : "COMPLETE_NO_PLAN_NEEDED"
        );
        return result;
    }

    const auto plan_start = std::chrono::steady_clock::now();

    const auto start_cell = planning_cost_map.geometry.containing_cell(start_map);
    const auto goal_cell = planning_cost_map.geometry.containing_cell(goal_plan);
    if (!start_cell || !goal_cell) return fail("planning endpoints are outside the map");

    // ── [1] 二维空间 A*：只选择空间拓扑 ──
    const SpatialGridAstar spatial_astar(config_.spatial_astar);
    const auto spatial_result = spatial_astar.search(
        planning_cost_map,
        *req.direction_map,
        req.terrain_constraints,
        *start_cell,
        *goal_cell,
        config_.occupied_threshold,
        config_.step_detection.detect_dot_threshold
    );
    if (!spatial_result.success) {
        RCLCPP_WARN(
            logger_, "Spatial planning failed: error=%s expansions=%d open_peak=%zu",
            spatial_result.error.c_str(), spatial_result.route.expansions,
            spatial_result.route.open_peak
        );
        return fail("spatial grid planning failed: " + spatial_result.error);
    }
    const SpatialRoute& spatial_route = spatial_result.route;
    if (config_.enable_diagnostics) {
        result.debug_spatial_path.reserve(spatial_route.raw_path.size());
        for (const Eigen::Vector2i& cell : spatial_route.raw_path) {
            result.debug_spatial_path.push_back(
                planning_cost_map.geometry.cell_center(cell)
            );
        }
        result.debug_spatial_path.front() = start_map;
        result.debug_spatial_path.back() = goal_plan;
    }
    const auto spatial_done = std::chrono::steady_clock::now();

    // ── [2] passage-preserving LOS 平滑、参考状态与指导时标 ──
    const ReferencePathBuilder reference_builder(config_.reference_path);
    const auto reference_result = reference_builder.build(
        spatial_route,
        start_map,
        goal_plan,
        std::clamp(std::max(req.current_velocity, 0.0),
            0.0, config_.state_lattice.dynamics.velocity_max),
        planning_cost_map,
        *req.direction_map,
        req.terrain_constraints,
        config_.state_lattice.dynamics,
        config_.occupied_threshold
    );
    if (!reference_result.success) {
        RCLCPP_WARN(
            logger_, "Reference path construction failed: error=%s spatial_expansions=%d raw_cells=%zu",
            reference_result.error.c_str(), spatial_route.expansions,
            spatial_route.raw_path.size()
        );
        return fail("reference path construction failed: " + reference_result.error);
    }
    const ReferencePath& reference_path = reference_result.path;
    if (config_.enable_diagnostics) {
        result.debug_smoothed_spatial_path.reserve(reference_path.points.size());
        for (const ReferencePoint& point : reference_path.points) {
            result.debug_smoothed_spatial_path.push_back(point.position);
        }
    }
    const auto reference_done = std::chrono::steady_clock::now();

    // ── [3] 自由空间测地距离场与固定宽度走廊 ──
    const GuideFieldBuilder guide_builder(config_.guide_field);
    const auto guide_result = guide_builder.build(
        reference_path, planning_cost_map, config_.occupied_threshold
    );
    if (!guide_result.success) {
        RCLCPP_WARN(
            logger_, "Guide field construction failed: error=%s spatial_expansions=%d raw_length=%.2f smoothed_length=%.2f",
            guide_result.error.c_str(), spatial_route.expansions,
            reference_path.raw_length, reference_path.smoothed_length
        );
        return fail("guide field construction failed: " + guide_result.error);
    }
    const GuideField& guide_field = guide_result.field;
    const auto guide_done = std::chrono::steady_clock::now();

    // ── [4] 走廊内 canonical state lattice：修正局部几何并搜索完整动力学状态 ──
    const LatticeFrame frame {
        .origin_map = start_map,
        .base_heading = req.current_yaw,
        .xy_resolution = config_.motion_primitives.xy_resolution,
        .heading_bins = config_.motion_primitives.heading_bins,
    };
    const double clamped_start_speed = std::clamp(
        std::max(req.current_velocity, 0.0),
        0.0,
        config_.state_lattice.dynamics.velocity_max
    );
    const double speed_bin_width = config_.state_lattice.dynamics.velocity_max
        / static_cast<double>(config_.state_lattice.speed_bin_count - 1);
    const int start_speed_bin = std::clamp(
        static_cast<int>(std::floor(clamped_start_speed / speed_bin_width + 1e-12)),
        0,
        config_.state_lattice.speed_bin_count - 1
    );
    std::vector<StateLatticeAstar::SearchRoot> roots {{
        .key = {0, 0, 0, start_speed_bin},
        .relaxation_bias = 0.0,
        .relaxed = false,
    }};
    // 仅静止离散根允许改变初始航向；非零速度的航向必须保持实测方向。
    if (start_speed_bin == 0) {
        for (int heading = 1; heading < frame.heading_bins; ++heading) {
            const double yaw_change = std::abs(wrap_angle(
                2.0 * std::numbers::pi * static_cast<double>(heading)
                    / static_cast<double>(frame.heading_bins)
            ));
            roots.push_back({
                .key = {0, 0, heading, 0},
                .relaxation_bias = config_.start_yaw_relaxation.root_bias_seconds
                    + config_.start_yaw_relaxation.yaw_bias_seconds_per_rad
                        * yaw_change,
                .relaxed = true,
            });
        }
    }
    const StateLatticeAstar lattice(config_.state_lattice, primitive_library_);
    const auto lattice_result = lattice.search(
        frame,
        roots,
        goal_plan,
        planning_cost_map,
        *req.direction_map,
        req.terrain_constraints,
        guide_field,
        reference_path,
        config_.occupied_threshold,
        config_.step_detection.detect_dot_threshold
    );
    if (!lattice_result.success) {
        const auto& diagnostics = lattice_result.diagnostics;
        RCLCPP_WARN(
            logger_, "Corridor lattice planning failed: error=%s spatial_expansions=%d "
            "raw_length=%.2f smoothed_length=%.2f corridor_cells=%zu "
            "lattice_expansions=%d generated=%d open_peak=%zu corridor_rejections=%zu",
            lattice_result.error.c_str(), spatial_route.expansions,
            reference_path.raw_length, reference_path.smoothed_length,
            guide_field.corridor_cell_count(), diagnostics.expansions,
            diagnostics.generated_states, diagnostics.open_peak,
            diagnostics.corridor_rejections
        );
        return fail("corridor state-lattice planning failed: " + lattice_result.error);
    }
    const auto lattice_done = std::chrono::steady_clock::now();
    const SpeedWitness& speed_witness = lattice_result.witness;
    if (config_.enable_diagnostics) {
        result.debug_kino_path = speed_witness.positions;
    }

    // ── 种子重采样为 MINCO 边界全状态 + 段时长 ──
    const auto minco_seed = build_minco_seed(
        speed_witness,
        config_.seed_resample_distance,
        config_.minco.directed_speed_min
    );
    if (minco_seed.durations.empty()) return fail("MINCO seed construction produced no segments");
    // ── [5] MINCO 物理时标见证 + 连续几何联合塑形 ──
    MincoOptimizer optimizer(config_.minco);
    const auto minco_start = std::chrono::steady_clock::now();
    const auto opt = optimizer.optimize(
        minco_seed.states,
        minco_seed.durations,
        minco_seed.tangents,
        planning_cost_map,
        *req.direction_map,
        req.terrain_constraints
    );
    if (!opt.success) return fail("MINCO optimization failed: " + opt.error);
    const auto minco_done = std::chrono::steady_clock::now();

    if (config_.enable_diagnostics && opt.diagnostics_valid) {
        const auto& s = opt.seed_costs;
        const auto& f = opt.final_costs;
        const std::string_view status = opt.optimizer_status_string();
        RCLCPP_DEBUG(
            logger_,
            "MINCO optimizer: status=%.*s accepted=%d evals=%d trials=%d rejected=%d nonfinite=%d | "
            "raw |grad|_inf %.3g -> %.3g (pos=%.3g, virtual_time=%.3g), "
            "scaled_max_block=%.3g, first_order=%.3g | "
            "radius initial=%.3g final=%.3g range=[%.3g,%.3g] shrink=%d expand=%d boundary=%d | "
            "history update=%d skip=%d reset=%d | last actual=%.3g predicted=%.3g rho=%.3g | "
            "cost_tail relative=%.3g count=%d/%d",
            static_cast<int>(status.size()), status.data(),
            opt.accepted_iterations, opt.function_evaluations, opt.trial_evaluations,
            opt.rejected_trials, opt.nonfinite_trials,
            opt.initial_grad_inf_norm, opt.final_grad_inf_norm,
            opt.final_grad_pos_inf_norm, opt.final_grad_time_inf_norm,
            opt.final_scaled_grad_max_block_norm,
            opt.final_first_order_optimality,
            opt.initial_radius, opt.final_radius, opt.min_radius, opt.max_radius,
            opt.radius_shrinks, opt.radius_expansions, opt.boundary_steps,
            opt.history_updates, opt.history_skips, opt.history_resets,
            opt.last_actual_reduction, opt.last_predicted_reduction, opt.last_reduction_ratio,
            opt.last_relative_cost_reduction,
            opt.consecutive_small_cost_reductions,
            config_.minco.optimizer.cost_convergence_window
        );
        RCLCPP_DEBUG(
            logger_,
            "MINCO objective: cost %.3g -> %.3g witness_time=%.2f s | "
            "waypoints free=%d disp(sum=%.3f m, max=%.3f m) | "
            "seed[energy=%.3g time=%.3g obstacle=%.3g v=%.3g at=%.3g omega=%.3g "
            "alpha=%.3g alat=%.3g directed=%.3g align=%.3g prohibited=%.3g runup=%.3g] "
            "final[energy=%.3g time=%.3g obstacle=%.3g v=%.3g at=%.3g omega=%.3g "
            "alpha=%.3g alat=%.3g directed=%.3g align=%.3g prohibited=%.3g runup=%.3g]",
            s.total(), f.total(),
            opt.trajectory.total_time(),
            opt.free_waypoint_count, opt.waypoint_total_displacement, opt.waypoint_max_displacement,
            s.energy, s.time, s.obstacle, s.velocity, s.tangential_acceleration,
            s.angular_velocity, s.angular_acceleration, s.lateral_acceleration,
            s.directed_regularity, s.traversal_alignment, s.prohibited_traversal,
            s.runup_curvature,
            f.energy, f.time, f.obstacle, f.velocity, f.tangential_acceleration,
            f.angular_velocity, f.angular_acceleration, f.lateral_acceleration,
            f.directed_regularity, f.traversal_alignment, f.prohibited_traversal,
            f.runup_curvature
        );
    }
    if (opt.trajectory.empty()) return fail("MINCO produced empty trajectory");

    if (const auto numerical_error = validate_trajectory_numerics(opt.trajectory)) {
        return fail("MINCO produced an invalid trajectory: " + *numerical_error);
    }
    const EnvironmentValidationReport environment_validation = validate_path_environment(
        opt.trajectory,
        planning_cost_map,
        *req.direction_map,
        req.terrain_constraints,
        config_.occupied_threshold,
        config_.step_detection.detect_dot_threshold,
        config_.environment_validation
    );
    if (environment_validation.rejection) {
        return fail("environment validation rejected the path: "
            + *environment_validation.rejection);
    }
    result.warnings.insert(
        result.warnings.end(),
        environment_validation.warnings.begin(),
        environment_validation.warnings.end()
    );

    // ── 构建不可变 AnnotatedPath ──
    auto path = std::make_shared<AnnotatedPath>(opt.trajectory);
    path->goal_pos = goal_map;
    path->goal_fixed = fixed;
    path->goal_id = req.goal.id;
    path->planning_performance = req.performance;

    // 台阶几何标注：基于 base（未掩码）方向场，扫描轴为累计弧长。
    path->step_segments = step_annotator::build_step_plan(
        config_.step_detection, path->trajectory, *req.direction_map, req.terrain_constraints, logger_
    );

    const Eigen::Vector2d current_velocity_map = req.current_velocity * Eigen::Vector2d(
        std::cos(req.current_yaw), std::sin(req.current_yaw)
    );
    const SpeedProfileOptimizer speed_optimizer(config_.speed_profile);
    auto speed_result = speed_optimizer.optimize(
        path->trajectory, path->step_segments, current_velocity_map
    );
    if (!speed_result.success) {
        return fail("Speed profile optimization failed: " + speed_result.error);
    }
    path->speed_profile = std::move(speed_result.profile);
    std::vector<StepTraversalConstraint> step_constraints;
    step_constraints.reserve(path->step_segments.size());
    for (const auto& segment : path->step_segments) {
        step_constraints.push_back(segment.traversal_constraint);
    }
    path->step_constraint_schedule = std::make_shared<const StepConstraintSchedule>(std::move(step_constraints));

    const auto& speed_diagnostics = speed_result.diagnostics;
    RCLCPP_DEBUG(
        logger_,
        "Speed profile: selection=%s nodes=%d constraints=(step=%d,lateral=%d) "
        "breakpoints=%d solve=%.3f ms travel=%.2f s "
        "cost=(speed=%.3g,step=%.3g,lateral=%.3g)",
        speed_profile_selection_string(speed_diagnostics.selection),
        speed_diagnostics.node_count,
        speed_diagnostics.traversal_window_constraint_count,
        speed_diagnostics.lateral_acceleration_constraint_count,
        speed_diagnostics.max_breakpoints,
        speed_diagnostics.solve_ms,
        speed_diagnostics.result_total_time,
        speed_diagnostics.speed_reward_cost,
        speed_diagnostics.traversal_window_cost,
        speed_diagnostics.lateral_acceleration_cost
    );
    if (speed_diagnostics.selection
        == SpeedProfileOptimizer::Diagnostics::Selection::FALLBACK) {
        std::string warning = "using validated reachable speed-profile fallback";
        if (!speed_diagnostics.fallback_reason.empty()) {
            warning += ": " + speed_diagnostics.fallback_reason;
        }
        result.warnings.push_back(std::move(warning));
    }
    size_t speed_window_violation_count = 0;
    double worst_speed_window_violation = 0.0;
    size_t worst_speed_window_segment = 0;
    double worst_speed_window_arc_length = 0.0;
    for (const auto& violation : speed_diagnostics.step_violations) {
        if (violation.max_under_speed <= 0.0 && violation.max_over_speed <= 0.0) continue;
        ++speed_window_violation_count;
        const double magnitude = std::max(
            violation.max_under_speed, violation.max_over_speed
        );
        if (magnitude > worst_speed_window_violation) {
            worst_speed_window_violation = magnitude;
            worst_speed_window_segment = violation.segment_index;
            worst_speed_window_arc_length = violation.arc_length;
        }
        RCLCPP_DEBUG(
            logger_,
            "Step #%zu speed window violation at s=%.2f: under=%.2f over=%.2f "
            "hard_max=%.2f target=[%.2f,%.2f]",
            violation.segment_index,
            violation.arc_length,
            violation.max_under_speed,
            violation.max_over_speed,
            violation.hard_velocity_upper,
            violation.target.min,
            violation.target.max
        );
    }
    if (speed_window_violation_count > 0) {
        result.warnings.push_back(
            "speed profile has " + std::to_string(speed_window_violation_count)
            + " soft step-window violation(s); worst="
            + std::to_string(worst_speed_window_violation)
            + " m/s at step #" + std::to_string(worst_speed_window_segment)
            + ", s=" + std::to_string(worst_speed_window_arc_length) + " m"
        );
    }

    // 台阶掩码层：针对本条轨迹产出。
    const auto layers = step_routing_mask_->compute(path->trajectory);
    path->step_cost_layer = layers.step_cost_layer;
    path->masked_direction_map = layers.masked_direction_map;

    result.kind = PlanResult::Kind::PATH;

    result.path = std::move(path);

    double seed_path_length = 0.0;
    for (size_t i = 1; i < speed_witness.positions.size(); ++i) {
        seed_path_length += (
            speed_witness.positions[i] - speed_witness.positions[i - 1]
        ).norm();
    }
    const int segment_count = opt.trajectory.segment_count();
    const int variable_count = 2 * std::max(segment_count - 1, 0) + segment_count; // 平坦：2D 路点 + 段时长
    const auto& search_diagnostics = lattice_result.diagnostics;
    RCLCPP_DEBUG(
        logger_, "Plan done (%.2f ms): Src(%.2f,%.2f)->Dst(%.2f,%.2f) %s, %zu step segments; "
        "spatial=%.2f ms, reference=%.2f ms, guide=%.2f ms, lattice=%.2f ms, MINCO=%.2f ms, "
        "spatial_search[exp=%d open=%zu raw_cells=%zu passages=%zu raw_length=%.2f smoothed_length=%.2f corridor_cells=%zu], "
        "witness_length=%.2f m, witness_states=%zu, "
        "search[root=%s bias=%.3f time=%.2f exp=%d states=%d improved=%d "
        "transitions=%d terminal=%d open=%zu stale=%zu "
        "geometry_cache=%zu hits=%zu terrain_spans=%zu corridor_rejections=%zu], "
        "primitives[count=%zu pos_res=%.3g heading_res=%.3g], "
        "segments=%d, vars=%d, optimizer=%.*s, accepted=%d, evals=%d, "
        "first_order=%.3g, scaled_grad=%.3g, raw_grad=%.3g",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - plan_start).count(),
        start_map.x(), start_map.y(), goal_plan.x(), goal_plan.y(),
        fixed ? "[FIXED]" : "", result.path->step_segments.size(),
        std::chrono::duration<double, std::milli>(spatial_done - plan_start).count(),
        std::chrono::duration<double, std::milli>(reference_done - spatial_done).count(),
        std::chrono::duration<double, std::milli>(guide_done - reference_done).count(),
        std::chrono::duration<double, std::milli>(lattice_done - guide_done).count(),
        std::chrono::duration<double, std::milli>(minco_done - minco_start).count(),
        spatial_route.expansions, spatial_route.open_peak,
        spatial_route.raw_path.size(), spatial_route.passages.size(),
        reference_path.raw_length, reference_path.smoothed_length,
        guide_field.corridor_cell_count(),
        seed_path_length, speed_witness.positions.size(),
        search_diagnostics.selected_relaxed_root ? "relaxed-yaw" : "strict",
        search_diagnostics.selected_relaxation_bias,
        search_diagnostics.search_time,
        search_diagnostics.expansions,
        search_diagnostics.generated_states,
        search_diagnostics.improved_states,
        search_diagnostics.transition_candidates,
        search_diagnostics.terminal_attempts,
        search_diagnostics.open_peak,
        search_diagnostics.stale_queue_entries,
        search_diagnostics.geometry_cache_entries,
        search_diagnostics.geometry_cache_hits,
        search_diagnostics.terrain_spans_checked,
        search_diagnostics.corridor_rejections,
        primitive_library_.primitive_count(),
        primitive_library_.max_position_residual(),
        primitive_library_.max_heading_residual(),
        segment_count, variable_count,
        static_cast<int>(opt.optimizer_status_string().size()), opt.optimizer_status_string().data(),
        opt.accepted_iterations, opt.function_evaluations,
        opt.final_first_order_optimality,
        opt.final_scaled_grad_max_block_norm,
        opt.final_grad_inf_norm
    );

    return result;
}

} // namespace nav_executor
