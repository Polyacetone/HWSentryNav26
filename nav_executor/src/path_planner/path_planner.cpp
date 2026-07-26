#include <nav_executor/path_planner/path_planner.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <queue>

#include <rclcpp/logging.hpp>

#include <nav_executor/common/trajectory_topology.hpp>

namespace nav_executor {

namespace {

double wrap_angle(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

// kinodynamic 前向平坦状态序列 → MINCO 平坦边界状态 + 段时长。
struct MincoSeed {
    std::vector<MincoMinJerk::BoundaryPVA> states; // 2D pos/vel/acc
    std::vector<double> durations;
};

// 距离和显著转向点会保留；段时长严格累加 A* 返回的真实逐边时长。
MincoSeed build_minco_seed(
    const std::vector<KinodynamicAstar::State>& raw,
    const double resample_distance,
    const std::vector<double>& raw_durations
) {
    MincoSeed seed;
    if (raw.size() < 2 || raw_durations.size() + 1 != raw.size()) return seed;

    std::vector<size_t> selected {0};
    const double distance_threshold = std::max(resample_distance, 0.05);
    constexpr double HEADING_THRESHOLD = 0.5;
    for (size_t i = 1; i + 1 < raw.size(); ++i) {
        const size_t last = selected.back();
        const double distance = (raw[i].position - raw[last].position).norm();
        const double heading = std::atan2(raw[i].velocity.y(), raw[i].velocity.x());
        const double last_heading = std::atan2(
            raw[last].velocity.y(), raw[last].velocity.x()
        );
        const double heading_change = std::abs(wrap_angle(heading - last_heading));
        if (distance >= distance_threshold || heading_change >= HEADING_THRESHOLD) {
            selected.push_back(i);
        }
    }
    selected.push_back(raw.size() - 1);

    const size_t n = selected.size();
    seed.states.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const auto& s = raw[selected[i]];
        auto& bs = seed.states[i];
        bs.pos = s.position;
        bs.vel = s.velocity;
        bs.acc.setZero();
    }
    seed.durations.resize(n - 1);
    for (size_t i = 0; i + 1 < n; ++i) {
        double duration = 0.0;
        for (size_t edge = selected[i]; edge < selected[i + 1]; ++edge) {
            duration += raw_durations[edge];
        }
        seed.durations[i] = std::max(duration, 0.1);
    }

    // 规划终点只约束位置停止；起点沿用当前运动状态。
    seed.states.back().vel.setZero();
    seed.states.back().acc.setZero();

    return seed;
}

bool validate_path_geometry(
    const MincoTrajectory& trajectory,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const int occupied_threshold,
    const double step_alignment_threshold,
    const PlannerConfig::GeometryValidationParams& validation,
    std::string& error,
    std::string& soft_diagnostic
) {
    const double total_time = trajectory.total_time();
    if (!std::isfinite(total_time) || total_time <= 0.0) {
        error = "trajectory has invalid total time";
        return false;
    }
    const double total_arc_length = trajectory.total_arc_length();
    if (!std::isfinite(total_arc_length) || total_arc_length < 0.0) {
        error = "trajectory has invalid total arc length";
        return false;
    }
    const auto self_intersection = find_disallowed_self_intersection(
        trajectory,
        {
            .flatness_tolerance = validation.self_intersection_flatness_tolerance,
            .max_edge_length = validation.self_intersection_max_edge_length,
        }
    );
    if (self_intersection) {
        error = "trajectory self-intersects between MINCO segments "
            + std::to_string(self_intersection->first_segment) + " and "
            + std::to_string(self_intersection->second_segment);
        return false;
    }

    const auto validate_sample = [&](const double tau) {
        const TrajSample s = trajectory.eval(tau);
        if (!s.p.allFinite() || !std::isfinite(s.theta) || !s.dp_dtau.allFinite()
            || !s.ddp_dtau.allFinite() || !std::isfinite(s.dtheta_dtau)) {
            error = "trajectory contains non-finite values at tau=" + std::to_string(tau)
                + ": p=(" + std::to_string(s.p.x()) + "," + std::to_string(s.p.y()) + ")"
                + " theta=" + std::to_string(s.theta)
                + " dp_dtau=(" + std::to_string(s.dp_dtau.x()) + "," + std::to_string(s.dp_dtau.y()) + ")"
                + " ddp_dtau=(" + std::to_string(s.ddp_dtau.x()) + "," + std::to_string(s.ddp_dtau.y()) + ")"
                + " dtheta_dtau=" + std::to_string(s.dtheta_dtau);
            return false;
        }
        const Eigen::Vector2d grid = cost_map.map_coord_to_grid(s.p);
        if (!cost_map.is_valid_coord(grid)) {
            error = "trajectory leaves planning map at tau=" + std::to_string(tau);
            return false;
        }
        if (cost_map.interpolate(grid) >= static_cast<double>(occupied_threshold)) {
            error = "trajectory intersects occupied cost at tau=" + std::to_string(tau)
                + ": cost=" + std::to_string(cost_map.interpolate(grid))
                + " >= threshold=" + std::to_string(occupied_threshold);
            return false;
        }

        const Eigen::Vector2d dir_grid = direction_map.map_coord_to_grid(s.p);
        if (!direction_map.is_valid_coord(dir_grid)) {
            error = "trajectory leaves direction map at tau=" + std::to_string(tau);
            return false;
        }
        if (direction_map.is_terrain_body_at(dir_grid)) {
            const Eigen::Array2i cell_array = dir_grid.array().floor().cast<int>();
            const Eigen::Vector2i cell(cell_array.x(), cell_array.y());
            const uint8_t label = direction_map.terrain_at(cell);
            const Eigen::Vector2d raw_dir = direction_map.at(cell);
            if (label < static_cast<uint8_t>(TerrainType::SLOPE)
                || raw_dir.squaredNorm() <= 1e-12) {
                error = "trajectory encountered invalid directional terrain body at tau="
                    + std::to_string(tau);
                return false;
            }
            const Eigen::Vector2d dir = raw_dir.normalized();
            const Eigen::Vector2d heading(std::cos(s.theta), std::sin(s.theta));
            const double alignment = heading.dot(dir);
            if (std::abs(alignment) <= step_alignment_threshold) {
                error = "trajectory is not aligned with directional terrain at tau=" + std::to_string(tau)
                    + ": |heading.dot(dir)|=" + std::to_string(std::abs(alignment))
                    + " <= threshold=" + std::to_string(step_alignment_threshold);
                return false;
            }
            const bool going_up = alignment >= 0.0;
            const TraversalMode* rule = terrain_constraints.selected_mode(label, going_up);
            if (!rule) {
                error = "trajectory uses prohibited directional terrain at tau=" + std::to_string(tau)
                    + ": label=" + std::to_string(static_cast<int>(label))
                    + ", direction=" + (going_up ? "up" : "down")
                    + ", |dir|=" + std::to_string(raw_dir.norm());
                return false;
            }
            if (soft_diagnostic.empty()) {
                const double angle_rad = std::acos(
                    std::clamp(std::abs(alignment), 0.0, 1.0)
                );
                if (angle_rad > validation.traversal_angle_tolerance) {
                    soft_diagnostic = "trajectory deviates from stair direction at tau="
                        + std::to_string(tau)
                        + ": angle=" + std::to_string(angle_rad)
                        + " rad, tolerance=" + std::to_string(validation.traversal_angle_tolerance)
                        + " rad";
                }
            }
        }
        return true;
    };

    // 时间均匀采样负责动态量硬验收，并同时复用同一个位置/地形检查器。
    const int samples_per_segment = std::max(validation.samples_per_segment, 1);
    for (int segment = 0; segment < trajectory.segment_count(); ++segment) {
        const double tau_begin = trajectory.segment_boundary_tau(segment);
        const double tau_end = trajectory.segment_boundary_tau(segment + 1);
        for (int sample = 0; sample <= samples_per_segment; ++sample) {
            const double fraction = static_cast<double>(sample)
                / static_cast<double>(samples_per_segment);
            const double tau = std::lerp(tau_begin, tau_end, fraction);
            if (!validate_sample(tau)) return false;
        }
    }

    // 额外按弧长采样，避免物理本体缩窄为原始格后被固定 tau 网格跨过。
    const double max_spatial_spacing = direction_map.resolution * 0.5;
    const int spatial_intervals = std::max(
        1, static_cast<int>(std::ceil(total_arc_length / max_spatial_spacing))
    );
    for (int sample = 0; sample <= spatial_intervals; ++sample) {
        const double arc_length = total_arc_length * static_cast<double>(sample)
            / static_cast<double>(spatial_intervals);
        const double tau = trajectory.tau_at_arc_length(arc_length);
        if (!validate_sample(tau)) return false;
    }
    return true;
}

} // anonymous namespace

PathPlanner::PathPlanner(
    const PlannerConfig& config,
    std::shared_ptr<StepRoutingMask> step_routing_mask,
    rclcpp::Logger logger
) : config_(config),
    step_routing_mask_(std::move(step_routing_mask)),
    logger_(logger) {}

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
    const Eigen::Vector2d cost_grid = cost_map.map_coord_to_grid(map_pt);
    const Eigen::Vector2d dir_grid = direction_map.map_coord_to_grid(map_pt);
    if (!cost_map.is_valid_coord(cost_grid) || !direction_map.is_valid_coord(dir_grid)) return false;

    return cost_map.interpolate(cost_grid) < config_.occupied_threshold
        && direction_map.interpolate(dir_grid).norm() < config_.on_step_threshold;
}

std::optional<Eigen::Vector2d> PathPlanner::nudge_point_to_free(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const Eigen::Vector2d& map_pt,
    const double max_nudge_distance
) const {
    const Eigen::Vector2d grid_pt = cost_map.map_coord_to_grid(map_pt);
    const int width = cost_map.width;
    const int height = cost_map.height;

    const auto key = [width](const int x, const int y) {
        return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
    };

    if (cost_map.is_valid_coord(grid_pt)
        && is_map_point_feasible(cost_map, direction_map, map_pt)) {
        return map_pt;
    }

    const int sx = static_cast<int>(std::round(grid_pt.x()));
    const int sy = static_cast<int>(std::round(grid_pt.y()));
    if (sx < 0 || sx >= width || sy < 0 || sy >= height) return std::nullopt;

    std::vector<uint8_t> visited(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
    std::queue<Eigen::Vector2i> q;
    q.push(Eigen::Vector2i(sx, sy));
    visited[key(sx, sy)] = 1;

    static constexpr int dx[] = {0, 1, 0, -1, 1, 1, -1, -1};
    static constexpr int dy[] = {1, 0, -1, 0, 1, -1, 1, -1};

    while (!q.empty()) {
        const auto current = q.front();
        q.pop();

        const double dist = (current.cast<double>() - grid_pt).norm() * cost_map.resolution;
        if (dist > max_nudge_distance) continue;

        const Eigen::Vector2d candidate_map = cost_map.grid_coord_to_map(current.cast<double>());
        if (is_map_point_feasible(cost_map, direction_map, candidate_map)) {
            return candidate_map;
        }

        for (int i = 0; i < 8; i++) {
            const int nx = current.x() + dx[i];
            const int ny = current.y() + dy[i];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            const double ndist = (Eigen::Vector2d(nx, ny) - grid_pt).norm()
                * cost_map.resolution;
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

    if (!req.global_cost_map || !req.merged_cost_map || !req.direction_map || !req.terrain_constraints.blocked_cost_layer) {
        RCLCPP_WARN(logger_, "Plan aborted: map snapshot incomplete");
        result.kind = PlanResult::Kind::FAILED;
        return result;
    }

    const Eigen::Vector2d goal_map = req.goal.position_map;
    const bool fixed = req.goal.fixed;

    const auto fail = [&](const std::string& msg) {
        RCLCPP_ERROR(logger_, "Plan failed: %s", msg.c_str());
        result.kind = PlanResult::Kind::FAILED;
        return result;
    };

    const CostMap global_feasibility_cost = req.global_cost_map->merge(
        *req.terrain_constraints.blocked_cost_layer
    );
    const CostMap planning_cost_map = req.merged_cost_map->merge(
        *req.terrain_constraints.blocked_cost_layer
    );

    Eigen::Vector2d start_map = req.current_pos_map;
    Eigen::Vector2d goal_plan = goal_map;

    // ── 起点 global 严格检查 ──
    {
        const Eigen::Vector2d sg = global_feasibility_cost.map_coord_to_grid(start_map);
        if (!global_feasibility_cost.is_valid_coord(sg)) return fail("Start is out of bound");
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
        const Eigen::Vector2d gg = global_feasibility_cost.map_coord_to_grid(goal_map);
        if (!global_feasibility_cost.is_valid_coord(gg)) return fail("Goal is out of bound");
        if (!is_map_point_feasible(global_feasibility_cost, *req.direction_map, goal_map)) return fail("Goal is not feasible on global map");
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
        RCLCPP_INFO(
            logger_, "Feasible goal within reached distance (%.2f m): %s",
            config_.goal_reached_distance,
            fixed ? "USE_AS_FIXED_GOAL" : "COMPLETE_NO_PLAN_NEEDED"
        );
        return result;
    }

    const Eigen::Vector2d goal_grid = planning_cost_map.map_coord_to_grid(goal_plan);
    const auto plan_start = std::chrono::steady_clock::now();

    // ── [1] Dijkstra cost-to-goal（map 系工作，供 kinodynamic h + 开阔 seed）──
    DijkstraCostToGoal dijkstra;
    dijkstra.build(planning_cost_map, goal_grid.cast<int>(), config_.dijkstra);
    if (!dijkstra.ready() || std::isinf(dijkstra.at_map(start_map))) {
        return fail("Dijkstra: goal unreachable from start");
    }
    const auto dijkstra_done = std::chrono::steady_clock::now();

    // 空间原语逐子步约束：碰撞、方向地形通行方向以及局部允许速度范围。
    const CostMap* const planning_map_ptr = &planning_cost_map;
    const auto transition_constraint = [
        planning_map_ptr,
        direction_map = req.direction_map.get(),
        &terrain = req.terrain_constraints,
        &config = config_
    ](const KinodynamicAstar::Pose& from, const KinodynamicAstar::Pose& to)
        -> std::optional<KinodynamicAstar::SpeedRange> {
        const Eigen::Vector2d& map_pt = to.position;
        const Eigen::Vector2d g = planning_map_ptr->map_coord_to_grid(map_pt);
        if (!planning_map_ptr->is_valid_coord(g)) return std::nullopt;
        if (planning_map_ptr->interpolate(g) >= config.occupied_threshold) {
            return std::nullopt;
        }

        const Eigen::Vector2d dg = direction_map->map_coord_to_grid(map_pt);
        if (!direction_map->is_valid_coord(dg)) return std::nullopt;
        if (!direction_map->is_terrain_body_at(dg)) {
            return KinodynamicAstar::SpeedRange {
                .min = 0.0,
                .max = config.kinodynamic.state_limits.speed_max,
            };
        }
        const Eigen::Array2i cell_array = dg.array().floor().cast<int>();
        const Eigen::Vector2i cell(cell_array.x(), cell_array.y());
        const uint8_t label = direction_map->terrain_at(cell);
        const Eigen::Vector2d raw_dir = direction_map->at(cell);
        if (label < static_cast<uint8_t>(TerrainType::SLOPE)
            || raw_dir.squaredNorm() <= 1e-12) return std::nullopt;

        const Eigen::Vector2d displacement = to.position - from.position;
        if (displacement.norm() < 1e-6) return std::nullopt;
        const Eigen::Vector2d dir = raw_dir.normalized();
        const double travel_alignment = displacement.normalized().dot(dir);
        if (std::abs(travel_alignment) <= config.step_detection.detect_dot_threshold) {
            return std::nullopt;
        }
        const TraversalMode* rule = terrain.selected_mode(label, travel_alignment >= 0.0);
        if (!rule) return std::nullopt;

        return KinodynamicAstar::SpeedRange {
            .min = rule->velocity_window.min,
            .max = rule->velocity_window.max,
        };
    };

    // ── [2] Kinodynamic A*：空间曲率原语 + 可达速度区间搜索 ──
    KinodynamicAstar::State strict_start;
    strict_start.position = start_map;
    const double start_reference_speed = std::clamp(
        req.current_velocity,
        config_.forward_velocity_bounds.min,
        config_.forward_velocity_bounds.max
    );
    strict_start.velocity = start_reference_speed * Eigen::Vector2d(
        std::cos(req.current_yaw), std::sin(req.current_yaw)
    );

    std::vector<KinodynamicAstar::SearchRoot> kino_roots;
    kino_roots.push_back({
        .state = strict_start,
        .initial_cost = 0.0,
        .relaxed = false,
    });
    if (std::abs(req.current_velocity)
        <= config_.start_yaw_relaxation.speed_threshold) {
        const int heading_bins = std::max(
            1,
            static_cast<int>(std::llround(
                2.0 * M_PI / config_.kinodynamic.dedup_theta
            ))
        );
        const double relaxed_speed = config_.forward_velocity_bounds.min;
        for (int bin = 0; bin < heading_bins; ++bin) {
            const double yaw = wrap_angle(
                req.current_yaw
                + 2.0 * M_PI * static_cast<double>(bin)
                    / static_cast<double>(heading_bins)
            );
            const double yaw_change = std::abs(wrap_angle(yaw - req.current_yaw));
            KinodynamicAstar::State relaxed_start;
            relaxed_start.position = start_map;
            relaxed_start.velocity = relaxed_speed * Eigen::Vector2d(
                std::cos(yaw), std::sin(yaw)
            );
            kino_roots.push_back({
                .state = relaxed_start,
                .initial_cost = config_.start_yaw_relaxation.root_penalty
                    + config_.start_yaw_relaxation.yaw_penalty * yaw_change,
                .relaxed = true,
            });
        }
    }

    KinodynamicAstar::Params kinodynamic_params = config_.kinodynamic;
    MincoOptimizer::Params minco_params = config_.minco;

    KinodynamicAstar astar(kinodynamic_params);
    const auto kino = astar.search(
        kino_roots, goal_plan, dijkstra, transition_constraint
    );
    if (!kino.success) {
        return fail(
            "Kinodynamic A* failed: " + kino.error
            + " (expansions=" + std::to_string(kino.expansions)
            + ", labels=" + std::to_string(kino.generated_labels)
            + ", dominated=" + std::to_string(kino.dominated_labels)
            + ", transitions=" + std::to_string(kino.transition_checks)
            + ", goal_attempts=" + std::to_string(kino.goal_connection_attempts)
            + ", open_peak=" + std::to_string(kino.open_peak) + ")"
        );
    }
    const std::vector<KinodynamicAstar::State>& seed_states_raw = kino.states;
    const auto kinodynamic_done = std::chrono::steady_clock::now();

    // ── 种子重采样为 MINCO 边界全状态 + 段时长 ──
    const auto minco_seed = build_minco_seed(
        seed_states_raw,
        config_.seed_resample_distance,
        kino.durations
    );
    if (minco_seed.durations.empty()) return fail("MINCO seed construction produced no segments");
    // ── [3] MINCO 时空优化 ──
    MincoOptimizer optimizer(minco_params);
    const auto minco_start = std::chrono::steady_clock::now();
    const auto opt = optimizer.optimize(
        minco_seed.states,
        minco_seed.durations,
        planning_cost_map,
        *req.direction_map,
        req.terrain_constraints
    );
    if (!opt.success) return fail("MINCO optimization failed: " + opt.error);
    const auto minco_done = std::chrono::steady_clock::now();

    if (config_.enable_debug && opt.diagnostics_valid) {
        const auto& s = opt.seed_costs;
        const auto& f = opt.final_costs;
        const std::string_view status = opt.optimizer_status_string();
        RCLCPP_DEBUG(
            logger_,
            "MINCO optimizer: status=%.*s accepted=%d evals=%d trials=%d rejected=%d nonfinite=%d | "
            "raw |grad|_inf %.3g -> %.3g (pos=%.3g, virtual_time=%.3g), normalized_scaled_max_block=%.3g | "
            "radius initial=%.3g final=%.3g range=[%.3g,%.3g] shrink=%d expand=%d boundary=%d | "
            "history update=%d skip=%d reset=%d",
            static_cast<int>(status.size()), status.data(),
            opt.accepted_iterations, opt.function_evaluations, opt.trial_evaluations,
            opt.rejected_trials, opt.nonfinite_trials,
            opt.initial_grad_inf_norm, opt.final_grad_inf_norm,
            opt.final_grad_pos_inf_norm, opt.final_grad_time_inf_norm,
            opt.final_normalized_scaled_grad_max_block_norm,
            opt.initial_radius, opt.final_radius, opt.min_radius, opt.max_radius,
            opt.radius_shrinks, opt.radius_expansions, opt.boundary_steps,
            opt.history_updates, opt.history_skips, opt.history_resets
        );
        RCLCPP_DEBUG(
            logger_,
            "MINCO objective: cost %.3g -> %.3g | waypoints free=%d disp(sum=%.3f m, max=%.3f m) | "
            "seed[E=%.3g T=%.3g O=%.3g V=%.3g Lat=%.3g W=%.3g A=%.3g SA=%.3g SV=%.3g SP=%.3g RA=%.3g RW=%.3g] "
            "final[E=%.3g T=%.3g O=%.3g V=%.3g Lat=%.3g W=%.3g A=%.3g SA=%.3g SV=%.3g SP=%.3g RA=%.3g RW=%.3g]",
            s.total(), f.total(),
            opt.free_waypoint_count, opt.waypoint_total_displacement, opt.waypoint_max_displacement,
            s.energy, s.time, s.obstacle, s.trajectory_velocity, s.lateral_acc,
            s.omega, s.accel, s.traversal_alignment, s.traversal_velocity_target,
            s.prohibited_traversal, s.runup_accel, s.runup_omega,
            f.energy, f.time, f.obstacle, f.trajectory_velocity, f.lateral_acc,
            f.omega, f.accel, f.traversal_alignment, f.traversal_velocity_target,
            f.prohibited_traversal, f.runup_accel, f.runup_omega
        );
        if (opt.grad_check_max_rel_err >= 0.0) {
            RCLCPP_DEBUG(
                logger_,
                "MINCO grad check (seed): max_abs_err=%.3g max_rel_err=%.3g worst_var=%d",
                opt.grad_check_max_abs_err, opt.grad_check_max_rel_err,
                opt.grad_check_worst_index
            );
        }
    }
    if (opt.trajectory.empty()) return fail("MINCO produced empty trajectory");
    std::string trajectory_error;
    std::string trajectory_soft_diagnostic;
    if (!validate_path_geometry(
            opt.trajectory,
            planning_cost_map,
            *req.direction_map,
            req.terrain_constraints,
            config_.occupied_threshold,
            config_.step_detection.detect_dot_threshold,
            config_.geometry_validation,
            trajectory_error,
            trajectory_soft_diagnostic
        )) {
        return fail("MINCO output rejected: " + trajectory_error);
    }
    if (!trajectory_soft_diagnostic.empty()) {
        RCLCPP_WARN(logger_, "MINCO soft target diagnostic: %s", trajectory_soft_diagnostic.c_str());
    }

    // ── 构建不可变 AnnotatedPath ──
    auto path = std::make_shared<AnnotatedPath>(opt.trajectory);
    path->goal_pos = goal_map;
    path->goal_fixed = fixed;
    path->goal_id = req.goal.id;
    path->planning_performance = req.performance;

    // 台阶几何标注：基于 base（未掩码）方向场，扫描轴为 MINCO 参数 τ。
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
    RCLCPP_INFO(
        logger_,
        "Speed profile: nodes=%d vars=%d constraints=%d windows=%d iterations=%d "
        "residual=(%.3g,%.3g) violation=%.3g rho_updates=%d time=%.2f ms "
        "travel=%.2f s cost=(speed=%.3g,step=%.3g) fallback=%s polish=%s/%s",
        speed_diagnostics.node_count,
        speed_diagnostics.variable_count,
        speed_diagnostics.constraint_count,
        speed_diagnostics.soft_window_node_count,
        speed_diagnostics.iterations,
        speed_diagnostics.primal_residual,
        speed_diagnostics.dual_residual,
        speed_diagnostics.max_constraint_violation,
        speed_diagnostics.rho_updates,
        speed_diagnostics.factorization_ms + speed_diagnostics.iteration_ms,
        speed_diagnostics.result_total_time,
        speed_diagnostics.speed_reward_cost,
        speed_diagnostics.traversal_window_cost,
        speed_diagnostics.used_fallback ? "yes" : "no",
        speed_diagnostics.polish_attempted ? "attempted" : "skipped",
        speed_diagnostics.polish_accepted ? "accepted" : "rejected"
    );
    for (const auto& violation : speed_diagnostics.step_violations) {
        if (violation.max_under_speed <= 0.0 && violation.max_over_speed <= 0.0) continue;
        RCLCPP_WARN(
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

    // 台阶掩码层：针对本条轨迹产出。
    const auto layers = step_routing_mask_->compute(path->trajectory);
    path->step_cost_layer = layers.step_cost_layer;
    path->masked_direction_map = layers.masked_direction_map;

    result.kind = PlanResult::Kind::PATH;

    if (config_.enable_debug) {
        std::vector<Eigen::Vector2d> seed_pts;
        seed_pts.reserve(seed_states_raw.size());
        for (const auto& s : seed_states_raw) seed_pts.push_back(s.position);
        result.debug_rough_path = std::move(seed_pts);
    }

    result.path = std::move(path);

    double seed_path_length = 0.0;
    for (size_t i = 1; i < seed_states_raw.size(); ++i) {
        seed_path_length += (
            seed_states_raw[i].position - seed_states_raw[i - 1].position
        ).norm();
    }
    const int segment_count = opt.trajectory.segment_count();
    const int variable_count = 2 * std::max(segment_count - 1, 0) + segment_count; // 平坦：2D 路点 + 段时长
    RCLCPP_INFO(
        logger_, "Plan done (%.2f ms): Src(%.2f,%.2f)->Dst(%.2f,%.2f) %s, %zu step segments; "
        "Dijkstra=%.2f ms, Kino=%.2f ms, MINCO=%.2f ms, seed_length=%.2f m, raw_states=%zu, "
        "kino[root=%s root_cost=%.2f exp=%d labels=%d dominated=%d transitions=%d goal=%d open=%zu], "
        "segments=%d, vars=%d, optimizer=%.*s, accepted=%d, evals=%d, "
        "normalized_scaled_grad=%.3g, raw_grad=%.3g",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - plan_start).count(),
        start_map.x(), start_map.y(), goal_plan.x(), goal_plan.y(),
        fixed ? "[FIXED]" : "", result.path->step_segments.size(),
        std::chrono::duration<double, std::milli>(dijkstra_done - plan_start).count(),
        std::chrono::duration<double, std::milli>(kinodynamic_done - dijkstra_done).count(),
        std::chrono::duration<double, std::milli>(minco_done - minco_start).count(),
        seed_path_length, seed_states_raw.size(),
        kino.selected_relaxed_root ? "relaxed-yaw" : "strict",
        kino.selected_root_cost,
        kino.expansions, kino.generated_labels, kino.dominated_labels,
        kino.transition_checks, kino.goal_connection_attempts, kino.open_peak,
        segment_count, variable_count,
        static_cast<int>(opt.optimizer_status_string().size()), opt.optimizer_status_string().data(),
        opt.accepted_iterations, opt.function_evaluations,
        opt.final_normalized_scaled_grad_max_block_norm, opt.final_grad_inf_norm
    );

    return result;
}

} // namespace nav_executor
