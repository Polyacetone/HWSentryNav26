#include <nav_executor/path_planner/path_planner.hpp>

#include <algorithm>
#include <array>
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

// kinodynamic 状态序列 → MINCO 平坦边界状态 + 段时长 + 换向拓扑。
struct MincoSeed {
    std::vector<MincoMinJerk::BoundaryPVA> states; // 2D pos/vel/acc
    std::vector<double> durations;
    std::vector<double> gears;                     // 每段 ±1
    std::vector<char> cusp_waypoints;              // 每内部节点是否换向尖点
};

// 把 kinodynamic (x,y,θ,v) 序列重采样后转成 MINCO 平坦边界状态 + 换向拓扑。
//   - 平坦速度 = (v cosθ, v sinθ)（2D）；加速度置零（MINCO 会调整）。
//   - 每段 gear = 段内主导速度符号（前进 +1 / 倒车 −1），由前端冻结。
//   - gear 在相邻段翻转处标记为换向尖点：该内部节点两侧 v=0（MINCO 结构化施加）。
//   - 距离、显著转向和换向点都会保留，避免把倒车尖点重采样掉。
//   - 段时长沿用 A* 原语时间，避免用低速处的弧长/速度比制造异常长分段。
MincoSeed build_minco_seed(
    const std::vector<KinodynamicAstar::State>& raw,
    const double resample_distance,
    const double state_interval
) {
    MincoSeed seed;
    if (raw.size() < 2) return seed;

    std::vector<size_t> selected {0};
    const double distance_threshold = std::max(resample_distance, 0.05);
    constexpr double HEADING_THRESHOLD = 0.5;
    constexpr double VELOCITY_EPS = 0.05;
    for (size_t i = 1; i + 1 < raw.size(); ++i) {
        const size_t last = selected.back();
        const double distance = std::hypot(raw[i].x - raw[last].x, raw[i].y - raw[last].y);
        const double heading_change = std::abs(wrap_angle(raw[i].theta - raw[last].theta));
        const bool gear_change = (raw[i - 1].v < -VELOCITY_EPS && raw[i + 1].v > VELOCITY_EPS)
            || (raw[i - 1].v > VELOCITY_EPS && raw[i + 1].v < -VELOCITY_EPS);
        if (distance >= distance_threshold || heading_change >= HEADING_THRESHOLD || gear_change) {
            selected.push_back(i);
        }
    }
    selected.push_back(raw.size() - 1);

    const size_t n = selected.size();

    // ── 每段 gear：段内所有 raw 速度的带符号均值符号；接近 0 时沿用上一段。──
    seed.gears.assign(n - 1, 1.0);
    double prev_gear = raw[selected[0]].v < -VELOCITY_EPS ? -1.0 : 1.0;
    for (size_t i = 0; i + 1 < n; ++i) {
        double v_sum = 0.0;
        for (size_t r = selected[i]; r <= selected[i + 1]; ++r) v_sum += raw[r].v;
        double gear = prev_gear;
        if (std::abs(v_sum) > VELOCITY_EPS) gear = v_sum < 0.0 ? -1.0 : 1.0;
        seed.gears[i] = gear;
        prev_gear = gear;
    }

    // ── 换向尖点：相邻段 gear 翻转的内部节点（waypoint index = i-1，节点在段 i-1|i 之间）。──
    seed.cusp_waypoints.assign(n >= 2 ? n - 1 : 0, 0);
    for (size_t i = 1; i + 1 < n; ++i) {
        if (seed.gears[i - 1] != seed.gears[i]) {
            seed.cusp_waypoints[i - 1] = 1;
        }
    }

    seed.states.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const auto& s = raw[selected[i]];
        auto& bs = seed.states[i];
        bs.pos << s.x, s.y;
        bs.vel << s.v * std::cos(s.theta), s.v * std::sin(s.theta);
        bs.acc.setZero();
    }
    seed.durations.resize(n - 1);
    for (size_t i = 0; i + 1 < n; ++i) {
        seed.durations[i] = std::max(
            state_interval * static_cast<double>(selected[i + 1] - selected[i]),
            0.1
        );
    }

    // 规划终点只约束位置停止；起点沿用当前运动状态。
    seed.states.back().vel.setZero();
    seed.states.back().acc.setZero();

    return seed;
}

bool validate_trajectory(
    const MincoTrajectory& trajectory,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const int occupied_threshold,
    const double step_alignment_threshold,
    const MincoOptimizer::TrajectoryLimits& limits,
    const PlannerConfig::TrajectoryValidationParams& validation,
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
            .cusp_retrace_alignment_threshold = validation.cusp_retrace_alignment_threshold,
        }
    );
    if (self_intersection) {
        error = "trajectory self-intersects between MINCO segments "
            + std::to_string(self_intersection->first_segment) + " and "
            + std::to_string(self_intersection->second_segment);
        return false;
    }

    const auto validate_sample = [&](const double tau, const bool check_dynamics) {
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
        const double longitudinal_velocity = trajectory.longitudinal_velocity(s);
        const double omega = trajectory.angular_velocity(s);
        if (check_dynamics) {
            const Eigen::Vector2d acceleration = s.ddp_dtau / (total_time * total_time);
            // 非完整约束在平坦表示下恒等满足（θ=atan2(gear·ṗ)），无需校验。
            if (longitudinal_velocity < limits.velocity.min - validation.velocity_tolerance
                || longitudinal_velocity > limits.velocity.max + validation.velocity_tolerance) {
                error = "trajectory violates velocity limit at tau=" + std::to_string(tau)
                    + ": v=" + std::to_string(longitudinal_velocity)
                    + " (allowed=[" + std::to_string(limits.velocity.min)
                    + "," + std::to_string(limits.velocity.max)
                    + "], tolerance=" + std::to_string(validation.velocity_tolerance) + ")";
                return false;
            }
            if (std::abs(omega)
                > limits.angular_velocity_max + validation.omega_tolerance) {
                error = "trajectory violates angular velocity limit at tau=" + std::to_string(tau)
                    + ": |omega|=" + std::to_string(std::abs(omega))
                    + " > angular_velocity_max="
                    + std::to_string(limits.angular_velocity_max)
                    + " + tolerance=" + std::to_string(validation.omega_tolerance)
                    + " = " + std::to_string(
                        limits.angular_velocity_max + validation.omega_tolerance
                    );
                return false;
            }
            if (acceleration.norm()
                > limits.acceleration_max + validation.acceleration_tolerance) {
                error = "trajectory violates acceleration limit at tau=" + std::to_string(tau)
                    + ": |acc|=" + std::to_string(acceleration.norm())
                    + " > acceleration_max=" + std::to_string(limits.acceleration_max)
                    + " + tolerance=" + std::to_string(validation.acceleration_tolerance)
                    + " = " + std::to_string(
                        limits.acceleration_max + validation.acceleration_tolerance
                    );
                return false;
            }
            if (std::abs(longitudinal_velocity * omega)
                > limits.lateral_acceleration_max
                    + validation.lateral_acceleration_tolerance) {
                error = "trajectory violates lateral acceleration limit at tau=" + std::to_string(tau)
                    + ": |v*omega|=" + std::to_string(std::abs(longitudinal_velocity * omega))
                    + " > lateral_acceleration_max="
                    + std::to_string(limits.lateral_acceleration_max)
                    + " + tolerance=" + std::to_string(validation.lateral_acceleration_tolerance)
                    + " = " + std::to_string(
                        limits.lateral_acceleration_max
                        + validation.lateral_acceleration_tolerance
                    );
                return false;
            }
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
            if (s.gear < 0.0) {
                error = "trajectory reverses on directional terrain body at tau="
                    + std::to_string(tau);
                return false;
            }
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
            const auto& target = rule->velocity_window;
            if (soft_diagnostic.empty()
                && (longitudinal_velocity
                        < target.min - validation.traversal_velocity_target_tolerance
                    || longitudinal_velocity
                        > target.max + validation.traversal_velocity_target_tolerance)) {
                soft_diagnostic = "trajectory misses traversal velocity target at tau="
                    + std::to_string(tau)
                    + ": v=" + std::to_string(longitudinal_velocity) + ", required=["
                    + std::to_string(target.min) + "," + std::to_string(target.max)
                    + "]";
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
            if (!validate_sample(tau, true)) return false;
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
        if (!validate_sample(tau, false)) return false;
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

Eigen::Vector2d PathPlanner::adjust_reachable_start_on_segment(
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const Eigen::Vector2d& from_map,
    const Eigen::Vector2d& to_map
) const {
    const Eigen::Vector2d delta = to_map - from_map;
    const double length = delta.norm();
    if (length <= 1e-6) return from_map;

    const double step = std::max(1e-3, config_.start_prediction_collision_check_step);
    const int n = std::max(1, static_cast<int>(std::ceil(length / step)));
    const Eigen::Vector2d dir = delta / length;

    Eigen::Vector2d last_feasible = from_map;
    for (int i = 0; i <= n; ++i) {
        const double d = length * (static_cast<double>(i) / static_cast<double>(n));
        const Eigen::Vector2d pt = from_map + dir * d;
        if (!is_map_point_feasible(cost_map, direction_map, pt)) break;
        last_feasible = pt;
    }
    return last_feasible;
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

Eigen::Vector2d PathPlanner::predict_start_map(
    const PlanRequest& req,
    const CostMap& cost_map,
    const DirectionMap& direction_map
) const {
    const Eigen::Vector2d current_map = req.current_pos_map;
    if (!config_.start_prediction_enable) return current_map;

    const double speed = req.current_velocity;
    if (speed < std::max(0.0, config_.start_prediction_min_speed)) return current_map;

    const Eigen::Vector2d heading_dir(std::cos(req.current_yaw), std::sin(req.current_yaw));

    const double brake_distance = speed * speed / (2.0 * config_.start_prediction_max_accel);
    const double delay_distance = speed * std::max(0.0, config_.start_prediction_planning_delay);
    const double total_distance = brake_distance + delay_distance;

    const Eigen::Vector2d predicted = current_map + heading_dir * total_distance;
    return adjust_reachable_start_on_segment(
        cost_map, direction_map, current_map, predicted
    );
}

// ═══════════════════════ 规划主流程 ═══════════════════════

PlanResult PathPlanner::plan(const PlanRequest& req) const {
    PlanResult result;
    result.goal_id = req.goal.id;
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

    Eigen::Vector2d start_map = predict_start_map(
        req, planning_cost_map, *req.direction_map
    );
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
    const double dist = (goal_plan - start_map).norm();

    const auto plan_start = std::chrono::steady_clock::now();

    // ── [1] Dijkstra cost-to-goal（map 系工作，供 kinodynamic h + 开阔 seed）──
    DijkstraCostToGoal dijkstra;
    dijkstra.build(planning_cost_map, goal_grid.cast<int>(), config_.dijkstra);
    if (!dijkstra.ready() || std::isinf(dijkstra.at_map(start_map))) {
        return fail("Dijkstra: goal unreachable from start");
    }
    const auto dijkstra_done = std::chrono::steady_clock::now();

    // 原语逐子步可行性：碰撞 + 方向地形的通行方向、对齐和入口速度窗。
    const CostMap* const planning_map_ptr = &planning_cost_map;
    const auto transition_feasible = [
        planning_map_ptr,
        direction_map = req.direction_map.get(),
        &terrain = req.terrain_constraints,
        &config = config_
    ](const KinodynamicAstar::State& from, const KinodynamicAstar::State& to) {
        const Eigen::Vector2d map_pt(to.x, to.y);
        const Eigen::Vector2d g = planning_map_ptr->map_coord_to_grid(map_pt);
        if (!planning_map_ptr->is_valid_coord(g)) return false;
        if (planning_map_ptr->interpolate(g) >= config.occupied_threshold) return false;

        const Eigen::Vector2d dg = direction_map->map_coord_to_grid(map_pt);
        if (!direction_map->is_valid_coord(dg)) return false;
        if (!direction_map->is_terrain_body_at(dg)) return true;
        const Eigen::Array2i cell_array = dg.array().floor().cast<int>();
        const Eigen::Vector2i cell(cell_array.x(), cell_array.y());
        const uint8_t label = direction_map->terrain_at(cell);
        const Eigen::Vector2d raw_dir = direction_map->at(cell);
        if (label < static_cast<uint8_t>(TerrainType::SLOPE)
            || raw_dir.squaredNorm() <= 1e-12) return false;

        const Eigen::Vector2d displacement(to.x - from.x, to.y - from.y);
        if (displacement.norm() < 1e-6) return false;
        const Eigen::Vector2d dir = raw_dir.normalized();
        const double travel_alignment = displacement.normalized().dot(dir);
        if (std::abs(travel_alignment) <= config.step_detection.detect_dot_threshold) return false;
        const TraversalMode* rule = terrain.selected_mode(label, travel_alignment >= 0.0);
        if (!rule) return false;

        // 台阶模式速度窗定义为车身正向速度；倒车只用于台阶外的拓扑机动。
        return to.v >= rule->velocity_window.min - 1e-6
            && to.v <= rule->velocity_window.max + 1e-6;
    };

    // 位置容差内仍需证明可由 MINCO 在一条短、无方向地形本体、无碰撞连接段上精确接到目标。
    // 该 predicate 防止目标在台阶另一侧时，搜索在台阶前提前成功。
    const auto goal_reached = [
        planning_map_ptr,
        direction_map = req.direction_map.get(),
        goal_plan,
        &config = config_
    ](const KinodynamicAstar::State& state) {
        const Eigen::Vector2d start(state.x, state.y);
        const double distance = (goal_plan - start).norm();
        if (distance > config.kinodynamic.goal_tolerance) return false;
        const int samples = std::max(1, static_cast<int>(std::ceil(distance / 0.03)));
        for (int i = 1; i <= samples; ++i) {
            const double fraction = static_cast<double>(i) / static_cast<double>(samples);
            const Eigen::Vector2d point = start + fraction * (goal_plan - start);
            const Eigen::Vector2d cost_grid = planning_map_ptr->map_coord_to_grid(point);
            if (!planning_map_ptr->is_valid_coord(cost_grid)
                || planning_map_ptr->interpolate(cost_grid) >= config.occupied_threshold) {
                return false;
            }
            const Eigen::Vector2d direction_grid = direction_map->map_coord_to_grid(point);
            if (!direction_map->is_valid_coord(direction_grid)) return false;
            if (direction_map->is_terrain_body_at(direction_grid)) {
                return false;
            }
        }
        return true;
    };

    // ── [2] Kinodynamic A*：全状态机动 seed（含倒车/绕行/原地转拓扑）──
    KinodynamicAstar::State kino_start;
    kino_start.x = start_map.x();
    kino_start.y = start_map.y();
    kino_start.theta = req.current_yaw;
    kino_start.v = req.current_velocity;

    KinodynamicAstar::Params kinodynamic_params = config_.kinodynamic;
    MincoOptimizer::Params minco_params = config_.minco;

    std::vector<KinodynamicAstar::State> seed_states_raw;
    bool direct_segment_avoids_terrain_body = true;
    if (dist > 1e-9) {
        const int samples = std::max(2, static_cast<int>(std::ceil(dist / 0.03)) + 1);
        for (int i = 0; i <= samples; ++i) {
            const Eigen::Vector2d p = start_map
                + (goal_plan - start_map) * (static_cast<double>(i) / static_cast<double>(samples));
            const Eigen::Vector2d dg = req.direction_map->map_coord_to_grid(p);
            if (req.direction_map->is_valid_coord(dg)
                && req.direction_map->is_terrain_body_at(dg)) {
                direct_segment_avoids_terrain_body = false;
                break;
            }
        }
    }
    if (dist < config_.skip_distance && direct_segment_avoids_terrain_body) {
        // 近距离直连：起点 + 终点两状态，MINCO 直接优化。
        KinodynamicAstar::State goal_state;
        goal_state.x = goal_plan.x();
        goal_state.y = goal_plan.y();
        goal_state.theta = req.current_yaw;
        goal_state.v = 0.0;
        seed_states_raw = {kino_start, goal_state};
    } else {
        KinodynamicAstar astar(kinodynamic_params);
        const auto kino = astar.search(kino_start, dijkstra, transition_feasible, goal_reached);
        if (!kino.success) return fail("Kinodynamic A* failed: " + kino.error);
        seed_states_raw = kino.states;
    }
    const auto kinodynamic_done = std::chrono::steady_clock::now();

    // ── 种子重采样为 MINCO 边界全状态 + 段时长 ──
    // A* predicate 已证明容差内末节点存在无地形本体的短连接；显式追加精确目标，不改写搜索状态。
    if (std::hypot(seed_states_raw.back().x - goal_plan.x(), seed_states_raw.back().y - goal_plan.y()) > 1e-9) {
        KinodynamicAstar::State exact_goal = seed_states_raw.back();
        exact_goal.x = goal_plan.x();
        exact_goal.y = goal_plan.y();
        exact_goal.v = 0.0;
        seed_states_raw.push_back(exact_goal);
    } else {
        seed_states_raw.back().v = 0.0;
    }
    const auto minco_seed = build_minco_seed(
        seed_states_raw,
        config_.seed_resample_distance,
        kinodynamic_params.primitive_duration / std::max(kinodynamic_params.collision_substeps, 1)
    );
    if (minco_seed.durations.empty()) return fail("MINCO seed construction produced no segments");
    // ── [3] MINCO 时空优化 ──
    MincoOptimizer optimizer(minco_params);
    const auto minco_start = std::chrono::steady_clock::now();
    const auto opt = optimizer.optimize(
        minco_seed.states,
        minco_seed.durations,
        minco_seed.gears,
        minco_seed.cusp_waypoints,
        planning_cost_map,
        *req.direction_map,
        req.terrain_constraints
    );
    if (!opt.success) return fail("MINCO optimization failed: " + opt.error);
    const auto minco_done = std::chrono::steady_clock::now();

    if (config_.enable_debug && opt.diagnostics_valid) {
        const auto& s = opt.seed_costs;
        const auto& f = opt.final_costs;
        static constexpr const char* LBFGS_STATUS[] = {"CONVERGED", "MAX_ITER", "LS_FAILED"};
        const int status_index = std::clamp(opt.lbfgs_status, 0, 2);
        RCLCPP_DEBUG(
            logger_,
            "MINCO diag: status=%s iters=%d |grad|_inf %.3g -> %.3g (pos=%.3g, time=%.3g) | "
            "cost %.3g -> %.3g | waypoints free=%d disp(sum=%.3f m, max=%.3f m) | "
            "seed[E=%.3g T=%.3g O=%.3g V=%.3g Lat=%.3g W=%.3g A=%.3g SA=%.3g SV=%.3g SP=%.3g RA=%.3g RW=%.3g] "
            "final[E=%.3g T=%.3g O=%.3g V=%.3g Lat=%.3g W=%.3g A=%.3g SA=%.3g SV=%.3g SP=%.3g RA=%.3g RW=%.3g]",
            LBFGS_STATUS[status_index], opt.iterations,
            opt.initial_grad_inf_norm, opt.final_grad_inf_norm,
            opt.final_grad_pos_inf_norm, opt.final_grad_time_inf_norm,
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
    if (!validate_trajectory(
            opt.trajectory,
            planning_cost_map,
            *req.direction_map,
            req.terrain_constraints,
            config_.occupied_threshold,
            config_.step_detection.detect_dot_threshold,
            minco_params.trajectory_limits,
            config_.trajectory_validation,
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
    std::vector<StepTraversalConstraint> step_constraints;
    step_constraints.reserve(path->step_segments.size());
    for (const auto& segment : path->step_segments) {
        step_constraints.push_back(segment.traversal_constraint);
    }
    path->step_constraint_schedule = std::make_shared<const StepConstraintSchedule>(std::move(step_constraints));

    // 台阶掩码层：针对本条轨迹产出。
    const auto layers = step_routing_mask_->compute(path->trajectory);
    path->step_cost_layer = layers.step_cost_layer;
    path->masked_direction_map = layers.masked_direction_map;

    result.kind = PlanResult::Kind::PATH;

    if (config_.enable_debug) {
        std::vector<Eigen::Vector2d> seed_pts;
        seed_pts.reserve(seed_states_raw.size());
        for (const auto& s : seed_states_raw) seed_pts.emplace_back(s.x, s.y);
        result.debug_rough_path = std::move(seed_pts);
    }

    result.path = std::move(path);

    double seed_path_length = 0.0;
    for (size_t i = 1; i < seed_states_raw.size(); ++i) {
        seed_path_length += std::hypot(
            seed_states_raw[i].x - seed_states_raw[i - 1].x,
            seed_states_raw[i].y - seed_states_raw[i - 1].y
        );
    }
    const int segment_count = opt.trajectory.segment_count();
    const int variable_count = 2 * std::max(segment_count - 1, 0) + segment_count; // 平坦：2D 路点 + 段时长
    int cusp_count = 0;
    for (const char c : minco_seed.cusp_waypoints) cusp_count += c ? 1 : 0;
    RCLCPP_INFO(
        logger_, "Plan done (%.2f ms): Src(%.2f,%.2f)->Dst(%.2f,%.2f) %s, %zu step segments; "
        "Dijkstra=%.2f ms, Kino=%.2f ms, MINCO=%.2f ms, seed_length=%.2f m, raw_states=%zu, "
        "segments=%d, vars=%d, cusps=%d, iters=%d, |grad|=%.3g",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - plan_start).count(),
        start_map.x(), start_map.y(), goal_plan.x(), goal_plan.y(),
        fixed ? "[FIXED]" : "", result.path->step_segments.size(),
        std::chrono::duration<double, std::milli>(dijkstra_done - plan_start).count(),
        std::chrono::duration<double, std::milli>(kinodynamic_done - dijkstra_done).count(),
        std::chrono::duration<double, std::milli>(minco_done - minco_start).count(),
        seed_path_length, seed_states_raw.size(), segment_count, variable_count, cusp_count,
        opt.iterations, opt.final_grad_inf_norm
    );

    return result;
}

} // namespace nav_executor
