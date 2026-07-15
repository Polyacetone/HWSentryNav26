#include <nav_executor/path_planner/path_planner.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <queue>

#include <rclcpp/logging.hpp>

namespace nav_executor {

namespace {

double wrap_angle(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

// kinodynamic 状态序列 → MINCO 边界全状态 + 段时长。
struct MincoSeed {
    std::vector<MincoMinJerk::BoundaryPVA> states;
    std::vector<double> durations;
    std::vector<MincoOptimizer::HardWaypoint> hard_waypoints;
    std::vector<MincoOptimizer::StepEntrySpeedWindow> step_entry_speed_windows;
};

// 把 kinodynamic (x,y,θ,v) 序列重采样后转成 MINCO min-jerk 边界状态。
//   - 位置速度 = (v cosθ, v sinθ)；朝向速度 θ̇ 由相邻状态差分估计；加速度置零（MINCO 会调整）。
//   - 距离、显著转向和换向点都会保留，避免把倒车尖点或原地旋转重采样掉。
//   - 段时长沿用 A* 原语时间，避免用低速处的弧长/速度比制造异常长分段。
MincoSeed build_minco_seed(
    const std::vector<KinodynamicAstar::State>& raw,
    const double resample_distance,
    const double state_interval,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const double step_norm_threshold
) {
    MincoSeed seed;
    if (raw.size() < 2) return seed;

    std::vector<bool> is_step_entry(raw.size(), false);
    for (size_t i = 1; i < raw.size(); ++i) {
        const Eigen::Vector2d prev_grid = direction_map.map_coord_to_grid({raw[i - 1].x, raw[i - 1].y});
        const Eigen::Vector2d grid = direction_map.map_coord_to_grid({raw[i].x, raw[i].y});
        if (!direction_map.is_valid_coord(grid)) continue;
        const uint8_t label = direction_map.terrain_at(grid);
        const Eigen::Vector2d dir = direction_map.interpolate(grid);
        const bool on_step = label >= static_cast<uint8_t>(TerrainType::SLOPE)
            && dir.norm() >= step_norm_threshold;
        const bool prev_on_same_step = direction_map.is_valid_coord(prev_grid)
            && direction_map.terrain_at(prev_grid) == label
            && direction_map.interpolate(prev_grid).norm() >= step_norm_threshold;
        is_step_entry[i] = on_step && !prev_on_same_step;
    }

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
        if (distance >= distance_threshold || heading_change >= HEADING_THRESHOLD || gear_change
            || is_step_entry[i]) {
            selected.push_back(i);
        }
    }
    selected.push_back(raw.size() - 1);

    const size_t n = selected.size();
    seed.states.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const auto& s = raw[selected[i]];
        auto& bs = seed.states[i];
        bs.pos << s.x, s.y, s.theta;
        bs.vel << s.v * std::cos(s.theta), s.v * std::sin(s.theta), 0.0;
        bs.acc.setZero();
    }
    seed.durations.resize(n - 1);
    for (size_t i = 0; i + 1 < n; ++i) {
        seed.durations[i] = std::max(
            state_interval * static_cast<double>(selected[i + 1] - selected[i]),
            0.1
        );
    }
    // θ̇ 估计填入边界速度第三分量。
    for (size_t i = 0; i < n; ++i) {
        double theta_rate = 0.0;
        if (i == 0 && n >= 2) {
            theta_rate = wrap_angle(
                raw[selected[1]].theta - raw[selected[0]].theta
            ) / seed.durations[0];
        } else if (i + 1 == n) {
            theta_rate = wrap_angle(
                raw[selected[i]].theta - raw[selected[i - 1]].theta
            ) / seed.durations[i - 1];
        } else {
            const double dth = wrap_angle(
                raw[selected[i + 1]].theta - raw[selected[i - 1]].theta
            );
            theta_rate = dth / (seed.durations[i - 1] + seed.durations[i]);
        }
        seed.states[i].vel(2) = theta_rate;
    }

    // 规划终点只约束位置；执行到达前应有可停止的参考。终端 θ 在优化器中是自由变量。
    seed.states.back().vel.setZero();
    seed.states.back().acc.setZero();

    for (size_t i = 1; i + 1 < n; ++i) {
        const size_t raw_index = selected[i];
        if (!is_step_entry[raw_index]) continue;
        const auto& state = raw[raw_index];
        const Eigen::Vector2d grid = direction_map.map_coord_to_grid({state.x, state.y});
        const uint8_t label = direction_map.terrain_at(grid);
        const Eigen::Vector2d dir = direction_map.interpolate(grid).normalized();
        const Eigen::Vector2d displacement(
            state.x - raw[selected[i - 1]].x,
            state.y - raw[selected[i - 1]].y
        );
        const Eigen::Vector2d heading(std::cos(state.theta), std::sin(state.theta));
        const bool positive_direction = (displacement.norm() > 1e-6 ? displacement : heading).dot(dir) >= 0.0;
        const Eigen::Vector2d traversal_direction = positive_direction ? dir : -dir;
        const TerrainStepRule* rule = terrain_constraints.selected_mode(label, positive_direction);
        if (!rule) continue;
        seed.hard_waypoints.push_back(MincoOptimizer::HardWaypoint{
            .waypoint_index = static_cast<int>(i - 1),
            .position = {state.x, state.y},
            .theta = std::atan2(traversal_direction.y(), traversal_direction.x()),
        });
        seed.step_entry_speed_windows.push_back(MincoOptimizer::StepEntrySpeedWindow{
            .waypoint_index = static_cast<int>(i - 1),
            .speed_min = rule->speed.min,
            .speed_max = rule->speed.max,
        });
    }
    return seed;
}

bool validate_trajectory(
    const MincoTrajectory& trajectory,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const int occupied_threshold,
    const double step_norm_threshold,
    const double step_alignment_threshold,
    const MincoOptimizer::Limits& limits,
    const double nonholonomic_tolerance,
    const double step_entry_window_fraction,
    const PlannerConfig::TrajectoryValidationParams& validation,
    const std::vector<MincoOptimizer::StepEntrySpeedWindow>& step_entry_speed_windows,
    std::string& error
) {
    const double total_time = trajectory.total_time();
    if (!std::isfinite(total_time) || total_time <= 0.0) {
        error = "trajectory has invalid total time";
        return false;
    }
    const int samples_per_segment = std::max(validation.samples_per_segment, 1);
    for (int segment = 0; segment < trajectory.segment_count(); ++segment) {
        const double tau_begin = trajectory.segment_boundary_tau(segment);
        const double tau_end = trajectory.segment_boundary_tau(segment + 1);
        for (int sample = 0; sample <= samples_per_segment; ++sample) {
            const double fraction = static_cast<double>(sample) / static_cast<double>(samples_per_segment);
            const double tau = std::lerp(tau_begin, tau_end, fraction);
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
        const Eigen::Vector2d velocity = s.dp_dtau / total_time;
        const Eigen::Vector2d acceleration = s.ddp_dtau / (total_time * total_time);
        const double longitudinal_velocity = trajectory.longitudinal_velocity(s);
        const double omega = trajectory.angular_velocity(s);
        const double nonholonomic_violation = velocity.x() * std::sin(s.theta) - velocity.y() * std::cos(s.theta);
        if (std::abs(nonholonomic_violation) > nonholonomic_tolerance) {
            error = "trajectory violates nonholonomic constraint at tau=" + std::to_string(tau)
                + ": |dx/dt*sin(theta)-dy/dt*cos(theta)|=" + std::to_string(std::abs(nonholonomic_violation))
                + " > tolerance=" + std::to_string(nonholonomic_tolerance);
            return false;
        }
        if (longitudinal_velocity < limits.vel_min - validation.velocity_tolerance
            || longitudinal_velocity > limits.vel_max + validation.velocity_tolerance) {
            error = "trajectory violates velocity limit at tau=" + std::to_string(tau)
                + ": v=" + std::to_string(longitudinal_velocity)
                + " (allowed=[" + std::to_string(limits.vel_min)
                + "," + std::to_string(limits.vel_max)
                + "], tolerance=" + std::to_string(validation.velocity_tolerance) + ")";
            return false;
        }
        if (std::abs(omega) > limits.omega_max + validation.omega_tolerance) {
            error = "trajectory violates angular velocity limit at tau=" + std::to_string(tau)
                + ": |omega|=" + std::to_string(std::abs(omega))
                + " > omega_max=" + std::to_string(limits.omega_max)
                + " + tolerance=" + std::to_string(validation.omega_tolerance)
                + " = " + std::to_string(limits.omega_max + validation.omega_tolerance);
            return false;
        }
        if (acceleration.norm() > limits.acc_max + validation.acceleration_tolerance) {
            error = "trajectory violates acceleration limit at tau=" + std::to_string(tau)
                + ": |acc|=" + std::to_string(acceleration.norm())
                + " > acc_max=" + std::to_string(limits.acc_max)
                + " + tolerance=" + std::to_string(validation.acceleration_tolerance)
                + " = " + std::to_string(limits.acc_max + validation.acceleration_tolerance);
            return false;
        }
        if (std::abs(longitudinal_velocity * omega)
            > limits.a_lat_max + validation.lateral_acceleration_tolerance) {
            error = "trajectory violates lateral acceleration limit at tau=" + std::to_string(tau)
                + ": |v*omega|=" + std::to_string(std::abs(longitudinal_velocity * omega))
                + " > a_lat_max=" + std::to_string(limits.a_lat_max)
                + " + tolerance=" + std::to_string(validation.lateral_acceleration_tolerance)
                + " = " + std::to_string(limits.a_lat_max + validation.lateral_acceleration_tolerance);
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
        const uint8_t label = direction_map.terrain_at(dir_grid);
        const Eigen::Vector2d raw_dir = direction_map.interpolate(dir_grid);
        if (label >= static_cast<uint8_t>(TerrainType::SLOPE)
            && raw_dir.norm() >= step_norm_threshold) {
            const Eigen::Vector2d dir = raw_dir.normalized();
            const Eigen::Vector2d heading(std::cos(s.theta), std::sin(s.theta));
            const double alignment = heading.dot(dir);
            if (std::abs(alignment) < step_alignment_threshold) {
                error = "trajectory is not aligned with directional terrain at tau=" + std::to_string(tau)
                    + ": |heading.dot(dir)|=" + std::to_string(std::abs(alignment))
                    + " < threshold=" + std::to_string(step_alignment_threshold);
                return false;
            }
            const bool going_up = alignment >= 0.0;
            const TerrainStepRule* rule = terrain_constraints.selected_mode(label, going_up);
            if (!rule) {
                error = "trajectory uses prohibited directional terrain at tau=" + std::to_string(tau)
                    + ": label=" + std::to_string(static_cast<int>(label))
                    + ", direction=" + (going_up ? "up" : "down")
                    + ", |dir|=" + std::to_string(raw_dir.norm());
                return false;
            }
            if (longitudinal_velocity < rule->speed.min - validation.step_velocity_tolerance
                || longitudinal_velocity > rule->speed.max + validation.step_velocity_tolerance) {
                error = "trajectory violates step velocity window at tau=" + std::to_string(tau)
                    + ": v=" + std::to_string(longitudinal_velocity) + ", required=["
                    + std::to_string(rule->speed.min) + "," + std::to_string(rule->speed.max)
                    + "], accepted=[" + std::to_string(rule->speed.min - validation.step_velocity_tolerance)
                    + "," + std::to_string(rule->speed.max + validation.step_velocity_tolerance) + "]";
                return false;
            }
        }

        for (const auto& entry : step_entry_speed_windows) {
            const bool in_incoming_window = segment == entry.waypoint_index
                && fraction >= 1.0 - step_entry_window_fraction;
            const bool in_outgoing_window = segment == entry.waypoint_index + 1
                && fraction <= step_entry_window_fraction;
            if (!in_incoming_window && !in_outgoing_window) continue;
            if (longitudinal_velocity < entry.speed_min - validation.velocity_tolerance
                || longitudinal_velocity > entry.speed_max + validation.velocity_tolerance) {
                error = "trajectory violates step-entry velocity window at tau=" + std::to_string(tau)
                    + " (waypoint " + std::to_string(entry.waypoint_index) + ")"
                    + ": v=" + std::to_string(longitudinal_velocity)
                    + " (required=[" + std::to_string(entry.speed_min)
                    + "," + std::to_string(entry.speed_max)
                    + "], tolerance=" + std::to_string(validation.velocity_tolerance) + ")";
                return false;
            }
        }
        }
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

bool PathPlanner::is_map_point_feasible(const CostMap& cost_map, const DirectionMap& direction_map, const TerrainTraversalConstraints& terrain_constraints, const Eigen::Vector2d& map_pt) const {
    const Eigen::Vector2d cost_grid = cost_map.map_coord_to_grid(map_pt);
    const Eigen::Vector2d dir_grid = direction_map.map_coord_to_grid(map_pt);
    if (!cost_map.is_valid_coord(cost_grid) || !direction_map.is_valid_coord(dir_grid)) return false;

    return cost_map.interpolate(cost_grid) < config_.occupied_threshold
        && direction_map.interpolate(dir_grid).norm() < config_.on_step_threshold
        && !terrain_constraints.has_blocked_corner(direction_map, dir_grid);
}

Eigen::Vector2d PathPlanner::adjust_reachable_start_on_segment(const PlanRequest& req, const Eigen::Vector2d& from_map, const Eigen::Vector2d& to_map) const {
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
        if (!is_map_point_feasible(*req.merged_cost_map, *req.direction_map, req.terrain_constraints, pt)) break;
        last_feasible = pt;
    }
    return last_feasible;
}

std::optional<Eigen::Vector2d> PathPlanner::nudge_point_to_free(const PlanRequest& req, const Eigen::Vector2d& map_pt, const double max_nudge_distance) const {
    const CostMap& merged = *req.merged_cost_map;
    const DirectionMap& dir = *req.direction_map;
    const Eigen::Vector2d grid_pt = merged.map_coord_to_grid(map_pt);
    const int width = merged.width;
    const int height = merged.height;

    const auto key = [width](const int x, const int y) {
        return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
    };

    if (merged.is_valid_coord(grid_pt) && is_map_point_feasible(merged, dir, req.terrain_constraints, map_pt)) {
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

        const double dist = (current.cast<double>() - grid_pt).norm() * merged.resolution;
        if (dist > max_nudge_distance) continue;

        const Eigen::Vector2d candidate_map = merged.grid_coord_to_map(current.cast<double>());
        if (is_map_point_feasible(merged, dir, req.terrain_constraints, candidate_map)) {
            return candidate_map;
        }

        for (int i = 0; i < 8; i++) {
            const int nx = current.x() + dx[i];
            const int ny = current.y() + dy[i];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            const double ndist = (Eigen::Vector2d(nx, ny) - grid_pt).norm() * merged.resolution;
            if (ndist > max_nudge_distance) continue;
            const size_t nk = key(nx, ny);
            if (visited[nk]) continue;
            visited[nk] = 1;
            q.push(Eigen::Vector2i(nx, ny));
        }
    }

    return std::nullopt;
}

Eigen::Vector2d PathPlanner::predict_start_map(const PlanRequest& req) const {
    const Eigen::Vector2d current_map = req.current_pos_map;
    if (!config_.start_prediction_enable) return current_map;

    const double speed = req.current_velocity;
    if (speed < std::max(0.0, config_.start_prediction_min_speed)) return current_map;

    const Eigen::Vector2d heading_dir(std::cos(req.current_yaw), std::sin(req.current_yaw));

    const double brake_distance = speed * speed / (2.0 * config_.start_prediction_max_accel);
    const double delay_distance = speed * std::max(0.0, config_.start_prediction_planning_delay);
    const double total_distance = brake_distance + delay_distance;

    const Eigen::Vector2d predicted = current_map + heading_dir * total_distance;
    return adjust_reachable_start_on_segment(req, current_map, predicted);
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

    // ── 近距离短路判断：robot-to-goal ──
    if ((req.current_pos_map - goal_map).norm() < config_.goal_reached_distance) {
        result.kind = fixed ? PlanResult::Kind::USE_AS_FIXED_GOAL : PlanResult::Kind::COMPLETE_NO_PLAN_NEEDED;
        RCLCPP_INFO(
            logger_, "Goal within reached distance (%.2f m): %s",
            config_.goal_reached_distance, fixed ? "USE_AS_FIXED_GOAL" : "COMPLETE_NO_PLAN_NEEDED"
        );
        return result;
    }

    const auto fail = [&](const std::string& msg) {
        RCLCPP_ERROR(logger_, "Plan failed: %s", msg.c_str());
        result.kind = PlanResult::Kind::FAILED;
        return result;
    };

    Eigen::Vector2d start_map = predict_start_map(req);
    Eigen::Vector2d goal_plan = goal_map;

    // ── 起点 global 严格检查 ──
    {
        const Eigen::Vector2d sg = req.global_cost_map->map_coord_to_grid(start_map);
        if (!req.global_cost_map->is_valid_coord(sg)) return fail("Start is out of bound");
        if (!is_map_point_feasible(*req.global_cost_map, *req.direction_map, req.terrain_constraints, start_map)) return fail("Start is not feasible on global map");
    }

    // ── 起点 nudge（merged 上最近 free）──
    {
        const auto nudged = nudge_point_to_free(req, start_map, config_.nudge_max_distance);
        if (!nudged) return fail("Cannot nudge start to a free cell");
        start_map = *nudged;
    }

    // ── 终点 global 严格检查 ──
    {
        const Eigen::Vector2d gg = req.global_cost_map->map_coord_to_grid(goal_map);
        if (!req.global_cost_map->is_valid_coord(gg)) return fail("Goal is out of bound");
        if (!is_map_point_feasible(*req.global_cost_map, *req.direction_map, req.terrain_constraints, goal_map)) return fail("Goal is not feasible on global map");
    }

    // ── 终点 merged 检查 / nudge（取决于 fixed）──
    if (fixed) {
        if (!is_map_point_feasible(*req.merged_cost_map, *req.direction_map, req.terrain_constraints, goal_map)) return fail("Fixed goal is occupied by a dynamic obstacle");
    } else {
        const auto nudged = nudge_point_to_free(req, goal_map, config_.nudge_max_distance);
        if (!nudged) return fail("Cannot nudge goal to a free cell");
        goal_plan = *nudged;
    }

    const CostMap planning_cost_map = req.merged_cost_map->merge(*req.terrain_constraints.blocked_cost_layer);
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
        const uint8_t label = direction_map->terrain_at(dg);
        const Eigen::Vector2d raw_dir = direction_map->interpolate(dg);
        if (label < static_cast<uint8_t>(TerrainType::SLOPE)
            || raw_dir.norm() < config.step_detection.detect_norm_threshold) {
            return true;
        }

        const Eigen::Vector2d displacement(to.x - from.x, to.y - from.y);
        if (displacement.norm() < 1e-6) return false;
        const Eigen::Vector2d dir = raw_dir.normalized();
        const double travel_alignment = displacement.normalized().dot(dir);
        if (std::abs(travel_alignment) < config.step_detection.detect_dot_threshold) return false;
        const TerrainStepRule* rule = terrain.selected_mode(label, travel_alignment >= 0.0);
        if (!rule) return false;

        // 台阶模式速度窗定义为车身正向速度；倒车只用于台阶外的拓扑机动。
        return to.v >= rule->speed.min - 1e-6 && to.v <= rule->speed.max + 1e-6;
    };

    // 位置容差内仍需证明可由 MINCO 在一条短、平坦、无碰撞连接段上精确接到目标。
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
            if (direction_map->terrain_at(direction_grid) >= static_cast<uint8_t>(TerrainType::SLOPE)
                && direction_map->interpolate(direction_grid).norm()
                    >= config.step_detection.detect_norm_threshold) {
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

    // 本次规划速度上限至少覆盖所有已选地形模式的最低入口速度。选择最低可行速度，
    // 避免生成超过对应 capability profile 必要范围的参考。
    double required_vel_max = config_.kinodynamic.vel_max;
    std::array<bool, TERRAIN_LABEL_COUNT> terrain_present {};
    for (const uint8_t label : req.direction_map->terrain) {
        if (label < TERRAIN_LABEL_COUNT) terrain_present[label] = true;
    }
    for (uint8_t label = static_cast<uint8_t>(TerrainType::SLOPE); label < TERRAIN_LABEL_COUNT; ++label) {
        if (!terrain_present[label]) continue;
        for (const bool is_up : {false, true}) {
            if (const TerrainStepRule* rule = req.terrain_constraints.selected_mode(label, is_up)) {
                required_vel_max = std::max(required_vel_max, rule->speed.min);
            }
        }
    }
    KinodynamicAstar::Params kinodynamic_params = config_.kinodynamic;
    kinodynamic_params.vel_max = required_vel_max;
    MincoOptimizer::Params minco_params = config_.minco;

    std::vector<KinodynamicAstar::State> seed_states_raw;
    bool direct_segment_is_flat = true;
    if (dist > 1e-9) {
        const int samples = std::max(2, static_cast<int>(std::ceil(dist / 0.03)) + 1);
        for (int i = 0; i <= samples; ++i) {
            const Eigen::Vector2d p = start_map
                + (goal_plan - start_map) * (static_cast<double>(i) / static_cast<double>(samples));
            const Eigen::Vector2d dg = req.direction_map->map_coord_to_grid(p);
            if (req.direction_map->is_valid_coord(dg)
                && req.direction_map->terrain_at(dg) >= static_cast<uint8_t>(TerrainType::SLOPE)
                && req.direction_map->interpolate(dg).norm() >= config_.step_detection.detect_norm_threshold) {
                direct_segment_is_flat = false;
                break;
            }
        }
    }
    if (dist < config_.skip_distance && direct_segment_is_flat) {
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
    // A* predicate 已证明容差内末节点存在平坦短连接；显式追加精确目标，不改写搜索状态。
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
        kinodynamic_params.primitive_duration / std::max(kinodynamic_params.collision_substeps, 1),
        *req.direction_map,
        req.terrain_constraints,
        config_.step_detection.detect_norm_threshold
    );
    if (minco_seed.durations.empty()) return fail("MINCO seed construction produced no segments");
    for (const auto& entry : minco_seed.step_entry_speed_windows) {
        minco_params.limits.vel_max = std::max(minco_params.limits.vel_max, entry.speed_max);
    }

    // ── [3] MINCO 时空优化 ──
    MincoOptimizer optimizer(minco_params);
    const auto minco_start = std::chrono::steady_clock::now();
    const auto opt = optimizer.optimize(
        minco_seed.states,
        minco_seed.durations,
        planning_cost_map,
        *req.direction_map,
        req.terrain_constraints,
        minco_seed.hard_waypoints,
        minco_seed.step_entry_speed_windows
    );
    if (!opt.success) return fail("MINCO optimization failed: " + opt.error);
    const auto minco_done = std::chrono::steady_clock::now();
    if (opt.trajectory.empty()) return fail("MINCO produced empty trajectory");
    if (opt.max_nonholonomic_violation > minco_params.nonholonomic_tolerance) {
        RCLCPP_WARN(
            logger_,
            "MINCO accepted with nonholonomic residual %.4f m/s (target %.4f, acceptance %.4f)",
            opt.max_nonholonomic_violation,
            minco_params.nonholonomic_tolerance,
            minco_params.nonholonomic_acceptance_tolerance
        );
    }
    std::string trajectory_error;
    if (!validate_trajectory(
            opt.trajectory,
            planning_cost_map,
            *req.direction_map,
            req.terrain_constraints,
            config_.occupied_threshold,
            config_.step_detection.detect_norm_threshold,
            config_.step_detection.detect_dot_threshold,
            minco_params.limits,
            minco_params.nonholonomic_acceptance_tolerance,
            minco_params.step_entry_window_fraction,
            config_.trajectory_validation,
            minco_seed.step_entry_speed_windows,
            trajectory_error
        )) {
        return fail("MINCO output rejected: " + trajectory_error);
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
        std::vector<Eigen::Vector2d> traj_pts;
        constexpr int TRAJ_SAMPLES = 100;
        for (int i = 0; i <= TRAJ_SAMPLES; ++i) {
            traj_pts.push_back(path->trajectory.position(static_cast<double>(i) / TRAJ_SAMPLES));
        }
        result.global_path = std::move(traj_pts);
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
    const int variable_count = 3 * std::max(segment_count - 1, 0) + segment_count + 1;
    const double dense_workspace_kib = static_cast<double>(
        8LL * (3LL * 6LL * segment_count * 6LL * segment_count + 12LL * segment_count)
    ) / 1024.0;
    RCLCPP_INFO(
        logger_, "Plan done (%.2f ms): Src(%.2f,%.2f)->Dst(%.2f,%.2f) %s, %zu step segments; "
        "Dijkstra=%.2f ms, Kino=%.2f ms, MINCO=%.2f ms, seed_length=%.2f m, raw_states=%zu, "
        "segments=%d, vars=%d, dense_workspace_est=%.1f KiB, AL=%d, rho=%.1f, nonholo=%.4f, |grad|=%.3g",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - plan_start).count(),
        start_map.x(), start_map.y(), goal_plan.x(), goal_plan.y(),
        fixed ? "[FIXED]" : "", result.path->step_segments.size(),
        std::chrono::duration<double, std::milli>(dijkstra_done - plan_start).count(),
        std::chrono::duration<double, std::milli>(kinodynamic_done - dijkstra_done).count(),
        std::chrono::duration<double, std::milli>(minco_done - minco_start).count(),
        seed_path_length, seed_states_raw.size(), segment_count, variable_count, dense_workspace_kib,
        opt.al_rounds, opt.final_rho, opt.max_nonholonomic_violation, opt.final_grad_inf_norm
    );

    return result;
}

} // namespace nav_executor
