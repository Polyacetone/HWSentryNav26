#include <nav_executor/planner/path_planner.hpp>

#include <algorithm>
#include <chrono>
#include <queue>

#include <rclcpp/logging.hpp>

namespace nav_executor {

namespace {

std::vector<Eigen::Vector2d> make_direct_init_path(
    const Eigen::Vector2d& start_grid,
    const Eigen::Vector2d& goal_grid,
    int width,
    int height
) {
    const Eigen::Vector2d delta = goal_grid - start_grid;
    const double dist = delta.norm();

    if (dist <= 1e-6) {
        Eigen::Vector2d axis = Eigen::Vector2d::UnitX();
        if (width <= 1 && height > 1) axis = Eigen::Vector2d::UnitY();
        Eigen::Vector2d mid = start_grid + 0.5 * axis;
        mid.x() = std::clamp(mid.x(), 0.0, static_cast<double>(std::max(0, width - 1)));
        mid.y() = std::clamp(mid.y(), 0.0, static_cast<double>(std::max(0, height - 1)));
        return {start_grid, mid, goal_grid};
    }

    const int point_count = std::max(3, static_cast<int>(std::ceil(dist)) + 1);
    std::vector<Eigen::Vector2d> path;
    path.reserve(static_cast<size_t>(point_count));
    for (int i = 0; i < point_count; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(point_count - 1);
        path.push_back((1.0 - t) * start_grid + t * goal_grid);
    }
    return path;
}

} // anonymous namespace

PathPlanner::PathPlanner(
    const PlannerConfig& config,
    std::shared_ptr<const AStarPlanner> a_star,
    std::shared_ptr<const BSplineOptimizer> optimizer,
    std::shared_ptr<StepRoutingMask> step_routing_mask,
    rclcpp::Logger logger
) : config_(config),
    a_star_(std::move(a_star)),
    optimizer_(std::move(optimizer)),
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

bool PathPlanner::is_map_point_feasible(const CostMap& cost_map, const DirectionMap& direction_map, const Eigen::Vector2d& map_pt) const {
    const Eigen::Vector2d cost_grid = cost_map.map_coord_to_grid(map_pt);
    const Eigen::Vector2d dir_grid = direction_map.map_coord_to_grid(map_pt);
    if (!cost_map.is_valid_coord(cost_grid) || !direction_map.is_valid_coord(dir_grid)) return false;

    return cost_map.interpolate(cost_grid) < config_.occupied_threshold
        && direction_map.interpolate(dir_grid).norm() < config_.on_step_threshold
        && !direction_map.has_blocked_corner(dir_grid);
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
        if (!is_map_point_feasible(*req.merged_cost_map, *req.direction_map, pt)) break;
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

    if (merged.is_valid_coord(grid_pt) && is_map_point_feasible(merged, dir, map_pt)) {
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
        if (is_map_point_feasible(merged, dir, candidate_map)) {
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

    if (!req.global_cost_map || !req.merged_cost_map || !req.direction_map) {
        RCLCPP_WARN(logger_, "Plan aborted: map snapshot incomplete");
        result.kind = PlanResult::Kind::FAILED;
        return result;
    }

    const Eigen::Vector2d goal_map = req.goal.position_map;
    const bool fixed = req.goal.fixed;

    // ── 近距离短路判断（§6.4）：robot-to-goal ──
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
        if (!is_map_point_feasible(*req.global_cost_map, *req.direction_map, start_map)) return fail("Start is not feasible on global map");
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
        if (!is_map_point_feasible(*req.global_cost_map, *req.direction_map, goal_map)) return fail("Goal is not feasible on global map");
    }

    // ── 终点 merged 检查 / nudge（取决于 fixed）──
    if (fixed) {
        if (!is_map_point_feasible(*req.merged_cost_map, *req.direction_map, goal_map)) return fail("Fixed goal is occupied by a dynamic obstacle");
    } else {
        const auto nudged = nudge_point_to_free(req, goal_map, config_.nudge_max_distance);
        if (!nudged) return fail("Cannot nudge goal to a free cell");
        goal_plan = *nudged;
    }

    const CostMap& planning_cost_map = *req.merged_cost_map;
    const Eigen::Vector2d start_grid = planning_cost_map.map_coord_to_grid(start_map);
    const Eigen::Vector2d goal_grid = planning_cost_map.map_coord_to_grid(goal_plan);
    const double dist = (goal_plan - start_map).norm();

    const auto plan_start = std::chrono::steady_clock::now();

    // ── 粗路径：近距离直连初值，否则 A* ──
    std::vector<Eigen::Vector2d> rough_path;
    if (dist < config_.skip_distance) {
        rough_path = make_direct_init_path(start_grid, goal_grid, planning_cost_map.width, planning_cost_map.height);
    } else {
        const auto plan_search = a_star_->search_path(
            planning_cost_map, *req.direction_map, start_grid.cast<int>(), goal_grid.cast<int>()
        );
        if (!plan_search) return fail("A* search failed: " + plan_search.error());
        rough_path.reserve(plan_search->size());
        for (const auto& pt : *plan_search) rough_path.push_back(pt.cast<double>());
    }

    // ── 样条优化 ──
    const auto optimize_result = optimizer_->optimize(
        planning_cost_map, *req.direction_map, rough_path, start_grid, goal_grid
    );
    if (!optimize_result) return fail("Path optimization failed: " + optimize_result.error());

    const auto& [control_points_grid, warmup_path_grid, optimized_path_grid] = *optimize_result;
    if (control_points_grid.size() < 3) return fail("Optimizer returned insufficient control points");

    const auto grid_to_map = [&](const std::vector<Eigen::Vector2d>& pts) {
        std::vector<Eigen::Vector2d> out;
        out.reserve(pts.size());
        for (const auto& p : pts) out.push_back(planning_cost_map.grid_coord_to_map(p));
        return out;
    };

    std::vector<Eigen::Vector2d> control_points_map = grid_to_map(control_points_grid);

    // ── 构建不可变 AnnotatedPath ──
    auto path = std::make_shared<AnnotatedPath>(SplinePath(control_points_map));
    path->goal_pos = goal_map;
    path->goal_fixed = fixed;
    path->goal_id = req.goal.id;

    // 台阶几何标注：基于 base（未掩码）方向场（§16）。
    path->step_segments = step_annotator::build_step_plan(
        config_.step_detection, path->spline, *req.direction_map, logger_
    );

    // 台阶掩码层：针对本条样条产出（Q3）。
    const auto layers = step_routing_mask_->compute(path->spline);
    path->step_cost_layer = layers.step_cost_layer;
    path->masked_direction_map = layers.masked_direction_map;

    result.kind = PlanResult::Kind::PATH;
    result.path = std::move(path);

    if (config_.enable_debug) {
        result.debug_rough_path = grid_to_map(rough_path);
        result.debug_warmup_path = grid_to_map(warmup_path_grid);
        result.debug_optimized_path = grid_to_map(optimized_path_grid);
    }

    RCLCPP_INFO(
        logger_, "Plan done (%.2f ms): Src(%.2f,%.2f)->Dst(%.2f,%.2f) %s, %zu step segments",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - plan_start).count(),
        start_map.x(), start_map.y(), goal_plan.x(), goal_plan.y(),
        fixed ? "[FIXED]" : "", result.path->step_segments.size()
    );

    return result;
}

} // namespace nav_executor
