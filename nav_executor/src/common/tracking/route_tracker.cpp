#include <nav_executor/common/tracking/route_tracker.hpp>

#include <algorithm>
#include <cmath>

namespace nav_executor {

namespace {

double wrap_angle(const double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

} // anonymous namespace

RouteTracker::RouteTracker(RouteTrackerParams params) : params_(params) {}

void RouteTracker::reset() {
    path_.reset();
    hypotheses_.clear();
    reported_arc_length_ = 0.0;
    last_stamp_.reset();
}

std::optional<RouteEstimate> RouteTracker::update(
    AnnotatedPath::ConstPtr path,
    const Eigen::Vector3d& chassis_pose_map,
    const double chassis_velocity,
    const std::chrono::steady_clock::time_point stamp
) {
    if (!path || path->trajectory.empty()) {
        reset();
        return std::nullopt;
    }
    if (path != path_) {
        path_ = std::move(path);
        hypotheses_.clear();
        reported_arc_length_ = 0.0;
        last_stamp_.reset();
    }

    const MincoTrajectory& geometry = path_->trajectory;
    const double total_length = geometry.total_arc_length();
    const Eigen::Vector2d position = chassis_pose_map.head<2>();
    const double heading = chassis_pose_map.z();
    const Eigen::Vector2d heading_axis(std::cos(heading), std::sin(heading));
    const Eigen::Vector2d velocity_map = chassis_velocity * heading_axis;
    const double measured_path_speed = std::max(0.0, chassis_velocity);
    const double spacing = params_.hypothesis_spacing;

    // 观测代价：位置、航向和速度方向共同判定有向分支。空间相邻但切向相反的
    // 回头弯分支在航向与速度项上代价迥异，因此不会被误选。
    const auto observation_cost = [&](const double arc_length, const double path_speed) {
        const TrajSample sample = geometry.eval_arc_length(arc_length);
        const Eigen::Vector2d tangent(std::cos(sample.theta), std::sin(sample.theta));
        const double position_residual = (position - sample.p).norm() / params_.position_scale;
        const double heading_residual =
            wrap_angle(heading - sample.theta) / params_.heading_scale;
        const double velocity_residual =
            (velocity_map - path_speed * tangent).norm() / params_.velocity_scale;
        return position_residual * position_residual
            + heading_residual * heading_residual
            + velocity_residual * velocity_residual;
    };

    // 在 [lo, hi] 上按 hypothesis_spacing 枚举有向进度候选，端点必取。
    const auto for_each_candidate = [&](const double lo, const double hi, auto&& visit) {
        visit(lo);
        const int first = static_cast<int>(std::floor(lo / spacing)) + 1;
        const int last = static_cast<int>(std::ceil(hi / spacing)) - 1;
        for (int index = first; index <= last; ++index) {
            visit(static_cast<double>(index) * spacing);
        }
        if (hi > lo) visit(hi);
    };

    std::vector<Hypothesis> candidates;
    if (hypotheses_.empty()) {
        // 新路径只在起点附近建立初始进度，覆盖规划延迟但禁止跳到后段。
        const double initial_hi = std::min(total_length, params_.initial_search_distance);
        for_each_candidate(0.0, initial_hi, [&](const double arc_length) {
            candidates.push_back({
                .arc_length = arc_length,
                .path_speed = measured_path_speed,
                .cost = observation_cost(arc_length, measured_path_speed),
            });
        });
    } else {
        const double dt = last_stamp_
            ? std::clamp(
                std::chrono::duration<double>(stamp - *last_stamp_).count(),
                0.0, params_.prediction_time_limit
            )
            : 0.0;
        for (const Hypothesis& previous : hypotheses_) {
            const double path_speed = std::lerp(
                previous.path_speed, measured_path_speed, params_.path_speed_filter_alpha
            );
            // 路径域运动模型：s_k = s_{k-1} + Δt·(ν_{k-1}+ν_k)/2。
            const double advance = 0.5 * (previous.path_speed + path_speed) * dt;
            const double predicted = previous.arc_length + advance;
            // 进度允许观测噪声，但不允许回退：候选下界同时不低于已上报进度，
            // 因此分支切换也无法让对外可见的弧长后退。
            const double lo = std::max(previous.arc_length, reported_arc_length_);
            const double hi = std::min(total_length, predicted + advance + spacing);
            for_each_candidate(lo, std::max(lo, hi), [&](const double arc_length) {
                const double transition = (arc_length - predicted) / params_.transition_scale;
                candidates.push_back({
                    .arc_length = arc_length,
                    .path_speed = path_speed,
                    .cost = previous.cost + transition * transition
                        + observation_cost(arc_length, path_speed),
                });
            });
        }
    }
    if (candidates.empty()) {
        reset();
        return std::nullopt;
    }

    // 按代价保留领先的竞争分支；弧长过近的候选只留最优者，避免假设集退化为同一分支。
    std::sort(
        candidates.begin(), candidates.end(),
        [](const Hypothesis& a, const Hypothesis& b) { return a.cost < b.cost; }
    );
    const double best_cost = candidates.front().cost;
    const double cost_ceiling = best_cost
        + params_.hypothesis_prune_ratio * std::max(std::abs(best_cost), 1.0);
    hypotheses_.clear();
    for (const Hypothesis& candidate : candidates) {
        if (static_cast<int>(hypotheses_.size()) >= params_.max_hypotheses) break;
        if (candidate.cost > cost_ceiling) break;
        const bool duplicate = std::any_of(
            hypotheses_.begin(), hypotheses_.end(),
            [&](const Hypothesis& kept) {
                return std::abs(kept.arc_length - candidate.arc_length) < spacing;
            }
        );
        if (duplicate) continue;
        hypotheses_.push_back(candidate);
    }
    // 代价按帧归一，防止长时间跟随后累计量溢出并保持假设间的相对可比性。
    for (Hypothesis& hypothesis : hypotheses_) hypothesis.cost -= best_cost;

    const Hypothesis& best = hypotheses_.front();
    const TrajSample reference = geometry.eval_arc_length(best.arc_length);
    const double tracking_error = (reference.p - position).norm();
    reported_arc_length_ = best.arc_length;
    last_stamp_ = stamp;
    return RouteEstimate {
        .path = path_,
        .status = tracking_error <= params_.max_tracking_error
            ? RouteTrackingStatus::TRACKED
            : RouteTrackingStatus::LOST,
        .arc_length = best.arc_length,
        .path_speed = best.path_speed,
        .remaining_length = std::max(0.0, total_length - best.arc_length),
        .reference_position = reference.p,
        .tracking_error = tracking_error,
    };
}

} // namespace nav_executor
