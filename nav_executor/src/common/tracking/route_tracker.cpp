#include <nav_executor/common/tracking/route_tracker.hpp>

#include <algorithm>
#include <cmath>

namespace nav_executor {

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
    const Eigen::Vector2d velocity_map = chassis_velocity
        * Eigen::Vector2d(std::cos(heading), std::sin(heading));
    const double spacing = params_.hypothesis_spacing;

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

    const double dt = last_stamp_
        ? std::clamp(
            std::chrono::duration<double>(stamp - *last_stamp_).count(),
            0.0, params_.prediction_time_limit
        )
        : 0.0;
    const bool initializing = hypotheses_.empty();
    const std::vector<Hypothesis> initial_state {{
        .arc_length = 0.0,
        .path_speed = 0.0,
        .cost = 0.0,
    }};
    const std::vector<Hypothesis>& previous_states = initializing
        ? initial_state
        : hypotheses_;

    const double velocity_weight = 1.0 / std::pow(params_.velocity_sigma, 2);
    const double progress_weight = 1.0 / std::pow(params_.progress_sigma, 2);
    const bool has_speed_profile = !path_->speed_profile.empty();
    const double profile_weight = has_speed_profile
        ? 1.0 / std::pow(params_.profile_speed_sigma, 2)
        : 0.0;
    const double dynamics_weight = 1.0 / std::pow(params_.speed_dynamics_sigma, 2);

    std::vector<Hypothesis> candidates;
    for (const Hypothesis& previous : previous_states) {
        const double nominal_advance = previous.path_speed * dt;
        const double lo = initializing ? 0.0 : reported_arc_length_;
        const double hi = initializing
            ? std::min(total_length, params_.initial_search_distance)
            : std::min(
                total_length,
                previous.arc_length + 2.0 * nominal_advance + spacing
            );
        for_each_candidate(lo, std::max(lo, hi), [&](const double arc_length) {
            const TrajSample sample = geometry.eval_arc_length(arc_length);
            const Eigen::Vector2d tangent(std::cos(sample.theta), std::sin(sample.theta));
            const double profile_speed = !has_speed_profile
                ? previous.path_speed
                : path_->speed_profile.eval_arc_length(arc_length).velocity;

            // 固定 s 后，统一后验关于 nu 是一维二次函数，可直接求其受限最小值。
            const double half_dt = 0.5 * dt;
            const double progress_offset = arc_length - previous.arc_length
                - half_dt * previous.path_speed;
            const double denominator = velocity_weight + profile_weight + dynamics_weight
                + progress_weight * half_dt * half_dt;
            const double numerator = velocity_weight * tangent.dot(velocity_map)
                + profile_weight * profile_speed
                + dynamics_weight * previous.path_speed
                + progress_weight * half_dt * progress_offset;
            const double path_speed = std::clamp(
                numerator / denominator, 0.0, params_.max_path_speed
            );
            const double predicted = previous.arc_length
                + half_dt * (previous.path_speed + path_speed);
            const double position_residual =
                (position - sample.p).norm() / params_.position_sigma;
            const double velocity_residual =
                (velocity_map - path_speed * tangent).norm() / params_.velocity_sigma;
            const double progress_residual =
                (arc_length - predicted) / params_.progress_sigma;
            const double profile_residual = has_speed_profile
                ? (path_speed - profile_speed) / params_.profile_speed_sigma
                : 0.0;
            const double dynamics_residual =
                (path_speed - previous.path_speed) / params_.speed_dynamics_sigma;
            candidates.push_back({
                .arc_length = arc_length,
                .path_speed = path_speed,
                .cost = previous.cost
                    + position_residual * position_residual
                    + velocity_residual * velocity_residual
                    + progress_residual * progress_residual
                    + profile_residual * profile_residual
                    + dynamics_residual * dynamics_residual,
            });
        });
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
