#pragma once

#include <Eigen/Core>
#include <limits>
#include <optional>
#include <path_follower/path/spline_path.hpp>
#include <path_follower/common/nav_map.hpp>

namespace path_follower {
struct RecoveryParams;
}

namespace path_follower::recovery_helpers {

struct FieldSample {
    double cost = 0.0;
    double step_norm = 0.0;
};

struct PathScore {
    double score = std::numeric_limits<double>::infinity();
    bool end_safe = false;
    FieldSample end_sample;
};

std::optional<FieldSample> sample_fields(
    const CostMap& cost_map,
    const DirectionMap& dir_map,
    const Eigen::Vector2d& pos
);

bool is_safe_goal(const RecoveryParams& p, const FieldSample& s);

std::optional<PathScore> score_candidate_by_path_integral(
    const RecoveryParams& p,
    const CostMap& cost_map,
    const DirectionMap& dir_map,
    const Eigen::Vector2d& origin,
    const Eigen::Vector2d& goal,
    double radius,
    const DirectionMap* base_dir_map = nullptr
);

std::optional<Eigen::Vector2d> find_goal(
    const RecoveryParams& p,
    const CostMap& cost_map,
    const DirectionMap& dir_map,
    const Eigen::Vector3d& chassis_pose,
    const DirectionMap* base_dir_map = nullptr
);

Eigen::Vector2d rotate_vector(const Eigen::Vector2d& v, double angle);

std::optional<double> max_cost_along_segment(
    const CostMap& cost_map,
    const Eigen::Vector2d& a_map,
    const Eigen::Vector2d& b_map,
    int samples
);

} // namespace path_follower::recovery_helpers
