#pragma once

#include <limits>
#include <optional>
#include <uniform_bspline/uniform_bspline.hpp>
#include <Eigen/Core>
#include <path_follower/nav_map.hpp>

namespace path_follower {
template <typename ValueType>
using SplineT = ubs::UniformBSpline<ValueType, 2, ValueType, Eigen::Matrix<ValueType, 2, 1>, std::vector<Eigen::Matrix<ValueType, 2, 1>>>;
using SplineD = ubs::UniformBSpline<double, 2, double, Eigen::Vector2d, std::vector<Eigen::Vector2d>>;
}

namespace path_follower{
inline constexpr double PATH_U_EXTRAP_MIN = -0.35;
inline constexpr double PATH_U_EXTRAP_MAX = 1.35;

double clamp_path_u_extrapolated(
    double u,
    double u_min = PATH_U_EXTRAP_MIN,
    double u_max = PATH_U_EXTRAP_MAX
);

double project_to_spline_u(
    const SplineD& spline,
    const Eigen::Vector2d& pos,
    double u_hint,
    int num_samples,
    double search_window,
    double local_search_lazy_distance
);

double project_to_spline_u_extrapolated(
    const SplineD& spline,
    const Eigen::Vector2d& pos,
    double u_hint,
    int num_samples,
    double search_window,
    double local_search_lazy_distance,
    double u_min = PATH_U_EXTRAP_MIN,
    double u_max = PATH_U_EXTRAP_MAX
);
}

namespace path_follower {
struct RecoveryParams;
struct NavigationParams;
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

std::optional<double> score_runup_path_integral(
    const NavigationParams& params,
    const CostMap& cost_map,
    const DirectionMap& dir_map,
    const Eigen::Vector2d& origin,
    const Eigen::Vector2d& goal
);

std::optional<double> max_cost_along_segment(
    const CostMap& cost_map,
    const Eigen::Vector2d& a_map,
    const Eigen::Vector2d& b_map,
    int samples
);

} // namespace path_follower::recovery_helpers