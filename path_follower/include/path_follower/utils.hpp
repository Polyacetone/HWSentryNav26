#pragma once

#include <Eigen/Core>
#include <limits>
#include <optional>
#include <vector>
#include <uniform_bspline/uniform_bspline.hpp>
#include <path_follower/nav_map.hpp>

namespace path_follower {
template <typename ValueType>
using SplineT = ubs::UniformBSpline<ValueType, 2, ValueType, Eigen::Matrix<ValueType, 2, 1>, std::vector<Eigen::Matrix<ValueType, 2, 1>>>;
using SplineD = ubs::UniformBSpline<double, 2, double, Eigen::Vector2d, std::vector<Eigen::Vector2d>>;
}

namespace path_follower{
inline constexpr double PATH_U_EXTRAP_MIN = -1.0;
inline constexpr double PATH_U_EXTRAP_MAX = 1.5;

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

void eval_quadratic_bspline2(
    const std::vector<Eigen::Vector2d>& cps,
    double u,
    Eigen::Vector2d* p,
    Eigen::Vector2d* d1,
    Eigen::Vector2d* d2
);

void eval_quadratic_bspline2_extrapolated(
    const std::vector<Eigen::Vector2d>& cps,
    double u,
    Eigen::Vector2d* p,
    Eigen::Vector2d* d1,
    Eigen::Vector2d* d2
);

double quadratic_bspline_curvature(const Eigen::Vector2d& d1, const Eigen::Vector2d& d2);
}

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
