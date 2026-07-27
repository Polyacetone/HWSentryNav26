#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/trajectory/minco_trajectory.hpp>
#include <nav_executor/common/trajectory/trajectory_limits.hpp>

namespace nav_executor {

struct GeometryCertificateParams {
    // 自适应区间细分的最大深度；达到深度仍无法给出证书即判定违规。
    int max_subdivision_depth;

    // 有向 SE(2) 可分辨性：弧长相距 arc_separation 以上的两点若同时位置接近且航向接近，
    // 则仅凭车身位姿观测无法区分，路径在规划期即被拒绝。
    double observability_arc_separation;
    double observability_position_distance;
    double observability_heading_angle;
    double observability_sample_spacing;
};

// 独立几何证书。通过即保证：曲线有向正则（无内部零速点、无逆向、航向连续）、
// 真实曲率与曲率变化率在包络内、且不存在有向不可观测的近重合分支。
// 判据使用 Bezier 凸包上的保守区间界配合自适应细分，而不是固定采样点，
// 因此固定网格之间的窄尖峰也无法通过。
struct GeometryCertificate {
    bool valid = false;
    std::string rejection; // valid=false 时记录首个失败项及定位信息
};

// seed_tangents 为 A* 有向种子在各段边界处的单位切向，长度须为 segment_count()+1。
[[nodiscard]] GeometryCertificate certify_trajectory_geometry(
    const MincoTrajectory& trajectory,
    const std::vector<Eigen::Vector2d>& seed_tangents,
    const TrajectoryLimits& limits,
    const GeometryCertificateParams& params
);

} // namespace nav_executor
