#pragma once

#include <chrono>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/trajectory/annotated_path.hpp>

namespace nav_executor {

struct RouteTrackerParams {
    // 新路径只允许在起点附近建立初始进度 (m)。
    double initial_search_distance;
    // 与有向参考点的距离超过该值视为跟踪丢失 (m)。
    double max_tracking_error;
    // 状态掉帧后最多积分这么久的路径速度 (s)。
    double prediction_time_limit;

    // 候选进度假设的弧长间距 (m)；决定多假设网格的分辨率。
    double hypothesis_spacing;
    // 保留的竞争假设数量上限。
    int max_hypotheses;
    // 假设权重低于领先假设该比例时被淘汰。
    double hypothesis_prune_ratio;

    // 统一 (s, nu) 状态估计器的残差标准差。
    double position_sigma;
    double velocity_sigma;
    double progress_sigma;
    double profile_speed_sigma;
    double speed_dynamics_sigma;
    // 实际路径速度状态的物理上界 (m/s)。
    double max_path_speed;
};

enum class RouteTrackingStatus : uint8_t {
    TRACKED = 0,
    LOST = 1,
};

struct RouteEstimate {
    AnnotatedPath::ConstPtr path;
    RouteTrackingStatus status = RouteTrackingStatus::LOST;
    double arc_length = 0.0;
    double path_speed = 0.0;
    double remaining_length = 0.0;
    Eigen::Vector2d reference_position = Eigen::Vector2d::Zero();
    double tracking_error = 0.0;
};

// 有向路径进度观测器。
//
// 进度是有向路径上的时序状态，不是每帧独立的最近点：
//   - 位置与地图系速度矢量共同观测 (s, nu)，速度剖面仅作为 nu 的弱先验；
//   - 维护多个竞争的弧长假设，错误分支随后续观测被淘汰，不会因单次错误最近点永久锁死；
//   - 前进跟随时进度单调不减，杜绝沿错误分支回退。
class RouteTracker {
public:
    explicit RouteTracker(RouteTrackerParams params);

    std::optional<RouteEstimate> update(
        AnnotatedPath::ConstPtr path,
        const Eigen::Vector3d& chassis_pose_map,
        double chassis_velocity,
        std::chrono::steady_clock::time_point stamp
    );

    void reset();

private:
    // 一个有向进度假设：弧长、路径方向速度，以及累计代价（越小越可信）。
    struct Hypothesis {
        double arc_length = 0.0;
        double path_speed = 0.0;
        double cost = 0.0;
    };

    RouteTrackerParams params_;
    AnnotatedPath::ConstPtr path_;
    std::vector<Hypothesis> hypotheses_;
    // 已上报进度的下界。分支切换不得让对外可见的进度回退。
    double reported_arc_length_ = 0.0;
    std::optional<std::chrono::steady_clock::time_point> last_stamp_;
};

} // namespace nav_executor
