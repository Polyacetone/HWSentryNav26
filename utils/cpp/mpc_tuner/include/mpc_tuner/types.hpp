#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/common/annotated_path.hpp>
#include <nav_executor/path_executor/step_controller.hpp>
#include <nav_executor/path_executor/solver/mpc_types.hpp>
#include <nav_executor/path_planner/bspline_optimizer.hpp>
#include <nav_executor/path_planner/path_planner.hpp>
#include <nav_executor/path_planner/step_routing_mask.hpp>

#include <mpc_tuner/parameter_space.hpp>

namespace mpc_tuner {

enum class ExecutedFeature : size_t {
    LATERAL_TUBE,
    HEADING,
    PROGRESS,
    COMMAND_V,
    COMMAND_OMEGA,
    COMMAND_DV,
    COMMAND_DOMEGA,
    ACC_LIMIT,
    ALPHA_LIMIT,
    LAT_ACCEL,
    OBSTACLE,
    STEP_DIRECTION,
    STEP_VELOCITY,
    STEP_REACHABILITY_LO,
    STEP_REACHABILITY_HI,
    STEP_OMEGA,
    STEP_DV,
    STEP_DOMEGA,
    TERMINAL_BRAKE,
    TERMINAL_PROGRESS,
    TERMINAL_LATERAL,
    COUNT,
};

constexpr size_t EXECUTED_FEATURE_COUNT = static_cast<size_t>(ExecutedFeature::COUNT);

constexpr std::array<std::string_view, EXECUTED_FEATURE_COUNT> EXECUTED_FEATURE_NAMES {
    "lateral_tube", "heading", "progress", "command_v", "command_omega", "command_dv", "command_domega",
    "acc_limit", "alpha_limit", "lat_accel", "obstacle", "step_direction", "step_velocity",
    "step_reachability_lo", "step_reachability_hi", "step_omega", "step_dv", "step_domega",
    "terminal_brake", "terminal_progress", "terminal_lateral",
};

// 闭环实际执行轨迹上的场景覆盖统计，不依赖 MPCSolver 的内部预测或诊断接口。
struct ExecutedFeatureCoverage {
    std::array<double, EXECUTED_FEATURE_COUNT> raw_sum {};
    std::array<uint64_t, EXECUTED_FEATURE_COUNT> active_samples {};
    std::array<uint64_t, EXECUTED_FEATURE_COUNT> sample_count {};
};

struct ScenarioSpec {
    std::string name;
    std::string split;
    Eigen::Vector3d start_pose = Eigen::Vector3d::Zero();
    Eigen::Vector2d goal = Eigen::Vector2d::Zero();
    std::vector<uint64_t> seeds;
};

struct MapSnapshot {
    int width = 0;
    int height = 0;
    double resolution = 0.0;
    double origin_x = 0.0;
    double origin_y = 0.0;
    std::vector<uint8_t> global_cost_data;
    std::vector<uint8_t> direction_image_data;
};

struct StoredRoute {
    ScenarioSpec spec;
    std::vector<Eigen::Vector2d> spline_control_points;
    std::vector<nav_executor::StepPlanSegment> step_segments;
    std::vector<uint8_t> control_cost_data;
};

struct SceneBundle {
    static constexpr uint32_t FORMAT_VERSION = 2;
    static constexpr uint32_t LEGACY_FORMAT_VERSION = 1;

    uint32_t format_version = FORMAT_VERSION;
    std::string split;
    std::string created_at;
    MapSnapshot map;
    std::vector<StoredRoute> routes;
};

struct CompiledScenario {
    ScenarioSpec spec;
    nav_executor::AnnotatedPath::ConstPtr path;
    nav_executor::CostMap::ConstPtr global_cost_map;
    nav_executor::CostMap::ConstPtr control_cost_map;
    nav_executor::DirectionMap::ConstPtr direction_map;
};

struct StudyConfig {
    uint64_t seed = 2026;
    int population_size = 64;
    double elite_fraction = 0.2;
    int generations = 10;
    double initial_std = 0.45;
    double min_std = 0.03;
    int parallel_workers = 0;
    double progress_interval_seconds = 10.0;
    double regularization_lambda = 0.1; // 归一化空间向基线锚定的软正则强度
};

// 软标量各项的特征尺度：把每个原始物理指标除以其“多大算很多”的尺度，得到 O(1) 量，
// 再乘以下方 SoftWeights。尺度是物理/经验问题，权重是价值取舍问题，两者刻意分离。
struct SoftScales {
    double reference_speed = 2.0;          // 参考通行速度 (m/s)，用于每路线时间归一 t_ref = 弧长 / v_ref
    double arrival_speed_band = 0.2;       // 超出目标到达速度部分的归一化尺度 (m/s)，越小惩罚越强
    double cross_track = 0.3;              // 横向偏离归一尺度 (m)，取管廊半宽 y_tube
    double high_cost_integral = 1.0;       // 经过代价积分软膝盖尺度
    double accel_floor = 1.8;              // 平滑度地板：线加速度阈值 (m/s²)，取 acc_max
    double alpha_floor = 7.0;              // 平滑度地板：角加速度阈值 (rad/s²)，取 alpha_max
};

// 软标量各项相对偏好权重。时间为唯一的主导项 (1.0)，其余单项 ≤ 0.3，
// 保证“没有任何单个质量项能盖过一次有意义的时间改善”。
struct SoftWeights {
    double time = 1.0;
    double high_cost = 0.30;
    double arrival_speed = 0.25;
    double step_speed_minor = 0.20;
    double step_heading_minor = 0.20;
    double cross_track = 0.15;
    double smoothness = 0.10;
};

struct EpisodeConfig {
    double timeout = 20.0;
    double goal_radius = 0.5;
    double target_arrival_speed = 0.1;
    double high_cost_threshold = 30.0;
    double lethal_cost_threshold = 250.0;
    double severe_step_heading_error = 0.5235987755982988; // 30° in rad
    double severe_step_speed_margin = 0.2;
    double forward_progress_epsilon = 1e-4; // 判定“前进”的 Δu 死区，用于 run-up 段违规门控
    SoftScales soft_scales;
    SoftWeights soft_weights;
};

struct TunerConfig {
    StudyConfig study;
    EpisodeConfig episode;
    std::array<SearchRange, PARAMETER_COUNT> search_ranges;
};

struct RuntimeConfig {
    struct AStarConfig {
        double step_alignment_weight = 0.0;
        double obstacle_weight = 0.0;
        double step_proximity_weight = 0.0;
        double step_mode_dot_threshold = 0.0;
        int downsampled_waypoint_max_interval = 0;
        int feasible_threshold = 0;
    };

    nav_executor::MPCParams mpc;
    nav_executor::TerrainProfiles terrain_profiles;
    nav_executor::ProfileBlendParams profile_blend;
    nav_executor::PlannerConfig planner;
    nav_executor::BSplineOptimizer::Params path_optimizer;
    nav_executor::StepRoutingMaskParams step_mask;
    AStarConfig a_star;
    double step_dist_offset = 0.1;
};

struct ParameterCandidate {
    std::array<double, PARAMETER_COUNT> normalized {};
    std::array<double, PARAMETER_COUNT> values {};
};

// 单条 episode 的原始度量。硬 / 软的语义划分在 fitness 层完成；此处只如实记录物理量。
// 台阶违规区分“严重”(硬门槛) 与“轻微”(软标量)，run-up 段仅在前进时累积（见 episode_runner）。
struct EpisodeMetrics {
    std::string scenario_name;
    uint64_t seed = 0;
    bool reached = false;
    bool solver_failed = false;
    double elapsed_time = 0.0;
    double final_progress = 0.0;
    double path_length = 0.0; // 路线弧长 (m)，用于每路线时间归一

    double arrival_speed = 0.0;
    double arrival_omega = 0.0;

    // ── 硬门槛伴随量 ──
    double hazard_duration = 0.0;             // 处于致命代价的停留时长 (s)
    int severe_step_speed_events = 0;         // 台阶速度严重违规次数（含跨越瞬间超限）
    double severe_step_speed_excess = 0.0;    // 对应超限积分/幅值，作字典序伴随量
    int severe_step_heading_events = 0;       // 台阶角严重违规次数（含跨越瞬间超限）
    double severe_step_heading_excess = 0.0;  // 对应超限积分/幅值

    // ── 软标量原始量 ──
    double high_cost_integral = 0.0;          // 高代价区（未致命）停留积分
    double max_cost = 0.0;
    double minor_step_speed_violation = 0.0;  // 台阶速度轻微违规积分（前进段，未达严重阈值部分）
    double minor_step_heading_violation = 0.0;// 台阶角轻微违规积分（前进段）
    double cross_track_squared_integral = 0.0;
    double smoothness_excess_integral = 0.0;  // 超出物理加速度地板的抖动积分

    ExecutedFeatureCoverage feature_coverage;
};

// 两层适应度：
//   第一层（硬门槛）——字典序元组，安全绝对优先，各类事件计数 + 连续伴随量，无需权重。
//     层级顺序（按用户约定，越靠前越优先）：
//       1) hazard            —— 致命障碍/卡死停留，绝对优先
//       2) 到达目标          —— failed_scenarios + progress_deficit（未到达时越接近完成越好）
//       3) 台阶区域安全      —— 严重速度 + 严重航向合并计数（角度+速度同层）
//   第二层（软标量）——单一连续加权代价，时间主导，仅对安全且到达的候选做平滑排序。
struct Fitness {
    // 第一层：硬门槛
    int hazard_events = 0;
    double hazard_duration = 0.0;
    int severe_step_speed_events = 0;
    double severe_step_speed_excess = 0.0;
    int severe_step_heading_events = 0;
    double severe_step_heading_excess = 0.0;
    int failed_scenarios = 0;
    double progress_deficit = 0.0;

    // 第二层：软标量（已含向基线的正则项）
    double soft_cost = std::numeric_limits<double>::infinity();

    [[nodiscard]] auto ordering_key() const {
        // 排名层级（字典序，越靠前越优先）：
        //   1. hazard            —— 致命障碍/卡死停留，安全绝对优先
        //   2. 到达目标          —— 未到达次数 + 未到达时进度缺口
        //   3. 台阶区域安全      —— 严重速度 + 严重航向违规合并计数（角度+速度同层）
        //   4. 软代价            —— 仅用于安全且到达的候选间平滑排序
        const int step_zone_violations = severe_step_speed_events + severe_step_heading_events;
        const double step_zone_excess = severe_step_speed_excess + severe_step_heading_excess;
        return std::tuple {
            hazard_events, hazard_duration,
            failed_scenarios, progress_deficit,
            step_zone_violations, step_zone_excess,
            soft_cost,
        };
    }

    bool operator<(const Fitness& other) const { return ordering_key() < other.ordering_key(); }
};

} // namespace mpc_tuner
