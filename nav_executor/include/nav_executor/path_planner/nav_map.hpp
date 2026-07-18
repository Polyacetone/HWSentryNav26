#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

namespace nav_executor {

class CostMap;
class DirectionMap;

// ════════════════════════════════════════════════════════════════
//  地形标签
// ════════════════════════════════════════════════════════════════

enum class TerrainType : uint8_t {
    FLAT = 0,
    OBSTACLE = 1,
    SLOPE = 2,
    STEP_L1 = 3,
    STEP_L2 = 4,
    FLY_SLOPE = 5,
    STEP_HIGH = 6,
};
constexpr size_t TERRAIN_LABEL_COUNT = 7;

// ════════════════════════════════════════════════════════════════
//  地形跨越模式与规划期约束
// ════════════════════════════════════════════════════════════════

struct TerrainRule {
    bool forward_allowed = true;
    bool backward_allowed = true;
};
using TerrainRuleTable = std::array<TerrainRule, TERRAIN_LABEL_COUNT>;

// ════════════════════════════════════════════════════════════════
//  能力档位 / 地形 profile（供 executor 台阶模式 + MPC 约束）
// ════════════════════════════════════════════════════════════════

struct SignedVelocityBounds {
    double min;
    double max;
};

struct SignedAngularVelocityBounds {
    double min;
    double max;
};

// 下位机对控制指令实施的硬限幅。它约束 command，而不是规划轨迹或预测状态。
struct CommandEnvelope {
    SignedVelocityBounds velocity;
    SignedAngularVelocityBounds angular_velocity;
};

// 控制器内部用于惩罚指令变化和侧向加速度的动态尺度，不是 box constraint。
struct CommandDynamicsLimits {
    double velocity_rate_max;
    double angular_velocity_rate_max;
    double lateral_acceleration_max;
};

struct CapabilityProfile {
    CommandEnvelope command_envelope;
    CommandDynamicsLimits command_dynamics;
};

enum class CapabilityLevel : uint8_t {
    LOW = 0,
    MEDIUM = 1,
    HIGH = 2,
};

inline CapabilityLevel capability_level_from_string(const std::string& s) {
    if (s == "low") return CapabilityLevel::LOW;
    if (s == "medium") return CapabilityLevel::MEDIUM;
    if (s == "high") return CapabilityLevel::HIGH;
    throw std::invalid_argument("Unknown capability level: \"" + s + "\" (expected low/medium/high)");
}

// 台阶穿越时对 v_act/v_pred 的共享速度窗口。A* 将其作为可行性条件，MINCO/MPCC
// 将其作为软目标；该类型本身不隐含 enforcement 方式。
struct TraversalVelocityWindow {
    double min = 0.0;
    double max = 0.0;
};

struct TraversalMode {
    std::string name;
    uint8_t chassis_mode = 0;
    CapabilityLevel capability = CapabilityLevel::LOW;
    TraversalVelocityWindow velocity_window;
    bool requires_high_performance = false;
    double run_up = 0.0; // 助跑提前量 (m)：约束锚点自物理台阶边缘上游回退该距离（冲量动作填 0）
};

struct DirectionalTraversalModes {
    std::vector<TraversalMode> up;
    std::vector<TraversalMode> down;
};

struct SelectedTerrainModes {
    std::optional<TraversalMode> up;
    std::optional<TraversalMode> down;
};

struct TraversalConfiguration {
    std::array<CapabilityProfile, 3> capability_profiles;
    std::array<DirectionalTraversalModes, 5> directional_labels;
    double high_performance_buffercap_threshold = 0.0;
    double high_performance_supercap_threshold = 0.0;
    double high_performance_rfr_pwr_limit_threshold = 0.0;
};

struct PerformanceState {
    bool high_performance = false;
};

struct TerrainTraversalConstraints {
    TerrainRuleTable rules{};
    std::array<SelectedTerrainModes, 5> selected_modes{};
    std::shared_ptr<const CostMap> blocked_cost_layer;

    [[nodiscard]] bool has_blocked_corner(const DirectionMap& direction_map, const Eigen::Vector2d& grid_coord) const;
    [[nodiscard]] bool is_direction_prohibited(const DirectionMap& direction_map, const Eigen::Vector2i& grid_coord, const Eigen::Vector2d& move_dir, double dot_threshold) const;
    [[nodiscard]] const TraversalMode* selected_mode(uint8_t label, bool is_up) const;
};

[[nodiscard]] TerrainTraversalConstraints build_terrain_traversal_constraints(
    const DirectionMap& direction_map,
    const TraversalConfiguration& configuration,
    PerformanceState performance
);

// ════════════════════════════════════════════════════════════════
//  代价地图
// ════════════════════════════════════════════════════════════════

class CostMap {
public:
    using Ptr = std::shared_ptr<CostMap>;
    using ConstPtr = std::shared_ptr<const CostMap>;

    explicit CostMap(int width, int height, double resolution, double origin_x, double origin_y, const std::vector<uint8_t>& data);
    explicit CostMap(const nav_msgs::msg::OccupancyGrid& occupancy_grid);

    CostMap merge(const CostMap& other) const;

    Eigen::Vector2d map_coord_to_grid(const Eigen::Vector2d& map_coord) const;
    Eigen::Vector2d grid_coord_to_map(const Eigen::Vector2d& grid_coord) const;

    bool is_valid_coord(const Eigen::Vector2i& grid_coord) const;
    bool is_valid_coord(const Eigen::Vector2d& grid_coord) const;

    uint8_t at(const Eigen::Vector2i& grid_coord) const;
    double interpolate(const Eigen::Vector2d& grid_coord) const;
    Eigen::Vector2d gradient(const Eigen::Vector2d& grid_coord) const;

    const int width, height;
    const double resolution, origin_x, origin_y;
    const std::vector<uint8_t> data;
};

// ════════════════════════════════════════════════════════════════
//  方向场地图（携带方向向量 + 地形标签 + 双份只读配置）
// ════════════════════════════════════════════════════════════════

class DirectionMap {
public:
    using Ptr = std::shared_ptr<DirectionMap>;
    using ConstPtr = std::shared_ptr<const DirectionMap>;

    struct DirectionSample {
        Eigen::Vector2d value;
        Eigen::Matrix2d gradient; // d(value)/d(grid_coord)，每列对应 grid_x/grid_y 偏导
    };

    explicit DirectionMap(
        const cv::Mat& direction_map, double resolution, double origin_x, double origin_y
    );

    explicit DirectionMap(
        int width, int height, double resolution, double origin_x, double origin_y,
        std::vector<Eigen::Vector2d> dir_data, std::vector<uint8_t> terrain_data
    );

private:
    explicit DirectionMap(
        int width, int height, double resolution, double origin_x, double origin_y,
        std::pair<std::vector<Eigen::Vector2d>, std::vector<uint8_t>> decoded
    );

    static std::pair<std::vector<Eigen::Vector2d>, std::vector<uint8_t>>
    decode_mat(const cv::Mat& mat);

public:
    Eigen::Vector2d map_coord_to_grid(const Eigen::Vector2d& map_coord) const;
    Eigen::Vector2d grid_coord_to_map(const Eigen::Vector2d& grid_coord) const;

    bool is_valid_coord(const Eigen::Vector2i& grid_coord) const;
    bool is_valid_coord(const Eigen::Vector2d& grid_coord) const;

    Eigen::Vector2d at(const Eigen::Vector2i& grid_coord) const;
    Eigen::Vector2d interpolate(const Eigen::Vector2d& grid_coord) const;
    DirectionSample interpolate_with_gradient(const Eigen::Vector2d& grid_coord) const;

    uint8_t terrain_at(const Eigen::Vector2i& grid_coord) const;
    uint8_t terrain_at(const Eigen::Vector2d& grid_coord) const;

    const int width, height;
    const double resolution, origin_x, origin_y;
    const std::vector<Eigen::Vector2d> data;
    const std::vector<uint8_t> terrain;

};

} // namespace nav_executor
