#pragma once

#include <optional>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/common/chassis_defs.hpp>
#include <nav_executor/path/annotated_path.hpp>
#include <nav_executor/path/spline_path.hpp>
#include <nav_executor/planner/nav_map.hpp>
#include <nav_executor/solver/mpc_solver.hpp>

namespace nav_executor {

struct ProfileBlendParams {
    double v_step;
    double w_step;
    double acc_step;
    double alpha_step;
    double a_lat_step;
};

// 台阶运行时逻辑。
//
// 台阶几何标注（build_step_plan）已上移到 planner 的 step_annotator。此处只
// 消费 AnnotatedPath.step_segments，负责运行时的：
//   - 台阶段激活/持有跟踪（沿 u 推进）
//   - 能力档位（profile）时间域融合
//   - 台阶底盘模式激活判定
//   - 台阶距离上报
//
// 不再持有/锁存 path（锁存机制已移除；不可抢占态直接禁止规划）。
class StepController {
public:
    StepController(
        double step_dist_offset,
        const CapabilityProfile& normal_profile,
        const std::array<CapabilityProfile, 3>& capability_profiles,
        const ProfileBlendParams& blend_params,
        rclcpp::Logger logger
    );

    // 路径切换时重置段跟踪状态并绑定新的 step_segments。
    void set_path(const AnnotatedPath* path);
    // 无 path 时清空运行时状态。
    void clear();

    // 每周期沿 u 更新当前持有的台阶段。
    void update_active_segment(double current_u);

    [[nodiscard]] std::optional<size_t> find_active_segment_index(double current_u) const;
    [[nodiscard]] const StepPlanSegment* active_segment(double current_u) const;
    [[nodiscard]] const StepPlanSegment* current_command_segment(double current_u) const;

    [[nodiscard]] std::optional<ActiveStepMode> current_active_step_mode(double current_u) const;
    [[nodiscard]] bool is_step_active(double current_u) const;
    [[nodiscard]] bool should_activate_step_mode(double current_u) const;
    [[nodiscard]] uint8_t compute_step_distance_cm(const SplinePath& path, double current_u) const;

    void tick_profile_blend();
    [[nodiscard]] const CapabilityProfile& current_blended_profile() const { return current_profile_; }

private:
    double step_dist_offset_;
    ProfileBlendParams blend_params_;
    CapabilityProfile normal_profile_;
    std::array<CapabilityProfile, 3> capability_profiles_;
    rclcpp::Logger logger_;

    CapabilityProfile current_profile_;
    CapabilityProfile target_profile_;

    const std::vector<StepPlanSegment>* step_plan_ = nullptr;
    std::optional<size_t> held_step_segment_index_;
};

} // namespace nav_executor
