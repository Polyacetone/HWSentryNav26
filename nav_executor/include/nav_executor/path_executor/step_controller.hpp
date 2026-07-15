#pragma once

#include <optional>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <nav_executor/common/chassis_defs.hpp>
#include <nav_executor/common/annotated_path.hpp>
#include <nav_executor/common/minco_trajectory.hpp>
#include <nav_executor/path_planner/nav_map.hpp>
#include <nav_executor/path_executor/solver/mpc_solver.hpp>

namespace nav_executor {

struct ProfileBlendParams {
    double v_step;
    double w_step;
    double acc_step;
    double alpha_step;
    double a_lat_step;
};

// 台阶运行时逻辑：消费 AnnotatedPath.step_segments，负责段跟踪、profile 时间域融合、台阶模式激活判定与距离上报。
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
    void set_path(AnnotatedPath::ConstPtr path);
    // 无 path 时清空运行时状态。
    void clear();

    // 每周期沿 u 更新当前持有的台阶段。
    void update_active_segment(double current_u);

    [[nodiscard]] std::optional<size_t> find_active_segment_index(double current_u) const;
    [[nodiscard]] const StepPlanSegment* active_segment(double current_u) const;
    [[nodiscard]] const StepPlanSegment* current_command_segment(double current_u) const;

    [[nodiscard]] const StepChassisCommand* current_chassis_command(double current_u) const;
    [[nodiscard]] bool is_step_nonpreemptible(double current_u) const;
    [[nodiscard]] bool should_activate_chassis_mode(double current_u) const;
    [[nodiscard]] uint8_t compute_step_distance_cm(const MincoTrajectory& path, double current_u) const;

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

    AnnotatedPath::ConstPtr path_;
    std::optional<size_t> held_step_segment_index_;
};

} // namespace nav_executor
