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
#include <nav_executor/path_executor/state_machine.hpp>

namespace nav_executor {

struct ProfileBlendParams {
    double v_step;
    double w_step;
    double acc_step;
    double alpha_step;
    double a_lat_step;
};

struct StepPhaseObservation {
    StepPhase phase = StepPhase::NONE;
    std::optional<size_t> segment_index;
};

// 对任意不可变路径做无状态阶段判定，供 TaskManager 在本控制周期接纳目标或规划结果前
// 检查 commit 边界。运行期 StepController::observe_step_phase() 还会结合已持有区段，
// 避免路径投影小幅回退时丢失当前台阶段。
[[nodiscard]] StepPhaseObservation classify_step_phase(const AnnotatedPath& path, double current_u);

// 台阶运行时：跟踪当前区段，并协调能力档、底盘模式和距离上报。
class StepController {
public:
    StepController(
        double step_dist_offset,
        const CapabilityProfile& normal_profile,
        const std::array<CapabilityProfile, 3>& capability_profiles,
        const ProfileBlendParams& blend_params,
        rclcpp::Logger logger
    );

    void set_path(AnnotatedPath::ConstPtr path);
    void clear();

    void update_active_segment(double current_u);

    [[nodiscard]] std::optional<size_t> find_active_segment_index(double current_u) const;
    [[nodiscard]] const StepPlanSegment* active_segment(double current_u) const;
    [[nodiscard]] const StepPlanSegment* current_command_segment(double current_u) const;
    [[nodiscard]] StepPhaseObservation observe_step_phase(double current_u) const;

    [[nodiscard]] const StepChassisCommand* current_chassis_command(double current_u) const;
    [[nodiscard]] uint8_t compute_step_distance_cm(const MincoTrajectory& path, double current_u) const;

    void tick_profile_blend();
    [[nodiscard]] const CapabilityProfile& effective_capability() const {
        return current_profile_;
    }

private:
    double step_dist_offset_;
    ProfileBlendParams blend_params_;
    CapabilityProfile normal_profile_;
    std::array<CapabilityProfile, 3> capability_profiles_;
    rclcpp::Logger logger_;

    CapabilityProfile current_profile_;
    CapabilityProfile target_profile_;

    AnnotatedPath::ConstPtr path_;
    size_t next_step_segment_index_ = 0;
    std::optional<size_t> held_step_segment_index_;
};

} // namespace nav_executor
