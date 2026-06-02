#pragma once

#include <optional>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/logger.hpp>

#include <path_follower/common/chassis_defs.hpp>
#include <path_follower/solver/mpc_solver.hpp>
#include <path_follower/common/nav_map.hpp>
#include <path_follower/path/spline_path.hpp>

namespace path_follower {

struct StepDetectionParams {
    double detect_norm_threshold;
    double detect_dot_threshold;
    double path_sample_resolution;
    double prepare_distance;
    double active_distance;
    double release_distance;
};

struct StepBlockReplanParams {
    bool enable;
    double lookahead_distance;
    double sample_resolution;
    double step_norm_threshold;
    double obstacle_cost_threshold;
    double predicted_obstacle_ratio_threshold;
};

struct ProfileBlendParams {
    double v_step;
    double w_step;
    double acc_step;
    double alpha_step;
    double a_lat_step;
};

class StepController {
public:
    struct StepPlanSegment {
        int path_version = 0;
        double prepare_u = 0.0;
        double active_u = 0.0;
        double step_enter_u = 0.0;
        double step_exit_u = 0.0;
        double release_u = 1.0;
        Eigen::Vector2d step_enter_pos_map = Eigen::Vector2d::Zero();
        Eigen::Vector2d step_exit_pos_map = Eigen::Vector2d::Zero();
        Eigen::Vector2d dir_map = Eigen::Vector2d::Zero();
        StepDirection direction = StepDirection::UP;
        ActiveStepMode command;
        uint8_t terrain_label = 0;
    };

    StepController(
        const StepDetectionParams& step_detection,
        const StepBlockReplanParams& step_block_replan,
        double step_dist_offset,
        const CapabilityProfile& normal_profile,
        const std::array<CapabilityProfile, 3>& capability_profiles,
        const ProfileBlendParams& blend_params,
        rclcpp::Logger logger
    );

    void update_plan_for_path_change(
        bool has_new_path,
        const std::optional<SplinePath>& path,
        const DirectionMap* direction_map
    );
    void clear_plan();
    void clear_runtime_state();

    void update_active_segment(
        double current_u,
        const std::optional<SplinePath>& global_path,
        bool fixed_goal,
        const Eigen::Vector2d& fixed_goal_pos
    );

    [[nodiscard]] std::optional<size_t> find_active_segment_index(double current_u) const;
    [[nodiscard]] const StepPlanSegment* active_segment(double current_u) const;
    [[nodiscard]] const StepPlanSegment* current_command_segment(double current_u) const;

    [[nodiscard]] std::optional<ActiveStepMode> current_active_step_mode(double current_u) const;
    [[nodiscard]] bool is_step_active(double current_u) const;
    [[nodiscard]] bool should_activate_step_mode(double current_u) const;
    [[nodiscard]] uint8_t compute_step_distance_cm(const SplinePath& path, double current_u) const;

    void tick_profile_blend();
    [[nodiscard]] const CapabilityProfile& current_blended_profile() const { return current_profile_; }

    [[nodiscard]] bool check_block_replan(
        const SplinePath& path,
        double current_u,
        const DirectionMap* masked_direction_map,
        const CostMap* current_dynamic_cost_map,
        const std::vector<const CostMap*>& per_step_dynamic_cost_maps
    ) const;

    [[nodiscard]] bool is_path_locked() const {
        return held_step_segment_index_.has_value() && step_locked_path_.has_value();
    }
    [[nodiscard]] const std::optional<SplinePath>& locked_path() const { return step_locked_path_; }
    [[nodiscard]] bool locked_fixed_goal() const { return step_locked_fixed_goal_; }
    [[nodiscard]] const Eigen::Vector2d& locked_fixed_goal_pos() const { return step_locked_fixed_goal_pos_; }

    [[nodiscard]] bool consume_deferred_update() {
        bool v = deferred_external_path_update_;
        deferred_external_path_update_ = false;
        return v;
    }
    void set_deferred_update() { deferred_external_path_update_ = true; }

private:
    struct FollowStepBlockSampleStats {
        int sample_count = 0;
        int step_sample_count = 0;
        int blocked_step_sample_count = 0;
    };

    std::vector<StepPlanSegment> build_step_plan(
        const SplinePath& path,
        const DirectionMap& direction_map
    ) const;
    std::optional<ActiveStepMode> build_step_command(
        StepDirection direction,
        const Eigen::Vector2d& step_enter_pos_map,
        double step_enter_u,
        const DirectionMap& direction_map
    ) const;

    double advance_path_u_by_distance(const SplinePath& path, double start_u, double distance) const;
    double retreat_path_u_by_distance(const SplinePath& path, double start_u, double distance) const;

    static std::optional<FollowStepBlockSampleStats> sample_block_replan_stats(
        const StepBlockReplanParams& p,
        const SplinePath& path,
        double start_u,
        const CostMap* dynamic_cost_map,
        const std::vector<const CostMap*>& dynamic_prediction_maps,
        const DirectionMap& direction_map
    );

    StepDetectionParams step_detection_;
    StepBlockReplanParams step_block_replan_;
    double step_dist_offset_;
    ProfileBlendParams blend_params_;
    CapabilityProfile normal_profile_;
    std::array<CapabilityProfile, 3> capability_profiles_;
    rclcpp::Logger logger_;

    CapabilityProfile current_profile_;
    CapabilityProfile target_profile_;

    int path_version_ = 0;

    std::vector<StepPlanSegment> step_plan_;

    std::optional<size_t> held_step_segment_index_;
    std::optional<SplinePath> step_locked_path_;
    bool step_locked_fixed_goal_ = false;
    Eigen::Vector2d step_locked_fixed_goal_pos_ = Eigen::Vector2d::Zero();
    bool deferred_external_path_update_ = false;
};

} // namespace path_follower
