#pragma once

#include <nav_executor/path_executor/solver/mpc_types.hpp>
#include <nav_executor/common/spline_path.hpp>
#include <nav_executor/path_executor/solver/follow_problem.hpp>
#include <nav_executor/path_executor/solver/stop_problem.hpp>
#include <nav_executor/path_executor/solver/hold_problem.hpp>

namespace nav_executor {

class MPCSolver {
public:
    enum class FollowSolveStatus : uint8_t {
        FOLLOW = 0,
        STOP_AND_WAIT_REPLAN = 1,
    };

    struct FollowSolveResult {
        Eigen::Vector2d command = Eigen::Vector2d::Zero();
        MPCPrediction prediction;
        FollowSolveStatus status = FollowSolveStatus::FOLLOW;
        std::optional<RolloutLethalObstacleInfo> lethal_obstacle;
    };

    explicit MPCSolver(const MPCParams& params);

    void set_last_cmd(const Eigen::Vector2d& cmd);
    void reset_warm_start();

    void update_observer(const ChassisMotionState& chassis_state);
    void reset_observer();
    void set_energy_state(double remaining_energy, double rfr_pwr_limit);

    [[nodiscard]] double hidden_state_estimate() const {
        return x_h_hat_;
    }

    std::expected<FollowSolveResult, std::string> solve_follow(
        const SplinePath& global_path,
        const Eigen::Vector3d& chassis_pose_map,
        const ChassisMotionState& chassis_state,
        const CostMap& cost_map,
        const CostMap& masked_global_map,
        const std::vector<const CostMap*>& per_step_cost_maps,
        double prediction_dt,
        const DirectionMap& direction_map,
        const CapabilityProfile& blended_profile,
        std::optional<ActiveStepMode> active_step_mode,
        bool check_lethal_status
    );

    std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> solve_stop(
        const Eigen::Vector3d& chassis_pose_map,
        const ChassisMotionState& chassis_state,
        const CostMap& cost_map
    );

    std::expected<std::tuple<Eigen::Vector2d, MPCPrediction>, std::string> solve_hold(
        const Eigen::Vector2d& goal_map,
        const Eigen::Vector3d& chassis_pose_map,
        const ChassisMotionState& chassis_state,
        const CostMap& cost_map
    );

    [[nodiscard]] const MPCParams& params() const {
        return params_;
    }

private:
    MPCParams params_;
    Eigen::Vector2d last_cmd_ = Eigen::Vector2d::Zero();
    double last_u_ = 0.0;

    fddp::Solver<FollowProblem> follow_solver_;
    fddp::Solver<StopProblem> stop_solver_;
    fddp::Solver<HoldProblem> hold_solver_;
    bool follow_warm_ = false;
    bool stop_warm_ = false;
    bool hold_warm_ = false;

    std::vector<CostMapGridView> step_cost_grids_cache_;

    std::optional<SplinePath> prev_ref_control_points_;

    double x_h_hat_ = 0.0;
    double prev_v_act_ = 0.0;
    double prev_w_act_ = 0.0;
    double prev_schedule_rho_ = 0.0;
    bool observer_initialized_ = false;

    double remaining_energy_ = 200.0;
    double rfr_pwr_limit_ = 90.0;

    int fddp_lethal_consecutive_count_ = 0;

    StateVec make_initial_state(
        const Eigen::Vector3d& pose,
        const ChassisMotionState& chassis_state,
        const Eigen::Vector2d& cmd_clamped,
        double path_u
    ) const;
};

} // namespace nav_executor
