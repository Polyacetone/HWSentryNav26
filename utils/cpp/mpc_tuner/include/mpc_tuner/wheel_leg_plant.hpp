#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/path_executor/solver/mpc_types.hpp>

namespace mpc_tuner {

struct PlantSample {
    Eigen::Vector3d pose = Eigen::Vector3d::Zero();
    nav_executor::ChassisMotionState chassis;
};

class WheelLegPlant {
public:
    WheelLegPlant();

    void reset(const Eigen::Vector3d& pose, uint64_t seed);
    [[nodiscard]] std::vector<PlantSample> step(const Eigen::Vector2d& command, double duration);
    [[nodiscard]] PlantSample sample() const;

private:
    static constexpr double DT = 0.001;
    static constexpr int STATE_DIM = 10;
    static constexpr int INPUT_DIM = 4;

    using State = Eigen::Matrix<double, STATE_DIM, 1>;
    using StateMatrix = Eigen::Matrix<double, STATE_DIM, STATE_DIM>;
    using InputMatrix = Eigen::Matrix<double, STATE_DIM, INPUT_DIM>;
    using GainMatrix = Eigen::Matrix<double, INPUT_DIM, STATE_DIM>;
    using Input = Eigen::Matrix<double, INPUT_DIM, 1>;

    void update_model_cache();
    void lqr_substep();
    static std::pair<double, double> clamp_command(double velocity, double omega);
    static double rate_limit(double current, double target, double max_rate);

    State state_ = State::Zero();
    Eigen::Vector3d pose_ = Eigen::Vector3d::Zero();
    StateMatrix a_ = StateMatrix::Zero();
    InputMatrix b_ = InputMatrix::Zero();
    GainMatrix k_ = GainMatrix::Zero();
    Input u_d_ = Input::Zero();

    double leg_left_ = 0.15;
    double leg_right_ = 0.15;
    double velocity_target_ = 0.0;
    double omega_target_ = 0.0;
    double velocity_applied_ = 0.0;
    double omega_applied_ = 0.0;
    double theta_target_ = 0.0;
    double s_reference_ = 0.0;
    std::mt19937_64 rng_;
    std::normal_distribution<double> standard_normal_ {0.0, 1.0};
};

} // namespace mpc_tuner
