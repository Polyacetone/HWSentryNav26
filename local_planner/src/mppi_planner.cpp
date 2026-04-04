#include <local_planner/mppi_planner.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <omp.h>

namespace local_planner {

MPPIPlanner::MPPIPlanner(const MPPIParams& params)
    : params_(params),
      mean_v_(static_cast<size_t>(params.horizon), 0.0),
      mean_omega_(static_cast<size_t>(params.horizon), 0.0) {
    // 初始化每线程独立 RNG
    std::random_device rd;
    const int num_threads = std::max(1, params_.num_threads);
    thread_rngs_.reserve(static_cast<size_t>(num_threads));
    for (int i = 0; i < num_threads; ++i) {
        thread_rngs_.emplace_back(rd() + static_cast<unsigned>(i));
    }
}

MPPIState MPPIPlanner::dynamics_step(const MPPIState& s, double v_cmd, double omega_cmd) const {
    const double dt = params_.dt;
    MPPIState sn;
    // 一阶滞后速度模型
    sn.v_hat = params_.Av * s.v_hat + (1.0 - params_.Av) * v_cmd;
    sn.omega_hat = params_.A22 * s.omega_hat + params_.A24 * omega_cmd;
    // 运动学积分（梯形法则近似）
    sn.theta = s.theta + 0.5 * (s.omega_hat + sn.omega_hat) * dt;
    const double ct0 = std::cos(s.theta), st0 = std::sin(s.theta);
    const double ct1 = std::cos(sn.theta), st1 = std::sin(sn.theta);
    sn.x = s.x + 0.5 * (s.v_hat * ct0 + sn.v_hat * ct1) * dt;
    sn.y = s.y + 0.5 * (s.v_hat * st0 + sn.v_hat * st1) * dt;
    return sn;
}

double MPPIPlanner::evaluate_step_cost(
    const MPPIState& state,
    double v_cmd, double omega_cmd,
    double prev_v_cmd, double prev_omega_cmd,
    const SplineD& path, double path_u,
    const CostMap& cost_map,
    const DirectionMap& direction_map
) const {
    double cost = 0.0;

    // 1. 障碍物代价
    const Eigen::Vector2d pos(state.x, state.y);
    const Eigen::Vector2d grid = cost_map.map_coord_to_grid(pos);
    if (cost_map.is_valid_coord(grid)) {
        const double obs = cost_map.interpolate(grid) / 255.0;
        cost += params_.w_obstacle * obs * obs;
        // 超过阈值则给碰撞惩罚
        if (obs > params_.obstacle_threshold / 255.0) {
            cost += params_.collision_cost;
        }
    } else {
        cost += params_.collision_cost; // 越界高惩罚
    }

    // 2. 路径跟随代价
    const double u = std::clamp(path_u, 0.0, 1.0);
    const Eigen::Vector2d ref = path.evaluate(u);
    const Eigen::Vector2d d1 = path.derivative(u, 1);
    const double d1_norm = d1.norm();

    const Eigen::Vector2d err = pos - ref;
    double lateral_err = 0.0;
    if (d1_norm > 1e-6) {
        const Eigen::Vector2d tangent = d1 / d1_norm;
        lateral_err = std::abs(-err.x() * tangent.y() + err.y() * tangent.x());
    } else {
        lateral_err = err.norm();
    }
    cost += params_.w_path_follow * lateral_err * lateral_err;

    // 3. 朝向对齐代价
    if (d1_norm > 1e-6) {
        const double ref_theta = std::atan2(d1.y(), d1.x());
        double heading_err = state.theta - ref_theta;
        heading_err = std::atan2(std::sin(heading_err), std::cos(heading_err));
        cost += params_.w_heading * heading_err * heading_err;
    }

    // 4. 进度代价：鼓励沿路径前进
    cost += params_.w_progress * (1.0 - u) * (1.0 - u);

    // 5. 方向场代价（台阶对齐）
    const Eigen::Vector2d dir_grid = direction_map.map_coord_to_grid(pos);
    if (direction_map.is_valid_coord(dir_grid)) {
        const Eigen::Vector2d dir = direction_map.interpolate(dir_grid);
        const double dir_norm = dir.norm();
        if (dir_norm > params_.step_norm_threshold) {
            const Eigen::Vector2d heading(std::cos(state.theta), std::sin(state.theta));
            const double cross = heading.x() * dir.y() - heading.y() * dir.x();
            cost += params_.w_direction * cross * cross;
        }
    }

    // 6. 控制量平方代价
    cost += params_.w_control_v * v_cmd * v_cmd;
    cost += params_.w_control_omega * omega_cmd * omega_cmd;

    // 7. 控制变化率代价（平滑性）
    const double dv = v_cmd - prev_v_cmd;
    const double dw = omega_cmd - prev_omega_cmd;
    cost += params_.w_control_dv * dv * dv;
    cost += params_.w_control_domega * dw * dw;

    return cost;
}

double MPPIPlanner::evaluate_terminal_cost(
    const MPPIState& state,
    const SplineD& path, double path_u
) const {
    const double u = std::clamp(path_u, 0.0, 1.0);
    const Eigen::Vector2d pos(state.x, state.y);
    const Eigen::Vector2d ref = path.evaluate(u);
    const double dist = (pos - ref).norm();

    // 终端代价：到路径参考点距离 + 剩余路径长度
    return params_.w_terminal_goal * (dist * dist + (1.0 - u) * (1.0 - u));
}

double MPPIPlanner::project_to_path(
    const SplineD& path,
    const Eigen::Vector2d& pos,
    double u_hint
) const {
    return project_to_spline_u(
        path, pos, u_hint,
        params_.proj_num_samples,
        params_.proj_search_window,
        params_.proj_lazy_dist
    );
}

MPPIOutput MPPIPlanner::plan(const MPPIInput& input) {
    const auto plan_start = std::chrono::steady_clock::now();
    const int K = params_.num_rollouts;
    const int T = params_.horizon;
    const double dt = params_.dt;
    const int num_iterations = std::max(1, params_.num_iterations);
    const int num_threads = std::max(1, params_.num_threads);

    omp_set_num_threads(num_threads);

    // 初始化控制序列均值（warm start）
    if (!warm_started_) {
        std::fill(mean_v_.begin(), mean_v_.end(), 0.3);
        std::fill(mean_omega_.begin(), mean_omega_.end(), 0.0);
    }

    // 初始状态
    MPPIState s0;
    s0.x = input.robot_pose.x();
    s0.y = input.robot_pose.y();
    s0.theta = input.robot_pose.z();
    s0.v_hat = input.robot_vel.x();
    s0.omega_hat = input.robot_vel.y();

    // 机器人当前在路径上的投影（不随 horizon 前传）
    const double robot_u = project_to_path(*input.global_path, Eigen::Vector2d(s0.x, s0.y), input.global_path_u_hint);

    // 初始路径投影用于 rollout
    const double u0 = robot_u;

    // 预分配 rollout 缓冲
    std::vector<double> total_costs(static_cast<size_t>(K), 0.0);
    std::vector<std::vector<double>> rollout_v(static_cast<size_t>(K), std::vector<double>(static_cast<size_t>(T)));
    std::vector<std::vector<double>> rollout_omega(static_cast<size_t>(K), std::vector<double>(static_cast<size_t>(T)));

    const bool collect_debug_rollouts = input.collect_debug_rollouts && input.max_debug_rollouts > 0;

    MPPIOutput output;

    // ─── 多次迭代 MPPI ───
    for (int iter = 0; iter < num_iterations; ++iter) {
        const bool last_iter = (iter == num_iterations - 1);

        // 并行采样 rollouts
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < K; ++k) {
            const int tid = omp_get_thread_num();
            auto& rng = thread_rngs_[static_cast<size_t>(tid % static_cast<int>(thread_rngs_.size()))];
            std::normal_distribution<double> noise_v(0.0, params_.noise_sigma_v);
            std::normal_distribution<double> noise_omega(0.0, params_.noise_sigma_omega);

            MPPIState state = s0;
            double path_u = u0;
            double cost = 0.0;
            double prev_v = mean_v_[0];
            double prev_w = mean_omega_[0];

            for (int t = 0; t < T; ++t) {
                // 第一个 rollout 使用零噪声（确保均值序列本身被评估）
                double eps_v = (k == 0) ? 0.0 : noise_v(rng);
                double eps_w = (k == 0) ? 0.0 : noise_omega(rng);

                double v_cmd = std::clamp(mean_v_[static_cast<size_t>(t)] + eps_v, params_.v_cmd_min, params_.v_cmd_max);
                double w_cmd = std::clamp(mean_omega_[static_cast<size_t>(t)] + eps_w, params_.omega_cmd_min, params_.omega_cmd_max);

                rollout_v[static_cast<size_t>(k)][static_cast<size_t>(t)] = v_cmd;
                rollout_omega[static_cast<size_t>(k)][static_cast<size_t>(t)] = w_cmd;

                cost += evaluate_step_cost(
                    state, v_cmd, w_cmd, prev_v, prev_w,
                    *input.global_path, path_u,
                    *input.final_cost_map, *input.direction_map
                );

                prev_v = v_cmd;
                prev_w = w_cmd;
                state = dynamics_step(state, v_cmd, w_cmd);
                path_u = project_to_path(*input.global_path, Eigen::Vector2d(state.x, state.y), path_u);
            }

            // 终端代价
            cost += evaluate_terminal_cost(state, *input.global_path, path_u);

            total_costs[static_cast<size_t>(k)] = cost;
        }

        // MPPI 权重计算
        const double min_cost = *std::min_element(total_costs.begin(), total_costs.end());
        std::vector<double> weights(static_cast<size_t>(K));
        double weight_sum = 0.0;
        for (int k = 0; k < K; ++k) {
            weights[static_cast<size_t>(k)] = std::exp(-(total_costs[static_cast<size_t>(k)] - min_cost) / std::max(params_.temperature, 1e-6));
            weight_sum += weights[static_cast<size_t>(k)];
        }
        if (weight_sum < 1e-12) weight_sum = 1e-12;
        for (int k = 0; k < K; ++k) {
            weights[static_cast<size_t>(k)] /= weight_sum;
        }

        // 加权平均更新控制序列
        for (int t = 0; t < T; ++t) {
            double v_sum = 0.0, w_sum = 0.0;
            for (int k = 0; k < K; ++k) {
                v_sum += weights[static_cast<size_t>(k)] * rollout_v[static_cast<size_t>(k)][static_cast<size_t>(t)];
                w_sum += weights[static_cast<size_t>(k)] * rollout_omega[static_cast<size_t>(k)][static_cast<size_t>(t)];
            }
            mean_v_[static_cast<size_t>(t)] = std::clamp(v_sum, params_.v_cmd_min, params_.v_cmd_max);
            mean_omega_[static_cast<size_t>(t)] = std::clamp(w_sum, params_.omega_cmd_min, params_.omega_cmd_max);
        }

        // 时间维度上对控制序列做指数平滑，减少跳变
        if (params_.smooth_alpha > 1e-6 && T > 1) {
            const double alpha = params_.smooth_alpha;
            for (int t = 1; t < T; ++t) {
                mean_v_[static_cast<size_t>(t)] = alpha * mean_v_[static_cast<size_t>(t - 1)] + (1.0 - alpha) * mean_v_[static_cast<size_t>(t)];
                mean_omega_[static_cast<size_t>(t)] = alpha * mean_omega_[static_cast<size_t>(t - 1)] + (1.0 - alpha) * mean_omega_[static_cast<size_t>(t)];
            }
        }

        // 收集 debug rollouts（仅最后一次迭代）
        if (last_iter && collect_debug_rollouts) {
            const int rollout_stride = std::max(1, K / std::max(1, input.max_debug_rollouts));
            output.debug_rollouts.reserve(static_cast<size_t>(input.max_debug_rollouts));
            for (int k = 0; k < K && static_cast<int>(output.debug_rollouts.size()) < input.max_debug_rollouts; k += rollout_stride) {
                std::vector<Eigen::Vector2d> rollout;
                rollout.reserve(static_cast<size_t>(T + 1));
                MPPIState state = s0;
                rollout.emplace_back(state.x, state.y);
                for (int t = 0; t < T; ++t) {
                    state = dynamics_step(state, rollout_v[static_cast<size_t>(k)][static_cast<size_t>(t)],
                                                  rollout_omega[static_cast<size_t>(k)][static_cast<size_t>(t)]);
                    rollout.emplace_back(state.x, state.y);
                }
                output.debug_rollouts.push_back(std::move(rollout));
            }
        }
    }
    warm_started_ = true;

    // 用更新后的最优控制序列进行前向传播，生成最优轨迹
    output.best_trajectory.reserve(static_cast<size_t>(T + 1));
    output.best_timestamps.reserve(static_cast<size_t>(T + 1));

    MPPIState state = s0;
    double path_u = u0;
    output.best_trajectory.emplace_back(state.x, state.y);
    output.best_timestamps.push_back(0.0f);

    bool raw_step_up = false, raw_step_down = false;

    for (int t = 0; t < T; ++t) {
        state = dynamics_step(state, mean_v_[static_cast<size_t>(t)], mean_omega_[static_cast<size_t>(t)]);
        path_u = project_to_path(*input.global_path, Eigen::Vector2d(state.x, state.y), path_u);
        output.best_trajectory.emplace_back(state.x, state.y);
        output.best_timestamps.push_back(static_cast<float>((t + 1) * dt));

        // 台阶前瞻检测（在最优轨迹上）
        if (input.direction_map) {
            const Eigen::Vector2d pos(state.x, state.y);
            const Eigen::Vector2d g = input.direction_map->map_coord_to_grid(pos);
            if (input.direction_map->is_valid_coord(Eigen::Vector2i(
                static_cast<int>(std::floor(g.x())),
                static_cast<int>(std::floor(g.y()))
            ))) {
                const Eigen::Vector2d dir = input.direction_map->interpolate(g);
                if (dir.norm() > params_.step_norm_threshold) {
                    const Eigen::Vector2d heading(std::cos(state.theta), std::sin(state.theta));
                    const double dot_val = dir.normalized().dot(heading);
                    if (dot_val > 0.5) raw_step_up = true;
                    if (dot_val < -0.5) raw_step_down = true;
                }
            }
        }
    }

    output.robot_u = robot_u;
    output.updated_u_hint = path_u;

    const double max_cost = *std::max_element(total_costs.begin(), total_costs.end());
    const double min_cost_final = *std::min_element(total_costs.begin(), total_costs.end());
    double cost_sum = 0.0;
    for (double cost : total_costs) cost_sum += cost;
    output.debug_cost_min = min_cost_final;
    output.debug_cost_max = max_cost;
    output.debug_cost_mean = cost_sum / static_cast<double>(std::max(K, 1));
    output.step_up_ahead = raw_step_up;
    output.step_down_ahead = raw_step_down;

    // Shift 控制序列用于下一次 warm start
    for (int t = 0; t < T - 1; ++t) {
        mean_v_[static_cast<size_t>(t)] = mean_v_[static_cast<size_t>(t + 1)];
        mean_omega_[static_cast<size_t>(t)] = mean_omega_[static_cast<size_t>(t + 1)];
    }
    // 最后一步用前一步的值填充
    if (T >= 2) {
        mean_v_[static_cast<size_t>(T - 1)] = mean_v_[static_cast<size_t>(T - 2)];
        mean_omega_[static_cast<size_t>(T - 1)] = mean_omega_[static_cast<size_t>(T - 2)];
    }

    output.debug_plan_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - plan_start
    ).count();

    return output;
}

} // namespace local_planner