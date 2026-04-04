#pragma once

#include <Eigen/Core>
#include <random>
#include <vector>

#include <local_planner/nav_map.hpp>
#include <local_planner/utils.hpp>

namespace local_planner {

// ═══════════════════════ MPPI 参数 ═══════════════════════════

struct MPPIParams {
    // 采样参数
    int num_rollouts;           // 采样轨迹数量 K
    int horizon;                // 预测步数 T
    double dt;                  // 时间步长（秒）
    int num_iterations;         // MPPI 迭代次数（每次 plan() 内重复采样+更新）
    int num_threads;            // OMP 线程数

    // 5 状态简化模型参数
    double Av;                  // v 通道一阶滞后系数（≈0.905）
    double A22;                 // ω 通道 ZOH 系数
    double A24;                 // ω 通道 ZOH 输入系数

    // 控制量上下限
    double v_cmd_max;
    double v_cmd_min;
    double omega_cmd_max;
    double omega_cmd_min;

    // 噪声标准差
    double noise_sigma_v;
    double noise_sigma_omega;

    // 温度参数（MPPI 指数加权）
    double temperature;

    // 代价函数权重
    double w_obstacle;          // 障碍物代价权重
    double w_path_follow;       // 路径跟随代价权重
    double w_heading;           // 朝向对齐代价权重
    double w_progress;          // 沿路径进度代价权重
    double w_direction;         // 台阶方向对齐代价权重
    double w_control_v;         // 速度控制平滑代价权重
    double w_control_omega;     // 角速度控制平滑代价权重
    double w_terminal_goal;     // 终端代价权重（到路径终点距离）
    double w_control_dv;        // 速度变化率平滑代价权重
    double w_control_domega;    // 角速度变化率平滑代价权重

    // 台阶方向场参数
    double step_norm_threshold; // 台阶方向场模长阈值

    // 路径投影参数
    int proj_num_samples;
    double proj_search_window;
    double proj_lazy_dist;

    // 障碍物代价阈值（0~255 归一化后的阈值）
    double obstacle_threshold;

    // 碰撞代价（当超过 obstacle_threshold 时的额外大惩罚）
    double collision_cost;

    // 控制序列平滑参数
    double smooth_alpha;        // 指数平滑系数（0=不平滑, 1=全平滑）
};

// ═══════════════════════ MPPI 输入 ═══════════════════════════

struct MPPIInput {
    Eigen::Vector3d robot_pose;     // [x, y, θ]
    Eigen::Vector2d robot_vel;      // [v_actual, ω_actual]（来自 mpc_controller 反馈）
    const SplineD* global_path;     // 全局 B-spline 路径
    double global_path_u_hint;      // 路径投影 hint
    const CostMap* final_cost_map;  // 包含台阶掩码的最终代价图
    const DirectionMap* direction_map; // 掩码后的方向场
    bool collect_debug_rollouts = false;
    int max_debug_rollouts = 0;
};

// ═══════════════════════ MPPI 输出 ═══════════════════════════

struct MPPIOutput {
    std::vector<Eigen::Vector2d> best_trajectory;   // 最优空间轨迹
    std::vector<float> best_timestamps;             // 各点时间戳（相对于规划时刻，秒）
    std::vector<std::vector<Eigen::Vector2d>> debug_rollouts; // 调试用 rollout 子集
    double robot_u;                                 // 机器人当前在路径上的投影 u（不含 horizon 前向传播）
    double updated_u_hint;                          // MPPI horizon 末端 u（用于下次 warm start hint）
    double debug_plan_time_ms = 0.0;
    double debug_cost_min = 0.0;
    double debug_cost_mean = 0.0;
    double debug_cost_max = 0.0;
    bool step_up_ahead;                             // 沿轨迹前方有上台阶
    bool step_down_ahead;                           // 沿轨迹前方有下台阶
};

// ═══════════════════════ 5 状态 MPPI Rollout ════════════════

struct MPPIState {
    double x, y, theta;
    double v_hat, omega_hat;    // 一阶滞后有效速度
};

// ═══════════════════════ MPPI 规划器 ═════════════════════════

class MPPIPlanner {
public:
    explicit MPPIPlanner(const MPPIParams& params);

    /// 执行一次 MPPI 规划
    MPPIOutput plan(const MPPIInput& input);

    /// 重置 warm start（状态变化时调用）
    void reset() { warm_started_ = false; }

private:
    // 5 状态动力学前向传播
    MPPIState dynamics_step(const MPPIState& state, double v_cmd, double omega_cmd) const;

    // 单步代价评估
    double evaluate_step_cost(
        const MPPIState& state,
        double v_cmd, double omega_cmd,
        double prev_v_cmd, double prev_omega_cmd,
        const SplineD& path, double path_u,
        const CostMap& cost_map,
        const DirectionMap& direction_map
    ) const;

    // 终端代价
    double evaluate_terminal_cost(
        const MPPIState& state,
        const SplineD& path, double path_u
    ) const;

    // 路径投影：找到最近的 u
    double project_to_path(
        const SplineD& path,
        const Eigen::Vector2d& pos,
        double u_hint
    ) const;

    MPPIParams params_;
    std::vector<std::mt19937> thread_rngs_; // 每线程独立 RNG

    // 预分配控制序列缓冲
    std::vector<double> mean_v_, mean_omega_;   // 上一次最优控制序列（warm start）
    bool warm_started_ = false;
};

} // namespace local_planner