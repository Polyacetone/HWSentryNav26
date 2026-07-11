#include <nav_executor/path_executor/solver/search/search_cost.hpp>

#include <algorithm>
#include <cmath>

#include <nav_executor/path_executor/solver/bilinear_sampling.hpp>
#include <nav_executor/path_executor/solver/mpc_utils.hpp>

namespace nav_executor::search {

namespace {
constexpr double REACH_EPS = 1e-6;
// 搜索期样条投影参数：粗搜不需要 FDDP 那么密的采样。
constexpr int PROJ_SAMPLES = 20;
constexpr double PROJ_WINDOW = 0.2;
constexpr double PROJ_LAZY = 0.1;
} // namespace

StateKey SearchEnvironment::make_key(const SearchState& s) const {
    const double gx = (s.x - cost_info.origin_x) * cost_info.inv_resolution;
    const double gy = (s.y - cost_info.origin_y) * cost_info.inv_resolution;

    const double two_pi = 2.0 * M_PI;
    double theta_norm = std::fmod(s.theta, two_pi);
    if (theta_norm < 0.0) theta_norm += two_pi;
    const int theta_bin = static_cast<int>(theta_norm / two_pi * params.theta_bins) % params.theta_bins;

    const int v_bin = static_cast<int>(std::lround(s.v / std::max(params.v_bin_size, 1e-3)));

    return StateKey {
        .col = static_cast<int>(std::lround(gx)),
        .row = static_cast<int>(std::lround(gy)),
        .theta_bin = theta_bin,
        .v_bin = v_bin,
    };
}

bool SearchEnvironment::is_feasible(const SearchState& s) const {
    const double gx = (s.x - cost_info.origin_x) * cost_info.inv_resolution;
    const double gy = (s.y - cost_info.origin_y) * cost_info.inv_resolution;
    if (gx < 0.0 || gy < 0.0 || gx > cost_info.width - 1 || gy > cost_info.height - 1) {
        return false;
    }
    return eval_cost_bilinear(cost_grid, cost_info, s.x, s.y).value < params.collision_threshold;
}

bool SearchEnvironment::is_goal(const SearchState& s) const {
    return (Eigen::Vector2d(s.x, s.y) - goal_xy).norm() < params.goal_tolerance;
}

double SearchEnvironment::edge_cost(const SearchState& next) const {
    // 各避障/跟踪/台阶项二次化，与 FDDP running cost（0.5·(w·residual)²）同构，
    // 使搜索 basin 与 FDDP basin 形状一致，令双解 cost 比较更公平。
    // w_time 保持线性——它是 A* 的 g-cost 基线，也是 H0 admissible 下界的折算依据。
    // 二次项均 ≥ 0，故 g 单调非减，A* 完备性/有界次优不受影响。
    double cost = params.w_time; // 基础步代价（线性）

    // ── 避障（↔ 0.5·(environment_weights.obstacle·cost/255)²）──
    const double obs = eval_cost_bilinear(cost_grid, cost_info, next.x, next.y).value;
    const double r_obs = params.w_obstacle * obs / 255.0;
    cost += 0.5 * r_obs * r_obs;

    // ── Frenet 横向误差（↔ 0.5·(q_y·ey)²）──
    const double u = spline.project_extrapolated(
        Eigen::Vector2d(next.x, next.y), start_u, PROJ_SAMPLES, PROJ_WINDOW, PROJ_LAZY
    );
    const auto se = spline.eval(u);
    const double ex = next.x - se.p.x();
    const double ey_w = next.y - se.p.y();
    const double ey = -ex * se.sin_r + ey_w * se.cos_r;
    const double r_lat = params.w_lateral * ey;
    cost += 0.5 * r_lat * r_lat;

    // ── Frenet 航向误差（↔ 0.5·(q_theta·etheta)²）──
    const double etheta = wrap_pi(next.theta - se.thetar);
    const double r_head = params.w_heading * etheta;
    cost += 0.5 * r_head * r_head;

    // ── 台阶方向对齐（↔ 0.5·(terrain_weights.direction·|heading × dir|)²）──
    const Eigen::Vector2d dir = eval_dir_bilinear_value_only(dir_grid, dir_info, next.x, next.y);
    if (dir.squaredNorm() > 1e-8) {
        const Eigen::Vector2d heading(std::cos(next.theta), std::sin(next.theta));
        const double cross = heading.x() * dir.y() - heading.y() * dir.x();
        const double r_align = params.w_step_align * cross;
        cost += 0.5 * r_align * r_align;
    }

    // ── 台阶入口可达速度（↔ 0.5·(sqrt(step_reachability_*)·relu)² = 0.5·w·relu²）──
    if (active_step_mode.has_value() && active_step_mode->step_entry_u.has_value()) {
        const double entry_u = std::clamp(*active_step_mode->step_entry_u, 0.0, 1.0);
        const double cur_u = std::clamp(u, 0.0, 1.0);
        if (cur_u < entry_u) {
            const double d = spline.arc_length(cur_u, entry_u, 8);
            const double a_guide = std::max(step_guide_acc, REACH_EPS);
            const double v = next.v;
            const double r_lo = std::sqrt(std::max(0.0, std::max(0.0, v) * std::max(0.0, v) + 2.0 * a_guide * d));
            const double r_hi = std::sqrt(std::max(0.0, v * v - 2.0 * a_guide * d));
            const double relu_lo = positive_part(active_step_mode->speed_min - r_lo);
            const double relu_hi = positive_part(r_hi - active_step_mode->speed_max);
            cost += 0.5 * params.w_step_reach * relu_lo * relu_lo;
            cost += 0.5 * params.w_step_reach * relu_hi * relu_hi;
        }
    }

    return cost;
}

} // namespace nav_executor::search
