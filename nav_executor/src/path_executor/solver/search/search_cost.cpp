#include <nav_executor/path_executor/solver/search/search_cost.hpp>

#include <algorithm>
#include <cmath>

#include <nav_executor/path_executor/solver/bilinear_sampling.hpp>
#include <nav_executor/path_executor/solver/mpc_utils.hpp>

namespace nav_executor::search {

namespace {
constexpr double REACH_EPS = 1e-6;
// 搜索期样条投影/弧长积分采样数：粗搜不需要 FDDP 那么密。
constexpr int ARC_SAMPLES = 8;
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

bool SearchEnvironment::is_move_allowed(const SearchState& from, const SearchState& to) const {
    // 台阶方向硬约束：单/双向可通行由 terrain rule 编码，夹角判据由 dot_threshold 控制。
    // 使用 base（未掩码）方向场，避免走廊掩码擦除方向矢量导致约束失效。
    if (base_dir == nullptr || terrain == nullptr) return true;

    const Eigen::Vector2d g = base_dir->map_coord_to_grid(Eigen::Vector2d(to.x, to.y));
    if (!base_dir->is_valid_coord(g)) return true;

    // 运动方向：位移向量；原地转向退化时用航向近似。
    Eigen::Vector2d move(to.x - from.x, to.y - from.y);
    if (move.squaredNorm() < 1e-12) {
        move = Eigen::Vector2d(std::cos(to.theta), std::sin(to.theta));
    }

    const Eigen::Vector2i gi(static_cast<int>(std::lround(g.x())), static_cast<int>(std::lround(g.y())));
    return !terrain->is_direction_prohibited(*base_dir, gi, move, params.step_dir_dot_threshold);
}

double SearchEnvironment::s_remaining(double u) const {
    const double uc = std::clamp(u, 0.0, 1.0);
    if (uc >= 1.0) return 0.0;
    return spline.arc_length(uc, 1.0, ARC_SAMPLES);
}

double SearchEnvironment::edge_cost(const SearchState& next, double u) const {
    // 各项与 FDDP running cost 主项同构：二次项 0.5·(w·residual)²，线性项 w·quantity，
    // 令搜索 basin 与 FDDP basin 形状一致，双解 cost 比较更公平。
    // 二次/线性项均 ≥ 0，故 g 单调非减，A* 完备性 / 有界次优不受影响。
    double cost = params.w_time; // 基础步代价（线性，anchor admissible 下界基线）

    // ── 沿样条进度（↔ tracking_weights.q_u·s_remaining，取代前瞻目标点）──
    // 线性、随剩余弧长递减，自发牵引"沿路径推进"；终点处饱和为 0，不奖励越过终点。
    cost += params.w_progress * s_remaining(u);

    // ── 避障（↔ 0.5·(environment_weights.obstacle·cost/255)²）──
    const double obs = eval_cost_bilinear(cost_grid, cost_info, next.x, next.y).value;
    const double r_obs = params.w_obstacle * obs / 255.0;
    cost += 0.5 * r_obs * r_obs;

    // ── Frenet 横向 / 航向误差（↔ 0.5·(q_y·ey)² + 0.5·(q_theta·etheta)²）──
    const auto se = spline.eval(std::clamp(u, 0.0, 1.0));
    const double ex = next.x - se.p.x();
    const double ey_w = next.y - se.p.y();
    const double ey = -ex * se.sin_r + ey_w * se.cos_r;
    const double r_lat = params.w_lateral * ey;
    cost += 0.5 * r_lat * r_lat;

    const double etheta = wrap_pi(next.theta - se.thetar);
    const double r_head = params.w_heading * etheta;
    cost += 0.5 * r_head * r_head;

    // ── 终端减速（↔ 0.5·(terminal_weights.q_v_final·relu(v - v_allowed))²）──
    // v_allowed = sqrt(2·a_brake·s_remaining + v_target²)：临近终点时压低允许速度，
    // 令超越终点的高速方案被抑制，无需 is_goal 硬终止即自然停在终点。
    const double v_allowed = std::sqrt(
        std::max(0.0, 2.0 * brake_decel * s_remaining(u)) + brake_v_target * brake_v_target
    );
    const double r_brake = params.w_brake * positive_part(next.v - v_allowed);
    cost += 0.5 * r_brake * r_brake;

    // ── 台阶方向对齐（↔ 0.5·(terrain_weights.direction·|heading × dir|)²）──
    const Eigen::Vector2d dir = eval_dir_bilinear_value_only(dir_grid, dir_info, next.x, next.y);
    const double dir_norm_sq = dir.squaredNorm();
    if (dir_norm_sq > 1e-8) {
        const Eigen::Vector2d heading(std::cos(next.theta), std::sin(next.theta));
        const double cross = heading.x() * dir.y() - heading.y() * dir.x();
        const double r_align = params.w_step_align * cross;
        cost += 0.5 * r_align * r_align;
    }

    if (active_step_mode.has_value()) {
        const double speed_min = active_step_mode->speed_min;
        const double speed_max = active_step_mode->speed_max;

        // ── 台阶内部速度区间（↔ 0.5·(terrain_weights.step_vel_weight·dir_norm·relu_vstep)²）──
        //    高权重软约束：在台阶面上把速度维持在 [speed_min, speed_max]。
        if (dir_norm_sq > 1e-10) {
            const double dir_norm = std::sqrt(dir_norm_sq);
            const double v = next.v;
            const double v_err = v < speed_min ? (speed_min - v) : (v > speed_max ? (v - speed_max) : 0.0);
            const double r_vstep = params.w_step_vel * dir_norm * v_err;
            cost += 0.5 * r_vstep * r_vstep;
        }

        // ── 台阶入口可达速度（↔ 0.5·(sqrt(step_reachability_*)·relu)²）──
        //    高权重软约束：入口前以引导加速度推算可达速度区间，欠速时牵引"先退后进"腾挪加速。
        if (active_step_mode->step_entry_u.has_value()) {
            const double entry_u = std::clamp(*active_step_mode->step_entry_u, 0.0, 1.0);
            const double cur_u = std::clamp(u, 0.0, 1.0);
            if (cur_u < entry_u) {
                const double d = spline.arc_length(cur_u, entry_u, ARC_SAMPLES);
                const double a_guide = std::max(step_guide_acc, REACH_EPS);
                const double v = next.v;
                const double r_lo = std::sqrt(std::max(0.0, std::max(0.0, v) * std::max(0.0, v) + 2.0 * a_guide * d));
                const double r_hi = std::sqrt(std::max(0.0, v * v - 2.0 * a_guide * d));
                const double relu_lo = positive_part(speed_min - r_lo);
                const double relu_hi = positive_part(r_hi - speed_max);
                cost += 0.5 * params.w_step_reach * relu_lo * relu_lo;
                cost += 0.5 * params.w_step_reach * relu_hi * relu_hi;
            }
        }
    }

    return cost;
}

} // namespace nav_executor::search
