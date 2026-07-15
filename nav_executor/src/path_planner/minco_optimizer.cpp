#include <nav_executor/path_planner/minco_optimizer.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <nav_executor/path_planner/lbfgs_minimizer.hpp>

namespace nav_executor {

namespace {

constexpr int DIM = MincoMinJerk::DIM;
constexpr int NCOEF = MincoMinJerk::NCOEF;
constexpr double EPS = 1e-9;

// 平滑 relu 及其导数（约束罚：违反量 g>0 时二次惩罚）。
inline double violation(double g) { return std::max(g, 0.0); }
inline double violation_grad(double g) { return g > 0.0 ? 1.0 : 0.0; }

// 虚拟时间正性重参数化：T = softplus(tau_v) = ln(1+e^{tau_v})，保证 T>0 且光滑。
// dT/dtau_v = sigmoid(tau_v) = 1/(1+e^{-tau_v})。
inline double virtual_to_time(double tau_v, double min_t) {
    const double sp = tau_v > 30.0 ? tau_v : std::log1p(std::exp(tau_v));
    return sp + min_t;
}
inline double virtual_to_time_grad(double tau_v) {
    return tau_v > 30.0 ? 1.0 : 1.0 / (1.0 + std::exp(-tau_v));
}
inline double time_to_virtual(double t, double min_t) {
    const double sp = std::max(t - min_t, 1e-6);
    // 反 softplus：ln(e^{sp}-1)
    return sp > 30.0 ? sp : std::log(std::expm1(sp));
}

} // anonymous namespace

// ── 优化工作区：一次 optimize 内复用，避免反复分配 ──
struct MincoOptimizer::Workspace {
    int n_segments = 0;
    int n_waypoints = 0; // n_segments - 1

    MincoMinJerk minco;
    MincoMinJerk::BoundaryPVA head;
    MincoMinJerk::BoundaryPVA tail;

    const CostMap* cost_map = nullptr;
    const DirectionMap* direction_map = nullptr;
    const TerrainTraversalConstraints* terrain_constraints = nullptr;

    // 硬中间路点：index → 固定 (pos, θ)（速度软约束，不固定）。
    std::vector<MincoOptimizer::HardWaypoint> hard;
    std::vector<bool> waypoint_is_hard;
    std::vector<Eigen::Vector3d> hard_pos_theta; // 每个硬路点的 (x,y,θ)
    std::vector<MincoOptimizer::StepEntrySpeedWindow> step_entry_speed_windows;

    // 非完整约束的 AL 乘子（每采样点一个），rho 外层放大。
    std::vector<double> nonholo_lambda;
    double nonholo_rho = 1.0;

    // 当前解缓存
    Eigen::Matrix<double, DIM, Eigen::Dynamic> waypoints;
    std::vector<double> times;
};

MincoOptimizer::MincoOptimizer(Params params)
    : params_(std::move(params)) {}

// 采样点罚 + 能量 + 时间；梯度累积到 grad_c / grad_t_explicit。返回代价。
double MincoOptimizer::accumulate_penalties(
    Workspace& ws,
    Eigen::MatrixXd& grad_c,
    Eigen::VectorXd& grad_t_explicit
) const {
    const auto& coeffs = ws.minco.coefficients();
    const int n = ws.n_segments;
    const auto& w = params_.weights;
    const auto& lim = params_.limits;

    double cost = 0.0;
    grad_c.setZero(NCOEF * n, DIM);
    grad_t_explicit.setZero(n);

    int global_sample = 0;
    for (int seg = 0; seg < n; ++seg) {
        const double T = ws.times[static_cast<size_t>(seg)];
        const int c_off = NCOEF * seg;

        // ── min-jerk 能量：E = ∫₀ᵀ ‖p'''(t)‖² dt ──
        // p'''(t) = 6c₃ + 24c₄t + 60c₅t²，逐项积分（每维独立）：
        //   E = 36c₃²T + 144c₃c₄T² + (240c₃c₅+192c₄²)T³ + 720c₄c₅T⁴ + 720c₅²T⁵
        // （系数来自 ∫t^m dt = T^{m+1}/(m+1)：240=720/3, 192=576/3, 720=2880/4, 720=3600/5）
        const double T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
        for (int d = 0; d < DIM; ++d) {
            const double c3 = coeffs(c_off + 3, d);
            const double c4 = coeffs(c_off + 4, d);
            const double c5 = coeffs(c_off + 5, d);

            const double energy =
                  36.0 * c3 * c3 * T
                + 144.0 * c3 * c4 * T2
                + (240.0 * c3 * c5 + 192.0 * c4 * c4) * T3
                + 720.0 * c4 * c5 * T4
                + 720.0 * c5 * c5 * T5;
            cost += w.energy * energy;

            // ∂E/∂c₃, ∂E/∂c₄, ∂E/∂c₅
            grad_c(c_off + 3, d) += w.energy * (72.0 * c3 * T + 144.0 * c4 * T2 + 240.0 * c5 * T3);
            grad_c(c_off + 4, d) += w.energy * (144.0 * c3 * T2 + 384.0 * c4 * T3 + 720.0 * c5 * T4);
            grad_c(c_off + 5, d) += w.energy * (240.0 * c3 * T3 + 720.0 * c4 * T4 + 1440.0 * c5 * T5);
            // ∂E/∂T（显式，固定 c）
            grad_t_explicit(seg) += w.energy * (
                  36.0 * c3 * c3
                + 288.0 * c3 * c4 * T
                + 3.0 * (240.0 * c3 * c5 + 192.0 * c4 * c4) * T2
                + 2880.0 * c4 * c5 * T3
                + 3600.0 * c5 * c5 * T4
            );
        }

        // ── 时间罚 w_time · T ──
        cost += w.time * T;
        grad_t_explicit(seg) += w.time;

        // ── 采样点罚 ──
        const int S = std::max(params_.samples_per_segment, 1);
        for (int s = 0; s < S; ++s) {
            const double t = T * static_cast<double>(s) / static_cast<double>(S);
            // β 基与导数
            double b0[NCOEF], b1[NCOEF], b2[NCOEF], b3[NCOEF];
            {
                double tp = 1.0;
                for (int k = 0; k < NCOEF; ++k) { b0[k] = tp; tp *= t; }
                b1[0] = 0.0; tp = 1.0;
                for (int k = 1; k < NCOEF; ++k) { b1[k] = k * tp; tp *= t; }
                b2[0] = 0.0; b2[1] = 0.0; tp = 1.0;
                for (int k = 2; k < NCOEF; ++k) { b2[k] = k * (k - 1) * tp; tp *= t; }
                b3[0] = 0.0; b3[1] = 0.0; b3[2] = 0.0; tp = 1.0;
                for (int k = 3; k < NCOEF; ++k) { b3[k] = k * (k - 1) * (k - 2) * tp; tp *= t; }
            }
            // 物理量
            double px = 0, py = 0, th = 0;
            double vx = 0, vy = 0, vth = 0;
            double ax = 0, ay = 0, ath = 0;
            double jx = 0, jy = 0; // jerk（横坐标漂移用）
            for (int k = 0; k < NCOEF; ++k) {
                px += coeffs(c_off + k, 0) * b0[k];
                py += coeffs(c_off + k, 1) * b0[k];
                th += coeffs(c_off + k, 2) * b0[k];
                vx += coeffs(c_off + k, 0) * b1[k];
                vy += coeffs(c_off + k, 1) * b1[k];
                vth += coeffs(c_off + k, 2) * b1[k];
                ax += coeffs(c_off + k, 0) * b2[k];
                ay += coeffs(c_off + k, 1) * b2[k];
                ath += coeffs(c_off + k, 2) * b2[k];
                jx += coeffs(c_off + k, 0) * b3[k];
                jy += coeffs(c_off + k, 1) * b3[k];
            }

            const double cth = std::cos(th), sth = std::sin(th);
            const double v_signed = vx * cth + vy * sth; // 带符号纵向速度

            const double dt = T / static_cast<double>(S); // 积分测度

            // 累积**密度** ρ（不含 dt）与其对物理量 (px,py,th,vx,vy,vth,ax,ay) 的梯度。
            // 之后统一乘 dt 得代价；T 梯度含两支：测度 ∂dt/∂T·ρ 与横坐标漂移 dt·∂ρ/∂t·(s/S)。
            double density = 0.0;
            double gpx = 0, gpy = 0, gth = 0, gvx = 0, gvy = 0, gvth = 0, gax = 0, gay = 0;

            // (a) 障碍罚：使用双线性插值的精确 map 坐标梯度。
            if (ws.cost_map && w.obstacle > 0.0) {
                const Eigen::Vector2d grid = ws.cost_map->map_coord_to_grid(Eigen::Vector2d(px, py));
                if (ws.cost_map->is_valid_coord(grid)) {
                    const int x0 = static_cast<int>(std::floor(grid.x()));
                    const int y0 = static_cast<int>(std::floor(grid.y()));
                    const double tx = grid.x() - static_cast<double>(x0);
                    const double ty = grid.y() - static_cast<double>(y0);
                    const double c00 = static_cast<double>(ws.cost_map->at({x0, y0}));
                    const double c10 = static_cast<double>(ws.cost_map->at({x0 + 1, y0}));
                    const double c01 = static_cast<double>(ws.cost_map->at({x0, y0 + 1}));
                    const double c11 = static_cast<double>(ws.cost_map->at({x0 + 1, y0 + 1}));
                    const double cval = (
                        (1.0 - tx) * (1.0 - ty) * c00
                        + tx * (1.0 - ty) * c10
                        + (1.0 - tx) * ty * c01
                        + tx * ty * c11
                    ) / 255.0;
                    const double inv_scale = 1.0 / (255.0 * ws.cost_map->resolution);
                    const Eigen::Vector2d cgrad(
                        ((1.0 - ty) * (c10 - c00) + ty * (c11 - c01)) * inv_scale,
                        ((1.0 - tx) * (c01 - c00) + tx * (c11 - c10)) * inv_scale
                    );
                    density += w.obstacle * 0.5 * cval * cval;
                    gpx += w.obstacle * cval * cgrad.x();
                    gpy += w.obstacle * cval * cgrad.y();
                } else {
                    // 地图外必须比致命障碍更差，并提供指向地图内部的梯度。
                    const double x_min = ws.cost_map->origin_x;
                    const double y_min = ws.cost_map->origin_y;
                    const double x_max = x_min + static_cast<double>(ws.cost_map->width - 1) * ws.cost_map->resolution;
                    const double y_max = y_min + static_cast<double>(ws.cost_map->height - 1) * ws.cost_map->resolution;
                    const Eigen::Vector2d p_map(px, py);
                    const Eigen::Vector2d clamped(
                        std::clamp(px, x_min, x_max),
                        std::clamp(py, y_min, y_max)
                    );
                    const Eigen::Vector2d delta = p_map - clamped;
                    const double dist = std::sqrt(delta.squaredNorm() + EPS);
                    const double cval = 1.0 + dist / ws.cost_map->resolution;
                    const Eigen::Vector2d cgrad = delta / (dist * ws.cost_map->resolution);
                    density += w.obstacle * 0.5 * cval * cval;
                    gpx += w.obstacle * cval * cgrad.x();
                    gpy += w.obstacle * cval * cgrad.y();
                }
            }

            // (b) 速度窗（带符号）：v ≤ v_max, v ≥ v_min
            {
                const double over = v_signed - lim.vel_max;
                const double under = lim.vel_min - v_signed;
                const double go = violation(over), gu = violation(under);
                density += w.velocity * 0.5 * (go * go + gu * gu);
                const double dv = w.velocity * (go * violation_grad(over) - gu * violation_grad(under));
                gvx += dv * cth;
                gvy += dv * sth;
                gth += dv * (-vx * sth + vy * cth);
            }

            // 台阶入口相邻两段的软速度窗。waypoint i 是段 i 与 i+1 的公共边界。
            for (const auto& entry : ws.step_entry_speed_windows) {
                const double fraction = static_cast<double>(s) / static_cast<double>(S);
                const bool in_incoming_window = seg == entry.waypoint_index
                    && fraction >= 1.0 - params_.step_entry_window_fraction;
                const bool in_outgoing_window = seg == entry.waypoint_index + 1
                    && fraction <= params_.step_entry_window_fraction;
                if (!in_incoming_window && !in_outgoing_window) continue;
                const double over = v_signed - entry.speed_max;
                const double under = entry.speed_min - v_signed;
                const double go = violation(over), gu = violation(under);
                density += w.step_velocity * 0.5 * (go * go + gu * gu);
                const double dv = w.step_velocity
                    * (go * violation_grad(over) - gu * violation_grad(under));
                gvx += dv * cth;
                gvy += dv * sth;
                gth += dv * (-vx * sth + vy * cth);
            }

            // (c) 侧向加速度 |v·ω| ≤ a_lat
            {
                const double a_lat = v_signed * vth;
                const double mag = std::abs(a_lat) - lim.a_lat_max;
                const double gm = violation(mag);
                density += w.lateral_acc * 0.5 * gm * gm;
                const double coeff = w.lateral_acc * gm * violation_grad(mag) * ((a_lat > 0) ? 1.0 : -1.0);
                const double dvs = coeff * vth;
                gvx += dvs * cth;
                gvy += dvs * sth;
                gth += dvs * (-vx * sth + vy * cth);
                gvth += coeff * v_signed;
            }

            // 方向地形约束：车身轴与方向场共线，并满足对应上下行模式的正向速度窗。
            if (ws.direction_map && ws.terrain_constraints
                && (w.step_alignment > 0.0 || w.step_velocity > 0.0)) {
                const Eigen::Vector2d dg = ws.direction_map->map_coord_to_grid({px, py});
                if (ws.direction_map->is_valid_coord(dg)) {
                    const uint8_t label = ws.direction_map->terrain_at(dg);
                    const DirectionMap::DirectionSample direction_sample =
                        ws.direction_map->interpolate_with_gradient(dg);
                    const Eigen::Vector2d& raw_dir = direction_sample.value;
                    if (label >= static_cast<uint8_t>(TerrainType::SLOPE) && raw_dir.norm() > EPS) {
                        const double dir_norm = raw_dir.norm();
                        const Eigen::Vector2d dir = raw_dir / dir_norm;
                        const Eigen::Matrix2d dir_jacobian_map = (
                            (Eigen::Matrix2d::Identity() - dir * dir.transpose()) / dir_norm
                        ) * direction_sample.gradient / ws.direction_map->resolution;
                        const double axis_dot = cth * dir.x() + sth * dir.y();
                        const double cross = cth * dir.y() - sth * dir.x();
                        density += w.step_alignment * 0.5 * cross * cross;
                        gth += w.step_alignment * cross * (-sth * dir.y() - cth * dir.x());
                        const Eigen::Vector2d cross_grad_dir(-sth, cth);
                        const Eigen::Vector2d cross_grad_pos = dir_jacobian_map.transpose() * cross_grad_dir;
                        gpx += w.step_alignment * cross * cross_grad_pos.x();
                        gpy += w.step_alignment * cross * cross_grad_pos.y();

                        const TerrainStepRule* rule = ws.terrain_constraints->selected_mode(label, axis_dot >= 0.0);
                        if (rule) {
                            const double over = v_signed - rule->speed.max;
                            const double under = rule->speed.min - v_signed;
                            const double go = violation(over), gu = violation(under);
                            density += w.step_velocity * 0.5 * (go * go + gu * gu);
                            const double dv = w.step_velocity
                                * (go * violation_grad(over) - gu * violation_grad(under));
                            gvx += dv * cth;
                            gvy += dv * sth;
                            gth += dv * (-vx * sth + vy * cth);
                        }
                    }
                }
            }

            // (d) ω 界 |vth| ≤ omega_max
            {
                const double mag = std::abs(vth) - lim.omega_max;
                const double gm = violation(mag);
                density += w.omega * 0.5 * gm * gm;
                gvth += w.omega * gm * violation_grad(mag) * ((vth > 0) ? 1.0 : -1.0);
            }

            // (e) 加速度界 |a| ≤ acc_max
            {
                const double amag = std::sqrt(ax * ax + ay * ay + EPS);
                const double mag = amag - lim.acc_max;
                const double gm = violation(mag);
                density += w.accel * 0.5 * gm * gm;
                const double coeff = w.accel * gm * violation_grad(mag) / amag;
                gax += coeff * ax;
                gay += coeff * ay;
            }

            // (f) θ̇ 正则
            {
                density += w.theta_rate * 0.5 * vth * vth;
                gvth += w.theta_rate * vth;
            }

            // (g) 朝向跟随：约束车头轴与运动方向共线（模 π），因此前进和倒车都不受误罚。
            if (w.heading_follow > 0.0) {
                const double speed2 = vx * vx + vy * vy;
                constexpr double GATE_SCALE2 = 0.25;
                const double gate_denom = speed2 + GATE_SCALE2;
                const double gate = speed2 / gate_denom;
                const double move_dir = std::atan2(vy, vx);
                const double axis_error = th - move_dir;
                const double herr = 0.5 * std::atan2(
                    std::sin(2.0 * axis_error), std::cos(2.0 * axis_error)
                );
                density += w.heading_follow * 0.5 * gate * herr * herr;
                gth += w.heading_follow * gate * herr;
                if (speed2 > EPS) {
                    gvx += w.heading_follow * gate * herr * (vy / speed2);
                    gvy -= w.heading_follow * gate * herr * (vx / speed2);
                }
                const double gate_grad_scale = 2.0 * GATE_SCALE2 / (gate_denom * gate_denom);
                gvx += 0.5 * w.heading_follow * herr * herr * gate_grad_scale * vx;
                gvy += 0.5 * w.heading_follow * herr * herr * gate_grad_scale * vy;
            }

            // (h) 非完整约束 AL：h = ẋ sinθ − ẏ cosθ = 0
            {
                const double h = vx * sth - vy * cth;
                const double lambda = ws.nonholo_lambda[static_cast<size_t>(global_sample)];
                const double rho = ws.nonholo_rho;
                density += lambda * h + 0.5 * rho * h * h;
                const double coeff = lambda + rho * h;
                gvx += coeff * sth;
                gvy += coeff * (-cth);
                gth += coeff * (vx * cth + vy * sth);
            }

            cost += dt * density;

            // ── 物理量梯度（含 dt）经 β 映射回 grad_c ──
            for (int k = 0; k < NCOEF; ++k) {
                grad_c(c_off + k, 0) += dt * (gpx * b0[k] + gvx * b1[k] + gax * b2[k]);
                grad_c(c_off + k, 1) += dt * (gpy * b0[k] + gvy * b1[k] + gay * b2[k]);
                grad_c(c_off + k, 2) += dt * (gth * b0[k] + gvth * b1[k]);
            }

            // ── 显式 T 梯度：两支 ──
            //   测度：∂(dt·ρ)/∂T|_measure = ρ/S
            //   横坐标漂移：t=T·(s/S)，∂phys/∂T = (s/S)·(dphys/dt)。
            //     dp/dt=(vx,vy), dθ/dt=vth, dv/dt=(ax,ay), da/dt≈(jerk) 忽略（罚不含 a 的更高阶敏感）。
            const double dtdT = static_cast<double>(s) / static_cast<double>(S);
            grad_t_explicit(seg) += density / static_cast<double>(S);
            grad_t_explicit(seg) += dt * dtdT * (
                  gpx * vx + gpy * vy + gth * vth       // ∂p/∂t, ∂θ/∂t
                + gvx * ax + gvy * ay + gvth * ath      // ∂v/∂t, ∂ω/∂t
                + gax * jx + gay * jy                   // ∂a/∂t
            );

            ++global_sample; // 每采样点推进全局索引（与 AL 乘子一一对应）
        }
    }

    return cost;
}

double MincoOptimizer::evaluate(Workspace& ws, const Eigen::VectorXd& vars, Eigen::VectorXd& grad) const {
    const int n = ws.n_segments;
    const int nw = ws.n_waypoints;
    const int time_offset = DIM * nw;
    const int terminal_theta_index = time_offset + n;

    // ── 解包决策变量：[waypoints DIM×nw] + [virtual times n] ──
    for (int i = 0; i < nw; ++i) {
        if (ws.waypoint_is_hard[static_cast<size_t>(i)]) {
            // 硬路点：位置/θ 固定，不从 vars 取。
            ws.waypoints.col(i) = ws.hard_pos_theta[static_cast<size_t>(i)];
        } else {
            ws.waypoints(0, i) = vars(DIM * i + 0);
            ws.waypoints(1, i) = vars(DIM * i + 1);
            ws.waypoints(2, i) = vars(DIM * i + 2);
        }
    }
    for (int i = 0; i < n; ++i) {
        ws.times[static_cast<size_t>(i)] = virtual_to_time(vars(time_offset + i), params_.min_segment_time);
    }
    ws.tail.pos(2) = vars(terminal_theta_index);

    // ── MINCO 求解 ──
    ws.minco.generate(ws.times, ws.head, ws.tail, ws.waypoints);

    // ── 罚 + 梯度 ──
    Eigen::MatrixXd grad_c;
    Eigen::VectorXd grad_t_expl;
    const double cost = accumulate_penalties(ws, grad_c, grad_t_expl);

    // ── 回传到 Q / T ──
    Eigen::Matrix<double, DIM, Eigen::Dynamic> grad_q;
    Eigen::VectorXd grad_t;
    Eigen::Vector3d grad_tail_pos = Eigen::Vector3d::Zero();
    ws.minco.propagate_gradient(grad_c, grad_t_expl, grad_q, grad_t, &grad_tail_pos);

    // ── 打包梯度 ──
    grad.setZero(vars.size());
    for (int i = 0; i < nw; ++i) {
        if (ws.waypoint_is_hard[static_cast<size_t>(i)]) continue; // 硬路点梯度丢弃
        grad(DIM * i + 0) = grad_q(0, i);
        grad(DIM * i + 1) = grad_q(1, i);
        grad(DIM * i + 2) = grad_q(2, i);
    }
    for (int i = 0; i < n; ++i) {
        // 链式：∂/∂tau_v = ∂/∂T · dT/dtau_v
        grad(time_offset + i) = grad_t(i) * virtual_to_time_grad(vars(time_offset + i));
    }
    grad(terminal_theta_index) = grad_tail_pos(2);

    return cost;
}

double MincoOptimizer::update_nonholonomic_multipliers(Workspace& ws) const {
    const auto& coeffs = ws.minco.coefficients();
    const int n = ws.n_segments;
    const int S = std::max(params_.samples_per_segment, 1);
    double max_violation = 0.0;
    int gs = 0;
    for (int seg = 0; seg < n; ++seg) {
        const double T = ws.times[static_cast<size_t>(seg)];
        const int c_off = NCOEF * seg;
        for (int s = 0; s < S; ++s) {
            const double t = T * static_cast<double>(s) / static_cast<double>(S);
            double b0[NCOEF], b1[NCOEF];
            double tp = 1.0;
            for (int k = 0; k < NCOEF; ++k) { b0[k] = tp; tp *= t; }
            b1[0] = 0.0; tp = 1.0;
            for (int k = 1; k < NCOEF; ++k) { b1[k] = k * tp; tp *= t; }
            double th = 0, vx = 0, vy = 0;
            for (int k = 0; k < NCOEF; ++k) {
                th += coeffs(c_off + k, 2) * b0[k];
                vx += coeffs(c_off + k, 0) * b1[k];
                vy += coeffs(c_off + k, 1) * b1[k];
            }
            const double h = vx * std::sin(th) - vy * std::cos(th);
            ws.nonholo_lambda[static_cast<size_t>(gs)] += ws.nonholo_rho * h;
            max_violation = std::max(max_violation, std::abs(h));
            ++gs;
        }
    }
    return max_violation;
}

MincoOptimizer::Result MincoOptimizer::optimize(
    const std::vector<MincoMinJerk::BoundaryPVA>& seed_states,
    const std::vector<double>& seed_durations,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints,
    const std::vector<HardWaypoint>& hard_waypoints,
    const std::vector<StepEntrySpeedWindow>& step_entry_speed_windows
) const {
    Result result;
    const int n = static_cast<int>(seed_durations.size());
    if (n < 1 || static_cast<int>(seed_states.size()) != n + 1) {
        result.error = "seed size mismatch: " + std::to_string(seed_states.size())
                     + " states for " + std::to_string(n) + " segments (need n+1)";
        return result;
    }

    Workspace ws;
    ws.n_segments = n;
    ws.n_waypoints = n - 1;
    ws.head = seed_states.front();
    ws.tail = seed_states.back();
    ws.cost_map = &cost_map;
    ws.direction_map = &direction_map;
    ws.terrain_constraints = &terrain_constraints;
    ws.step_entry_speed_windows = step_entry_speed_windows;
    ws.times = seed_durations;
    ws.waypoints.setZero(DIM, std::max(n - 1, 0));

    // 硬路点表
    ws.waypoint_is_hard.assign(static_cast<size_t>(std::max(n - 1, 0)), false);
    ws.hard_pos_theta.assign(static_cast<size_t>(std::max(n - 1, 0)), Eigen::Vector3d::Zero());
    for (const auto& hw : hard_waypoints) {
        if (hw.waypoint_index >= 0 && hw.waypoint_index < n - 1) {
            ws.waypoint_is_hard[static_cast<size_t>(hw.waypoint_index)] = true;
            ws.hard_pos_theta[static_cast<size_t>(hw.waypoint_index)] =
                Eigen::Vector3d(hw.position.x(), hw.position.y(), hw.theta);
        }
    }

    // 采样点总数（AL 乘子每采样点一个）
    const int total_samples = n * std::max(params_.samples_per_segment, 1);
    ws.nonholo_lambda.assign(static_cast<size_t>(total_samples), 0.0);
    ws.nonholo_rho = params_.nonholonomic_rho_init;

    // ── 初值：内部路点取 seed 中间状态位置，虚拟时间由 seed_durations ──
    for (int i = 0; i < n - 1; ++i) {
        ws.waypoints.col(i) = seed_states[static_cast<size_t>(i + 1)].pos;
        if (ws.waypoint_is_hard[static_cast<size_t>(i)]) {
            ws.waypoints.col(i) = ws.hard_pos_theta[static_cast<size_t>(i)];
        }
    }

    const int nw = n - 1;
    const int time_offset = DIM * nw;
    const int terminal_theta_index = time_offset + n;
    Eigen::VectorXd vars(DIM * nw + n + 1);
    for (int i = 0; i < nw; ++i) {
        vars(DIM * i + 0) = ws.waypoints(0, i);
        vars(DIM * i + 1) = ws.waypoints(1, i);
        vars(DIM * i + 2) = ws.waypoints(2, i);
    }
    for (int i = 0; i < n; ++i) {
        vars(time_offset + i) = time_to_virtual(seed_durations[static_cast<size_t>(i)], params_.min_segment_time);
    }
    vars(terminal_theta_index) = ws.tail.pos(2);

    LbfgsMinimizer::Options lopt;
    lopt.max_iterations = params_.max_iterations;
    LbfgsMinimizer solver(lopt);

    Eigen::VectorXd scratch_grad(vars.size());
    auto cost_fn = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
        return evaluate(ws, x, g);
    };

    // ── 梯度自检：解析 vs 中心差分（诊断用）──
    if (params_.debug_check_gradient) {
        Eigen::VectorXd g_analytic(vars.size());
        evaluate(ws, vars, g_analytic);
        double max_abs_err = 0.0;
        double max_rel_err = 0.0;
        const double h = 1e-6;
        Eigen::VectorXd dummy(vars.size());
        for (int i = 0; i < vars.size(); ++i) {
            Eigen::VectorXd vp = vars, vm = vars;
            vp(i) += h;
            vm(i) -= h;
            const double cp = evaluate(ws, vp, dummy);
            const double cm = evaluate(ws, vm, dummy);
            const double num = (cp - cm) / (2.0 * h);
            const double abs_err = std::abs(num - g_analytic(i));
            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
                result.grad_check_worst_index = i;
            }
            max_rel_err = std::max(max_rel_err, abs_err / (std::abs(num) + 1.0));
        }
        result.grad_check_max_abs_err = max_abs_err;
        result.grad_check_max_rel_err = max_rel_err;
        result.grad_check_num_time_vars = n;
    }

    // ── 增广拉格朗日外层：优化 → 更新乘子 lambda += rho·h → 放大 rho ──
    LbfgsMinimizer::Result lr;
    double max_violation = 0.0;
    double last_good_violation = std::numeric_limits<double>::infinity();
    Eigen::VectorXd last_good_vars = vars;
    bool have_good_iterate = false;
    for (int round = 0; round < std::max(params_.nonholonomic_al_rounds, 1); ++round) {
        lr = solver.minimize(cost_fn, vars);

        result.al_rounds = round + 1;
        result.iterations += lr.iterations;
        result.line_search_iterations += lr.line_search_iterations;
        result.final_grad_inf_norm = lr.grad_inf_norm;
        result.final_rho = ws.nonholo_rho;

        if (lr.status == LbfgsMinimizer::Status::LINE_SEARCH_FAILED) {
            if (!have_good_iterate) {
                result.error = "L-BFGS line search failed in AL round " + std::to_string(round + 1)
                    + " (rho=" + std::to_string(ws.nonholo_rho)
                    + ", cost=" + std::to_string(lr.cost)
                    + ", |grad|_inf=" + std::to_string(lr.grad_inf_norm) + ")";
                return result;
            }
            vars = last_good_vars;
            max_violation = last_good_violation;
            break;
        }

        // 用收敛解刷新 ws（times/waypoints/minco），再逐采样点算非完整违反 h 更新乘子。
        evaluate(ws, vars, scratch_grad);
        max_violation = update_nonholonomic_multipliers(ws);
        have_good_iterate = true;
        last_good_vars = vars;
        last_good_violation = max_violation;

        if (max_violation <= params_.nonholonomic_tolerance) break;
        ws.nonholo_rho = std::min(ws.nonholo_rho * params_.nonholonomic_rho_scale, params_.nonholonomic_rho_max);
    }

    // 最终解：确保 ws 反映最终 vars。
    evaluate(ws, vars, scratch_grad);
    result.trajectory = ws.minco.to_trajectory();
    result.cost = lr.cost;
    result.max_nonholonomic_violation = max_violation;
    if (!std::isfinite(max_violation)) {
        result.error = "non-finite nonholonomic violation";
        return result;
    }
    if (max_violation > params_.nonholonomic_acceptance_tolerance) {
        result.error = "nonholonomic constraint exceeds acceptance limit: max_violation="
            + std::to_string(max_violation) + ", target="
            + std::to_string(params_.nonholonomic_tolerance) + ", acceptance_limit="
            + std::to_string(params_.nonholonomic_acceptance_tolerance);
        return result;
    }
    result.success = true;
    return result;
}

} // namespace nav_executor
