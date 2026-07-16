#include <nav_executor/path_planner/minco_optimizer.hpp>

#include <algorithm>
#include <cmath>

#include <nav_executor/path_planner/lbfgs_minimizer.hpp>

namespace nav_executor {

namespace {

constexpr int DIM = MincoMinJerk::DIM;
constexpr int NCOEF = MincoMinJerk::NCOEF;
constexpr double EPS = 1e-9;
// ω 分母速度正则尺度（与 MincoTrajectory 一致，保证优化目标与跟随参考的 ω 定义相同）。
constexpr double OMEGA_SPEED_REG = 0.05;
constexpr double OMEGA_SPEED_REG_SQ = OMEGA_SPEED_REG * OMEGA_SPEED_REG;

// 平滑 relu 及其导数（约束罚：违反量 g>0 时二次惩罚）。
inline double violation(double g) { return std::max(g, 0.0); }
inline double violation_grad(double g) { return g > 0.0 ? 1.0 : 0.0; }

// C1 光滑门控 smoothstep：x<lo → 0，x>hi → 1，区间内 3t²−2t³ 过渡。
// 返回 {值, d值/dx}。用于把方向地形罚在台阶边界连续开关，消除代价断崖。
inline std::pair<double, double> smoothstep_gate(double x, double lo, double hi) {
    if (hi <= lo) return {x >= hi ? 1.0 : 0.0, 0.0};
    const double t = (x - lo) / (hi - lo);
    if (t <= 0.0) return {0.0, 0.0};
    if (t >= 1.0) return {1.0, 0.0};
    const double value = t * t * (3.0 - 2.0 * t);
    const double dvalue = 6.0 * t * (1.0 - t) / (hi - lo);
    return {value, dvalue};
}

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

    std::vector<double> gears;          // 每段 ±1
    std::vector<char> cusp;             // 每内部节点是否尖点（size n-1）

    // 硬中间路点：index → 固定 pos（2D）。速度不固定。
    std::vector<bool> waypoint_is_hard;
    std::vector<Eigen::Vector2d> hard_pos;
    std::vector<MincoOptimizer::StepEntrySpeedWindow> step_entry_speed_windows;

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
    Eigen::VectorXd& grad_t_explicit,
    CostTerms* terms
) const {
    const auto& coeffs = ws.minco.coefficients();
    const int n = ws.n_segments;
    const auto& w = params_.weights;
    const auto& lim = params_.limits;
    constexpr double SPEED_EPS = 1e-4;

    double cost = 0.0;
    grad_c.setZero(NCOEF * n, DIM);
    grad_t_explicit.setZero(n);
    if (terms) *terms = CostTerms {};

    for (int seg = 0; seg < n; ++seg) {
        const double T = ws.times[static_cast<size_t>(seg)];
        const int c_off = NCOEF * seg;
        const double gear = ws.gears[static_cast<size_t>(seg)];

        // ── min-jerk 能量（每维 x,y 独立）──
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
            if (terms) terms->energy += w.energy * energy;
            grad_c(c_off + 3, d) += w.energy * (72.0 * c3 * T + 144.0 * c4 * T2 + 240.0 * c5 * T3);
            grad_c(c_off + 4, d) += w.energy * (144.0 * c3 * T2 + 384.0 * c4 * T3 + 720.0 * c5 * T4);
            grad_c(c_off + 5, d) += w.energy * (240.0 * c3 * T3 + 720.0 * c4 * T4 + 1440.0 * c5 * T5);
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
        if (terms) terms->time += w.time * T;
        grad_t_explicit(seg) += w.time;

        // ── 采样点罚 ──
        const int S = std::max(params_.samples_per_segment, 1);
        for (int s = 0; s < S; ++s) {
            const double t = T * static_cast<double>(s) / static_cast<double>(S);
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
            // 平坦物理量（对真实时间 t）
            double px = 0, py = 0;
            double vx = 0, vy = 0;
            double ax = 0, ay = 0;
            double jx = 0, jy = 0;
            for (int k = 0; k < NCOEF; ++k) {
                px += coeffs(c_off + k, 0) * b0[k];
                py += coeffs(c_off + k, 1) * b0[k];
                vx += coeffs(c_off + k, 0) * b1[k];
                vy += coeffs(c_off + k, 1) * b1[k];
                ax += coeffs(c_off + k, 0) * b2[k];
                ay += coeffs(c_off + k, 1) * b2[k];
                jx += coeffs(c_off + k, 0) * b3[k];
                jy += coeffs(c_off + k, 1) * b3[k];
            }

            const double dt = T / static_cast<double>(S); // 积分测度

            // 密度 ρ 及其对 (px,py,vx,vy,ax,ay) 的梯度。
            double density = 0.0;
            double gpx = 0, gpy = 0, gvx = 0, gvy = 0, gax = 0, gay = 0;
            // 分项密度（诊断用，terms 非空时按 dt 加权累积到 terms）。
            double d_obstacle = 0, d_velocity = 0, d_lateral_acc = 0;
            double d_omega = 0, d_accel = 0, d_step_alignment = 0, d_step_velocity = 0;

            const double speed2 = vx * vx + vy * vy;
            const double speed = std::sqrt(speed2);
            const double omega_den = speed2 + OMEGA_SPEED_REG_SQ;
            const double cross = vx * ay - vy * ax;
            const double omega = cross / omega_den;
            const double v_signed = gear * speed;

            // (a) 障碍罚：双线性插值 map 坐标梯度。
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
                        (1.0 - tx) * (1.0 - ty) * c00 + tx * (1.0 - ty) * c10
                        + (1.0 - tx) * ty * c01 + tx * ty * c11
                    ) / 255.0;
                    const double inv_scale = 1.0 / (255.0 * ws.cost_map->resolution);
                    const Eigen::Vector2d cgrad(
                        ((1.0 - ty) * (c10 - c00) + ty * (c11 - c01)) * inv_scale,
                        ((1.0 - tx) * (c01 - c00) + tx * (c11 - c10)) * inv_scale
                    );
                    density += w.obstacle * 0.5 * cval * cval;
                    if (terms) d_obstacle += w.obstacle * 0.5 * cval * cval;
                    gpx += w.obstacle * cval * cgrad.x();
                    gpy += w.obstacle * cval * cgrad.y();
                } else {
                    const double x_min = ws.cost_map->origin_x;
                    const double y_min = ws.cost_map->origin_y;
                    const double x_max = x_min + static_cast<double>(ws.cost_map->width - 1) * ws.cost_map->resolution;
                    const double y_max = y_min + static_cast<double>(ws.cost_map->height - 1) * ws.cost_map->resolution;
                    const Eigen::Vector2d p_map(px, py);
                    const Eigen::Vector2d clamped(std::clamp(px, x_min, x_max), std::clamp(py, y_min, y_max));
                    const Eigen::Vector2d delta = p_map - clamped;
                    const double dist = std::sqrt(delta.squaredNorm() + EPS);
                    const double cval = 1.0 + dist / ws.cost_map->resolution;
                    const Eigen::Vector2d cgrad = delta / (dist * ws.cost_map->resolution);
                    density += w.obstacle * 0.5 * cval * cval;
                    if (terms) d_obstacle += w.obstacle * 0.5 * cval * cval;
                    gpx += w.obstacle * cval * cgrad.x();
                    gpy += w.obstacle * cval * cgrad.y();
                }
            }

            // ∂v_signed/∂(vx,vy) = gear·(vx,vy)/speed（speed>eps 时有效）。
            const bool speed_ok = speed > SPEED_EPS;
            const double dvs_dvx = speed_ok ? gear * vx / speed : 0.0;
            const double dvs_dvy = speed_ok ? gear * vy / speed : 0.0;

            // (b) 带符号速度窗：v ≤ vel_max, v ≥ vel_min
            {
                const double over = v_signed - lim.vel_max;
                const double under = lim.vel_min - v_signed;
                const double go = violation(over), gu = violation(under);
                density += w.velocity * 0.5 * (go * go + gu * gu);
                if (terms) d_velocity += w.velocity * 0.5 * (go * go + gu * gu);
                const double dv = w.velocity * (go - gu); // dρ/dv_signed
                gvx += dv * dvs_dvx;
                gvy += dv * dvs_dvy;
            }

            // 台阶入口相邻两段的软速度窗（带符号 v）。
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
                if (terms) d_step_velocity += w.step_velocity * 0.5 * (go * go + gu * gu);
                const double dv = w.step_velocity * (go - gu);
                gvx += dv * dvs_dvx;
                gvy += dv * dvs_dvy;
            }

            // ∂ω/∂(vx,vy,ax,ay)
            const double inv_den = 1.0 / omega_den;
            const double domega_dvx = ay * inv_den - cross * 2.0 * vx * inv_den * inv_den;
            const double domega_dvy = -ax * inv_den - cross * 2.0 * vy * inv_den * inv_den;
            const double domega_dax = -vy * inv_den;
            const double domega_day = vx * inv_den;

            // (c) 侧向加速度 |v·ω| ≤ a_lat
            {
                const double a_lat = v_signed * omega;
                const double mag = std::abs(a_lat) - lim.a_lat_max;
                const double gm = violation(mag);
                density += w.lateral_acc * 0.5 * gm * gm;
                if (terms) d_lateral_acc += w.lateral_acc * 0.5 * gm * gm;
                const double coeff = w.lateral_acc * gm * ((a_lat > 0) ? 1.0 : -1.0);
                // ∂a_lat/∂· = ω·∂v_signed/∂· + v_signed·∂ω/∂·
                gvx += coeff * (omega * dvs_dvx + v_signed * domega_dvx);
                gvy += coeff * (omega * dvs_dvy + v_signed * domega_dvy);
                gax += coeff * (v_signed * domega_dax);
                gay += coeff * (v_signed * domega_day);
            }

            // (d) ω 界 |ω| ≤ omega_max
            {
                const double mag = std::abs(omega) - lim.omega_max;
                const double gm = violation(mag);
                density += w.omega * 0.5 * gm * gm;
                if (terms) d_omega += w.omega * 0.5 * gm * gm;
                const double coeff = w.omega * gm * ((omega > 0) ? 1.0 : -1.0);
                gvx += coeff * domega_dvx;
                gvy += coeff * domega_dvy;
                gax += coeff * domega_dax;
                gay += coeff * domega_day;
            }

            // (e) 加速度界 |a| ≤ acc_max
            {
                const double amag = std::sqrt(ax * ax + ay * ay + EPS);
                const double mag = amag - lim.acc_max;
                const double gm = violation(mag);
                density += w.accel * 0.5 * gm * gm;
                if (terms) d_accel += w.accel * 0.5 * gm * gm;
                const double coeff = w.accel * gm / amag;
                gax += coeff * ax;
                gay += coeff * ay;
            }

            // 方向地形约束：运动方向与方向场共线（模 π），并满足所选模式速度窗。
            // 门控 gate=smoothstep(‖dir‖) 连续开关，替代 label/阈值硬开关，保证目标沿采样点
            // 位移分段光滑（strong Wolfe 前提）。gate 依赖位置，其梯度按乘积法则流回 gpx/gpy。
            if (ws.direction_map && ws.terrain_constraints && speed_ok
                && (w.step_alignment > 0.0 || w.step_velocity > 0.0)) {
                const Eigen::Vector2d dg = ws.direction_map->map_coord_to_grid({px, py});
                if (ws.direction_map->is_valid_coord(dg)) {
                    const uint8_t label = ws.direction_map->terrain_at(dg);
                    const DirectionMap::DirectionSample direction_sample =
                        ws.direction_map->interpolate_with_gradient(dg);
                    const Eigen::Vector2d& raw_dir = direction_sample.value;
                    const double dir_norm = raw_dir.norm();
                    // 门控只在有方向语义（label>=SLOPE）的地形上开启；标签本身离散，但其切换发生在
                    // ‖dir‖→0 的区域（gate 已连续压到 0），故不引入代价跳变。
                    const bool directional = label >= static_cast<uint8_t>(TerrainType::SLOPE);
                    const auto [gate, dgate_dnorm] = directional
                        ? smoothstep_gate(dir_norm, params_.terrain_gate.norm_lo, params_.terrain_gate.norm_hi)
                        : std::pair<double, double> {0.0, 0.0};
                    if (gate > 0.0 && dir_norm > EPS) {
                        const Eigen::Vector2d dir = raw_dir / dir_norm;
                        const Eigen::Matrix2d dir_jacobian_map = (
                            (Eigen::Matrix2d::Identity() - dir * dir.transpose()) / dir_norm
                        ) * direction_sample.gradient / ws.direction_map->resolution;
                        // ∂‖dir‖/∂pos = dirᵀ·(∂raw_dir/∂pos)；∂raw_dir/∂pos = gradient/resolution（map系）。
                        const Eigen::Vector2d dnorm_dpos =
                            (direction_sample.gradient.transpose() * dir) / ws.direction_map->resolution;
                        const Eigen::Vector2d dgate_dpos = dgate_dnorm * dnorm_dpos;

                        // 门控内累积的方向地形密度（用于把 gate 的位置梯度按乘积法则回传）。
                        double terrain_density = 0.0;

                        // (1) 对齐罚：运动方向单位向量 u=(vx,vy)/speed；e = u×dir。
                        const double ux = vx / speed, uy = vy / speed;
                        const double e = ux * dir.y() - uy * dir.x();
                        const double align_density = w.step_alignment * 0.5 * e * e;
                        terrain_density += align_density;
                        density += gate * align_density;
                        if (terms) d_step_alignment += gate * align_density;
                        const double de = gate * w.step_alignment * e;
                        const double inv_s = 1.0 / speed;
                        const double inv_s3 = inv_s * inv_s * inv_s;
                        const double dux_dvx = (speed2 - vx * vx) * inv_s3;
                        const double dux_dvy = -vx * vy * inv_s3;
                        const double duy_dvx = -vx * vy * inv_s3;
                        const double duy_dvy = (speed2 - vy * vy) * inv_s3;
                        gvx += de * (dir.y() * dux_dvx - dir.x() * duy_dvx);
                        gvy += de * (dir.y() * dux_dvy - dir.x() * duy_dvy);
                        // ∂e/∂dir = (−uy, ux)；dir 依赖位置（经 dir_jacobian_map）。
                        const Eigen::Vector2d de_ddir(-uy, ux);
                        const Eigen::Vector2d de_dpos = dir_jacobian_map.transpose() * de_ddir;
                        gpx += de * de_dpos.x();
                        gpy += de * de_dpos.y();

                        // (2) 速度窗罚（所选模式）。
                        const TerrainStepRule* rule = ws.terrain_constraints->selected_mode(label, gear >= 0.0);
                        if (rule) {
                            const double over = v_signed - rule->speed.max;
                            const double under = rule->speed.min - v_signed;
                            const double go = violation(over), gu = violation(under);
                            const double vel_density = w.step_velocity * 0.5 * (go * go + gu * gu);
                            terrain_density += vel_density;
                            density += gate * vel_density;
                            if (terms) d_step_velocity += gate * vel_density;
                            const double dv = gate * w.step_velocity * (go - gu);
                            gvx += dv * dvs_dvx;
                            gvy += dv * dvs_dvy;
                        }

                        // gate 对位置的梯度（乘积法则）：∂(gate·terrain_density)/∂pos ⊃ terrain_density·∂gate/∂pos。
                        gpx += terrain_density * dgate_dpos.x();
                        gpy += terrain_density * dgate_dpos.y();
                    }
                }
            }

            cost += dt * density;
            if (terms) {
                terms->obstacle += dt * d_obstacle;
                terms->velocity += dt * d_velocity;
                terms->lateral_acc += dt * d_lateral_acc;
                terms->omega += dt * d_omega;
                terms->accel += dt * d_accel;
                terms->step_alignment += dt * d_step_alignment;
                terms->step_velocity += dt * d_step_velocity;
            }

            // 物理量梯度（含 dt）经 β 映射回 grad_c。
            for (int k = 0; k < NCOEF; ++k) {
                grad_c(c_off + k, 0) += dt * (gpx * b0[k] + gvx * b1[k] + gax * b2[k]);
                grad_c(c_off + k, 1) += dt * (gpy * b0[k] + gvy * b1[k] + gay * b2[k]);
            }

            // 显式 T 梯度：测度 ρ/S + 横坐标漂移 dt·(s/S)·dρ/dt。
            const double dtdT = static_cast<double>(s) / static_cast<double>(S);
            grad_t_explicit(seg) += density / static_cast<double>(S);
            grad_t_explicit(seg) += dt * dtdT * (
                  gpx * vx + gpy * vy
                + gvx * ax + gvy * ay
                + gax * jx + gay * jy
            );
        }
    }

    return cost;
}

double MincoOptimizer::evaluate(Workspace& ws, const Eigen::VectorXd& vars, Eigen::VectorXd& grad) const {
    const int n = ws.n_segments;
    const int nw = ws.n_waypoints;
    const int time_offset = DIM * nw;

    // 解包决策变量：[waypoints DIM×nw] + [virtual times n]
    for (int i = 0; i < nw; ++i) {
        if (ws.waypoint_is_hard[static_cast<size_t>(i)]) {
            ws.waypoints.col(i) = ws.hard_pos[static_cast<size_t>(i)];
        } else {
            ws.waypoints(0, i) = vars(DIM * i + 0);
            ws.waypoints(1, i) = vars(DIM * i + 1);
        }
    }
    for (int i = 0; i < n; ++i) {
        ws.times[static_cast<size_t>(i)] = virtual_to_time(vars(time_offset + i), params_.min_segment_time);
    }

    ws.minco.generate(ws.times, ws.head, ws.tail, ws.waypoints, ws.cusp);

    Eigen::MatrixXd grad_c;
    Eigen::VectorXd grad_t_expl;
    const double cost = accumulate_penalties(ws, grad_c, grad_t_expl);

    Eigen::Matrix<double, DIM, Eigen::Dynamic> grad_q;
    Eigen::VectorXd grad_t;
    ws.minco.propagate_gradient(grad_c, grad_t_expl, grad_q, grad_t);

    grad.setZero(vars.size());
    for (int i = 0; i < nw; ++i) {
        if (ws.waypoint_is_hard[static_cast<size_t>(i)]) continue;
        grad(DIM * i + 0) = grad_q(0, i);
        grad(DIM * i + 1) = grad_q(1, i);
    }
    for (int i = 0; i < n; ++i) {
        grad(time_offset + i) = grad_t(i) * virtual_to_time_grad(vars(time_offset + i));
    }

    return cost;
}

MincoOptimizer::Result MincoOptimizer::optimize(
    const std::vector<MincoMinJerk::BoundaryPVA>& seed_states,
    const std::vector<double>& seed_durations,
    const std::vector<double>& seed_gears,
    const std::vector<char>& cusp_waypoints,
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
    if (static_cast<int>(seed_gears.size()) != n) {
        result.error = "seed gears size mismatch: " + std::to_string(seed_gears.size())
                     + " for " + std::to_string(n) + " segments";
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
    ws.gears = seed_gears;
    ws.cusp.assign(static_cast<size_t>(std::max(n - 1, 0)), 0);
    for (int i = 0; i < n - 1 && i < static_cast<int>(cusp_waypoints.size()); ++i) {
        ws.cusp[static_cast<size_t>(i)] = cusp_waypoints[static_cast<size_t>(i)];
    }
    ws.waypoints.setZero(DIM, std::max(n - 1, 0));

    // 硬路点表（仅固定 2D 位置）。
    ws.waypoint_is_hard.assign(static_cast<size_t>(std::max(n - 1, 0)), false);
    ws.hard_pos.assign(static_cast<size_t>(std::max(n - 1, 0)), Eigen::Vector2d::Zero());
    for (const auto& hw : hard_waypoints) {
        if (hw.waypoint_index >= 0 && hw.waypoint_index < n - 1) {
            ws.waypoint_is_hard[static_cast<size_t>(hw.waypoint_index)] = true;
            ws.hard_pos[static_cast<size_t>(hw.waypoint_index)] = hw.position;
        }
    }

    // 初值：内部路点取 seed 中间状态位置，虚拟时间由 seed_durations。
    for (int i = 0; i < n - 1; ++i) {
        ws.waypoints.col(i) = seed_states[static_cast<size_t>(i + 1)].pos;
        if (ws.waypoint_is_hard[static_cast<size_t>(i)]) {
            ws.waypoints.col(i) = ws.hard_pos[static_cast<size_t>(i)];
        }
    }

    const int nw = n - 1;
    const int time_offset = DIM * nw;
    Eigen::VectorXd vars(DIM * nw + n);
    for (int i = 0; i < nw; ++i) {
        vars(DIM * i + 0) = ws.waypoints(0, i);
        vars(DIM * i + 1) = ws.waypoints(1, i);
    }
    for (int i = 0; i < n; ++i) {
        vars(time_offset + i) = time_to_virtual(seed_durations[static_cast<size_t>(i)], params_.min_segment_time);
    }

    LbfgsMinimizer::Options lopt;
    lopt.max_iterations = params_.max_iterations;
    LbfgsMinimizer solver(lopt);

    auto cost_fn = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
        return evaluate(ws, x, g);
    };

    // 梯度自检：解析 vs 中心差分（诊断用）。
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

    // 收敛诊断：记录种子处分项代价、初始梯度范数、自由路点初值位置。
    Eigen::VectorXd seed_vars;
    if (params_.debug_diagnostics) {
        seed_vars = vars;
        Eigen::VectorXd g_seed(vars.size());
        evaluate(ws, vars, g_seed); // 重建 ws 至种子解
        Eigen::MatrixXd grad_c_dbg;
        Eigen::VectorXd grad_t_dbg;
        accumulate_penalties(ws, grad_c_dbg, grad_t_dbg, &result.seed_costs);
        result.initial_grad_inf_norm = g_seed.lpNorm<Eigen::Infinity>();
    }

    const LbfgsMinimizer::Result lr = solver.minimize(cost_fn, vars);
    result.iterations = lr.iterations;
    result.line_search_iterations = lr.line_search_iterations;
    result.final_grad_inf_norm = lr.grad_inf_norm;
    if (lr.status == LbfgsMinimizer::Status::LINE_SEARCH_FAILED && lr.iterations == 0) {
        result.error = "L-BFGS line search failed at start (cost=" + std::to_string(lr.cost)
            + ", |grad|_inf=" + std::to_string(lr.grad_inf_norm) + ")";
        return result;
    }

    // 最终解：确保 ws 反映最终 vars。
    Eigen::VectorXd scratch_grad(vars.size());
    evaluate(ws, vars, scratch_grad);
    result.trajectory = ws.minco.to_trajectory(ws.gears);
    result.cost = lr.cost;
    result.success = true;

    // 收敛诊断：最终分项代价 + 自由路点位移（相对种子）。
    if (params_.debug_diagnostics) {
        Eigen::MatrixXd grad_c_dbg;
        Eigen::VectorXd grad_t_dbg;
        accumulate_penalties(ws, grad_c_dbg, grad_t_dbg, &result.final_costs);
        // 最终梯度按类别拆分：前 DIM*nw 为位置变量，其余 n 为时间变量。
        double g_pos = 0.0, g_time = 0.0;
        for (int i = 0; i < time_offset; ++i) g_pos = std::max(g_pos, std::abs(scratch_grad(i)));
        for (int i = time_offset; i < scratch_grad.size(); ++i) g_time = std::max(g_time, std::abs(scratch_grad(i)));
        result.final_grad_pos_inf_norm = g_pos;
        result.final_grad_time_inf_norm = g_time;
        result.lbfgs_status = static_cast<int>(lr.status);
        double disp_sum = 0.0;
        double disp_max = 0.0;
        int free_count = 0;
        for (int i = 0; i < nw; ++i) {
            if (ws.waypoint_is_hard[static_cast<size_t>(i)]) continue;
            const double dx = vars(DIM * i + 0) - seed_vars(DIM * i + 0);
            const double dy = vars(DIM * i + 1) - seed_vars(DIM * i + 1);
            const double d = std::hypot(dx, dy);
            disp_sum += d;
            disp_max = std::max(disp_max, d);
            ++free_count;
        }
        result.waypoint_total_displacement = disp_sum;
        result.waypoint_max_displacement = disp_max;
        result.free_waypoint_count = free_count;
        result.diagnostics_valid = true;
    }
    return result;
}

} // namespace nav_executor
