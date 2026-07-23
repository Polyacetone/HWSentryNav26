#include <nav_executor/path_planner/minco_optimizer.hpp>

#include <algorithm>
#include <array>
#include <cmath>

#include <nav_executor/path_planner/lbfgs_minimizer.hpp>

namespace nav_executor {

namespace {

constexpr int DIM = MincoMinJerk::DIM;
constexpr int NCOEF = MincoMinJerk::NCOEF;
constexpr double EPS = 1e-9;
// 与轨迹求值使用同一角速度正则，保证优化目标和跟随参考一致。
constexpr double OMEGA_SPEED_REG = 0.05;
constexpr double OMEGA_SPEED_REG_SQ = OMEGA_SPEED_REG * OMEGA_SPEED_REG;

inline double violation(double g) { return std::max(g, 0.0); }

// 用 C1 过渡启停地形罚，避免台阶边界出现代价跳变。
inline std::pair<double, double> smoothstep_gate(double x, double lo, double hi) {
    if (hi <= lo) return {x >= hi ? 1.0 : 0.0, 0.0};
    const double t = (x - lo) / (hi - lo);
    if (t <= 0.0) return {0.0, 0.0};
    if (t >= 1.0) return {1.0, 0.0};
    const double value = t * t * (3.0 - 2.0 * t);
    const double dvalue = 6.0 * t * (1.0 - t) / (hi - lo);
    return {value, dvalue};
}

// 在助跑范围外以 C1 过渡衰减。
inline std::pair<double, double> runup_distance_gate(
    const double distance, const double radius, const double transition
) {
    if (distance <= radius) return {1.0, 0.0};
    if (transition <= 0.0 || distance >= radius + transition) return {0.0, 0.0};
    const double x = (radius + transition - distance) / transition;
    const double value = x * x * (3.0 - 2.0 * x);
    const double derivative = -6.0 * x * (1.0 - x) / transition;
    return {value, derivative};
}

struct SmoothMotion {
    double speed = 0.0;
    Eigen::Vector2d speed_gradient = Eigen::Vector2d::Zero();
    Eigen::Vector2d direction = Eigen::Vector2d::Zero();
    Eigen::Matrix2d direction_jacobian = Eigen::Matrix2d::Zero();
};

SmoothMotion smooth_motion(const Eigen::Vector2d& velocity, const double speed_scale) {
    const double speed_squared = velocity.squaredNorm();
    const double scale_squared = speed_scale * speed_scale;
    const double denominator_squared = speed_squared + scale_squared;
    const double denominator = std::sqrt(denominator_squared);
    const double denominator_cubed = denominator_squared * denominator;

    // s=q/sqrt(D): ds/dv=(q+2v0²)v/D^(3/2)；u=v/sqrt(D): du/dv=I/sqrt(D)-vvᵀ/D^(3/2)。
    SmoothMotion result;
    result.speed = speed_squared / denominator;
    result.speed_gradient = velocity * (
        (speed_squared + 2.0 * scale_squared) / denominator_cubed
    );
    result.direction = velocity / denominator;
    result.direction_jacobian = Eigen::Matrix2d::Identity() / denominator
        - velocity * velocity.transpose() / denominator_cubed;
    return result;
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
    return sp > 30.0 ? sp : std::log(std::expm1(sp));
}

} // anonymous namespace

// 单次优化复用的工作区。
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
    const auto& lim = params_.trajectory_limits;

    double cost = 0.0;
    grad_c.setZero(NCOEF * n, DIM);
    grad_t_explicit.setZero(n);
    if (terms) *terms = CostTerms {};

    struct RunupSource {
        double value = 0.0;
        double radius = 0.0;
        Eigen::Vector2d position_gradient = Eigen::Vector2d::Zero();
    };
    struct LookaheadSample {
        int segment = 0;
        double local_time = 0.0;
        double dt = 0.0;
        double sample_time_fraction = 0.0;
        Eigen::Vector2d position = Eigen::Vector2d::Zero();
        Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
        Eigen::Vector2d acceleration = Eigen::Vector2d::Zero();
        SmoothMotion motion;
        double omega = 0.0;
        double arc_measure = 0.0;
        uint8_t terrain_label = static_cast<uint8_t>(TerrainType::FLAT);
        double direction_norm = 0.0;
        Eigen::Vector2d terrain_direction = Eigen::Vector2d::Zero();
        Eigen::Matrix2d terrain_direction_jacobian = Eigen::Matrix2d::Zero();
        double terrain_gate = 0.0;
        Eigen::Vector2d terrain_gate_position_gradient = Eigen::Vector2d::Zero();
        std::array<RunupSource, TERRAIN_LABEL_COUNT - 2> sources;
        int source_count = 0;
    };

    const int samples_per_segment = std::max(params_.samples_per_segment, 1);
    std::vector<LookaheadSample> lookahead_samples;
    lookahead_samples.reserve(static_cast<size_t>(n * samples_per_segment));

    // 方向场幅值决定连续强度，双线性标签权重只用于 runup source。主 profile 仍取包含
    // 采样点的单个栅格 label：跨 label 重合属于地图异常并由 map_server 聚合告警，本阶段
    // 不引入 per-label 连续混合；典型单 label 外边界在 terrain gate 开启前已衰减掉。
    for (int seg = 0; seg < n; ++seg) {
        const double T = ws.times[static_cast<size_t>(seg)];
        const int c_off = NCOEF * seg;
        for (int sample_index = 0; sample_index < samples_per_segment; ++sample_index) {
            const double fraction = static_cast<double>(sample_index)
                / static_cast<double>(samples_per_segment);
            const double t = T * fraction;
            double b0[NCOEF], b1[NCOEF], b2[NCOEF];
            double tp = 1.0;
            for (int k = 0; k < NCOEF; ++k) { b0[k] = tp; tp *= t; }
            b1[0] = 0.0;
            tp = 1.0;
            for (int k = 1; k < NCOEF; ++k) { b1[k] = k * tp; tp *= t; }
            b2[0] = 0.0;
            b2[1] = 0.0;
            tp = 1.0;
            for (int k = 2; k < NCOEF; ++k) { b2[k] = k * (k - 1) * tp; tp *= t; }

            LookaheadSample sample;
            sample.segment = seg;
            sample.local_time = t;
            sample.dt = T / static_cast<double>(samples_per_segment);
            sample.sample_time_fraction = fraction;
            for (int k = 0; k < NCOEF; ++k) {
                sample.position.x() += coeffs(c_off + k, 0) * b0[k];
                sample.position.y() += coeffs(c_off + k, 1) * b0[k];
                sample.velocity.x() += coeffs(c_off + k, 0) * b1[k];
                sample.velocity.y() += coeffs(c_off + k, 1) * b1[k];
                sample.acceleration.x() += coeffs(c_off + k, 0) * b2[k];
                sample.acceleration.y() += coeffs(c_off + k, 1) * b2[k];
            }
            sample.motion = smooth_motion(
                sample.velocity, params_.terrain_gate.motion_speed_scale
            );
            sample.arc_measure = sample.dt * sample.motion.speed;
            const double speed2 = sample.velocity.squaredNorm();
            sample.omega = (
                sample.velocity.x() * sample.acceleration.y()
                - sample.velocity.y() * sample.acceleration.x()
            ) / (speed2 + OMEGA_SPEED_REG_SQ);

            if (ws.direction_map && ws.terrain_constraints) {
                const Eigen::Vector2d grid = ws.direction_map->map_coord_to_grid(sample.position);
                if (ws.direction_map->is_valid_coord(grid)) {
                    const auto direction_sample = ws.direction_map->interpolate_with_gradient(grid);
                    const double direction_norm = direction_sample.value.norm();
                    sample.terrain_label = ws.direction_map->terrain_at(grid);
                    sample.direction_norm = direction_norm;
                    if (direction_norm > EPS) {
                        const Eigen::Vector2d direction = direction_sample.value / direction_norm;
                        sample.terrain_direction = direction;
                        sample.terrain_direction_jacobian = (
                            (Eigen::Matrix2d::Identity()
                                - direction * direction.transpose()) / direction_norm
                        ) * direction_sample.gradient / ws.direction_map->resolution;
                        const Eigen::Vector2d dnorm_dpos =
                            (direction_sample.gradient.transpose() * direction)
                            / ws.direction_map->resolution;
                        const auto [terrain_gate, dterrain_gate_dnorm] = smoothstep_gate(
                            direction_norm,
                            params_.terrain_gate.norm_lo,
                            params_.terrain_gate.norm_hi
                        );
                        sample.terrain_gate = terrain_gate;
                        sample.terrain_gate_position_gradient =
                            dterrain_gate_dnorm * dnorm_dpos;

                        const auto [body_gate, dbody_gate_dnorm] = smoothstep_gate(
                            direction_norm,
                            params_.runup_body_norm_lo,
                            params_.runup_body_norm_hi
                        );
                        if (body_gate <= 0.0) {
                            lookahead_samples.push_back(std::move(sample));
                            continue;
                        }
                        const Eigen::Vector2d dbody_gate_dpos =
                            dbody_gate_dnorm * dnorm_dpos;
                        const bool going_up = sample.velocity.dot(direction) >= 0.0;

                        const int x0 = static_cast<int>(std::floor(grid.x()));
                        const int y0 = static_cast<int>(std::floor(grid.y()));
                        const double tx = grid.x() - static_cast<double>(x0);
                        const double ty = grid.y() - static_cast<double>(y0);
                        struct Corner {
                            Eigen::Vector2i grid;
                            double weight;
                            Eigen::Vector2d gradient_grid;
                        };
                        const std::array<Corner, 4> corners {{
                            {{x0, y0}, (1.0 - tx) * (1.0 - ty), {-(1.0 - ty), -(1.0 - tx)}},
                            {{x0 + 1, y0}, tx * (1.0 - ty), {1.0 - ty, -tx}},
                            {{x0, y0 + 1}, (1.0 - tx) * ty, {-ty, 1.0 - tx}},
                            {{x0 + 1, y0 + 1}, tx * ty, {ty, tx}},
                        }};

                        std::array<double, TERRAIN_LABEL_COUNT> label_weights {};
                        std::array<Eigen::Vector2d, TERRAIN_LABEL_COUNT> label_gradients;
                        label_gradients.fill(Eigen::Vector2d::Zero());
                        for (const Corner& corner : corners) {
                            const uint8_t label = ws.direction_map->terrain_at(corner.grid);
                            if (label >= TERRAIN_LABEL_COUNT) continue;
                            label_weights[label] += corner.weight;
                            label_gradients[label] += corner.gradient_grid
                                / ws.direction_map->resolution;
                        }

                        for (uint8_t label = static_cast<uint8_t>(TerrainType::SLOPE);
                             label < TERRAIN_LABEL_COUNT; ++label) {
                            const double label_weight = label_weights[label];
                            if (label_weight <= 0.0) continue;
                            const TraversalMode* rule =
                                ws.terrain_constraints->selected_mode(label, going_up);
                            if (!rule) continue;
                            sample.sources[static_cast<size_t>(sample.source_count++)] = RunupSource {
                                .value = body_gate * label_weight,
                                .radius = rule->run_up,
                                .position_gradient = label_weight * dbody_gate_dpos
                                    + body_gate * label_gradients[label],
                            };
                        }
                    }
                }
            }
            lookahead_samples.push_back(std::move(sample));
        }
    }

    const int sample_count = static_cast<int>(lookahead_samples.size());
    double max_runup_radius = 0.0;
    if (ws.terrain_constraints) {
        for (uint8_t label = static_cast<uint8_t>(TerrainType::SLOPE);
             label < TERRAIN_LABEL_COUNT; ++label) {
            for (const bool going_up : {false, true}) {
                if (const TraversalMode* rule =
                        ws.terrain_constraints->selected_mode(label, going_up)) {
                    max_runup_radius = std::max(max_runup_radius, rule->run_up);
                }
            }
        }
    }
    const double lookahead_limit = max_runup_radius + params_.runup_transition_distance;
    std::vector<double> runup_exposure(static_cast<size_t>(sample_count), 0.0);
    std::vector<double> runup_gate(static_cast<size_t>(sample_count), 0.0);
    for (int i = 0; i < sample_count; ++i) {
        double distance = 0.0;
        double exposure = 0.0;
        for (int j = i; j < sample_count; ++j) {
            const auto& future = lookahead_samples[static_cast<size_t>(j)];
            for (int source_index = 0; source_index < future.source_count; ++source_index) {
                const RunupSource& source = future.sources[static_cast<size_t>(source_index)];
                const auto [distance_gate, unused] = runup_distance_gate(
                    distance, source.radius, params_.runup_transition_distance
                );
                (void)unused;
                exposure += source.value * distance_gate * future.arc_measure;
            }
            distance += future.arc_measure;
            if (distance > lookahead_limit) break;
        }
        runup_exposure[static_cast<size_t>(i)] = exposure;
        runup_gate[static_cast<size_t>(i)] = 1.0 - std::exp(
            -exposure / params_.runup_saturation_length
        );
    }

    // 状态正则 gate = 1-(1-runup_gate)(1-terrain_gate)。这里反传其 runup 分支；
    // terrain 分支的位置梯度在局部采样循环中累积。
    std::vector<Eigen::Vector2d> lookahead_position_gradient(
        static_cast<size_t>(sample_count), Eigen::Vector2d::Zero()
    );
    std::vector<double> arc_measure_gradient(static_cast<size_t>(sample_count), 0.0);
    std::vector<double> arc_range_difference(static_cast<size_t>(sample_count + 1), 0.0);
    for (int i = 0; i < sample_count; ++i) {
        const auto& current = lookahead_samples[static_cast<size_t>(i)];
        const double acceleration_density = 0.5 * w.runup_accel
            * current.acceleration.squaredNorm();
        const double omega_density = 0.5 * w.runup_omega * current.omega * current.omega;
        const double dcost_dexposure = current.dt
            * (acceleration_density + omega_density)
            * (1.0 - current.terrain_gate)
            * std::exp(-runup_exposure[static_cast<size_t>(i)]
                / params_.runup_saturation_length)
            / params_.runup_saturation_length;
        if (dcost_dexposure == 0.0) continue;

        double distance = 0.0;
        for (int j = i; j < sample_count; ++j) {
            const auto& future = lookahead_samples[static_cast<size_t>(j)];
            for (int source_index = 0; source_index < future.source_count; ++source_index) {
                const RunupSource& source = future.sources[static_cast<size_t>(source_index)];
                const auto [distance_gate, dgate_ddistance] = runup_distance_gate(
                    distance, source.radius, params_.runup_transition_distance
                );
                if (distance_gate <= 0.0 && dgate_ddistance == 0.0) continue;
                const double source_scale = dcost_dexposure
                    * distance_gate * future.arc_measure;
                lookahead_position_gradient[static_cast<size_t>(j)] +=
                    source_scale * source.position_gradient;
                arc_measure_gradient[static_cast<size_t>(j)] +=
                    dcost_dexposure * source.value * distance_gate;

                if (j > i && dgate_ddistance != 0.0) {
                    const double range_scale = dcost_dexposure * source.value
                        * future.arc_measure * dgate_ddistance;
                    arc_range_difference[static_cast<size_t>(i)] += range_scale;
                    arc_range_difference[static_cast<size_t>(j)] -= range_scale;
                }
            }
            distance += future.arc_measure;
            if (distance > lookahead_limit) break;
        }
    }
    double accumulated_range_gradient = 0.0;
    for (int i = 0; i < sample_count; ++i) {
        accumulated_range_gradient += arc_range_difference[static_cast<size_t>(i)];
        arc_measure_gradient[static_cast<size_t>(i)] += accumulated_range_gradient;
    }

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

            double density = 0.0;
            double gpx = 0, gpy = 0, gvx = 0, gvy = 0, gax = 0, gay = 0;
            double d_obstacle = 0, d_velocity = 0, d_lateral_acc = 0;
            double d_omega = 0, d_accel = 0, d_traversal_alignment = 0;
            double d_traversal_velocity = 0, d_prohibited_traversal = 0;
            double d_runup_accel = 0, d_runup_omega = 0;

            const double speed2 = vx * vx + vy * vy;
            const double speed = std::sqrt(speed2);
            const double omega_den = speed2 + OMEGA_SPEED_REG_SQ;
            const double cross = vx * ay - vy * ax;
            const double omega = cross / omega_den;
            const double v_signed = gear * speed;
            const int flat_sample_index = seg * S + s;
            const LookaheadSample& cached_sample = lookahead_samples[
                static_cast<size_t>(flat_sample_index)
            ];
            const Eigen::Vector2d velocity(vx, vy);
            const SmoothMotion& motion = cached_sample.motion;

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

            // 全局物理速度界使用真实 ‖v‖；仅在精确零速处取安全的零次梯度。
            const double inverse_speed = speed > EPS ? 1.0 / speed : 0.0;
            const double dvs_dvx = gear * vx * inverse_speed;
            const double dvs_dvy = gear * vy * inverse_speed;

            // (b) 带符号速度窗：v ≤ vel_max, v ≥ vel_min
            {
                const double over = v_signed - lim.velocity.max;
                const double under = lim.velocity.min - v_signed;
                const double go = violation(over), gu = violation(under);
                density += w.trajectory_velocity * 0.5 * (go * go + gu * gu);
                if (terms) {
                    d_velocity += w.trajectory_velocity * 0.5 * (go * go + gu * gu);
                }
                const double dv = w.trajectory_velocity * (go - gu); // dρ/dv_signed
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
                const double mag = std::abs(a_lat) - lim.lateral_acceleration_max;
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
                const double mag = std::abs(omega) - lim.angular_velocity_max;
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
                const double mag = amag - lim.acceleration_max;
                const double gm = violation(mag);
                density += w.accel * 0.5 * gm * gm;
                if (terms) d_accel += w.accel * 0.5 * gm * gm;
                const double coeff = w.accel * gm / amag;
                gax += coeff * ax;
                gay += coeff * ay;
            }

            // 状态正则从助跑贯穿本体/膨胀过渡：gate 是 runup 与当前位置 terrain 的平滑并集。
            // runup 的非局部梯度稍后回传；当前位置 terrain 分支在这里按乘积法则回传。
            const double current_runup_gate = runup_gate[
                static_cast<size_t>(flat_sample_index)
            ];
            const double state_gate = 1.0
                - (1.0 - current_runup_gate) * (1.0 - cached_sample.terrain_gate);
            const double runup_accel_density = 0.5 * w.runup_accel
                * (ax * ax + ay * ay);
            const double runup_omega_density = 0.5 * w.runup_omega * omega * omega;
            const double state_regularization_density =
                runup_accel_density + runup_omega_density;
            if (state_gate > 0.0) {
                density += state_gate * state_regularization_density;
                if (terms) {
                    d_runup_accel += state_gate * runup_accel_density;
                    d_runup_omega += state_gate * runup_omega_density;
                }
                gax += state_gate * w.runup_accel * ax;
                gay += state_gate * w.runup_accel * ay;
                const double omega_coeff = state_gate * w.runup_omega * omega;
                gvx += omega_coeff * domega_dvx;
                gvy += omega_coeff * domega_dvy;
                gax += omega_coeff * domega_dax;
                gay += omega_coeff * domega_day;
            }
            const double terrain_union_scale = (1.0 - current_runup_gate)
                * state_regularization_density;
            gpx += terrain_union_scale
                * cached_sample.terrain_gate_position_gradient.x();
            gpy += terrain_union_scale
                * cached_sample.terrain_gate_position_gradient.y();

            // 膨胀方向场连续控制对齐强度；离散 terrain label 只选择速度 profile。
            if (ws.terrain_constraints
                && (w.traversal_alignment > 0.0
                    || w.traversal_velocity_target > 0.0
                    || w.prohibited_traversal > 0.0)
                && cached_sample.terrain_gate > 0.0
                && cached_sample.direction_norm > EPS) {
                const double gate = cached_sample.terrain_gate;
                const uint8_t label = cached_sample.terrain_label;
                const Eigen::Vector2d& dir = cached_sample.terrain_direction;
                const Eigen::Matrix2d& dir_jacobian_map =
                    cached_sample.terrain_direction_jacobian;

                double terrain_density = 0.0;

                // (1) 对齐罚。低速方向 u=v/sqrt(‖v‖²+v0²)，无零速激活断崖。
                const Eigen::Vector2d& smooth_direction = motion.direction;
                const double e = smooth_direction.x() * dir.y()
                    - smooth_direction.y() * dir.x();
                const double alignment = smooth_direction.dot(dir);
                const double align_density = w.traversal_alignment * 0.5 * e * e;
                terrain_density += align_density;
                density += gate * align_density;
                if (terms) d_traversal_alignment += gate * align_density;
                const double de_scale = gate * w.traversal_alignment * e;
                const Eigen::Vector2d de_dsmooth_direction(dir.y(), -dir.x());
                const Eigen::Vector2d de_dvelocity =
                    motion.direction_jacobian.transpose() * de_dsmooth_direction;
                gvx += de_scale * de_dvelocity.x();
                gvy += de_scale * de_dvelocity.y();
                const Eigen::Vector2d de_ddir(
                    -smooth_direction.y(), smooth_direction.x()
                );
                const Eigen::Vector2d de_dposition =
                    dir_jacobian_map.transpose() * de_ddir;
                gpx += de_scale * de_dposition.x();
                gpy += de_scale * de_dposition.y();

                // (2) 两侧 mode 均存在时，速度窗在低投影速度附近平滑混合；仅一侧存在时
                // 始终使用该可用窗口，禁止方向由 prohibited 项负责，避免 availability 权重
                // 在零速把梯度推向缺失侧。目标速度统一使用 gear·smooth_speed。
                const TraversalMode* up_rule =
                    ws.terrain_constraints->selected_mode(label, true);
                const TraversalMode* down_rule =
                    ws.terrain_constraints->selected_mode(label, false);
                const double traversal_speed = gear * motion.speed;
                struct VelocityWindowCost {
                    double density = 0.0;
                    double speed_derivative = 0.0;
                };
                const auto velocity_window_cost = [&](const TraversalMode* mode) {
                    VelocityWindowCost result;
                    if (!mode) return result;
                    const double over = traversal_speed - mode->velocity_window.max;
                    const double under = mode->velocity_window.min - traversal_speed;
                    const double go = violation(over);
                    const double gu = violation(under);
                    result.density = 0.5 * w.traversal_velocity_target
                        * (go * go + gu * gu);
                    result.speed_derivative = w.traversal_velocity_target * (go - gu);
                    return result;
                };
                const VelocityWindowCost up_velocity = velocity_window_cost(up_rule);
                const VelocityWindowCost down_velocity = velocity_window_cost(down_rule);

                double up_weight = 0.0;
                double down_weight = 0.0;
                double dup_weight_dprojected_speed = 0.0;
                if (up_rule && down_rule) {
                    const double projected_speed = velocity.dot(dir);
                    const double profile_scale_squared =
                        params_.terrain_gate.motion_speed_scale
                        * params_.terrain_gate.motion_speed_scale;
                    const double projection_denominator = std::sqrt(
                        projected_speed * projected_speed + profile_scale_squared
                    );
                    up_weight = 0.5 * (
                        1.0 + projected_speed / projection_denominator
                    );
                    down_weight = 1.0 - up_weight;
                    dup_weight_dprojected_speed = 0.5 * profile_scale_squared
                        / (projection_denominator * projection_denominator
                            * projection_denominator);
                } else if (up_rule) {
                    up_weight = 1.0;
                } else if (down_rule) {
                    down_weight = 1.0;
                }
                const double velocity_density = up_weight * up_velocity.density
                    + down_weight * down_velocity.density;
                terrain_density += velocity_density;
                density += gate * velocity_density;
                if (terms) d_traversal_velocity += gate * velocity_density;

                const double speed_density_derivative = up_weight
                    * up_velocity.speed_derivative
                    + down_weight * down_velocity.speed_derivative;
                const Eigen::Vector2d speed_velocity_gradient = gear
                    * motion.speed_gradient;
                const double profile_weight_gradient_scale =
                    (up_velocity.density - down_velocity.density)
                    * dup_weight_dprojected_speed;
                const Eigen::Vector2d velocity_density_gradient =
                    speed_density_derivative * speed_velocity_gradient
                    + profile_weight_gradient_scale * dir;
                gvx += gate * velocity_density_gradient.x();
                gvy += gate * velocity_density_gradient.y();

                const Eigen::Vector2d projected_speed_position_gradient =
                    dir_jacobian_map.transpose() * velocity;
                const Eigen::Vector2d profile_weight_position_gradient =
                    profile_weight_gradient_scale
                    * projected_speed_position_gradient;
                gpx += gate * profile_weight_position_gradient.x();
                gpy += gate * profile_weight_position_gradient.y();

                // (3) 单向禁止仅表示对应 up/down traversal mode 缺失。
                const bool directional_label =
                    label >= static_cast<uint8_t>(TerrainType::SLOPE);
                const double prohibited_up = directional_label && !up_rule
                    ? violation(alignment)
                    : 0.0;
                const double prohibited_down = directional_label && !down_rule
                    ? violation(-alignment)
                    : 0.0;
                const double prohibited_density = 0.5 * w.prohibited_traversal
                    * (prohibited_up * prohibited_up + prohibited_down * prohibited_down);
                terrain_density += prohibited_density;
                density += gate * prohibited_density;
                if (terms) d_prohibited_traversal += gate * prohibited_density;
                const double alignment_scale = gate * w.prohibited_traversal
                    * (prohibited_up - prohibited_down);
                const Eigen::Vector2d dalignment_dvelocity =
                    motion.direction_jacobian.transpose() * dir;
                gvx += alignment_scale * dalignment_dvelocity.x();
                gvy += alignment_scale * dalignment_dvelocity.y();
                const Eigen::Vector2d alignment_position_gradient =
                    dir_jacobian_map.transpose() * smooth_direction;
                gpx += alignment_scale * alignment_position_gradient.x();
                gpy += alignment_scale * alignment_position_gradient.y();

                gpx += terrain_density
                    * cached_sample.terrain_gate_position_gradient.x();
                gpy += terrain_density
                    * cached_sample.terrain_gate_position_gradient.y();
            }

            cost += dt * density;
            if (terms) {
                terms->obstacle += dt * d_obstacle;
                terms->trajectory_velocity += dt * d_velocity;
                terms->lateral_acc += dt * d_lateral_acc;
                terms->omega += dt * d_omega;
                terms->accel += dt * d_accel;
                terms->traversal_alignment += dt * d_traversal_alignment;
                terms->traversal_velocity_target += dt * d_traversal_velocity;
                terms->prohibited_traversal += dt * d_prohibited_traversal;
                terms->runup_accel += dt * d_runup_accel;
                terms->runup_omega += dt * d_runup_omega;
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

    // arc=dt·smooth_speed：对 v 的伴随为 arc_adjoint·dt·ds/dv；显式 T 导数还包含
    // smooth_speed/S 与采样横坐标漂移。将这些非局部项回传到多项式系数和段时长。
    for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
        const auto& sample = lookahead_samples[static_cast<size_t>(sample_index)];
        const int c_off = NCOEF * sample.segment;
        const double arc_adjoint = arc_measure_gradient[static_cast<size_t>(sample_index)];
        const Eigen::Vector2d velocity_adjoint = arc_adjoint * sample.dt
            * sample.motion.speed_gradient;
        const Eigen::Vector2d& position_adjoint =
            lookahead_position_gradient[static_cast<size_t>(sample_index)];

        double b0[NCOEF], b1[NCOEF];
        double tp = 1.0;
        for (int k = 0; k < NCOEF; ++k) { b0[k] = tp; tp *= sample.local_time; }
        b1[0] = 0.0;
        tp = 1.0;
        for (int k = 1; k < NCOEF; ++k) { b1[k] = k * tp; tp *= sample.local_time; }
        for (int k = 0; k < NCOEF; ++k) {
            grad_c(c_off + k, 0) += position_adjoint.x() * b0[k]
                + velocity_adjoint.x() * b1[k];
            grad_c(c_off + k, 1) += position_adjoint.y() * b0[k]
                + velocity_adjoint.y() * b1[k];
        }

        grad_t_explicit(sample.segment) += arc_adjoint * sample.motion.speed
            / static_cast<double>(samples_per_segment);
        grad_t_explicit(sample.segment) += sample.sample_time_fraction * (
            position_adjoint.dot(sample.velocity)
            + velocity_adjoint.dot(sample.acceleration)
        );
    }

    return cost;
}

double MincoOptimizer::evaluate(Workspace& ws, const Eigen::VectorXd& vars, Eigen::VectorXd& grad) const {
    const int n = ws.n_segments;
    const int nw = ws.n_waypoints;
    const int time_offset = DIM * nw;

    for (int i = 0; i < nw; ++i) {
        ws.waypoints(0, i) = vars(DIM * i + 0);
        ws.waypoints(1, i) = vars(DIM * i + 1);
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
    const TerrainTraversalConstraints& terrain_constraints
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
    ws.times = seed_durations;
    ws.gears = seed_gears;
    ws.cusp.assign(static_cast<size_t>(std::max(n - 1, 0)), 0);
    for (int i = 0; i < n - 1 && i < static_cast<int>(cusp_waypoints.size()); ++i) {
        ws.cusp[static_cast<size_t>(i)] = cusp_waypoints[static_cast<size_t>(i)];
    }
    ws.waypoints.setZero(DIM, std::max(n - 1, 0));

    for (int i = 0; i < n - 1; ++i) {
        ws.waypoints.col(i) = seed_states[static_cast<size_t>(i + 1)].pos;
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

    Eigen::VectorXd seed_vars;
    if (params_.debug_diagnostics) {
        seed_vars = vars;
        Eigen::VectorXd g_seed(vars.size());
        evaluate(ws, vars, g_seed);
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

    Eigen::VectorXd scratch_grad(vars.size());
    evaluate(ws, vars, scratch_grad);
    result.trajectory = ws.minco.to_trajectory(ws.gears);
    result.cost = lr.cost;
    result.success = true;

    if (params_.debug_diagnostics) {
        Eigen::MatrixXd grad_c_dbg;
        Eigen::VectorXd grad_t_dbg;
        accumulate_penalties(ws, grad_c_dbg, grad_t_dbg, &result.final_costs);
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
