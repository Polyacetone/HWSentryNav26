#include <nav_executor/path_planner/trajectory/minco_optimizer.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace nav_executor {

namespace {

constexpr int DIM = MincoMinJerk::DIM;
constexpr int NCOEF = MincoMinJerk::NCOEF;
constexpr double EPS = 1e-9;
// 速度模长的平滑正则尺度 (m/s)。有向正则性罚把 ‖p_τ‖ 推离零，因此这里只需
// 一个远小于工作速度的数值下限来保证 |v|、方向及其雅可比处处可微。
constexpr double SPEED_REG = 1e-4;
constexpr double SPEED_REG_SQ = SPEED_REG * SPEED_REG;

inline double violation(const double g) { return std::max(g, 0.0); }

inline double cross_2d(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}

inline Eigen::Vector2d perpendicular(const Eigen::Vector2d& v) {
    return {-v.y(), v.x()};
}

// 用 C1 过渡启停地形罚，避免台阶边界出现代价跳变。
inline std::pair<double, double> smoothstep_gate(
    const double x, const double lo, const double hi
) {
    if (hi <= lo) return {x >= hi ? 1.0 : 0.0, 0.0};
    const double t = (x - lo) / (hi - lo);
    if (t <= 0.0) return {0.0, 0.0};
    if (t >= 1.0) return {1.0, 0.0};
    return {t * t * (3.0 - 2.0 * t), 6.0 * t * (1.0 - t) / (hi - lo)};
}

// 在助跑范围外以 C1 过渡衰减。
inline std::pair<double, double> runup_distance_gate(
    const double distance, const double radius, const double transition
) {
    if (distance <= radius) return {1.0, 0.0};
    if (transition <= 0.0 || distance >= radius + transition) return {0.0, 0.0};
    const double x = (radius + transition - distance) / transition;
    return {x * x * (3.0 - 2.0 * x), -6.0 * x * (1.0 - x) / transition};
}

// 真实几何曲率及其弧长变化率，连同对 (v, a, j) 的解析梯度。
// κ = det(v,a)/‖v‖³，dκ/ds = [det(v,j)/‖v‖³ − 3κ(v·a)/‖v‖²]/‖v‖。
// 两者都与参数化无关，因此规划器与跟随层共享同一份定义。
struct CurvatureJet {
    double kappa = 0.0;
    double kappa_rate = 0.0;
    Eigen::Vector2d dkappa_dvelocity = Eigen::Vector2d::Zero();
    Eigen::Vector2d dkappa_dacceleration = Eigen::Vector2d::Zero();
    Eigen::Vector2d drate_dvelocity = Eigen::Vector2d::Zero();
    Eigen::Vector2d drate_dacceleration = Eigen::Vector2d::Zero();
    Eigen::Vector2d drate_djerk = Eigen::Vector2d::Zero();
};

CurvatureJet curvature_jet(
    const Eigen::Vector2d& velocity,
    const Eigen::Vector2d& acceleration,
    const Eigen::Vector2d& jerk
) {
    const double speed_squared = velocity.squaredNorm() + SPEED_REG_SQ;
    const double speed = std::sqrt(speed_squared);
    const double speed3 = speed_squared * speed;
    const double speed5 = speed3 * speed_squared;
    const double speed4 = speed_squared * speed_squared;
    const double speed6 = speed4 * speed_squared;

    const double turn = cross_2d(velocity, acceleration);
    const double twist = cross_2d(velocity, jerk);
    const double tangential = velocity.dot(acceleration);

    CurvatureJet jet;
    jet.kappa = turn / speed3;
    jet.dkappa_dvelocity = Eigen::Vector2d(acceleration.y(), -acceleration.x()) / speed3
        - 3.0 * turn * velocity / speed5;
    jet.dkappa_dacceleration = perpendicular(velocity) / speed3;

    // κ' = A − B，A = det(v,j)/‖v‖⁴，B = 3κ(v·a)/‖v‖³。
    const double a_term = twist / speed4;
    const double b_term = 3.0 * jet.kappa * tangential / speed3;
    jet.kappa_rate = a_term - b_term;

    const Eigen::Vector2d da_dvelocity = Eigen::Vector2d(jerk.y(), -jerk.x()) / speed4
        - 4.0 * twist * velocity / speed6;
    const Eigen::Vector2d db_dvelocity =
        3.0 * (jet.dkappa_dvelocity * tangential + jet.kappa * acceleration) / speed3
        - 9.0 * jet.kappa * tangential * velocity / speed5;
    jet.drate_dvelocity = da_dvelocity - db_dvelocity;
    jet.drate_dacceleration = -3.0
        * (jet.dkappa_dacceleration * tangential + jet.kappa * velocity) / speed3;
    jet.drate_djerk = perpendicular(velocity) / speed4;
    return jet;
}

// 虚拟时间正性重参数化：T = softplus(tau_v) + min_t，保证 T>0 且光滑。
inline double virtual_to_time(const double tau_v, const double min_t) {
    const double softplus = tau_v > 30.0 ? tau_v : std::log1p(std::exp(tau_v));
    return softplus + min_t;
}
inline double virtual_to_time_grad(const double tau_v) {
    return tau_v > 30.0 ? 1.0 : 1.0 / (1.0 + std::exp(-tau_v));
}
inline double time_to_virtual(const double t, const double min_t) {
    const double softplus = std::max(t - min_t, 1e-6);
    return softplus > 30.0 ? softplus : std::log(std::expm1(softplus));
}

// 多项式基 β^{(order)}(t)，order = 0..4。
struct PolynomialBasis {
    std::array<std::array<double, NCOEF>, 5> value {};

    explicit PolynomialBasis(const double t) {
        static constexpr std::array<int, 5> FIRST_NONZERO {0, 1, 2, 3, 4};
        for (size_t order = 0; order < value.size(); ++order) {
            auto& row = value[order];
            row.fill(0.0);
            double power = 1.0;
            for (int k = FIRST_NONZERO[order]; k < NCOEF; ++k) {
                double factor = 1.0;
                for (size_t d = 0; d < order; ++d) {
                    factor *= static_cast<double>(k - static_cast<int>(d));
                }
                row[static_cast<size_t>(k)] = factor * power;
                power *= t;
            }
        }
    }

    [[nodiscard]] const std::array<double, NCOEF>& operator[](const int order) const {
        return value[static_cast<size_t>(order)];
    }
};

struct RunupSource {
    double value = 0.0;
    double radius = 0.0;
    Eigen::Vector2d position_gradient = Eigen::Vector2d::Zero();
};

// 每个采样点的完整几何与地形快照。罚项与非局部助跑回传都基于这份缓存，
// 避免同一采样点被重复求值。
struct Sample {
    int segment = 0;
    double local_time = 0.0;
    double dt = 0.0;            // 积分测度 T/S
    double time_fraction = 0.0; // s/S，用于 T 的采样横坐标漂移
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
    Eigen::Vector2d acceleration = Eigen::Vector2d::Zero();
    Eigen::Vector2d jerk = Eigen::Vector2d::Zero();
    Eigen::Vector2d snap = Eigen::Vector2d::Zero();
    Eigen::Vector2d seed_tangent = Eigen::Vector2d::UnitX();

    double speed = 0.0;                     // sqrt(‖v‖² + SPEED_REG²)
    Eigen::Vector2d speed_gradient = Eigen::Vector2d::Zero();
    Eigen::Vector2d direction = Eigen::Vector2d::Zero();
    Eigen::Matrix2d direction_jacobian = Eigen::Matrix2d::Zero();
    CurvatureJet curvature;
    double arc_measure = 0.0;               // dt · speed

    uint8_t terrain_label = static_cast<uint8_t>(TerrainType::FLAT);
    double direction_norm = 0.0;
    Eigen::Vector2d terrain_direction = Eigen::Vector2d::Zero();
    Eigen::Matrix2d terrain_direction_jacobian = Eigen::Matrix2d::Zero();
    double terrain_gate = 0.0;
    Eigen::Vector2d terrain_gate_position_gradient = Eigen::Vector2d::Zero();
    std::array<RunupSource, TERRAIN_LABEL_COUNT - 2> sources {};
    int source_count = 0;
};

// 采样点罚的梯度累加器：对物理量的偏导，最终经 β 映射回多项式系数。
struct StateGradient {
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
    Eigen::Vector2d acceleration = Eigen::Vector2d::Zero();
    Eigen::Vector2d jerk = Eigen::Vector2d::Zero();
};

} // anonymous namespace

// 单次优化复用的工作区。
struct MincoOptimizer::Workspace {
    int n_segments = 0;
    int n_waypoints = 0; // n_segments - 1

    MincoMinJerk minco;
    MincoMinJerk::BoundaryPVA head;
    MincoMinJerk::BoundaryPVA tail;
    std::vector<Eigen::Vector2d> seed_tangents; // 段边界处的有向单位切向，size = n+1

    const CostMap* cost_map = nullptr;
    const DirectionMap* direction_map = nullptr;
    const TerrainTraversalConstraints* terrain_constraints = nullptr;

    // 当前解缓存
    Eigen::Matrix<double, DIM, Eigen::Dynamic> waypoints;
    std::vector<double> times;
};

MincoOptimizer::MincoOptimizer(Params params)
    : params_(std::move(params)) {}

// 采样点罚 + 能量 + 参数长度；梯度累积到 grad_c / grad_t_explicit。返回代价。
double MincoOptimizer::accumulate_penalties(
    Workspace& ws,
    Eigen::MatrixXd& grad_c,
    Eigen::VectorXd& grad_t_explicit,
    CostTerms* terms
) const {
    const auto& coeffs = ws.minco.coefficients();
    const int n = ws.n_segments;
    const auto& w = params_.weights;
    const auto& limits = params_.limits;
    const double curvature_max = limits.curvature_max();
    const double curvature_rate_max = limits.curvature_rate_max();
    const int samples_per_segment = std::max(params_.samples_per_segment, 1);

    double cost = 0.0;
    grad_c.setZero(NCOEF * n, DIM);
    grad_t_explicit.setZero(n);
    if (terms) *terms = CostTerms {};

    // ── 采样点几何与地形快照 ──
    // 方向场幅值决定连续强度，双线性标签权重只用于 runup source。主 profile 仍取包含
    // 采样点的单个栅格 label：跨 label 重合属于地图异常并由 map_server 聚合告警。
    std::vector<Sample> samples;
    samples.reserve(static_cast<size_t>(n * samples_per_segment));
    for (int segment = 0; segment < n; ++segment) {
        const double duration = ws.times[static_cast<size_t>(segment)];
        const int c_off = NCOEF * segment;
        const Eigen::Vector2d& tangent_begin = ws.seed_tangents[static_cast<size_t>(segment)];
        const Eigen::Vector2d& tangent_end = ws.seed_tangents[static_cast<size_t>(segment + 1)];
        for (int index = 0; index < samples_per_segment; ++index) {
            const double fraction = static_cast<double>(index)
                / static_cast<double>(samples_per_segment);
            const PolynomialBasis basis(duration * fraction);

            Sample sample;
            sample.segment = segment;
            sample.local_time = duration * fraction;
            sample.dt = duration / static_cast<double>(samples_per_segment);
            sample.time_fraction = fraction;
            for (int k = 0; k < NCOEF; ++k) {
                const Eigen::Vector2d c(coeffs(c_off + k, 0), coeffs(c_off + k, 1));
                sample.position += c * basis[0][static_cast<size_t>(k)];
                sample.velocity += c * basis[1][static_cast<size_t>(k)];
                sample.acceleration += c * basis[2][static_cast<size_t>(k)];
                sample.jerk += c * basis[3][static_cast<size_t>(k)];
                sample.snap += c * basis[4][static_cast<size_t>(k)];
            }

            const Eigen::Vector2d seed_blend = tangent_begin * (1.0 - fraction)
                + tangent_end * fraction;
            if (seed_blend.norm() > EPS) sample.seed_tangent = seed_blend.normalized();

            const double speed_squared = sample.velocity.squaredNorm() + SPEED_REG_SQ;
            sample.speed = std::sqrt(speed_squared);
            sample.speed_gradient = sample.velocity / sample.speed;
            sample.direction = sample.velocity / sample.speed;
            sample.direction_jacobian = Eigen::Matrix2d::Identity() / sample.speed
                - sample.velocity * sample.velocity.transpose()
                    / (speed_squared * sample.speed);
            sample.curvature = curvature_jet(
                sample.velocity, sample.acceleration, sample.jerk
            );
            sample.arc_measure = sample.dt * sample.speed;

            if (ws.direction_map && ws.terrain_constraints) {
                const Eigen::Vector2d grid =
                    ws.direction_map->map_coord_to_grid(sample.position);
                if (ws.direction_map->is_valid_coord(grid)) {
                    const auto field = ws.direction_map->interpolate_with_gradient(grid);
                    const double norm = field.value.norm();
                    sample.terrain_label = ws.direction_map->terrain_at(grid);
                    sample.direction_norm = norm;
                    if (norm > EPS) {
                        const Eigen::Vector2d direction = field.value / norm;
                        sample.terrain_direction = direction;
                        sample.terrain_direction_jacobian = (
                            (Eigen::Matrix2d::Identity() - direction * direction.transpose())
                                / norm
                        ) * field.gradient / ws.direction_map->resolution;
                        const Eigen::Vector2d dnorm_dposition =
                            (field.gradient.transpose() * direction)
                            / ws.direction_map->resolution;
                        const auto [terrain_gate, dterrain_gate] = smoothstep_gate(
                            norm, params_.terrain_gate.norm_lo, params_.terrain_gate.norm_hi
                        );
                        sample.terrain_gate = terrain_gate;
                        sample.terrain_gate_position_gradient =
                            dterrain_gate * dnorm_dposition;

                        const auto [body_gate, dbody_gate] = smoothstep_gate(
                            norm, params_.runup_body_norm_lo, params_.runup_body_norm_hi
                        );
                        if (body_gate > 0.0) {
                            const Eigen::Vector2d dbody_gate_dposition =
                                dbody_gate * dnorm_dposition;
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
                                const uint8_t label =
                                    ws.direction_map->terrain_at(corner.grid);
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
                                sample.sources[static_cast<size_t>(sample.source_count++)] =
                                    RunupSource {
                                        .value = body_gate * label_weight,
                                        .radius = rule->run_up,
                                        .position_gradient =
                                            label_weight * dbody_gate_dposition
                                            + body_gate * label_gradients[label],
                                    };
                            }
                        }
                    }
                }
            }
            samples.push_back(std::move(sample));
        }
    }

    // ── 助跑门控：沿前方弧长积分台阶暴露量 ──
    const int sample_count = static_cast<int>(samples.size());
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
            const Sample& future = samples[static_cast<size_t>(j)];
            for (int index = 0; index < future.source_count; ++index) {
                const RunupSource& source = future.sources[static_cast<size_t>(index)];
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

    // 状态正则 gate = 1−(1−runup_gate)(1−terrain_gate)。这里反传其 runup 分支；
    // terrain 分支的位置梯度在主采样循环中按乘积法则累积。
    std::vector<Eigen::Vector2d> lookahead_position_gradient(
        static_cast<size_t>(sample_count), Eigen::Vector2d::Zero()
    );
    std::vector<double> arc_measure_gradient(static_cast<size_t>(sample_count), 0.0);
    std::vector<double> arc_range_difference(static_cast<size_t>(sample_count + 1), 0.0);
    for (int i = 0; i < sample_count; ++i) {
        const Sample& current = samples[static_cast<size_t>(i)];
        const double regularization_density = 0.5 * w.runup_curvature
            * current.curvature.kappa * current.curvature.kappa;
        const double dcost_dexposure = current.dt * regularization_density
            * (1.0 - current.terrain_gate)
            * std::exp(-runup_exposure[static_cast<size_t>(i)]
                / params_.runup_saturation_length)
            / params_.runup_saturation_length;
        if (dcost_dexposure == 0.0) continue;

        double distance = 0.0;
        for (int j = i; j < sample_count; ++j) {
            const Sample& future = samples[static_cast<size_t>(j)];
            for (int index = 0; index < future.source_count; ++index) {
                const RunupSource& source = future.sources[static_cast<size_t>(index)];
                const auto [distance_gate, dgate_ddistance] = runup_distance_gate(
                    distance, source.radius, params_.runup_transition_distance
                );
                if (distance_gate <= 0.0 && dgate_ddistance == 0.0) continue;
                lookahead_position_gradient[static_cast<size_t>(j)] +=
                    dcost_dexposure * distance_gate * future.arc_measure
                    * source.position_gradient;
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

    // ── 段级项：min-jerk 能量与参数长度 ──
    for (int segment = 0; segment < n; ++segment) {
        const double duration = ws.times[static_cast<size_t>(segment)];
        const int c_off = NCOEF * segment;
        const double t2 = duration * duration;
        const double t3 = t2 * duration;
        const double t4 = t3 * duration;
        const double t5 = t4 * duration;
        for (int d = 0; d < DIM; ++d) {
            const double c3 = coeffs(c_off + 3, d);
            const double c4 = coeffs(c_off + 4, d);
            const double c5 = coeffs(c_off + 5, d);
            const double energy =
                  36.0 * c3 * c3 * duration
                + 144.0 * c3 * c4 * t2
                + (240.0 * c3 * c5 + 192.0 * c4 * c4) * t3
                + 720.0 * c4 * c5 * t4
                + 720.0 * c5 * c5 * t5;
            cost += w.energy * energy;
            if (terms) terms->energy += w.energy * energy;
            grad_c(c_off + 3, d) += w.energy
                * (72.0 * c3 * duration + 144.0 * c4 * t2 + 240.0 * c5 * t3);
            grad_c(c_off + 4, d) += w.energy
                * (144.0 * c3 * t2 + 384.0 * c4 * t3 + 720.0 * c5 * t4);
            grad_c(c_off + 5, d) += w.energy
                * (240.0 * c3 * t3 + 720.0 * c4 * t4 + 1440.0 * c5 * t5);
            grad_t_explicit(segment) += w.energy * (
                  36.0 * c3 * c3
                + 288.0 * c3 * c4 * duration
                + 3.0 * (240.0 * c3 * c5 + 192.0 * c4 * c4) * t2
                + 2880.0 * c4 * c5 * t3
                + 3600.0 * c5 * c5 * t4
            );
        }
        cost += w.time * duration;
        if (terms) terms->time += w.time * duration;
        grad_t_explicit(segment) += w.time;
    }

    // ── 采样点罚 ──
    for (int index = 0; index < sample_count; ++index) {
        const Sample& sample = samples[static_cast<size_t>(index)];
        const CurvatureJet& jet = sample.curvature;
        const double dt = sample.dt;

        double density = 0.0;
        StateGradient gradient;
        CostTerms sample_terms;

        // (a) 障碍罚：双线性插值 map 坐标梯度；越界时按到边界的距离外推。
        if (ws.cost_map && w.obstacle > 0.0) {
            const Eigen::Vector2d grid = ws.cost_map->map_coord_to_grid(sample.position);
            double value = 0.0;
            Eigen::Vector2d value_gradient = Eigen::Vector2d::Zero();
            if (ws.cost_map->is_valid_coord(grid)) {
                const int x0 = static_cast<int>(std::floor(grid.x()));
                const int y0 = static_cast<int>(std::floor(grid.y()));
                const double tx = grid.x() - static_cast<double>(x0);
                const double ty = grid.y() - static_cast<double>(y0);
                const double c00 = static_cast<double>(ws.cost_map->at({x0, y0}));
                const double c10 = static_cast<double>(ws.cost_map->at({x0 + 1, y0}));
                const double c01 = static_cast<double>(ws.cost_map->at({x0, y0 + 1}));
                const double c11 = static_cast<double>(ws.cost_map->at({x0 + 1, y0 + 1}));
                value = (
                    (1.0 - tx) * (1.0 - ty) * c00 + tx * (1.0 - ty) * c10
                    + (1.0 - tx) * ty * c01 + tx * ty * c11
                ) / 255.0;
                const double inv_scale = 1.0 / (255.0 * ws.cost_map->resolution);
                value_gradient = {
                    ((1.0 - ty) * (c10 - c00) + ty * (c11 - c01)) * inv_scale,
                    ((1.0 - tx) * (c01 - c00) + tx * (c11 - c10)) * inv_scale,
                };
            } else {
                const double x_min = ws.cost_map->origin_x;
                const double y_min = ws.cost_map->origin_y;
                const double x_max = x_min
                    + static_cast<double>(ws.cost_map->width - 1) * ws.cost_map->resolution;
                const double y_max = y_min
                    + static_cast<double>(ws.cost_map->height - 1) * ws.cost_map->resolution;
                const Eigen::Vector2d clamped(
                    std::clamp(sample.position.x(), x_min, x_max),
                    std::clamp(sample.position.y(), y_min, y_max)
                );
                const Eigen::Vector2d delta = sample.position - clamped;
                const double distance = std::sqrt(delta.squaredNorm() + EPS);
                value = 1.0 + distance / ws.cost_map->resolution;
                value_gradient = delta / (distance * ws.cost_map->resolution);
            }
            const double obstacle = w.obstacle * 0.5 * value * value;
            density += obstacle;
            sample_terms.obstacle += obstacle;
            gradient.position += w.obstacle * value * value_gradient;
        }

        // (b) 参数化速度上界：MINCO 只需保持 τ 与弧长量级可比，物理时标由速度剖面决定。
        {
            const double over = violation(sample.speed - limits.velocity_max);
            const double penalty = w.parameterization_velocity * 0.5 * over * over;
            density += penalty;
            sample_terms.parameterization_velocity += penalty;
            gradient.velocity += w.parameterization_velocity * over * sample.speed_gradient;
        }

        // (c) 有向正则性 p_τ·t̂_seed ≥ directed_speed_min。
        // 仅限制 ‖p_τ‖>0 不够：曲线仍可从正向穿过零点后反向，因此必须约束有向分量。
        {
            const double directed = sample.velocity.dot(sample.seed_tangent);
            const double under = violation(limits.directed_speed_min - directed);
            const double penalty = w.directed_regularity * 0.5 * under * under;
            density += penalty;
            sample_terms.directed_regularity += penalty;
            gradient.velocity -= w.directed_regularity * under * sample.seed_tangent;
        }

        // (d) 真实几何曲率 |κ| ≤ κ_max。
        {
            const double over = violation(std::abs(jet.kappa) - curvature_max);
            const double penalty = w.curvature * 0.5 * over * over;
            density += penalty;
            sample_terms.curvature += penalty;
            const double scale = w.curvature * over * (jet.kappa >= 0.0 ? 1.0 : -1.0);
            gradient.velocity += scale * jet.dkappa_dvelocity;
            gradient.acceleration += scale * jet.dkappa_dacceleration;
        }

        // (e) 曲率变化率 |dκ/ds| ≤ κ'_max：阻止在极短弧长内建立大角速度。
        {
            const double over = violation(std::abs(jet.kappa_rate) - curvature_rate_max);
            const double penalty = w.curvature_rate * 0.5 * over * over;
            density += penalty;
            sample_terms.curvature_rate += penalty;
            const double scale = w.curvature_rate * over
                * (jet.kappa_rate >= 0.0 ? 1.0 : -1.0);
            gradient.velocity += scale * jet.drate_dvelocity;
            gradient.acceleration += scale * jet.drate_dacceleration;
            gradient.jerk += scale * jet.drate_djerk;
        }

        // (f) 台阶助跑与本体的 κ² 正则：gate 是 runup 与当前位置 terrain 的平滑并集。
        // runup 的非局部梯度稍后回传；当前位置 terrain 分支在这里按乘积法则回传。
        const double current_runup_gate = runup_gate[static_cast<size_t>(index)];
        const double state_gate = 1.0
            - (1.0 - current_runup_gate) * (1.0 - sample.terrain_gate);
        const double regularization_density = 0.5 * w.runup_curvature
            * jet.kappa * jet.kappa;
        if (state_gate > 0.0) {
            density += state_gate * regularization_density;
            sample_terms.runup_curvature += state_gate * regularization_density;
            const double scale = state_gate * w.runup_curvature * jet.kappa;
            gradient.velocity += scale * jet.dkappa_dvelocity;
            gradient.acceleration += scale * jet.dkappa_dacceleration;
        }
        gradient.position += (1.0 - current_runup_gate) * regularization_density
            * sample.terrain_gate_position_gradient;

        // (g) 膨胀方向场提供连续的通行方向对齐强度；离散 label 只决定禁止方向。
        if (ws.terrain_constraints
            && (w.traversal_alignment > 0.0 || w.prohibited_traversal > 0.0)
            && sample.terrain_gate > 0.0
            && sample.direction_norm > EPS) {
            const double gate = sample.terrain_gate;
            const Eigen::Vector2d& terrain = sample.terrain_direction;
            const Eigen::Matrix2d& terrain_jacobian = sample.terrain_direction_jacobian;
            const Eigen::Vector2d& heading = sample.direction;
            double terrain_density = 0.0;

            // 对齐罚：车身方向与台阶方向的叉积应为零（同向或反向均可）。
            const double cross = cross_2d(heading, terrain);
            const double alignment = heading.dot(terrain);
            const double align_density = w.traversal_alignment * 0.5 * cross * cross;
            terrain_density += align_density;
            density += gate * align_density;
            sample_terms.traversal_alignment += gate * align_density;
            const double cross_scale = gate * w.traversal_alignment * cross;
            gradient.velocity += cross_scale * sample.direction_jacobian.transpose()
                * Eigen::Vector2d(terrain.y(), -terrain.x());
            gradient.position += cross_scale * terrain_jacobian.transpose()
                * perpendicular(heading);

            // 单向禁止仅表示对应 up/down traversal mode 缺失。
            const uint8_t label = sample.terrain_label;
            const bool directional = label >= static_cast<uint8_t>(TerrainType::SLOPE);
            const bool up_allowed = ws.terrain_constraints->selected_mode(label, true);
            const bool down_allowed = ws.terrain_constraints->selected_mode(label, false);
            const double prohibited_up = directional && !up_allowed
                ? violation(alignment) : 0.0;
            const double prohibited_down = directional && !down_allowed
                ? violation(-alignment) : 0.0;
            const double prohibited_density = 0.5 * w.prohibited_traversal
                * (prohibited_up * prohibited_up + prohibited_down * prohibited_down);
            terrain_density += prohibited_density;
            density += gate * prohibited_density;
            sample_terms.prohibited_traversal += gate * prohibited_density;
            const double alignment_scale = gate * w.prohibited_traversal
                * (prohibited_up - prohibited_down);
            gradient.velocity += alignment_scale
                * sample.direction_jacobian.transpose() * terrain;
            gradient.position += alignment_scale * terrain_jacobian.transpose() * heading;

            gradient.position += terrain_density * sample.terrain_gate_position_gradient;
        }

        // ── 非局部助跑项：arc = dt·speed 的伴随，以及位置源梯度 ──
        const double arc_adjoint = arc_measure_gradient[static_cast<size_t>(index)];
        const Eigen::Vector2d arc_velocity_gradient = arc_adjoint * sample.speed_gradient;
        const Eigen::Vector2d& lookahead_position =
            lookahead_position_gradient[static_cast<size_t>(index)];

        cost += dt * density;
        if (terms) {
            terms->obstacle += dt * sample_terms.obstacle;
            terms->parameterization_velocity += dt * sample_terms.parameterization_velocity;
            terms->directed_regularity += dt * sample_terms.directed_regularity;
            terms->curvature += dt * sample_terms.curvature;
            terms->curvature_rate += dt * sample_terms.curvature_rate;
            terms->traversal_alignment += dt * sample_terms.traversal_alignment;
            terms->prohibited_traversal += dt * sample_terms.prohibited_traversal;
            terms->runup_curvature += dt * sample_terms.runup_curvature;
        }

        // 物理量梯度（含积分测度）经 β 映射回 grad_c。
        const PolynomialBasis basis(sample.local_time);
        const int c_off = NCOEF * sample.segment;
        const Eigen::Vector2d position_gradient = dt * gradient.position
            + lookahead_position;
        const Eigen::Vector2d velocity_gradient = dt * gradient.velocity
            + dt * arc_velocity_gradient;
        const Eigen::Vector2d acceleration_gradient = dt * gradient.acceleration;
        const Eigen::Vector2d jerk_gradient = dt * gradient.jerk;
        for (int k = 0; k < NCOEF; ++k) {
            for (int d = 0; d < DIM; ++d) {
                grad_c(c_off + k, d) +=
                    position_gradient(d) * basis[0][static_cast<size_t>(k)]
                    + velocity_gradient(d) * basis[1][static_cast<size_t>(k)]
                    + acceleration_gradient(d) * basis[2][static_cast<size_t>(k)]
                    + jerk_gradient(d) * basis[3][static_cast<size_t>(k)];
            }
        }

        // 显式 T 梯度：积分测度 ρ/S、arc 测度，以及采样横坐标随 T 的漂移。
        const double inv_samples = 1.0 / static_cast<double>(samples_per_segment);
        grad_t_explicit(sample.segment) += density * inv_samples;
        grad_t_explicit(sample.segment) += arc_adjoint * sample.speed * inv_samples;
        grad_t_explicit(sample.segment) += sample.time_fraction * (
            position_gradient.dot(sample.velocity)
            + velocity_gradient.dot(sample.acceleration)
            + acceleration_gradient.dot(sample.jerk)
            + jerk_gradient.dot(sample.snap)
        );
    }

    return cost;
}

double MincoOptimizer::evaluate(
    Workspace& ws, const Eigen::VectorXd& vars, Eigen::VectorXd& grad
) const {
    const int n = ws.n_segments;
    const int nw = ws.n_waypoints;
    const int time_offset = DIM * nw;

    for (int i = 0; i < nw; ++i) {
        ws.waypoints(0, i) = vars(DIM * i + 0);
        ws.waypoints(1, i) = vars(DIM * i + 1);
    }
    for (int i = 0; i < n; ++i) {
        ws.times[static_cast<size_t>(i)] = virtual_to_time(
            vars(time_offset + i), params_.min_segment_time
        );
    }

    ws.minco.generate(ws.times, ws.head, ws.tail, ws.waypoints);

    Eigen::MatrixXd grad_c;
    Eigen::VectorXd grad_t_explicit;
    const double cost = accumulate_penalties(ws, grad_c, grad_t_explicit);

    Eigen::Matrix<double, DIM, Eigen::Dynamic> grad_q;
    Eigen::VectorXd grad_t;
    ws.minco.propagate_gradient(grad_c, grad_t_explicit, grad_q, grad_t);

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
    const std::vector<Eigen::Vector2d>& seed_tangents,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints
) const {
    Result result;
    const int n = static_cast<int>(seed_durations.size());
    if (n < 1 || static_cast<int>(seed_states.size()) != n + 1
        || static_cast<int>(seed_tangents.size()) != n + 1) {
        result.error = "seed size mismatch: " + std::to_string(seed_states.size())
            + " states and " + std::to_string(seed_tangents.size())
            + " tangents for " + std::to_string(n) + " segments (need n+1)";
        return result;
    }
    Workspace ws;
    ws.n_segments = n;
    ws.n_waypoints = n - 1;
    ws.head = seed_states.front();
    ws.tail = seed_states.back();
    ws.seed_tangents = seed_tangents;
    ws.cost_map = &cost_map;
    ws.direction_map = &direction_map;
    ws.terrain_constraints = &terrain_constraints;
    ws.times = seed_durations;
    ws.waypoints.setZero(DIM, std::max(n - 1, 0));

    const int nw = n - 1;
    for (int i = 0; i < nw; ++i) {
        ws.waypoints.col(i) = seed_states[static_cast<size_t>(i + 1)].pos;
    }

    const int time_offset = DIM * nw;
    Eigen::VectorXd vars(DIM * nw + n);
    for (int i = 0; i < nw; ++i) {
        vars(DIM * i + 0) = ws.waypoints(0, i);
        vars(DIM * i + 1) = ws.waypoints(1, i);
    }
    for (int i = 0; i < n; ++i) {
        vars(time_offset + i) = time_to_virtual(
            seed_durations[static_cast<size_t>(i)], params_.min_segment_time
        );
    }

    LbfgsMinimizer::Options options;
    options.max_iterations = params_.max_iterations;
    options.max_function_evaluations = params_.optimizer.max_function_evaluations;
    options.history_size = params_.optimizer.history_size;
    options.gradient_tolerance = params_.optimizer.gradient_tolerance;
    options.scaled_step_tolerance = params_.optimizer.scaled_step_tolerance;
    options.trust_region = params_.optimizer.trust_region;
    options.curvature_relative_threshold = params_.optimizer.curvature_relative_threshold;
    options.history_acceptance_ratio = params_.optimizer.history_acceptance_ratio;
    LbfgsMinimizer solver(options);

    std::vector<LbfgsMinimizer::VariableBlock> variable_blocks;
    variable_blocks.reserve(static_cast<size_t>(nw + n));
    for (int i = 0; i < nw; ++i) {
        variable_blocks.push_back({
            .offset = DIM * i,
            .size = DIM,
            .scale = params_.optimizer.position_scale,
        });
    }
    for (int i = 0; i < n; ++i) {
        const double time_jacobian = virtual_to_time_grad(vars(time_offset + i));
        double virtual_time_scale = params_.optimizer.max_virtual_time_scale;
        if (std::isfinite(time_jacobian) && time_jacobian > 0.0) {
            virtual_time_scale = std::clamp(
                params_.optimizer.physical_time_scale / time_jacobian,
                params_.optimizer.physical_time_scale,
                params_.optimizer.max_virtual_time_scale
            );
        }
        variable_blocks.push_back({
            .offset = time_offset + i,
            .size = 1,
            .scale = virtual_time_scale,
        });
    }

    auto cost_fn = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
        return evaluate(ws, x, g);
    };

    Eigen::VectorXd seed_vars;
    if (params_.debug_diagnostics) {
        seed_vars = vars;
        Eigen::VectorXd seed_grad(vars.size());
        evaluate(ws, vars, seed_grad);
        Eigen::MatrixXd grad_c;
        Eigen::VectorXd grad_t;
        accumulate_penalties(ws, grad_c, grad_t, &result.seed_costs);
    }

    const LbfgsMinimizer::Result lr = solver.minimize(cost_fn, vars, variable_blocks);
    result.optimizer_status = lr.status;
    result.accepted_iterations = lr.accepted_iterations;
    result.function_evaluations = lr.function_evaluations;
    result.trial_evaluations = lr.trial_evaluations;
    result.rejected_trials = lr.rejected_trials;
    result.nonfinite_trials = lr.nonfinite_trials;
    result.initial_grad_inf_norm = lr.initial_grad_inf_norm;
    result.final_grad_inf_norm = lr.grad_inf_norm;
    result.final_normalized_scaled_grad_max_block_norm =
        lr.normalized_scaled_grad_max_block_norm;
    result.initial_radius = lr.initial_radius;
    result.final_radius = lr.final_radius;
    result.min_radius = lr.min_radius;
    result.max_radius = lr.max_radius;
    result.radius_shrinks = lr.radius_shrinks;
    result.radius_expansions = lr.radius_expansions;
    result.boundary_steps = lr.boundary_steps;
    result.history_updates = lr.history_updates;
    result.history_skips = lr.history_skips;
    result.history_resets = lr.history_resets;

    const bool failed_initial_evaluation =
        lr.status == LbfgsMinimizer::Status::INITIAL_EVALUATION_NONFINITE;
    const bool terminal_failure_without_progress = lr.accepted_iterations == 0
        && (lr.status == LbfgsMinimizer::Status::TRUST_REGION_TOO_SMALL
            || lr.status == LbfgsMinimizer::Status::STAGNATED
            || lr.status == LbfgsMinimizer::Status::NUMERICAL_FAILURE);
    if (failed_initial_evaluation || terminal_failure_without_progress) {
        result.error = "L-BFGS terminated with "
            + std::string(LbfgsMinimizer::status_string(lr.status))
            + " before accepting a step (cost=" + std::to_string(lr.cost)
            + ", |grad|_inf=" + std::to_string(lr.grad_inf_norm)
            + ", evals=" + std::to_string(lr.function_evaluations)
            + ", rejected=" + std::to_string(lr.rejected_trials) + ")";
        return result;
    }

    Eigen::VectorXd final_grad(vars.size());
    const double final_cost = evaluate(ws, vars, final_grad);
    if (!std::isfinite(final_cost) || !final_grad.allFinite()) {
        result.error = "L-BFGS finite incumbent could not be reconstructed";
        return result;
    }
    result.trajectory = ws.minco.to_trajectory();
    result.cost = lr.cost;
    result.success = true;

    if (params_.debug_diagnostics) {
        Eigen::MatrixXd grad_c;
        Eigen::VectorXd grad_t;
        accumulate_penalties(ws, grad_c, grad_t, &result.final_costs);
        for (int i = 0; i < time_offset; ++i) {
            result.final_grad_pos_inf_norm = std::max(
                result.final_grad_pos_inf_norm, std::abs(final_grad(i))
            );
        }
        for (int i = time_offset; i < final_grad.size(); ++i) {
            result.final_grad_time_inf_norm = std::max(
                result.final_grad_time_inf_norm, std::abs(final_grad(i))
            );
        }
        for (int i = 0; i < nw; ++i) {
            const double displacement = std::hypot(
                vars(DIM * i + 0) - seed_vars(DIM * i + 0),
                vars(DIM * i + 1) - seed_vars(DIM * i + 1)
            );
            result.waypoint_total_displacement += displacement;
            result.waypoint_max_displacement = std::max(
                result.waypoint_max_displacement, displacement
            );
        }
        result.free_waypoint_count = nw;
        result.diagnostics_valid = true;
    }
    return result;
}

} // namespace nav_executor
