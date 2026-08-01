#include <nav_executor/path_planner/trajectory/minco_optimizer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <nav_executor/path_planner/trajectory/runup_gate.hpp>

namespace nav_executor {

namespace {

constexpr int DIM = MincoMinJerk::DIM;
constexpr int NCOEF = MincoMinJerk::NCOEF;
constexpr double EPS = 1e-9;

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

// 时标不变的曲率 jet。输入仍是时间导数，但先转换到单段归一化参数 u=t/T；
// 返回梯度再通过链式法则换回时间导数坐标。
struct CurvatureJet {
    double kappa = 0.0;
    double kappa_rate = 0.0;
    Eigen::Vector2d dkappa_dvelocity = Eigen::Vector2d::Zero();
    Eigen::Vector2d dkappa_dacceleration = Eigen::Vector2d::Zero();
    Eigen::Vector2d dkappa_rate_dvelocity = Eigen::Vector2d::Zero();
    Eigen::Vector2d dkappa_rate_dacceleration = Eigen::Vector2d::Zero();
    Eigen::Vector2d dkappa_rate_djerk = Eigen::Vector2d::Zero();
};

CurvatureJet curvature_jet(
    const Eigen::Vector2d& velocity,
    const Eigen::Vector2d& acceleration,
    const Eigen::Vector2d& jerk,
    const double duration,
    const double tangent_regularization
) {
    const Eigen::Vector2d first = duration * velocity;
    const Eigen::Vector2d second = duration * duration * acceleration;
    const Eigen::Vector2d third = duration * duration * duration * jerk;
    const double q = first.squaredNorm()
        + tangent_regularization * tangent_regularization;
    const double speed = std::sqrt(q);
    const double q2 = q * q;
    const double q3 = q2 * q;
    const double q4 = q3 * q;
    const double turn = cross_2d(first, second);
    const double twist = cross_2d(first, third);
    const double stretch = first.dot(second);
    const Eigen::Vector2d dturn_dfirst(second.y(), -second.x());
    const Eigen::Vector2d dturn_dsecond = perpendicular(first);
    const Eigen::Vector2d dtwist_dfirst(third.y(), -third.x());
    const Eigen::Vector2d dtwist_dthird = perpendicular(first);

    CurvatureJet jet;
    jet.kappa = turn / (q * speed);
    jet.kappa_rate = twist / q2 - 3.0 * turn * stretch / q3;
    const Eigen::Vector2d dkappa_dfirst = dturn_dfirst / (q * speed)
        - 3.0 * turn * first / (q2 * speed);
    const Eigen::Vector2d dkappa_dsecond = dturn_dsecond / (q * speed);
    const Eigen::Vector2d dkappa_rate_dfirst = dtwist_dfirst / q2
        - 4.0 * twist * first / q3
        - 3.0 * (dturn_dfirst * stretch + turn * second) / q3
        + 18.0 * turn * stretch * first / q4;
    const Eigen::Vector2d dkappa_rate_dsecond = -3.0
        * (dturn_dsecond * stretch + turn * first) / q3;
    const Eigen::Vector2d dkappa_rate_dthird = dtwist_dthird / q2;
    jet.dkappa_dvelocity = duration * dkappa_dfirst;
    jet.dkappa_dacceleration = duration * duration * dkappa_dsecond;
    jet.dkappa_rate_dvelocity = duration * dkappa_rate_dfirst;
    jet.dkappa_rate_dacceleration = duration * duration * dkappa_rate_dsecond;
    jet.dkappa_rate_djerk = duration * duration * duration * dkappa_rate_dthird;
    return jet;
}

struct ScalarConstraintGradient {
    double penalty = 0.0;
    double derivative = 0.0;
};

ScalarConstraintGradient symmetric_limit_penalty(
    const double value,
    const double limit,
    const double weight
) {
    const double over = violation(std::abs(value) - limit);
    return {
        .penalty = 0.5 * weight * over * over,
        .derivative = weight * over * (value >= 0.0 ? 1.0 : -1.0),
    };
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

// 助跑源。value 是本体门控与 label 双线性权重、方向混合权重的乘积，全部连续，
// 因此源的出现与消失都是渐变而非阶跃。
struct RunupSource {
    double value = 0.0;
    double radius = 0.0;
    TraversalVelocityWindow velocity_window;
    // ∂value/∂position：含 body_gate 与 label 权重两条路径。
    Eigen::Vector2d position_gradient = Eigen::Vector2d::Zero();
    // ∂value/∂velocity：来自上/下行方向混合权重。
    Eigen::Vector2d velocity_gradient = Eigen::Vector2d::Zero();
};

// 前视窗口内所有助跑源的加权汇总。方向与速度窗都是权重归一化的平均值，避免
// 「第一个有效源」这种离散选择在源排序交换时产生代价跳变。
struct TraversalContext {
    bool valid = false;
    bool direction_valid = false;
    double weight_total = 0.0;   // Σ w_k
    double summed_norm = 0.0;    // ‖Σ w_k d̂_k‖，方向归一化的分母（≠ weight_total）
    TraversalVelocityWindow velocity_window; // 归一化后的加权平均窗
    Eigen::Vector2d direction = Eigen::Vector2d::Zero(); // 归一化后的加权平均方向
};

// context 聚合中单个源的贡献，供伴随回传按同一权重分摊。source/source_index
// 精确定位贡献来自哪个采样点的哪一个源，回传时不能跨源平摊。
struct TraversalContribution {
    int sample = -1;
    int source_index = -1;
    double weight = 0.0;          // source.value · distance_gate
    double distance_gate = 0.0;
    double dgate_ddistance = 0.0; // ∂distance_gate/∂前视弧长
    double source_value = 0.0;
    Eigen::Vector2d direction = Eigen::Vector2d::Zero(); // 该源的地形方向
    TraversalVelocityWindow velocity_window;
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

    double speed = 0.0;                     // ‖dp/dt‖，只用于构造弧长积分测度
    Eigen::Vector2d speed_gradient = Eigen::Vector2d::Zero();
    Eigen::Vector2d direction = Eigen::Vector2d::Zero();
    Eigen::Matrix2d direction_jacobian = Eigen::Matrix2d::Zero();
    CurvatureJet curvature;
    double arc_measure = 0.0;               // dt · speed

    double direction_norm = 0.0;
    Eigen::Vector2d terrain_direction = Eigen::Vector2d::Zero();
    Eigen::Matrix2d terrain_direction_jacobian = Eigen::Matrix2d::Zero();
    double terrain_gate = 0.0;
    Eigen::Vector2d terrain_gate_position_gradient = Eigen::Vector2d::Zero();
    double body_gate = 0.0;
    Eigen::Vector2d body_gate_position_gradient = Eigen::Vector2d::Zero();
    // 每个方向性 label 各占一个上行源与一个下行源。
    std::array<RunupSource, 2 * (TERRAIN_LABEL_COUNT - 2)> sources {};
    int source_count = 0;
    // 各 label 的双线性权重及其位置导数，用于让禁止方向罚跨 cell 连续。
    DirectionMap::LabelWeights label_weights;
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
    GeometricBoundary head;
    GeometricBoundary tail;
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

// 几何采样罚 + 参数域 min-jerk 正则；梯度累积到 grad_c / grad_t_explicit。
double MincoOptimizer::accumulate_penalties(
    Workspace& ws,
    Eigen::MatrixXd& grad_c,
    Eigen::VectorXd& grad_t_explicit,
    CostTerms* terms
) const {
    const auto& coeffs = ws.minco.coefficients();
    const int n = ws.n_segments;
    const auto& w = params_.weights;
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

            sample.speed = sample.velocity.norm();
            if (sample.speed > 0.0) sample.speed_gradient = sample.velocity / sample.speed;
            const Eigen::Vector2d normalized_tangent = duration * sample.velocity;
            const double direction_norm_squared = normalized_tangent.squaredNorm()
                + params_.geometry.tangent_regularization
                    * params_.geometry.tangent_regularization;
            const double direction_norm = std::sqrt(direction_norm_squared);
            sample.direction = normalized_tangent / direction_norm;
            sample.direction_jacobian = duration * (
                Eigen::Matrix2d::Identity() / direction_norm
                - normalized_tangent * normalized_tangent.transpose()
                    / (direction_norm_squared * direction_norm)
            );
            sample.curvature = curvature_jet(
                sample.velocity, sample.acceleration, sample.jerk, duration,
                params_.geometry.tangent_regularization
            );
            sample.arc_measure = sample.dt * sample.speed;

            if (ws.direction_map && ws.terrain_constraints) {
                // 越出方向图 footprint 时复制边界值而非整体归零，避免地形罚在
                // 图边界从满强度阶跃到 0。
                const auto field =
                    ws.direction_map->sample_map_clamped(sample.position);
                const double norm = field.value.norm();
                sample.direction_norm = norm;
                sample.label_weights =
                    ws.direction_map->label_weights_clamped(sample.position);
                if (norm > EPS) {
                    const Eigen::Vector2d direction = field.value / norm;
                    sample.terrain_direction = direction;
                    sample.terrain_direction_jacobian = (
                        (Eigen::Matrix2d::Identity() - direction * direction.transpose())
                            / norm
                    ) * field.jacobian;
                    const Eigen::Vector2d dnorm_dposition =
                        field.jacobian.transpose() * direction;
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
                        sample.body_gate = body_gate;
                        sample.body_gate_position_gradient = dbody_gate_dposition;

                        // 上/下行不再由符号硬切：在几何方向余弦带内平滑混合
                        // 带内平滑混合两侧 mode，使禁止方向的源渐隐而非骤然消失。
                        const double directed = sample.direction.dot(direction);
                        const double band = params_.directed_cosine_min;
                        const auto [up_weight, dup_weight] = smoothstep_gate(
                            directed, -band, band
                        );
                        // ∂directed/∂v = d̂；方向本身随位置变化的项经
                        // terrain_direction_jacobian 进入位置梯度。
                        const Eigen::Vector2d ddirected_dposition =
                            sample.terrain_direction_jacobian.transpose()
                            * sample.direction;

                        // label 按双线性权重混合，跨 cell 时参数连续过渡。
                        for (uint8_t label = static_cast<uint8_t>(TerrainType::SLOPE);
                             label < TERRAIN_LABEL_COUNT; ++label) {
                            const double label_weight =
                                sample.label_weights.weights[label];
                            if (label_weight <= 0.0) continue;
                            const Eigen::Vector2d& dlabel_weight =
                                sample.label_weights.dweights[label];
                            for (const bool is_up : {true, false}) {
                                const TraversalMode* rule =
                                    ws.terrain_constraints->selected_mode(label, is_up);
                                if (!rule) continue;
                                const double direction_weight =
                                    is_up ? up_weight : 1.0 - up_weight;
                                if (direction_weight <= 0.0) continue;
                                const double ddirection_weight =
                                    is_up ? dup_weight : -dup_weight;
                                const double value =
                                    body_gate * label_weight * direction_weight;
                                if (value <= 0.0) continue;
                                sample.sources[
                                    static_cast<size_t>(sample.source_count++)
                                ] = RunupSource {
                                    .value = value,
                                    .radius = rule->run_up,
                                    .velocity_window = rule->velocity_window,
                                    .position_gradient =
                                        dbody_gate_dposition * label_weight
                                            * direction_weight
                                        + body_gate * dlabel_weight * direction_weight
                                        + body_gate * label_weight
                                            * ddirection_weight * ddirected_dposition,
                                    .velocity_gradient = body_gate * label_weight
                                        * ddirection_weight
                                        * sample.direction_jacobian.transpose() * direction,
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
    std::vector<TraversalContext> traversal_contexts(
        static_cast<size_t>(sample_count)
    );
    // 每个采样点的 context 由哪些源按什么权重合成，供 alignment 与速度窗的
    // 非局部伴随按同一权重分摊。
    std::vector<std::vector<TraversalContribution>> traversal_contributions(
        static_cast<size_t>(sample_count)
    );
    for (int i = 0; i < sample_count; ++i) {
        double distance = 0.0;
        double exposure = 0.0;
        TraversalContext& context = traversal_contexts[static_cast<size_t>(i)];
        auto& contributions = traversal_contributions[static_cast<size_t>(i)];
        for (int j = i; j < sample_count; ++j) {
            const Sample& future = samples[static_cast<size_t>(j)];
            for (int index = 0; index < future.source_count; ++index) {
                const RunupSource& source = future.sources[static_cast<size_t>(index)];
                const auto [distance_gate, dgate_ddistance] = runup_distance_gate(
                    distance, source.radius, params_.runup_transition_distance
                );
                exposure += source.value * distance_gate * future.arc_measure;
                // 全部在范围内的源按 value·distance_gate 加权累加，取代
                // 「第一个有效源」的离散选择。
                if (source.value > 0.0 && distance_gate > 0.0
                    && future.terrain_direction.squaredNorm() > EPS) {
                    const double weight = source.value * distance_gate;
                    context.valid = true;
                    context.weight_total += weight;
                    context.direction += weight * future.terrain_direction;
                    context.velocity_window.min +=
                        weight * source.velocity_window.min;
                    context.velocity_window.max +=
                        weight * source.velocity_window.max;
                    contributions.push_back(TraversalContribution {
                        .sample = j,
                        .source_index = index,
                        .weight = weight,
                        .distance_gate = distance_gate,
                        .dgate_ddistance = dgate_ddistance,
                        .source_value = source.value,
                        .direction = future.terrain_direction,
                        .velocity_window = source.velocity_window,
                    });
                }
            }
            distance += future.arc_measure;
            if (distance > lookahead_limit) break;
        }
        if (context.valid && context.weight_total > 0.0) {
            const double inverse_weight = 1.0 / context.weight_total;
            context.velocity_window.min *= inverse_weight;
            context.velocity_window.max *= inverse_weight;
            const Eigen::Vector2d summed = context.direction;
            const double summed_norm = summed.norm();
            if (summed_norm > EPS) {
                context.direction = summed / summed_norm;
                context.summed_norm = summed_norm;
                context.direction_valid = true;
            } else {
                // 前视窗口内的方向互相抵消（例如两侧对向台阶）：没有可用的
                // 对齐参考，但速度窗仍使用全部源的加权平均。
                context.direction.setZero();
            }
        } else {
            context.valid = false;
            contributions.clear();
        }
        runup_exposure[static_cast<size_t>(i)] = exposure;
        runup_gate[static_cast<size_t>(i)] = 1.0 - std::exp(
            -exposure / params_.runup_saturation_length
        );
    }

    // traversal gate = 1−(1−runup_gate)(1−body_gate)，严格覆盖 runup 起点到本体
    // 出口。先汇总所有受该 gate 控制的密度，用同一套非局部伴随反传 gate。
    std::vector<double> traversal_base_density(
        static_cast<size_t>(sample_count), 0.0
    );
    for (int i = 0; i < sample_count; ++i) {
        const Sample& sample = samples[static_cast<size_t>(i)];
        const TraversalContext& context = traversal_contexts[static_cast<size_t>(i)];
        double base_density = 0.5 * w.runup_curvature
            * sample.curvature.kappa * sample.curvature.kappa;
        if (context.valid) {
            const double speed_under = violation(
                context.velocity_window.min - sample.speed
            );
            const double speed_over = violation(
                sample.speed - context.velocity_window.max
            );
            base_density += 0.5 * w.traversal_velocity_window
                * (speed_under * speed_under + speed_over * speed_over);
            if (context.direction_valid) {
                const double cross = cross_2d(sample.direction, context.direction);
                base_density += 0.5 * w.traversal_alignment * cross * cross;
            }
        }
        traversal_base_density[static_cast<size_t>(i)] = base_density;
    }

    std::vector<Eigen::Vector2d> lookahead_position_gradient(
        static_cast<size_t>(sample_count), Eigen::Vector2d::Zero()
    );
    // 源强度还依赖速度（上/下行混合权重），需要与位置梯度并行的非局部通道。
    std::vector<Eigen::Vector2d> lookahead_velocity_gradient(
        static_cast<size_t>(sample_count), Eigen::Vector2d::Zero()
    );
    std::vector<double> arc_measure_gradient(static_cast<size_t>(sample_count), 0.0);
    std::vector<double> arc_range_difference(static_cast<size_t>(sample_count + 1), 0.0);

    // ── context 的非局部伴随 ──
    // context.direction = normalize(Σ_k w_k d̂_k)，w_k = value_k · distance_gate_k。
    // 三条依赖都要回传：源方向 d̂_k、源强度 value_k，以及 distance_gate_k 所依赖的前视弧长。最后一条必须在
    // arc_range_difference 做前缀和之前累加，因此这里先于罚项主循环求值。
    for (int i = 0; i < sample_count; ++i) {
        const Sample& sample = samples[static_cast<size_t>(i)];
        const TraversalContext& traversal = traversal_contexts[static_cast<size_t>(i)];
        const auto& contributions = traversal_contributions[static_cast<size_t>(i)];
        if (!traversal.valid || contributions.empty()
            || traversal.weight_total <= 0.0) {
            continue;
        }
        const double state_gate = 1.0
            - (1.0 - runup_gate[static_cast<size_t>(i)]) * (1.0 - sample.body_gate);
        if (state_gate <= 0.0) continue;

        const double integration_measure = sample.arc_measure;
        const Eigen::Vector2d& heading = sample.direction;
        const Eigen::Vector2d& terrain = traversal.direction;
        const double cross = traversal.direction_valid
            ? cross_2d(heading, terrain) : 0.0;
        const double cross_scale = state_gate * w.traversal_alignment * cross;
        const double speed_under = violation(
            traversal.velocity_window.min - sample.speed
        );
        const double speed_over = violation(
            sample.speed - traversal.velocity_window.max
        );
        const double dcost_dwindow_min = integration_measure * state_gate
            * w.traversal_velocity_window * speed_under;
        const double dcost_dwindow_max = -integration_measure * state_gate
            * w.traversal_velocity_window * speed_over;
        // d̂ = Σ/‖Σ‖ ⇒ ∂d̂/∂Σ = (I − d̂d̂ᵀ)/‖Σ‖。归一化分母是 ‖Σ‖ 而非 Σw_k：
        // 两者仅在所有源方向平行时才相等。
        const Eigen::Vector2d dcost_ddirection =
            integration_measure * cross_scale * perpendicular(heading);
        Eigen::Vector2d dcost_dsummed = Eigen::Vector2d::Zero();
        if (traversal.direction_valid) {
            dcost_dsummed = (
                Eigen::Matrix2d::Identity() - terrain * terrain.transpose()
            ) * dcost_ddirection / traversal.summed_norm;
        }
        const double inverse_weight = 1.0 / traversal.weight_total;
        for (const TraversalContribution& contribution : contributions) {
            const auto source_index = static_cast<size_t>(contribution.sample);
            const Sample& source = samples[source_index];
            // (1) 经该源的地形方向 d̂_k：∂Σ/∂d̂_k = w_k·I。
            lookahead_position_gradient[source_index] +=
                contribution.weight
                * source.terrain_direction_jacobian.transpose() * dcost_dsummed;
            // (2) 经权重 w_k：方向侧 ∂Σ/∂w_k = d̂_k；窗侧为加权平均的偏导。
            const double dcost_dweight =
                dcost_dsummed.dot(contribution.direction)
                + inverse_weight * (
                    dcost_dwindow_min * (
                        contribution.velocity_window.min
                        - traversal.velocity_window.min
                    )
                    + dcost_dwindow_max * (
                        contribution.velocity_window.max
                        - traversal.velocity_window.max
                    )
                );
            if (dcost_dweight == 0.0) continue;
            // (2a) w_k = value_k · distance_gate_k 的 value 一侧。
            const double value_scale =
                dcost_dweight * contribution.distance_gate;
            const RunupSource& contributing_source =
                source.sources[static_cast<size_t>(contribution.source_index)];
            lookahead_position_gradient[source_index] +=
                value_scale * contributing_source.position_gradient;
            lookahead_velocity_gradient[source_index] +=
                value_scale * contributing_source.velocity_gradient;
            // (2b) distance_gate 一侧：前视弧长是 [i, j) 区间内 arc_measure 之和，
            // 用区间差分累加，随后与 exposure 通道共享同一前缀和。
            if (contribution.sample > i && contribution.dgate_ddistance != 0.0) {
                const double range_scale = dcost_dweight
                    * contribution.source_value * contribution.dgate_ddistance;
                arc_range_difference[static_cast<size_t>(i)] += range_scale;
                arc_range_difference[source_index] -= range_scale;
            }
        }
    }

    for (int i = 0; i < sample_count; ++i) {
        const Sample& current = samples[static_cast<size_t>(i)];
        const double dcost_dexposure = current.arc_measure
            * traversal_base_density[static_cast<size_t>(i)]
            * (1.0 - current.body_gate)
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
                const double source_scale =
                    dcost_dexposure * distance_gate * future.arc_measure;
                lookahead_position_gradient[static_cast<size_t>(j)] +=
                    source_scale * source.position_gradient;
                lookahead_velocity_gradient[static_cast<size_t>(j)] +=
                    source_scale * source.velocity_gradient;
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

    // ── 段级项：参数域 min-jerk 能量与参数区间长度 ──
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
        double density = 0.0;
        StateGradient gradient;
        CostTerms sample_terms;

        // (a) 障碍罚：双线性插值 map 坐标梯度。越界时在被复制的边界值之上叠加
        // 距离斜坡，而不是替换为常数 1 + d/res —— 后者会在 footprint 边界让
        // 自由区的罚从 ≈0 阶跃到满值。
        if (ws.cost_map && w.obstacle > 0.0) {
            const auto cost_sample = ws.cost_map->sample_map_clamped(sample.position);
            double value = cost_sample.value / 255.0;
            Eigen::Vector2d value_gradient = cost_sample.gradient / 255.0;
            const Eigen::Vector2d clamped =
                ws.cost_map->geometry.clamp_to_footprint(sample.position);
            const Eigen::Vector2d delta = sample.position - clamped;
            const double outside_squared = delta.squaredNorm();
            if (outside_squared > 0.0) {
                // 斜坡取 d²/(2·res·reg)（d<reg）与 d−reg/2 的 C1 拼接：边界处
                // 值与梯度都为 0，避免 sqrt 在 d→0 处的无界导数与常数偏置。
                const double regularization = EPS;
                const double distance = std::sqrt(outside_squared);
                const double resolution = ws.cost_map->geometry.resolution();
                if (distance < regularization) {
                    value += 0.5 * outside_squared / (resolution * regularization);
                    value_gradient += delta / (resolution * regularization);
                } else {
                    value += (distance - 0.5 * regularization) / resolution;
                    value_gradient += delta / (distance * resolution);
                }
            }
            const double obstacle = w.obstacle * 0.5 * value * value;
            density += obstacle;
            sample_terms.obstacle += obstacle;
            gradient.position += w.obstacle * value * value_gradient;
        }

        // (b) 纯几何曲率上界。
        {
            const ScalarConstraintGradient penalty = symmetric_limit_penalty(
                jet.kappa, params_.geometry.curvature_max, w.curvature
            );
            density += penalty.penalty;
            sample_terms.curvature += penalty.penalty;
            gradient.velocity += penalty.derivative * jet.dkappa_dvelocity;
            gradient.acceleration += penalty.derivative * jet.dkappa_dacceleration;
        }

        // (c) 弧长曲率变化率上界。该量不含任何物理速度分母。
        {
            const ScalarConstraintGradient penalty = symmetric_limit_penalty(
                jet.kappa_rate, params_.geometry.curvature_rate_max, w.curvature_rate
            );
            density += penalty.penalty;
            sample_terms.curvature_rate += penalty.penalty;
            gradient.velocity += penalty.derivative * jet.dkappa_rate_dvelocity;
            gradient.acceleration += penalty.derivative * jet.dkappa_rate_dacceleration;
            gradient.jerk += penalty.derivative * jet.dkappa_rate_djerk;
        }

        // (d) 有向几何正则性：只约束切线方向余弦，不约束伪物理速度。
        {
            const double directed = sample.direction.dot(sample.seed_tangent);
            const double under = violation(params_.directed_cosine_min - directed);
            const double penalty = w.directed_regularity * 0.5 * under * under;
            density += penalty;
            sample_terms.directed_regularity += penalty;
            gradient.velocity -= w.directed_regularity * under
                * sample.direction_jacobian.transpose() * sample.seed_tangent;
        }

        // (h) runup 起点至台阶出口的统一 traversal gate。
        const double current_runup_gate = runup_gate[static_cast<size_t>(index)];
        const double state_gate = 1.0
            - (1.0 - current_runup_gate) * (1.0 - sample.body_gate);
        const TraversalContext& traversal =
            traversal_contexts[static_cast<size_t>(index)];

        // 台阶延长区内的 κ² 正则。
        const double regularization_density = 0.5 * w.runup_curvature
            * jet.kappa * jet.kappa;
        if (state_gate > 0.0) {
            const double weighted_density = state_gate * regularization_density;
            density += weighted_density;
            sample_terms.runup_curvature += weighted_density;
            const double scale = state_gate
                * w.runup_curvature * jet.kappa;
            gradient.velocity += scale * jet.dkappa_dvelocity;
            gradient.acceleration += scale * jet.dkappa_dacceleration;
        }

        // 速度窗使用 MINCO 内部速度见证，只负责联合塑形，不会写入执行速度剖面。
        if (state_gate > 0.0 && traversal.valid) {
            const double speed_under = violation(
                traversal.velocity_window.min - sample.speed
            );
            const double speed_over = violation(
                sample.speed - traversal.velocity_window.max
            );
            const double speed_density = 0.5 * w.traversal_velocity_window
                * (speed_under * speed_under + speed_over * speed_over);
            density += state_gate * speed_density;
            sample_terms.traversal_velocity_window += state_gate * speed_density;
            gradient.velocity += state_gate * w.traversal_velocity_window
                * (speed_over - speed_under) * sample.speed_gradient;
        }

        // 地形方向对齐同样只依赖几何切向。
        if (state_gate > 0.0 && traversal.direction_valid) {
            const Eigen::Vector2d& heading = sample.direction;
            const Eigen::Vector2d& terrain = traversal.direction;
            const double cross = cross_2d(heading, terrain);
            const double align_density = 0.5 * w.traversal_alignment
                * cross * cross;
            density += state_gate * align_density;
            sample_terms.traversal_alignment += state_gate * align_density;
            const double cross_scale = state_gate * w.traversal_alignment * cross;
            gradient.velocity += cross_scale * sample.direction_jacobian.transpose()
                * Eigen::Vector2d(terrain.y(), -terrain.x());
        }

        gradient.position += (1.0 - current_runup_gate)
            * traversal_base_density[static_cast<size_t>(index)]
            * sample.body_gate_position_gradient;

        // (i) 膨胀方向场仅负责禁止方向；允许方向的对齐由 traversal gate 处理。
        if (ws.terrain_constraints && w.prohibited_traversal > 0.0
            && sample.terrain_gate > 0.0 && sample.direction_norm > EPS) {
            const double gate = sample.terrain_gate;
            const Eigen::Vector2d& terrain = sample.terrain_direction;
            const Eigen::Matrix2d& terrain_jacobian = sample.terrain_direction_jacobian;
            const Eigen::Vector2d& heading = sample.direction;
            const double alignment = heading.dot(terrain);
            // label 不再取单格硬值：按双线性权重对各 label 的禁止性加权，
            // 使罚在 label 边界连续过渡。
            double prohibited_density = 0.0;
            double alignment_scale = 0.0;
            Eigen::Vector2d label_weight_gradient = Eigen::Vector2d::Zero();
            for (uint8_t label = static_cast<uint8_t>(TerrainType::SLOPE);
                 label < TERRAIN_LABEL_COUNT; ++label) {
                const double label_weight = sample.label_weights.weights[label];
                if (label_weight <= 0.0) continue;
                const bool up_allowed =
                    ws.terrain_constraints->selected_mode(label, true);
                const bool down_allowed =
                    ws.terrain_constraints->selected_mode(label, false);
                const double prohibited_up = up_allowed
                    ? 0.0 : violation(alignment);
                const double prohibited_down = down_allowed
                    ? 0.0 : violation(-alignment);
                const double label_density = 0.5 * w.prohibited_traversal
                    * (prohibited_up * prohibited_up
                        + prohibited_down * prohibited_down);
                prohibited_density += label_weight * label_density;
                alignment_scale += label_weight * w.prohibited_traversal
                    * (prohibited_up - prohibited_down);
                label_weight_gradient +=
                    label_density * sample.label_weights.dweights[label];
            }
            density += gate * prohibited_density;
            sample_terms.prohibited_traversal += gate * prohibited_density;
            alignment_scale *= gate;
            gradient.velocity += alignment_scale
                * sample.direction_jacobian.transpose() * terrain;
            gradient.position += alignment_scale * terrain_jacobian.transpose() * heading;
            gradient.position += gate * label_weight_gradient;
            gradient.position += prohibited_density
                * sample.terrain_gate_position_gradient;
        }

        // ── 非局部助跑项：arc = dt·speed 的伴随，以及位置源梯度 ──
        const double arc_adjoint = arc_measure_gradient[static_cast<size_t>(index)];
        const Eigen::Vector2d arc_velocity_gradient = arc_adjoint * sample.speed_gradient;
        const Eigen::Vector2d& lookahead_position =
            lookahead_position_gradient[static_cast<size_t>(index)];
        const Eigen::Vector2d& lookahead_velocity =
            lookahead_velocity_gradient[static_cast<size_t>(index)];

        const double integration_measure = sample.arc_measure;
        cost += integration_measure * density;
        if (terms) {
            terms->obstacle += integration_measure * sample_terms.obstacle;
            terms->curvature += integration_measure * sample_terms.curvature;
            terms->curvature_rate += integration_measure * sample_terms.curvature_rate;
            terms->directed_regularity +=
                integration_measure * sample_terms.directed_regularity;
            terms->traversal_velocity_window +=
                integration_measure * sample_terms.traversal_velocity_window;
            terms->traversal_alignment +=
                integration_measure * sample_terms.traversal_alignment;
            terms->prohibited_traversal +=
                integration_measure * sample_terms.prohibited_traversal;
            terms->runup_curvature +=
                integration_measure * sample_terms.runup_curvature;
        }

        // 几何量梯度（含弧长积分测度）经 β 映射回 grad_c。
        const PolynomialBasis basis(sample.local_time);
        const int c_off = NCOEF * sample.segment;
        const Eigen::Vector2d position_gradient = integration_measure * gradient.position
            + lookahead_position;
        const Eigen::Vector2d velocity_gradient = integration_measure * gradient.velocity
            + sample.dt * density * sample.speed_gradient
            + sample.dt * arc_velocity_gradient + lookahead_velocity;
        const Eigen::Vector2d acceleration_gradient =
            integration_measure * gradient.acceleration;
        const Eigen::Vector2d jerk_gradient = integration_measure * gradient.jerk;
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
        grad_t_explicit(sample.segment) +=
            (density + arc_adjoint) * sample.speed * inv_samples;
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

    if (!ws.minco.generate(
            ws.times, ws.head, ws.tail, ws.waypoints, params_.geometry
        )) {
        grad.setConstant(vars.size(), std::numeric_limits<double>::quiet_NaN());
        return std::numeric_limits<double>::infinity();
    }

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
    const std::vector<GeometricBoundary>& seed_boundaries,
    const std::vector<double>& seed_durations,
    const CostMap& cost_map,
    const DirectionMap& direction_map,
    const TerrainTraversalConstraints& terrain_constraints
) const {
    Result result;
    const int n = static_cast<int>(seed_durations.size());
    if (n < 1 || static_cast<int>(seed_boundaries.size()) != n + 1) {
        result.error = "seed size mismatch: " + std::to_string(seed_boundaries.size())
            + " geometric boundaries for " + std::to_string(n)
            + " segments (need n+1)";
        return result;
    }
    if (!std::all_of(seed_boundaries.begin(), seed_boundaries.end(), [](const auto& boundary) {
            return boundary.position.allFinite() && boundary.tangent.allFinite()
                && boundary.tangent.norm() > 0.0 && std::isfinite(boundary.curvature);
        }) || !std::all_of(seed_durations.begin(), seed_durations.end(), [&](const double time) {
            return std::isfinite(time) && time >= params_.min_segment_time;
        })) {
        result.error = "MINCO seed contains an invalid geometric boundary or segment time";
        return result;
    }
    Workspace ws;
    ws.n_segments = n;
    ws.n_waypoints = n - 1;
    ws.head = seed_boundaries.front();
    ws.tail = seed_boundaries.back();
    ws.head.tangent.normalize();
    ws.tail.tangent.normalize();
    ws.seed_tangents.reserve(seed_boundaries.size());
    for (const GeometricBoundary& boundary : seed_boundaries) {
        ws.seed_tangents.push_back(boundary.tangent.normalized());
    }
    ws.cost_map = &cost_map;
    ws.direction_map = &direction_map;
    ws.terrain_constraints = &terrain_constraints;
    ws.times = seed_durations;
    ws.waypoints.setZero(DIM, std::max(n - 1, 0));

    const int nw = n - 1;
    for (int i = 0; i < nw; ++i) {
        ws.waypoints.col(i) = seed_boundaries[static_cast<size_t>(i + 1)].position;
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
    if (!ws.minco.generate(
            ws.times, ws.head, ws.tail, ws.waypoints, params_.geometry
        )) {
        result.error = "MINCO coefficient system factorization failed for the seed geometry";
        return result;
    }

    LbfgsMinimizer::Options options;
    options.max_iterations = params_.max_iterations;
    options.max_function_evaluations = params_.optimizer.max_function_evaluations;
    options.history_size = params_.optimizer.history_size;
    options.gradient_tolerance = params_.optimizer.gradient_tolerance;
    options.cost_window_relative_tolerance =
        params_.optimizer.cost_window_relative_tolerance;
    options.cost_window_size = params_.optimizer.cost_window_size;
    options.cost_plateau_gradient_tolerance =
        params_.optimizer.cost_plateau_gradient_tolerance;
    options.scaled_step_tolerance = params_.optimizer.scaled_step_tolerance;
    options.step_control = params_.optimizer.step_control;
    options.curvature_cosine_threshold = params_.optimizer.curvature_cosine_threshold;
    options.history_update_min_model_ratio =
        params_.optimizer.history_update_min_model_ratio;
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
    result.final_scaled_gradient_max_block_norm =
        lr.scaled_gradient_max_block_norm;
    result.initial_step_cap = lr.initial_step_cap;
    result.final_step_cap = lr.final_step_cap;
    result.min_step_cap = lr.min_step_cap;
    result.max_step_cap = lr.max_step_cap;
    result.step_cap_shrinks = lr.step_cap_shrinks;
    result.step_cap_expansions = lr.step_cap_expansions;
    result.step_cap_hits = lr.step_cap_hits;
    result.history_updates = lr.history_updates;
    result.history_skips = lr.history_skips;
    result.history_resets = lr.history_resets;
    result.last_relative_cost_reduction = lr.last_relative_cost_reduction;
    result.window_relative_cost_reduction = lr.window_relative_cost_reduction;
    result.cost_plateau_recoveries = lr.cost_plateau_recoveries;
    result.last_actual_reduction = lr.last_actual_reduction;
    result.last_predicted_reduction = lr.last_predicted_reduction;
    result.last_model_ratio = lr.last_model_ratio;

    // 求解终止原因与 incumbent 可用性正交。初值恰好收敛可以发布；除此之外，必须
    // 至少取得一次 accepted progress，避免把失败后的原始 seed 当成优化结果。
    const bool publishable_incumbent = lr.has_finite_incumbent
        && (lr.converged() || lr.made_progress);
    if (!publishable_incumbent) {
        result.error = "L-BFGS terminated with "
            + std::string(LbfgsMinimizer::status_string(lr.status))
            + " without a publishable optimized result (cost=" + std::to_string(lr.cost)
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
