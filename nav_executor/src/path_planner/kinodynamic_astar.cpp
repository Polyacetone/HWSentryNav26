#include <nav_executor/path_planner/kinodynamic_astar.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <queue>
#include <unordered_map>

namespace nav_executor {

namespace {

constexpr double EPS = 1e-9;
constexpr double FORWARD_SPEED_EPS = 1e-4;

double cross_2d(const Eigen::Vector2d& lhs, const Eigen::Vector2d& rhs) {
    return lhs.x() * rhs.y() - lhs.y() * rhs.x();
}

struct StateKey {
    uint64_t packed = 0;
    bool operator==(const StateKey& other) const { return packed == other.packed; }
};

struct StateKeyHash {
    size_t operator()(const StateKey& key) const {
        return std::hash<uint64_t>{}(key.packed);
    }
};

int64_t quantize_component(const double value, const double resolution) {
    return static_cast<int64_t>(std::llround(value / std::max(resolution, EPS)));
}

StateKey make_key(const KinodynamicAstar::State& state, const KinodynamicAstar::Params& params) {
    double theta = std::atan2(state.velocity.y(), state.velocity.x());
    if (theta < 0.0) theta += 2.0 * M_PI;
    const int64_t ix = quantize_component(state.position.x(), params.dedup_xy);
    const int64_t iy = quantize_component(state.position.y(), params.dedup_xy);
    const int64_t theta_bins = std::max<int64_t>(
        1, std::llround(2.0 * M_PI / std::max(params.dedup_theta, EPS))
    );
    const int64_t it = quantize_component(theta, params.dedup_theta) % theta_bins;
    const int64_t iv = quantize_component(state.velocity.norm(), params.dedup_speed);
    return {
        .packed = (static_cast<uint64_t>(ix) & 0xFFFFULL) << 48
            | (static_cast<uint64_t>(iy) & 0xFFFFULL) << 32
            | (static_cast<uint64_t>(it) & 0xFFFFULL) << 16
            | (static_cast<uint64_t>(iv) & 0xFFFFULL),
    };
}

struct SearchNode {
    KinodynamicAstar::State state;
    double g = 0.0;
    double f = 0.0;
    int parent = -1;
    Eigen::Vector2d applied_acceleration = Eigen::Vector2d::Zero();
};

struct OpenEntry {
    double f = 0.0;
    int node_index = -1;
    bool operator>(const OpenEntry& other) const { return f > other.f; }
};

struct GoalShot {
    std::vector<KinodynamicAstar::State> states;
    std::vector<double> durations;
};

using ShotCoefficients = Eigen::Matrix<double, 6, 2>;
using Polynomial = std::vector<double>;

Polynomial polynomial_multiply(const Polynomial& lhs, const Polynomial& rhs) {
    Polynomial result(lhs.size() + rhs.size() - 1, 0.0);
    for (size_t i = 0; i < lhs.size(); ++i) {
        for (size_t j = 0; j < rhs.size(); ++j) result[i + j] += lhs[i] * rhs[j];
    }
    return result;
}

Polynomial polynomial_add(
    const Polynomial& lhs,
    const Polynomial& rhs,
    const double rhs_scale = 1.0
) {
    Polynomial result(std::max(lhs.size(), rhs.size()), 0.0);
    for (size_t i = 0; i < lhs.size(); ++i) result[i] += lhs[i];
    for (size_t i = 0; i < rhs.size(); ++i) result[i] += rhs_scale * rhs[i];
    return result;
}

double binomial(const int n, const int k) {
    if (k < 0 || k > n) return 0.0;
    double result = 1.0;
    for (int i = 1; i <= k; ++i) {
        result *= static_cast<double>(n - k + i) / static_cast<double>(i);
    }
    return result;
}

std::vector<double> interval_bernstein_coefficients(
    const Polynomial& polynomial,
    const double interval_begin,
    const double interval_end
) {
    const int degree = static_cast<int>(polynomial.size()) - 1;
    const double span = interval_end - interval_begin;
    Polynomial local(static_cast<size_t>(degree + 1), 0.0);
    for (int local_order = 0; local_order <= degree; ++local_order) {
        double shifted_coefficient = 0.0;
        for (int global_order = local_order; global_order <= degree; ++global_order) {
            shifted_coefficient += polynomial[static_cast<size_t>(global_order)]
                * binomial(global_order, local_order)
                * std::pow(interval_begin, global_order - local_order);
        }
        local[static_cast<size_t>(local_order)] = shifted_coefficient
            * std::pow(span, local_order);
    }

    std::vector<double> bernstein(static_cast<size_t>(degree + 1), 0.0);
    for (int control = 0; control <= degree; ++control) {
        for (int order = 0; order <= control; ++order) {
            bernstein[static_cast<size_t>(control)] +=
                binomial(control, order) / binomial(degree, order)
                * local[static_cast<size_t>(order)];
        }
    }
    return bernstein;
}

bool polynomial_nonpositive_on_interval(
    const Polynomial& polynomial,
    const double interval_begin,
    const double interval_end
) {
    constexpr double CERTIFICATION_TOLERANCE = 1e-9;
    const auto coefficients = interval_bernstein_coefficients(
        polynomial, interval_begin, interval_end
    );
    return std::all_of(coefficients.begin(), coefficients.end(), [](const double value) {
        return value <= CERTIFICATION_TOLERANCE;
    });
}

ShotCoefficients make_goal_shot_coefficients(
    const KinodynamicAstar::State& start,
    const Eigen::Vector2d& goal,
    const double duration
) {
    const Eigen::Vector2d displacement = goal - start.position;
    const double t2 = duration * duration;
    const double t3 = t2 * duration;
    const double t4 = t3 * duration;
    const double t5 = t4 * duration;
    ShotCoefficients coefficients = ShotCoefficients::Zero();
    coefficients.row(0) = start.position.transpose();
    coefficients.row(1) = start.velocity.transpose();
    coefficients.row(3) = (10.0 * displacement / t3 - 6.0 * start.velocity / t2).transpose();
    coefficients.row(4) = (-15.0 * displacement / t4 + 8.0 * start.velocity / t3).transpose();
    coefficients.row(5) = (6.0 * displacement / t5 - 3.0 * start.velocity / t4).transpose();
    return coefficients;
}

KinodynamicAstar::State evaluate_goal_shot(
    const ShotCoefficients& coefficients,
    const double time,
    Eigen::Vector2d& acceleration
) {
    KinodynamicAstar::State state;
    double power = 1.0;
    for (int order = 0; order < 6; ++order) {
        state.position += coefficients.row(order).transpose() * power;
        power *= time;
    }
    power = 1.0;
    for (int order = 1; order < 6; ++order) {
        state.velocity += static_cast<double>(order)
            * coefficients.row(order).transpose() * power;
        power *= time;
    }
    acceleration.setZero();
    power = 1.0;
    for (int order = 2; order < 6; ++order) {
        acceleration += static_cast<double>(order * (order - 1))
            * coefficients.row(order).transpose() * power;
        power *= time;
    }
    return state;
}

bool goal_shot_dynamically_feasible(
    const ShotCoefficients& coefficients,
    const double duration,
    const int interval_count,
    const Eigen::Vector2d& goal_displacement,
    const KinodynamicAstar::Params::StateLimits& limits
) {
    Polynomial velocity_x(5, 0.0);
    Polynomial velocity_y(5, 0.0);
    Polynomial acceleration_x(4, 0.0);
    Polynomial acceleration_y(4, 0.0);
    for (int order = 1; order < 6; ++order) {
        velocity_x[static_cast<size_t>(order - 1)] = static_cast<double>(order)
            * coefficients(order, 0);
        velocity_y[static_cast<size_t>(order - 1)] = static_cast<double>(order)
            * coefficients(order, 1);
    }
    for (int order = 2; order < 6; ++order) {
        acceleration_x[static_cast<size_t>(order - 2)]
            = static_cast<double>(order * (order - 1)) * coefficients(order, 0);
        acceleration_y[static_cast<size_t>(order - 2)]
            = static_cast<double>(order * (order - 1)) * coefficients(order, 1);
    }

    const Polynomial speed_squared = polynomial_add(
        polynomial_multiply(velocity_x, velocity_x),
        polynomial_multiply(velocity_y, velocity_y)
    );
    const Polynomial acceleration_squared = polynomial_add(
        polynomial_multiply(acceleration_x, acceleration_x),
        polynomial_multiply(acceleration_y, acceleration_y)
    );
    const Polynomial cross = polynomial_add(
        polynomial_multiply(velocity_x, acceleration_y),
        polynomial_multiply(velocity_y, acceleration_x),
        -1.0
    );
    const Polynomial cross_squared = polynomial_multiply(cross, cross);
    Polynomial speed_limit = speed_squared;
    speed_limit[0] -= limits.speed_max * limits.speed_max;
    Polynomial acceleration_limit = acceleration_squared;
    acceleration_limit[0] -= limits.acceleration_max * limits.acceleration_max;
    const Polynomial lateral_limit = polynomial_add(
        cross_squared,
        speed_squared,
        -limits.lateral_acceleration_max * limits.lateral_acceleration_max
    );
    const Polynomial angular_limit = polynomial_add(
        cross_squared,
        polynomial_multiply(speed_squared, speed_squared),
        -limits.angular_velocity_max * limits.angular_velocity_max
    );
    Polynomial goal_projection(5, 0.0);
    for (size_t order = 0; order < goal_projection.size(); ++order) {
        goal_projection[order] = velocity_x[order] * goal_displacement.x()
            + velocity_y[order] * goal_displacement.y();
    }

    const double interval_duration = duration / static_cast<double>(interval_count);
    for (int interval = 0; interval < interval_count; ++interval) {
        const double begin = interval_duration * static_cast<double>(interval);
        const double end = interval_duration * static_cast<double>(interval + 1);
        const auto projection_bounds = interval_bernstein_coefficients(
            goal_projection, begin, end
        );
        const bool final_interval = interval + 1 == interval_count;
        bool strictly_forward = true;
        for (size_t control = 0; control < projection_bounds.size(); ++control) {
            // 终点 v=0 且 a=0，因此四次速度投影的最后两个 Bernstein
            // 控制系数必为零；其余控制系数严格为正即可保证 t<T 时持续前进。
            const bool terminal_zero_control = final_interval
                && control + 2 >= projection_bounds.size();
            if (terminal_zero_control) {
                strictly_forward = projection_bounds[control] >= -1e-9;
            } else if (projection_bounds[control] <= 1e-10) {
                strictly_forward = false;
            }
            if (!strictly_forward) break;
        }
        if (!polynomial_nonpositive_on_interval(speed_limit, begin, end)
            || !polynomial_nonpositive_on_interval(acceleration_limit, begin, end)
            || !polynomial_nonpositive_on_interval(lateral_limit, begin, end)
            || !polynomial_nonpositive_on_interval(angular_limit, begin, end)
            || !strictly_forward) {
            return false;
        }
    }
    return true;
}

bool dynamically_feasible(
    const KinodynamicAstar::State& state,
    const Eigen::Vector2d& acceleration,
    const KinodynamicAstar::Params::StateLimits& limits,
    const bool allow_zero_speed
) {
    const double speed = state.velocity.norm();
    if ((!allow_zero_speed && speed < FORWARD_SPEED_EPS)
        || speed > limits.speed_max + EPS
        || acceleration.norm() > limits.acceleration_max + EPS) {
        return false;
    }
    if (speed < FORWARD_SPEED_EPS) return allow_zero_speed;
    const double lateral_acceleration = std::abs(cross_2d(state.velocity, acceleration)) / speed;
    const double angular_velocity = lateral_acceleration / speed;
    return lateral_acceleration <= limits.lateral_acceleration_max + EPS
        && angular_velocity <= limits.angular_velocity_max + EPS;
}

bool constant_acceleration_dynamically_feasible(
    const Eigen::Vector2d& initial_velocity,
    const Eigen::Vector2d& acceleration,
    const double duration,
    const KinodynamicAstar::Params::StateLimits& limits
) {
    const double acceleration_squared = acceleration.squaredNorm();
    const double closest_time = acceleration_squared > EPS
        ? std::clamp(
            -initial_velocity.dot(acceleration) / acceleration_squared,
            0.0,
            duration
        )
        : 0.0;
    KinodynamicAstar::State closest_state;
    closest_state.velocity = initial_velocity + closest_time * acceleration;
    KinodynamicAstar::State initial_state;
    initial_state.velocity = initial_velocity;
    return dynamically_feasible(initial_state, acceleration, limits, false)
        && dynamically_feasible(closest_state, acceleration, limits, false);
}

std::optional<GoalShot> try_goal_shot(
    const KinodynamicAstar::State& start,
    const Eigen::Vector2d& goal,
    const KinodynamicAstar::Params& params,
    const KinodynamicAstar::TransitionFeasibleFn& transition_feasible
) {
    const double distance = (goal - start.position).norm();
    const double speed = start.velocity.norm();
    const Eigen::Vector2d goal_displacement = goal - start.position;
    const double stopping_distance = speed * speed
        / (2.0 * params.state_limits.acceleration_max);
    if (distance > stopping_distance + speed * params.primitive_duration
            + params.goal_tolerance) {
        return std::nullopt;
    }

    const double minimum_duration = std::max({
        params.primitive_duration,
        1.6 * speed / params.state_limits.acceleration_max,
        distance / params.state_limits.speed_max,
    });
    constexpr std::array<double, 8> DURATION_SCALES {
        1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
    };
    const double primitive_substep = params.primitive_duration
        / static_cast<double>(std::max(params.collision_substeps, 1));
    constexpr double GOAL_SHOT_SPATIAL_STEP = 0.03;
    constexpr double GOAL_SHOT_TIME_STEP = 0.01;
    const double verification_step = std::min({
        primitive_substep,
        GOAL_SHOT_TIME_STEP,
        GOAL_SHOT_SPATIAL_STEP / params.state_limits.speed_max,
    });
    for (const double scale : DURATION_SCALES) {
        const double duration = minimum_duration * scale;
        const int sample_count = std::max(
            params.collision_substeps,
            static_cast<int>(std::ceil(duration / verification_step))
        );
        const double dt = duration / static_cast<double>(sample_count);
        const ShotCoefficients coefficients = make_goal_shot_coefficients(start, goal, duration);
        if (!goal_shot_dynamically_feasible(
                coefficients,
                duration,
                sample_count,
                goal_displacement,
                params.state_limits
            )) {
            continue;
        }
        GoalShot shot;
        shot.states.reserve(static_cast<size_t>(sample_count));
        shot.durations.assign(static_cast<size_t>(sample_count), dt);
        KinodynamicAstar::State previous = start;
        bool feasible = true;
        for (int sample = 1; sample <= sample_count; ++sample) {
            Eigen::Vector2d acceleration;
            KinodynamicAstar::State state = evaluate_goal_shot(
                coefficients, dt * static_cast<double>(sample), acceleration
            );
            const bool final_sample = sample == sample_count;
            if (final_sample) {
                state.position = goal;
                state.velocity.setZero();
            }
            const bool advances_toward_goal = final_sample
                || state.velocity.dot(goal_displacement) > FORWARD_SPEED_EPS * distance;
            if (!dynamically_feasible(
                    state, acceleration, params.state_limits, final_sample
                ) || !advances_toward_goal
                || !transition_feasible(previous, state)) {
                feasible = false;
                break;
            }
            shot.states.push_back(state);
            previous = state;
        }
        if (feasible) return shot;
    }
    return std::nullopt;
}

} // anonymous namespace

KinodynamicAstar::Result KinodynamicAstar::search(
    const State& start,
    const Eigen::Vector2d& goal_position,
    const DijkstraCostToGoal& dijkstra,
    const TransitionFeasibleFn& transition_feasible
) const {
    Result result;
    if (!dijkstra.ready()) {
        result.error = "Dijkstra field not built";
        return result;
    }
    if (!dynamically_feasible(
            start, Eigen::Vector2d::Zero(), params_.state_limits, false
        )) {
        result.error = "start state is not a feasible forward flat state";
        return result;
    }

    const auto heuristic = [&](const State& state) {
        const double distance_cost = dijkstra.at_map(state.position);
        if (std::isinf(distance_cost)) return DijkstraCostToGoal::UNREACHABLE;
        const double travel_time = distance_cost / params_.state_limits.speed_max;
        const double stop_time = state.velocity.norm() / params_.state_limits.acceleration_max;
        return params_.heuristic_weight * std::max(travel_time, stop_time);
    };

    std::vector<SearchNode> nodes;
    nodes.reserve(4096);
    std::unordered_map<StateKey, double, StateKeyHash> best_g;
    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<>> open;
    nodes.push_back({.state = start, .g = 0.0, .f = heuristic(start)});
    best_g[make_key(start, params_)] = 0.0;
    open.push({nodes.front().f, 0});

    const int substeps = std::max(params_.collision_substeps, 1);
    const double sub_dt = params_.primitive_duration / static_cast<double>(substeps);
    int goal_node_index = -1;
    GoalShot goal_shot;

    while (!open.empty()) {
        if (result.expansions >= params_.max_expansions) {
            result.error = "expansion limit reached";
            break;
        }
        const OpenEntry entry = open.top();
        open.pop();
        const SearchNode current = nodes[static_cast<size_t>(entry.node_index)];
        const auto best = best_g.find(make_key(current.state, params_));
        if (best != best_g.end() && current.g > best->second + EPS) continue;
        ++result.expansions;

        if (const auto shot = try_goal_shot(
                current.state, goal_position, params_, transition_feasible
            )) {
            goal_node_index = entry.node_index;
            goal_shot = *shot;
            result.success = true;
            break;
        }

        const double speed = current.state.velocity.norm();
        const Eigen::Vector2d tangent = current.state.velocity / speed;
        const Eigen::Vector2d normal(-tangent.y(), tangent.x());
        for (int tangential_index = 0;
             tangential_index < params_.tangential_accel_samples;
             ++tangential_index) {
            const double tangential_fraction = params_.tangential_accel_samples <= 1
                ? 0.5
                : static_cast<double>(tangential_index)
                    / static_cast<double>(params_.tangential_accel_samples - 1);
            const double tangential_acceleration = params_.state_limits.acceleration_max
                * (2.0 * tangential_fraction - 1.0);
            const double normal_acceleration_max = std::min({
                speed * params_.state_limits.angular_velocity_max,
                params_.state_limits.lateral_acceleration_max,
                std::sqrt(std::max(
                    0.0,
                    params_.state_limits.acceleration_max
                        * params_.state_limits.acceleration_max
                        - tangential_acceleration * tangential_acceleration
                )),
            });
            const int normal_samples = normal_acceleration_max <= EPS
                ? 1 : params_.normal_accel_samples;
            for (int normal_index = 0; normal_index < normal_samples; ++normal_index) {
                const double normal_fraction = normal_samples <= 1
                    ? 0.5
                    : static_cast<double>(normal_index)
                        / static_cast<double>(normal_samples - 1);
                const double normal_acceleration = normal_acceleration_max
                    * (2.0 * normal_fraction - 1.0);
                const Eigen::Vector2d acceleration = tangential_acceleration * tangent
                    + normal_acceleration * normal;

                State state = current.state;
                bool feasible = true;
                for (int substep = 0; substep < substeps; ++substep) {
                    const State previous = state;
                    if (!constant_acceleration_dynamically_feasible(
                            previous.velocity,
                            acceleration,
                            sub_dt,
                            params_.state_limits
                        )) {
                        feasible = false;
                        break;
                    }
                    state.position += state.velocity * sub_dt
                        + 0.5 * acceleration * sub_dt * sub_dt;
                    state.velocity += acceleration * sub_dt;
                    if (!dynamically_feasible(
                            state, acceleration, params_.state_limits, false
                        ) || !transition_feasible(previous, state)) {
                        feasible = false;
                        break;
                    }
                }
                if (!feasible) continue;

                const double h = heuristic(state);
                if (std::isinf(h)) continue;
                const double new_g = current.g + params_.primitive_duration;
                const StateKey key = make_key(state, params_);
                const auto incumbent = best_g.find(key);
                if (incumbent != best_g.end() && new_g >= incumbent->second - EPS) continue;

                best_g[key] = new_g;
                const int node_index = static_cast<int>(nodes.size());
                nodes.push_back({
                    .state = state,
                    .g = new_g,
                    .f = new_g + h,
                    .parent = entry.node_index,
                    .applied_acceleration = acceleration,
                });
                open.push({new_g + h, node_index});
            }
        }
    }

    if (goal_node_index < 0) {
        if (result.error.empty()) result.error = "no feasible kinodynamic path";
        result.success = false;
        return result;
    }

    std::vector<int> reversed_indices;
    for (int index = goal_node_index; index >= 0;
         index = nodes[static_cast<size_t>(index)].parent) {
        reversed_indices.push_back(index);
    }
    std::reverse(reversed_indices.begin(), reversed_indices.end());
    result.states.push_back(start);
    State replay = start;
    for (size_t i = 1; i < reversed_indices.size(); ++i) {
        const SearchNode& node = nodes[static_cast<size_t>(reversed_indices[i])];
        for (int substep = 0; substep < substeps; ++substep) {
            replay.position += replay.velocity * sub_dt
                + 0.5 * node.applied_acceleration * sub_dt * sub_dt;
            replay.velocity += node.applied_acceleration * sub_dt;
            result.states.push_back(replay);
            result.durations.push_back(sub_dt);
        }
    }
    result.states.insert(result.states.end(), goal_shot.states.begin(), goal_shot.states.end());
    result.durations.insert(
        result.durations.end(), goal_shot.durations.begin(), goal_shot.durations.end()
    );
    return result;
}

} // namespace nav_executor
