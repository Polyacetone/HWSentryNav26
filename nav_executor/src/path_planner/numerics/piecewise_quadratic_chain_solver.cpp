#include <nav_executor/path_planner/numerics/piecewise_quadratic_chain_solver.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>

namespace nav_executor {

namespace {

using Clock = std::chrono::steady_clock;

constexpr double INF = std::numeric_limits<double>::infinity();

struct Knot {
    double x = 0.0;
    double left = 0.0;
    double right = 0.0;
};

// 凸分段二次函数的次梯度图。端点的无穷跳跃编码定义域 indicator。
class ConvexDerivative {
public:
    explicit ConvexDerivative(std::vector<Knot> knots) : knots_(std::move(knots)) {
        if (knots_.size() < 2) return;
        double previous = knots_.front().right;
        for (size_t i = 1; i < knots_.size(); ++i) {
            repair_roundoff(previous, knots_[i].left);
            previous = knots_[i].left;
            if (i + 1 < knots_.size()) {
                repair_roundoff(previous, knots_[i].right);
                previous = knots_[i].right;
            }
        }
    }

    [[nodiscard]] static ConvexDerivative singleton(const double x) {
        return ConvexDerivative({{x, -INF, INF}});
    }

    [[nodiscard]] double lower_domain() const { return knots_.front().x; }
    [[nodiscard]] double upper_domain() const { return knots_.back().x; }
    [[nodiscard]] size_t breakpoint_count() const { return knots_.size(); }
    [[nodiscard]] const std::vector<Knot>& knots() const { return knots_; }

    [[nodiscard]] double left_limit(const double x) const {
        return limits_at(x).first;
    }

    [[nodiscard]] double right_limit(const double x) const {
        return limits_at(x).second;
    }

    [[nodiscard]] std::pair<double, double> minimizer_interval() const {
        if (knots_.size() == 1) return {knots_.front().x, knots_.front().x};

        double lower = knots_.back().x;
        for (size_t i = 0; i < knots_.size(); ++i) {
            const Knot& knot = knots_[i];
            if (knot.right >= 0.0) {
                lower = knot.x;
                break;
            }
            if (i + 1 >= knots_.size()) continue;
            const Knot& next = knots_[i + 1];
            if (next.left >= 0.0) {
                const double fraction = -knot.right / (next.left - knot.right);
                lower = std::lerp(knot.x, next.x, fraction);
                break;
            }
        }

        double upper = knots_.front().x;
        for (size_t reverse = knots_.size(); reverse > 0; --reverse) {
            const size_t i = reverse - 1;
            const Knot& knot = knots_[i];
            if (knot.left <= 0.0) {
                upper = knot.x;
                break;
            }
            if (i == 0) continue;
            const Knot& previous = knots_[i - 1];
            if (previous.right <= 0.0) {
                const double fraction = -previous.right
                    / (knot.left - previous.right);
                upper = std::lerp(previous.x, knot.x, fraction);
                break;
            }
        }
        return {lower, upper};
    }

    [[nodiscard]] bool valid() const {
        if (knots_.empty()) return false;
        if (!std::isfinite(knots_.front().x)
            || knots_.front().left != -INF
            || knots_.back().right != INF) {
            return false;
        }
        if (knots_.size() == 1) {
            return knots_.front().right == INF;
        }
        for (size_t i = 0; i < knots_.size(); ++i) {
            const Knot& knot = knots_[i];
            if (!std::isfinite(knot.x) || std::isnan(knot.left)
                || std::isnan(knot.right) || knot.left > knot.right) {
                return false;
            }
            if ((i > 0 && !std::isfinite(knot.left))
                || (i + 1 < knots_.size() && !std::isfinite(knot.right))) {
                return false;
            }
            if (i > 0 && (!(knots_[i - 1].x < knot.x)
                    || knots_[i - 1].right > knot.left)) {
                return false;
            }
        }
        return true;
    }

private:
    static void repair_roundoff(const double previous, double& current) {
        if (current >= previous) return;
        const double scale = std::max({1.0, std::abs(previous), std::abs(current)});
        if (previous - current
            <= 64.0 * std::numeric_limits<double>::epsilon() * scale) {
            current = previous;
        }
    }

    [[nodiscard]] std::pair<double, double> limits_at(const double x) const {
        const auto it = std::lower_bound(
            knots_.begin(), knots_.end(), x,
            [](const Knot& knot, const double value) { return knot.x < value; }
        );
        if (it != knots_.end() && it->x == x) return {it->left, it->right};
        if (it == knots_.begin() || it == knots_.end()) return {NAN, NAN};
        const Knot& previous = *(it - 1);
        const double fraction = (x - previous.x) / (it->x - previous.x);
        const double value = std::lerp(previous.right, it->left, fraction);
        return {value, value};
    }

    std::vector<Knot> knots_;
};

ConvexDerivative canonicalize(std::vector<Knot> knots) {
    std::vector<Knot> merged;
    merged.reserve(knots.size());
    for (const Knot& knot : knots) {
        if (!merged.empty() && merged.back().x == knot.x) {
            // 同坐标事件按次梯度图遍历顺序合并，保留最外侧极限。
            merged.back().right = knot.right;
        } else {
            merged.push_back(knot);
        }
    }
    return ConvexDerivative(std::move(merged));
}

ConvexDerivative moving_min(
    const ConvexDerivative& source,
    const double distance
) {
    const auto [minimizer_lower, minimizer_upper] = source.minimizer_interval();
    std::vector<Knot> knots;
    knots.reserve(source.breakpoint_count() + 2);
    for (const Knot& knot : source.knots()) {
        if (knot.x < minimizer_lower) {
            knots.push_back({knot.x - distance, knot.left, knot.right});
        }
    }
    knots.push_back({
        minimizer_lower - distance,
        std::min(source.left_limit(minimizer_lower), 0.0),
        0.0,
    });
    knots.push_back({
        minimizer_upper + distance,
        0.0,
        std::max(source.right_limit(minimizer_upper), 0.0),
    });
    for (const Knot& knot : source.knots()) {
        if (knot.x > minimizer_upper) {
            knots.push_back({knot.x + distance, knot.left, knot.right});
        }
    }
    return canonicalize(std::move(knots));
}

std::optional<ConvexDerivative> clip_domain(
    const ConvexDerivative& source,
    const double box_lower,
    const double box_upper
) {
    const double lower = std::max(source.lower_domain(), box_lower);
    const double upper = std::min(source.upper_domain(), box_upper);
    if (lower > upper) return std::nullopt;
    if (lower == upper) return ConvexDerivative::singleton(lower);

    std::vector<Knot> knots;
    knots.reserve(source.breakpoint_count() + 2);
    knots.push_back({lower, -INF, source.right_limit(lower)});
    for (const Knot& knot : source.knots()) {
        if (knot.x > lower && knot.x < upper) knots.push_back(knot);
    }
    knots.push_back({upper, source.left_limit(upper), INF});
    return canonicalize(std::move(knots));
}

struct SlopeEvent {
    double x = 0.0;
    double delta = 0.0;
    double slope_after = 0.0;
    double intercept_after = 0.0;
};

class NodeCostDerivative {
public:
    NodeCostDerivative(
        const double linear_reward,
        const std::vector<ChainProblem::SoftWindow>& windows
    ) : initial_intercept_(-linear_reward) {
        std::vector<SlopeEvent> events;
        for (const ChainProblem::SoftWindow& window : windows) {
            if (window.weight == 0.0) continue;
            if (window.lower == window.upper) {
                initial_slope_ += 2.0 * window.weight;
                initial_intercept_ -= 2.0 * window.weight * window.lower;
                continue;
            }
            initial_slope_ += 2.0 * window.weight;
            initial_intercept_ -= 2.0 * window.weight * window.lower;
            events.push_back({window.lower, -2.0 * window.weight, 0.0, 0.0});
            events.push_back({window.upper, 2.0 * window.weight, 0.0, 0.0});
        }
        std::sort(events.begin(), events.end(), [](const SlopeEvent& a, const SlopeEvent& b) {
            return a.x < b.x;
        });

        double slope = initial_slope_;
        double intercept = initial_intercept_;
        for (size_t begin = 0; begin < events.size();) {
            size_t end = begin + 1;
            double delta = events[begin].delta;
            while (end < events.size() && events[end].x == events[begin].x) {
                delta += events[end].delta;
                ++end;
            }
            if (delta != 0.0) {
                intercept -= delta * events[begin].x;
                slope += delta;
                events_.push_back({events[begin].x, delta, slope, intercept});
            }
            begin = end;
        }
    }

    [[nodiscard]] double value(const double x) const {
        const auto it = std::lower_bound(
            events_.begin(), events_.end(), x,
            [](const SlopeEvent& event, const double value) { return event.x < value; }
        );
        if (it == events_.begin()) return initial_slope_ * x + initial_intercept_;
        const SlopeEvent& event = *(it - 1);
        return event.slope_after * x + event.intercept_after;
    }

    [[nodiscard]] const std::vector<SlopeEvent>& events() const { return events_; }

private:
    double initial_slope_ = 0.0;
    double initial_intercept_ = 0.0;
    std::vector<SlopeEvent> events_;
};

ConvexDerivative add_node_cost(
    const ConvexDerivative& source,
    const NodeCostDerivative& cost
) {
    if (source.lower_domain() == source.upper_domain()) return source;

    std::vector<double> coordinates;
    coordinates.reserve(source.breakpoint_count() + cost.events().size());
    size_t event_index = 0;
    for (const Knot& knot : source.knots()) {
        while (event_index < cost.events().size()
            && cost.events()[event_index].x < knot.x) {
            const double event_x = cost.events()[event_index].x;
            if (event_x > source.lower_domain()) coordinates.push_back(event_x);
            ++event_index;
        }
        coordinates.push_back(knot.x);
        while (event_index < cost.events().size()
            && cost.events()[event_index].x == knot.x) {
            ++event_index;
        }
    }

    std::vector<Knot> knots;
    knots.reserve(coordinates.size());
    size_t source_index = 0;
    for (const double x : coordinates) {
        while (source_index + 1 < source.knots().size()
            && source.knots()[source_index + 1].x <= x) {
            ++source_index;
        }
        double source_left = 0.0;
        double source_right = 0.0;
        if (source.knots()[source_index].x == x) {
            source_left = source.knots()[source_index].left;
            source_right = source.knots()[source_index].right;
        } else {
            const Knot& lower = source.knots()[source_index];
            const Knot& upper = source.knots()[source_index + 1];
            const double fraction = (x - lower.x) / (upper.x - lower.x);
            source_left = std::lerp(lower.right, upper.left, fraction);
            source_right = source_left;
        }
        const double node_derivative = cost.value(x);
        knots.push_back({
            x,
            x == source.lower_domain() ? -INF : source_left + node_derivative,
            x == source.upper_domain() ? INF : source_right + node_derivative,
        });
    }
    return ConvexDerivative(std::move(knots));
}

struct BacktrackState {
    double lower_domain = 0.0;
    double upper_domain = 0.0;
    double upper_minimizer = 0.0;
};

bool finite_nonnegative(const double value) {
    return std::isfinite(value) && value >= 0.0;
}

std::optional<std::string> validate_problem(const ChainProblem& problem) {
    const size_t node_count = problem.node_upper.size();
    if (node_count == 0 || problem.linear_reward.size() != node_count
        || problem.step_limit.size() + 1 != node_count) {
        return "chain problem dimensions are inconsistent";
    }
    if (!finite_nonnegative(problem.initial_value)) {
        return "chain initial value must be finite and non-negative";
    }
    for (const double value : problem.linear_reward) {
        if (!finite_nonnegative(value)) return "chain linear rewards must be finite and non-negative";
    }
    for (const double value : problem.node_upper) {
        if (!finite_nonnegative(value)) return "chain node upper bounds must be finite and non-negative";
    }
    for (const double value : problem.step_limit) {
        if (!finite_nonnegative(value)) return "chain step limits must be finite and non-negative";
    }
    for (const ChainProblem::SoftWindow& window : problem.soft_windows) {
        if (window.node_index >= node_count || !finite_nonnegative(window.lower)
            || !finite_nonnegative(window.upper) || window.lower > window.upper
            || !finite_nonnegative(window.weight)) {
            return "chain soft window is invalid";
        }
    }
    return std::nullopt;
}

double objective_value(const ChainProblem& problem, const std::vector<double>& value) {
    double objective = 0.0;
    for (size_t i = 0; i < value.size(); ++i) {
        objective -= problem.linear_reward[i] * value[i];
    }
    for (const ChainProblem::SoftWindow& window : problem.soft_windows) {
        const double under = std::max(window.lower - value[window.node_index], 0.0);
        const double over = std::max(value[window.node_index] - window.upper, 0.0);
        objective += window.weight * (under * under + over * over);
    }
    return objective;
}

} // anonymous namespace

PiecewiseQuadraticChainSolver::Result PiecewiseQuadraticChainSolver::solve(
    const ChainProblem& problem
) {
    const auto start = Clock::now();
    Result result;
    const auto finish = [&]() {
        result.solve_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - start
        ).count();
        return result;
    };

    if (const auto error = validate_problem(problem)) {
        result.error = *error;
        return finish();
    }

    const size_t node_count = problem.node_upper.size();
    std::vector<std::vector<ChainProblem::SoftWindow>> windows_by_node(node_count);
    for (const ChainProblem::SoftWindow& window : problem.soft_windows) {
        windows_by_node[window.node_index].push_back(window);
    }

    std::vector<BacktrackState> backtrack(node_count);
    ConvexDerivative value = ConvexDerivative::singleton(problem.initial_value);
    auto clipped = clip_domain(value, 0.0, problem.node_upper.front());
    if (!clipped) {
        result.status = Status::INFEASIBLE;
        result.error = "initial value violates the first node bound";
        return finish();
    }
    value = std::move(*clipped);
    if (node_count == 1) {
        clipped = clip_domain(value, 0.0, 0.0);
        if (!clipped) {
            result.status = Status::INFEASIBLE;
            result.error = "the single node cannot satisfy both endpoint constraints";
            return finish();
        }
        value = std::move(*clipped);
    }
    value = add_node_cost(
        value, NodeCostDerivative(problem.linear_reward.front(), windows_by_node.front())
    );
    if (!value.valid()) {
        result.error = "chain value-function invariant failed at node 0";
        return finish();
    }
    backtrack.front() = {
        value.lower_domain(), value.upper_domain(), value.minimizer_interval().second,
    };
    result.max_breakpoints = static_cast<int>(value.breakpoint_count());

    for (size_t i = 1; i < node_count; ++i) {
        value = moving_min(value, problem.step_limit[i - 1]);
        if (!value.valid()) {
            result.error = "chain moving-min produced an invalid value function at node "
                + std::to_string(i);
            return finish();
        }
        clipped = clip_domain(value, 0.0, problem.node_upper[i]);
        if (!clipped) {
            result.status = Status::INFEASIBLE;
            result.error = "reachable domain and node bound do not intersect at node "
                + std::to_string(i);
            return finish();
        }
        value = std::move(*clipped);
        if (i + 1 == node_count) {
            clipped = clip_domain(value, 0.0, 0.0);
            if (!clipped) {
                result.status = Status::INFEASIBLE;
                result.error = "terminal value is unreachable";
                return finish();
            }
            value = std::move(*clipped);
        }
        value = add_node_cost(
            value, NodeCostDerivative(problem.linear_reward[i], windows_by_node[i])
        );
        if (!value.valid()) {
            result.error = "chain value-function invariant failed at node "
                + std::to_string(i);
            return finish();
        }
        backtrack[i] = {
            value.lower_domain(), value.upper_domain(), value.minimizer_interval().second,
        };
        result.max_breakpoints = std::max(
            result.max_breakpoints, static_cast<int>(value.breakpoint_count())
        );
    }

    result.value.assign(node_count, 0.0);
    for (size_t next = node_count - 1; next > 0; --next) {
        const size_t i = next - 1;
        const BacktrackState& state = backtrack[i];
        double lower = std::max(
            state.lower_domain, result.value[next] - problem.step_limit[i]
        );
        double upper = std::min(
            state.upper_domain, result.value[next] + problem.step_limit[i]
        );
        if (lower > upper) {
            const double scale = std::max({1.0, std::abs(lower), std::abs(upper)});
            const double roundoff = 16.0 * std::numeric_limits<double>::epsilon() * scale;
            if (lower - upper > roundoff) {
                result.value.clear();
                result.error = "chain backtracking domain is empty at node "
                    + std::to_string(i);
                return finish();
            }
            const double boundary = std::midpoint(lower, upper);
            lower = boundary;
            upper = boundary;
        }
        result.value[i] = std::clamp(state.upper_minimizer, lower, upper);
    }

    result.objective = objective_value(problem, result.value);
    if (!std::isfinite(result.objective)) {
        result.value.clear();
        result.error = "chain objective is non-finite";
        return finish();
    }
    result.status = Status::OPTIMAL;
    return finish();
}

} // namespace nav_executor
