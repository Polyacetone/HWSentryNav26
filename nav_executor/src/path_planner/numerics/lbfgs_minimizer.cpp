#include <nav_executor/path_planner/numerics/lbfgs_minimizer.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nav_executor {

namespace {

constexpr double GAMMA_MIN = 1e-6;
constexpr double GAMMA_MAX = 1e6;
constexpr double BOUNDARY_FRACTION = 0.9;
constexpr double PREDICTION_EPS_FACTOR = 10.0;

struct HistoryPair {
    Eigen::VectorXd s;
    Eigen::VectorXd y;
    double inverse_curvature = 0.0;
};

bool finite_positive(const double value) {
    return std::isfinite(value) && value > 0.0;
}

void validate_options(const LbfgsMinimizer::Options& options) {
    const auto& trust = options.trust_region;
    if (options.max_iterations <= 0 || options.max_function_evaluations <= 0
        || options.history_size <= 0) {
        throw std::invalid_argument("L-BFGS iteration, evaluation, and history limits must be positive");
    }
    if (!finite_positive(options.gradient_tolerance)
        || !finite_positive(options.scaled_step_tolerance)
        || !finite_positive(options.curvature_relative_threshold)
        || options.curvature_relative_threshold >= 1.0) {
        throw std::invalid_argument("L-BFGS convergence or curvature tolerance is invalid");
    }
    if (!finite_positive(trust.initial_radius)
        || !finite_positive(trust.min_radius)
        || !finite_positive(trust.max_radius)
        || trust.min_radius > trust.initial_radius
        || trust.initial_radius > trust.max_radius) {
        throw std::invalid_argument("L-BFGS trust-region radii are invalid");
    }
    if (!std::isfinite(trust.acceptance_ratio)
        || !std::isfinite(trust.shrink_ratio)
        || !std::isfinite(trust.expansion_ratio)
        || trust.acceptance_ratio < 0.0
        || trust.acceptance_ratio >= trust.shrink_ratio
        || trust.shrink_ratio >= trust.expansion_ratio
        || trust.expansion_ratio >= 1.0) {
        throw std::invalid_argument("L-BFGS trust-region reduction ratios are invalid");
    }
    if (!finite_positive(trust.shrink_factor) || trust.shrink_factor >= 1.0
        || !std::isfinite(trust.expansion_factor) || trust.expansion_factor <= 1.0) {
        throw std::invalid_argument("L-BFGS trust-region radius factors are invalid");
    }
    if (trust.max_consecutive_rejections <= 0
        || trust.history_reset_after_rejections <= 0
        || trust.history_reset_after_rejections >= trust.max_consecutive_rejections) {
        throw std::invalid_argument("L-BFGS trust-region rejection limits are invalid");
    }
    if (!std::isfinite(options.history_acceptance_ratio)
        || options.history_acceptance_ratio < trust.acceptance_ratio
        || options.history_acceptance_ratio >= 1.0) {
        throw std::invalid_argument("L-BFGS history acceptance ratio is invalid");
    }
}

Eigen::VectorXd validate_blocks_and_make_scales(
    const int variable_count,
    const std::vector<LbfgsMinimizer::VariableBlock>& blocks
) {
    if (variable_count <= 0 || blocks.empty()) {
        throw std::invalid_argument("L-BFGS variables must be covered by at least one block");
    }

    Eigen::VectorXd scales(variable_count);
    std::vector<char> covered(static_cast<size_t>(variable_count), 0);
    for (const auto& block : blocks) {
        if (block.offset < 0 || block.size <= 0
            || block.offset > variable_count - block.size
            || !finite_positive(block.scale)) {
            throw std::invalid_argument("L-BFGS variable block is out of range or has an invalid scale");
        }
        for (int i = block.offset; i < block.offset + block.size; ++i) {
            if (covered[static_cast<size_t>(i)]) {
                throw std::invalid_argument("L-BFGS variable blocks overlap");
            }
            covered[static_cast<size_t>(i)] = 1;
            scales(i) = block.scale;
        }
    }
    if (std::any_of(covered.begin(), covered.end(), [](const char value) { return value == 0; })) {
        throw std::invalid_argument("L-BFGS variable blocks do not completely cover x");
    }
    return scales;
}

double block_norm(
    const Eigen::VectorXd& value,
    const std::vector<LbfgsMinimizer::VariableBlock>& blocks
) {
    double norm = 0.0;
    for (const auto& block : blocks) {
        norm = std::max(norm, value.segment(block.offset, block.size).stableNorm());
    }
    return norm;
}

double infinity_norm(const Eigen::VectorXd& value) {
    return value.cwiseAbs().maxCoeff();
}

} // anonymous namespace

std::string_view LbfgsMinimizer::status_string(const Status status) noexcept {
    switch (status) {
        case Status::CONVERGED: return "CONVERGED";
        case Status::MAX_ITERATIONS: return "MAX_ITERATIONS";
        case Status::MAX_EVALUATIONS: return "MAX_EVALUATIONS";
        case Status::TRUST_REGION_TOO_SMALL: return "TRUST_REGION_TOO_SMALL";
        case Status::STAGNATED: return "STAGNATED";
        case Status::INITIAL_EVALUATION_NONFINITE: return "INITIAL_EVALUATION_NONFINITE";
        case Status::NUMERICAL_FAILURE: return "NUMERICAL_FAILURE";
    }
    return "UNKNOWN";
}

LbfgsMinimizer::Result LbfgsMinimizer::minimize(
    const CostFunction& cost_fn,
    Eigen::VectorXd& x,
    const std::vector<VariableBlock>& blocks
) const {
    validate_options(opt_);
    const int variable_count = static_cast<int>(x.size());
    const Eigen::VectorXd variable_scales = validate_blocks_and_make_scales(variable_count, blocks);

    Result result;
    double radius = opt_.trust_region.initial_radius;
    result.initial_radius = radius;
    result.final_radius = radius;
    result.min_radius = radius;
    result.max_radius = radius;

    Eigen::VectorXd raw_gradient = Eigen::VectorXd::Constant(
        variable_count, std::numeric_limits<double>::quiet_NaN()
    );
    double raw_cost = cost_fn(x, raw_gradient);
    result.function_evaluations = 1;
    result.cost = raw_cost;
    const bool initial_gradient_finite = raw_gradient.size() == variable_count
        && raw_gradient.allFinite();
    if (!std::isfinite(raw_cost) || !initial_gradient_finite) {
        result.status = Status::INITIAL_EVALUATION_NONFINITE;
        result.initial_grad_inf_norm = initial_gradient_finite
            ? infinity_norm(raw_gradient)
            : std::numeric_limits<double>::infinity();
        result.grad_inf_norm = result.initial_grad_inf_norm;
        return result;
    }
    result.initial_grad_inf_norm = infinity_norm(raw_gradient);

    const Eigen::VectorXd initial_variable_scaled_gradient =
        variable_scales.array() * raw_gradient.array();
    if (!initial_variable_scaled_gradient.allFinite()) {
        result.status = Status::NUMERICAL_FAILURE;
        result.grad_inf_norm = result.initial_grad_inf_norm;
        result.normalized_scaled_grad_max_block_norm =
            std::numeric_limits<double>::infinity();
        return result;
    }
    const double initial_scaled_block_norm = block_norm(initial_variable_scaled_gradient, blocks);
    if (!std::isfinite(initial_scaled_block_norm)) {
        result.status = Status::NUMERICAL_FAILURE;
        result.grad_inf_norm = result.initial_grad_inf_norm;
        result.normalized_scaled_grad_max_block_norm = initial_scaled_block_norm;
        return result;
    }
    // 精确按初始 scaled gradient 归一化，保证目标整体常数缩放不改变算法轨迹。
    const double objective_scale = initial_scaled_block_norm > 0.0
        ? 1.0 / initial_scaled_block_norm
        : 1.0;
    if (!finite_positive(objective_scale)) {
        result.status = Status::NUMERICAL_FAILURE;
        result.grad_inf_norm = result.initial_grad_inf_norm;
        result.normalized_scaled_grad_max_block_norm = initial_scaled_block_norm;
        return result;
    }
    result.objective_scale = objective_scale;

    Eigen::VectorXd gradient = initial_variable_scaled_gradient * objective_scale;
    if (!gradient.allFinite()) {
        result.status = Status::NUMERICAL_FAILURE;
        result.grad_inf_norm = result.initial_grad_inf_norm;
        result.normalized_scaled_grad_max_block_norm = block_norm(gradient, blocks);
        return result;
    }

    std::deque<HistoryPair> history;
    int accepted_iterations = 0;

    const auto update_incumbent_diagnostics = [&]() {
        result.cost = raw_cost;
        result.grad_inf_norm = infinity_norm(raw_gradient);
        result.normalized_scaled_grad_max_block_norm = block_norm(gradient, blocks);
        result.accepted_iterations = accepted_iterations;
        result.final_radius = radius;
    };
    const auto finish = [&](const Status status) {
        result.status = status;
        update_incumbent_diagnostics();
        return result;
    };
    const auto reset_history = [&]() {
        if (history.empty()) return;
        history.clear();
        ++result.history_resets;
    };
    const auto lbfgs_direction = [&]() -> Eigen::VectorXd {
        Eigen::VectorXd q = gradient;
        const int history_count = static_cast<int>(history.size());
        std::vector<double> alpha(static_cast<size_t>(history_count), 0.0);
        for (int i = history_count - 1; i >= 0; --i) {
            const auto index = static_cast<size_t>(i);
            alpha[index] = history[index].inverse_curvature * history[index].s.dot(q);
            q -= alpha[index] * history[index].y;
        }

        double gamma = 1.0;
        if (!history.empty()) {
            const HistoryPair& latest = history.back();
            const double y_norm = latest.y.stableNorm();
            const double sy = latest.s.dot(latest.y);
            if (std::isfinite(y_norm) && std::isfinite(sy)
                && y_norm > 0.0 && sy > 0.0) {
                const double gamma_candidate = (sy / y_norm) / y_norm;
                if (!std::isnan(gamma_candidate)) {
                    gamma = std::clamp(gamma_candidate, GAMMA_MIN, GAMMA_MAX);
                }
            }
        }

        Eigen::VectorXd direction = gamma * q;
        for (int i = 0; i < history_count; ++i) {
            const auto index = static_cast<size_t>(i);
            const double beta = history[index].inverse_curvature
                * history[index].y.dot(direction);
            direction += (alpha[index] - beta) * history[index].s;
        }
        return -direction;
    };

    while (accepted_iterations < opt_.max_iterations) {
        update_incumbent_diagnostics();
        if (result.normalized_scaled_grad_max_block_norm <= opt_.gradient_tolerance) {
            return finish(Status::CONVERGED);
        }
        if (result.function_evaluations >= opt_.max_function_evaluations) {
            return finish(Status::MAX_EVALUATIONS);
        }

        Eigen::VectorXd direction = lbfgs_direction();
        double directional_derivative = gradient.dot(direction);
        if (!direction.allFinite() || !std::isfinite(directional_derivative)
            || directional_derivative >= 0.0) {
            reset_history();
            direction = -gradient;
            directional_derivative = gradient.dot(direction);
        }
        if (!direction.allFinite() || !std::isfinite(directional_derivative)
            || directional_derivative >= 0.0) {
            return finish(Status::NUMERICAL_FAILURE);
        }

        int consecutive_rejections = 0;
        bool history_reset_at_current = false;
        while (true) {
            if (result.function_evaluations >= opt_.max_function_evaluations) {
                return finish(Status::MAX_EVALUATIONS);
            }

            const double direction_norm = block_norm(direction, blocks);
            if (!std::isfinite(direction_norm) || direction_norm <= 0.0) {
                return finish(Status::NUMERICAL_FAILURE);
            }
            const double alpha = std::min(1.0, radius / direction_norm);
            const Eigen::VectorXd step = alpha * direction;
            const double step_norm = block_norm(step, blocks);
            if (!std::isfinite(step_norm)) {
                return finish(Status::NUMERICAL_FAILURE);
            }
            if (step_norm <= opt_.scaled_step_tolerance) {
                if (!history.empty()) {
                    reset_history();
                    direction = -gradient;
                    continue;
                }
                return finish(Status::STAGNATED);
            }

            const bool boundary_step = step_norm >= BOUNDARY_FRACTION * radius;
            if (boundary_step) ++result.boundary_steps;

            const double step_derivative = gradient.dot(step);
            const double predicted_reduction = -(1.0 - 0.5 * alpha) * step_derivative;
            // floor 只依赖当前局部模型，不让目标常数偏置参与状态机。
            const double prediction_floor = PREDICTION_EPS_FACTOR
                * std::numeric_limits<double>::epsilon() * std::abs(step_derivative);
            if (!std::isfinite(predicted_reduction) || predicted_reduction < 0.0) {
                if (!history.empty()) {
                    reset_history();
                    direction = -gradient;
                    continue;
                }
                return finish(Status::NUMERICAL_FAILURE);
            }
            if (predicted_reduction <= prediction_floor) {
                if (!history.empty()) {
                    reset_history();
                    direction = -gradient;
                    continue;
                }
                return finish(Status::STAGNATED);
            }

            const Eigen::VectorXd trial_x = x
                + (variable_scales.array() * step.array()).matrix();
            Eigen::VectorXd trial_raw_gradient = Eigen::VectorXd::Constant(
                variable_count, std::numeric_limits<double>::quiet_NaN()
            );
            const double trial_raw_cost = cost_fn(trial_x, trial_raw_gradient);
            ++result.function_evaluations;
            ++result.trial_evaluations;

            bool trial_finite = std::isfinite(trial_raw_cost)
                && trial_raw_gradient.size() == variable_count
                && trial_raw_gradient.allFinite();
            Eigen::VectorXd trial_gradient(variable_count);
            if (trial_finite) {
                trial_gradient = (
                    variable_scales.array() * trial_raw_gradient.array()
                ) * objective_scale;
                trial_finite = trial_gradient.allFinite();
            }

            double actual_reduction = -std::numeric_limits<double>::infinity();
            double reduction_ratio = -std::numeric_limits<double>::infinity();
            if (trial_finite) {
                // 先消去 raw 目标中的常数偏置，再施加归一化尺度。
                actual_reduction = (raw_cost - trial_raw_cost) * objective_scale;
                reduction_ratio = actual_reduction / predicted_reduction;
                if (!std::isfinite(reduction_ratio)) trial_finite = false;
            }
            if (!trial_finite) ++result.nonfinite_trials;

            result.last_actual_reduction = actual_reduction;
            result.last_predicted_reduction = predicted_reduction;
            result.last_reduction_ratio = reduction_ratio;

            const double previous_radius = radius;
            const bool attempted_at_min_radius = previous_radius <= opt_.trust_region.min_radius;
            if (!trial_finite || reduction_ratio < opt_.trust_region.shrink_ratio) {
                radius = std::max(
                    opt_.trust_region.min_radius,
                    opt_.trust_region.shrink_factor * step_norm
                );
                if (radius < previous_radius) ++result.radius_shrinks;
            } else if (reduction_ratio > opt_.trust_region.expansion_ratio && boundary_step) {
                radius = std::min(
                    opt_.trust_region.max_radius,
                    opt_.trust_region.expansion_factor * previous_radius
                );
                if (radius > previous_radius) ++result.radius_expansions;
            }
            result.min_radius = std::min(result.min_radius, radius);
            result.max_radius = std::max(result.max_radius, radius);
            result.final_radius = radius;

            const bool accepted = trial_finite
                && reduction_ratio >= opt_.trust_region.acceptance_ratio;
            if (!accepted) {
                ++result.rejected_trials;
                ++consecutive_rejections;

                if (!history_reset_at_current
                    && consecutive_rejections >= opt_.trust_region.history_reset_after_rejections
                    && !history.empty()) {
                    reset_history();
                    history_reset_at_current = true;
                    direction = -gradient;
                }

                if (attempted_at_min_radius
                    || consecutive_rejections >= opt_.trust_region.max_consecutive_rejections) {
                    return finish(Status::TRUST_REGION_TOO_SMALL);
                }
                continue;
            }

            const Eigen::VectorXd y = trial_gradient - gradient;
            bool history_updated = false;
            if (reduction_ratio >= opt_.history_acceptance_ratio
                && y.allFinite()) {
                const double sy = step.dot(y);
                const double s_norm = step.stableNorm();
                const double y_norm = y.stableNorm();
                const double relative_curvature = s_norm > 0.0 && y_norm > 0.0
                    ? (sy / s_norm) / y_norm
                    : 0.0;
                const double inverse_curvature = 1.0 / sy;
                if (std::isfinite(sy) && std::isfinite(relative_curvature)
                    && finite_positive(inverse_curvature)
                    && relative_curvature > opt_.curvature_relative_threshold) {
                    if (static_cast<int>(history.size()) >= opt_.history_size) {
                        history.pop_front();
                    }
                    history.push_back(HistoryPair {
                        .s = step,
                        .y = y,
                        .inverse_curvature = inverse_curvature,
                    });
                    ++result.history_updates;
                    history_updated = true;
                }
            }
            if (!history_updated) ++result.history_skips;

            x = trial_x;
            raw_cost = trial_raw_cost;
            raw_gradient = std::move(trial_raw_gradient);
            gradient = std::move(trial_gradient);
            ++accepted_iterations;
            break;
        }
    }

    update_incumbent_diagnostics();
    if (result.normalized_scaled_grad_max_block_norm <= opt_.gradient_tolerance) {
        return finish(Status::CONVERGED);
    }
    return finish(Status::MAX_ITERATIONS);
}

} // namespace nav_executor
