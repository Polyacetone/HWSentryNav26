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
constexpr double ARMIJO_C1 = 1e-4;

struct HistoryPair {
    Eigen::VectorXd s;
    Eigen::VectorXd y;
    double inverse_curvature = 0.0;
};

bool finite_positive(const double value) {
    return std::isfinite(value) && value > 0.0;
}

void validate_options(const LbfgsMinimizer::Options& options) {
    const auto& step_control = options.step_control;
    if (options.max_iterations <= 0 || options.max_function_evaluations <= 0
        || options.history_size <= 0) {
        throw std::invalid_argument("L-BFGS iteration, evaluation, and history limits must be positive");
    }
    if (!finite_positive(options.gradient_tolerance)
        || !finite_positive(options.cost_window_relative_tolerance)
        || !finite_positive(options.cost_plateau_gradient_tolerance)
        || options.cost_plateau_gradient_tolerance < options.gradient_tolerance
        || !finite_positive(options.scaled_step_tolerance)
        || !finite_positive(options.curvature_cosine_threshold)
        || options.curvature_cosine_threshold >= 1.0) {
        throw std::invalid_argument("L-BFGS convergence or curvature tolerance is invalid");
    }
    if (options.cost_window_size <= 0) {
        throw std::invalid_argument("L-BFGS cost convergence window must be positive");
    }
    if (!finite_positive(step_control.initial_step_cap)
        || !finite_positive(step_control.min_step_cap)
        || !finite_positive(step_control.max_step_cap)
        || step_control.min_step_cap > step_control.initial_step_cap
        || step_control.initial_step_cap > step_control.max_step_cap) {
        throw std::invalid_argument("L-BFGS step caps are invalid");
    }
    if (!finite_positive(step_control.expansion_min_model_ratio)
        || step_control.expansion_min_model_ratio >= 1.0
        || !finite_positive(step_control.backtrack_factor)
        || step_control.backtrack_factor >= 1.0
        || !std::isfinite(step_control.expansion_factor)
        || step_control.expansion_factor <= 1.0) {
        throw std::invalid_argument("L-BFGS step-control factors are invalid");
    }
    if (step_control.max_rejections_per_iteration <= 0
        || step_control.recovery_after_rejections <= 0
        || step_control.recovery_after_rejections
            >= step_control.max_rejections_per_iteration) {
        throw std::invalid_argument("L-BFGS step-control rejection limits are invalid");
    }
    if (!std::isfinite(options.history_update_min_model_ratio)
        || options.history_update_min_model_ratio < 0.0
        || options.history_update_min_model_ratio >= 1.0) {
        throw std::invalid_argument("L-BFGS history model-ratio threshold is invalid");
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
        case Status::CONVERGED_FIRST_ORDER: return "CONVERGED_FIRST_ORDER";
        case Status::CONVERGED_COST: return "CONVERGED_COST";
        case Status::ITERATION_LIMIT: return "ITERATION_LIMIT";
        case Status::EVALUATION_LIMIT: return "EVALUATION_LIMIT";
        case Status::STALLED_SMALL_STEP: return "STALLED_SMALL_STEP";
        case Status::STALLED_LINE_SEARCH: return "STALLED_LINE_SEARCH";
        case Status::STALLED_COST_PLATEAU: return "STALLED_COST_PLATEAU";
        case Status::INVALID_INITIAL_EVALUATION: return "INVALID_INITIAL_EVALUATION";
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
    double step_cap = opt_.step_control.initial_step_cap;
    result.initial_step_cap = step_cap;
    result.final_step_cap = step_cap;
    result.min_step_cap = step_cap;
    result.max_step_cap = step_cap;

    Eigen::VectorXd raw_gradient = Eigen::VectorXd::Constant(
        variable_count, std::numeric_limits<double>::quiet_NaN()
    );
    double raw_cost = cost_fn(x, raw_gradient);
    const double initial_raw_cost = raw_cost;
    result.function_evaluations = 1;
    result.cost = raw_cost;
    const bool initial_gradient_finite = raw_gradient.size() == variable_count
        && raw_gradient.allFinite();
    if (!std::isfinite(raw_cost) || !initial_gradient_finite) {
        result.status = Status::INVALID_INITIAL_EVALUATION;
        result.initial_grad_inf_norm = initial_gradient_finite
            ? infinity_norm(raw_gradient)
            : std::numeric_limits<double>::infinity();
        result.grad_inf_norm = result.initial_grad_inf_norm;
        result.scaled_gradient_max_block_norm = std::numeric_limits<double>::infinity();
        return result;
    }
    result.has_finite_incumbent = true;
    result.initial_grad_inf_norm = infinity_norm(raw_gradient);

    const Eigen::VectorXd initial_variable_scaled_gradient =
        variable_scales.array() * raw_gradient.array();
    if (!initial_variable_scaled_gradient.allFinite()) {
        result.status = Status::NUMERICAL_FAILURE;
        result.grad_inf_norm = result.initial_grad_inf_norm;
        result.scaled_gradient_max_block_norm = std::numeric_limits<double>::infinity();
        return result;
    }
    const double initial_scaled_block_norm = block_norm(initial_variable_scaled_gradient, blocks);
    if (!std::isfinite(initial_scaled_block_norm)) {
        result.status = Status::NUMERICAL_FAILURE;
        result.grad_inf_norm = result.initial_grad_inf_norm;
        result.scaled_gradient_max_block_norm = initial_scaled_block_norm;
        return result;
    }
    // 初始 scaled gradient 只用于调理算法内部的目标量级，使目标整体常数缩放不改变
    // L-BFGS 与 step-control 的轨迹。收敛判据不能复用这一固定基准，否则种子处一次性
    // 的巨大罚项会永久放宽后续的一阶驻点要求。
    const double objective_scale = initial_scaled_block_norm > 0.0
        ? 1.0 / initial_scaled_block_norm
        : 1.0;
    if (!finite_positive(objective_scale)) {
        result.status = Status::NUMERICAL_FAILURE;
        result.grad_inf_norm = result.initial_grad_inf_norm;
        result.scaled_gradient_max_block_norm = initial_scaled_block_norm;
        return result;
    }
    result.objective_scale = objective_scale;

    Eigen::VectorXd gradient = initial_variable_scaled_gradient * objective_scale;
    if (!gradient.allFinite()) {
        result.status = Status::NUMERICAL_FAILURE;
        result.grad_inf_norm = result.initial_grad_inf_norm;
        result.scaled_gradient_max_block_norm = initial_scaled_block_norm;
        return result;
    }

    enum class SolverMode { LBFGS, STEEPEST_DESCENT_RECOVERY };

    std::deque<HistoryPair> history;
    std::deque<double> accepted_cost_window {raw_cost};
    int accepted_iterations = 0;
    double last_relative_cost_reduction = 0.0;
    double window_relative_cost_reduction = 0.0;
    SolverMode mode = SolverMode::LBFGS;

    const auto update_incumbent_diagnostics = [&]() {
        const Eigen::VectorXd variable_scaled_gradient =
            variable_scales.array() * raw_gradient.array();
        result.cost = raw_cost;
        result.grad_inf_norm = infinity_norm(raw_gradient);
        result.scaled_gradient_max_block_norm = block_norm(variable_scaled_gradient, blocks);
        result.accepted_iterations = accepted_iterations;
        const double relative_progress = (initial_raw_cost - raw_cost)
            / std::max(1.0, std::abs(initial_raw_cost));
        result.made_progress = std::isfinite(relative_progress)
            && relative_progress >= opt_.cost_window_relative_tolerance;
        result.last_relative_cost_reduction = last_relative_cost_reduction;
        result.window_relative_cost_reduction = window_relative_cost_reduction;
        result.final_step_cap = step_cap;
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
    const auto reset_cost_window = [&]() {
        accepted_cost_window.clear();
        accepted_cost_window.push_back(raw_cost);
        window_relative_cost_reduction = 0.0;
    };
    const auto update_window_reduction = [&]() {
        if (accepted_cost_window.size()
            < static_cast<size_t>(opt_.cost_window_size + 1)) {
            window_relative_cost_reduction = 0.0;
            return false;
        }
        const double oldest_cost = accepted_cost_window.front();
        window_relative_cost_reduction = (oldest_cost - raw_cost)
            / std::max(1.0, std::abs(oldest_cost));
        return window_relative_cost_reduction <= opt_.cost_window_relative_tolerance;
    };
    const auto enter_recovery = [&]() {
        reset_history();
        reset_cost_window();
        mode = SolverMode::STEEPEST_DESCENT_RECOVERY;
        ++result.cost_plateau_recoveries;
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
    const auto steepest_direction = [&]() -> Eigen::VectorXd {
        const double norm = block_norm(gradient, blocks);
        Eigen::VectorXd direction = -gradient;
        if (finite_positive(norm)) direction /= norm;
        return direction;
    };

    while (accepted_iterations < opt_.max_iterations) {
        update_incumbent_diagnostics();
        if (result.scaled_gradient_max_block_norm <= opt_.gradient_tolerance) {
            return finish(Status::CONVERGED_FIRST_ORDER);
        }
        const bool cost_window_full = accepted_cost_window.size()
            >= static_cast<size_t>(opt_.cost_window_size + 1);
        if (update_window_reduction()) {
            if (result.scaled_gradient_max_block_norm
                <= opt_.cost_plateau_gradient_tolerance) {
                return finish(Status::CONVERGED_COST);
            }
            if (mode == SolverMode::STEEPEST_DESCENT_RECOVERY) {
                return finish(Status::STALLED_COST_PLATEAU);
            }
            enter_recovery();
        } else if (cost_window_full
            && mode == SolverMode::STEEPEST_DESCENT_RECOVERY) {
            // 恢复窗口取得了足够的累计收益，重新允许构建 L-BFGS 历史。
            mode = SolverMode::LBFGS;
            reset_cost_window();
        }
        if (result.function_evaluations >= opt_.max_function_evaluations) {
            return finish(Status::EVALUATION_LIMIT);
        }

        bool using_steepest_direction =
            mode == SolverMode::STEEPEST_DESCENT_RECOVERY || history.empty();
        Eigen::VectorXd direction = using_steepest_direction
            ? steepest_direction() : lbfgs_direction();
        double directional_derivative = gradient.dot(direction);
        if (!direction.allFinite() || !std::isfinite(directional_derivative)
            || directional_derivative >= 0.0) {
            if (using_steepest_direction) {
                return finish(Status::NUMERICAL_FAILURE);
            }
            reset_history();
            reset_cost_window();
            mode = SolverMode::STEEPEST_DESCENT_RECOVERY;
            direction = steepest_direction();
            using_steepest_direction = true;
            directional_derivative = gradient.dot(direction);
        }
        if (!direction.allFinite() || !std::isfinite(directional_derivative)
            || directional_derivative >= 0.0) {
            return finish(Status::NUMERICAL_FAILURE);
        }

        int consecutive_rejections = 0;
        while (true) {
            if (result.function_evaluations >= opt_.max_function_evaluations) {
                return finish(Status::EVALUATION_LIMIT);
            }

            const double direction_norm = block_norm(direction, blocks);
            if (!std::isfinite(direction_norm) || direction_norm <= 0.0) {
                return finish(Status::NUMERICAL_FAILURE);
            }
            const double alpha = std::min(1.0, step_cap / direction_norm);
            const Eigen::VectorXd step = alpha * direction;
            const double step_norm = block_norm(step, blocks);
            if (!std::isfinite(step_norm)) {
                return finish(Status::NUMERICAL_FAILURE);
            }
            if (step_norm <= opt_.scaled_step_tolerance) {
                if (!using_steepest_direction) {
                    reset_history();
                    reset_cost_window();
                    mode = SolverMode::STEEPEST_DESCENT_RECOVERY;
                    direction = steepest_direction();
                    using_steepest_direction = true;
                    continue;
                }
                return finish(Status::STALLED_SMALL_STEP);
            }

            const bool capped_step = step_norm >= BOUNDARY_FRACTION * step_cap;
            if (capped_step) ++result.step_cap_hits;

            const double step_derivative = gradient.dot(step);
            const double predicted_reduction = -(1.0 - 0.5 * alpha) * step_derivative;
            if (!std::isfinite(predicted_reduction) || predicted_reduction <= 0.0) {
                if (!using_steepest_direction) {
                    reset_history();
                    reset_cost_window();
                    mode = SolverMode::STEEPEST_DESCENT_RECOVERY;
                    direction = steepest_direction();
                    using_steepest_direction = true;
                    continue;
                }
                return finish(Status::NUMERICAL_FAILURE);
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
            double model_ratio = -std::numeric_limits<double>::infinity();
            if (trial_finite) {
                // 先消去 raw 目标中的常数偏置，再施加归一化尺度。
                actual_reduction = (raw_cost - trial_raw_cost) * objective_scale;
                model_ratio = actual_reduction / predicted_reduction;
                if (!std::isfinite(model_ratio)) trial_finite = false;
            }
            if (!trial_finite) ++result.nonfinite_trials;

            result.last_actual_reduction = actual_reduction;
            result.last_predicted_reduction = predicted_reduction;
            result.last_model_ratio = model_ratio;

            const double previous_step_cap = step_cap;
            const bool attempted_at_min_step_cap =
                previous_step_cap <= opt_.step_control.min_step_cap;
            const bool accepted = trial_finite
                && actual_reduction >= ARMIJO_C1 * (-step_derivative);
            if (!accepted) {
                step_cap = std::max(
                    opt_.step_control.min_step_cap,
                    opt_.step_control.backtrack_factor
                        * std::min(previous_step_cap, step_norm)
                );
                if (step_cap < previous_step_cap) ++result.step_cap_shrinks;
            } else if (model_ratio > opt_.step_control.expansion_min_model_ratio
                && capped_step) {
                step_cap = std::min(
                    opt_.step_control.max_step_cap,
                    opt_.step_control.expansion_factor * previous_step_cap
                );
                if (step_cap > previous_step_cap) ++result.step_cap_expansions;
            }
            result.min_step_cap = std::min(result.min_step_cap, step_cap);
            result.max_step_cap = std::max(result.max_step_cap, step_cap);
            result.final_step_cap = step_cap;

            if (!accepted) {
                ++result.rejected_trials;
                ++consecutive_rejections;

                const bool repeated_interior_step = alpha == 1.0
                    && step_cap >= direction_norm;
                if (!using_steepest_direction
                    && (attempted_at_min_step_cap || repeated_interior_step
                        || consecutive_rejections
                            >= opt_.step_control.recovery_after_rejections)) {
                    reset_history();
                    reset_cost_window();
                    mode = SolverMode::STEEPEST_DESCENT_RECOVERY;
                    direction = steepest_direction();
                    using_steepest_direction = true;
                    consecutive_rejections = 0;
                    continue;
                }
                if (attempted_at_min_step_cap || repeated_interior_step
                    || consecutive_rejections
                        >= opt_.step_control.max_rejections_per_iteration) {
                    return finish(Status::STALLED_LINE_SEARCH);
                }
                continue;
            }

            const Eigen::VectorXd y = trial_gradient - gradient;
            const bool history_eligible = mode == SolverMode::LBFGS
                && consecutive_rejections == 0
                && model_ratio >= opt_.history_update_min_model_ratio;
            if (!history_eligible) {
                reset_history();
            }
            bool history_updated = false;
            if (history_eligible && y.allFinite()) {
                const double sy = step.dot(y);
                const double s_norm = step.stableNorm();
                const double y_norm = y.stableNorm();
                const double curvature_cosine = s_norm > 0.0 && y_norm > 0.0
                    ? (sy / s_norm) / y_norm
                    : 0.0;
                const double inverse_curvature = 1.0 / sy;
                if (std::isfinite(sy) && std::isfinite(curvature_cosine)
                    && finite_positive(inverse_curvature)
                    && curvature_cosine > opt_.curvature_cosine_threshold) {
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

            last_relative_cost_reduction = (raw_cost - trial_raw_cost)
                / std::max(1.0, std::abs(raw_cost));
            x = trial_x;
            raw_cost = trial_raw_cost;
            raw_gradient = std::move(trial_raw_gradient);
            gradient = std::move(trial_gradient);
            ++accepted_iterations;

            accepted_cost_window.push_back(raw_cost);
            while (accepted_cost_window.size()
                > static_cast<size_t>(opt_.cost_window_size + 1)) {
                accepted_cost_window.pop_front();
            }
            if (mode == SolverMode::STEEPEST_DESCENT_RECOVERY
                && model_ratio >= opt_.history_update_min_model_ratio) {
                // 恢复是否成功由局部模型可信度决定，不能再依赖含常数基线的总 cost。
                mode = SolverMode::LBFGS;
                reset_cost_window();
            }
            break;
        }
    }

    update_incumbent_diagnostics();
    if (result.scaled_gradient_max_block_norm <= opt_.gradient_tolerance) {
        return finish(Status::CONVERGED_FIRST_ORDER);
    }
    if (update_window_reduction()
        && result.scaled_gradient_max_block_norm
            <= opt_.cost_plateau_gradient_tolerance) {
        return finish(Status::CONVERGED_COST);
    }
    return finish(Status::ITERATION_LIMIT);
}

} // namespace nav_executor
