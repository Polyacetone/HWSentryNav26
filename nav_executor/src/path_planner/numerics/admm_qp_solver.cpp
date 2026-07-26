#include <nav_executor/path_planner/numerics/admm_qp_solver.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/QR>

namespace nav_executor {

namespace {

using Clock = std::chrono::steady_clock;

Eigen::VectorXd project_interval(
    const Eigen::VectorXd& value,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper
) {
    Eigen::VectorXd projected = value;
    for (Eigen::Index i = 0; i < projected.size(); ++i) {
        projected(i) = std::clamp(projected(i), lower(i), upper(i));
    }
    return projected;
}

double max_constraint_violation(
    const Eigen::VectorXd& value,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper
) {
    double violation = 0.0;
    for (Eigen::Index i = 0; i < value.size(); ++i) {
        violation = std::max(violation, lower(i) - value(i));
        violation = std::max(violation, value(i) - upper(i));
    }
    return std::max(violation, 0.0);
}

double objective_value(const SparseQpProblem& problem, const Eigen::VectorXd& x) {
    return 0.5 * x.dot(problem.quadratic * x) + problem.linear.dot(x);
}

bool finite_problem(const SparseQpProblem& problem) {
    if (problem.quadratic.rows() != problem.quadratic.cols()
        || problem.quadratic.rows() != problem.linear.size()
        || problem.constraint_matrix.cols() != problem.linear.size()
        || problem.constraint_matrix.rows() != problem.lower.size()
        || problem.lower.size() != problem.upper.size()) {
        return false;
    }
    if (!problem.linear.allFinite()) return false;
    for (Eigen::Index i = 0; i < problem.lower.size(); ++i) {
        if (std::isnan(problem.lower(i)) || std::isnan(problem.upper(i))
            || problem.lower(i) > problem.upper(i)) {
            return false;
        }
    }
    for (int outer = 0; outer < problem.quadratic.outerSize(); ++outer) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(problem.quadratic, outer); it; ++it) {
            if (!std::isfinite(it.value())) return false;
        }
    }
    for (int outer = 0; outer < problem.constraint_matrix.outerSize(); ++outer) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(problem.constraint_matrix, outer); it; ++it) {
            if (!std::isfinite(it.value())) return false;
        }
    }
    return true;
}

bool try_polish(
    const SparseQpProblem& problem,
    const double active_tolerance,
    Eigen::VectorXd& solution
) {
    const Eigen::VectorXd mapped = problem.constraint_matrix * solution;
    std::vector<int> active_rows;
    std::vector<double> active_values;
    active_rows.reserve(static_cast<size_t>(mapped.size()));
    active_values.reserve(static_cast<size_t>(mapped.size()));
    for (Eigen::Index row = 0; row < mapped.size(); ++row) {
        const bool equality = std::isfinite(problem.lower(row))
            && std::isfinite(problem.upper(row))
            && std::abs(problem.upper(row) - problem.lower(row)) <= active_tolerance;
        const double lower_distance = std::isfinite(problem.lower(row))
            ? std::abs(mapped(row) - problem.lower(row))
            : std::numeric_limits<double>::infinity();
        const double upper_distance = std::isfinite(problem.upper(row))
            ? std::abs(mapped(row) - problem.upper(row))
            : std::numeric_limits<double>::infinity();
        if (!equality && lower_distance > active_tolerance
            && upper_distance > active_tolerance) {
            continue;
        }
        active_rows.push_back(static_cast<int>(row));
        active_values.push_back(
            equality || lower_distance <= upper_distance
                ? problem.lower(row)
                : problem.upper(row)
        );
    }
    if (active_rows.empty()) return false;

    const int variables = static_cast<int>(problem.linear.size());
    const int active_count = static_cast<int>(active_rows.size());
    std::vector<Eigen::Triplet<double>> active_triplets;
    for (int column = 0; column < problem.constraint_matrix.outerSize(); ++column) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(
                 problem.constraint_matrix, column); it; ++it) {
            const auto found = std::lower_bound(
                active_rows.begin(), active_rows.end(), it.row()
            );
            if (found == active_rows.end() || *found != it.row()) continue;
            active_triplets.emplace_back(
                static_cast<int>(std::distance(active_rows.begin(), found)),
                it.col(), it.value()
            );
        }
    }
    Eigen::SparseMatrix<double> active(active_count, variables);
    active.setFromTriplets(active_triplets.begin(), active_triplets.end());
    const Eigen::MatrixXd active_dense(active);
    const Eigen::VectorXd baseline_gradient = problem.quadratic * solution
        + problem.linear;
    const Eigen::VectorXd baseline_multipliers = active_dense.transpose()
        .completeOrthogonalDecomposition()
        .solve(-baseline_gradient);
    const double baseline_kkt_residual = (
        baseline_gradient + active_dense.transpose() * baseline_multipliers
    ).lpNorm<Eigen::Infinity>();

    std::vector<Eigen::Triplet<double>> kkt_triplets;
    kkt_triplets.reserve(
        static_cast<size_t>(problem.quadratic.nonZeros() + 2 * active.nonZeros() + variables)
    );
    for (int column = 0; column < problem.quadratic.outerSize(); ++column) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(problem.quadratic, column); it; ++it) {
            kkt_triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }
    constexpr double POLISH_REGULARIZATION = 1e-9;
    for (int i = 0; i < variables; ++i) {
        kkt_triplets.emplace_back(i, i, POLISH_REGULARIZATION);
    }
    for (int column = 0; column < active.outerSize(); ++column) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(active, column); it; ++it) {
            kkt_triplets.emplace_back(it.col(), variables + it.row(), it.value());
            kkt_triplets.emplace_back(variables + it.row(), it.col(), it.value());
        }
    }
    Eigen::SparseMatrix<double> kkt(variables + active_count, variables + active_count);
    kkt.setFromTriplets(kkt_triplets.begin(), kkt_triplets.end());

    Eigen::VectorXd rhs(variables + active_count);
    rhs.head(variables) = -problem.linear;
    for (int i = 0; i < active_count; ++i) rhs(variables + i) = active_values[static_cast<size_t>(i)];

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(kkt);
    solver.factorize(kkt);
    if (solver.info() != Eigen::Success) return false;
    const Eigen::VectorXd polished = solver.solve(rhs);
    if (solver.info() != Eigen::Success || !polished.allFinite()) return false;
    const Eigen::VectorXd candidate = polished.head(variables);
    const Eigen::VectorXd candidate_multipliers = polished.tail(active_count);
    if (max_constraint_violation(
            problem.constraint_matrix * candidate, problem.lower, problem.upper
        ) > active_tolerance) {
        return false;
    }
    if (objective_value(problem, candidate)
        > objective_value(problem, solution) + active_tolerance) {
        return false;
    }
    const double candidate_stationarity = (
        problem.quadratic * candidate + problem.linear
        + active_dense.transpose() * candidate_multipliers
    ).lpNorm<Eigen::Infinity>();
    Eigen::VectorXd active_target(active_count);
    for (int i = 0; i < active_count; ++i) {
        active_target(i) = active_values[static_cast<size_t>(i)];
    }
    const double candidate_equality_residual = (
        active_dense * candidate - active_target
    ).lpNorm<Eigen::Infinity>();
    const double candidate_kkt_residual = std::max(
        candidate_stationarity, candidate_equality_residual
    );
    if (!std::isfinite(candidate_kkt_residual)
        || candidate_kkt_residual >= baseline_kkt_residual) {
        return false;
    }
    solution = candidate;
    return true;
}

} // anonymous namespace

AdmmQpSolver::Result AdmmQpSolver::solve(
    const SparseQpProblem& problem,
    const Eigen::VectorXd& initial_guess
) const {
    Result result;
    result.constraint_tolerance = params_.constraint_tolerance;
    result.final_rho = params_.rho;
    result.polish_status = params_.enable_polish
        ? PolishStatus::NOT_RUN
        : PolishStatus::DISABLED;
    if (!finite_problem(problem)
        || initial_guess.size() != problem.linear.size()
        || !initial_guess.allFinite()) {
        result.error = "invalid or non-finite QP data";
        return result;
    }

    const Eigen::Index variables = problem.linear.size();
    Eigen::SparseMatrix<double> identity(variables, variables);
    identity.setIdentity();
    const Eigen::SparseMatrix<double> ata =
        problem.constraint_matrix.transpose() * problem.constraint_matrix;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> linear_solver;
    double rho = params_.rho;

    const auto factorize = [&]() {
        const auto start = Clock::now();
        Eigen::SparseMatrix<double> system = problem.quadratic
            + params_.sigma * identity + rho * ata;
        linear_solver.compute(system);
        result.factorization_ms += std::chrono::duration<double, std::milli>(
            Clock::now() - start
        ).count();
        return linear_solver.info() == Eigen::Success;
    };
    if (!factorize()) {
        result.error = "ADMM linear system factorization failed";
        return result;
    }

    Eigen::VectorXd x = initial_guess;
    Eigen::VectorXd ax = problem.constraint_matrix * x;
    Eigen::VectorXd y = project_interval(ax, problem.lower, problem.upper);
    Eigen::VectorXd dual = Eigen::VectorXd::Zero(problem.lower.size());
    const auto iteration_start = Clock::now();
    for (int iteration = 1; iteration <= params_.max_iterations; ++iteration) {
        const Eigen::VectorXd rhs = -problem.linear + params_.sigma * x
            + rho * problem.constraint_matrix.transpose() * (y - dual);
        const Eigen::VectorXd next_x = linear_solver.solve(rhs);
        if (linear_solver.info() != Eigen::Success || !next_x.allFinite()) {
            result.error = "ADMM linear solve produced a non-finite result";
            result.iterations = iteration;
            break;
        }
        const Eigen::VectorXd previous_y = y;
        ax = problem.constraint_matrix * next_x;
        const Eigen::VectorXd relaxed = params_.relaxation * ax
            + (1.0 - params_.relaxation) * previous_y;
        y = project_interval(relaxed + dual, problem.lower, problem.upper);
        dual += relaxed - y;
        x = next_x;

        result.iterations = iteration;
        result.primal_residual = (ax - y).lpNorm<Eigen::Infinity>();
        result.dual_residual = (
            rho * problem.constraint_matrix.transpose() * (y - previous_y)
        ).lpNorm<Eigen::Infinity>();
        result.max_constraint_violation = max_constraint_violation(
            ax, problem.lower, problem.upper
        );
        result.primal_tolerance = params_.absolute_tolerance
            + params_.relative_tolerance * std::max(
                ax.lpNorm<Eigen::Infinity>(), y.lpNorm<Eigen::Infinity>()
            );
        result.dual_tolerance = params_.absolute_tolerance
            + params_.relative_tolerance * (
                rho * problem.constraint_matrix.transpose() * dual
            ).lpNorm<Eigen::Infinity>();
        if (result.primal_residual <= result.primal_tolerance
            && result.dual_residual <= result.dual_tolerance
            && result.max_constraint_violation <= params_.constraint_tolerance) {
            result.status = Status::SOLVED;
            break;
        }

        if (params_.rho_update_interval > 0
            && iteration % params_.rho_update_interval == 0
            && result.rho_updates < 10) {
            double new_rho = rho;
            if (result.primal_residual > 10.0 * result.dual_residual) {
                new_rho = rho * 2.0;
            } else if (result.dual_residual > 10.0 * result.primal_residual) {
                new_rho = rho * 0.5;
            }
            new_rho = std::clamp(new_rho, 1e-6, 1e6);
            if (new_rho != rho) {
                dual *= rho / new_rho;
                rho = new_rho;
                ++result.rho_updates;
                if (!factorize()) {
                    result.status = Status::NUMERICAL_FAILURE;
                    result.error = "ADMM rho update factorization failed";
                    break;
                }
            }
        }
    }
    result.iteration_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - iteration_start
    ).count();
    if (result.status == Status::NUMERICAL_FAILURE && result.error.empty()) {
        result.status = Status::MAX_ITERATIONS;
        result.error = "ADMM reached the iteration limit";
    }
    result.solution = x;
    result.final_rho = rho;

    if (result.status == Status::SOLVED && params_.enable_polish) {
        result.polish_status = try_polish(
            problem, params_.constraint_tolerance, result.solution
        ) ? PolishStatus::ACCEPTED : PolishStatus::REJECTED;
    }
    result.objective = objective_value(problem, result.solution);
    result.max_constraint_violation = max_constraint_violation(
        problem.constraint_matrix * result.solution, problem.lower, problem.upper
    );
    return result;
}

} // namespace nav_executor
