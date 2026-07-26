#pragma once

#include <string>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace nav_executor {

struct SparseQpProblem {
    Eigen::SparseMatrix<double> quadratic;
    Eigen::VectorXd linear;
    Eigen::SparseMatrix<double> constraint_matrix;
    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
};

class AdmmQpSolver {
public:
    struct Params {
        int max_iterations = 4000;
        double absolute_tolerance = 1e-5;
        double relative_tolerance = 1e-4;
        double constraint_tolerance = 1e-5;
        double rho = 0.5;
        double sigma = 1e-6;
        double relaxation = 1.6;
        int rho_update_interval = 25;
        bool enable_polish = true;
    };

    enum class Status {
        SOLVED,
        MAX_ITERATIONS,
        NUMERICAL_FAILURE,
    };

    struct Result {
        Status status = Status::NUMERICAL_FAILURE;
        Eigen::VectorXd solution;
        int iterations = 0;
        int rho_updates = 0;
        double primal_residual = 0.0;
        double dual_residual = 0.0;
        double max_constraint_violation = 0.0;
        double objective = 0.0;
        double factorization_ms = 0.0;
        double iteration_ms = 0.0;
        bool polish_attempted = false;
        bool polish_accepted = false;
        std::string error;
    };

    explicit AdmmQpSolver(Params params) : params_(params) {}

    [[nodiscard]] Result solve(
        const SparseQpProblem& problem,
        const Eigen::VectorXd& initial_guess
    ) const;

private:
    Params params_;
};

} // namespace nav_executor
