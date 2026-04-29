//
// Created by Ferdinand Vanmaele on 28.04.26.
//

#ifndef GPE_FAS_H
#define GPE_FAS_H

#include <gpe/ropt/coarse.h>

namespace gpe
{

// TODO: factor out the cycle_smooth() methods (consolidate with descent.h)
//       convergence check (cf. EnergySimulator::run)
template <int dim, typename SmoothOracle, typename CoarseModel>
class FullApproximationScheme
{
public:
    using OperatorType  = LinearCombination<SparseMatrix<double>,Vector<double>>;
    using MatrixType    = SparseMatrix<double>;
    using InverseOpType = PreconditionInverse<OperatorType, SparseMatrix<double>>;

    // TODO: to support recursion, we need to be able to take arbitrary Oracles at levels 0...{n-1}
    FullApproximationScheme(const GrossPitaevskiiProblem<dim>& problem_coarse,
                            const GrossPitaevskiiProblem<dim>& problem_fine,
                            const LinearTransferBase& transfer, double beta,
                            SolverOptions options, SolverOptions options_coarse)
        : O_fine(problem_fine, beta, options)
        // TODO: separate preconditioners for gradient descent, and inverse of M (coarse gradients)
        , O_coarse(problem_coarse, problem_fine, transfer, beta, options, options_coarse)
        , problem_coarse(problem_coarse)
        , problem_fine(problem_fine)
    {
        O_coarse.set_timer(timer);
    }

    void cycle_condition(const Vector<double>& x0, std::ostream& os,
                         DescentOptions options_gd, DescentOptions options_gd_coarse,
                         double kappa, double eps, unsigned coarse_every = 1)
    {
        timer.reset();

        // 0. Initialize oracle and coarse model
        Vector<double> x(x0);
        O_fine.update(x);
        Vector<double> x_grad(x.size());
        Vector<double> dk(x.size());

        const auto& M_fine = problem_fine.get_M();
        const auto& M_coarse = problem_coarse.get_M();
        bool check_coarse_cond = true;

        for (unsigned i = 0; i < options_gd.max_iter; i++) {
            // Compute coarse condition
            if (check_coarse_cond && (i == 0 || i % coarse_every == 0)) {
                // TODO: If the coarse model is evaluated in the A-gradient (or the fine objected solved
                //       in the M-metric), this step results in negligible additional effort.
                //       Metric-free formulation of the coarse condition?
                auto coarse_step = O_coarse.setup(x);
                // Norm of fine (M-)gradient
                double x_grad_norm = M_norm(M_fine, coarse_step.x_grad);
                // Norm of restricted (M-)gradient
                double x_grad_restr_norm = M_norm(M_coarse, coarse_step.x_grad_restr);
                convergence_table.add_value("grad_restr_norm", x_grad_restr_norm);
                convergence_table.add_value("grad_norm", x_grad_norm);

                if (x_grad_restr_norm <= eps) {
                    check_coarse_cond = false;  // stop coarse condition evaluation once threshold was reached
                }
                std::cerr << "x_grad_restr_norm: " << x_grad_restr_norm << "\n";
                std::cerr << "x_grad_norm: " << x_grad_norm << "\n";
                if (x_grad_restr_norm >= kappa*x_grad_norm && x_grad_restr_norm > eps) {
                    // Coarse step
                    O_coarse.solve(coarse_step, options_gd_coarse, dk);

                    // TODO: debug step for checking descent direction (to gradient of corresponding fine oracle)
                    CycleInfo info = cycle_smooth(x, dk, options_gd);
                    info.iter      = i;
                    info.coarse    = true;
                    info.lac_iter  = 0;

                    cycle_eval(O_fine, x, convergence_table, info);
                } else {
                    goto fine_step;
                }
            } else {
                convergence_table.add_value("grad_restr_norm", 0);
                convergence_table.add_value("grad_norm", 0);
    fine_step:
                // Update gradient
                std::cerr << "[" << timer.cpu_time() << "] fine: A-gradient\n";
                auto lac_iter = O_fine.gradient(x, x_grad);
                dk  = x_grad;
                dk *= -1.0;

                // Evaluate directional derivative in A-norm
                CycleInfo info = cycle_smooth(x, dk, options_gd);
                info.iter      = i;
                info.coarse    = false;
                info.lac_iter  = lac_iter;

                cycle_eval(O_fine, x, convergence_table, info);
            }
        }
        convergence_table.set_precision("grad_restr_norm", 4);
        convergence_table.set_precision("grad_norm", 4);
        convergence_table.set_scientific("grad_restr_norm", true);
        convergence_table.set_scientific("grad_norm", true);

        cycle_finalize(convergence_table, os, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }

private:
    dealii::ConvergenceTable convergence_table;
    dealii::Timer timer;
    SmoothOracle O_fine;
    CoarseModel O_coarse;  // encodes both the objective, and the method to solve it
    const GrossPitaevskiiProblem<dim>& problem_coarse;
    const GrossPitaevskiiProblem<dim>& problem_fine;

    // Implementation of smoothing steps
    CycleInfo cycle_smooth(Vector<double>& x, const Vector<double>& eta, DescentOptions options_gd)
    {
        timer.start();
        double step_size = options_gd.step_size;

        // Update point (fixed step or line search)
        if (options_gd.line_search) {
            std::cerr << "[" << timer.cpu_time() << "] " << "fine: line search" << std::endl;
            double Ex = O_fine.value(x);
            double dir_deriv = O_fine.directional_derivative(x, eta);

            step_size = armijo_line_search(O_fine, x, eta, Ex, dir_deriv, options_gd);

            if (step_size <= options_gd.ls.min) {
                std::cerr << "  -> Step rejected by line search." << std::endl;
            }
        }
        else {
            std::cerr << "[" << timer.cpu_time() << "] " << "fine: retraction" << std::endl;
            O_fine.retract(eta, x, options_gd.step_size);  // update y

            std::cerr << "[" << timer.cpu_time() << "] " << "fine: assembly" << std::endl;
            O_fine.update(x);
        }

        timer.stop();
        return {.step_size = step_size, .elapsed = timer.cpu_time()};
    }
};

} // namespace gpe

#endif //GPE_FAS_H
