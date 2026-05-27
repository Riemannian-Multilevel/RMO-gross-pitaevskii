//
// Created by Ferdinand Vanmaele on 08.04.26.
//
// TODO: move to fas.h
#ifndef GPE_MAIN_COARSE_H
#define GPE_MAIN_COARSE_H

#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/base/convergence_table.h>

#include <gpe/lac.h>
#include <gpe/problem/oracle.h>
#include <gpe/problem/oracle_coarse.h>
#include <gpe/ropt/transport.h>

namespace gpe
{
template <typename MatrixType>
double M_norm(const MatrixType& M, const Vector<double>& x)
{
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    return std::sqrt(x*Mx);
}


struct CycleInfo
{
    unsigned iter;
    unsigned lac_iter;
    double   step_size;
    double   elapsed;
    bool     coarse;
};


class SolverBase
{
public:
    virtual ~SolverBase() = default;

    // Both FAS and GradientDescent must implement this interface.
    // Note: 'x' must be non-const so the solver can mutate the initial guess
    virtual void cycle(Vector<double>& x, std::ostream& os) = 0;
    virtual void cycle(Vector<double>& x) = 0;
};


// Implementation of smoothing steps
template <typename Oracle>
CycleInfo cycle_smooth(Oracle& O_fine, const ManifoldBase& manifold,
                       Vector<double>& x, const Vector<double>& eta,
                       const dealii::Timer& timer,
                       DescentOptions options_gd)
{
    double step_size = options_gd.step_size;

    // Update point (fixed step or line search)
    if (options_gd.line_search) {
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] " << "fine: line search" << std::endl;
#endif
        double Ex = O_fine.value(x);
        double dir_deriv = O_fine.directional_derivative(x, eta);

        step_size = armijo_line_search(O_fine, manifold, x, eta, Ex, dir_deriv, options_gd);

        if (step_size <= options_gd.ls.min) {
            std::cerr << "  -> Step rejected by line search." << std::endl;
        }
    }
    else {
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] " << "fine: retraction" << std::endl;
#endif
        manifold.retract(eta, x, options_gd.step_size);   // update y

#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] " << "fine: assembly" << std::endl;
#endif
        O_fine.update(x);
    }

    return {.step_size = step_size, .elapsed = timer.cpu_time()};
}


template <typename Oracle>
void cycle_eval(const Oracle& O, const Vector<double>& y,
                dealii::ConvergenceTable& convergence_table,
                const CycleInfo info)
{
    const double residual = O.residual(y);
    const double energy   = O.value(y);

    convergence_table.add_value("iter", info.iter);
    convergence_table.add_value("coarse", info.coarse ? "*" : " ");
    convergence_table.add_value("lac_iter", info.lac_iter);
    convergence_table.add_value("residual", residual);
    convergence_table.add_value("energy", energy);
    convergence_table.add_value("step",info.step_size);
    convergence_table.add_value("elapsed",info.elapsed);
}


inline void cycle_finalize(dealii::ConvergenceTable& convergence_table, std::ostream& os,
                           dealii::TableHandler::TextOutputFormat format)
{
    convergence_table.set_precision("residual", 4);
    convergence_table.set_precision("energy", 8);
    convergence_table.set_precision("step", 4);
    convergence_table.set_precision("elapsed", 4);

    convergence_table.set_scientific("lambda", true);
    convergence_table.set_scientific("residual", true);
    convergence_table.set_scientific("energy", true);
    convergence_table.set_scientific("step", true);
    convergence_table.set_scientific("elapsed", true);

    convergence_table.evaluate_convergence_rates("residual", dealii::ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates("residual", dealii::ConvergenceTable::reduction_rate_log2);
    convergence_table.write_text(os, format);
}


template <int dim>
class GradientDescent : public SolverBase
{
public:
    // O_fine: oracle used for computing gradient descent steps on the fine level
    GradientDescent(OracleBase& O_fine, const ManifoldBase& manifold, DescentOptions options_gd)
        : O_fine(O_fine)
        , manifold(manifold)
        , options_gd(options_gd)
    {}

    void cycle(Vector<double>& x) override
    {
        timer.restart();
        convergence_table.clear();

        // x is updated in-place
        O_fine.update(x);
        Vector<double> x_grad(x.size());
        Vector<double> dk(x.size());

        for (unsigned i = 0; i < options_gd.max_iter; i++) {
            // Update gradient
#ifdef CPU_TIME
            std::cerr << "[" << timer.cpu_time() << "] fine: A-gradient\n";
#endif
            auto info_grad = O_fine.gradient(x, x_grad);
            dk  = x_grad;
            dk *= -1.0;

            // Evaluate directional derivative in A-norm
            // -> runs OracleBase::update()
            CycleInfo info = cycle_smooth(O_fine, manifold, x, dk, timer, options_gd);
            info.iter      = i;
            info.coarse    = false;
            info.lac_iter  = info_grad.num_iter;

            cycle_eval(O_fine, x, convergence_table, info);
        }
        timer.stop();
    }

    void cycle(Vector<double>& x, std::ostream& os) override
    {
        cycle(x);
        cycle_finalize(convergence_table, os, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }

private:
    dealii::ConvergenceTable convergence_table;
    dealii::Timer timer;

    OracleBase& O_fine;
    const ManifoldBase& manifold;
    DescentOptions options_gd;
};


// Decouple Descent Oracle from the FAS Manager, and add the fine manifold
template <int dim, typename DescentOracleType, typename FASManagerType, typename CoarseModelType>
class FullApproximationScheme : public SolverBase
{
public:
    FullApproximationScheme(DescentOracleType& O_fine,
                        // Coarse model parameters
                            FASManagerType& fas,
                        // Oracle for coarse model
                            CoarseModelType& q_k,
                        // Problem geometry
                            const ManifoldBase& fine_manifold,
                            const ManifoldBase& coarse_manifold,
                            const VectorTransportBase& vector_transport,
                        // Solver parameters
                            SolverBase&    coarse_solver, // Reference to the next level
                            DescentOptions options_gd,
                            FAS_Options    options_fas)
        : O_fine(O_fine)
        , fas(fas)
        , q_k(q_k)
        , fine_manifold(fine_manifold)
        , coarse_manifold(coarse_manifold)
        , vector_transport(vector_transport)
        , coarse_solver(coarse_solver)
        , options_gd(options_gd)
        , options_fas(options_fas)
    {
        fas.set_timer(timer);
    }

    void cycle(Vector<double>& x) override
    {
        // Clear and start the clock
        timer.restart();
        convergence_table.clear();

        // Update the fine oracle on the initial guess
        O_fine.update(x);

        Vector<double> x_grad(x.size());
        Vector<double> dk(x.size());

        for (unsigned i = 0; i < options_gd.max_iter; i++) {
            bool do_coarse_step = false;

            if (i == 0 || i % options_fas.coarse_every == 0) {
                fas.update_model(x);  // update coarse model and oracle
                const auto& state = fas.get_state();

                double norm_fine = fas.fine().norm(state.x_grad);
                double norm_coarse = fas.coarse().norm(state.x_grad_restr);

                convergence_table.add_value("grad_norm", norm_fine);
                convergence_table.add_value("grad_restr_norm", norm_coarse);

                if (norm_coarse >= options_fas.kappa * norm_fine && norm_coarse > options_fas.eps) {
                    do_coarse_step = true;
                }
            }
            else {
                convergence_table.add_value("grad_restr_norm", 0);
                convergence_table.add_value("grad_norm", 0);
            }

            if (do_coarse_step) {
                const auto& state = fas.get_state();
                // Initialize coarse trial point as the restricted fine point
                Vector<double> zk = state.y;
                // Solve the coarse model q_k(zk)
                // TODO: solve() method in CoarseModelType to ensure consistency
                coarse_solver.cycle(zk);

#ifdef CPU_TIME
                std::cerr << "[" << timer.cpu_time() << "] coarse: inverse retraction\n";
#endif
                coarse_manifold.retract_inv(zk, state.y);

#ifdef CPU_TIME
                std::cerr << "[" << timer.cpu_time() << "] coarse: vector prolongation\n";
#endif
                vector_transport.vector_prolongation(state.x, state.y, zk, dk);

                // Pass fine_manifold into cycle_smooth
                // -> runs O_fine.update(x)
                CycleInfo info = cycle_smooth(O_fine, fine_manifold, x, dk, timer, options_gd);

                info.iter      = i;
                info.coarse    = true;
                info.lac_iter  = 0;

                cycle_eval(O_fine, x, convergence_table, info);
            }
            else {
                auto info_grad = O_fine.gradient(x, x_grad);
                dk  = x_grad;
                dk *= -1.0;

                // 3. Pass fine_manifold into cycle_smooth
                // -> runs O_fine.update(x)
                CycleInfo info = cycle_smooth(O_fine, fine_manifold, x, dk, timer, options_gd);

                info.iter      = i;
                info.coarse    = false;
                info.lac_iter  = info_grad.num_iter;

                cycle_eval(O_fine, x, convergence_table, info);

            }
        }
        timer.stop();
    }

    void cycle(Vector<double>& x, std::ostream& os) override
    {
        cycle(x);

        convergence_table.set_precision("grad_restr_norm", 4);
        convergence_table.set_precision("grad_norm", 4);
        convergence_table.set_scientific("grad_restr_norm", true);
        convergence_table.set_scientific("grad_norm", true);

        cycle_finalize(convergence_table, os, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }

private:
    dealii::ConvergenceTable convergence_table;
    mutable dealii::Timer timer;

    DescentOracleType& O_fine;
    FASManagerType& fas;
    CoarseModelType& q_k;

    const ManifoldBase& fine_manifold;
    const ManifoldBase& coarse_manifold;
    const VectorTransportBase& vector_transport;

    SolverBase& coarse_solver;
    DescentOptions options_gd;
    FAS_Options options_fas;
};


} // namespace gpe

#endif //GPE_MAIN_COARSE_H
