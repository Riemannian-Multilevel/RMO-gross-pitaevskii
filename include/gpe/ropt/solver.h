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
#include <gpe/ropt/descent.h>

namespace gpe
{

struct CycleInfo
{
    unsigned iter;
    unsigned lac_iter;
    double   step_size;
    double   elapsed;
    bool     coarse;
    unsigned level;
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
                       double dir_deriv,
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
        //double dir_deriv = O_fine.directional_derivative(x, eta);
        // Runs O_fine.update(x)
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

    convergence_table.add_value("iter",     info.iter);
    convergence_table.add_value("level",    info.level);
    convergence_table.add_value("coarse",   info.coarse ? "*" : " ");
    convergence_table.add_value("lac_iter", info.lac_iter);
    convergence_table.add_value("residual", residual);
    convergence_table.add_value("energy",   energy);
    convergence_table.add_value("step",     info.step_size);
    convergence_table.add_value("elapsed",  info.elapsed);
}


inline void cycle_finalize(dealii::ConvergenceTable& convergence_table, std::ostream& os,
                           dealii::TableHandler::TextOutputFormat format)
{
    convergence_table.set_precision("residual", 4);
    convergence_table.set_precision("energy", 16);
    convergence_table.set_precision("step", 4);
    convergence_table.set_precision("elapsed", 4);

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

            double dir_deriv = O_fine.directional_derivative(x, dk);

            // Evaluate directional derivative in A-norm
            // -> runs OracleBase::update()
            CycleInfo info = cycle_smooth(O_fine, manifold, x, dk, dir_deriv, timer, options_gd);
            info.iter      = i;
            info.coarse    = false;
            info.lac_iter  = info_grad.num_iter;
            info.level     = 0;

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

} // namespace gpe

#endif //GPE_MAIN_COARSE_H
