//
// Created by Ferdinand Vanmaele on 08.04.26.
//
// TODO: move to fas.h
#ifndef RMO_ROPT_SOLVER_H
#define RMO_ROPT_SOLVER_H

#include <boost/describe.hpp>

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include <deal.II/numerics/data_postprocessor.h>

#include <rmo/ropt/observer.h>
#include <rmo/ropt/oracle.h>
#include <rmo/ropt/residual.h>
#include <rmo/ropt/descent.h>


namespace rmo
{

//! One evaluated iterate of a (multilevel) descent method. The described fields are reported by
//! every solver; optional values go into `extra`.
struct CycleInfo
{
    unsigned iter      = 0;
    unsigned level     = 0;
    bool     coarse    = false;  // step came from a coarse correction
    unsigned lac_iter  = 0;
    double   residual  = 0.0;
    double   energy    = 0.0;
    double   step      = 0.0;
    double   min_value = 0.0;
    double   elapsed   = 0.0;

    //! Optional values, reported as additional columns (e.g. coarse condition norms)
    std::vector<std::pair<std::string, double>> extra;
};

BOOST_DESCRIBE_STRUCT(CycleInfo, (),
    (iter, level, coarse, lac_iter, residual, energy, step, min_value, elapsed));



class SolverBase : public ObservableSolver<CycleInfo>
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

    return {.step = step_size, .elapsed = timer.cpu_time()};
}


// Evaluates and reports the residual (of R) and the energy (of O) at the iterate y
template <typename Oracle>
std::pair<double,double>
cycle_eval(const Oracle& O, const ResidualBase& R, const Vector<double>& y, IterationObserver<CycleInfo>* observer,
           CycleInfo info)
{
    const double residual = R.residual(y);
    const double energy   = O.value(y);
    const auto min_element = std::ranges::min_element(y);

    info.residual  = residual;
    info.energy    = energy;
    info.min_value = *min_element;

    if (observer != nullptr) {
        observer->add(info);
    }
    return std::make_pair(residual, energy);
}


class GradientDescent : public SolverBase
{
public:
    // O_fine: oracle used for computing gradient descent steps on the fine level
    // R_fine: residual of the fine problem (stopping criterion; its value is passed to O_fine.gradient()),
    //         usually O_fine.get_residual()
    GradientDescent(OracleBase& O_fine, const ResidualBase& R_fine, const ManifoldBase& manifold,
                    DescentOptions options_gd)
        : O_fine(O_fine)
        , R_fine(R_fine)
        , manifold(manifold)
        , options_gd(options_gd)
    {
        // A residual of another problem (e.g. another level) has another dimension
        AssertDimension(R_fine.n_dofs(), O_fine.n_dofs());
    }

    void cycle(Vector<double>& x) override
    {
        timer.restart();
        if (m_observer != nullptr) {
            m_observer->begin_level(0);
        }

        // x is updated in-place
        O_fine.update(x);
        Vector<double> x_grad(x.size());
        Vector<double> dk(x.size());

        // Evaluate starting value
        CycleInfo info;
        info.level  = 0;
        info.coarse = false;
        info.iter   = 0;

        // Residual at x, passed to the next gradient (inner tolerance)
        double residual = cycle_eval(O_fine, R_fine, x, m_observer, info).first;
        x_hist.clear();
        x_hist.emplace_back(x);

        if (residual < options_gd.tol_residual) {
            return;
        }

        // Execute gradient descent iterations
        for (unsigned i = 1; i <= options_gd.max_iter; i++) {
            // Update gradient
#ifdef CPU_TIME
            std::cerr << "[" << timer.cpu_time() << "] fine: A-gradient\n";
#endif
            auto info_grad = O_fine.gradient(x, x_grad, residual);
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

            residual = cycle_eval(O_fine, R_fine, x, m_observer, info).first;
            x_hist.emplace_back(x);

            if (residual < options_gd.tol_residual) {
                // trick so that convergence_table is updated for last step
                // n iterations + starting solution -> n+1 table entries
                break;
            }

            // Avoid a stalling line search where the solution x does not change
            if (options_gd.line_search && info.step == 0.0) {
                std::cerr << "  -> no progress possible (line search stalled), stopping early" << std::endl;
                break;
            }
        }
        timer.stop();
    }

    void cycle(Vector<double>& x, std::ostream& os) override
    {
        cycle(x);
        if (m_observer != nullptr) {
            m_observer->end_level(0, os);
        }
    }

    const auto& history() const { return x_hist; }

private:
    dealii::Timer timer;

    OracleBase& O_fine;
    const ResidualBase& R_fine;
    const ManifoldBase& manifold;
    DescentOptions options_gd;

    std::vector<Vector<double>> x_hist;
};

} // namespace rmo

#endif //RMO_ROPT_SOLVER_H
