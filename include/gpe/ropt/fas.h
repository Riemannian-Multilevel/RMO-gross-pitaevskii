//
// Created by Ferdinand Vanmaele on 08.04.26.
//
// TODO: move to fas.h
#ifndef GPE_MAIN_COARSE_H
#define GPE_MAIN_COARSE_H

#include <deal.II/numerics/data_postprocessor.h>
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
    auto state   = O.residual(y);
    state.energy = O.value(y);

    convergence_table.add_value("iter", info.iter);
    convergence_table.add_value("coarse", info.coarse ? "*" : " ");
    convergence_table.add_value("lac_iter", info.lac_iter);
    convergence_table.add_value("mass", state.mass);
    convergence_table.add_value("lambda", state.lambda);
    convergence_table.add_value("residual", state.residual);
    convergence_table.add_value("energy", state.energy);
    convergence_table.add_value("step",info.step_size);
    convergence_table.add_value("elapsed",info.elapsed);
}


inline void cycle_finalize(dealii::ConvergenceTable& convergence_table, std::ostream& os,
                           dealii::TableHandler::TextOutputFormat format)
{
    convergence_table.set_precision("mass", 4);
    convergence_table.set_precision("lambda", 4);
    convergence_table.set_precision("residual", 4);
    convergence_table.set_precision("energy", 8);
    convergence_table.set_precision("step", 4);
    convergence_table.set_precision("elapsed", 4);

    convergence_table.set_scientific("lambda", true);
    convergence_table.set_scientific("residual", true);
    convergence_table.set_scientific("energy", true);
    convergence_table.set_scientific("step", true);
    convergence_table.set_scientific("elapsed", true);

    convergence_table.evaluate_convergence_rates("residual", reduction_rate);
    convergence_table.evaluate_convergence_rates("residual", reduction_rate_log2);
    convergence_table.write_text(os, format);
}

template <int dim>
class GradientDescent
{
public:
    // O_fine: oracle used for computing gradient descent steps on the fine level
    GradientDescent(OracleBase& O_fine, const ManifoldBase& manifold)
        : O_fine(O_fine)
        , manifold(manifold)
    {}

    void cycle(const Vector<double>& x0, std::ostream& os, DescentOptions options_gd)
    {
        timer.restart();

        // 0. Initialize oracle and coarse model
        Vector<double> x(x0);
        O_fine.update(x);
        Vector<double> x_grad(x.size());
        Vector<double> dk(x.size());

        for (unsigned i = 0; i < options_gd.max_iter; i++) {
            // Update gradient
#ifdef CPU_TIME
            std::cerr << "[" << timer.cpu_time() << "] fine: A-gradient\n";
#endif
            auto lac_iter = O_fine.gradient(x, x_grad);
            dk  = x_grad;
            dk *= -1.0;

            // Evaluate directional derivative in A-norm
            CycleInfo info = cycle_smooth(O_fine, manifold, x, dk, timer, options_gd);
            info.iter      = i;
            info.coarse    = false;
            info.lac_iter  = lac_iter;

            cycle_eval(O_fine, x, convergence_table, info);
        }
        cycle_finalize(convergence_table, os, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }

private:
    dealii::ConvergenceTable convergence_table;
    dealii::Timer timer;
    OracleBase& O_fine;
    const ManifoldBase& manifold;
};


// TODO: Use cycle_fine() for consistency instead of gradient_descent()
// Deduce the FAS Manager via a template parameter
template <typename CoarseModelType, typename FASManagerType>
inline void coarse_solve(CoarseModelType& q_k,
                         const FASManagerType& fas,
                         const Vector<double>& x,
                         const ManifoldBase& coarse_manifold,
                         const VectorTransportBase& vector_transport,
                         DescentOptions options_gd,
                         Vector<double>& dst,
                         [[maybe_unused]] dealii::Timer& timer)
{
    const auto& state = fas.get_state();
    Vector<double> zk = state.y;

#ifdef CPU_TIME
    std::cerr << "[" << timer.cpu_time() << "] coarse: " << CoarseModelType::id << "-gradient descent\n";
#endif
    zk = gradient_descent(q_k, coarse_manifold, zk, options_gd, std::cerr);

#ifdef CPU_TIME
    std::cerr << "[" << timer.cpu_time() << "] coarse: inverse retraction\n";
#endif
    coarse_manifold.retract_inv(zk, state.y);

#ifdef CPU_TIME
    std::cerr << "[" << timer.cpu_time() << "] coarse: vector prolongation\n";
#endif
    vector_transport.vector_prolongation(x, state.y, zk, dst);
}


// Decouple Descent Oracle from the FAS Manager, and add the fine manifold
template <int dim, typename DescentOracleType, typename FASManagerType, typename CoarseModelType>
class FullApproximationScheme
{
public:
    FullApproximationScheme(DescentOracleType& O_fine,
                            FASManagerType& fas,
                            CoarseModelType& q_k,
                            const ManifoldBase& fine_manifold,
                            const ManifoldBase& coarse_manifold,
                            const VectorTransportBase& vector_transport)
        : O_fine(O_fine)
        , fas(fas)
        , q_k(q_k)
        , fine_manifold(fine_manifold)     // <--- ADDED
        , coarse_manifold(coarse_manifold)
        , vector_transport(vector_transport)
    {
        fas.set_timer(timer);
    }

    void cycle(Vector<double>& x, std::ostream& os,
               DescentOptions options_gd, DescentOptions options_gd_coarse,
               double kappa, double eps, unsigned coarse_every = 1)
    {
        // Clear and start the clock
        timer.restart();

        Vector<double> x_grad(x.size());
        Vector<double> dk(x.size());
        bool check_coarse_cond = true;
        bool is_updated = false;

        for (unsigned i = 0; i < options_gd.max_iter; i++) {

            bool do_coarse_step = false;

            if (check_coarse_cond && (i == 0 || i % coarse_every == 0)) {
                fas.update(x);
                is_updated = true;
                const auto& state = fas.get_state();

                double norm_fine = fas.objective_fine().norm(state.x_grad);
                double norm_coarse = fas.objective_coarse().norm(state.x_grad_restr);

                convergence_table.add_value("grad_norm", norm_fine);
                convergence_table.add_value("grad_restr_norm", norm_coarse);

                if (norm_coarse <= eps) { check_coarse_cond = false; }
                if (norm_coarse >= kappa * norm_fine && norm_coarse > eps) {
                    do_coarse_step = true;
                }
            } else {
                convergence_table.add_value("grad_restr_norm", 0);
                convergence_table.add_value("grad_norm", 0);
            }

            if (do_coarse_step) {
                coarse_solve(q_k, fas, x, coarse_manifold, vector_transport, options_gd_coarse, dk, timer);

                // 3. Pass fine_manifold into cycle_smooth
                CycleInfo info = cycle_smooth(O_fine, fine_manifold, x, dk, timer, options_gd);
                info.iter      = i;
                info.coarse    = true;
                info.lac_iter  = 0;

                cycle_eval(O_fine, x, convergence_table, info);
                is_updated = false;
            } else {
                if (!is_updated) {
                    O_fine.update(x);
                    is_updated = true;
                }

                auto lac_iter = O_fine.gradient(x, x_grad);
                dk  = x_grad;
                dk *= -1.0;

                // 3. Pass fine_manifold into cycle_smooth
                CycleInfo info = cycle_smooth(O_fine, fine_manifold, x, dk, timer, options_gd);
                info.iter      = i;
                info.coarse    = false;
                info.lac_iter  = lac_iter;

                cycle_eval(O_fine, x, convergence_table, info);
                is_updated = false;
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
    mutable dealii::Timer timer;

    DescentOracleType& O_fine;
    FASManagerType& fas;
    CoarseModelType& q_k;

    const ManifoldBase& fine_manifold;     // <--- ADDED
    const ManifoldBase& coarse_manifold;
    const VectorTransportBase& vector_transport;
};


} // namespace gpe

#endif //GPE_MAIN_COARSE_H
