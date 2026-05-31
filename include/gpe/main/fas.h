//
// Created by Ferdinand Vanmaele on 27.05.26.
//

#ifndef GPE_FAS_H
#define GPE_FAS_H

#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/base/mg_level_object.h>

#include <gpe/lac.h>
#include <gpe/problem/oracle.h>
#include <gpe/problem/oracle_coarse.h>

#include <gpe/ropt/transport.h>
#include <gpe/ropt/solver.h>

#include <utility>

namespace gpe
{
using dealii::MGLevelObject;

// Model which creates oracles on the fly, depending on specified types (descent - coarse correction - coarse model)
// Vector and point transfers are independent (a-priori) of the chosen metric for the coarse model
// (the FullApproximationScheme constructor is not templated)
template <int dim>
class FullApproximationScheme
{
public:
    // Components are sorted in ascending level of discretization (from coarse to fine)
    FullApproximationScheme(MGLevelObject<std::shared_ptr<ManifoldBase>>          manifold_mg,
                            MGLevelObject<std::shared_ptr<ManifoldTransferBase>>  point_transfer_mg,
                            MGLevelObject<std::shared_ptr<VectorTransportBase>>   vector_transport_mg,
                            // TODO: later generalization: shared_ptr<FunctionalBase> (base for constructing oracles)
                            MGLevelObject<std::shared_ptr<GrossPitaevskiiFunctional<dim>>>  objective_mg,
                            MGLevelObject<DescentOptions>  options_descent_mg,
                            MGLevelObject<SolverOptions>   options_solver_mg,
                            FAS_Options options_fas)
        : m_manifold_mg         (std::move(manifold_mg))
        , m_point_transfer_mg   (std::move(point_transfer_mg))
        , m_vector_transport_mg (std::move(vector_transport_mg))
    // TODO: dependency injection
        , m_objective_mg        (std::move(objective_mg))
        , options_descent_mg    (std::move(options_descent_mg))
        , options_solver_mg     (std::move(options_solver_mg))
        , options_fas(options_fas)
    {
        min_level = m_manifold_mg.min_level();
        max_level = m_manifold_mg.max_level();

        conv_table_mg.resize(min_level, max_level);
    }

    // TiltOracleType: The oracle used to evaluate the coarse objective and build 'w' (e.g. MassOracle)
    // CoarseModelType: The surrogate descent model (e.g. MassCoarseOracleEnergyAdaptive)
    // OracleBase&: The oracle used to evaluate the level objective
    template <typename TiltOracleType, typename CoarseModelType>
    void cycle(OracleBase& O_level, Vector<double>& x, unsigned level)
    {
        AssertIndexRange(level - min_level, max_level - min_level + 1);

        // Clear and start the clock on finest level
        if (level == max_level) {
            timer.restart();
        }
        // Clear table for (W-)cycle
        auto& convergence_table = conv_table_mg[level];
        convergence_table.clear();

        // Fine descent direction (level)
        Vector<double> x_grad(x.size());
        // Coarse descent direction (level-1 -> level)
        Vector<double> dk(x.size());

        // Update the level oracle on the initial guess
        // This matches m_objective_mg[level]->update(x)
        O_level.update(x);

        if (level == min_level) {
            // Coarse condition is always false on coarsest level
            // -> gradient descent
            for (unsigned i = 0; i < options_descent_mg[level].max_iter; i++) {
                auto info_grad = O_level.gradient(x, x_grad);
                dk  = x_grad;
                dk *= -1.0;

                // Pass level manifold into cycle_smooth()
                // -> runs O_level.update(x)
                CycleInfo info = cycle_smooth(O_level, *m_manifold_mg[level], x, dk, timer, options_descent_mg[level]);
                info.iter      = i;
                info.coarse    = false;
                info.lac_iter  = info_grad.num_iter;

                cycle_eval(O_level, x, convergence_table, info);
            }
            return;
        }

        // Reference coarse oracle for next level
        TiltOracleType T_coarse(*m_objective_mg[level-1], options_solver_mg[level-1]);
        TiltOracleType T_level (*m_objective_mg[level],   options_solver_mg[level]);

        // Define the coarse model (for levels min_level+1..max_level)
        // Required to evaluate the coarse condition
        // Note: CoarseOracleBase is problem-independent and only depends on the OracleBase interface
        CoarseOracleBase<dim> qk_base(T_level, T_coarse,
            *m_manifold_mg[level-1], *m_point_transfer_mg[level], *m_vector_transport_mg[level]);
        // Evaluation of coarse model
        CoarseModelType qk(qk_base, options_solver_mg[level-1]);

        // T_level.update(x)
        // TODO: clearly encode that T_level and O_level point to the same GrossPitaevskiiSystem (~Functional)

        bool check_coarse_cond = true;

        // Begin (W-)cycle
        for (unsigned i = 0; i < options_descent_mg[level].max_iter; i++) {
            if (check_coarse_cond && (i == 0 || i % options_fas.coarse_every == 0)) {
                // Update coarse model for current level estimate x
                qk_base.update_model(x);

                // Compute coarse condition
                const auto& state  = qk_base.get_state();
                double norm_level  = T_level.norm(state.x_grad);
                double norm_coarse = T_coarse.norm(state.x_grad_restr);

                convergence_table.add_value("grad_norm", norm_level);
                convergence_table.add_value("grad_restr_norm", norm_coarse);

                if (norm_coarse <= options_fas.eps) {
                    check_coarse_cond = false;  // stop coarse condition evaluation once threshold was reached
                }

                // Different values of kappa for different levels?
                if (norm_coarse >= options_fas.kappa * norm_level && norm_coarse > options_fas.eps) {
                    // Initialize coarse trial point as the restricted fine point
                    Vector<double> zk = state.y;

                    // Solve the coarse model q_k(zk)
                    this->template cycle<TiltOracleType, CoarseModelType>(qk, zk, level-1);

#ifdef CPU_TIME
                    std::cerr << "[" << timer.cpu_time() << "] coarse: inverse retraction\n";
#endif
                    m_manifold_mg[level-1]->retract_inv(zk, state.y);

#ifdef CPU_TIME
                    std::cerr << "[" << timer.cpu_time() << "] coarse: vector prolongation\n";
#endif
                    m_vector_transport_mg[level]->vector_prolongation(state.x, state.y, zk, dk);

                    // Pass fine_manifold into cycle_smooth
                    // -> runs O_level.update(x)
                    CycleInfo info = cycle_smooth(O_level, *m_manifold_mg[level], x,
                        dk, timer, options_descent_mg[level]);
                    info.iter      = i;
                    info.coarse    = true;
                    info.lac_iter  = 0;

                    cycle_eval(O_level, x, convergence_table, info);
                } else {
                    goto fine_step;
                }
            }
            else {
                convergence_table.add_value("grad_restr_norm", 0);
                convergence_table.add_value("grad_norm", 0);
fine_step:
                // Update gradient
                auto info_grad = O_level.gradient(x, x_grad);
                dk  = x_grad;
                dk *= -1.0;

                // Pass fine_manifold into cycle_smooth
                // -> runs O_fine.update(x)
                CycleInfo info = cycle_smooth(O_level, *m_manifold_mg[level], x,
                    dk, timer, options_descent_mg[level]);
                info.iter      = i;
                info.coarse    = false;
                info.lac_iter  = info_grad.num_iter;

                cycle_eval(O_level, x, convergence_table, info);
            }
        }
        timer.stop();
    }

    // TODO: only output on certain levels OR include the current level in the table
    template <typename TiltOracleType, typename CoarseModelType>
    void cycle(OracleBase& O_level, Vector<double>& x, unsigned level, std::ostream& os)
    {
        cycle<TiltOracleType, CoarseModelType>(O_level, x, level);
        auto& convergence_table = conv_table_mg[level];

        convergence_table.set_precision("grad_restr_norm", 4);
        convergence_table.set_precision("grad_norm", 4);
        convergence_table.set_scientific("grad_restr_norm", true);
        convergence_table.set_scientific("grad_norm", true);

        cycle_finalize(convergence_table, os, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }


private:
    MGLevelObject<dealii::ConvergenceTable> conv_table_mg;
    mutable dealii::Timer timer;
    unsigned min_level, max_level;

    MGLevelObject<std::shared_ptr<ManifoldBase>>          m_manifold_mg;
    MGLevelObject<std::shared_ptr<ManifoldTransferBase>>  m_point_transfer_mg;
    MGLevelObject<std::shared_ptr<VectorTransportBase>>   m_vector_transport_mg;
    MGLevelObject<std::shared_ptr<GrossPitaevskiiFunctional<dim>>>  m_objective_mg;
    MGLevelObject<DescentOptions>                         options_descent_mg;
    MGLevelObject<SolverOptions>                          options_solver_mg;
    FAS_Options                                           options_fas;
};

} // namespace gpe

#endif //GPE_FAS_H
