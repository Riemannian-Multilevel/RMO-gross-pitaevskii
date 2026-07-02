//
// Created by Ferdinand Vanmaele on 27.05.26.
//

#ifndef GPE_FAS_H
#define GPE_FAS_H

#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/base/mg_level_object.h>

#include <gpe/problem/oracle.h>
#include <gpe/problem/oracle_coarse.h>

#include <gpe/ropt/transport.h>
#include <gpe/ropt/solver.h>

#include <utility>

namespace gpe
{
using dealii::MGLevelObject;

struct ConvergenceTable : dealii::ConvergenceTable {
    bool has_column(const std::string& key) const {
        return columns.count(key) > 0;
    }
};


// Model which creates oracles on the fly, depending on specified types (descent - coarse correction - coarse model)
// Vector and point transfers are independent (a-priori) of the chosen metric for the coarse model
// (the FullApproximationScheme constructor is not templated)
template <int dim>
class FullApproximationScheme
{
public:
    // Components are sorted in ascending level of discretization (from coarse to fine)
    // TODO: dependency injection
    FullApproximationScheme(MGLevelObject<std::shared_ptr<ManifoldBase>>          manifold_mg,
                            MGLevelObject<std::shared_ptr<ManifoldTransferBase>>  point_transfer_mg,
                            MGLevelObject<std::shared_ptr<VectorTransportBase>>   vector_transport_mg,
                            // TODO: generalization: shared_ptr<FunctionalBase> (base for constructing oracles)
                            MGLevelObject<std::shared_ptr<GrossPitaevskiiFunctional<dim>>>  objective_mg,
                            const std::vector<unsigned> &level_indices,
                            MGLevelObject<DescentOptions>  options_descent_mg,
                            MGLevelObject<SolverOptions>   options_solver_mg,
                            FAS_Options options_fas)
        : level_indices(level_indices)
        , m_manifold_mg         (std::move(manifold_mg))
        , m_point_transfer_mg   (std::move(point_transfer_mg))
        , m_vector_transport_mg (std::move(vector_transport_mg))
        , m_objective_mg        (std::move(objective_mg))
        , options_descent_mg    (std::move(options_descent_mg))
        , options_solver_mg     (std::move(options_solver_mg))
        , options_fas(options_fas)
    {
        // Check that level indices are strictly ascending
        Assert(std::ranges::adjacent_find(level_indices, std::greater_equal<unsigned>()) == level_indices.end(),
            dealii::ExcInternalError("level indices not sorted in strictly ascending order"));

        // Check that MG objects are of the same size
        min_level = m_manifold_mg.min_level();
        max_level = m_manifold_mg.max_level();
        AssertDimension(m_point_transfer_mg.min_level(),   min_level);
        AssertDimension(m_point_transfer_mg.max_level(),   max_level);
        AssertDimension(m_vector_transport_mg.min_level(), min_level);
        AssertDimension(m_vector_transport_mg.max_level(), max_level);
        AssertDimension(m_objective_mg.min_level(),        min_level);
        AssertDimension(m_objective_mg.max_level(),        max_level);

        // Check that level indices are contained within MGLevelObject
        AssertIndexRange(min_level, level_indices.front()+1);  // open range
        AssertIndexRange(max_level, level_indices.back() +1);

        // Create a convergence table for every level
        conv_table_mg.resize(min_level, max_level);
    }

    // TiltOracleType:       The oracle used to evaluate the coarse objective and build 'w' (e.g. MassOracle)
    // TiltCoarseModelType:  The coarse oracle used for building 'w' (e.g. MassCoarseOracle)
    // CoarseModelType:      The coarse descent model for gradients (e.g. MassCoarseOracleEnergyAdaptive)
    // OracleBase&:          The oracle used to evaluate the level objective
    template <typename TiltOracleType, typename TiltCoarseOracleType, typename CoarseModelType>
    void cycle(OracleBase& O_level, OracleBase& T_level, Vector<double>& x, unsigned level_idx)
    {
        AssertIndexRange(level_idx, level_indices.size());
        unsigned level = level_indices.at(level_idx);
        std::cerr << "level: " << level << std::endl;
        //AssertIndexRange(level - min_level, max_level - min_level + 1);

        // Clear and start the clock on finest level
        if (level_idx == level_indices.size() - 1) {
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
        // TODO: O_level and T_level should point to same underlying state (GrossPitaevskiiSystem)

        if (level_idx == 0) {
        //if (level == min_level) {
            // Coarse condition is always false on coarsest level
            // -> gradient descent
            for (unsigned i = 0; i < options_descent_mg[level].max_iter; i++) {
                level_log.push_back(level);

                auto info_grad = O_level.gradient(x, x_grad);
                dk  = x_grad;
                dk *= -1.0;

                double dir_deriv = O_level.directional_derivative(x, dk);

                // Pass level manifold into cycle_smooth()
                // -> runs O_level.update(x)
                CycleInfo info = cycle_smooth(O_level, *m_manifold_mg[level], x, dk, dir_deriv,
                    timer, options_descent_mg[level]);
                info.iter      = i;
                info.coarse    = false;
                info.lac_iter  = info_grad.num_iter;
                info.level     = level;

                cycle_eval(O_level, x, convergence_table, info);
            }
            return;
        }

        // Reference coarse oracle for next level
        unsigned coarse_level_idx = level_idx - 1;
        unsigned coarse_level     = level_indices.at(coarse_level_idx);

        TiltOracleType T_coarse(*m_objective_mg[coarse_level], options_solver_mg[coarse_level]);

        // Define the coarse model (for levels min_level+1..max_level)
        // Required to evaluate the coarse condition
        // Note: CoarseOracleBase is problem-independent and only depends on the OracleBase interface
        // Recursive call (example)
        // - T_level:  MassCoarseOracle
        // - T_coarse: MassOracle
        // - O_level:  MassCoarseOracleEnergyAdaptive
        CoarseOracleBase<dim> qk_base(T_level, T_coarse, *m_manifold_mg[coarse_level], *m_point_transfer_mg[level], *m_vector_transport_mg[level]);
        // Evaluation of coarse model gradient  (-> descent direction, A-gradient)
        CoarseModelType qk(qk_base, options_solver_mg[coarse_level]);
        // Evaluation of coarse model objective (-> correction term w, M-gradient)
        TiltCoarseOracleType qk_m(qk_base, options_solver_mg[coarse_level]);

        // T_level.update(x)
        // TODO: clearly encode that T_level and O_level point to the same GrossPitaevskiiSystem (~Functional)

        bool check_coarse_cond = true;  // allow disabling coarse steps entirely at some point in the iteration
        bool prev_fine_step = false;

        // Begin (W-)cycle
        for (unsigned i = 0; i < options_descent_mg[level].max_iter; i++) {
            //level_log.push_back(level_indices.at(level_idx));

            //if (check_coarse_cond && (i == 0 || i % options_fas.coarse_every == 0)) {
            if (check_coarse_cond && (i == 0 || prev_fine_step)) {
                // Update coarse model for current level estimate x
                // -> runs T_coarse.update(y) <-> m_objective_mg[level-1]->update(y)
                // TODO: set fixed tolerance (multiplied by options.tol_inner_res)
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
                    // TODO: pass on ostream (-> callback strategy)
                    //       throw/catch exception when we do not have a descent direction on the fine level
                    this->template cycle<TiltOracleType, TiltCoarseOracleType, CoarseModelType>(qk, qk_m,
                        zk, coarse_level_idx, std::cerr);

#ifdef CPU_TIME
                    std::cerr << "[" << timer.cpu_time() << "] coarse: inverse retraction\n";
#endif
                    m_manifold_mg[coarse_level]->retract_inv(zk, state.y);

#ifdef CPU_TIME
                    std::cerr << "[" << timer.cpu_time() << "] coarse: vector prolongation\n";
#endif
                    m_vector_transport_mg[level]->vector_prolongation(state.x, state.y, zk, dk);

                    // Fallback to gradient descent in case of ascend direction
                    // TODO: skip coarse steps for remainder of the cycle?
                    double dir_deriv = O_level.directional_derivative(x, dk);

                    if (dir_deriv >= 0) {
                        std::cerr << "warning: not a descent direction (" << dir_deriv
                                  << std::setprecision(12) << ")" << std::endl;
                        std::cerr << "falling back to gradient step" << std::endl;

                        goto fine_step;
                    }
                    prev_fine_step = false;

                    // Pass fine_manifold into cycle_smooth
                    // -> runs O_level.update(x)
                    CycleInfo info = cycle_smooth(O_level, *m_manifold_mg[level], x, dk, dir_deriv,
                        timer, options_descent_mg[level]);
                    info.iter      = i;
                    info.coarse    = true;
                    info.lac_iter  = 0;
                    info.level     = level;

                    cycle_eval(O_level, x, convergence_table, info);
                }
                else {
                    goto fine_step;
                }
            }
            else {
                convergence_table.add_value("grad_restr_norm", 0);
                convergence_table.add_value("grad_norm", 0);
fine_step:
                prev_fine_step = true;
                // Record that a fine step was taken on this level
                level_log.push_back(level_indices.at(level_idx));
                // Update gradient
                auto info_grad = O_level.gradient(x, x_grad);
                dk  = x_grad;
                dk *= -1.0;

                // TODO: this can still be an ascend direction for coarser levels (numerical issues?)
                //       return early from the cycle in this case?
                double dir_deriv = O_level.directional_derivative(x, dk);

                // Pass fine_manifold into cycle_smooth
                // -> runs O_fine.update(x)
                CycleInfo info = cycle_smooth(O_level, *m_manifold_mg[level], x, dk, dir_deriv,
                    timer, options_descent_mg[level]);
                info.iter      = i;
                info.coarse    = false;
                info.lac_iter  = info_grad.num_iter;
                info.level     = level;

                cycle_eval(O_level, x, convergence_table, info);
            }
        }
        if (level_idx == level_indices.size() - 1) {
            timer.stop();
        }
    }

    // TODO: only output on certain levels OR include the current level in the table
    template <typename TiltOracleType, typename TiltCoarseOracleType, typename CoarseModelType>
    void cycle(OracleBase& O_level, OracleBase& T_level, Vector<double>& x, unsigned level_idx, std::ostream& os)
    {
        cycle<TiltOracleType, TiltCoarseOracleType, CoarseModelType>(O_level, T_level, x, level_idx);

        unsigned level = level_indices.at(level_idx);
        auto& convergence_table = conv_table_mg[level];

        // TODO: this is only set if an actual coarse step was taken
        for (const char* col : {"grad_norm", "grad_restr_norm"}) {
            if (convergence_table.has_column(col)) {
                convergence_table.set_precision(col, 4);
                convergence_table.set_scientific(col, true);
            }
        }
        cycle_finalize(convergence_table, os, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }

    const std::vector<unsigned> cycle_log() const
    {
        return level_log;
    }

private:
    MGLevelObject<ConvergenceTable> conv_table_mg;
    mutable dealii::Timer timer;
    unsigned min_level, max_level;
    std::vector<unsigned> level_indices;
    std::vector<unsigned> level_log;

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
