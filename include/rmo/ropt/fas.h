//
// Created by Ferdinand Vanmaele on 27.05.26.
//

#ifndef RMO_ROPT_FAS_H
#define RMO_ROPT_FAS_H

#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/base/mg_level_object.h>

#include <rmo/ropt/condition.h>
#include <rmo/ropt/observer.h>
#include <rmo/ropt/oracle_base.h>
#include <rmo/ropt/oracle_coarse_base.h>

#include <rmo/ropt/transport.h>
#include <rmo/ropt/solver.h>

#include <functional>
#include <optional>
#include <utility>

namespace rmo
{
using dealii::MGLevelObject;

// Norm on one level, used by FullApproximationScheme to evaluate the coarse condition
//   ||R g||_{l-1} >= kappa ||g||_l
using LevelNorm = std::function<double(const Vector<double>&)>;


// Model which creates oracles on the fly, depending on specified types (descent - coarse correction - coarse model)
// Vector and point transfers are independent (a-priori) of the chosen metric for the coarse model
// (the FullApproximationScheme constructor is not templated)
template <typename Functional>
class FullApproximationScheme : public ObservableSolver
{
public:
    // Components are sorted in ascending level of discretization (from coarse to fine)
    // cond_norm_mg[l]: norm on level l for the coarse condition; an empty entry falls back to the
    //                  norm of the tilt oracle on that level (i.e. the metric of the coarse model).
    // condition_mg[l]: coarse-correction trigger (eq. 16) evaluated on level l against level l-1;
    //                  a null entry falls back to DefaultCoarseCondition.
    FullApproximationScheme(MGLevelObject<std::shared_ptr<ManifoldBase>>          manifold_mg,
                            MGLevelObject<std::shared_ptr<ManifoldTransferBase>>  point_transfer_mg,
                            MGLevelObject<std::shared_ptr<VectorTransportBase>>   vector_transport_mg,
                            MGLevelObject<std::shared_ptr<Functional>>            objective_mg,
                            const std::vector<unsigned> &level_indices,
                            MGLevelObject<DescentOptions>  options_descent_mg,
                            MGLevelObject<SolverOptions>   options_solver_mg,
                            FAS_Options options_fas,
                            std::optional<MGLevelObject<LevelNorm>> cond_norm_mg = std::nullopt,
                            std::optional<MGLevelObject<std::shared_ptr<CoarseConditionBase>>> condition_mg = std::nullopt)
        : level_indices(level_indices)
        , m_manifold_mg         (std::move(manifold_mg))
        , m_point_transfer_mg   (std::move(point_transfer_mg))
        , m_vector_transport_mg (std::move(vector_transport_mg))
    // m_manifold_mg is initialized before m_cond_norm_mg/m_condition_mg, so its level bounds
    // size the defaults
        , m_cond_norm_mg        (cond_norm_mg ? std::move(*cond_norm_mg)
                                              : MGLevelObject<LevelNorm>(m_manifold_mg.min_level(),
                                                                         m_manifold_mg.max_level()))
        , m_condition_mg        (condition_mg ? std::move(*condition_mg)
                                              : MGLevelObject<std::shared_ptr<CoarseConditionBase>>(
                                                    m_manifold_mg.min_level(), m_manifold_mg.max_level()))
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
        AssertDimension(m_cond_norm_mg.min_level(),        min_level);
        AssertDimension(m_cond_norm_mg.max_level(),        max_level);
        AssertDimension(m_condition_mg.min_level(),        min_level);
        AssertDimension(m_condition_mg.max_level(),        max_level);

        // Check that level indices are contained within MGLevelObject
        AssertIndexRange(min_level, level_indices.front()+1);  // open range
        AssertIndexRange(max_level, level_indices.back() +1);

    }

    // TiltOracleType:       The oracle used to evaluate the coarse objective and build 'w' (e.g. MassOracle)
    // TiltCoarseModelType:  The coarse oracle used for building 'w' (e.g. MassCoarseOracle)
    // CoarseModelType:      The coarse descent model for gradients (e.g. MassCoarseOracleEnergyAdaptive)
    // OracleBase&:          The oracle used to evaluate the level objective
    // TODO: report the iterate history through the observer as well (x_hist)
    template <typename TiltOracleType, CoarseOracle TiltCoarseOracleType, CoarseOracle CoarseModelType>
        requires TiltOracle<TiltOracleType, Functional>
    void cycle(OracleBase& O_level, OracleBase& T_level, Vector<double>& x, unsigned level_idx)
    {
        AssertIndexRange(level_idx, level_indices.size());
        unsigned level = level_indices.at(level_idx);
        //AssertIndexRange(level - min_level, max_level - min_level + 1);

        // Clear and start the clock on finest level
        if (level_idx == level_indices.size() - 1) {
            timer.restart();
        }
        // Clear table for (W-)cycle
        if (m_observer != nullptr) {
            m_observer->begin_level(level);
        }

        // Fine descent direction (level)
        Vector<double> x_grad(x.size());
        // Coarse descent direction (level-1 -> level)
        Vector<double> dk(x.size());

        // Update the level oracle on the initial guess.
        // Invariant: O_level and T_level evaluate the same functional (m_objective_mg[level]),
        // so this update is visible to T_level as well.
        O_level.update(x);

        // Solve on coarsest level
        if (level_idx == 0) {
            // Evaluate starting value
            CycleInfo info;
            info.iter      = 0;
            info.coarse    = false;
            info.lac_iter  = 0;
            info.level     = level;
            info.step_size = 0.0;   // no step taken yet
            info.elapsed   = timer.cpu_time();

            cycle_eval(O_level, x, m_observer, info);

            // Coarse condition is always false on coarsest level
            // -> gradient descent
            for (unsigned i = 1; i <= options_descent_mg[level].max_iter; i++) {
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

                //auto [residual, _] = cycle_eval(O_level, x, m_observer, info);
                cycle_eval(O_level, x, m_observer, info);

                // Avoid a stalling line search where the solution x does not change
                if (options_descent_mg[level].line_search && info.step_size == 0.0) {
                    std::cerr << "  -> no progress possible (line search stalled), stopping early" << std::endl;
                    break;
                }

                // TODO: residual check on coarser levels?
                // if (residual < options_descent_mg[level].tol_residual) {
                //     break;
                // }
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
        CoarseOracleBase qk_base(T_level, T_coarse, *m_manifold_mg[coarse_level],
            *m_point_transfer_mg[level], *m_vector_transport_mg[level]);
        // Evaluation of coarse model gradient  (-> descent direction, A-gradient)
        CoarseModelType qk(qk_base, options_solver_mg[coarse_level]);
        // Evaluation of coarse model objective (-> correction term w, M-gradient)
        TiltCoarseOracleType qk_m(qk_base, options_solver_mg[coarse_level]);

        // T_level.update(x) is implied by O_level.update(x) above (shared functional)

        bool check_coarse_cond = true;

        // Evaluate starting value (finer levels)
        CycleInfo info;
        info.iter      = 0;
        info.coarse    = false;
        info.lac_iter  = 0;
        info.level     = level;
        info.step_size = 0.0;   // no step taken yet
        info.elapsed   = timer.cpu_time();

        info.coarse_cond = true;  // this level has a coarser one: report the condition norms

        auto [residual, _] = cycle_eval(O_level, x, m_observer, info);

        // Plot history of iterates
        if (level_idx == level_indices.size() - 1) {
            x_hist.emplace_back(x);

            // Residual check on finest level
            if (residual < options_descent_mg[level].tol_residual) {
                return;
            }
        }

        // Begin (W-)cycle
        for (unsigned i = 1; i <= options_descent_mg[level].max_iter; i++) {
            //level_log.push_back(level_indices.at(level_idx));

            // Coarse-condition norms for this step; left at 0 when the condition is not evaluated
            double cond_grad_norm       = 0.0;
            double cond_grad_restr_norm = 0.0;

            if (check_coarse_cond && (i == 1 || (i-1) % options_fas.coarse_every == 0)) {
                // Update coarse model for current level estimate x
                // -> runs T_coarse.update(y) <-> m_objective_mg[level-1]->update(y)
                // TODO: set fixed tolerance (multiplied by options.tol_inner_res)
                qk_base.update_model(x);

                // Compute coarse condition (in the configured norm, or the oracle's metric by default)
                const auto& state  = qk_base.get_state();
                double norm_level  = cond_norm(level,        T_level,  state.x_grad);
                double norm_coarse = cond_norm(coarse_level, T_coarse, state.x_grad_restr);

                cond_grad_norm       = norm_level;
                cond_grad_restr_norm = norm_coarse;

                if (norm_level <= options_fas.eps) {
                    check_coarse_cond = false;  // stop coarse condition evaluation once threshold was reached
                }

                if (coarse_condition_trigger(level, norm_level, norm_coarse)) {
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

                    // Pass fine_manifold into cycle_smooth
                    // -> runs O_level.update(x)
                    CycleInfo info = cycle_smooth(O_level, *m_manifold_mg[level], x, dk, dir_deriv,
                        timer, options_descent_mg[level]);
                    info.iter      = i;
                    info.coarse    = true;
                    info.lac_iter  = 0;
                    info.level     = level;
                    info.coarse_cond     = true;
                    info.grad_norm       = cond_grad_norm;
                    info.grad_restr_norm = cond_grad_restr_norm;

                    auto [residual, _] = cycle_eval(O_level, x, m_observer, info);

                    // Plot history of iterates
                    if (level_idx == level_indices.size() - 1) {
                        x_hist.emplace_back(x);

                        // Residual check on finest level
                        if (residual < options_descent_mg[level].tol_residual) {
                            return;
                        }
                    }

                    // Avoid a stalling line search where the solution x does not change
                    if (options_descent_mg[level].line_search && info.step_size == 0.0) {
                        std::cerr << "  -> no progress possible (line search stalled), stopping early" << std::endl;
                        break;
                    }

                    // TODO: residual check on coarser levels?
                    // if (residual < options_descent_mg[level].tol_residual) {
                    //     break;
                    // }
                }
                else {
                    goto fine_step;
                }
            }
            else {
fine_step:
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
                info.coarse_cond     = true;
                info.grad_norm       = cond_grad_norm;
                info.grad_restr_norm = cond_grad_restr_norm;

                auto [residual, _] = cycle_eval(O_level, x, m_observer, info);

                // Plot history of iterates
                if (level_idx == level_indices.size() - 1) {
                    x_hist.emplace_back(x);

                    // Residual check on finest level
                    if (residual < options_descent_mg[level].tol_residual) {
                        return;
                    }
                }

                // Avoid a stalling line search where the solution x does not change
                if (options_descent_mg[level].line_search && info.step_size == 0.0) {
                    std::cerr << "  -> no progress possible (line search stalled), stopping early" << std::endl;
                    break;
                }

                // TODO: residual check on coarser levels?
                // if (residual < options_descent_mg[level].tol_residual) {
                //     break;
                // }
            }
        }

        if (level_idx == level_indices.size() - 1) {
            timer.stop();
        }
    }

    // TODO: only output on certain levels OR include the current level in the table
    template <typename TiltOracleType, CoarseOracle TiltCoarseOracleType, CoarseOracle CoarseModelType>
        requires TiltOracle<TiltOracleType, Functional>
    void cycle(OracleBase& O_level, OracleBase& T_level, Vector<double>& x, unsigned level_idx, std::ostream& os)
    {
        cycle<TiltOracleType, TiltCoarseOracleType, CoarseModelType>(O_level, T_level, x, level_idx);

        unsigned level = level_indices.at(level_idx);
        if (m_observer != nullptr) {
            m_observer->end_level(level, os);
        }
    }

    const std::vector<unsigned> cycle_log() const
    {
        return level_log;
    }

    const auto& history() const { return x_hist; }

private:
    double cond_norm(unsigned level, const OracleBase& T, const Vector<double>& v) const
    {
        const LevelNorm& norm = m_cond_norm_mg[level];
        if (norm)              // a norm was configured for this level
            return norm(v);
        return T.norm(v);      // default: metric of the (tilt) oracle on this level
    }

    bool coarse_condition_trigger(unsigned level, double norm_level, double norm_coarse) const
    {
        if (m_condition_mg[level])   // a condition was configured for this level
            return m_condition_mg[level]->trigger(norm_level, norm_coarse, options_fas);
        static const DefaultCoarseCondition default_condition;
        return default_condition.trigger(norm_level, norm_coarse, options_fas);
    }

    mutable dealii::Timer timer;
    unsigned min_level, max_level;
    std::vector<unsigned> level_indices;
    std::vector<unsigned> level_log;

    MGLevelObject<std::shared_ptr<ManifoldBase>>          m_manifold_mg;
    MGLevelObject<std::shared_ptr<ManifoldTransferBase>>  m_point_transfer_mg;
    MGLevelObject<std::shared_ptr<VectorTransportBase>>   m_vector_transport_mg;
    MGLevelObject<LevelNorm>                              m_cond_norm_mg;
    MGLevelObject<std::shared_ptr<CoarseConditionBase>>   m_condition_mg;
    MGLevelObject<std::shared_ptr<Functional>>            m_objective_mg;
    MGLevelObject<DescentOptions>                         options_descent_mg;
    MGLevelObject<SolverOptions>                          options_solver_mg;
    FAS_Options                                           options_fas;

    std::vector<Vector<double>> x_hist;
};

} // namespace rmo

#endif //RMO_ROPT_FAS_H
