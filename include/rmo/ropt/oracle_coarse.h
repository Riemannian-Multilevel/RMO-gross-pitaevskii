#ifndef RMO_ROPT_ORACLE_COARSE_H
#define RMO_ROPT_ORACLE_COARSE_H

#include <rmo/ropt/oracle.h>
#include <rmo/ropt/residual.h>
#include <rmo/ropt/manifold.h>
#include <rmo/ropt/transport.h>

#include <deal.II/base/timer.h>

#include <concepts>

/**
 * @file
 * @brief CoarseOracleBase: the problem-independent part of the coarse models of FullApproximationScheme, and the
 * concepts of the coarse oracles and coarse residuals.
 */
namespace rmo
{

//! Points, gradients and correction of the current coarse model, see CoarseOracleBase::update_model().
// TODO: keep track of fine vector for consistency
struct CoarseState
{
    Vector<double> x;             // fine point
    Vector<double> y;             // restricted point (base point for coarse model)
    Vector<double> y_grad;        // gradient of restricted point
    Vector<double> x_grad;        // fine gradient
    Vector<double> x_grad_restr;  // restricted gradient
    Vector<double> w;             // correction vector

    CoarseState(unsigned n_fine, unsigned n_coarse)
        : x(n_fine)
        , y(n_coarse)
        , y_grad(n_coarse)
        , x_grad(n_fine)
        , x_grad_restr(n_coarse)
        , w(n_coarse)
    {}
};


/**
 * @brief Problem-independent part of the coarse model: from a fine and a coarse tilt oracle (with the same
 * metric) and the transfers, update_model() computes the restricted point \f$ y = r(x) \f$ and the correction
 * \f$ w = \grad f_H(y) - \mathcal{R} \grad f_h(x) \f$ used by the coarse oracles.
 */
class CoarseOracleBase
{
public:
    CoarseOracleBase(OracleBase &T_fine,
                     OracleBase &T_coarse,
                     const ManifoldBase &coarse_manifold,
                     const ManifoldTransferBase &point_transfer,
                     const VectorTransportBase  &vector_transport)
    // Problem evaluation
        : T_fine(T_fine), T_coarse(T_coarse)
        , coarse_manifold(coarse_manifold)
        , n_fine(T_fine.n_dofs())
        , n_coarse(T_coarse.n_dofs())

    // Grid transfer
        , point_transfer(point_transfer)
        , vector_transport(vector_transport)

    // Coarse parameter initialization
        , m_state(n_fine, n_coarse)
    {
        AssertThrow(T_fine.metric().kind() == T_coarse.metric().kind(),
            dealii::ExcInternalError("non-corresponding metrics for coarse and fine oracle types"));
    }

    // Compute parameters for coarse model
    // Note that unlike the coarse models - which take a coarse vector as argument - this takes a fine vector.
    // Models are based on the difference w of coarse gradients, and restricted fine gradients.
    void update_model(const Vector<double>& x_fine, const double model_tol = -1.0)
    {
        AssertDimension(x_fine.size(), n_fine);
        m_state.x = x_fine;
        // TODO: Underlying state of T_fine assumed to match x (OracleBase::update -> GrossPitaevskiiSystem::update)

        // Compute base point for coarse model
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: point transfer\n";
#endif
        point_transfer.restriction(m_state.x, m_state.y);
        AssertDimension(m_state.y.size(), n_coarse);

#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: assemble matrix\n";
#endif
        T_coarse.update(m_state.y);

        // Compute coarse (F or M)-gradient
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << T_coarse.id() << "-coarse gradient\n";
#endif
        // Set tolerance for coarse gradient defining the coarse model
        // TODO: model_tol is multiplied by options.tol_inner_res inside OracleBase::gradient() implementations
        if (model_tol > 0.0) {
            T_coarse.gradient(m_state.y, m_state.y_grad, model_tol);
        } else {
            T_coarse.gradient(m_state.y, m_state.y_grad);  // set based on residual of coarse objective
        }
        AssertDimension(m_state.y_grad.size(), n_coarse);

        // Compute fine (F or M)-gradient
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << T_fine.id() << "-fine gradient\n";
#endif
        // Set tolerance for fine gradient defining the coarse model
        // TODO: model_tol is multiplied by options.tol_inner_res inside OracleBase::gradient() implementations
        if (model_tol > 0.0) {
            T_fine.gradient(m_state.x, m_state.x_grad, model_tol);
        } else {
            T_fine.gradient(m_state.x, m_state.x_grad);   // set based on residual of fine objective
        }

        AssertDimension(m_state.x_grad.size(), n_fine);

        // Compute restricted gradient
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: M-vector restriction\n";
#endif
        vector_transport.vector_restriction(m_state.y, m_state.x, m_state.x_grad, m_state.x_grad_restr);
        AssertDimension(m_state.x_grad_restr.size(), n_coarse);

        // Compute correction term
        m_state.w = m_state.y_grad;
        m_state.w.add(-1.0, m_state.x_grad_restr);
    }

    //! Metric of the coarse model, i.e. of the coarse tilt oracle
    const MetricBase& metric() const { return T_coarse.metric(); }

    void set_timer(const dealii::Timer& timer_new) const { timer = timer_new; }
    const CoarseState& get_state() const { return m_state; }

    const OracleBase& fine() const { return T_fine; }  // fine tilt oracle
    OracleBase& fine() { return T_fine; }

    const OracleBase& coarse() const { return T_coarse; }  // coarse tilt oracle
    OracleBase& coarse() { return T_coarse; }

    const ManifoldBase& manifold() const { return coarse_manifold; }


protected:
    // Coarse and fine level evaluation for correction vector w
    OracleBase &T_fine, &T_coarse;
    const ManifoldBase &coarse_manifold;
    unsigned n_fine, n_coarse;

    // Operators for transferring solutions and gradients
    const ManifoldTransferBase &point_transfer;
    const VectorTransportBase  &vector_transport;

    // Coarse model parameters
    CoarseState m_state;

    // Benchmarking
    mutable dealii::Timer timer;
};


// Contract for the coarse oracles handed to FullApproximationScheme::cycle():
// an OracleBase built on top of the problem-independent CoarseOracleBase.
template <typename T>
concept CoarseOracle = std::derived_from<T, OracleBase>
                    && std::constructible_from<T, CoarseOracleBase&, SolverOptions>;

// Contract for the residual of a coarse model, built by FullApproximationScheme::cycle() from the coarse
// model and the functional of the coarse level
template <typename T, typename Functional>
concept CoarseResidual = std::derived_from<T, ResidualBase>
                      && std::constructible_from<T, const CoarseOracleBase&, const Functional&>;

} // namespace rmo

#endif //RMO_ROPT_ORACLE_COARSE_H
