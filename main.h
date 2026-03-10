//
// Created by Ferdinand Vanmaele on 24.02.26.
//
#ifndef GPE_MAIN_H
#define GPE_MAIN_H

#include "manifold.h"
#include "function.h"
#include "lac.h"
#include "gpe.h"
#include "grid_operators.h"

namespace gpe
{

// TODO: define interface and move to `objective.h`, merge functions from `manifold.h`
//       support other preconditioner types
/**
 * @brief Oracle for the Gross-Pitaevskii energy functional.
 * This class provides the interface required by the @ref gradient_descent algorithm.
 * It translates physical concepts (matrices and assembly) into optimization concepts
 * (functional values and gradients).
 *
 * @tparam dim The spatial dimension of the problem.
 */
template <int dim>
class OracleBase
{
public:
    static constexpr int dimension = dim;
    /** @brief Alias for the combined linear operator type. */
    // Light-weight wrapper for the matrices stored in GrossPitaevskiiProblem
    using OperatorType  = LinearCombination<SparseMatrix<double>, Vector<double>>;
    /** @brief Alias for the preconditioned inverse operator type. */
    // Heavy-weight wrapper for matrix preconditioning
    using InverseOpType = PreconditionInverse<OperatorType,SparseMatrix<double>>;

    /**
     * @brief Constructs the Oracle by referencing an existing GPE problem.
     * @param problem_ Reference to the assembled GPE problem.
     * @param beta_ Non-linear coupling constant (interaction strength).
     * @param options_ Solver and tolerance options for the inner iteration.
     * @note The Oracle does not own the problem; it holds a reference. The Problem object
     * must outlive this Oracle.
     */
    // TODO: LinearOptions instead of DescentOptions (separate inner solve and rgd options)
    //       dealii::ObserverPointer instead of const reference for `problem`
    OracleBase(const GrossPitaevskiiProblem<dim>& problem_, double beta_,
               DescentOptions options_)
        : problem(problem_)
        , beta(beta_)
        , options(options_)
        , A(problem.get_operator_A(beta))
        , A_inv(A, options.solver, options.precond, options.max_inner, options.tol_inner)
    {
        A_inv.update_static(problem.get_A0());     // for ILU or AMG
    }
    virtual ~OracleBase() = default;

    // TODO: store `x` argument
    //       set markers `needs_assembly=true`, `need_gradient=true`
    void update(const Vector<double>& x)
    {
        problem.assemble_nonlinear_term(x);        // update M_pp <- A

        A_inv.update_dynamic(A.diagonal());   // updates preconditioner stored in A_inv
    }

    /**
     * @brief Retracts a tangent vector back to the unit-mass manifold.
     * \f[ R_x(z) = \frac{x + z}{\|x + z\|_M} \f]
     * @param z The update vector.
     * @param x The base vector (modified in place).
     * @param factor Step size scaling factor.
     */
    void retract(const Vector<double>& z, Vector<double>& x, double factor = 1.0) const
    {
        const auto M = problem.get_M();
        ellipsoid::retract_by_norm(M, z, x, factor);
    }

    void retract(const Vector<double>& z, const Vector<double>& x,
                 Vector<double>& output, double factor = 1.0) const
    {
        output = x;
        retract(z, output, factor);
    }

    auto get_M() { return problem.get_operator_M(); }
    auto get_A() const { return problem.get_operator_A(beta); }
    double get_beta() const { return beta; }
    unsigned n_dofs() const { return problem.n_dofs(); }

    // Problem / metric-dependent methods
    virtual double value(const Vector<double>& x) const = 0;

    virtual unsigned gradient(const Vector<double>& x, Vector<double>& output) const = 0;

    virtual double metric(const Vector<double>& y, const Vector<double>& z) const = 0;

    virtual iteration::State residual(const Vector<double>& x) const = 0;

    virtual bool check_convergence(const iteration::State& current,
                                   const iteration::State& previous) const { return false; };

private:
    /** @brief Reference to the problem matrices. */
    const GrossPitaevskiiProblem<dim>& problem;
    /** @brief Interaction strength parameter. */
    double beta;
    DescentOptions options;

    /** @brief Total operator \f$ A_0 + \beta M_{pp} \f$. */
    OperatorType A;
    /** @brief Inverse operator \f$(A_0 + \beta M_{pp})^{-1} \f$. */
    InverseOpType A_inv;
};


template <int dim>
class EnergyOracle : public OracleBase<dim>
{
public:
    using OracleBase<dim>::OracleBase;

    /**
     * @brief Computes the Gross-Pitaevskii energy functional value.
     * \f[ E(\phi) = \langle \phi, A_0 \phi \rangle + \frac{\beta}{2} \langle \phi, M_{pp}(\phi) \phi \rangle \f]
     * @param x The current state vector.
     * @return The energy value.
     */
    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_assembly`
    double value(const Vector<double>& x) const override
    {
        const auto A0  = this->problem.get_A0();
        const auto Mpp = this->problem.get_Mpp();

        return ellipsoid::function_value(x, A0, Mpp, this->beta);
    }

    /**
     * @brief Computes the Riemannian gradient.
     * Solves the inner linear system \f$ A^{-1} \nabla E \f$ using the @ref InverseMatrix
     * wrapper and the provided solver options.
     * @param x The current state vector.
     * @param output Vector to store the computed gradient.
     * @return The number of inner iterations performed by the linear solver.
     */
    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_gradient`
    //       enum for selecting metric
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        const auto M = this->problem.get_M();
        ellipsoid::gradient(this->A_inv, M, x, output);

        return this->A_inv.control().last_step();
    }

    /**
     * @brief Computes the iteration state (eigenvalue, residual, etc.) at point x.
     */
    iteration::State residual(const Vector<double>& x) const override
    {
        const auto M = this->problem.get_M();

        return iteration::residual(x, this->A, M);
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        AssertDimension(y.size(), z.size());

        Vector<double> Az(z.size());
        this->A.vmult(Az, y);
        return y*Az;
    }

    /**
     * @brief Checks if the optimization has converged.
     * Convergence is achieved if both the eigenvalue (\f$ \lambda \f$) and the
     * non-linear residual fall below specified tolerances.
     * @param current
     * @param previous
     */
    bool check_convergence(const iteration::State& current,
                           const iteration::State& previous) const override
    {
        const double lmb_diff   = std::abs(current.lambda - previous.lambda);
        const double lmb_factor = 1.0 + std::abs(current.lambda);

        return (lmb_diff < this->options.tol_lambda * lmb_factor && current.residual < this->options.tol_residual);
    }
};


template <int dim>
class CoarseOracle : public OracleBase<dim>
{
public:
    void update_parameters(const Vector<double>& w_new, const Vector<double>& phi_new)
    {
        w = w_new;
        phi = phi_new;
    }

    double value(const Vector<double>& x) const override
    {
        const auto M   = this->problem.get_M();
        const auto A0  = this->problem.get_A0();
        const auto Mpp = this->problem.get_Mpp();

        return coarse::function_value(x, phi, w, M, A0, Mpp, this->beta);
    }

    // TODO: the coarse model has the coarse correction in the M-metric.
    //       either formulate the coarse correction in the A-metric, or compute the M-gradient?
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        const auto M = this->problem.get_M();
        coarse::gradient(M, this->A_inv, x, phi, w, output);

        return this->A_inv.control().last_step();
    }

    // TODO: the coarse model is formulated in the M-metric
    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        AssertDimension(y.size(), z.size());

        Vector<double> Az(z.size());
        this->A.vmult(Az, y);
        return y*Az;
    }

    // XXX: methods for gradient_descent()
    iteration::State residual(const Vector<double>& x) const override
    {
        return {.energy=value(x)};
    }

private:
    Vector<double> w;
    Vector<double> phi;
};

/**
 * @brief Orchestrator for Gross-Pitaevskii simulations.
 * The @ref EnergySimulator manages the persistent @ref GrossPitaevskiiPackage
 * (discretization) and coordinates the execution of the energy minimization
 * using the @ref EnergyOracle.
 *
 * @tparam dim The spatial dimension.
 */
template <int dim>
class EnergySimulator
{
public:
    /**
     * @brief Constructor.
     * @tparam Potential Functor or class representing the external potential \f$ V(x) \f$.
     * @param V The potential object.
     * @param options General options for GPE discretization.
     * @param n_levels Number of global mesh refinements.
     */
    template <typename Potential>
    EnergySimulator(Potential&& V, const GPE_Options& options, unsigned int n_levels)
        : package(options, n_levels)
        , problem(package.problem(std::forward<Potential>(V)))
        , options(options)
    {}

    // Allow to change the potential without re-discretizing the domain.
    template <typename Potential>
    void reinit(Potential&& V)
    {
        problem = package.problem(std::forward<Potential>(V));
    }

    void distribute(Vector<double>& x) const
    {
        package.distribute(x);
    }

    /**
     * @brief Runs the energy minimization for a given potential.
     * @param beta The interaction strength constant.
     * @param options_gd Options for the gradient descent algorithm.
     */
    // TODO factor this out to caller, see get_oracle()
    Vector<double>
    run(const Vector<double>& x0, double beta,
        const DescentOptions& options_gd, std::ostream& os) const
    {
        Assert(x0.size() == package.n_dofs(), dealii::ExcDimensionMismatch(x0.size(), package.n_dofs()));
        // Create the oracle (light-weight object, references problem matrices)
        // TODO: separate linear options (for inner solve/gradient computation)
        //       and descent options (tolerance and step size)
        OracleBase<dim> oracle(problem, beta, options_gd);

        // Riemannian gradient descent
        // Note: the update strategy can be arbitrary complex (e.g. for multilevel algorithms)
        return gradient_descent(oracle, x0, options_gd, os);
    }

    /** @brief Access the discretization package. */
    const GrossPitaevskiiPackage<dim>& get_package() const { return package; }
    const GrossPitaevskiiProblem<dim>& get_problem() const { return problem; }

    unsigned int n_dofs() const { return package.n_dofs(); }

    const dealii::DoFHandler<dim>&
    get_dofs() const { return package.get_dofs(); }

    const dealii::AffineConstraints<double>&
    get_constraints() const { return package.get_constraints(); }

    OracleBase<dim> get_oracle(double beta, DescentOptions options_gd) const
    {
        return OracleBase<dim>(problem, beta, options_gd);
    }

    auto get_M() const { return problem.get_operator_M(); }

    auto get_M_inv(DescentOptions options_gd) const
    {
        using InverseOpType = InverseMatrix<decltype(this->get_M()), dealii::PreconditionIdentity>;

        return InverseOpType(get_M(), options_gd.solver, dealii::PreconditionIdentity{},
            options_gd.max_inner, options_gd.tol_inner);
    }

    auto get_A(double beta) const { return problem.get_operator_A(beta); }

    auto get_A_inv(double beta, DescentOptions options_gd) const
    {
        using InverseOpType = InverseMatrix<decltype(this->get_A(beta)), dealii::PreconditionIdentity>;

        return InverseOpType(get_A(beta), options_gd.solver, dealii::PreconditionIdentity{},
            options_gd.max_inner, options_gd.tol_inner);
    }

private:
    /** @brief Persistent discretization infrastructure. */
    GrossPitaevskiiPackage<dim> package;
    /** @brief Assembly and storage of matrices. */
    GrossPitaevskiiProblem<dim> problem;
    /** @brief Problem configuration options. */
    GPE_Options options;
};


template <typename MatrixType>
double M_norm(const MatrixType& M, const Vector<double>& x)
{
    Vector<double> Ax(x.size());
    M.vmult(Ax, x);
    return std::sqrt(x*Ax);
}

struct CycleInfo
{
    unsigned iter;
    unsigned lac_iter;
    double step_size;
    double elapsed;
    bool coarse;
};

template <typename Oracle>
void cycle_eval(const Oracle& O, const Vector<double>& y,
                dealii::ConvergenceTable& convergence_table,
                CycleInfo info)
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
class FullApproximationScheme
{
public:
    using OperatorType  = LinearCombination<SparseMatrix<double>,Vector<double>>;
    using MatrixType    = SparseMatrix<double>;
    using InverseOpType = InverseMatrix<OperatorType, dealii::PreconditionIdentity>;

    FullApproximationScheme(const EnergySimulator<dim>& GP_coarse,
                            const EnergySimulator<dim>& GP_fine,
                            double beta, DescentOptions options, DescentOptions options_coarse)
        // Function evaluation
        : O_coarse(GP_coarse.get_oracle(beta, options_coarse))
        , O_fine(GP_fine.get_oracle(beta, options))
        , qk(GP_coarse.get_problem(), beta, {}, {})

        // Problem components
        , n_coarse(GP_coarse.n_dofs())
        , n_fine(GP_fine.n_dofs())
        , M_coarse(GP_coarse.get_M()), M_fine(GP_fine.get_M())
        , A_coarse(GP_coarse.get_A(beta)), A_fine(GP_fine.get_A(beta))
        //, M_inv_coarse(GP_coarse.get_M_inv(options)), M_inv_fine(GP_fine.get_M_inv(options))

        // Grid operators
        , transfer(GP_coarse.get_dofs(), GP_fine.get_dofs(), GP_coarse.get_constraints(), GP_fine.get_constraints())
        , point_transfer(M_coarse, M_fine, transfer)
        , vector_transport(M_coarse, M_fine, transfer, point_transfer)
    {} // TODO: Set up preconditioner for M

    void cycle(const Vector<double>& y0, std::ostream& os,
               DescentOptions options, DescentOptions options_coarse,
               unsigned n_pre = 1, unsigned n_post = 1)
    {
        dealii::Timer timer;
        timer.reset();

        // 0. Initialize oracle and coarse model
        Vector<double> y(y0);
        O_fine.update(y);

        unsigned i = 1;
        while (true) {
            // 1. Pre-smoothing
            Vector<double> y_grad(O_fine.n_dofs());

            for (unsigned pre = 0; pre < n_pre; pre++) {
                std::cerr << "Pre-smoothing: " << pre << "\n";
                if (i > options.max_iter) break;

                CycleInfo info = cycle_fine(y, y_grad, options);
                info.iter = i++;
                info.coarse = false;

                cycle_eval(O_fine, y, convergence_table, info);
            }

            // 2. Coarse step
            {
                if (i > options.max_iter) break;

                CycleInfo info = cycle_coarse(y, options, options_coarse);
                info.iter = i++;
                info.coarse = true;

                cycle_eval(O_fine, y, convergence_table, info);
            }

            // 3. Post-smoothing
            for (unsigned post = 0; post < n_post; ++post) {
                std::cerr << "Post-smoothing: " << post << "\n";
                if (i > options.max_iter) break;

                CycleInfo info = cycle_fine(y, y_grad, options);
                info.iter = i++;
                info.coarse = false;

                cycle_eval(O_fine, y, convergence_table, info);
            }
        }
        cycle_finalize(convergence_table, os, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }

private:
    dealii::ConvergenceTable convergence_table;
    dealii::Timer timer;

    // Function evaluation
    OracleBase<dim> O_coarse, O_fine;
    CoarseOracle<dim> qk;
    unsigned n_coarse, n_fine;

    // Grid operators
    LinearTransfer<dim> transfer;
    ManifoldTransfer<dim, OperatorType> point_transfer;
    //EnergyProjectionTransport<dim, MatrixType, InverseOpType> vector_transport;
    ProjectionTransport<dim, OperatorType> vector_transport;  // TODO select components

    OperatorType M_coarse, M_fine;
    OperatorType A_coarse, A_fine;
    //InverseOpType M_inv_coarse, M_inv_fine;

    // Methods
    double line_search(Vector<double>& y, const Vector<double>& y_grad, const Vector<double>& eta,
                       DescentOptions options)
    {
        double threshold = 1e-4;
        double Ex   = O_fine.value(y);
        double dd   = O_fine.metric(y_grad, eta);
        double step = armijo_line_search(O_fine, y, eta, Ex, dd, options, threshold);

        if (step < threshold) {
            std::cerr << "  -> Coarse step rejected by line search." << std::endl;
        }
        return step;
    }

    // TODO: CoarseOracle is *NOT* a light-weight object; it sets up the preconditioner for A0
    //       like EnergyOracle would. Therefore, it should not be initialized for every iteration.
    Vector<double>
    descent_coarse(const Vector<double>& y, DescentOptions options, DescentOptions options_coarse)
    {
        // TODO: use class preconditioner
        //       set up inverse objects in constructor
        InverseOpType M_coarse_inv(M_coarse, options_coarse.solver, dealii::PreconditionIdentity{},
            options_coarse.max_inner, options_coarse.tol_inner);
        InverseOpType M_fine_inv(M_fine, options.solver, dealii::PreconditionIdentity{},
            options.max_inner, options.tol_inner);

        // Starting coarse point
        std::cerr << "[" << timer.cpu_time() << "] coarse: point transfer\n";
        Vector<double> x(n_coarse);
        point_transfer.restriction(y, x);
        O_coarse.update(x);

        // Compute coarse M-gradient
        std::cerr << "[" << timer.cpu_time() << "] coarse: M-coarse gradient\n";
        Vector<double> x_grad_m(n_coarse);
        ellipsoid::gradient(M_coarse_inv, A_coarse, M_coarse, x, x_grad_m);

        // Compute fine M-gradient
        std::cerr << "[" << timer.cpu_time() << "] coarse: M-fine gradient\n";
        Vector<double> y_grad_m(n_fine);
        ellipsoid::gradient(M_fine_inv, A_fine, M_fine, y, y_grad_m);

        // Compute coarse correction step
        std::cerr << "[" << timer.cpu_time() << "] coarse: correction\n";
        Vector<double> w(n_coarse);
        coarse_correction(vector_transport, x_grad_m, y_grad_m, x, y, w);

        // Set up coarse model
        // TODO: preconditioner for coarse A0 is computed for every step
        //       factor this out and call some update function?
        //CoarseOracle<dim> qk(prob_coarse, O_coarse.get_beta(), w, x);
        qk.update_parameters(w, x);
        Vector<double> zk(n_coarse);

        // Find zk such that qk(zk) < qk(x)
        // TODO: variable descent method, or use cycle_fine() for consistency
        std::cerr << "[" << timer.cpu_time() << "] coarse: gradient descent\n";
        zk = gradient_descent(qk, x, options_coarse, std::cerr);

        // Compute the search direction, zk <- L_x(zk)
        std::cerr << "[" << timer.cpu_time() << "] coarse: inverse retraction\n";
        ellipsoid::retract_inv_by_norm(M_coarse, zk, x);
        Vector<double> dk(n_fine);

        // Coarse base point not required for ProjectionTransport
        std::cerr << "[" << timer.cpu_time() << "] coarse: M-vector prolongation\n";
        vector_transport.vector_prolongation(y, {}, zk, dk);
        return dk;
    }

    // Implementation of fine and coarse cycles
    // TODO: factor out Oracle::update (possible with matrix-free line search)
    CycleInfo cycle_fine(Vector<double>& y, Vector<double>& y_grad, DescentOptions options)
    {
        timer.start();

        // Update gradient
        std::cerr << "[" << timer.cpu_time() << "] fine: A-gradient\n";
        auto lac_iter = O_fine.gradient(y, y_grad);
        double step_size = options.step_size;

        // Update point (fixed step or line search)
        if (options.line_search) {
            Vector<double> eta(y_grad);
            eta *= -1.0;

            step_size = line_search(y, y_grad, eta, options);
        }
        else {
            std::cerr << "[" << timer.cpu_time() << "] " << "fine: retraction" << std::endl;
            O_fine.retract(y_grad, y, -options.step_size);  // update y
            std::cerr << "[" << timer.cpu_time() << "] " << "fine: assembly" << std::endl;
            O_fine.update(y);
        }

        timer.stop();
        return {.lac_iter = lac_iter, .step_size = step_size, .elapsed = timer.cpu_time()};
    }

    // TODO: factor out Oracle::update (possible with matrix-free line search)
    CycleInfo cycle_coarse(Vector<double>& y, DescentOptions options, DescentOptions options_coarse)
    {
        timer.start();

        // Coarse descent direction
        Vector<double> dk = descent_coarse(y, options, options_coarse);
        double step_size = options.step_size;

        // Update point (fixed step or line search)
        if (options.line_search) {
            // TODO: dk is computed in the A-metric, but the coarse model is formulated in the M-metric.
            //       This means that we need to compute both the M and the A-gradient, former for the coarse model,
            //       and later for evaluating the Armijo condition on dk.
            Vector<double> y_grad(n_fine);
            auto lac_iter = O_fine.gradient(y, y_grad);

            step_size = line_search(y, y_grad, dk, options);
        }
        else {
            std::cerr << "[" << timer.cpu_time() << "] " << "fine: retraction" << std::endl;
            O_fine.retract(dk, y, options.step_size);
            std::cerr << "[" << timer.cpu_time() << "] " << "fine: assembly" << std::endl;
            O_fine.update(y);
        }

        timer.stop();
        return {.lac_iter = 0, .step_size = step_size, .elapsed = timer.cpu_time()};
    }
};

} // namespace gpe

#endif //GPE_MAIN_H