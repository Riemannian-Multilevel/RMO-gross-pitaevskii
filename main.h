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

// TODO: refactor into UnitMassSphere (<- Manifold) and Oracle
/**
 * @brief Base Oracle for the Gross-Pitaevskii energy functional.
 * This class translates physical concepts (matrices and assembly) into optimization concepts
 * (functional values and gradients) for use in Riemannian descent algorithms.
 *
 * @tparam dim The spatial dimension of the problem.
 */
template <int dim>
class OracleBase
{
public:
    static constexpr int dimension = dim;
    using OperatorType  = LinearCombination<SparseMatrix<double>, Vector<double>>;
    using InverseOpType = PreconditionInverse<OperatorType, SparseMatrix<double>>;

    /**
     * @brief Constructs the Oracle by referencing an existing GPE problem.
     * @note The Oracle holds a reference; the Problem object must outlive this Oracle.
     */
    OracleBase(const GrossPitaevskiiProblem<dim>& problem_, double beta_, SolverOptions options_)
        : problem(problem_)
        , beta(beta_)
        , options(options_)
        , A(problem.get_operator_A(beta))
        , A_inv(A, options_)
    {
        A_inv.update_static(problem.get_A0());
    }

    virtual ~OracleBase() = default;

    /**
     * @brief Updates the problem state and preconditioner for a new evaluation point.
     * @note This method is NOT const, as it mutates the internal operator state.
     */
    void update(const Vector<double>& x)
    {
        problem.assemble_nonlinear_term(x);
        A_inv.update_dynamic(A.diagonal());
    }

    /**
     * @brief Retracts a tangent vector back to the unit-mass manifold.
     * $$ R_x(z) = \frac{x + z}{\|x + z\|_M} $$
     */
    void retract(const Vector<double>& z, Vector<double>& x, double factor = 1.0) const
    {
        ellipsoid::retract_by_norm(problem.get_M(), z, x, factor);
    }

    void retract(const Vector<double>& z, const Vector<double>& x,
                 Vector<double>& output, double factor = 1.0) const
    {
        output = x;
        retract(z, output, factor);
    }

    // Accessors
    const auto& get_M() const { return problem.get_M(); }
    const auto& get_A() const { return A; }
    double get_beta() const { return beta; }
    unsigned n_dofs() const { return problem.n_dofs(); }

    // Pure virtual interface
    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_assembly`
    virtual double value(const Vector<double>&) const = 0;

    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_gradient
    virtual unsigned gradient(const Vector<double>&, Vector<double>&) const = 0;

    virtual double metric(const Vector<double>&, const Vector<double>&) const = 0;
    virtual iteration::State residual(const Vector<double>&) const = 0;

protected:
    const GrossPitaevskiiProblem<dim>& problem;
    double beta;
    SolverOptions options;
    OperatorType A;
    InverseOpType A_inv;
};


template <int dim>
class EnergyOracle : public OracleBase<dim>
{
public:
    // Inherit constructors from OracleBase
    using OracleBase<dim>::OracleBase;

    /**
     * @brief Computes the Gross-Pitaevskii energy functional value.
     * $$ E(\phi) = \langle \phi, A_0 \phi \rangle + \frac{\beta}{2} \langle \phi, M_{pp}(\phi) \phi \rangle $$
     */
    double value(const Vector<double>& x) const override
    {
        return ellipsoid::function_value(x, this->problem.get_A0(), this->problem.get_Mpp(), this->beta);
    }

    /**
     * @brief Computes the Riemannian gradient in the A-metric.
     * Solves the inner linear system $ A^{-1} \nabla E $ using the PreconditionInverse wrapper.
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        ellipsoid::gradient(this->A_inv, this->problem.get_M(), x, output);
        return this->A_inv.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        return iteration::residual(x, this->A, this->problem.get_operator_M());
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        AssertDimension(y.size(), z.size());
        Vector<double> Az(z.size());
        this->A.vmult(Az, y);
        return y * Az;
    }
};


template <int dim>
class CoarseOracle : public OracleBase<dim>
{
public:
    // Explicit constructor to initialize the correction parameters
    CoarseOracle(const GrossPitaevskiiProblem<dim>& problem_, double beta_,
                 const Vector<double>& w_, const Vector<double>& phi_,
                 SolverOptions options_)
        : OracleBase<dim>(problem_, beta_, options_)
        , w(w_), phi(phi_)
    {}

    void update_parameters(const Vector<double>& w_new, const Vector<double>& phi_new)
    {
        w = w_new;
        phi = phi_new;
    }

    double value(const Vector<double>& x) const override
    {
        return coarse::function_value(x, phi, w, this->problem.get_M(),
                                      this->problem.get_A0(), this->problem.get_Mpp(), this->beta);
    }

    /**
     * @brief Computes the coarse model gradient in the M-metric.
     * $$ \nabla_M q_k(x) = \nabla_M E_H(x) - w $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        coarse::gradient(this->problem.get_M(), this->A_inv, x, phi, w, output);
        //coarse::gradient(this->problem.get_M(), x, phi, w, output);
        return 0; // No linear solver needed for pure M-gradient
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const override
    {
        // Metric for line search evaluation on the coarse grid
        AssertDimension(y.size(), z.size());
        Vector<double> Az(z.size());
        this->A.vmult(Az, y);
        return y * Az;
    }

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
     * @param x0
     * @param beta The interaction strength constant.
     * @param options_inner
     * @param options_gd Options for the gradient descent algorithm.
     */
    // TODO factor this out to caller, see get_oracle()
    Vector<double>
    run(const Vector<double>& x0, double beta,
        const SolverOptions&  options_inner,
        const DescentOptions& options_gd, std::ostream& os) const
    {
        Assert(x0.size() == package.n_dofs(), dealii::ExcDimensionMismatch(x0.size(), package.n_dofs()));
        // Create the oracle (light-weight object, references problem matrices)
        EnergyOracle<dim> oracle(problem, beta, options_inner);

        // Termination criterium
        auto conv_check = [&options_gd](const iteration::State& current, const iteration::State& previous)
        {
            const double lmb_diff   = std::abs(current.lambda - previous.lambda);
            const double lmb_factor = 1.0 + std::abs(current.lambda);

            return (lmb_diff < options_gd.tol_lambda * lmb_factor && current.residual < options_gd.tol_residual);
        };

        // Riemannian gradient descent
        // Note: the update strategy can be arbitrary complex (e.g. for multilevel algorithms)
        return gradient_descent(oracle, x0, options_gd, os, conv_check);
    }

    /** @brief Access the discretization package. */
    const GrossPitaevskiiPackage<dim>& get_package() const { return package; }
    const GrossPitaevskiiProblem<dim>& get_problem() const { return problem; }

    unsigned int n_dofs() const { return package.n_dofs(); }

    const dealii::DoFHandler<dim>&
    get_dofs() const { return package.get_dofs(); }

    const dealii::AffineConstraints<double>&
    get_constraints() const { return package.get_constraints(); }

    EnergyOracle<dim> get_oracle(double beta, SolverOptions options_gd) const
    {
        return EnergyOracle<dim>(problem, beta, options_gd);
    }

    auto get_M() const { return problem.get_operator_M(); }

    auto get_M_inv(SolverOptions options_gd) const
    {
        using InverseOpType = InverseMatrix<decltype(this->get_M()), dealii::PreconditionIdentity>;

        return InverseOpType(get_M(), options_gd.solver, dealii::PreconditionIdentity{},
            options_gd.max_inner, options_gd.tol_inner);
    }

    auto get_A(double beta) const { return problem.get_operator_A(beta); }

    auto get_A_inv(double beta, SolverOptions options_gd) const
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

struct CoarseStep
{
    CoarseStep(const Vector<double>& y, unsigned n_coarse, unsigned n_fine)
        : y(y), x(n_coarse), x_grad(n_coarse), y_grad(n_fine), y_grad_restr(n_coarse)
    {
        AssertDimension(y.size(),n_fine);
    }

    const Vector<double>& y;     // fine point (unchanged in coarse cycle)
    Vector<double> x;            // coarse point
    Vector<double> x_grad;       // (M-)coarse gradient
    Vector<double> y_grad;       // (M-)fine gradient
    Vector<double> y_grad_restr; // restricted gradient
};


template <int dim>
class CoarseModel
{
public:
    using OperatorType  = LinearCombination<SparseMatrix<double>,Vector<double>>;
    using MatrixType    = SparseMatrix<double>;
    using InverseOpType = PreconditionInverse<OperatorType, SparseMatrix<double>>;

    CoarseModel(const GrossPitaevskiiProblem<dim>& problem_coarse,
                const GrossPitaevskiiProblem<dim>& problem_fine,
                const LinearTransferBase& transfer,
                double beta, SolverOptions options, SolverOptions options_coarse)
        // Function evaluation
        : options(options), options_coarse(options_coarse)
        , O_coarse(problem_coarse, beta, options_coarse)
        , qk(problem_coarse, beta, {}, {}, options)

        // Problem components
        , n_coarse(problem_coarse.n_dofs())
        , n_fine(problem_fine.n_dofs())
        , M_coarse(problem_coarse.get_operator_M())
        , M_fine(problem_fine.get_operator_M())
        , A_coarse(problem_coarse.get_operator_A(beta))
        , A_fine(problem_fine.get_operator_A(beta))  // only needed for A-gradient
        , M_coarse_inv(M_coarse, options_coarse)
        , M_fine_inv(M_fine, options)

        // Grid operators
        , transfer(transfer)
        , point_transfer(M_coarse, M_fine, transfer)
        , vector_transport(M_coarse, M_fine, transfer, point_transfer)
    {}

    // We separate the "setup" phase (fine/coarse vectors) from the "model" phase
    // so that a coarse criterion can be efficiently evaluated.
    CoarseStep setup(const Vector<double>& y)
    {
        CoarseStep step(y, n_coarse, n_fine);

        // Starting coarse point
        std::cerr << "[" << timer.cpu_time() << "] coarse: point transfer\n";
        point_transfer.restriction(y, step.x);

        std::cerr << "[" << timer.cpu_time() << "] coarse: assemble matrix\n";
        O_coarse.update(step.x);

        // Compute coarse M-gradient
        std::cerr << "[" << timer.cpu_time() << "] coarse: M-coarse gradient\n";
        ellipsoid::gradient(M_coarse_inv, A_coarse, M_coarse, step.x, step.x_grad);

        // Compute fine M-gradient
        std::cerr << "[" << timer.cpu_time() << "] coarse: M-fine gradient\n";
        ellipsoid::gradient(M_fine_inv, A_fine, M_fine, y, step.y_grad);

        // Compute coarse correction step
        std::cerr << "[" << timer.cpu_time() << "] coarse: vector transfer\n";
        vector_transport.vector_restriction(step.x, y, step.y_grad, step.y_grad_restr);

        return step;
    }

    // TODO: Use cycle_fine() for consistency instead of gradient_descent()
    void solve(const CoarseStep& step, DescentOptions options_gd, Vector<double>& dst)
    {
        Vector<double> w(n_coarse);
        w = step.x;
        w.add(-1.0, step.y_grad_restr);

        // Set up coarse model
        qk.update_parameters(w, step.x);
        Vector<double> zk(n_coarse);

        // Find zk such that qk(zk) < qk(x)
        std::cerr << "[" << timer.cpu_time() << "] coarse: gradient descent\n";
        zk = gradient_descent(qk, step.x, options_gd, std::cerr);

        // Compute the search direction, zk <- L_x(zk)
        std::cerr << "[" << timer.cpu_time() << "] coarse: inverse retraction\n";
        ellipsoid::retract_inv_by_norm(M_coarse, zk, step.x);
        Vector<double> dk(n_fine);

        // Coarse base point not required for ProjectionTransport
        std::cerr << "[" << timer.cpu_time() << "] coarse: M-vector prolongation\n";
        vector_transport.vector_prolongation(step.y, {}, zk, dst);
    }

private:
    dealii::Timer timer; // TODO: global timer?
    SolverOptions options, options_coarse;

    // Function evaluation
    EnergyOracle<dim> O_coarse;
    CoarseOracle<dim> qk;
    unsigned n_coarse, n_fine;

    OperatorType M_coarse, M_fine;
    OperatorType A_coarse, A_fine;
    InverseOpType M_coarse_inv, M_fine_inv;

    // Grid operators
    const LinearTransferBase& transfer;
    ManifoldTransfer<dim, OperatorType> point_transfer;
    //EnergyProjectionTransport<dim, MatrixType, InverseOpType> vector_transport;
    ProjectionTransport<dim, OperatorType> vector_transport;  // TODO select vector transport (enum)
};


template <int dim>
class FullApproximationScheme
{
public:
    using OperatorType  = LinearCombination<SparseMatrix<double>,Vector<double>>;
    using MatrixType    = SparseMatrix<double>;
    using InverseOpType = PreconditionInverse<OperatorType, SparseMatrix<double>>;

    // TODO: Only keep coarse step logic, to allow for different schemes (FAS, Gratton, ...)
    FullApproximationScheme(const GrossPitaevskiiProblem<dim>& problem_coarse,
                            const GrossPitaevskiiProblem<dim>& problem_fine,
                            const LinearTransferBase& transfer,
                            double beta, SolverOptions options, SolverOptions options_coarse)
        // Function evaluation
        : options(options), options_coarse(options_coarse)
        , O_coarse(problem_coarse, beta, options_coarse)
        , O_fine(problem_fine, beta, options)
        , qk(problem_coarse, beta, {}, {}, options)

        // Problem components
        , n_coarse(problem_coarse.n_dofs())
        , n_fine(problem_fine.n_dofs())
        , M_coarse(problem_coarse.get_operator_M())
        , M_fine(problem_fine.get_operator_M())
        , A_coarse(problem_coarse.get_operator_A(beta))
        , A_fine(problem_fine.get_operator_A(beta))
        , M_coarse_inv(M_coarse, options_coarse)
        , M_fine_inv(M_fine, options)

        // Grid operators
        , transfer(transfer)
        , point_transfer(M_coarse, M_fine, transfer)
        , vector_transport(M_coarse, M_fine, transfer, point_transfer)
    {} // TODO: convergence check (cf. EnergySimulator::run)

    void cycle(const Vector<double>& y0, std::ostream& os,
               DescentOptions options_gd, DescentOptions options_gd_coarse,
               unsigned n_pre = 1, unsigned n_post = 1)
    {
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
                if (i > options_gd.max_iter) break;

                CycleInfo info = cycle_fine(y, y_grad, options_gd);
                info.iter = i++;
                info.coarse = false;

                cycle_eval(O_fine, y, convergence_table, info);
            }

            // 2. Coarse step
            {
                if (i > options_gd.max_iter) break;

                CycleInfo info = cycle_coarse(y, options_gd, options_gd_coarse);
                info.iter = i++;
                info.coarse = true;

                cycle_eval(O_fine, y, convergence_table, info);
            }

            // 3. Post-smoothing
            for (unsigned post = 0; post < n_post; ++post) {
                std::cerr << "Post-smoothing: " << post << "\n";
                if (i > options_gd.max_iter) break;

                CycleInfo info = cycle_fine(y, y_grad, options_gd);
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
    SolverOptions options, options_coarse;

    // Function evaluation
    EnergyOracle<dim> O_coarse, O_fine;
    CoarseOracle<dim> qk;
    unsigned n_coarse, n_fine;

    OperatorType M_coarse, M_fine;
    OperatorType A_coarse, A_fine;
    InverseOpType M_coarse_inv, M_fine_inv;

    // Grid operators
    const LinearTransferBase& transfer;
    ManifoldTransfer<dim, OperatorType> point_transfer;
    //EnergyProjectionTransport<dim, MatrixType, InverseOpType> vector_transport;
    ProjectionTransport<dim, OperatorType> vector_transport;  // TODO select vector transport (enum)

    // Methods
    double line_search(Vector<double>& y, const Vector<double>& y_grad, const Vector<double>& eta,
                       DescentOptions options_gd)
    {
        double threshold = 1e-4;
        double Ex   = O_fine.value(y);
        double dd   = O_fine.metric(y_grad, eta);
        double step = armijo_line_search(O_fine, y, eta, Ex, dd, options_gd, threshold);

        if (step < threshold) {
            std::cerr << "  -> Coarse step rejected by line search." << std::endl;
        }
        return step;
    }

    // TODO: Separate computation of coarse vectors (for coarse condition) from minimization of coarse model
    //       CoarseDescent class?
    Vector<double>
    descent_coarse(const Vector<double>& y, DescentOptions options_gd_coarse)
    {
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
        qk.update_parameters(w, x);
        Vector<double> zk(n_coarse);

        // Find zk such that qk(zk) < qk(x)
        // TODO: Use cycle_fine() for consistency
        std::cerr << "[" << timer.cpu_time() << "] coarse: gradient descent\n";
        zk = gradient_descent(qk, x, options_gd_coarse, std::cerr);

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
    // TODO: refactor to cycle_smoother()
    CycleInfo cycle_fine(Vector<double>& y, Vector<double>& y_grad, DescentOptions options_gd)
    {
        timer.start();

        // Update gradient
        std::cerr << "[" << timer.cpu_time() << "] fine: A-gradient\n";
        auto lac_iter = O_fine.gradient(y, y_grad);
        double step_size = options_gd.step_size;

        // Update point (fixed step or line search)
        if (options_gd.line_search) {
            Vector<double> eta(y_grad);
            eta *= -1.0;

            step_size = line_search(y, y_grad, eta, options_gd);
        }
        else {
            std::cerr << "[" << timer.cpu_time() << "] " << "fine: retraction" << std::endl;
            O_fine.retract(y_grad, y, -options_gd.step_size);  // update y
            std::cerr << "[" << timer.cpu_time() << "] " << "fine: assembly" << std::endl;
            O_fine.update(y);
        }

        timer.stop();
        return {.lac_iter = lac_iter, .step_size = step_size, .elapsed = timer.cpu_time()};
    }

    // TODO: refactor to cycle_smoother()
    CycleInfo cycle_coarse(Vector<double>& y, DescentOptions options_gd, DescentOptions options_gd_coarse)
    {
        timer.start();

        // Coarse descent direction
        Vector<double> dk = descent_coarse(y, options_gd_coarse);
        double step_size = options_gd.step_size;

        // Update point (fixed step or line search)
        if (options_gd.line_search) {
            // TODO: dk is computed in the A-metric, but the coarse model is formulated in the M-metric.
            //       This means that we need to compute both the M and the A-gradient, former for the coarse model,
            //       and later for evaluating the Armijo condition on dk.
            //       Just computed both in either the A- or M-metric?
            Vector<double> y_grad(n_fine);
            auto lac_iter = O_fine.gradient(y, y_grad);

            step_size = line_search(y, y_grad, dk, options_gd);
        }
        else {
            std::cerr << "[" << timer.cpu_time() << "] " << "fine: retraction" << std::endl;
            O_fine.retract(dk, y, options_gd.step_size);
            std::cerr << "[" << timer.cpu_time() << "] " << "fine: assembly" << std::endl;
            O_fine.update(y);
        }

        timer.stop();
        return {.lac_iter = 0, .step_size = step_size, .elapsed = timer.cpu_time()};
    }
};

} // namespace gpe

#endif //GPE_MAIN_H