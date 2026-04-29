//
// Created by Ferdinand Vanmaele on 08.04.26.
//
#ifndef GPE_MAIN_COARSE_H
#define GPE_MAIN_COARSE_H

#include <gpe/lac.h>
#include <gpe/problem/oracle.h>
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


struct CoarseStep
{
    CoarseStep(const Vector<double>& y, const unsigned n_coarse, const unsigned n_fine)
        : x(y), y(n_coarse), y_grad(n_coarse), x_grad(n_fine), x_grad_restr(n_coarse)
    {
        AssertDimension(y.size(),n_fine);
    }

    const Vector<double>& x;     // fine point (unchanged in coarse cycle)
    Vector<double> y;            // coarse point
    Vector<double> y_grad;       // (M-)coarse gradient
    Vector<double> x_grad;       // (M-)fine gradient
    Vector<double> x_grad_restr; // restricted gradient
};


// TODO: CoarseModel decides which metric DEFINES the Nash model.
//       CoarseOracle decides which metric SOLVES the Nash model.
//       -> Do not hardcode the Oracle, but take a common interface.
template <int dim, typename Oracle, typename CoarseOracle, typename VectorTransport>
class CoarseModel
{
public:
    using OperatorType   = LinearCombination<SparseMatrix<double>,Vector<double>>;
    using MatrixType     = SparseMatrix<double>;
    using InverseOpType  = PreconditionInverse<OperatorType, SparseMatrix<double>>;
    using Context        = MatrixContext<OperatorType>;
    using InverseContext = InverseMatrixContext<InverseOpType>;

    // CoarseModel(const OracleBase<dim>& O_coarse, const OracleBase<dim>& O_fine, const OracleBase<dim>& qk,
    //             const Context& ctx, const InverseContext& inv_ctx)

    CoarseModel(const GrossPitaevskiiProblem<dim>& problem_coarse,
                const GrossPitaevskiiProblem<dim>& problem_fine,
                const LinearTransferBase& transfer, double beta,
                SolverOptions options, SolverOptions options_coarse)
        // Function evaluation
        : options(options), options_coarse(options_coarse)
        , O_coarse(problem_coarse, beta, options_coarse)
        , O_fine(problem_fine, beta, options)
        , qk(problem_coarse, beta, options)

        // Problem components
        , n_coarse(problem_coarse.n_dofs())
        , n_fine(problem_fine.n_dofs())
        , M_coarse(problem_coarse.get_operator_M())
        , M_fine(problem_fine.get_operator_M())
        , A_coarse(problem_coarse.get_operator_A(beta))
        , A_fine(problem_fine.get_operator_A(beta))  // only needed for A-gradient
        // TODO: separate preconditioners for gradient descent (qk), and inverse of M (coarse gradients)
        , M_coarse_inv(M_coarse, options_coarse)
        , M_fine_inv(M_fine, options)
        , A_coarse_inv(A_coarse, options_coarse)
        , A_fine_inv(A_fine, options)

        // Grid operators
        , transfer(transfer)
        , context(M_coarse, M_fine, A_coarse, A_fine)
        , inverse_context(M_coarse_inv, M_fine_inv, A_coarse_inv, A_fine_inv)
        , point_transfer(context, transfer)
        // TODO: VectorTransport<Context,void,Transfer,PointTransfer> instead of if constexpr
        , vector_transport([&]() {
            if constexpr (VectorTransport::requires_inverse) {
                return VectorTransport(context, inverse_context, transfer, point_transfer);
            } else {
                return VectorTransport(context, transfer, point_transfer);
            }
        }())
    {}

    void set_timer(const dealii::Timer& timer_new)
    {
        timer = timer_new;
    }

    // We separate the "setup" phase (fine/coarse vectors) from the "model" phase
    // so that a coarse criterion can be efficiently evaluated.
    CoarseStep
    setup(const Vector<double>& x)
    {
        AssertDimension(x.size(),n_fine);
        CoarseStep step(x, n_coarse, n_fine);

        // Starting coarse point
        std::cerr << "[" << timer.cpu_time() << "] coarse: point transfer\n";
        point_transfer.restriction(x, step.y);

        std::cerr << "[" << timer.cpu_time() << "] coarse: assemble matrix\n";
        O_coarse.update(step.y);

        // Compute coarse M-gradient
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << O_coarse.id << "-coarse gradient\n";
        O_coarse.gradient(step.y, step.y_grad);

        // Compute fine M-gradient
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << O_fine.id << "-fine gradient\n";
        O_fine.gradient(x, step.x_grad);

        // Compute coarse correction step
        std::cerr << "[" << timer.cpu_time() << "] coarse: M-vector restriction\n";
        vector_transport.vector_restriction(step.y, x, step.x_grad, step.x_grad_restr);

        return step;
    }

    // TODO: Use cycle_fine() for consistency instead of gradient_descent()
    void solve(const CoarseStep& step, DescentOptions options_gd, Vector<double>& dst)
    {
        AssertDimension(dst.size(),n_fine);

        Vector<double> w(n_coarse);
        w = step.y_grad;
        w.add(-1.0, step.x_grad_restr);

        // Set up coarse model
        qk.update_parameters(w, step.y);
        Vector<double> zk(n_coarse);

        // Find zk such that qk(zk) < qk(x)
        // TODO: variable method
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << qk.id << "-gradient descent\n";
        zk = gradient_descent(qk, step.y, options_gd, std::cerr);

        // Compute the search direction, zk <- L_x(zk)
        // FIXME? use separate variable to hold ambient tangent vector L_x(zk)
        std::cerr << "[" << timer.cpu_time() << "] coarse: inverse retraction\n";
        ellipsoid::retract_inv_by_norm(M_coarse, zk, step.y);

        std::cerr << "[" << timer.cpu_time() << "] coarse: " << vector_transport.id << "-vector prolongation\n";
        vector_transport.vector_prolongation(step.x, step.y, zk, dst);
    }

    unsigned n_dofs() const { return n_coarse; }

private:
    dealii::Timer timer;
    SolverOptions options, options_coarse;

    // Function evaluation
    Oracle O_coarse, O_fine;
    CoarseOracle qk;
    // std::unique_ptr<OracleBase<dim>> O_coarse;  // coarse objective
    // std::unique_ptr<OracleBase<dim>> O_fine;    // fine objective
    // std::unique_ptr<CoarseOracleBase<dim>> qk;  // coarse objective + linear shift
    unsigned n_coarse, n_fine;

    OperatorType M_coarse, M_fine;
    OperatorType A_coarse, A_fine;
    InverseOpType M_coarse_inv, M_fine_inv;
    InverseOpType A_coarse_inv, A_fine_inv;

    // Grid operators
    const LinearTransferBase& transfer;
    Context context;  // TODO: redundant
    InverseContext inverse_context;
    ManifoldTransfer<OperatorType> point_transfer;
    VectorTransport vector_transport;
};

} // namespace gpe

#endif //GPE_MAIN_COARSE_H
