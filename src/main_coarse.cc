#include <gpe/ropt/manifold.h>
#include <gpe/ropt/coarse.h>
#include <gpe/problem/oracle.h>
#include <gpe/problem/gpe.h>
#include <gpe/option.h>

using namespace dealii;
using namespace gpe;

// TODO: separate preconditioners for gradient descent, and inverse of M (coarse gradients)
using OperatorType   = LinearCombination<SparseMatrix<double>, Vector<double>>;
using InverseOpType  = PreconditionInverse<OperatorType, SparseMatrix<double>>;


template <int dim>
struct ContextMultiLevel
{
    ContextMultiLevel(const OracleBase<dim>& O_descent,
                      const OracleBase<dim>& O_tilt, const OracleBase<dim>& O_tilt_coarse,
                      const CoarseOracleBase<dim>& O_coarse,
                      const LinearTransferBase& transfer,
                      const ManifoldTransferBase& point_transfer,
                      const VectorTransportBase& vector_transport)
        : O_descent(O_descent)
        , O_tilt(O_tilt), O_tilt_coarse(O_tilt_coarse)
        , O_coarse(O_coarse)
        , transfer(transfer), point_transfer(point_transfer)
        , vector_transport(vector_transport)
    {}

    const OracleBase<dim>& O_descent;               // Oracle for gradient descent on fine level
    const OracleBase<dim>& O_tilt;                  // Oracle for coarse correction term on fine level
    const OracleBase<dim>& O_tilt_coarse;           // Oracle for coarse correction term on coarse level
    const CoarseOracleBase<dim>& O_coarse;          // Oracle for coarse model

    const LinearTransferBase& transfer;             // Linear interpolation with Galerkin condition
    const ManifoldTransferBase& point_transfer;     // Point transfer operator
    const VectorTransportBase& vector_transport;    // Vector transport operatorf
};


// Reference problem using energy adaptive gradient descent
template <int dim>
struct ExperimentSingleLevel
{
    ExperimentSingleLevel(const OracleBase<dim>& O_descent, DescentOptions options)
        : m_solver(O_descent), m_options(options)
    {}

    void cycle(const Vector<double>& x0, std::ostream os)
    {
        m_solver.cycle(x0, os, m_options);
    }

private:
    GradientDescent<dim> m_solver;
    DescentOptions m_options;
};


template <int dim, typename TiltOracle>
struct ExperimentMultiLevel
{
    ExperimentMultiLevel(ContextMultiLevel<dim> ctx,
                         DescentOptions options, DescentOptions options_coarse)
        : model(ctx.O_tilt_coarse, ctx.O_tilt, ctx.O_coarse, ctx.point_transfer, ctx.vector_transport)
        , solver(ctx.O_descent, model)
        , m_options(options)
        , m_options_coarse(options_coarse)
    {}

    void cycle(const Vector<double>& x0, std::ostream os,
               double kappa, double eps, unsigned coarse_every)
    {
        solver.cycle(x0, os, m_options, m_options_coarse, kappa, eps, coarse_every);
    }

private:
    CoarseModel<dim, TiltOracle> model;
    FullApproximationScheme<dim, TiltOracle> solver;
    DescentOptions m_options;
    DescentOptions m_options_coarse;
};


int main(int argc, char* argv[])
{
    GPE_Options    options{};
    DescentOptions options_gd{};
    SolverOptions  options_slv{};
    MG_Options     options_mg{};
    FAS_Options    options_fas{};


    // TODO: add configuration file (cf. boost tutorial)
    try {
        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(descent_cli_options());
        all.add(mg_cli_options());
        all.add(inner_cli_options());
        all.add(fas_cli_options());

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.contains("help")) {
            std::cout << all << "\n";
            return 0;
        }

        apply_gpe_options(vm, options);
        apply_descent_options(vm, options_gd);
        apply_mg_options(vm, options_mg);
        apply_inner_options(vm, options_slv);
        apply_fas_options(vm, options_fas);


        with_dimension(options.dimension, [&]<typename T0>(T0)
        {
            constexpr int dim = T0::value;
            SolverOptions options_slv_coarse = options_slv;
            // TODO: reduce tolerance for coarse level

            DescentOptions options_gd_coarse = options_gd;
            // TODO: heuristic: take steps UNTIL armijo line search fails OR max_iter encountered
            options_gd_coarse.max_iter    = 4;
            options_gd_coarse.step_size   = 1.0;
            options_gd_coarse.line_search = false;

            // Default potential
            Square<dim> V;

            // Default refinement for coarse and fine grid
            const int min_i = options_mg.min_level;
            const int min_i_def = 8;
            const int max_i = options_mg.max_level;
            const int max_i_def = 9;
            const int levels_i = options_mg.n_levels;
            const int n_coarse_levels = (min_i == 0 ? min_i_def : min_i);
            const int n_fine_levels = (max_i == levels_i ? max_i_def : max_i);

            // TODO: support more than 2 levels
            AssertThrow(n_fine_levels - n_coarse_levels == 1,
                dealii::ExcNotImplemented("only 2 levels are supported"));


            // Set up discretization for fine and coarse level
            GrossPitaevskiiSimulator<dim, EnergyOracle<dim>> GP_coarse(V, options, n_coarse_levels);
            const auto& problem_coarse = GP_coarse.get_problem();
            const auto& dofs_coarse = GP_coarse.get_dofs();
            const auto& constraints_coarse = GP_coarse.get_constraints();
            const auto& M_coarse = GP_coarse.get_M();
            auto M_inv_coarse = InverseOpType(M_coarse, options_slv_coarse);

            GrossPitaevskiiSimulator<dim, EnergyOracle<dim>> GP_fine(V, options, n_fine_levels);
            const auto& problem_fine = GP_fine.get_problem();
            const auto& dofs_fine = GP_fine.get_dofs();
            const auto& constraints_fine = GP_fine.get_constraints();
            const auto& M_fine = GP_fine.get_M();
            auto M_inv_fine = InverseOpType(M_fine, options_slv);


            // Set up experiments
            // TODO: factory based on enums (option_types.h) <- identical constructor signature
            // Function evaluation
            MassOracle<dim> oracle_mass_coarse(problem_coarse, options.beta, options_slv_coarse);
            MassOracle<dim> oracle_mass_fine(problem_fine, options.beta, options_slv);
            EnergyOracle<dim> oracle_energy_coarse(problem_coarse, options.beta, options_slv_coarse);
            EnergyOracle<dim> oracle_energy_fine(problem_fine, options.beta, options_slv);
            FrobeniusOracle<dim> oracle_frobenius_coarse(problem_coarse, options.beta, options_slv_coarse);
            FrobeniusOracle<dim> oracle_frobenius_fine(problem_fine, options.beta, options_slv);

            // Coarse models
            MassCoarseOracle<dim> oracle_mass_coarse_model(problem_coarse, options.beta, options_slv_coarse);
            MassCoarseOracleEnergyAdaptive<dim> oracle_mass_ea_coarse_model(problem_coarse, options.beta, options_slv_coarse);
            FrobeniusCoarseOracle<dim> oracle_frobenius_coarse_model(problem_coarse, options.beta, options_slv_coarse);
            FrobeniusCoarseOracleEnergyAdaptive<dim> oracle_frobenius_ea_coarse_model(problem_coarse, options.beta, options_slv_coarse);

            // Operators (standard Galerkin condition)
            LinearTransferMatrix<dim> transfer(dofs_coarse, dofs_fine, constraints_coarse, constraints_fine);
            ManifoldTransfer<OperatorType> point_transfer(transfer, M_coarse, M_fine);
            MassProjectionTransport<OperatorType> proj_mass_transport(transfer, M_coarse, M_fine);
            FrobeniusProjectionTransport<OperatorType> proj_frobenius_transport(transfer, M_coarse, M_fine);
            DifferentialTransport<OperatorType> diff_transport(point_transfer, M_coarse, M_fine);

            // Operators (mass Galerkin condition)
            MassTransfer<dim,LinearTransferMatrix<dim>,OperatorType,InverseOpType> mass_transfer(
                dofs_coarse, dofs_fine, constraints_coarse, constraints_fine, M_fine, M_inv_coarse);
            ManifoldTransfer<OperatorType> mass_point_transfer(mass_transfer, M_coarse, M_fine);
            MassProjectionTransport<OperatorType> mass_proj_mass_transport(mass_transfer, M_coarse, M_fine);
            FrobeniusProjectionTransport<OperatorType> mass_proj_frobenius_transport(mass_transfer, M_coarse, M_fine);
            DifferentialTransport<OperatorType> mass_diff_transport(mass_point_transfer, M_coarse, M_fine);

        });
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "Exception of unknown type!\n";
        return 1;
    }
}

