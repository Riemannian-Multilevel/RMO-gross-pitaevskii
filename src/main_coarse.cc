#include <gpe/main/model.h>
#include <gpe/ropt/fas.h>
#include <gpe/problem/oracle_coarse.h>
#include <gpe/problem/gpe.h>
#include <gpe/option.h>

using namespace dealii;
using namespace gpe;


template <int dim>
struct ContextMultiLevel
{
    ContextMultiLevel(const GrossPitaevskiiOracle<dim>& O_descent,
                      const GrossPitaevskiiOracle<dim>& O_tilt, const GrossPitaevskiiOracle<dim>& O_tilt_coarse,
                      CoarseOracleBase<dim>& O_coarse,
                      const LinearTransferBase& transfer,
                      const ManifoldTransferBase& point_transfer,
                      const VectorTransportBase& vector_transport)
        : O_descent(O_descent)
        , O_tilt(O_tilt), O_tilt_coarse(O_tilt_coarse)
        , O_coarse(O_coarse)
        , transfer(transfer), point_transfer(point_transfer)
        , vector_transport(vector_transport)
    {}

    const GrossPitaevskiiOracle<dim>& O_descent;               // Oracle for gradient descent on fine level
    const GrossPitaevskiiOracle<dim>& O_tilt;                  // Oracle for coarse correction term on fine level
    const GrossPitaevskiiOracle<dim>& O_tilt_coarse;           // Oracle for coarse correction term on coarse level
    // TODO: const correctness (update_parameters) - calculation of tilt term inside CoarseOracleBase
    CoarseOracleBase<dim>& O_coarse;          // Oracle for coarse model

    const LinearTransferBase& transfer;             // Linear interpolation with Galerkin condition
    const ManifoldTransferBase& point_transfer;     // Point transfer operator
    const VectorTransportBase& vector_transport;    // Vector transport operatorf
};


// Reference problem using energy adaptive gradient descent
template <int dim>
struct ExperimentSingleLevel
{
    ExperimentSingleLevel(const GrossPitaevskiiOracle<dim>& O_descent, DescentOptions options)
        : m_solver(O_descent), m_options(options)
    {}

    void cycle(const Vector<double>& x0, std::ostream& os)
    {
        m_solver.cycle(x0, os, m_options);
    }

private:
    GradientDescent<dim> m_solver;
    DescentOptions m_options;
};


template <int dim>
struct ExperimentMultiLevel
{
    ExperimentMultiLevel(ContextMultiLevel<dim> ctx,
                         DescentOptions options,
                         DescentOptions options_coarse)
        : model(ctx.O_tilt_coarse, ctx.O_tilt, ctx.O_coarse, ctx.point_transfer, ctx.vector_transport)
        , solver(ctx.O_descent, model)
        , m_options(options)
        , m_options_coarse(options_coarse)
    {}

    void cycle(const Vector<double>& x0, std::ostream& os,
               double kappa, double eps, unsigned coarse_every)
    {
        solver.cycle(x0, os, m_options, m_options_coarse, kappa, eps, coarse_every);
    }

private:
    CoarseModel<dim> model;
    FullApproximationScheme<dim> solver;
    DescentOptions m_options;
    DescentOptions m_options_coarse;
};

void cat_file(std::string filename)
{
    std::ifstream inFile(filename);

    if (inFile) {
        // Use rdbuf() to send the file buffer directly to cout
        std::cout << inFile.rdbuf();
    } else {
        std::cerr << "Error: Could not open file for reading.\n";
    }
}

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

            // Starting values (non-zero) and cycle
            Vector<double> x0(GP_fine.n_dofs());
            x0 = 1.0;

            // Oracle for fine level descent
            EnergyOracle<dim> o_descent(problem_fine, options.beta, options_slv);

            // Oracle for correction term
            std::unique_ptr<GrossPitaevskiiOracle<dim>>       o_tilt         = nullptr;
            std::unique_ptr<GrossPitaevskiiOracle<dim>>       o_tilt_coarse  = nullptr;
            std::unique_ptr<CoarseOracleBase<dim>> o_coarse_model = nullptr;

            if (options_fas.metric_t == MetricKind::NONE) {
                ExperimentSingleLevel<dim>(o_descent, options_gd).cycle(x0, std::cout);
                return;
            }
            if (options_fas.metric_t == MetricKind::MASS) {
                o_tilt         = std::make_unique<MassOracle<dim>>(problem_fine, options.beta, options_slv);
                o_tilt_coarse  = std::make_unique<MassOracle<dim>>(problem_coarse, options.beta, options_slv_coarse);
                o_coarse_model = std::make_unique<MassCoarseOracleEnergyAdaptive<dim>>(problem_coarse, options.beta, options_slv_coarse);
            }
            else if (options_fas.metric_t == MetricKind::FROBENIUS) {
                o_tilt         = std::make_unique<FrobeniusOracle<dim>>(problem_fine, options.beta, options_slv);
                o_tilt_coarse  = std::make_unique<FrobeniusOracle<dim>>(problem_coarse, options.beta, options_slv_coarse);
                o_coarse_model = std::make_unique<FrobeniusCoarseOracleEnergyAdaptive<dim>>(problem_coarse, options.beta, options_slv_coarse);
            } else {
                throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
            }

            // Manifold transfer depending on Galerkin condition
            std::unique_ptr<LinearTransferBase>   transfer       = nullptr;
            std::unique_ptr<ManifoldTransferBase> point_transfer = nullptr;

            if (options_fas.interpol_t == Interpolate::NONE) {
                transfer       = std::make_unique<LinearTransferMatrix<dim>>(dofs_coarse, dofs_fine, constraints_coarse, constraints_fine);
                point_transfer = std::make_unique<ManifoldTransfer<OperatorType>>(*transfer, M_coarse, M_fine);
            }
            else if (options_fas.interpol_t == Interpolate::MASS) {
                transfer       = std::make_unique<MassTransfer<dim,LinearTransferMatrix<dim>,OperatorType,InverseOpType>>(
                dofs_coarse, dofs_fine, constraints_coarse, constraints_fine, M_fine, M_inv_coarse);
                point_transfer = std::make_unique<ManifoldTransfer<OperatorType>>(*transfer, M_coarse, M_fine);
            }
            else {
              throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
            }

            // Vector transport operators
            std::unique_ptr<VectorTransportBase> vector_transport = nullptr;

            if (options_fas.transport_t == Transport::DIFFERENTIAL) {
                vector_transport = std::make_unique<DifferentialTransport<OperatorType>>(*point_transfer, M_coarse, M_fine);
            }
            else if (options_fas.transport_t == Transport::MASS) {
                vector_transport = std::make_unique<MassProjectionTransport<OperatorType>>(*transfer, M_coarse, M_fine);
            }
            else if (options_fas.transport_t == Transport::FROBENIUS) {
                vector_transport = std::make_unique<FrobeniusProjectionTransport<OperatorType>>(*transfer, M_coarse, M_fine);
            }
            else if (options_fas.transport_t == Transport::ADJOINT_RESTRICTION) {
                vector_transport = std::make_unique<AdjointRestrictionTransport<OperatorType,InverseOpType>>(
                    *transfer, M_coarse, M_fine, M_inv_coarse);
            }
            else if (options_fas.transport_t == Transport::ADJOINT_RESTRICTION_SCALED) {
                vector_transport = std::make_unique<AdjointRestrictionTransportScaled<OperatorType,InverseOpType>>(
                    *transfer, M_coarse, M_fine, M_inv_coarse);
            }
            else {
                throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
            }

            ContextMultiLevel<dim> context(
                o_descent,           // Fine descent oracle (stack-allocated)
                *o_tilt,             // Fine tilt oracle (heap-allocated)
                *o_tilt_coarse,      // Coarse tilt oracle (heap-allocated)
                *o_coarse_model,     // Coarse model qk (heap-allocated)
                *transfer,           // Linear transfer
                *point_transfer,     // Point transfer
                *vector_transport    // Vector transport
            );
            ExperimentMultiLevel<dim>(context, options_gd, options_gd_coarse).cycle(x0, std::cout,
                options_fas.kappa, options_fas.eps, options_fas.coarse_every);
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

