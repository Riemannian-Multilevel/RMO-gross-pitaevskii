#include <gpe/lac.h>
#include <gpe/main/model.h>
#include <gpe/ropt/fas.h>
#include <gpe/problem/oracle_coarse.h>
#include <gpe/problem/gpe.h>
#include <gpe/option.h>
#include <gpe/ropt/manifold.h> // Assuming this is where UnitMassSphere lives

using namespace dealii;
using namespace gpe;

// -------------------------------------------------------------------------
// Single Level Experiment
// -------------------------------------------------------------------------
template <int dim, typename DescentOracleType>
struct ExperimentSingleLevel
{
    ExperimentSingleLevel(DescentOracleType& O_descent, const ManifoldBase& manifold, DescentOptions options)
        : m_solver(O_descent, manifold), m_options(options)
    {}

    void cycle(Vector<double>& x0, std::ostream& os)
    {
        m_solver.cycle(x0, os, m_options);
    }

private:
    GradientDescent<dim> m_solver;
    DescentOptions m_options;
};

// -------------------------------------------------------------------------
// Template Dispatcher for Multilevel Experiments
// -------------------------------------------------------------------------
// This replaces ContextMultiLevel and directly wires the FAS architecture
template <int dim, typename DescentOracleType, typename TiltFineType, typename TiltCoarseType, template<int, typename> class CoarseModelType>
void run_multilevel_experiment(
    DescentOracleType& o_descent,
    TiltFineType& o_tilt_fine,
    TiltCoarseType& o_tilt_coarse,
    const ManifoldTransferBase& point_transfer,
    const VectorTransportBase& vector_transport,
    const ManifoldBase& fine_manifold,
    const ManifoldBase& coarse_manifold,
    SolverOptions options_slv_coarse,
    DescentOptions options_gd,
    DescentOptions options_gd_coarse,
    FAS_Options options_fas,
    Vector<double>& x0)
{
    // Build the FAS State Manager
    CoarseOracleBase<dim, TiltFineType, TiltCoarseType> fas(
        o_tilt_fine, o_tilt_coarse, point_transfer, vector_transport);

    // Build the Surrogate Model (q_k)
    CoarseModelType<dim, TiltFineType> q_k(fas, options_slv_coarse);

    // Use decltype(fas) to easily pass the complex FAS manager type to the scheme!
    FullApproximationScheme<dim, DescentOracleType, decltype(fas), CoarseModelType<dim, TiltFineType>> solver(
        o_descent, fas, q_k, fine_manifold, coarse_manifold, vector_transport);

    // Run the Cycle
    solver.cycle(x0, std::cout, options_gd, options_gd_coarse,
                 options_fas.kappa, options_fas.eps, options_fas.coarse_every);
}


void cat_file(std::string filename)
{
    std::ifstream inFile(filename);
    if (inFile) {
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

            DescentOptions options_gd_coarse = options_gd;
            options_gd_coarse.max_iter    = 4;
            options_gd_coarse.step_size   = 1.0;
            options_gd_coarse.line_search = false;

            Square<dim> V;

            const int min_i = options_mg.min_level;
            const int min_i_def = 8;
            const int max_i = options_mg.max_level;
            const int max_i_def = 9;
            const int levels_i = options_mg.n_levels;
            const int n_coarse_levels = (min_i == 0 ? min_i_def : min_i);
            const int n_fine_levels = (max_i == levels_i ? max_i_def : max_i);

            AssertThrow(n_fine_levels - n_coarse_levels == 1,
                dealii::ExcNotImplemented("only 2 levels are supported"));

            // Setup coarse and fine physics
            ModelBuilder<dim> builder_coarse(V, options, n_coarse_levels);
            ModelBuilder<dim> builder_fine(V, options, n_fine_levels);

            // 1. Build the Functionals (Bridges Physics to Oracles)
            auto obj_coarse = builder_coarse.get_eval(options.beta);
            auto obj_fine   = builder_fine.get_eval(options.beta);

            // 2. Extract OperatorTypes (LinearCombinations) from the Functionals
            const auto& M_coarse = obj_coarse.get_M();
            const auto& M_fine   = obj_fine.get_M();

            auto M_inv_coarse = InverseOpType(M_coarse, options_slv_coarse);
            auto M_inv_fine   = InverseOpType(M_fine, options_slv);

            const auto& dofs_coarse = builder_coarse.get_package().get_dofs();
            const auto& constraints_coarse = builder_coarse.get_package().get_constraints();
            const auto& dofs_fine = builder_fine.get_package().get_dofs();
            const auto& constraints_fine = builder_fine.get_package().get_constraints();

            Vector<double> x0(builder_fine.n_dofs());
            x0 = 1.0;
            builder_fine.distribute(x0);

            // --- TRANSFER AND TRANSPORT OPERATORS ---
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

            // --- ORACLE AND SOLVER DISPATCH ---
            // Pass the functional 'obj_fine', NOT the raw system 'problem_fine'
            EnergyOracle<dim> o_descent(obj_fine, options_slv);

            // Build BOTH Manifolds (require OperatorType in template signature)
            UnitMassSphere<dim, OperatorType> fine_manifold(M_fine);
            UnitMassSphere<dim, OperatorType> coarse_manifold(M_coarse);

            if (options_fas.metric_t == MetricKind::NONE) {
                ExperimentSingleLevel<dim, EnergyOracle<dim>>(o_descent, fine_manifold, options_gd).cycle(x0, std::cout);
            }
            else if (options_fas.metric_t == MetricKind::MASS) {
                // Pass the functionals
                MassOracle<dim> o_tilt_fine(obj_fine, options_slv);
                MassOracle<dim> o_tilt_coarse(obj_coarse, options_slv_coarse);

                run_multilevel_experiment<dim, EnergyOracle<dim>, MassOracle<dim>, MassOracle<dim>, MassCoarseOracleEnergyAdaptive>(
                    o_descent, o_tilt_fine, o_tilt_coarse,
                    *point_transfer, *vector_transport,
                    fine_manifold, coarse_manifold,
                    options_slv_coarse, options_gd, options_gd_coarse, options_fas, x0);
            }
            else if (options_fas.metric_t == MetricKind::FROBENIUS) {
                // FrobeniusOracle's constructor only takes the functional
                FrobeniusOracle<dim> o_tilt_fine(obj_fine);
                FrobeniusOracle<dim> o_tilt_coarse(obj_coarse);

                run_multilevel_experiment<dim, EnergyOracle<dim>, FrobeniusOracle<dim>, FrobeniusOracle<dim>, FrobeniusCoarseOracleEnergyAdaptive>(
                    o_descent, o_tilt_fine, o_tilt_coarse,
                    *point_transfer, *vector_transport,
                    fine_manifold, coarse_manifold,
                    options_slv_coarse, options_gd, options_gd_coarse, options_fas, x0);
            }
            else {
                throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
            }
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