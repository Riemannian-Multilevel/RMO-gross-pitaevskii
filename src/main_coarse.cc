#include <gpe/lac.h>
#include <gpe/main/model.h>
#include <gpe/ropt/fas.h>
#include <gpe/problem/oracle_coarse.h>
#include <gpe/problem/gpe.h>
#include <gpe/option.h>
#include <gpe/ropt/manifold.h>

#include <memory>
#include <tuple>

using namespace dealii;
using namespace gpe;

// -------------------------------------------------------------------------
// Single Level Experiment
// -------------------------------------------------------------------------
template <int dim, typename DescentOracleType>
struct ExperimentSingleLevel
{
    ExperimentSingleLevel(DescentOracleType& O_descent, const ManifoldBase& manifold, DescentOptions options)
        : m_solver(O_descent, manifold, options), m_options(options)
    {}

    void cycle(Vector<double>& x0, std::ostream& os)
    {
        m_solver.cycle(x0, os);
    }

private:
    GradientDescent<dim> m_solver;
    DescentOptions m_options;
};

// -------------------------------------------------------------------------
// Template Dispatcher for 3-Level Experiments
// -------------------------------------------------------------------------
template <int dim, typename DescentOracleType, typename TiltType, template<int, typename> class CoarseModelType>
void run_3level_experiment(DescentOracleType& o_descent_2,
                           TiltType& o_tilt_2, TiltType& o_tilt_1, TiltType& o_tilt_0,
                           const ManifoldTransferBase& pt_21, const VectorTransportBase& vt_21,
                           const ManifoldTransferBase& pt_10, const VectorTransportBase& vt_10,
                           const ManifoldBase& manifold_2, const ManifoldBase& manifold_1, const ManifoldBase& manifold_0,
                           SolverOptions options_slv_coarse,
                           DescentOptions options_gd, DescentOptions options_gd_coarse, FAS_Options options_fas,
                           Vector<double>& x0)
{
    // Build the FAS State Managers (Bridges 2->1 and 1->0)
    using FASManagerType = CoarseOracleBase<dim, TiltType, TiltType>;
    FASManagerType fas_21(o_tilt_2, o_tilt_1, pt_21, vt_21);
    FASManagerType fas_10(o_tilt_1, o_tilt_0, pt_10, vt_10);

    // Build the Surrogate Models (q_k)
    CoarseModelType<dim, TiltType> q_2(fas_21, options_slv_coarse);
    CoarseModelType<dim, TiltType> q_1(fas_10, options_slv_coarse);

    // Level 0: Gradient Descent (minimizing q_1)
    GradientDescent<dim> solver_0(q_1, manifold_0, options_gd_coarse);

    // Level 1: FAS (minimizing q_2, using solver_0 for coarse steps)
    FullApproximationScheme<dim, CoarseModelType<dim, TiltType>, FASManagerType, CoarseModelType<dim, TiltType>>
        solver_1(q_2, fas_10, q_1, manifold_1, manifold_0, vt_10, solver_0, options_gd_coarse, options_fas);

    // Level 2: FAS (minimizing true objective, using solver_1 for coarse steps)
    FullApproximationScheme<dim, DescentOracleType, FASManagerType, CoarseModelType<dim, TiltType>>
        solver_2(o_descent_2, fas_21, q_2, manifold_2, manifold_1, vt_21, solver_1, options_gd, options_fas);

    // Run the Top-Level Cycle
    solver_2.cycle(x0, std::cout);
}


template <int dim, typename DescentOracleType, typename TiltType, template<int, typename> class CoarseModelType>
void run_2level_experiment(DescentOracleType& o_descent_1,
                           TiltType& o_tilt_1, TiltType& o_tilt_0,
                           const ManifoldTransferBase& pt_10, const VectorTransportBase& vt_10,
                           const ManifoldBase& manifold_1, const ManifoldBase& manifold_0,
                           SolverOptions options_slv_coarse,
                           DescentOptions options_gd, DescentOptions options_gd_coarse, FAS_Options options_fas,
                           Vector<double>& x0)
{
    // Build the FAS State Managers (Bridges 2->1 and 1->0)
    using FASManagerType = CoarseOracleBase<dim, TiltType, TiltType>;
    FASManagerType fas_10(o_tilt_1, o_tilt_0, pt_10, vt_10);

    // Build the Surrogate Models (q_k)
    CoarseModelType<dim, TiltType> q_1(fas_10, options_slv_coarse);

    // Level 0: Gradient Descent (minimizing q_1)
    GradientDescent<dim> solver_0(q_1, manifold_0, options_gd_coarse);

    // Level 1: FAS (minimizing q_2, using solver_0 for coarse steps)
    FullApproximationScheme<dim, DescentOracleType, FASManagerType, CoarseModelType<dim, TiltType>>
        solver_1(o_descent_1, fas_10, q_1, manifold_1, manifold_0, vt_10, solver_0, options_gd, options_fas);

    // Run the Top-Level Cycle
    solver_1.cycle(x0, std::cout);
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

            // Define the 3-Level Hierarchy
            const int n_level_2 = options_mg.max_level;
            const int n_level_1 = n_level_2 - 1;
            const int n_level_0 = n_level_2 - 2;

            AssertThrow(n_level_0 >= 0, dealii::ExcMessage("max_level must be at least 2 for a 3-level setup."));

            // Setup physics contexts
            ModelBuilder<dim> builder_0(V, options, n_level_0);
            ModelBuilder<dim> builder_1(V, options, n_level_1);
            ModelBuilder<dim> builder_2(V, options, n_level_2);

            // 1. Build the Functionals
            auto obj_0 = builder_0.get_eval(options.beta);
            auto obj_1 = builder_1.get_eval(options.beta);
            auto obj_2 = builder_2.get_eval(options.beta);

            // 2. Extract OperatorTypes
            const auto& M_0 = obj_0.get_M();
            const auto& M_1 = obj_1.get_M();
            const auto& M_2 = obj_2.get_M();

            auto M_inv_0 = InverseOpType(M_0, options_slv_coarse);
            auto M_inv_1 = InverseOpType(M_1, options_slv_coarse);
            auto M_inv_2 = InverseOpType(M_2, options_slv);

            const auto& dofs_0 = builder_0.get_package().get_dofs();
            const auto& dofs_1 = builder_1.get_package().get_dofs();
            const auto& dofs_2 = builder_2.get_package().get_dofs();

            const auto& constr_0 = builder_0.get_package().get_constraints();
            const auto& constr_1 = builder_1.get_package().get_constraints();
            const auto& constr_2 = builder_2.get_package().get_constraints();

            Vector<double> x0(builder_2.n_dofs());
            x0 = 1.0;
            builder_2.distribute(x0);

            // --- TRANSFER AND TRANSPORT OPERATORS FACTORY ---
            // TODO: take const& ModelBuilder as argument
            //       free function instead of lambda
            auto build_transfers = [&](const auto& dofs_c, const auto& dofs_f,
                                       const auto& constr_c, const auto& constr_f,
                                       const auto& M_c, const auto& M_f, const auto& M_inv_c)
            {
                std::shared_ptr<LinearTransferBase> transfer;
                std::shared_ptr<ManifoldTransferBase> point_transfer;
                std::shared_ptr<VectorTransportBase> vector_transport;

                if (options_fas.interpol_t == Interpolate::NONE) {
                    transfer = std::make_shared<LinearTransferMatrix<dim>>(dofs_c, dofs_f, constr_c, constr_f);
                    point_transfer = std::make_shared<ManifoldTransfer<OperatorType>>(*transfer, M_c, M_f);
                }
                else if (options_fas.interpol_t == Interpolate::MASS) {
                    transfer = std::make_shared<MassTransfer<dim,LinearTransferMatrix<dim>,OperatorType,InverseOpType>>(
                        dofs_c, dofs_f, constr_c, constr_f, M_f, M_inv_c);
                    point_transfer = std::make_shared<ManifoldTransfer<OperatorType>>(*transfer, M_c, M_f);
                }
                else {
                    throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
                }

                if (options_fas.transport_t == Transport::DIFFERENTIAL) {
                    vector_transport = std::make_shared<DifferentialTransport<OperatorType>>(*point_transfer, M_c, M_f);
                }
                else if (options_fas.transport_t == Transport::MASS) {
                    vector_transport = std::make_shared<MassProjectionTransport<OperatorType>>(*transfer, M_c, M_f);
                }
                else if (options_fas.transport_t == Transport::FROBENIUS) {
                    vector_transport = std::make_shared<FrobeniusProjectionTransport<OperatorType>>(*transfer, M_c, M_f);
                }
                else if (options_fas.transport_t == Transport::ADJOINT_RESTRICTION) {
                    vector_transport = std::make_shared<AdjointRestrictionTransport<OperatorType,InverseOpType>>(
                        *transfer, M_c, M_f, M_inv_c);
                }
                else if (options_fas.transport_t == Transport::ADJOINT_RESTRICTION_SCALED) {
                    vector_transport = std::make_shared<AdjointRestrictionTransportScaled<OperatorType,InverseOpType>>(
                        *transfer, M_c, M_f, M_inv_c);
                }
                else {
                    throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
                }

                return std::make_tuple(transfer, point_transfer, vector_transport);
            };

            auto [transfer_10, pt_10, vt_10] = build_transfers(dofs_0, dofs_1, constr_0, constr_1, M_0, M_1, M_inv_0);
            auto [transfer_21, pt_21, vt_21] = build_transfers(dofs_1, dofs_2, constr_1, constr_2, M_1, M_2, M_inv_1);

            // --- ORACLE AND SOLVER DISPATCH ---
            EnergyOracle<dim> o_descent_2(obj_2, options_slv);

            // Build Manifolds
            UnitMassSphere<dim, OperatorType> manifold_0(M_0);
            UnitMassSphere<dim, OperatorType> manifold_1(M_1);
            UnitMassSphere<dim, OperatorType> manifold_2(M_2);

            if (options_fas.metric_t == MetricKind::NONE) {
                ExperimentSingleLevel<dim, EnergyOracle<dim>>(o_descent_2, manifold_2, options_gd).cycle(x0, std::cout);
            }
            else if (options_fas.metric_t == MetricKind::MASS) {
                MassOracle<dim> o_tilt_0(obj_0, options_slv_coarse);
                MassOracle<dim> o_tilt_1(obj_1, options_slv_coarse);
                MassOracle<dim> o_tilt_2(obj_2, options_slv);

                run_3level_experiment<dim, EnergyOracle<dim>, MassOracle<dim>, MassCoarseOracleEnergyAdaptive>(
                    o_descent_2, o_tilt_2, o_tilt_1, o_tilt_0,
                    *pt_21, *vt_21, *pt_10, *vt_10,
                    manifold_2, manifold_1, manifold_0,
                    options_slv_coarse, options_gd, options_gd_coarse, options_fas, x0);

                // run_2level_experiment<dim, EnergyOracle<dim>, MassOracle<dim>, MassCoarseOracleEnergyAdaptive>(
                //     o_descent_2, o_tilt_2, o_tilt_1,
                //         *pt_21, *vt_21,
                //         manifold_2, manifold_1,
                //         options_slv_coarse, options_gd, options_gd_coarse, options_fas, x0);
            }
            else if (options_fas.metric_t == MetricKind::FROBENIUS) {
                FrobeniusOracle<dim> o_tilt_0(obj_0);
                FrobeniusOracle<dim> o_tilt_1(obj_1);
                FrobeniusOracle<dim> o_tilt_2(obj_2);

                run_3level_experiment<dim, EnergyOracle<dim>, FrobeniusOracle<dim>, FrobeniusCoarseOracleEnergyAdaptive>(
                    o_descent_2, o_tilt_2, o_tilt_1, o_tilt_0,
                    *pt_21, *vt_21, *pt_10, *vt_10,
                    manifold_2, manifold_1, manifold_0,
                    options_slv_coarse, options_gd, options_gd_coarse, options_fas, x0);

                // run_2level_experiment<dim, EnergyOracle<dim>, FrobeniusOracle<dim>, FrobeniusCoarseOracleEnergyAdaptive>(
                //     o_descent_2, o_tilt_2, o_tilt_1,
                //     *pt_21, *vt_21,
                //     manifold_2, manifold_1,
                //     options_slv_coarse, options_gd, options_gd_coarse, options_fas, x0);
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