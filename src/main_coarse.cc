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

    void run(Vector<double>& x0, std::ostream& os)
    {
        m_solver.cycle(x0, os);
    }

private:
    GradientDescent<dim> m_solver;
    DescentOptions m_options;
};


void cat_file(std::string filename)
{
    std::ifstream inFile(filename);
    if (inFile) {
        std::cout << inFile.rdbuf();
    } else {
        std::cerr << "Error: Could not open file for reading.\n";
    }
}


template <int dim>
auto build_transfers(const DoFHandler<dim>& dofs_c,
                     const DoFHandler<dim>& dofs_f,
                     const AffineConstraints<double>& constr_c,
                     const AffineConstraints<double>& constr_f,
                     const OperatorType& M_c, const OperatorType& M_f, const InverseOpType& M_inv_c,
                     FAS_Options options_fas)
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


template <int dim>
class Experiment2Level
{
public:
    template <typename Potential>
    Experiment2Level(Potential&& V, unsigned n_levels,
                     GPE_Options options, FAS_Options options_fas,
                     SolverOptions options_slv_f, SolverOptions options_slv_c)
        : n_levels(n_levels)
        , options(options), options_fas(options_fas), options_slv_c(options_slv_c), options_slv_f(options_slv_f)
        , builder_c(V, options, n_levels-1)
        , builder_f(V, options, n_levels)
        // FIXED: Added options.beta to get_eval()
        , obj_c(builder_c.get_eval(options.beta))
        , obj_f(builder_f.get_eval(options.beta))
        , M_c(obj_c.get_M())
        , M_f(obj_f.get_M())
        , M_inv_c(obj_c.get_M(), options_slv_c)
        , M_inv_f(obj_f.get_M(), options_slv_f)
        , manifold_c(M_c)
        , manifold_f(M_f)
    {
        const auto& dofs_c = builder_c.get_package().get_dofs();
        const auto& dofs_f = builder_f.get_package().get_dofs();

        const auto& constr_c = builder_c.get_package().get_constraints();
        const auto& constr_f = builder_f.get_package().get_constraints();

        std::tie(transfer, point_transfer, vector_transport) = build_transfers(dofs_c, dofs_f, constr_c, constr_f,
            M_c, M_f, M_inv_c, options_fas);
    }

    void distribute(Vector<double>& x)
    {
        builder_f.distribute(x);
    }

    unsigned n_dofs() const
    {
        return builder_f.n_dofs();
    }

    void run(Vector<double>& x0, DescentOptions options_gd_f, DescentOptions options_gd_c, std::ostream& os)
    {
        EnergyOracle<dim> o_descent_f(obj_f, options_slv_f);

        if (options_fas.metric_t == MetricKind::NONE) {
            // FIXED: Passed os instead of std::cout
            ExperimentSingleLevel<dim, EnergyOracle<dim>>(o_descent_f, manifold_f, options_gd_f).cycle(x0, os);
        }
        else if (options_fas.metric_t == MetricKind::MASS) {
            MassOracle<dim> o_tilt_c(obj_c, options_slv_c);
            MassOracle<dim> o_tilt_f(obj_f, options_slv_f);

            // FIXED: Call matches the run_impl signature exactly
            run_impl<EnergyOracle<dim>, MassOracle<dim>, MassCoarseOracleEnergyAdaptive>(
                x0, o_descent_f, o_tilt_f, o_tilt_c, options_gd_f, options_gd_c, os);
        }
        else if (options_fas.metric_t == MetricKind::FROBENIUS) {
            FrobeniusOracle<dim> o_tilt_c(obj_c);
            FrobeniusOracle<dim> o_tilt_f(obj_f);

            // FIXED: Call matches the run_impl signature exactly
            run_impl<EnergyOracle<dim>, FrobeniusOracle<dim>, FrobeniusCoarseOracleEnergyAdaptive>(
                x0, o_descent_f, o_tilt_f, o_tilt_c, options_gd_f, options_gd_c, os);
        }
    }

protected:
    template <typename DescentOracleType, typename TiltOracleType, template<int, typename> class CoarseModelType>
    void run_impl(Vector<double>& x0, DescentOracleType& o_descent_f,
                  TiltOracleType& o_tilt_f, TiltOracleType& o_tilt_c,
                  DescentOptions options_gd_f, DescentOptions options_gd_c, std::ostream& os)
    {
        // FAS orchestration (bridges 1->0)
        using FASManagerType = CoarseOracleBase<dim, TiltOracleType, TiltOracleType>;
        // FIXED: Dereferenced point_transfer and vector_transport
        FASManagerType fas(o_tilt_f, o_tilt_c, manifold_c, *point_transfer, *vector_transport);

        // Define the coarse models (q_k)
        CoarseModelType<dim, TiltOracleType> q_1(fas, options_slv_c);

        // Level 0: Gradient Descent (minimizing q_1)
        GradientDescent<dim> solver_c(q_1, manifold_c, options_gd_c);

        // Level 1: FAS (minimizing q_2, using solver_c for coarse steps)
        // FIXED: Dereferenced vector_transport
        FullApproximationScheme<dim, DescentOracleType, FASManagerType, CoarseModelType<dim, TiltOracleType>>
            solver_f(o_descent_f, fas, q_1, manifold_f, manifold_c, *vector_transport, solver_c, options_gd_f, options_fas);

        // Run the Top-Level Cycle
        solver_f.cycle(x0, os);
    }

private:
    unsigned n_levels;
    GPE_Options   options;
    FAS_Options   options_fas;
    SolverOptions options_slv_c;
    SolverOptions options_slv_f;

    ModelBuilder<dim> builder_c;  // coarse 0
    ModelBuilder<dim> builder_f;  // fine 1

    GrossPitaevskiiFunctional<dim> obj_c, obj_f;
    const OperatorType &M_c, &M_f;
    InverseOpType M_inv_c, M_inv_f;

    UnitMassSphere<dim, OperatorType> manifold_c;
    UnitMassSphere<dim, OperatorType> manifold_f;

    // Variable grid operators
    std::shared_ptr<LinearTransferBase> transfer;
    std::shared_ptr<ManifoldTransferBase> point_transfer;
    std::shared_ptr<VectorTransportBase> vector_transport;
};


template <int dim>
class Experiment3Level
{
public:
    template <typename Potential>
    Experiment3Level(Potential&& V, unsigned n_levels,
                     GPE_Options options, FAS_Options options_fas,
                     SolverOptions options_slv_2, SolverOptions options_slv_1, SolverOptions options_slv_0)
        : n_levels(n_levels)
        , options(options), options_fas(options_fas)
        , options_slv_2(options_slv_2), options_slv_1(options_slv_1), options_slv_0(options_slv_0)
        , builder_0(V, options, n_levels-2)
        , builder_1(V, options, n_levels-1)
        , builder_2(V, options, n_levels)
        , obj_0(builder_0.get_eval(options.beta))
        , obj_1(builder_1.get_eval(options.beta))
        , obj_2(builder_2.get_eval(options.beta))
        , M_0(obj_0.get_M())
        , M_1(obj_1.get_M())
        , M_2(obj_2.get_M())
        , M_inv_0(obj_0.get_M(), options_slv_0)
        , M_inv_1(obj_1.get_M(), options_slv_1)
        , M_inv_2(obj_2.get_M(), options_slv_2)
        , manifold_0(M_0)
        , manifold_1(M_1)
        , manifold_2(M_2)
    {
        const auto& dofs_0 = builder_0.get_package().get_dofs();
        const auto& dofs_1 = builder_1.get_package().get_dofs();
        const auto& dofs_2 = builder_2.get_package().get_dofs();

        const auto& constr_0 = builder_0.get_package().get_constraints();
        const auto& constr_1 = builder_1.get_package().get_constraints();
        const auto& constr_2 = builder_2.get_package().get_constraints();

        std::tie(transfer_10, pt_10, vt_10) = build_transfers(dofs_0, dofs_1, constr_0, constr_1,
            M_0, M_1, M_inv_0, options_fas);

        std::tie(transfer_21, pt_21, vt_21) = build_transfers(dofs_1, dofs_2, constr_1, constr_2,
            M_1, M_2, M_inv_1, options_fas);
    }

    void distribute(Vector<double>& x)
    {
        builder_2.distribute(x);
    }

    unsigned n_dofs() const
    {
        return builder_2.n_dofs();
    }

    void run(Vector<double>& x0, DescentOptions options_gd_2, DescentOptions options_gd_1, DescentOptions options_gd_0, std::ostream& os)
    {
        EnergyOracle<dim> o_descent_2(obj_2, options_slv_2);

        if (options_fas.metric_t == MetricKind::NONE) {
            ExperimentSingleLevel<dim, EnergyOracle<dim>>(o_descent_2, manifold_2, options_gd_2).cycle(x0, os);
        }
        else if (options_fas.metric_t == MetricKind::MASS) {
            MassOracle<dim> o_tilt_0(obj_0, options_slv_0);
            MassOracle<dim> o_tilt_1(obj_1, options_slv_1);
            MassOracle<dim> o_tilt_2(obj_2, options_slv_2);

            run_impl<EnergyOracle<dim>, MassOracle<dim>, MassCoarseOracleEnergyAdaptive>(
                x0, o_descent_2, o_tilt_2, o_tilt_1, o_tilt_0, options_gd_2, options_gd_1, options_gd_0, os);
        }
        else if (options_fas.metric_t == MetricKind::FROBENIUS) {
            FrobeniusOracle<dim> o_tilt_0(obj_0);
            FrobeniusOracle<dim> o_tilt_1(obj_1);
            FrobeniusOracle<dim> o_tilt_2(obj_2);

            run_impl<EnergyOracle<dim>, FrobeniusOracle<dim>, FrobeniusCoarseOracleEnergyAdaptive>(
                x0, o_descent_2, o_tilt_2, o_tilt_1, o_tilt_0, options_gd_2, options_gd_1, options_gd_0, os);
        }
    }

protected:
    template <typename DescentOracleType, typename TiltOracleType, template<int, typename> class CoarseModelType>
    void run_impl(Vector<double>& x0, DescentOracleType& o_descent_2,
                  TiltOracleType& o_tilt_2, TiltOracleType& o_tilt_1, TiltOracleType& o_tilt_0,
                  DescentOptions options_gd_2, DescentOptions options_gd_1, DescentOptions options_gd_0, std::ostream& os)
    {
        // FAS orchestration managers (bridges 2->1 and 1->0)
        using FASManagerType = CoarseOracleBase<dim, TiltOracleType, TiltOracleType>;
        FASManagerType fas_21(o_tilt_2, o_tilt_1, manifold_1, *pt_21, *vt_21);
        FASManagerType fas_10(o_tilt_1, o_tilt_0, manifold_0, *pt_10, *vt_10);

        // Define the coarse surrogate models (q_k)
        CoarseModelType<dim, TiltOracleType> q_1(fas_21, options_slv_1);
        CoarseModelType<dim, TiltOracleType> q_0(fas_10, options_slv_0);

        // Level 0: Gradient Descent (minimizing q_0)
        GradientDescent<dim> solver_0(q_0, manifold_0, options_gd_0);

        // Level 1: FAS (minimizing q_1, using solver_0 for coarse steps)
        FullApproximationScheme<dim, CoarseModelType<dim, TiltOracleType>, FASManagerType, CoarseModelType<dim, TiltOracleType>>
            solver_1(q_1, fas_10, q_0, manifold_1, manifold_0, *vt_10, solver_0, options_gd_1, options_fas);

        // Level 2: FAS (minimizing true objective E_2, using solver_1 for coarse steps)
        FullApproximationScheme<dim, DescentOracleType, FASManagerType, CoarseModelType<dim, TiltOracleType>>
            solver_2(o_descent_2, fas_21, q_1, manifold_2, manifold_1, *vt_21, solver_1, options_gd_2, options_fas);

        // Run the Top-Level Cycle
        solver_2.cycle(x0, os);
    }

private:
    unsigned n_levels;
    GPE_Options   options;
    FAS_Options   options_fas;
    SolverOptions options_slv_2;
    SolverOptions options_slv_1;
    SolverOptions options_slv_0;

    ModelBuilder<dim> builder_0;
    ModelBuilder<dim> builder_1;
    ModelBuilder<dim> builder_2;

    GrossPitaevskiiFunctional<dim> obj_0, obj_1, obj_2;
    const OperatorType &M_0, &M_1, &M_2;
    InverseOpType M_inv_0, M_inv_1, M_inv_2;

    UnitMassSphere<dim, OperatorType> manifold_0;
    UnitMassSphere<dim, OperatorType> manifold_1;
    UnitMassSphere<dim, OperatorType> manifold_2;

    std::shared_ptr<LinearTransferBase> transfer_10, transfer_21;
    std::shared_ptr<ManifoldTransferBase> pt_10, pt_21;
    std::shared_ptr<VectorTransportBase> vt_10, vt_21;
};


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

            int n_levels = options_mg.max_level;
            const int levels_count = options_mg.max_level - options_mg.min_level + 1;

            if (levels_count == 3) {
                AssertThrow(n_levels >= 2, dealii::ExcMessage("max_level must be >= 2 for 3-level setup"));

                Experiment3Level<dim> exp(V, n_levels, options, options_fas, options_slv, options_slv_coarse, options_slv_coarse);

                Vector<double> x0(exp.n_dofs());
                x0 = 1.0;
                exp.distribute(x0);

                exp.run(x0, options_gd, options_gd_coarse, options_gd_coarse, std::cout);
            }
            else if (levels_count == 2) {
                AssertThrow(n_levels >= 1, dealii::ExcMessage("max_level must be >= 1 for 2-level setup"));

                Experiment2Level<dim> exp(V, n_levels, options, options_fas, options_slv, options_slv_coarse);

                Vector<double> x0(exp.n_dofs());
                x0 = 1.0;
                exp.distribute(x0);

                exp.run(x0, options_gd, options_gd_coarse, std::cout);
            }
            else {
                AssertThrow(false, dealii::ExcMessage("Unsupported level count (only 2 or 3 supported in this example)"));
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