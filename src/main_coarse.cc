#include <gpe/lac.h>
#include <gpe/main/model.h>
#include <gpe/main/fas.h>
#include <gpe/problem/oracle_coarse.h>
#include <gpe/option.h>
#include <gpe/ropt/manifold.h>

using namespace dealii;
using namespace gpe;

// -------------------------------------------------------------------------
// Transfer Setup Helper
// -------------------------------------------------------------------------
template <int dim>
auto build_transfers(const DoFHandler<dim>& dofs_c, const DoFHandler<dim>& dofs_f,
                     const AffineConstraints<double>& constr_c, const AffineConstraints<double>& constr_f,
                     const OperatorType& M_c, const OperatorType& M_f, const InverseOpType& M_inv_c,
                     FAS_Options options_fas)
{
    std::shared_ptr<LinearTransferBase> transfer;
    std::shared_ptr<ManifoldTransferBase> point_transfer;
    std::shared_ptr<VectorTransportBase> vector_transport;

    if (options_fas.interpol_t == Interpolate::MASS) {
        transfer = std::make_shared<MassTransfer<dim,LinearTransferMatrix<dim>,OperatorType,InverseOpType>>(
            dofs_c, dofs_f, constr_c, constr_f, M_f, M_inv_c);
    } else {
        transfer = std::make_shared<LinearTransferMatrix<dim>>(dofs_c, dofs_f, constr_c, constr_f);
    }

    point_transfer = std::make_shared<ManifoldTransfer<OperatorType>>(*transfer, M_c, M_f);

    // TODO: include other operators
    if (options_fas.transport_t == Transport::MASS) {
        vector_transport = std::make_shared<MassProjectionTransport<OperatorType>>(*transfer, M_c, M_f);
    } else {
        vector_transport = std::make_shared<DifferentialTransport<OperatorType>>(*point_transfer, M_c, M_f);
    }

    return std::make_tuple(transfer, point_transfer, vector_transport);
}

template <int dim>
class MultiLevelExperiment
{
public:
    template <typename Potential>
    MultiLevelExperiment(Potential&& V, unsigned min_level, unsigned max_level,
                         GPE_Options options, FAS_Options options_fas,
                         SolverOptions options_slv, DescentOptions options_gd)
        : min_level(min_level), max_level(max_level)
        , objective_mg        (min_level, max_level)
        , manifold_mg         (min_level, max_level)
        , transfer_mg         (min_level, max_level)
        , point_transfer_mg   (min_level, max_level)
        , vector_transport_mg (min_level, max_level)
        , options_descent_mg  (min_level, max_level)
        , options_solver_mg   (min_level, max_level)
    {
        // 1. Build Physics and Manifolds for all levels
        for (unsigned l = min_level; l <= max_level; ++l) {
            builders.emplace_back(std::make_unique<ModelBuilder<dim>>(V, options, l));

            options_descent_mg[l] = options_gd;
            options_solver_mg [l] = options_slv;

            // Use shared_ptr to safely store objects with reference members
            // (default copy assignment operator for MGLevelObject)
            objective_mg[l] = std::make_shared<GrossPitaevskiiFunctional<dim>>(
                builders.back()->get_system(), options.beta, options_solver_mg[l]
            );
            manifold_mg [l] = std::make_shared<UnitMassSphere<dim, OperatorType>>(
                objective_mg[l]->get_M()
            );

        }

        // 2. Configure looser bounds for coarse grids
        // TODO: configurable (option.h)
        for (unsigned l = min_level; l < max_level; ++l) {
            options_descent_mg[l].max_iter = 4;
            options_descent_mg[l].line_search = false;
        }

        // 3. Build Transfer/Transport operators bridging each consecutive level
        for (unsigned l = min_level + 1; l <= max_level; ++l) {
            const auto&   dofs_c= builders[l - min_level - 1]->get_package().get_dofs();
            const auto&   constr_c = builders[l - min_level - 1]->get_package().get_constraints();
            const auto&   M_c= objective_mg[l-1]->get_M();
            InverseOpType M_inv_c(M_c, options_solver_mg[l-1]);

            const auto& dofs_f = builders[l - min_level]->get_package().get_dofs();
            const auto& constr_f = builders[l - min_level]->get_package().get_constraints();
            const auto& M_f= objective_mg[l]->get_M();

            auto [t, pt, vt] = build_transfers(
                dofs_c, dofs_f, constr_c, constr_f, M_c, M_f, M_inv_c, options_fas);

            transfer_mg[l]         = t;
            point_transfer_mg[l]   = pt;
            vector_transport_mg[l] = vt;
        }

        fas_solver = std::make_unique<FullApproximationScheme<dim>>(
            manifold_mg, point_transfer_mg, vector_transport_mg,
            objective_mg, options_descent_mg, options_solver_mg, options_fas
        );
    }

    unsigned n_dofs() const { return builders.back()->n_dofs(); }
    void distribute(Vector<double>& x) const { builders.back()->distribute(x); }

    void run(Vector<double>& x0, std::ostream& os)
    {
        // Execute the cycle on the finest level
        // (De-referencing objective_mg since it's a shared_ptr)
        EnergyOracle<dim> O_fine(*objective_mg[max_level], options_solver_mg[max_level]);

        fas_solver->template cycle<MassOracle<dim>, MassCoarseOracleEnergyAdaptive<dim>>(
            O_fine, x0, max_level, os
        );
    }

private:
    unsigned min_level, max_level;
    std::vector<std::unique_ptr<ModelBuilder<dim>>> builders;

    MGLevelObject<std::shared_ptr<GrossPitaevskiiFunctional<dim>>> objective_mg;
    MGLevelObject<std::shared_ptr<ManifoldBase>>                   manifold_mg;
    MGLevelObject<std::shared_ptr<LinearTransferBase>>             transfer_mg;
    MGLevelObject<std::shared_ptr<ManifoldTransferBase>>           point_transfer_mg;
    MGLevelObject<std::shared_ptr<VectorTransportBase>>            vector_transport_mg;
    MGLevelObject<DescentOptions>                                  options_descent_mg;
    MGLevelObject<SolverOptions>                                   options_solver_mg;

    std::unique_ptr<FullApproximationScheme<dim>> fas_solver;
};

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
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
            Square<dim> V;

            int n_levels = options_mg.max_level;

            MultiLevelExperiment<dim> exp(V, n_levels-1, n_levels, options, options_fas, options_slv, options_gd);

            Vector<double> x0(exp.n_dofs());
            x0 = 1.0;
            exp.distribute(x0);

            exp.run(x0, std::cout);
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