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
    }
    else if (options_fas.interpol_t == Interpolate::NONE) {
        transfer = std::make_shared<LinearTransferMatrix<dim>>(dofs_c, dofs_f, constr_c, constr_f);
    }
    else {
        std::abort();
    }

    point_transfer = std::make_shared<ManifoldTransfer<OperatorType>>(*transfer, M_c, M_f);

    // TODO: include other operators
    if (options_fas.transport_t == Transport::FROBENIUS) {
        vector_transport = std::make_shared<FrobeniusProjectionTransport<OperatorType>>(*transfer, M_c, M_f);
    }
    else if (options_fas.transport_t == Transport::MASS) {
        vector_transport = std::make_shared<MassProjectionTransport<OperatorType>>(*transfer, M_c, M_f);
    }
    else if (options_fas.transport_t == Transport::DIFFERENTIAL) {
        vector_transport = std::make_shared<DifferentialTransport<OperatorType>>(*point_transfer, M_c, M_f);
    }
    else if (options_fas.transport_t == Transport::ADJOINT_RESTRICTION) {
        vector_transport = std::make_shared<AdjointRestrictionTransport<OperatorType, InverseOpType>>(*transfer, M_c, M_f, M_inv_c);
    }
    else if (options_fas.transport_t == Transport::ADJOINT_DIFFERENTIAL) {
        vector_transport = std::make_shared<AdjointDifferentialTransport<OperatorType, InverseOpType>>(*transfer, *point_transfer, M_c, M_f, M_inv_c);
    }
    else {
        std::abort();
    }

    return std::make_tuple(transfer, point_transfer, vector_transport);
}


// Reference problem using Riemannian gradient descent
template <int dim>
class SingleLevelExperiment
{
public:
    template <typename Potential>
    SingleLevelExperiment(Potential&& V, unsigned level,
                          GPE_Options options,
                          SolverOptions options_slv,
                          DescentOptions options_gd)
        : builder(std::make_unique<ModelBuilder<dim>>(V, options, level))
        , options_slv(options_slv)
        , options_gd(options_gd)
    {
        // 1. Build Physics and Manifold for the single level
        objective = std::make_shared<GrossPitaevskiiFunctional<dim>>(
            builder->get_system(), options.beta, options_slv
        );
        manifold = std::make_shared<UnitMassSphere<dim, OperatorType>>(
            objective->get_M()
        );
    }

    unsigned n_dofs() const { return builder->n_dofs(); }
    void distribute(Vector<double>& x) const { builder->distribute(x); }

    void run(Vector<double>& x0, MetricKind metric_t, std::ostream& os)
    {
        std::unique_ptr<OracleBase> oracle;

        // 2. Instantiate the corresponding descent oracle
        if (metric_t == MetricKind::FROBENIUS) {
            oracle = std::make_unique<FrobeniusOracle<dim>>(*objective, options_slv);
        }
        else if (metric_t == MetricKind::MASS) {
            oracle = std::make_unique<MassOracle<dim>>(*objective, options_slv);
        }
        else if (metric_t == MetricKind::ENERGY_ADAPTIVE) {
            oracle = std::make_unique<EnergyOracle<dim>>(*objective, options_slv);
        }
        else {
            std::abort();
        }

        // 3. Execute the single-level gradient descent cycle
        GradientDescent<dim> solver(*oracle, *manifold, options_gd);
        solver.cycle(x0, os);
    }

private:
    std::unique_ptr<ModelBuilder<dim>>              builder;
    std::shared_ptr<GrossPitaevskiiFunctional<dim>> objective;
    std::shared_ptr<ManifoldBase>                   manifold;

    SolverOptions  options_slv;
    DescentOptions options_gd;
};


template <int dim>
class MultiLevelExperiment
{
public:
    template <typename Potential>
    MultiLevelExperiment(Potential&& V, const std::vector<unsigned> &levels,
                         GPE_Options options, FAS_Options options_fas,
                         SolverOptions options_slv, DescentOptions options_gd)
    // The level vector approach assumes that
    // 1. not every level may be populated in the multilevel hierarchy
    // 2. the exact level is required, since meshes are defined by number of refinements
    // 3. levels are processed from fine (N) to coarse (n), N > n refinements
        : m_levels(levels)
    // TODO: set min_level, max_level from m_levels in constructor body
        , min_level(*std::ranges::min_element(levels))
        , max_level(*std::ranges::max_element(levels))
        , builders_mg         (min_level, max_level)
        , objective_mg        (min_level, max_level)
        , manifold_mg         (min_level, max_level)
        , transfer_mg         (min_level, max_level)
        , point_transfer_mg   (min_level, max_level)
        , vector_transport_mg (min_level, max_level)
        , options_descent_mg  (min_level, max_level)
        , options_solver_mg   (min_level, max_level)
    {
        // TODO: duplicate effort with min_element, max_element
        // sort in ascending order, coarse -> fine
        std::ranges::sort(m_levels, std::ranges::less{});

        // TODO: use AssertThrow or always use the unique_copy
        Assert(std::ranges::adjacent_find(m_levels) == m_levels.end(),
            dealii::ExcInternalError("level indices are not unique"));

        // 1. Build Physics and Manifolds for all levels
        for (auto l: m_levels) {
            builders_mg[l] = std::make_unique<ModelBuilder<dim>>(V, options, l);

            options_descent_mg[l] = options_gd;
            options_solver_mg [l] = options_slv;

            // Use shared_ptr to safely store objects with reference members
            // (default copy assignment operator for MGLevelObject)
            objective_mg[l] = std::make_shared<GrossPitaevskiiFunctional<dim>>(
                builders_mg[l]->get_system(), options.beta, options_solver_mg[l]
            );
            manifold_mg[l] = std::make_shared<UnitMassSphere<dim, OperatorType>>(
                objective_mg[l]->get_M()
            );
        }

        // 2. Configure looser bounds for coarse grids
        // TODO: configurable (option.h)
        for (auto l: m_levels) {
            if (l == max_level) {  // use global options for finest level
                continue;
            }
            options_descent_mg[l].max_iter = 4;
            //options_descent_mg[l].line_search = false;
            // TODO: set options_solver_mg[], options_descent_mg[] per level
        }

        // 3. Build Transfer/Transport operators bridging each consecutive level
        // TODO: use set indices to iterate over consecutive levels
        //for (auto l: level_set) {
        for (unsigned i = 1; i < m_levels.size(); i++) {  // assumes (strictly) ascending sort order on levels()

            unsigned l   = m_levels.at(i);
            unsigned l_c = m_levels.at(i - 1);  // next coarsest level
            Assert(l > min_level, dealii::ExcMessage("min_level found in position > 0"));

            const auto& dofs_c= builders_mg[l_c]->get_package().get_dofs();
            const auto& constr_c = builders_mg[l_c]->get_package().get_constraints();
            const auto& M_c= objective_mg[l_c]->get_M();
            InverseOpType& M_inv_c = objective_mg[l_c]->get_M_inv();

            const auto& dofs_f = builders_mg[l]->get_package().get_dofs();
            const auto& constr_f = builders_mg[l]->get_package().get_constraints();
            const auto& M_f = objective_mg[l]->get_M();

            auto [t, pt, vt] = build_transfers(
                dofs_c, dofs_f, constr_c, constr_f, M_c, M_f, M_inv_c, options_fas);

            transfer_mg[l]         = t;
            point_transfer_mg[l]   = pt;
            vector_transport_mg[l] = vt;
        }

        fas_solver = std::make_unique<FullApproximationScheme<dim>>(
            manifold_mg, point_transfer_mg, vector_transport_mg, objective_mg, m_levels,
            options_descent_mg, options_solver_mg, options_fas
        );
    }

    unsigned n_dofs() const { return builders_mg[max_level]->n_dofs(); }
    void distribute(Vector<double>& x) const { builders_mg[max_level]->distribute(x); }

    void run(Vector<double>& x0, MetricKind metric_t, std::ostream& os)
    {
        // Execute the cycle on the finest level
        EnergyOracle<dim> O_fine (*objective_mg[max_level], options_solver_mg[max_level]);
        MassOracle<dim>   T_fine (*objective_mg[max_level], options_solver_mg[max_level]);

        if (metric_t == MetricKind::FROBENIUS) {
            fas_solver->template cycle<FrobeniusOracle<dim>, FrobeniusCoarseOracle<dim>, FrobeniusCoarseOracleEnergyAdaptive<dim>>(
                O_fine, T_fine, x0, m_levels.size() - 1, os
            );
        }
        else if (metric_t == MetricKind::MASS) {
            fas_solver->template cycle<MassOracle<dim>, MassCoarseOracle<dim>, MassCoarseOracleEnergyAdaptive<dim>>(
                O_fine, T_fine, x0, m_levels.size() - 1, os
            );
        }
        else if (metric_t == MetricKind::ENERGY_ADAPTIVE) {
            throw std::invalid_argument("metric not supported for coarse model");
        }
        else {
            std::abort();
        }
    }

    void log(std::ostream& os) const
    {
        auto levels = fas_solver->cycle_log();
        os << "Total iterations: " << levels.size() << std::endl;

        if (levels.size()) {
            os << "[";
            for (auto it = levels.begin(); it != levels.end()-1; ++it) {
                os << *it << ",";
            }
            os << levels.back() << "]";
        }
    }

private:
    std::vector<unsigned> m_levels;
    unsigned min_level, max_level;
    MGLevelObject<std::unique_ptr<ModelBuilder<dim>>> builders_mg;

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
    GPE_Options    options    {};
    DescentOptions options_gd {};
    SolverOptions  options_slv{};
    MG_Options     options_mg {};
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
            const unsigned n_levels  = options_mg.n_levels;
            auto potential_v = potential::get_potential<dim>(options.potential);

            if (options_fas.metric_t == MetricKind::NONE || options_mg.v_levels.size() == 1) {
                // Run standard single-level Riemannian gradient descent on the finest level
                auto exp = std::visit([&](auto&& arg) {
                    return SingleLevelExperiment<dim>(arg, n_levels, options, options_slv, options_gd);
                }, potential_v);

                Vector<double> x0(exp.n_dofs());
                x0 = 1.0;
                exp.distribute(x0);

                exp.run(x0, MetricKind::ENERGY_ADAPTIVE, std::cout);
            }
            else {
                auto exp = std::visit([&](auto&& arg) {
                    return MultiLevelExperiment<dim>(arg, options_mg.v_levels, options, options_fas, options_slv, options_gd);
                }, potential_v);

                Vector<double> x0(exp.n_dofs());
                x0 = 1.0;
                exp.distribute(x0);

                exp.run(x0, options_fas.metric_t, std::cout);
                exp.log(std::cerr);
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