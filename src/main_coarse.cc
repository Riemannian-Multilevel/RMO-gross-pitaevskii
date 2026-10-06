#include <rmo/lac.h>

#include <rmo/ropt/fas.h>
#include <rmo/ropt/observer_table.h>

#include <rmo/gpe/model.h>
#include <rmo/gpe/interpolate.h>
#include <rmo/gpe/oracle_coarse.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/transport.h>

#include <rmo/option.h>
#include <rmo/util/serialize.h>
#include <rmo/util/util.h>

#include <fmt/format.h>

using namespace dealii;
using namespace rmo;
using namespace rmo::gpe;

// -------------------------------------------------------------------------
// Transfer Setup Helper
// -------------------------------------------------------------------------
template <int dim, typename Operator, typename InverseM>
static auto build_transfers(const DoFHandler<dim>& dofs_c, const DoFHandler<dim>& dofs_f,
                            const AffineConstraints<double>& constr_c, const AffineConstraints<double>& constr_f,
                            const Operator& M_c, const Operator& M_f, const InverseM& M_inv_c,
                            const CoarseModelOptions& options_cm)
{
    std::shared_ptr<LinearTransferBase> transfer;

    if (options_cm.interpol_t == Interpolate::MASS) {
        transfer = std::make_shared<MassTransfer<dim,fe::LinearTransferMG<dim>,Operator,InverseM>>(
            dofs_c, dofs_f, constr_c, constr_f, M_f, M_inv_c);
    }
    else if (options_cm.interpol_t == Interpolate::NONE) {
        transfer = std::make_shared<fe::LinearTransferMG<dim>>(dofs_c, dofs_f, constr_c, constr_f);
    }
    else {
        std::abort();
    }

    std::shared_ptr<ManifoldTransferBase> point_transfer = std::make_shared<ManifoldTransfer<Operator>>(
        *transfer, M_c, M_f);
    std::shared_ptr<VectorTransportBase> vector_transport;

    // TODO: include other operators
    if (options_cm.transport_t == Transport::FROBENIUS) {
        vector_transport = std::make_shared<FrobeniusProjectionTransport<Operator>>(*transfer, M_c, M_f);
    }
    else if (options_cm.transport_t == Transport::MASS) {
        vector_transport = std::make_shared<MassProjectionTransport<Operator>>(*transfer, M_c, M_f);
    }
    else if (options_cm.transport_t == Transport::DIFFERENTIAL) {
        vector_transport = std::make_shared<DifferentialTransport<Operator>>(*point_transfer, M_c, M_f);
    }
    else if (options_cm.transport_t == Transport::ADJOINT_RESTRICTION) {
        vector_transport = std::make_shared<AdjointRestrictionTransport<Operator, InverseM>>(*transfer, M_c, M_f, M_inv_c);
    }
    else if (options_cm.transport_t == Transport::ADJOINT_DIFFERENTIAL) {
        vector_transport = std::make_shared<AdjointDifferentialTransport<Operator, InverseM>>(*transfer, *point_transfer, M_c, M_f, M_inv_c);
    }
    else if (options_cm.transport_t == Transport::ADJOINT_RESTRICTION_FROBENIUS) {
        vector_transport = std::make_shared<FrobeniusAdjointRestrictionTransport<Operator>>(*transfer, M_c, M_f);
    }
    else if (options_cm.transport_t == Transport::ADJOINT_DIFFERENTIAL_FROBENIUS) {
        vector_transport = std::make_shared<FrobeniusAdjointDifferentialTransport<Operator>>(*transfer, *point_transfer, M_c, M_f);
    }
    else if (options_cm.transport_t == Transport::DIFFERENTIAL_FROBENIUS) {
        vector_transport = std::make_shared<FrobeniusDifferentialTransport<Operator>>(*point_transfer, M_c, M_f);
    }
    else {
        std::abort();
    }

    return std::make_tuple(transfer, point_transfer, vector_transport);
}


// Metric for the coarse condition on one level, chosen by CoarseModelOptions::ccond_t
template <typename System>
static LevelMetric
build_cond_metric(const GrossPitaevskiiFunctional<System>& objective, MetricKind ccond_t)
{
    using OpMetric = OperatorMetric<typename GrossPitaevskiiFunctional<System>::Operator>;

    switch (ccond_t) {
        case MetricKind::MASS:
            return std::make_shared<OpMetric>(objective.get_M(), ccond_t);
        case MetricKind::ENERGY_ADAPTIVE:  // A(x) at the state the level functional was last updated to
            return std::make_shared<OpMetric>(objective.get_A(), ccond_t);
        case MetricKind::FROBENIUS:
            return std::make_shared<EuclideanMetric>();
        case MetricKind::NONE:             // use the metric of the coarse model (oracle metric)
            return nullptr;
        default:
            throw std::invalid_argument("unsupported metric for coarse condition");
    }
}


// Reference problem using Riemannian gradient descent
template <typename System>
class SingleLevelExperiment
{
public:
    static constexpr int dim = System::dimension;
    using Functional = GrossPitaevskiiFunctional<System>;
    using Operator   = typename Functional::Operator;

    template <typename Potential>
    SingleLevelExperiment(Potential&& V, unsigned level,
                          GPE_Options options,
                          SolverOptions options_slv,
                          DescentOptions options_gd)
        : builder(std::make_unique<ModelBuilder<System>>(V, options, level))
        , options_slv(options_slv)
        , options_gd(options_gd)
    {
        // 1. Build Physics and Manifold for the single level
        objective = std::make_shared<Functional>(
            builder->get_system(), options.beta, options_slv
        );
        manifold = std::make_shared<UnitMassSphere<Operator>>(
            objective->get_M()
        );
    }

    unsigned n_dofs() const { return builder->n_dofs(); }
    void distribute(Vector<double>& x) const { builder->distribute(x); }

    void run(Vector<double>& x0, MetricKind metric_t, std::ostream& os)
    {
        std::unique_ptr<GrossPitaevskiiOracle<System>> oracle;

        // 2. Instantiate the corresponding descent oracle
        if (metric_t == MetricKind::FROBENIUS) {
            oracle = std::make_unique<FrobeniusOracle<System>>(*objective, options_slv);
        }
        else if (metric_t == MetricKind::MASS) {
            oracle = std::make_unique<MassOracle<System>>(*objective, options_slv);
        }
        else if (metric_t == MetricKind::ENERGY_ADAPTIVE) {
            oracle = std::make_unique<EnergyOracle<System>>(*objective, options_slv);
        }
        else {
            std::abort();
        }

        // 3. Execute the single-level gradient descent cycle
        solver = std::make_unique<GradientDescent>(*oracle, oracle->get_residual(), *manifold, options_gd);
        solver->set_observer(conv_observer);
        solver->cycle(x0, os);
    }

    const auto& history() const { return solver->history(); }
    const typename System::MassMatrix& get_M() const { return builder->get_system().get_M(); }
    const auto& get_package() { return builder->get_package(); }

private:
    std::unique_ptr<ModelBuilder<System>>      builder;
    std::shared_ptr<Functional>                     objective;
    std::shared_ptr<ManifoldBase>                   manifold;

    SolverOptions  options_slv;
    DescentOptions options_gd;

    ConvergenceTableObserver<CycleInfo> conv_observer;
    std::unique_ptr<GradientDescent> solver;
};


template <typename System>
class MultiLevelExperiment
{
public:
    static constexpr int dim = System::dimension;
    using Functional = GrossPitaevskiiFunctional<System>;
    using Operator   = typename Functional::Operator;
    using InverseM   = typename Functional::InverseM;

    template <typename Potential>
    MultiLevelExperiment(Potential&& V, const std::vector<unsigned> &levels,
                         GPE_Options options, FAS_Options options_fas, CoarseModelOptions options_cm,
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
        , cond_metric_mg      (min_level, max_level)
        , manifold_mg         (min_level, max_level)
        , transfer_mg         (min_level, max_level)
        , point_transfer_mg   (min_level, max_level)
        , vector_transport_mg (min_level, max_level)
        , options_descent_mg  (min_level, max_level)
        , options_solver_mg   (min_level, max_level)
        , table_observer       (min_level, max_level)
    {
        // TODO: duplicate effort with min_element, max_element
        // sort in ascending order, coarse -> fine
        std::ranges::sort(m_levels, std::ranges::less{});

        // TODO: use AssertThrow or always use the unique_copy
        Assert(std::ranges::adjacent_find(m_levels) == m_levels.end(),
            dealii::ExcInternalError("level indices are not unique"));

        // 1. Build Physics and Manifolds for all levels
        for (auto l: m_levels) {
            builders_mg[l] = std::make_unique<ModelBuilder<System>>(V, options, l);

            options_descent_mg[l] = options_gd;
            options_solver_mg [l] = options_slv;

            // Use shared_ptr to safely store objects with reference members
            // (default copy assignment operator for MGLevelObject)
            objective_mg[l] = std::make_shared<Functional>(
                builders_mg[l]->get_system(), options.beta, options_solver_mg[l]
            );
            manifold_mg[l] = std::make_shared<UnitMassSphere<Operator>>(
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
            const Operator& M_c = objective_mg[l_c]->get_M();
            InverseM& M_inv_c = objective_mg[l_c]->get_M_inv();

            const auto& dofs_f = builders_mg[l]->get_package().get_dofs();
            const auto& constr_f = builders_mg[l]->get_package().get_constraints();
            const Operator& M_f = objective_mg[l]->get_M();

            auto [t, pt, vt] = build_transfers(
                dofs_c, dofs_f, constr_c, constr_f, M_c, M_f, M_inv_c, options_cm);

            transfer_mg[l]         = t;
            point_transfer_mg[l]   = pt;
            vector_transport_mg[l] = vt;
        }

        // 4. Metric for the coarse condition on each level
        for (auto l: m_levels) {
            cond_metric_mg[l] = build_cond_metric(*objective_mg[l], options_cm.ccond_t);
        }

        fas_solver = std::make_unique<FullApproximationScheme<Functional>>(
            manifold_mg, point_transfer_mg, vector_transport_mg, objective_mg, m_levels,
            options_descent_mg, options_solver_mg, options_fas, cond_metric_mg
        );
        fas_solver->set_observer(table_observer);
    }

    unsigned n_dofs() const { return builders_mg[max_level]->n_dofs(); }
    unsigned n_level_min() const { return min_level; }
    unsigned n_level_max() const { return max_level; }

    void distribute(Vector<double>& x) const { builders_mg[max_level]->distribute(x); }

    void run(Vector<double>& x0, MetricKind metric_t, std::ostream& os)
    {
        // Execute the cycle on the finest level
        EnergyOracle<System> O_fine(*objective_mg[max_level], options_solver_mg[max_level]);

        if (metric_t == MetricKind::FROBENIUS) {
            FrobeniusOracle<System> T_fine(*objective_mg[max_level], options_solver_mg[max_level]);

            fas_solver->template cycle<FrobeniusOracle<System>, FrobeniusCoarseOracle<System>,
                                       FrobeniusCoarseOracleEnergyAdaptive<System>, GrossPitaevskiiCoarseResidual<System>>(
                O_fine, T_fine, O_fine.get_residual(), x0, m_levels.size() - 1, os
            );
        }
        else if (metric_t == MetricKind::MASS) {
            MassOracle<System> T_fine(*objective_mg[max_level], options_solver_mg[max_level]);

            fas_solver->template cycle<MassOracle<System>, MassCoarseOracle<System>,
                                       MassCoarseOracleEnergyAdaptive<System>, GrossPitaevskiiCoarseResidual<System>>(
                O_fine, T_fine, O_fine.get_residual(), x0, m_levels.size() - 1, os
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

    const auto& history() const { return fas_solver->history(); }
    const typename System::MassMatrix& get_M(int level) const { return builders_mg[level]->get_system().get_M(); }
    const typename System::MassMatrix& get_M() const { return builders_mg[max_level]->get_system().get_M(); }
    const auto& get_package(int level) { return builders_mg[level]->get_package(); }
    const auto& get_package() { return builders_mg[max_level]->get_package(); }

private:
    std::vector<unsigned> m_levels;
    unsigned min_level, max_level;
    MGLevelObject<std::unique_ptr<ModelBuilder<System>>> builders_mg;

    MGLevelObject<std::shared_ptr<Functional>> objective_mg;
    MGLevelObject<LevelMetric> cond_metric_mg;
    MGLevelObject<std::shared_ptr<ManifoldBase>>                   manifold_mg;
    MGLevelObject<std::shared_ptr<LinearTransferBase>>             transfer_mg;
    MGLevelObject<std::shared_ptr<ManifoldTransferBase>>           point_transfer_mg;
    MGLevelObject<std::shared_ptr<VectorTransportBase>>            vector_transport_mg;
    MGLevelObject<DescentOptions>                                  options_descent_mg;
    MGLevelObject<SolverOptions>                                   options_solver_mg;

    ConvergenceTableObserver<CycleInfo>                            table_observer;

    std::unique_ptr<FullApproximationScheme<Functional>> fas_solver;
};


// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
// Writes <basename>_coords.bin and one <basename>_iter<k>.bin per iterate (see plot_solution.py)
template <int dim, typename History>
static void output_bin(const History& history, const GrossPitaevskiiPackage<dim>& package, const std::string& basename)
{
    write_support_points(package.get_dofs(), package.get_mapping(), basename + "_coords.bin");

    unsigned iter = 0;
    for (const auto& x : history) {
        write_solution(x, fmt::format("{}_iter{}.bin", basename, iter++));
    }
}


// Option groups of this program
struct ProgramOptions
{
    GPE_Options        gpe;
    DescentOptions     descent;
    SolverOptions      solver;
    MG_Options         mg;
    FAS_Options        fas;
    CoarseModelOptions coarse_model;
    OutputOptions      output;
};


// Single- or multilevel experiment, for the system type selected by --mass-lumping
template <typename System>
static void run_experiment(const ProgramOptions& opts)
{
    constexpr int dim = System::dimension;
    const unsigned n_levels = opts.mg.n_levels;
    auto potential_v = potential::get_potential<dim>(opts.gpe.potential, opts.gpe.potential_expr);

    if (opts.coarse_model.metric_t == MetricKind::NONE || opts.mg.v_levels.size() == 1) {
        // Run standard single-level Riemannian gradient descent on the finest level
        auto exp = std::visit([&](auto&& arg) {
            return SingleLevelExperiment<System>(arg, n_levels, opts.gpe, opts.solver, opts.descent);
        }, potential_v);

        Vector<double> x0(exp.n_dofs());
        x0 = 1.0;
        exp.distribute(x0);

        // Starting value on the sphere
        ellipsoid::retract_by_norm(exp.get_M(), x0);

        exp.run(x0, MetricKind::ENERGY_ADAPTIVE, std::cout);

        if (opts.output.output_bin) {
            const std::string basename = opts.output.bin_filename.empty()
                ? fmt::format("solution_{}d_sl_b{}_lvl{}", dim, opts.gpe.beta, opts.mg.v_levels.size())
                : opts.output.bin_filename;
            output_bin(exp.history(), exp.get_package(), basename);
        }
        if (opts.output.output_vtk) {
            const std::string name = opts.output.vtk_filename.empty() ? fmt::format("solution_{}d_sl", dim) : opts.output.vtk_filename;
            output_vtk(exp.history(), exp.get_package().get_dofs(), fmt::format("{}_lvl{}", name, n_levels),
                       opts.output.output_every);
        }
    }
    else {
        auto exp = std::visit([&](auto&& arg) {
            return MultiLevelExperiment<System>(arg, opts.mg.v_levels, opts.gpe, opts.fas, opts.coarse_model, opts.solver, opts.descent);
        }, potential_v);

        Vector<double> x0(exp.n_dofs());
        x0 = 1.0;
        exp.distribute(x0);

        // Starting value on the sphere
        ellipsoid::retract_by_norm(exp.get_M(), x0);

        exp.run(x0, opts.coarse_model.metric_t, std::cout);
        exp.log(std::cerr);

        if (opts.output.output_bin) {
            // TODO: Additional ML parameters in the default name?  (map for short names, e.g. OPTICAL_LATTICE -> ol)
            const std::string basename = opts.output.bin_filename.empty()
                ? fmt::format("solution_{}d_ml_b{}_lvl{}", dim, opts.gpe.beta, opts.mg.v_levels.size())
                : opts.output.bin_filename;
            output_bin(exp.history(), exp.get_package(), basename);
        }
        if (opts.output.output_vtk) {
            const std::string name = opts.output.vtk_filename.empty() ? fmt::format("solution_{}d_ml", dim) : opts.output.vtk_filename;
            output_vtk(exp.history(), exp.get_package().get_dofs(), fmt::format("{}_lvl{}", name, exp.n_level_max()),
                       opts.output.output_every);
        }
    }
}


int main(int argc, char* argv[])
{
    try {
        ProgramOptions opts{};
        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(descent_cli_options());
        all.add(mg_cli_options());
        all.add(inner_cli_options());
        all.add(fas_cli_options());
        all.add(coarse_model_cli_options());
        all.add(output_cli_options());

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.contains("help")) {
            std::cout << all << "\n";
            return 0;
        }

        apply_gpe_options(vm, opts.gpe);
        apply_descent_options(vm, opts.descent);
        apply_mg_options(vm, opts.mg);
        apply_inner_options(vm, opts.solver);
        apply_fas_options(vm, opts.fas);
        apply_coarse_model_options(vm, opts.coarse_model);
        apply_output_options(vm, opts.output);

        with_dimension(opts.gpe.dimension, [&]<typename T0>(T0)
        {
            with_system<T0::value>(opts.gpe.mass_lumping, [&]<typename System>()
            {
                run_experiment<System>(opts);
            });
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