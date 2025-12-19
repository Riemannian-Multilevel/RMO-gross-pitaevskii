#include "function.h"
#include "operators.h"
#include "gpe.h"
#include "descent.h"
#include "util.h"
#include "options.h"

#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/numerics/fe_field_function.h>
#include <iostream>
#include <fmt/format.h>

using namespace gpe;
using namespace dealii;


// TODO: inherit from GPE base class
template <int dim, typename ExecutionPolicy>
class GPE_Solve_MG
{
public:
    explicit GPE_Solve_MG(const GPE_Options& options,
        unsigned int min_level_ = 0,
        unsigned int max_level_ = numbers::invalid_unsigned_int)
    :
        problem(options), min_level(min_level_), max_level(max_level_)
    {
        if (max_level == numbers::invalid_unsigned_int) {
            max_level = problem.get_triangulation().n_global_levels();
        }
    }

    void setup()
    {
        problem.make_grid();
        problem.dofs_mg();
    }

    template <typename Function>
    void get_matrix(Function&& V, LevelMatrix& lm, unsigned int level) const
    {
        AssertIndexRange(level, problem.get_triangulation().n_global_levels());

        // Initialize sparsity pattern and compute entries (level)
        const AffineConstraints<double>& level_constraints = problem.get_level_constraints(level);
        const DoFHandler<dim>& dof_handler = problem.get_dofs();
        // "Note that there is [no] need to consider hanging nodes on the typical level matrices,
        // since only one level is considered."
        // TODO: we may still want to pass on Dirichlet constraints
        lm.reinit(make_sparsity_pattern_mg(dof_handler, level));

        assemble_mass(policy, lm.M, problem.get_dofs(), level_constraints, level);
        assemble_A0(policy, lm.A0, problem.get_dofs(), V, level_constraints, level);
    }

    template <typename Function>
    [[maybe_unused]] MGLevelObject<Vector<double>>
    run(Function&& V, const double x0d, double beta, GdOptions options_rgd, int n_check_res = 5) const
    {
        const Triangulation<dim>& triangulation = problem.get_triangulation();
        const DoFHandler<dim>& dof_handler = problem.get_dofs();
        const AffineConstraints<double>& constraints = problem.get_constraints();

        // FE matrices for every multigrid level
        //std::vector<LevelMatrix> lm_v(n_levels);
        MGLevelObject<LevelMatrix> lm_v(min_level, max_level-1);  // inclusive range
        for (unsigned int level = min_level; level < max_level; ++level) {
            this->get_matrix(V, lm_v[level], level);
        }

        // Build transfer operators
        MGTransferPrebuilt<Vector<double> > mg_transfer(problem.get_mg_dofs());
        mg_transfer.build(dof_handler);

        // Iterate over levels
        MGLevelObject<Vector<double> > x_v(min_level, max_level-1);
        //std::vector<Vector<double> > x_v(n_levels);

        for (unsigned int level = min_level; level < max_level; level++) {
            std::cerr << "Level: " << level << std::endl;
            std::cerr << "Number of cells: " << triangulation.n_cells(level) << std::endl;
            std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs(level) << std::endl;

            // Update weighed matrix for current solution + boundary conditions
            const AffineConstraints<double>& level_constraints = problem.get_level_constraints(level);

            // Define starting value
            // TODO: take MGLevelObject of starting vectors, then overload on constant value
            Vector<double> x0(dof_handler.n_dofs(level));
            x0 = x0d;

            // TODO: check for missing steps in tutorial/step-16
            // if (level == min_level) {
            //     // Constant value on coarsest level
            //     x0 = x0d;
            // }
            // else if (level+1 < max_level) {
            //     // Linear interpolation of solution on previous level
            //     mg_transfer.prolongate(level, x0, x_v[level-1]);
            //
            //     // Apply fine level constraints
            //     level_constraints.distribute(x0);
            //
            //     // Renormalize in M-norm
            //     Vector<double> Mx0(x0.size());
            //     lm_v[level].M.vmult(Mx0, x0);
            //     x0 /= std::sqrt(x0 * Mx0);
            // }

            auto update_mpp_level = [&dof_handler, level, &level_constraints](
                SparseMatrix<double>& Mpp, const Vector<double>& x)
            {
                assemble_mass_phiphi<dim>(policy, Mpp, dof_handler, x, level_constraints, level);
            };

            // Gradient descent + enforce boundary conditions
            x_v[level] = gp_energy_rgd<dim>(lm_v[level].A0, lm_v[level].M, lm_v[level].Mpp,
                update_mpp_level, x0, beta, level_constraints, options_rgd, n_check_res);
            std::cerr << std::endl;
        }
        return x_v;
    }

    const DoFHandler<dim>& get_dofs() const
    {
        return problem.get_dofs();
    }

private:
    // Problem parameters
    GPE<dim> problem;
    static constexpr ExecutionPolicy policy{};
    unsigned int min_level, max_level;
};

template <int dim, typename ExecutionPolicy>
void package(const unsigned min_level, const unsigned max_level,
    const GPE_Options& options, const GdOptions& opt_rgd)
{
    Square<dim> V;
    GPE_Solve_MG<dim, ExecutionPolicy> GS(options, min_level, max_level);
    GS.setup();

    auto xv = GS.run(V, 1.0, options.beta, opt_rgd);
}

template <int dim>
void run_package(bool parallel, const unsigned min_level, const unsigned max_level,
    const GPE_Options& options, const GdOptions& options_rgd)
{
    if (parallel) {
        package<dim, execution::par_t>(min_level, max_level, options, options_rgd);
    } else {
        package<dim, execution::seq_t>(min_level, max_level, options, options_rgd);
    }
}

int main(int argc, char* argv[])
{
    GPE_Options options{};
    GdOptions   options_rgd{};
    MG_Options  options_mg{};

    // TODO: add configuration file (cf. boost tutorial)
    try {
        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(gd_cli_options());
        all.add(mg_cli_options());

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << all << "\n";
            return 0;
        }
        apply_gpe_options(vm, options);
        apply_gd_options(vm, options_rgd);
        apply_mg_options(vm, options_mg);

        with_dimension(options.dimension, [&](auto D)
        {
            constexpr int dim = decltype(D)::value;

            run_package<dim>(options_mg.parallel, options_mg.min_level,
                options_mg.max_level, options, options_rgd);
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