/**
 * @file
 * @brief main_coarse on the positive sphere with mass lumping: the outputs of main_coarse (convergence tables of the
 * levels on stdout, the level sequence of the FAS cycle on stderr, --output-bin, --output-vtk) for the runs of
 * pos_sphere_ml (its options --smoother, --coarse-retraction, --point-transfer including the prototypes,
 * --coarse-grad, --coarse-max-iter, --floor, --metric-smooth; staging/pos_sphere_ml.h).
 *
 * With one level (--levels l, or --multilevel l) GradientDescent on the finest level; with two or more
 * (--multilevel l_1,...,l_n) FullApproximationScheme with the density coarse model. Requires --mass-lumping and
 * --dimension 2. Start as pos_sphere_ml: constant, thresholded, normalized.
 */
#include "pos_sphere_ml.h"

#include <rmo/ropt/observer_table.h>
#include <rmo/ropt/solver.h>
#include <rmo/util/serialize.h>
#include <rmo/util/util.h>

#include <iostream>
#include <string>

#include <fmt/format.h>

using namespace pos_sphere_ml;

namespace
{
struct ProgramOptions : Options
{
    OutputOptions output;
};

// Writes <basename>_coords.bin and one <basename>_iter<k>.bin per iterate (see staging/plot_solution.py)
template <typename History>
void output_bin(const History& history, const GrossPitaevskiiPackage<dim>& package, const std::string& basename)
{
    write_support_points(package.get_dofs(), package.get_mapping(), basename + "_coords.bin");

    unsigned iter = 0;
    for (const auto& x : history) {
        write_solution(x, fmt::format("{}_iter{}.bin", basename, iter++));
    }
}

template <typename History>
void write_outputs(const ProgramOptions& opts, const History& history, const Hierarchy& h)
{
    const char* kind = opts.multilevel() ? "ml" : "sl";
    const auto& package = h.builders[h.hi]->get_package();
    if (opts.output.output_bin) {
        const std::string basename = opts.output.bin_filename.empty()
            ? fmt::format("solution_{}d_pos_{}_b{}_lvl{}", dim, kind, opts.gpe.beta, h.levels.size())
            : opts.output.bin_filename;
        output_bin(history, package, basename);
    }
    if (opts.output.output_vtk) {
        const std::string name = opts.output.vtk_filename.empty() ? fmt::format("solution_{}d_pos_{}", dim, kind)
                                                                  : opts.output.vtk_filename;
        output_vtk(history, package.get_dofs(), fmt::format("{}_lvl{}", name, h.hi), opts.output.output_every);
    }
}

void run_single(Hierarchy& h, const ProgramOptions& opts)
{
    auto O = fine_oracle(opts, *h.objectives[h.hi]);
    GradientDescent gd(*O, O->get_residual(), *h.manifolds[h.hi], opts.descent);
    ConvergenceTableObserver<CycleInfo> observer;
    gd.set_observer(observer);
    Vector<double> x = h.start();
    gd.cycle(x, std::cout);
    write_outputs(opts, gd.history(), h);
}

void run_multilevel(Hierarchy& h, const ProgramOptions& opts)
{
    FullApproximationScheme<Functional> fas(h.manifolds, h.point_transfers, h.transports, h.objectives, h.levels,
                                            h.descent, h.solver, opts.fas);
    ConvergenceTableObserver<CycleInfo> observer(h.lo, h.hi);
    fas.set_observer(observer);
    auto O_fine = fine_oracle(opts, *h.objectives[h.hi]);
    MassOracle<System> T_fine(*h.objectives[h.hi], opts.solver);
    Vector<double> x = h.start();
    with_coarse_grad(opts.coarse_grad, [&]<DensityGradient G>() {
        fas_cycle<G>(fas, *O_fine, T_fine, x, h.levels.size(), std::cout);
    });

    // Level sequence of the cycle, as main_coarse
    const auto levels = fas.cycle_log();
    std::cerr << "Total iterations: " << levels.size() << std::endl;
    if (!levels.empty()) {
        std::cerr << "[";
        for (auto it = levels.begin(); it != levels.end() - 1; ++it) {
            std::cerr << *it << ",";
        }
        std::cerr << levels.back() << "]";
    }
    write_outputs(opts, fas.history(), h);
}

} // namespace


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
        all.add(smoother_cli_options());
        all.add(positive_cli_options());
        all.add(output_cli_options());

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.contains("help")) {
            std::cout << all << "\n";
            return 0;
        }

        apply_options(vm, opts, "main_coarse_lumping");
        apply_output_options(vm, opts.output);

        Hierarchy h(opts);
        opts.multilevel() ? run_multilevel(h, opts) : run_single(h, opts);
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
