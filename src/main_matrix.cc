//
// Created by Ferdinand Vanmaele on 09.10.26.
//
#include <rmo/fe/space.h>
#include <rmo/fe/grid.h>
#include <rmo/fe/util.h>

#include <rmo/util/util.h>
#include <rmo/option.h>

#include <fmt/format.h>
#include <iostream>

#include "rmo/gpe/gpe.h"
#include "rmo/util/serialize.h"

/** @file
 * @brief Program to evaluate properties of finite element matrices (mass, stiffness).
 * This includes
 * - Printing the sparsity pattern for various orderings of degrees-of-freedom.
 * - Exporting the mass and stiffness matrices to MatrixMarket files.
 * - Analyzing the graph induced by the stiffness matrix. (TODO)
 * - Analyzing the numerical properties of the stiffness matrix. (TODO)
 */

using namespace rmo;
using namespace rmo::gpe;
using namespace dealii;


struct ProgramOptions
{
    GPE_Options   gpe;
    MG_Options    mg;
    OutputOptions output;

    bool domain;
    bool matrix_market;
    bool matrix_props;
    std::string mm_filename;
};

template <int MatrixType>
class MatrixGraph
{
public:

private:
};

// Requirements:
// - [OK] GrossPitaevskiiPackage (discretization)
// - [OK] Sparsity pattern
// - [OK] Assembled mass (lumped, consistent) and stiffness matrices
//   performed inside this class; GrossPitaevskii{,Lumped}System assemble A0 = S + M_V; potential not needed
// - [OK] Matrix market export
// - Adjacency matrix to (undirected) graph -> boost graph library
// - Connected components

template <int dim, typename MassMatrix>
class MatrixExperiment
{
public:
    MatrixExperiment(const GPE_Options& options, unsigned int n_levels)
    // discretization
        : m_context(options, n_levels)
    // problem parameters
        , m_options(options)
    {
    // sparsity pattern for (constrained) matrix)
        auto dsp = fe::make_sparsity_pattern(m_context.get_dofs(), m_context.get_constraints());
        m_sparsity.copy_from(dsp);
    // stiffness matrix
        S.reinit(m_sparsity);
        fe::assemble_stiffness(S, m_context);
    // mass matrix
        if constexpr (is_diagonal_matrix_v<MassMatrix>) {
            // lumped system: diagonal matrix
            M.get_vector().reinit(m_context.n_dofs());
            fe::assemble_mass_lumped(M, m_context);
        }
        else {
            // system: sparse matrix with same sparsity as S
            M.reinit(m_sparsity);
            fe::assemble_mass(M, m_context);
        }
    }

    void write_domain(const std::string& prefix, unsigned int level) const
    {
        const std::string mesh_str = m_context.get_grid().has_simplex ? "simplex" : "quads";

        if (dim == 2) {
            write_grid(prefix, mesh_str, level);
        }
        write_dof(prefix, mesh_str, level);
        write_sparsity(prefix, mesh_str, level);
    }

    void write_matrix_market(const std::string& prefix, unsigned int level) const
    {
        std::string base = fmt::format("{}_lvl{}_dim{}", prefix, level, dim);

        rmo::write_matrix_market(M, base + "_M");
        rmo::write_matrix_market(S, base + "_S");
    }

private:
    // TODO: build prefix + mesh_str + level in caller
    void write_grid(const std::string& prefix, const std::string& mesh_str, unsigned int level) const
    {
        std::string grid_file = fmt::format("{}_{}d_{}_lvl{}",
            prefix, dim, mesh_str, level);

        fe::write_grid<dim>(grid_file + ".svg", m_context.get_grid().triangulation,
            GridOut::OutputFormat::svg);
        std::cerr << "Saving " + grid_file + ".svg" << std::endl;
    }

    // TODO: build prefix + mesh_str + level in caller
    void write_dof(const std::string& prefix, const std::string& mesh_str, unsigned int level) const
    {
        std::string dof_file = fmt::format("{}_{}d_{}_lvl{}_dof.gnuplot",
            prefix, dim, mesh_str, level);

        fe::write_dof_locations<dim>(m_context.get_dofs(), dof_file, m_context.get_mapping());
        std::cerr << "Saving " + dof_file << std::endl;
    }

    // TODO: build prefix + mesh_str + level in caller
    void write_sparsity(const std::string& prefix, const std::string& mesh_str, unsigned int level) const
    {
        std::string sparsity_file = fmt::format("{}_{}d_{}_lvl{}_sparsity.svg",
            prefix, dim, mesh_str, level);

        std::ofstream out(sparsity_file);
        m_sparsity.print_svg(out);
        std::cerr << "Saving " + sparsity_file << std::endl;
    }

    GrossPitaevskiiPackage<dim> m_context;
    GPE_Options m_options;
    SparsityPattern m_sparsity;

    SparseMatrix<double> S;
    MassMatrix M;
};

// ("matrix-market", po::value<std::string>()->implicit_value(""),
// "write mass and stiffness matrices in matrix market format, optionally with base name (--matrix-market=<name>)");
//     options_out.matrix_market = vm.contains("matrix-market");
// if (options_out.matrix_market) {
//    options_out.mm_filename = vm["matrix-market"].as<std::string>();
// }
int main(int argc, char** argv)
{
    ProgramOptions opts{};

    try {
        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(mg_cli_options());
        all.add(output_cli_options());

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << all << "\n";
            return 0;
        }
        apply_gpe_options(vm, opts.gpe);
        NumberGenerator::get().seed(opts.gpe.seed);

        apply_mg_options(vm, opts.mg);
        apply_output_options(vm, opts.output);

        // Main loop
        with_dimension(opts.gpe.dimension, [&]<typename T0>(T0)
        {
            constexpr int dim = T0::value;

            with_system<T0::value>(opts.gpe.mass_lumping, [&]<typename System>()
            {
                for (unsigned int level: opts.mg.v_levels) {
                    MatrixExperiment<dim, typename System::MassMatrix>
                    Analysis(opts.gpe, level+1);

                    // TODO: conditional write (ProgramOptions::domain, ProgramOptions::matrix_market)
                    Analysis.write_domain("domain", level);
                    Analysis.write_matrix_market("matrix", level);
                }
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