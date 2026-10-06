//
// Verifies that the stiffness matrix S, or A0 = S + M_V, is an irreducible M-matrix on the mesh of
// GrossPitaevskiiPackage, using Taussky's theorem (tex/taussky_theorem.tex): a symmetric matrix with
// positive diagonal, non-positive off-diagonal entries, weak diagonal dominance in every row, strict
// dominance in at least one row and a connected matrix graph is a non-singular M-matrix whose inverse
// is strictly positive.
//
// Only unconstrained (non-Dirichlet) DoFs are considered, and entries with |a_ij| <= tol max|a| are
// treated as zero. Exit code: 0 if all hypotheses hold on every level, 1 otherwise, 2 on errors.
//
#include <rmo/fe/assemble.h>
#include <rmo/gpe/gpe.h>
#include <rmo/option.h>
#include <rmo/util/sparsity.h>
#include <rmo/util/util.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

using namespace rmo;
using namespace rmo::gpe;
using namespace dealii;

namespace
{

struct CheckOptions
{
    std::string matrix;          // stiffness | A0
    double      tol;             // relative threshold for zero entries
    bool        print_components;
    unsigned    inverse_columns; // columns of A^{-1} to check for positivity (0: skip)
};

//! Properties of the matrix restricted to the unconstrained DoFs; values relative to max |a_ij|
struct Analysis
{
    unsigned n_free               = 0;
    double   symmetry_error       = 0;  // max |a_ij - a_ji|
    double   min_diagonal         = 0;
    unsigned n_positive_offdiag   = 0;
    double   max_positive_offdiag = 0;
    unsigned n_not_dominant       = 0;  // rows with a_ii < sum_{j != i} |a_ij|
    unsigned n_strictly_dominant  = 0;  // rows with a_ii > sum_{j != i} |a_ij|
    std::vector<unsigned> component_sizes;
};

Analysis analyze(const SparseMatrix<double>& A, const AffineConstraints<double>& constraints, double tol)
{
    Analysis an;
    const unsigned n = A.m();

    std::vector<unsigned> free_index(n, numbers::invalid_unsigned_int);
    for (unsigned i = 0; i < n; ++i) {
        if (!constraints.is_constrained(i)) {
            free_index[i] = an.n_free++;
        }
    }
    AssertThrow(an.n_free > 0, ExcMessage("all degrees of freedom are constrained"));

    double scale = 0.0;
    for (unsigned i = 0; i < n; ++i) {
        if (free_index[i] == numbers::invalid_unsigned_int) {
            continue;
        }
        for (auto it = A.begin(i); it != A.end(i); ++it) {
            if (free_index[it->column()] != numbers::invalid_unsigned_int) {
                scale = std::max(scale, std::abs(it->value()));
            }
        }
    }
    const double eps = tol * scale;

    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph(an.n_free);
    an.min_diagonal = std::numeric_limits<double>::max();

    for (unsigned i = 0; i < n; ++i) {
        if (free_index[i] == numbers::invalid_unsigned_int) {
            continue;
        }
        double diagonal = 0.0, off_diagonal_sum = 0.0;

        for (auto it = A.begin(i); it != A.end(i); ++it) {
            const unsigned j = it->column();
            const double   v = it->value();
            if (free_index[j] == numbers::invalid_unsigned_int) {
                continue;
            }
            if (j == i) {
                diagonal = v;
                continue;
            }
            an.symmetry_error = std::max(an.symmetry_error, std::abs(v - A.el(j, i)));
            if (v > eps) {
                ++an.n_positive_offdiag;
                an.max_positive_offdiag = std::max(an.max_positive_offdiag, v);
            }
            off_diagonal_sum += std::abs(v);

            if (std::abs(v) > eps && i < j) {
                boost::add_edge(free_index[i], free_index[j], graph);
            }
        }
        an.min_diagonal = std::min(an.min_diagonal, diagonal);

        const double margin = diagonal - off_diagonal_sum;
        if (margin < -eps) {
            ++an.n_not_dominant;
        }
        else if (margin > eps) {
            ++an.n_strictly_dominant;
        }
    }

    std::vector<unsigned> component(an.n_free);
    const unsigned n_components = boost::connected_components(graph, component.data());
    an.component_sizes.assign(n_components, 0);
    for (unsigned c : component) {
        ++an.component_sizes[c];
    }
    std::ranges::sort(an.component_sizes, std::greater<>());

    an.symmetry_error       /= scale;
    an.min_diagonal         /= scale;
    an.max_positive_offdiag /= scale;
    return an;
}

//! Smallest entry of A^{-1} e_i over `n_columns` unconstrained columns i, relative to the largest entry
double min_inverse_entry(const SparseMatrix<double>& A, const AffineConstraints<double>& constraints,
                         unsigned n_columns)
{
    const unsigned n = A.m();
    double min_rel = 1.0;
    PreconditionJacobi<SparseMatrix<double>> jacobi;
    jacobi.initialize(A);

    for (unsigned k = 1; k <= n_columns; ++k) {
        unsigned i = (k * n) / (n_columns + 1);
        while (i < n && constraints.is_constrained(i)) {
            ++i;
        }
        if (i == n) {
            continue;
        }
        Vector<double> e(n), y(n);
        e[i] = 1.0;

        SolverControl control(10 * n, 1e-14);
        SolverCG<Vector<double>> cg(control);
        cg.solve(A, y, e, jacobi);

        for (unsigned j = 0; j < n; ++j) {
            if (!constraints.is_constrained(j)) {
                min_rel = std::min(min_rel, y[j] / y.linfty_norm());
            }
        }
    }
    return min_rel;
}

std::string sci(double x)
{
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(2) << x;
    return ss.str();
}

//! Prints the report; returns true if all hypotheses of Taussky's theorem hold
bool report(const Analysis& an, const CheckOptions& opts)
{
    bool all = true;
    auto line = [&](bool ok, const std::string& name, const std::string& detail) {
        std::cout << (ok ? "  PASS  " : "  FAIL  ") << name << "  (" << detail << ")\n";
        all = all && ok;
    };

    line(an.symmetry_error <= opts.tol, "symmetric", "max |a_ij - a_ji| = " + sci(an.symmetry_error));
    line(an.min_diagonal > opts.tol, "positive diagonal", "min a_ii = " + sci(an.min_diagonal));
    line(an.n_positive_offdiag == 0, "non-positive off-diagonal entries (Z-matrix)",
         std::to_string(an.n_positive_offdiag) + " positive, max " + sci(an.max_positive_offdiag));
    line(an.n_not_dominant == 0, "weakly diagonally dominant in every row",
         std::to_string(an.n_not_dominant) + " rows not dominant");
    line(an.n_strictly_dominant > 0, "strictly diagonally dominant in at least one row",
         std::to_string(an.n_strictly_dominant) + " of " + std::to_string(an.n_free) + " rows");
    line(an.component_sizes.size() == 1, "irreducible (connected matrix graph)",
         std::to_string(an.component_sizes.size()) + " connected component(s)");

    if (opts.print_components && an.component_sizes.size() > 1) {
        std::cout << "        component sizes:";
        for (std::size_t c = 0; c < an.component_sizes.size() && c < 20; ++c) {
            std::cout << " " << an.component_sizes[c];
        }
        std::cout << (an.component_sizes.size() > 20 ? " ...\n" : "\n");
    }
    std::cout << (all ? "  => irreducible non-singular M-matrix (Taussky); its inverse is strictly positive\n"
                      : "  => hypotheses of Taussky's theorem not satisfied\n");
    return all;
}

template <int dim>
bool check_level(const GPE_Options& options, const CheckOptions& opts, unsigned level)
{
    GrossPitaevskiiPackage<dim> package(options, level);
    const auto& dofs        = package.get_dofs();
    const auto& constraints = package.get_constraints();

    std::cout << "level " << level << ": " << package.get_grid().triangulation.n_active_cells() << " cells, "
              << dofs.n_dofs() << " DoFs, matrix " << opts.matrix
              << (opts.matrix == "A0" ? (options.mass_lumping ? " (lumped)" : " (consistent)") : "") << "\n";

    auto check = [&](const SparseMatrix<double>& A) {
        const Analysis an = analyze(A, constraints, opts.tol);
        std::cout << "  " << an.n_free << " unconstrained DoFs\n";
        const bool ok = report(an, opts);
        if (opts.inverse_columns > 0) {
            std::cout << "  INFO  min relative entry of " << opts.inverse_columns << " columns of A^{-1}: "
                      << sci(min_inverse_entry(A, constraints, opts.inverse_columns)) << "\n";
        }
        return ok;
    };

    if (opts.matrix == "stiffness") {
        SparsityPattern sparsity;
        sparsity.copy_from(make_sparsity_pattern(dofs, constraints));
        SparseMatrix<double> S(sparsity);
        fe::assemble_stiffness(S, dofs, package.get_quadrature(), package.get_mapping(), constraints);
        return check(S);
    }

    auto potential_v = potential::get_potential<dim>(options.potential, options.potential_expr);
    return std::visit([&](auto&& V) {
        if (options.mass_lumping) {
            const auto system = package.template system<GrossPitaevskiiLumpedSystem<dim>>(V);
            return check(system.get_A0());
        }
        const auto system = package.template system<GrossPitaevskiiSystem<dim>>(V);
        return check(system.get_A0());
    }, potential_v);
}

} // namespace


int main(int argc, char* argv[])
{
    GPE_Options  options{};
    MG_Options   options_mg{};
    CheckOptions opts{};

    try {
        po::options_description check_options("M-matrix check options");
        check_options.add_options()
            ("matrix", po::value<std::string>()->default_value("stiffness"),
                "matrix to check (stiffness|A0); A0 = S + M_V, lumped with --mass-lumping")
            ("tol", po::value<double>()->default_value(1e-14),
                "relative threshold: entries with |a_ij| <= tol max|a| are treated as zero")
            ("print-components", po::bool_switch(),
                "print the sizes of the connected components if there are several")
            ("inverse-columns", po::value<unsigned>()->default_value(0),
                "also check positivity of this many columns of A^{-1} (0: skip)");

        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(mg_cli_options());
        all.add(check_options);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.contains("help")) {
            std::cout << "Checks the irreducible M-matrix property (Taussky's theorem) on each level of --levels or --multilevel\n\n"
                      << all << "\n";
            return 0;
        }
        apply_gpe_options(vm, options);
        apply_mg_options(vm, options_mg);

        opts.matrix           = vm["matrix"].as<std::string>();
        opts.tol              = vm["tol"].as<double>();
        opts.print_components = vm["print-components"].as<bool>();
        opts.inverse_columns  = vm["inverse-columns"].as<unsigned>();
        AssertThrow(opts.matrix == "stiffness" || opts.matrix == "A0",
                    ExcMessage("--matrix must be stiffness or A0"));

        bool all_ok = true;
        with_dimension(options.dimension, [&]<typename T0>(T0) {
            for (unsigned level : options_mg.v_levels) {
                all_ok = check_level<T0::value>(options, opts, level) && all_ok;
            }
        });
        return all_ok ? 0 : 1;
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
