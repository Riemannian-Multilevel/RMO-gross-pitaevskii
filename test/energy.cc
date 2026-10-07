/**
 * @file
 * @brief Checks the value of the Gross-Pitaevskii energy of the consistent system.
 *
 * GrossPitaevskiiFunctional::value() evaluates \f$ \frac\beta4 \int_\Omega u_h^4 \f$ as
 * \f$ \frac\beta4 x^\top M_{\phi\phi}(x) x \f$. This test integrates \f$ u_h^4 \f$ directly by a loop over the cells,
 * with the same quadrature, and compares the two energies at random points. The gradient test cannot detect a wrong
 * scaling of \f$ M_{\phi\phi} \f$, since value and gradient would stay consistent.
 */
#include "check.h"

#include <rmo/gpe/gpe.h>
#include <rmo/gpe/manifold.h>

#include <deal.II/fe/fe_values.h>

#include <cmath>
#include <string>
#include <vector>

using namespace rmo;
using namespace rmo::gpe;
using namespace rmo::test;
using namespace dealii;

namespace
{
constexpr int    dim  = 2;
constexpr double beta = 100.0;

//! \f$ \int_\Omega u_h^4 \f$ by a loop over the cells, with the quadrature and mapping of @p package.
double integrate_fourth_power(const GrossPitaevskiiPackage<dim>& package, const Vector<double>& x)
{
    const auto& dofs = package.get_dofs();
    FEValues<dim> fe_values(package.get_mapping(), dofs.get_fe(), package.get_quadrature(),
                            update_values | update_JxW_values);
    std::vector<double> u(package.get_quadrature().size());
    double integral = 0.0;

    for (const auto& cell : dofs.active_cell_iterators()) {
        fe_values.reinit(cell);
        fe_values.get_function_values(x, u);
        for (const unsigned q : fe_values.quadrature_point_indices()) {
            integral += std::pow(u[q], 4) * fe_values.JxW(q);
        }
    }
    return integral;
}

void check_energy(CheckReport& report, MeshKind mesh, int degree, const std::string& name)
{
    GPE_Options options{};
    options.dimension = dim;
    options.degree    = degree;
    options.radius    = 8.0;
    options.beta      = beta;
    options.bc        = BoundaryCondition::DIRICHLET;
    options.mesh_kind = mesh;

    const GrossPitaevskiiPackage<dim> package(options, 4);
    auto system = package.system(potential::Square<dim>());
    GrossPitaevskiiFunctional<GrossPitaevskiiSystem<dim>> func(system, beta, SolverOptions{});

    double max_diff = 0.0;
    for (unsigned trial = 0; trial < 5; ++trial) {
        Vector<double> x(package.n_dofs()), A0x(package.n_dofs());
        ellipsoid::random_point(x, func.get_M());
        package.distribute(x);
        func.update(x);

        system.get_A0().vmult(A0x, x);
        const double ref = 0.5 * (x * A0x) + 0.25 * beta * integrate_fourth_power(package, x);
        max_diff = std::max(max_diff, std::abs(func.value(x) - ref) / std::abs(ref));
    }
    report.check(max_diff < 1e-12, name + ": value() equals the energy integrated by a cell loop",
                 "max rel. diff " + sci(max_diff));
}

} // namespace


int main()
{
    std::cerr.setstate(std::ios::failbit);  // silence the mesh statistics of GrossPitaevskiiPackage

    return run_tests([](CheckReport& report) {
        check_energy(report, MeshKind::QUADRILATERAL, 1, "Q1");
        check_energy(report, MeshKind::QUADRILATERAL, 2, "Q2");
        check_energy(report, MeshKind::SIMPLEX, 1, "P1");
        check_energy(report, MeshKind::SIMPLEX, 2, "P2-bubble");
    });
}
