/**
 * @file
 * @brief Tests of mass lumping: the lumped matrices of GrossPitaevskiiLumpedSystem and the functional, oracles and
 * iterations built on it.
 *
 * - elements whose mass matrix has non-positive row sums are rejected;
 * - \f$ M_L \f$ holds the row sums of \f$ M \f$, independently of the quadrature;
 * - \f$ A_0 = S + M_{V,L} \f$ with the nodal quadrature of the potential term, and \f$ M_{\phi\phi} \f$ equals its
 *   assembly by a cell loop;
 * - the operator \f$ A = A_0 + \beta M_{\phi\phi} \f$, which mixes a sparse and a diagonal matrix;
 * - the lumped and the consistent energy of a smooth function differ by \f$ O(h^2) \f$;
 * - the functional: exact \f$ M^{-1} \f$, gradient by finite differences, \f$ A^{-1} \f$;
 * - the oracles: tangent gradient, metric of their kind, same gradient as the iterations.
 */
#include "check.h"
#include "finite_difference.h"

#include <rmo/gpe/gpe.h>
#include <rmo/gpe/iteration.h>
#include <rmo/gpe/oracle.h>

#include <deal.II/base/function.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <string>
#include <type_traits>
#include <vector>

using namespace rmo;
using namespace rmo::gpe;
using namespace rmo::test;
using namespace dealii;

namespace
{
constexpr int    dim    = 2;
constexpr double radius = 4.0;
constexpr double beta   = 100.0;

//! Element and mesh of a test case: Q<degree> on squares, or P1 / P2-bubble on simplices.
struct Discretization
{
    MeshKind mesh;
    int      degree;

    [[nodiscard]] std::string name() const
    {
        if (mesh == MeshKind::SIMPLEX) {
            return "P" + std::to_string(degree) + (degree > 1 ? "-bubble" : "");
        }
        return "Q" + std::to_string(degree);
    }

    [[nodiscard]] GPE_Options options(BoundaryCondition bc) const
    {
        GPE_Options options{};
        options.dimension = dim;
        options.degree    = degree;
        options.radius    = radius;
        options.beta      = beta;
        options.order     = Ordering::DEFAULT;
        options.bc        = bc;
        options.mesh_kind = mesh;
        options.potential = Potential::SQUARE;
        return options;
    }

    //! Gauss rule of higher order than the one of the package (degree + 1).
    [[nodiscard]] Quadrature<dim> gauss() const
    {
        if (mesh == MeshKind::SIMPLEX) {
            return QGaussSimplex<dim>(degree + 2);
        }
        return QGauss<dim>(degree + 2);
    }
};

const std::array discretizations{Discretization{MeshKind::QUADRILATERAL, 1}, Discretization{MeshKind::QUADRILATERAL, 2},
                                 Discretization{MeshKind::SIMPLEX, 1},       Discretization{MeshKind::SIMPLEX, 2}};

//! Package with the consistent and the lumped system of one discretization, for the square potential.
struct Systems
{
    Systems(const Discretization& disc, BoundaryCondition bc, unsigned n_levels)
        : name(disc.name())
        , package(disc.options(bc), n_levels)
        , consistent(package.system(V))
        , lumped(package.system<GrossPitaevskiiLumpedSystem<dim>>(V))
    {}

    [[nodiscard]] unsigned n_dofs() const { return package.n_dofs(); }

    const std::string name;
    const potential::Square<dim> V{};
    GrossPitaevskiiPackage<dim> package;
    GrossPitaevskiiSystem<dim> consistent;
    GrossPitaevskiiLumpedSystem<dim> lumped;
};

//! Deterministic test vector with entries f(i).
template <typename Function>
Vector<double> make_vector(unsigned n, Function&& f)
{
    Vector<double> v(n);
    for (unsigned i = 0; i < n; ++i) {
        v[i] = f(i);
    }
    return v;
}

Vector<double> test_point(unsigned n) { return make_vector(n, [](unsigned i) { return std::sin(0.37 * i) + 0.2; }); }
Vector<double> test_direction(unsigned n) { return make_vector(n, [](unsigned i) { return std::cos(1.3 * i); }); }

//! Energy \f$ \frac12 x^\top A_0 x + \frac\beta4 x^\top M_{\phi\phi}(x) x \f$; reassembles \f$ M_{\phi\phi}(x) \f$.
template <typename System>
double energy(System& system, const Vector<double>& x)
{
    system.assemble_nonlinear_term(x);
    Vector<double> Ax(x.size());
    system.get_operator_A(0.25 * beta, 0.5).vmult(Ax, x);
    return x * Ax;
}


// ---------------------------------------------------------------------------------------------
// Lumped matrices
// ---------------------------------------------------------------------------------------------

//! The vertex shape functions of FE_SimplexP(2) integrate to zero, so the system must reject it.
void check_lumpability(CheckReport& report)
{
    Triangulation<dim> tria;
    GridGenerator::subdivided_hyper_cube_with_simplices(tria, 2);
    const FE_SimplexP<dim> fe(2);
    DoFHandler<dim> dofs(tria);
    dofs.distribute_dofs(fe);

    const MappingFE<dim> mapping(FE_SimplexP<dim>(1));
    const QGaussSimplex<dim> quadrature(3);
    AffineConstraints<double> constraints;
    constraints.close();

    bool thrown = false;
    try {
        const GrossPitaevskiiLumpedSystem<dim> system(dofs, quadrature, mapping, constraints, potential::Square<dim>());
    }
    catch (const std::exception&) {
        thrown = true;
    }
    report.check(thrown, "FE_SimplexP(2) is rejected (zero row sums of M)");
}

//! M_L holds the (positive) row sums of M, and does not change with a higher-order quadrature.
void check_lumped_mass(CheckReport& report, const Systems& s, const Quadrature<dim>& gauss)
{
    const auto& M   = s.consistent.get_M();
    const auto& M_L = s.lumped.get_M().get_vector();

    double err = 0.0;
    for (unsigned i = 0; i < s.n_dofs(); ++i) {
        double row_sum = 0.0;
        for (auto it = M.begin(i); it != M.end(i); ++it) {
            row_sum += it->value();
        }
        err = std::max(err, std::abs(row_sum - M_L[i]) / M_L[i]);
    }
    report.check(err < 1e-12 && *std::ranges::min_element(M_L) > 0, s.name + ": M_L holds the positive row sums of M",
                 "max rel. diff " + sci(err));

    DiagonalMatrix<Vector<double>> M_gauss;
    M_gauss.get_vector().reinit(s.n_dofs());
    fe::assemble_mass_lumped(M_gauss, s.package.get_dofs(), gauss, s.package.get_mapping(), s.package.get_constraints());
    Vector<double> diff(M_gauss.get_vector());
    diff -= M_L;
    report.check(diff.linfty_norm() < 1e-14 * M_L.linfty_norm(),
                 s.name + ": M_L independent of the quadrature (higher-order Gauss)", "max diff " + sci(diff.linfty_norm()));
}

//! Reference for the lumped potential term: (M_{V,L})_ii by the nodal quadrature of deal.II.
Vector<double> nodal_potential_mass(const Systems& s)
{
    const auto& dofs = s.package.get_dofs();
    const Quadrature<dim> nodal = FETools::compute_nodal_quadrature(dofs.get_fe());
    FEValues<dim> fe_values(s.package.get_mapping(), dofs.get_fe(), nodal,
                            update_values | update_JxW_values | update_quadrature_points);
    std::vector<types::global_dof_index> indices(dofs.get_fe().n_dofs_per_cell());
    Vector<double> M_VL(s.n_dofs());

    for (const auto& cell : dofs.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell->get_dof_indices(indices);
        for (const unsigned q : fe_values.quadrature_point_indices()) {
            for (const unsigned j : fe_values.dof_indices()) {
                M_VL[indices[j]] += s.V(fe_values.quadrature_point(q)) * fe_values.shape_value(j, q) * fe_values.JxW(q);
            }
        }
    }
    return M_VL;
}

//! Off-diagonal entries of A0 equal S, diagonal entries S + M_{V,L}.
void check_lumped_A0(CheckReport& report, const Systems& s, const Quadrature<dim>& gauss)
{
    const auto& A0 = s.lumped.get_A0();
    SparseMatrix<double> S(A0.get_sparsity_pattern());
    fe::assemble_stiffness(S, s.package.get_dofs(), gauss, s.package.get_mapping(), s.package.get_constraints());
    const Vector<double> M_VL = nodal_potential_mass(s);

    double err_offdiag = 0.0, err_diag = 0.0;
    for (unsigned i = 0; i < s.n_dofs(); ++i) {
        for (auto it = A0.begin(i); it != A0.end(i); ++it) {
            const unsigned j = it->column();
            if (i == j) {
                err_diag = std::max(err_diag, std::abs(it->value() - S.el(i, j) - M_VL[i]));
            } else {
                err_offdiag = std::max(err_offdiag, std::abs(it->value() - S.el(i, j)));
            }
        }
    }
    const double scale = A0.linfty_norm();
    report.check(err_offdiag < 1e-12 * scale && err_diag < 1e-12 * scale, s.name + ": A0 = S + M_{V,L}",
                 "off-diag " + sci(err_offdiag) + ", diag " + sci(err_diag));
}

void check_lumped_Mpp(CheckReport& report, Systems& s, const Quadrature<dim>& gauss)
{
    const Vector<double> u = test_point(s.n_dofs());
    s.lumped.assemble_nonlinear_term(u);

    DiagonalMatrix<Vector<double>> Mpp;
    Mpp.get_vector().reinit(s.n_dofs());
    fe::assemble_mass_phiphi_lumped(Mpp, u, s.package.get_dofs(), gauss, s.package.get_mapping(),
                                    s.package.get_constraints());
    Vector<double> diff(Mpp.get_vector());
    diff -= s.lumped.get_Mpp().get_vector();
    report.check(diff.linfty_norm() < 1e-13 * Mpp.get_vector().linfty_norm(),
                 s.name + ": Mpp equals fe::assemble_mass_phiphi_lumped", "max diff " + sci(diff.linfty_norm()));
}


//! A = A0 + beta Mpp mixes a SparseMatrix and a DiagonalMatrix in a LinearCombination.
void check_lumped_operator(CheckReport& report, Systems& s)
{
    const unsigned n = s.n_dofs();
    const Vector<double> u = test_point(n);
    s.lumped.assemble_nonlinear_term(u);
    const auto& A0  = s.lumped.get_A0();
    const auto& Mpp = s.lumped.get_Mpp().get_vector();
    const auto  A   = s.lumped.get_operator_A(beta);

    Vector<double> Au(n), ref(n), tmp(u);
    A.vmult(Au, u);
    A0.vmult(ref, u);
    tmp.scale(Mpp);
    ref.add(beta, tmp);
    const double scale = ref.linfty_norm();

    Vector<double> diff(Au), ATu(n), diag = A.diagonal();
    diff -= ref;
    A.Tvmult(ATu, u);
    ATu -= Au;
    for (unsigned i = 0; i < n; ++i) {
        diag[i] -= A0.diag_element(i) + beta * Mpp[i];
    }
    report.check(std::max({diff.linfty_norm(), ATu.linfty_norm(), diag.linfty_norm()}) < 1e-12 * scale,
                 s.name + ": A = A0 + beta Mpp (vmult, Tvmult, diagonal)",
                 "vmult " + sci(diff.linfty_norm()) + ", Tvmult " + sci(ATu.linfty_norm()) + ", diagonal "
                 + sci(diag.linfty_norm()));
}


//! u(x) = prod_k cos(pi x_k / (2R)), which satisfies the Dirichlet condition on [-R, R]^dim.
class CosineBump : public Function<dim>
{
public:
    double value(const Point<dim>& p, unsigned) const override
    {
        double out = 1.0;
        for (unsigned k = 0; k < dim; ++k) {
            out *= std::cos(0.5 * std::numbers::pi * p[k] / radius);
        }
        return out;
    }
};

//! The lumped and the consistent energy of a smooth function differ by O(h^2).
void check_energy_convergence(CheckReport& report, const Discretization& disc, unsigned min_level, unsigned max_level)
{
    std::vector<double> diffs;
    for (unsigned level = min_level; level <= max_level; ++level) {
        Systems s(disc, BoundaryCondition::DIRICHLET, level);
        Vector<double> u(s.n_dofs());
        VectorTools::interpolate(s.package.get_mapping(), s.package.get_dofs(), CosineBump(), u);
        s.package.distribute(u);
        diffs.push_back(std::abs(energy(s.lumped, u) - energy(s.consistent, u)));
    }

    const double rate = std::log2(diffs[diffs.size() - 2] / diffs.back());
    std::string detail = "|E_L - E| =";
    for (double d : diffs) {
        detail += " " + sci(d);
    }
    report.check(rate > 1.8, disc.name() + ": lumped vs consistent energy converges at O(h^2)",
                 detail + ", rate " + sci(rate));
}


// ---------------------------------------------------------------------------------------------
// Functional, oracles and iterations on the lumped system
// ---------------------------------------------------------------------------------------------

SolverOptions solver_options(double tol_inner_res)
{
    SolverOptions options{};
    options.max_inner     = 2000;
    options.tol_inner     = 1e-12;
    options.tol_inner_res = tol_inner_res;
    options.solver        = SolverMethod::CG;
    options.precond       = Precondition::DIAGONAL;
    return options;
}

using LumpedFunctional = GrossPitaevskiiFunctional<GrossPitaevskiiLumpedSystem<dim>>;
static_assert(std::is_same_v<LumpedFunctional::InverseM, DiagonalInverse>);
static_assert(std::is_same_v<GrossPitaevskiiFunctional<GrossPitaevskiiSystem<dim>>::InverseM, InverseOpType>);

void check_lumped_functional(CheckReport& report, Systems& s)
{
    const std::string name = s.name + " functional";
    const unsigned n = s.n_dofs();
    LumpedFunctional func(s.lumped, beta, solver_options(1e-2));

    Vector<double> x = test_point(n), d = test_direction(n);
    s.package.distribute(x);
    s.package.distribute(d);
    func.update(x);

    Vector<double> Mx(n), y(n);
    func.get_M().vmult(Mx, x);
    func.get_M_inv().vmult(y, Mx);
    y -= x;
    report.check(y.linfty_norm() < 1e-14 * x.linfty_norm() && func.get_M_inv().control().last_step() == 0,
                 name + ": M_inv is the exact inverse of M_L", "max diff " + sci(y.linfty_norm()));

    // Central differences: the error drops ~100x from h = 1e-2 to 1e-3, unless the gradient is inconsistent
    Vector<double> g(n);
    func.gradient(x, g);
    auto value = [&](const Vector<double>& z) { func.update(z); return func.value(z); };
    const double error_1 = std::abs(central_difference(value, x, d, 1e-2) - g * d) / std::abs(g * d);
    const double error_2 = std::abs(central_difference(value, x, d, 1e-3) - g * d) / std::abs(g * d);
    report.check(error_2 < 1e-5 && error_1 / error_2 > 50, name + ": gradient() (finite differences)",
                 "rel. error " + sci(error_2) + ", ratio " + sci(error_1 / error_2));
    func.update(x);

    Vector<double> b(n), Ay(n);
    func.get_A().vmult(b, x);
    func.get_A_inv().vmult(y, b);
    func.get_A().vmult(Ay, y);
    Ay -= b;
    report.check(Ay.l2_norm() < 1e-8 * b.l2_norm(), name + ": A_inv solves A y = b (CG, Jacobi)",
                 "rel. residual " + sci(Ay.l2_norm() / b.l2_norm()) + ", "
                 + std::to_string(func.get_A_inv().control().last_step()) + " iterations");
}

//! The oracle gradient is tangent, measured in the metric of its kind, and equal to the gradient of the iteration.
template <typename Oracle, typename Iteration>
void check_oracle(CheckReport& report, const std::string& name, LumpedFunctional& func, const Vector<double>& x,
                  const SolverOptions& options)
{
    Oracle oracle(func, options);
    oracle.update(x);
    Vector<double> g(x.size()), g_iter(x.size());
    oracle.gradient(x, g);

    // Metrics of the functional, independent of the oracle
    const OperatorMetric M_metric(func.get_M(), MetricKind::MASS);
    const OperatorMetric A_metric(func.get_A(), MetricKind::ENERGY_ADAPTIVE);
    const EuclideanMetric F_metric;

    const double tangent = std::abs(M_metric.inner(x, g)) / g.l2_norm();
    report.check(tangent < 1e-8, name + ": gradient is tangent (x^T M g = 0)", "|x^T M g| / |g| = " + sci(tangent));

    const MetricBase& expected = [&]() -> const MetricBase& {
        switch (Oracle::metric_t) {
        case MetricKind::MASS:            return M_metric;
        case MetricKind::ENERGY_ADAPTIVE: return A_metric;
        default:                          return F_metric;
        }
    }();
    const double norm_diff = std::abs(oracle.metric().norm(g) - expected.norm(g));
    report.check(oracle.metric().kind() == Oracle::metric_t && norm_diff < 1e-12 * expected.norm(g),
                 name + ": metric() matches the metric of its kind", "norm difference " + sci(norm_diff));

    Iteration iteration(func, std::make_shared<const Vector<double>>(x), options);
    iteration.gradient(g_iter);
    g_iter -= g;
    report.check(g_iter.linfty_norm() < 1e-10 * g.linfty_norm(), name + ": iteration gradient equals oracle gradient",
                 "max diff " + sci(g_iter.linfty_norm()));
}

void check_lumped_oracles(CheckReport& report, Systems& s)
{
    using System = GrossPitaevskiiLumpedSystem<dim>;
    const SolverOptions options = solver_options(1e-6);
    LumpedFunctional func(s.lumped, beta, options);

    // On the mass sphere x^T M x = 1, as the tangent check assumes
    Vector<double> x = make_vector(s.n_dofs(), [](unsigned i) { return std::sin(0.37 * i) + 1.2; });
    Vector<double> Mx(s.n_dofs());
    s.package.distribute(x);
    s.lumped.get_M().vmult(Mx, x);
    x /= std::sqrt(x * Mx);

    check_oracle<MassOracle<System>, MassIteration<System>>(report, s.name + " MassOracle", func, x, options);
    check_oracle<EnergyOracle<System>, EnergyIteration<System>>(report, s.name + " EnergyOracle", func, x, options);
    check_oracle<FrobeniusOracle<System>, FrobeniusIteration<System>>(report, s.name + " FrobeniusOracle", func, x,
                                                                      options);
}

} // namespace


int main()
{
    return run_tests([](CheckReport& report) {
        check_lumpability(report);

        for (const auto& disc : discretizations) {
            // Neumann boundary: no constrained rows, so every row can be compared
            Systems s(disc, BoundaryCondition::NEUMANN, 4);
            const Quadrature<dim> gauss = disc.gauss();
            check_lumped_mass(report, s, gauss);
            check_lumped_A0(report, s, gauss);
            check_lumped_Mpp(report, s, gauss);
            check_lumped_operator(report, s);
        }
        check_energy_convergence(report, {MeshKind::QUADRILATERAL, 1}, 3, 6);
        check_energy_convergence(report, {MeshKind::SIMPLEX, 1}, 3, 6);

        for (const auto& disc : discretizations) {
            Systems s(disc, BoundaryCondition::DIRICHLET, 4);
            check_lumped_functional(report, s);
            check_lumped_oracles(report, s);
        }
    });
}
