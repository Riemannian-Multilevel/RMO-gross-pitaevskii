//
// Tests for mass lumping in the Gross-Pitaevskii discretization.
//
// 1. Nodal quadrature (fe::make_nodal_quadrature): weights for FE_Q, P1 and P2-bubble elements,
//    rejection of FE_SimplexP(2).
// 2. Lumped matrices (fe/assemble.h): row sums of the consistent mass matrix, positivity,
//    independence of the quadrature rule, structure of A0 = S + M_{V,L}.
// 3. GrossPitaevskiiLumpedSystem: nonlinear term, mixed sparse/diagonal operators
//    (LinearCombination), and consistency of energy and gradient (finite differences).
// 4. Convergence: the lumped and consistent energies of a smooth function differ by O(h^2).
// 5. Zero-weight components of LinearCombination are skipped.
// 6. GrossPitaevskiiFunctional on the lumped system: exact M^{-1}, value, gradient, A^{-1}.
// 7. Oracles and iterations on the lumped system: tangent gradients, oracle == iteration.
//
// Each check prints PASS/FAIL; the exit code is the number of failed checks (capped at 1).
//
#include <rmo/gpe/gpe.h>
#include <rmo/gpe/iteration.h>
#include <rmo/gpe/oracle.h>

#include <deal.II/base/function.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace rmo;
using namespace rmo::gpe;
using namespace dealii;

namespace
{
constexpr int dim = 2;
unsigned n_failed = 0;

void check(bool ok, const std::string& name, const std::string& detail = "")
{
    std::cout << (ok ? "PASS  " : "FAIL  ") << name;
    if (!detail.empty()) {
        std::cout << "  (" << detail << ")";
    }
    std::cout << "\n";
    n_failed += !ok;
}

std::string fmt_double(double x)
{
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(2) << x;
    return ss.str();
}

GPE_Options make_options(MeshKind mesh, int degree, BoundaryCondition bc)
{
    GPE_Options options{};
    options.dimension       = dim;
    options.degree          = degree;
    options.radius          = 4.0;
    options.beta            = 100.0;
    options.order           = Ordering::DEFAULT;
    options.bc              = bc;
    options.mesh_kind       = mesh;
    options.potential       = Potential::SQUARE;
    return options;
}

std::string config_name(MeshKind mesh, int degree)
{
    return (mesh == MeshKind::SIMPLEX ? (degree > 1 ? "P" + std::to_string(degree) + "-bubble"
                                                    : "P" + std::to_string(degree))
                                      : "Q" + std::to_string(degree));
}

//! Sorted weights of a quadrature rule, divided by the reference cell volume.
std::vector<double> sorted_weights(const Quadrature<dim>& q, double ref_volume)
{
    std::vector<double> w(q.get_weights());
    for (auto& wi : w) {
        wi /= ref_volume;
    }
    std::ranges::sort(w);
    return w;
}

// ---------------------------------------------------------------------------------------------
// 1. Nodal quadrature
// ---------------------------------------------------------------------------------------------
void test_nodal_quadrature()
{
    // FE_Q(p): the nodal rule is the Gauss-Lobatto rule (matched point by point)
    for (unsigned p = 1; p <= 3; ++p) {
        const FE_Q<dim> fe(p);
        const auto nodal = fe::make_nodal_quadrature(fe);
        const QGaussLobatto<dim> gll(p + 1);

        double err = (nodal.size() == gll.size()) ? 0.0 : 1.0;
        for (unsigned i = 0; i < nodal.size() && err < 1.0; ++i) {
            bool found = false;
            for (unsigned j = 0; j < gll.size(); ++j) {
                if (nodal.point(i).distance(gll.point(j)) < 1e-12) {
                    err   = std::max(err, std::abs(nodal.weight(i) - gll.weight(j)));
                    found = true;
                }
            }
            err = found ? err : 1.0;
        }
        check(err < 1e-14, "nodal quadrature FE_Q(" + std::to_string(p) + ") == QGaussLobatto(" +
              std::to_string(p + 1) + ")", "max weight diff " + fmt_double(err));
    }

    // FE_SimplexP(1): vertex weights |K|/(d+1)
    {
        const auto w = sorted_weights(fe::make_nodal_quadrature(FE_SimplexP<dim>(1)), 0.5);
        const bool ok = w.size() == 3 && std::ranges::all_of(w, [](double x) { return std::abs(x - 1.0/3) < 1e-14; });
        check(ok, "nodal quadrature FE_SimplexP(1): weights |K|/3");
    }

    // FE_SimplexP_Bubbles(2): |K| * (1/20 vertices, 2/15 edge midpoints, 9/20 centroid)
    {
        const auto w = sorted_weights(fe::make_nodal_quadrature(FE_SimplexP_Bubbles<dim>(2)), 0.5);
        const std::vector<double> expected = {1./20, 1./20, 1./20, 2./15, 2./15, 2./15, 9./20};
        bool ok = w.size() == expected.size();
        for (unsigned i = 0; ok && i < w.size(); ++i) {
            ok = std::abs(w[i] - expected[i]) < 1e-14;
        }
        check(ok, "nodal quadrature FE_SimplexP_Bubbles(2): weights |K|(1/20, 2/15, 9/20)");
    }

    // FE_SimplexP(2) is not lumpable (zero vertex weights)
    {
        bool thrown = false;
        try {
            fe::make_nodal_quadrature(FE_SimplexP<dim>(2));
        }
        catch (const std::exception&) {
            thrown = true;
        }
        check(thrown, "nodal quadrature FE_SimplexP(2) is rejected");
    }
}

// ---------------------------------------------------------------------------------------------
// 2./3. Lumped matrices and GrossPitaevskiiLumpedSystem, for one element/mesh configuration
// ---------------------------------------------------------------------------------------------
void test_lumped_system(MeshKind mesh, int degree, unsigned n_levels)
{
    const std::string name = config_name(mesh, degree);
    const potential::Square<dim> V;

    // Neumann: no constraints, so that all rows can be compared
    GrossPitaevskiiPackage<dim> package(make_options(mesh, degree, BoundaryCondition::NEUMANN), n_levels);
    const auto& dofs        = package.get_dofs();
    const auto& mapping     = package.get_mapping();
    const auto& constraints = package.get_constraints();
    const auto  n           = package.n_dofs();
    const auto  nodal       = fe::make_nodal_quadrature(dofs.get_fe());
    const QGaussSimplex<dim> q_gauss_simplex(degree + 2);
    const QGauss<dim>        q_gauss(degree + 1);
    const Quadrature<dim>&   gauss = (mesh == MeshKind::SIMPLEX) ? static_cast<const Quadrature<dim>&>(q_gauss_simplex)
                                                                 : static_cast<const Quadrature<dim>&>(q_gauss);

    auto consistent = package.system(V);
    auto lumped     = package.system<GrossPitaevskiiLumpedSystem<dim>>(V);
    const auto& M_L = lumped.get_M().get_vector();

    // Row sums of the consistent mass matrix
    {
        const auto& M = consistent.get_M();
        double err = 0.0;
        for (unsigned i = 0; i < n; ++i) {
            double row_sum = 0.0;
            for (auto it = M.begin(i); it != M.end(i); ++it) {
                row_sum += it->value();
            }
            err = std::max(err, std::abs(row_sum - M_L[i]) / M_L[i]);
        }
        check(err < 1e-12, name + ": M_L equals the row sums of M", "max rel. diff " + fmt_double(err));
    }

    // Positivity and total mass |Omega| = (2R)^dim
    {
        const double R = 4.0;
        const double total = M_L.mean_value() * n;
        check(*std::ranges::min_element(M_L) > 0, name + ": M_L is positive");
        check(std::abs(total - std::pow(2*R, dim)) < 1e-10 * total, name + ": sum of M_L equals |Omega|",
              "sum " + std::to_string(total));
    }

    // Gauss and nodal rules give the same lumped mass matrix
    {
        DiagonalMatrix<Vector<double>> M_gauss;
        M_gauss.get_vector().reinit(n);
        fe::assemble_mass_lumped(M_gauss, dofs, gauss, mapping, constraints);

        Vector<double> diff(M_gauss.get_vector());
        diff -= M_L;
        check(diff.linfty_norm() < 1e-14 * M_L.linfty_norm(), name + ": M_L independent of the quadrature (Gauss vs nodal)",
              "max diff " + fmt_double(diff.linfty_norm()));
    }

    // A0 = S + M_{V,L}: off-diagonal part equals S, diagonal part equals the lumped potential term
    {
        const auto& A0 = lumped.get_A0();
        SparseMatrix<double> S(A0.get_sparsity_pattern());
        fe::assemble_stiffness(S, dofs, gauss, mapping, constraints);

        DiagonalMatrix<Vector<double>> M_VL;
        M_VL.get_vector().reinit(n);
        fe::assemble_mass_lumped_weighted(M_VL, V, dofs, nodal, mapping, constraints);

        double err_offdiag = 0.0, err_diag = 0.0;
        for (unsigned i = 0; i < n; ++i) {
            for (auto it = A0.begin(i); it != A0.end(i); ++it) {
                const unsigned j = it->column();
                const double S_ij = S.el(i, j);
                if (i == j) {
                    err_diag = std::max(err_diag, std::abs(it->value() - S_ij - M_VL.get_vector()[i]));
                } else {
                    err_offdiag = std::max(err_offdiag, std::abs(it->value() - S_ij));
                }
            }
        }
        const double scale = A0.linfty_norm();
        check(err_offdiag < 1e-12 * scale && err_diag < 1e-12 * scale, name + ": A0 = S + M_{V,L}",
              "off-diag " + fmt_double(err_offdiag) + ", diag " + fmt_double(err_diag));
    }

    // Nonlinear term equals the cell-loop assembly
    Vector<double> u(n);
    for (unsigned i = 0; i < n; ++i) {
        u[i] = std::sin(0.37 * i) + 0.2;
    }
    lumped.assemble_nonlinear_term(u);
    {
        DiagonalMatrix<Vector<double>> Mpp_ref;
        Mpp_ref.get_vector().reinit(n);
        fe::assemble_mass_phiphi_lumped(Mpp_ref, u, dofs, nodal, mapping, constraints);

        Vector<double> diff(Mpp_ref.get_vector());
        diff -= lumped.get_Mpp().get_vector();
        check(diff.linfty_norm() < 1e-13 * Mpp_ref.get_vector().linfty_norm(),
              name + ": Mpp equals fe::assemble_mass_phiphi_lumped", "max diff " + fmt_double(diff.linfty_norm()));
    }

    // Mixed operator A = A0 + beta Mpp (SparseMatrix + DiagonalMatrix)
    {
        const double beta = 100.0;
        const auto A = lumped.get_operator_A(beta);
        const auto& mpp = lumped.get_Mpp().get_vector();

        Vector<double> Au(n), ref(n), tmp(n), ATu(n);
        A.vmult(Au, u);
        lumped.get_A0().vmult(ref, u);
        tmp = u;
        tmp.scale(mpp);
        ref.add(beta, tmp);
        Vector<double> diff(Au);
        diff -= ref;
        check(diff.linfty_norm() < 1e-12 * ref.linfty_norm(), name + ": LinearCombination<Sparse, Diagonal>::vmult",
              "max diff " + fmt_double(diff.linfty_norm()));

        A.Tvmult(ATu, u);
        ATu -= Au;
        check(ATu.linfty_norm() < 1e-12 * ref.linfty_norm(), name + ": LinearCombination::Tvmult equals vmult (symmetric)");

        Vector<double> diag = A.diagonal();
        for (unsigned i = 0; i < n; ++i) {
            diag[i] -= lumped.get_A0().diag_element(i) + beta * mpp[i];
        }
        check(diag.linfty_norm() < 1e-12 * ref.linfty_norm(), name + ": LinearCombination::diagonal()");

        const auto M_op = lumped.get_operator_M(2.0);
        Vector<double> Mu(n);
        M_op.vmult(Mu, u);
        tmp = u;
        tmp.scale(M_L);
        Mu.add(-2.0, tmp);
        check(Mu.linfty_norm() < 1e-14 * M_L.linfty_norm(), name + ": get_operator_M(w) applies w M_L");
    }
}

// ---------------------------------------------------------------------------------------------
// 3. Energy and gradient consistency: grad E(x) = A0 x + beta Mpp(x) x
// ---------------------------------------------------------------------------------------------
void test_gradient_consistency(MeshKind mesh, int degree, unsigned n_levels)
{
    const std::string name = config_name(mesh, degree);
    const double beta = 100.0;

    GrossPitaevskiiPackage<dim> package(make_options(mesh, degree, BoundaryCondition::DIRICHLET), n_levels);
    auto lumped = package.system<GrossPitaevskiiLumpedSystem<dim>>(potential::Square<dim>());
    const auto n = package.n_dofs();

    auto energy = [&](const Vector<double>& x) {
        lumped.assemble_nonlinear_term(x);
        const auto A = lumped.get_operator_A(0.25 * beta, 0.5);
        Vector<double> Ax(n);
        A.vmult(Ax, x);
        return x * Ax;
    };

    Vector<double> x(n), d(n), g(n);
    for (unsigned i = 0; i < n; ++i) {
        x[i] = std::sin(0.37 * i) + 0.2;
        d[i] = std::cos(1.3 * i);
    }
    package.distribute(x);
    package.distribute(d);

    lumped.assemble_nonlinear_term(x);
    lumped.get_operator_A(beta).vmult(g, x);
    const double gd = g * d;

    std::vector<double> errors;
    for (double h : {1e-2, 1e-3}) {
        Vector<double> xp(x), xm(x);
        xp.add(h, d);
        xm.add(-h, d);
        errors.push_back(std::abs((energy(xp) - energy(xm)) / (2*h) - gd) / std::abs(gd));
    }
    // Central differences: error O(h^2), i.e. a factor ~100 per decade of h. An inconsistent
    // gradient leaves an O(1) error that does not decrease with h.
    const double ratio = errors[0] / errors[1];
    check(errors[1] < 1e-5 && ratio > 50, name + ": gradient of the lumped energy (finite differences)",
          "rel. error " + fmt_double(errors[1]) + ", ratio " + fmt_double(ratio));
}

// ---------------------------------------------------------------------------------------------
// 4. Lumped vs consistent energy of a smooth function: O(h^2)
// ---------------------------------------------------------------------------------------------
void test_energy_convergence(MeshKind mesh, int degree, unsigned min_level, unsigned max_level)
{
    const std::string name = config_name(mesh, degree);
    const double beta = 100.0, R = 4.0;
    const potential::Square<dim> V;

    // u(x) = prod_d cos(pi x_d / (2R)), zero on the boundary of [-R, R]^dim
    class Scaled : public Function<dim>
    {
    public:
        explicit Scaled(double R) : R(R) {}
        double value(const Point<dim>& p, unsigned) const override
        {
            double out = 1.0;
            for (unsigned k = 0; k < dim; ++k) {
                out *= std::cos(0.5 * std::numbers::pi * p[k] / R);
            }
            return out;
        }
    private:
        double R;
    } u_exact(R);

    std::vector<double> diffs;
    for (unsigned level = min_level; level <= max_level; ++level) {
        GrossPitaevskiiPackage<dim> package(make_options(mesh, degree, BoundaryCondition::DIRICHLET), level);
        Vector<double> u(package.n_dofs());
        VectorTools::interpolate(package.get_mapping(), package.get_dofs(), u_exact, u);
        package.distribute(u);

        auto value = [&](auto& system) {
            system.assemble_nonlinear_term(u);
            Vector<double> Au(u.size());
            system.get_operator_A(0.25 * beta, 0.5).vmult(Au, u);
            return u * Au;
        };
        auto consistent = package.system(V);
        auto lumped     = package.system<GrossPitaevskiiLumpedSystem<dim>>(V);
        diffs.push_back(std::abs(value(lumped) - value(consistent)));
    }

    const double rate = std::log2(diffs[diffs.size() - 2] / diffs.back());
    std::string detail = "|E_L - E| =";
    for (double d : diffs) {
        detail += " " + fmt_double(d);
    }
    detail += ", rate " + fmt_double(rate);
    check(rate > 1.8, name + ": lumped vs consistent energy converges at O(h^2)", detail);
}

// ---------------------------------------------------------------------------------------------
// 5. Zero-weight components are skipped: A(beta = 0) = A0
// ---------------------------------------------------------------------------------------------
template <typename System>
void check_zero_weight(System& system, const std::string& name)
{
    const auto n = system.n_dofs();
    Vector<double> x(n), Ax(n), A0x(n);
    for (unsigned i = 0; i < n; ++i) {
        x[i] = std::sin(0.37 * i) + 0.2;
    }
    system.assemble_nonlinear_term(x);
    system.get_operator_A(0.0).vmult(Ax, x);
    system.get_A0().vmult(A0x, x);
    Ax -= A0x;
    check(Ax.linfty_norm() == 0.0, name + ": zero-weight component is skipped (A(0) = A0)");
}

void test_zero_weight()
{
    GrossPitaevskiiPackage<dim> package(make_options(MeshKind::QUADRILATERAL, 1, BoundaryCondition::DIRICHLET), 4);
    auto consistent = package.system(potential::Square<dim>());
    auto lumped     = package.system<GrossPitaevskiiLumpedSystem<dim>>(potential::Square<dim>());
    check_zero_weight(consistent, "consistent system");
    check_zero_weight(lumped, "lumped system");
}

// ---------------------------------------------------------------------------------------------
// 6. GrossPitaevskiiFunctional on the lumped system
// ---------------------------------------------------------------------------------------------
void test_lumped_functional(MeshKind mesh, int degree, unsigned n_levels)
{
    const std::string name = config_name(mesh, degree) + " functional";
    const double beta = 100.0;

    GrossPitaevskiiPackage<dim> package(make_options(mesh, degree, BoundaryCondition::DIRICHLET), n_levels);
    auto system = package.system<GrossPitaevskiiLumpedSystem<dim>>(potential::Square<dim>());
    const auto n = package.n_dofs();

    SolverOptions options{};
    options.max_inner     = 2000;
    options.tol_inner     = 1e-12;
    options.tol_inner_res = 1e-2;
    options.solver        = SolverMethod::CG;
    options.precond       = Precondition::DIAGONAL;

    using Functional = GrossPitaevskiiFunctional<GrossPitaevskiiLumpedSystem<dim>>;
    static_assert(std::is_same_v<Functional::InverseM, DiagonalInverse>);
    static_assert(std::is_same_v<GrossPitaevskiiFunctional<GrossPitaevskiiSystem<dim>>::InverseM, InverseOpType>);
    Functional func(system, beta, options);

    Vector<double> x(n), d(n);
    for (unsigned i = 0; i < n; ++i) {
        x[i] = std::sin(0.37 * i) + 0.2;
        d[i] = std::cos(1.3 * i);
    }
    package.distribute(x);
    package.distribute(d);
    func.update(x);

    // M^{-1} is exact and takes no iterations
    {
        Vector<double> Mx(n), y(n);
        func.get_M().vmult(Mx, x);
        func.get_M_inv().vmult(y, Mx);
        y -= x;
        check(y.linfty_norm() < 1e-14 * x.linfty_norm() && func.get_M_inv().control().last_step() == 0,
              name + ": M_inv is the exact inverse of M_L", "max diff " + fmt_double(y.linfty_norm()));
    }

    // value(x) = 1/2 x^T A0 x + beta/4 sum_i (M_L)_ii x_i^4
    {
        Vector<double> A0x(n);
        system.get_A0().vmult(A0x, x);
        double ref = 0.5 * (x * A0x);
        for (unsigned i = 0; i < n; ++i) {
            ref += 0.25 * beta * system.get_M().get_vector()[i] * std::pow(x[i], 4);
        }
        const double val = func.value(x);
        check(std::abs(val - ref) < 1e-12 * std::abs(ref), name + ": value() equals the lumped energy",
              "rel. diff " + fmt_double(std::abs(val - ref) / std::abs(ref)));
    }

    // gradient(x) against central differences of value()
    {
        Vector<double> g(n);
        func.gradient(x, g);
        const double gd = g * d;

        std::vector<double> errors;
        for (double h : {1e-2, 1e-3}) {
            Vector<double> xp(x), xm(x);
            xp.add(h, d);
            xm.add(-h, d);
            func.update(xp);
            const double Ep = func.value(xp);
            func.update(xm);
            const double Em = func.value(xm);
            errors.push_back(std::abs((Ep - Em) / (2*h) - gd) / std::abs(gd));
        }
        func.update(x);
        const double ratio = errors[0] / errors[1];
        check(errors[1] < 1e-5 && ratio > 50, name + ": gradient() (finite differences)",
              "rel. error " + fmt_double(errors[1]) + ", ratio " + fmt_double(ratio));
    }

    // A^{-1} solves with the mixed sparse/diagonal operator
    {
        Vector<double> b(n), y(n), Ay(n);
        func.get_A().vmult(b, x);
        func.get_A_inv().vmult(y, b);
        func.get_A().vmult(Ay, y);
        Ay -= b;
        check(Ay.l2_norm() < 1e-8 * b.l2_norm(), name + ": A_inv solves A y = b (CG, Jacobi)",
              "rel. residual " + fmt_double(Ay.l2_norm() / b.l2_norm()) + ", " +
              std::to_string(func.get_A_inv().control().last_step()) + " iterations");
    }
}

// ---------------------------------------------------------------------------------------------
// 7. Oracles and iterations on the lumped system
// ---------------------------------------------------------------------------------------------
template <typename Oracle, typename Iteration>
void check_oracle(const std::string& name, typename Oracle::Functional& func, const Vector<double>& x,
                  const SolverOptions& options, bool exact_solve)
{
    Oracle oracle(func, options);
    oracle.update(x);
    Vector<double> g(x.size()), g_iter(x.size()), Mg(x.size());
    const GradInfo info = oracle.gradient(x, g);

    // Riemannian gradients are tangent to the mass sphere: x^T M g = 0 (x normalized)
    oracle.get_M().vmult(Mg, g);
    const double tangent = std::abs(x * Mg) / g.l2_norm();
    check(tangent < 1e-8, name + ": gradient is tangent (x^T M g = 0)", "|x^T M g| / |g| = " + fmt_double(tangent));

    if (exact_solve) {
        check(info.num_iter == 0, name + ": exact M^{-1}, no solver iterations");
    }

    // The iteration interface computes the same gradient
    Iteration iteration(func, std::make_shared<const Vector<double>>(x), options);
    iteration.gradient(g_iter);
    g_iter -= g;
    check(g_iter.linfty_norm() < 1e-10 * g.linfty_norm(), name + ": iteration gradient equals oracle gradient",
          "max diff " + fmt_double(g_iter.linfty_norm()));
}

void test_lumped_oracles(MeshKind mesh, int degree, unsigned n_levels)
{
    const std::string name = config_name(mesh, degree);
    using System = GrossPitaevskiiLumpedSystem<dim>;

    GrossPitaevskiiPackage<dim> package(make_options(mesh, degree, BoundaryCondition::DIRICHLET), n_levels);
    System system = package.system<GrossPitaevskiiLumpedSystem<dim>>(potential::Square<dim>());
    const auto n = package.n_dofs();

    SolverOptions options{};
    options.max_inner     = 2000;
    options.tol_inner     = 1e-12;
    options.tol_inner_res = 1e-6;
    options.solver        = SolverMethod::CG;
    options.precond       = Precondition::DIAGONAL;
    GrossPitaevskiiFunctional<System> func(system, 100.0, options);

    // Point on the mass sphere x^T M_L x = 1
    Vector<double> x(n), Mx(n);
    for (unsigned i = 0; i < n; ++i) {
        x[i] = std::sin(0.37 * i) + 1.2;
    }
    package.distribute(x);
    system.get_M().vmult(Mx, x);
    x /= std::sqrt(x * Mx);

    check_oracle<MassOracle<System>, MassIteration<System>>(name + " MassOracle", func, x, options, true);
    check_oracle<EnergyOracle<System>, EnergyIteration<System>>(name + " EnergyOracle", func, x, options, false);
    check_oracle<FrobeniusOracle<System>, FrobeniusIteration<System>>(name + " FrobeniusOracle", func, x, options, false);
}

} // namespace


int main()
{
    try {
        test_nodal_quadrature();

        for (const auto& [mesh, degree] : {std::pair{MeshKind::QUADRILATERAL, 1}, std::pair{MeshKind::QUADRILATERAL, 2},
                                           std::pair{MeshKind::SIMPLEX, 1},       std::pair{MeshKind::SIMPLEX, 2}}) {
            test_lumped_system(mesh, degree, 4);
            test_gradient_consistency(mesh, degree, 4);
        }

        test_energy_convergence(MeshKind::QUADRILATERAL, 1, 3, 6);
        test_energy_convergence(MeshKind::SIMPLEX, 1, 3, 6);

        test_zero_weight();
        test_lumped_functional(MeshKind::QUADRILATERAL, 1, 4);
        test_lumped_functional(MeshKind::SIMPLEX, 2, 4);

        test_lumped_oracles(MeshKind::QUADRILATERAL, 1, 4);
        test_lumped_oracles(MeshKind::SIMPLEX, 2, 4);
    }
    catch (const std::exception& exc) {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return 1;
    }

    std::cout << (n_failed ? std::to_string(n_failed) + " check(s) failed" : "all checks passed") << std::endl;
    return n_failed ? 1 : 0;
}
