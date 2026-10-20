/**
 * @file
 * @brief Tests of the coarse model with the density lifting, DensityCoarseOracle (tex/drafts/positive_multilevel.tex),
 * on a lumped Q1 hierarchy (levels 4 -> 5, radius 10, Neumann boundary), with the coarse model built by a
 * CoarseOracleBase from M-metric tilt oracles, the unit-sphere point transfer (injection and normalization, positive
 * for Q1) and MassProjectionTransport:
 *
 * - \f$ q(y) = f_H(y) \f$, and \f$ q(z) = f_H(z) - \sum_j d_j w_j (z_j^2 - y_j^2) / (2 y_j) \f$ (the shifted-potential
 *   form against the definition with the density lifting);
 * - the directional derivative against central differences of \f$ q \f$;
 * - for every DensityGradient: the gradient is tangent, and \f$ \langle \grad q, v \rangle = \mathrm{D}q(z)[v] \f$ in
 *   the oracle's metric for tangent \f$ v \f$;
 * - coherence: \f$ \grad_M q(y) = \mathcal{R} \grad_M f_h(x) \f$;
 * - safeguard: with a large correction \f$ w \f$ (coarse functional with \f$ \beta \f$ 100 times the fine one), the
 *   ENERGY_MODEL_SHIFT step \f$ \eta = -\grad q \f$ satisfies \f$ \eta > -z \f$; the shift \f$ \sigma \f$, the
 *   ENERGY_MODEL step bound, and the number of updates at which \f$ \hat A(z) \f$ has a non-positive row sum are
 *   reported.
 */
#include "check.h"

#include <rmo/fe/interpolate.h>
#include <rmo/gpe/gpe.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/oracle.h>
#include <rmo/gpe/oracle_coarse.h>
#include <rmo/gpe/transport.h>

#include <deal.II/dofs/dof_tools.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

using namespace rmo;
using namespace rmo::gpe;
using namespace rmo::test;
using namespace dealii;

namespace
{
constexpr int dim = 2;
using System     = GrossPitaevskiiLumpedSystem<dim>;
using Functional = GrossPitaevskiiFunctional<System>;
using MassMatrix = System::MassMatrix;
using Operator   = Functional::Operator;

GPE_Options gpe_options()
{
    GPE_Options options{};
    options.dimension    = dim;
    options.degree       = 1;
    options.radius       = 10.0;
    options.beta         = 100.0;
    options.order        = Ordering::DEFAULT;
    options.bc           = BoundaryCondition::NEUMANN;
    options.mesh_kind    = MeshKind::QUADRILATERAL;
    options.potential    = Potential::SQUARE;
    options.mass_lumping = true;
    return options;
}

SolverOptions solver_options()
{
    SolverOptions options{};
    options.max_inner     = 5000;
    options.tol_inner     = 1e-14;
    options.tol_inner_res = 1e-12;  // accurate energy-adaptive gradients for the metric checks
    options.solver        = SolverMethod::CG;
    options.precond       = Precondition::DIAGONAL;
    return options;
}

struct Level
{
    explicit Level(unsigned n_levels)
        : package(gpe_options(), n_levels)
        , system(package.system<System>(V))
    {}

    [[nodiscard]] const MassMatrix& D() const { return system.get_M(); }
    [[nodiscard]] const Vector<double>& d() const { return system.get_M().get_vector(); }

    const potential::Square<dim> V{};
    GrossPitaevskiiPackage<dim> package;
    System system;
};

double d_inner(const Vector<double>& d, const Vector<double>& a, const Vector<double>& b)
{
    double s = 0;
    for (unsigned i = 0; i < a.size(); ++i) {
        s += d[i] * a[i] * b[i];
    }
    return s;
}

void normalize(const Vector<double>& d, Vector<double>& x) { x /= std::sqrt(d_inner(d, x, x)); }

//! Positive point with Gaussian tails and a smooth relative perturbation, normalized
Vector<double> positive_point(const Level& level, double width, double phase)
{
    std::vector<Point<dim>> points(level.package.n_dofs());
    DoFTools::map_dofs_to_support_points(level.package.get_mapping(), level.package.get_dofs(), points);
    Vector<double> x(points.size());
    for (unsigned i = 0; i < points.size(); ++i) {
        const Point<dim>& p = points[i];
        x[i] = std::exp(-p.square() / width) * (1 + 0.2 * std::sin(p[0] + 2 * p[1] + phase));
    }
    normalize(level.d(), x);
    return x;
}

//! Tangent vector at x: a smooth relative perturbation x * s, D-orthogonally projected
Vector<double> tangent(const Level& level, const Vector<double>& x, double freq)
{
    Vector<double> v(x.size());
    for (unsigned i = 0; i < x.size(); ++i) {
        v[i] = x[i] * std::cos(freq * i);
    }
    v.add(-d_inner(level.d(), x, v), x);
    return v;
}

//! Two-level setup: fine and coarse functionals, M-metric tilt oracles, transfers and the coarse model base
struct TwoLevel
{
    explicit TwoLevel(double beta_coarse)
        : fine(5), coarse(4)
        , func_f(fine.system, 100.0, solver_options())
        , func_c(coarse.system, beta_coarse, solver_options())
        , T_fine(func_f, solver_options())
        , T_coarse(func_c, solver_options())
        , transfer(coarse.package.get_dofs(), fine.package.get_dofs(),
                   coarse.package.get_constraints(), fine.package.get_constraints())
        , point_transfer(transfer, func_c.get_M(), func_f.get_M())
        , vector_transport(transfer, func_c.get_M(), func_f.get_M())
        , coarse_manifold(coarse.D())
        , base(T_fine, T_coarse, coarse_manifold, point_transfer, vector_transport)
        , x(positive_point(fine, 4.0, 0.0))
    {
        T_fine.update(x);
        base.update_model(x);
    }

    Level fine, coarse;
    Functional func_f, func_c;
    MassOracle<System> T_fine, T_coarse;
    fe::LinearTransferMG<dim> transfer;
    ManifoldTransfer<Operator> point_transfer;
    MassProjectionTransport<Operator> vector_transport;
    PosUnitMassSphere<MassMatrix> coarse_manifold;
    CoarseOracleBase base;
    Vector<double> x;
};

//! Coarse point near y: a smooth multiplicative perturbation of y, normalized
Vector<double> near(const Level& level, const Vector<double>& y)
{
    Vector<double> z(y);
    for (unsigned j = 0; j < z.size(); ++j) {
        z[j] *= std::exp(0.3 * std::sin(0.7 * j));
    }
    normalize(level.d(), z);
    return z;
}

void check_model(CheckReport& report)
{
    TwoLevel tl(100.0);
    const auto& state = tl.base.get_state();
    const Vector<double>& d = tl.coarse.d();
    DensityCoarseOracle<System, DensityGradient::MASS> q(tl.base, solver_options());

    report.info("coarse model", fmt::format("|w|_D = {}, max |w/y| = {}, min y = {}",
                                            sci(std::sqrt(d_inner(d, state.w, state.w))),
                                            sci([&] {
                                                double m = 0;
                                                for (unsigned j = 0; j < d.size(); ++j)
                                                    m = std::max(m, std::abs(state.w[j] / state.y[j]));
                                                return m;
                                            }()),
                                            sci(*std::ranges::min_element(state.y))));

    // q(y) = f_H(y)
    q.update(state.y);
    tl.T_coarse.update(state.y);
    const double fy = tl.T_coarse.value(state.y);
    const double err_y = std::abs(q.value(state.y) - fy) / std::abs(fy);
    report.check(err_y < 1e-13, "q(y) = f_H(y)", "relative error " + sci(err_y));

    // Shifted-potential form against the definition with the density lifting
    const Vector<double> z = near(tl.coarse, state.y);
    q.update(z);
    tl.T_coarse.update(z);
    double correction = 0;
    for (unsigned j = 0; j < z.size(); ++j) {
        correction += d[j] * state.w[j] * (z[j] * z[j] - state.y[j] * state.y[j]) / (2 * state.y[j]);
    }
    const double q_def = tl.T_coarse.value(z) - correction;
    const double err_form = std::abs(q.value(z) - q_def) / std::abs(q_def);
    report.check(err_form < 1e-13, "q(z) = f_H(z) - <w, L^m_y(z)>_D (shifted potential)",
                 "relative error " + sci(err_form));

    // Directional derivative against central differences, O(h^2)
    const Vector<double> v = tangent(tl.coarse, z, 0.37);
    q.update(z);
    const double dq = q.directional_derivative(z, v);
    auto fd = [&](double h) {
        Vector<double> zp(z), zm(z);
        zp.add(h, v);
        zm.add(-h, v);
        q.update(zp);
        const double qp = q.value(zp);
        q.update(zm);
        const double qm = q.value(zm);
        return std::abs((qp - qm) / (2 * h) - dq) / std::abs(dq);
    };
    const double e1 = fd(1e-3), e2 = fd(1e-4);
    report.check(e2 < 1e-7 && (e2 < 1e-10 || e1 / e2 > 50), "Dq(z)[v] (central differences, h^2)",
                 "relative error " + sci(e2) + ", ratio " + sci(e1 / e2));

    // Coherence: grad_M q(y) = R grad_M f_h(x)
    q.update(state.y);
    Vector<double> gy(d.size());
    q.gradient(state.y, gy);
    gy -= state.x_grad_restr;
    const double err_coh = std::sqrt(d_inner(d, gy, gy) / d_inner(d, state.x_grad_restr, state.x_grad_restr));
    report.check(err_coh < 1e-10, "coherence: grad_M q(y) = R grad_M f_h(x)", "relative error " + sci(err_coh));
}

//! Tangency and the gradient property <grad q, v>_G = Dq(z)[v] for one DensityGradient
template <DensityGradient G>
void check_gradient(CheckReport& report, TwoLevel& tl, const std::string& name)
{
    const Vector<double>& d = tl.coarse.d();
    DensityCoarseOracle<System, G> q(tl.base, solver_options());
    const Vector<double> z = near(tl.coarse, tl.base.get_state().y);
    q.update(z);
    Vector<double> g(z.size());
    q.gradient(z, g);

    const double tang = std::abs(d_inner(d, z, g)) / std::sqrt(d_inner(d, g, g));
    report.check(tang < 1e-10, name + ": gradient tangent", "|z^T D g| / |g|_D = " + sci(tang));

    double max_err = 0;
    for (const double freq : {0.37, 1.1, 2.3}) {
        const Vector<double> v = tangent(tl.coarse, z, freq);
        const double dq = q.directional_derivative(z, v);
        max_err = std::max(max_err, std::abs(q.metric().inner(g, v) - dq) / std::abs(dq));
    }
    report.check(max_err < 1e-8, name + ": <grad q, v> = Dq(z)[v] in its metric", "relative error " + sci(max_err));
}

void check_gradients(CheckReport& report)
{
    TwoLevel tl(100.0);
    check_gradient<DensityGradient::MASS>(report, tl, "MASS");
    check_gradient<DensityGradient::FROBENIUS>(report, tl, "FROBENIUS");
    check_gradient<DensityGradient::ENERGY_MODEL>(report, tl, "ENERGY_MODEL");
    check_gradient<DensityGradient::ENERGY_MODEL_SHIFT>(report, tl, "ENERGY_MODEL_SHIFT");
    check_gradient<DensityGradient::ENERGY_FINE>(report, tl, "ENERGY_FINE");
}

//! Large correction w: the coarse functional has beta 100 times the fine one
void check_safeguard(CheckReport& report)
{
    TwoLevel tl(1e4);
    const auto& state = tl.base.get_state();
    const Vector<double> z = near(tl.coarse, state.y);

    DensityCoarseOracle<System, DensityGradient::ENERGY_MODEL> q_model(tl.base, solver_options());
    q_model.update(z);
    DensityCoarseOracle<System, DensityGradient::ENERGY_MODEL_SHIFT> q_shift(tl.base, solver_options());
    q_shift.update(z);
    Vector<double> g(z.size());
    q_shift.gradient(z, g);

    double min_ratio = std::numeric_limits<double>::infinity();
    for (unsigned j = 0; j < z.size(); ++j) {
        min_ratio = std::min(min_ratio, (z[j] - g[j]) / z[j]);  // (eta + z) / z with eta = -g
    }
    report.check(min_ratio > 0, "ENERGY_MODEL_SHIFT: eta = -grad q > -z (large w)",
                 fmt::format("min (eta + z) / z = {}, sigma = {}", sci(min_ratio), sci(q_shift.get_sigma())));
    q_model.gradient(z, g);
    double min_model = std::numeric_limits<double>::infinity();
    for (unsigned j = 0; j < z.size(); ++j) {
        min_model = std::min(min_model, (z[j] - g[j]) / z[j]);
    }
    report.info("ENERGY_MODEL: min (eta + z) / z (large w)", sci(min_model));
    report.info("ENERGY_MODEL: row sum of A_hat(z) not positive",
                fmt::format("{} of {} updates", q_model.get_n_not_dominant(), q_model.get_n_updates()));
}

} // namespace


int main()
{
    return run_tests([](CheckReport& report) {
        check_model(report);
        check_gradients(report);
        check_safeguard(report);
    });
}
