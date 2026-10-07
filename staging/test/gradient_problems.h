#ifndef RMO_TEST_GRADIENT_PROBLEMS_H
#define RMO_TEST_GRADIENT_PROBLEMS_H

#include <rmo/lac.h>
#include <rmo/gpe/gpe.h>
#include <rmo/gpe/kernels.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/metric.h>
#include <rmo/ropt/metric.h>

/**
 * @file
 * @brief Test problems of staging/test/gradient.cc: an objective on the unit-mass sphere with a Riemannian gradient kernel.
 *
 * - ConstrainedSphere: the operators of the functional and the constraints, shared by all problems;
 * - EnergyAdaptive, Mass, Frobenius: metric, tangent spaces and gradient of the energy, for one metric;
 * - MassModel, FrobeniusModel: coarse model with the correction term in the M- or F-inner product, and its
 *   gradient in each metric for which a kernel exists;
 * - FineProblem (energy) and CoarseProblem (coarse model) implement GradientProblem for one metric.
 */
namespace rmo::gpe::test
{

template <int dim>
using Functional = GrossPitaevskiiFunctional<GrossPitaevskiiSystem<dim>>;

/** @brief Unit-mass sphere \f$ x^\top M x = 1 \f$ of a system, for vectors that satisfy its constraints. */
template <int dim>
class ConstrainedSphere
{
public:
    ConstrainedSphere(GrossPitaevskiiSystem<dim>& system, double beta, const SolverOptions& options)
        : m_func(system, beta, options)
        , m_constraints(system.get_constraints())
    {}

    //! Operators \f$ M \f$, \f$ A(x) \f$ and their inverses, at the point of the last update().
    [[nodiscard]] const Functional<dim>& functional() const { return m_func; }
    [[nodiscard]] unsigned n_dofs() const { return m_func.n_dofs(); }

    //! Reassembles \f$ A(x) \f$; value and gradient of all problems are evaluated at this point.
    void update(const Vector<double>& x) { m_func.update(x); }

    //! Applies the constraints: the assembled operators represent the energy only on such vectors.
    void distribute(Vector<double>& z) const { m_constraints.distribute(z); }

    //! Applies the constraints to @p x and scales it to \f$ x^\top M x = 1 \f$.
    void make_admissible(Vector<double>& x) const
    {
        distribute(x);
        ellipsoid::retract_by_norm(m_func.get_M(), x);
    }

    //! \f$ R_x(v) = (x + v) / \|x + v\|_M \f$
    void retract(const Vector<double>& x, const Vector<double>& v, Vector<double>& x_new) const
    {
        x_new = x;
        ellipsoid::retract_by_norm(m_func.get_M(), v, x_new);
    }

private:
    Functional<dim> m_func;
    const dealii::AffineConstraints<double>& m_constraints;
};


/**
 * @brief Energy-adaptive metric \f$ \langle u, v \rangle_x = u^\top A(x) v \f$: tangent spaces and gradient of
 * \f$ E \f$.
 */
template <int dim>
class EnergyAdaptive
{
public:
    explicit EnergyAdaptive(const Functional<dim>& f) : f(f), m_metric(f.get_A(), MetricKind::ENERGY_ADAPTIVE) {}

    [[nodiscard]] const MetricBase& metric() const { return m_metric; }

    void project(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const
    {
        metric::energy::project_onto_tangent_space(f.get_A_inv(), x, f.get_M(), v, v_proj);
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const
    {
        metric::energy::random_tangent_vector(f.get_A_inv(), x, f.get_M(), v);
    }

    void gradient(const Vector<double>& x, Vector<double>& g) const
    {
        kernels::grad_energy_adaptive(f.get_A_inv(), f.get_M(), x, g);
    }

private:
    const Functional<dim>& f;
    const OperatorMetric<OperatorType> m_metric;
};


/** @brief M-metric \f$ \langle u, v \rangle = u^\top M v \f$: tangent spaces and gradient of \f$ E \f$. */
template <int dim>
class Mass
{
public:
    explicit Mass(const Functional<dim>& f) : f(f), m_metric(f.get_M(), MetricKind::MASS) {}

    [[nodiscard]] const MetricBase& metric() const { return m_metric; }

    void project(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const
    {
        metric::mass::project_onto_tangent_space(x, f.get_M(), v, v_proj);
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const
    {
        metric::mass::random_tangent_vector(x, f.get_M(), v);
    }

    void gradient(const Vector<double>& x, Vector<double>& g) const
    {
        kernels::grad_mass(f.get_M_inv(), f.get_A(), f.get_M(), x, g);
    }

private:
    const Functional<dim>& f;
    const OperatorMetric<OperatorType> m_metric;
};


/** @brief F-metric \f$ \langle u, v \rangle = u^\top v \f$: tangent spaces and gradient of \f$ E \f$. */
template <int dim>
class Frobenius
{
public:
    explicit Frobenius(const Functional<dim>& f) : f(f) {}

    [[nodiscard]] const MetricBase& metric() const { return m_metric; }

    void project(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const
    {
        metric::frobenius::project_onto_tangent_space(x, f.get_M(), v, v_proj);
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const
    {
        metric::frobenius::random_tangent_vector(x, f.get_M(), v);
    }

    void gradient(const Vector<double>& x, Vector<double>& g) const
    {
        kernels::grad_frobenius(f.get_A(), f.get_M(), x, g);
    }

private:
    const Functional<dim>& f;
    const EuclideanMetric m_metric;
};


/**
 * @brief Coarse model \f$ \Psi(x) = E(x) - \langle w, R_\phi^{-1}(x) \rangle_M \f$, with gradient kernels for the M- and
 * the energy-adaptive metric.
 */
template <int dim>
class MassModel
{
public:
    explicit MassModel(const Functional<dim>& f) : f(f) {}

    [[nodiscard]] double value(const Vector<double>& x, const Vector<double>& phi, const Vector<double>& w) const
    {
        return kernels::coarse_mass_value(x, phi, w, f.get_M(), f.value(x));
    }

    //! Correction \f$ w \f$: @p w_ambient projected onto the tangent space at @p phi.
    void correction(const Vector<double>& phi, const Vector<double>& w_ambient, Vector<double>& w) const
    {
        metric::mass::project_onto_tangent_space(phi, f.get_M(), w_ambient, w);
    }

    void gradient(const Mass<dim>&, const Vector<double>& x, const Vector<double>& phi, const Vector<double>& w,
                  Vector<double>& g) const
    {
        kernels::coarse_mass_grad(f.get_M(), f.get_M_inv(), f.get_A(), x, phi, w, g);
    }

    void gradient(const EnergyAdaptive<dim>&, const Vector<double>& x, const Vector<double>& phi,
                  const Vector<double>& w, Vector<double>& g) const
    {
        kernels::coarse_mass_grad_energy_adaptive(f.get_M(), f.get_A_inv(), x, phi, w, g);
    }

private:
    const Functional<dim>& f;
};


/**
 * @brief Coarse model \f$ \Psi(x) = E(x) - w^\top R_\phi^{-1}(x) \f$, with gradient kernels for the F- and the
 * energy-adaptive metric.
 */
template <int dim>
class FrobeniusModel
{
public:
    explicit FrobeniusModel(const Functional<dim>& f) : f(f) {}

    [[nodiscard]] double value(const Vector<double>& x, const Vector<double>& phi, const Vector<double>& w) const
    {
        return kernels::coarse_frobenius_value(x, phi, w, f.get_M(), f.value(x));
    }

    //! Correction \f$ w \f$: @p w_ambient projected onto the tangent space at @p phi.
    void correction(const Vector<double>& phi, const Vector<double>& w_ambient, Vector<double>& w) const
    {
        metric::frobenius::project_onto_tangent_space(phi, f.get_M(), w_ambient, w);
    }

    void gradient(const Frobenius<dim>&, const Vector<double>& x, const Vector<double>& phi, const Vector<double>& w,
                  Vector<double>& g) const
    {
        kernels::coarse_frobenius_grad(f.get_M(), f.get_A(), x, phi, w, g);
    }

    void gradient(const EnergyAdaptive<dim>&, const Vector<double>& x, const Vector<double>& phi,
                  const Vector<double>& w, Vector<double>& g) const
    {
        kernels::coarse_frobenius_grad_energy_adaptive(f.get_M(), f.get_A_inv(), f.get_A(), x, phi, w, g);
    }

private:
    const Functional<dim>& f;
};


/** @brief Objective on the sphere with its Riemannian gradient, as seen by the gradient check. */
class GradientProblem
{
public:
    virtual ~GradientProblem() = default;

    //! Draws new parameters before each trial (coarse models: base point and correction).
    virtual void sample_parameters() {}

    //! Random point on the sphere.
    virtual void random_point(Vector<double>& x) const = 0;

    //! Value at @p x, which must be the point of the last ConstrainedSphere::update().
    [[nodiscard]] virtual double value(const Vector<double>& x) const = 0;
    virtual void gradient(const Vector<double>& x, Vector<double>& g) const = 0;

    //! Metric of gradient(): \f$ \langle \grad f(x), v \rangle_x = \mathrm{D}f(x)[v] \f$.
    [[nodiscard]] virtual const MetricBase& metric() const = 0;
    //! Orthogonal projection onto the tangent space at @p x, in metric().
    virtual void project(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const = 0;
    virtual void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const = 0;
};


/** @brief Problem with the metric and tangent spaces of @p Metric. */
template <int dim, template <int> class Metric>
class MetricProblem : public GradientProblem
{
public:
    explicit MetricProblem(const ConstrainedSphere<dim>& sphere) : sphere(sphere), metric_kernels(sphere.functional()) {}

    [[nodiscard]] const MetricBase& metric() const final { return metric_kernels.metric(); }

    void project(const Vector<double>& x, const Vector<double>& v, Vector<double>& v_proj) const final
    {
        metric_kernels.project(x, v, v_proj);
    }

    void random_tangent_vector(const Vector<double>& x, Vector<double>& v) const final
    {
        metric_kernels.random_tangent_vector(x, v);
    }

protected:
    const ConstrainedSphere<dim>& sphere;
    const Metric<dim> metric_kernels;
};


/** @brief The energy \f$ E \f$, with its Riemannian gradient in @p Metric. */
template <int dim, template <int> class Metric>
class FineProblem final : public MetricProblem<dim, Metric>
{
public:
    using MetricProblem<dim, Metric>::MetricProblem;

    void random_point(Vector<double>& x) const override
    {
        ellipsoid::random_point(x, this->sphere.functional().get_M());
    }

    [[nodiscard]] double value(const Vector<double>& x) const override { return this->sphere.functional().value(x); }

    void gradient(const Vector<double>& x, Vector<double>& g) const override { this->metric_kernels.gradient(x, g); }
};


/**
 * @brief Coarse model @p Model with its Riemannian gradient in @p Metric, on the mesh of the fine problem: a coarse
 * mesh and grid transfers are not needed to check the gradient.
 *
 * Each trial draws a random base point \f$ \phi \f$, and projects a fixed vector onto the tangent space at
 * \f$ \phi \f$ for the correction \f$ w \f$.
 */
template <int dim, template <int> class Model, template <int> class Metric>
class CoarseProblem final : public MetricProblem<dim, Metric>
{
public:
    CoarseProblem(const ConstrainedSphere<dim>& sphere, const Vector<double>& w)
        : MetricProblem<dim, Metric>(sphere)
        , model(sphere.functional())
        , w_ambient(w)
        , phi(w.size())
        , w(w.size())
    {}

    void sample_parameters() override
    {
        ellipsoid::random_point(phi, this->sphere.functional().get_M());
        this->sphere.make_admissible(phi);
        model.correction(phi, w_ambient, w);
    }

    //! \f$ R_\phi(v) \f$ for a random unit tangent vector \f$ v \f$ at \f$ \phi \f$, so that the point is near \f$ \phi \f$.
    void random_point(Vector<double>& x) const override
    {
        Vector<double> v(x.size());
        this->metric_kernels.random_tangent_vector(phi, v);
        v /= this->metric_kernels.metric().norm(v);
        this->sphere.retract(phi, v, x);
    }

    [[nodiscard]] double value(const Vector<double>& x) const override { return model.value(x, phi, w); }

    void gradient(const Vector<double>& x, Vector<double>& g) const override
    {
        model.gradient(this->metric_kernels, x, phi, w, g);
    }

private:
    const Model<dim> model;
    const Vector<double> w_ambient;
    Vector<double> phi, w;
};

} // namespace rmo::gpe::test

#endif // RMO_TEST_GRADIENT_PROBLEMS_H
