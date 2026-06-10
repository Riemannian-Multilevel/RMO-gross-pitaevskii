//
// Created by Ferdinand Vanmaele on 10.06.26.
//

#ifndef GPE_ITERATION_H
#define GPE_ITERATION_H

#include <gpe/lac.h>
#include <gpe/problem/gpe.h>
#include <gpe/ropt/manifold.h>
#include <gpe/ropt/transport.h>

#include <deal.II/base/timer.h>


namespace gpe
{

// Alternative interface setting the evaluation point in the constructor
class IterationBase
{
public:
    virtual ~IterationBase() = default;

    // Shared pointer to ensure lifetime of evaluation point, when IterationBase object relies on it
    IterationBase(std::shared_ptr<const Vector<double>> x_ptr)
        : x_ptr(x_ptr)
    {}

    virtual double value()    const = 0;
    virtual double residual() const = 0;
    virtual double directional_derivative(const Vector<double> &z) const = 0;

    virtual GradInfo gradient(Vector<double>& dst) const = 0;

    unsigned n_dofs() const { return x_ptr->size(); }

protected:
    std::shared_ptr<const Vector<double>> x_ptr;
};


template <int dim>
class GrossPitaevskiiIteration : public IterationBase
{
public:
    static constexpr int dimension = dim;

    GrossPitaevskiiIteration(GrossPitaevskiiFunctional<dim> &func,
                             std::shared_ptr<const Vector<double>> x_ptr)
        : IterationBase(x_ptr), m_func(func)
    {
        AssertDimension(x_ptr->size(), m_func.n_dofs());

        m_func.update(*x_ptr);
    }

    double value() const override
    {
        return m_func.value(*(this->x_ptr));
    }

    double directional_derivative(const Vector<double> &z) const override
    {
        return m_func.directional_derivative(*(this->x_ptr), z);
    }

    GradInfo gradient(Vector<double>& dst) const override
    {
        const double x_residual = residual();
        Assert(x_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info = gradient(dst, x_residual);
        info.residual = x_residual;

        return info;
    }

    virtual GradInfo gradient(Vector<double>& dst, double) const = 0;  // variable metric defined in child classes


protected:
    GrossPitaevskiiFunctional<dim> &m_func;
};


template <int dim>
class MassIteration : public GrossPitaevskiiIteration<dim>
{
public:
    MassIteration(GrossPitaevskiiFunctional<dim> &func,
                  std::shared_ptr<const Vector<double>> x_ptr,
                  SolverOptions options)
        : GrossPitaevskiiIteration<dim>(func, x_ptr), m_options(options)
    {}

    GradInfo gradient(Vector<double>& output, double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};
        auto& M_inv = this->m_func.get_M_inv();

        if (residual > 0) {
            M_inv.set_tol(residual * m_options.tol_inner_res);
        }

        timer.start();
        ellipsoid::mass::gradient(M_inv, this->m_func.get_A(), this->m_func.get_M(), *(this->x_ptr), output);

        info.num_iter     = M_inv.control().last_step();
        info.tolerance    = M_inv.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        timer.stop();
        return info;
    }

private:
    SolverOptions m_options;
};


template <int dim>
class EnergyIteration : public GrossPitaevskiiIteration<dim>
{
public:
    EnergyIteration(GrossPitaevskiiFunctional<dim> &func,
                    std::shared_ptr<const Vector<double>> x_ptr,
                    SolverOptions options)
        : GrossPitaevskiiIteration<dim>(func, x_ptr), m_options(options)
    {}


    GradInfo gradient(Vector<double>& output, double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};
        auto& A_inv = this->m_func.get_A_inv();

        if (residual > 0) {
            A_inv.set_tol(residual * m_options.tol_inner_res);
        }

        timer.start();
        ellipsoid::energy::gradient(A_inv, this->m_func.get_M(), *(this->x_ptr), output);

        info.num_iter     = A_inv.control().last_step();
        info.tolerance    = A_inv.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        timer.stop();
        return info;
    }

private:
    SolverOptions m_options;
};


template <int dim>
class FrobeniusIteration : public GrossPitaevskiiIteration<dim>
{
public:
    FrobeniusIteration(GrossPitaevskiiFunctional<dim> &func,
                       std::shared_ptr<const Vector<double>> x_ptr,
                       SolverOptions = {})
        : GrossPitaevskiiIteration<dim>(func, x_ptr)
    {}

    GradInfo gradient(Vector<double>& output) const final  // override from base, no matrix inversions (tolerance) needed
    {
        return gradient(output, -1.0);
    }

    GradInfo gradient(Vector<double>& output, double) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        timer.start();
        ellipsoid::frobenius::gradient(this->m_func.get_A(), this->m_func.get_M(), *(this->x_ptr), output);
        timer.stop();

        // F-gradient evaluation does not involve a linear solver.
        info.elapsed_time = timer.cpu_time();
        return info;
    }

};

} // namespace gpe

#endif //GPE_ITERATION_H
