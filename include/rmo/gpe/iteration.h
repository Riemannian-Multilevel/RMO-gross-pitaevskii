//
// Created by Ferdinand Vanmaele on 10.06.26.
//

#ifndef RMO_GPE_ITERATION_H
#define RMO_GPE_ITERATION_H

#include <rmo/ropt/oracle.h>
#include <rmo/gpe/gpe.h>
#include <rmo/gpe/kernels.h>
#include <rmo/gpe/residual.h>

#include <deal.II/base/timer.h>

namespace rmo::gpe {

/**
 * @brief Shared part of the iterations of a @p System: updates the functional at the point, and evaluates value,
 * residual and directional derivative there.
 */
template <typename System>
class GrossPitaevskiiIteration : public IterationBase
{
public:
    static constexpr int dimension = System::dimension;

    using Functional = GrossPitaevskiiFunctional<System>;
    using InverseM   = typename Functional::InverseM;
    using InverseA   = typename Functional::InverseA;

    GrossPitaevskiiIteration(Functional &func,
                             std::shared_ptr<const Vector<double>> x_ptr)
        : IterationBase(x_ptr), m_func(func), m_res(func)
    {
        AssertDimension(x_ptr->size(), m_func.n_dofs());

        m_func.update(*x_ptr);
    }

    [[nodiscard]] double value() const override
    {
        return m_func.value(*(this->x_ptr));
    }

    [[nodiscard]] double directional_derivative(const Vector<double> &z) const override
    {
        return m_func.directional_derivative(*(this->x_ptr), z);
    }

    [[nodiscard]] double residual() const override
    {
        return m_res.residual(*(this->x_ptr));
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
    Functional &m_func;

    const GrossPitaevskiiResidual<System> m_res;
};


/** @brief Riemannian gradient in the M-metric, see MassOracle. */
template <typename System>
class MassIteration : public GrossPitaevskiiIteration<System>
{
public:
    using Base       = GrossPitaevskiiIteration<System>;
    using Functional = typename Base::Functional;
    using InverseM   = typename Base::InverseM;
    using Base::gradient;  // keep gradient(dst), which computes the residual

    MassIteration(Functional &func,
                  std::shared_ptr<const Vector<double>> x_ptr,
                  SolverOptions options)
        : Base(func, x_ptr), m_options(options)
    {}

    GradInfo gradient(Vector<double>& output, double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};
        InverseM& M_inv = this->m_func.get_M_inv();

        if (residual > 0) {
            M_inv.set_tol(residual * m_options.tol_inner_res);
        }

        timer.start();
        kernels::grad_mass(M_inv, this->m_func.get_A(), this->m_func.get_M(), *(this->x_ptr), output);

        info.num_iter     = M_inv.control().last_step();
        info.tolerance    = M_inv.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        timer.stop();
        return info;
    }

private:
    SolverOptions m_options;
};


/** @brief Riemannian gradient in the energy-adaptive metric, see EnergyOracle. */
template <typename System>
class EnergyIteration : public GrossPitaevskiiIteration<System>
{
public:
    using Base       = GrossPitaevskiiIteration<System>;
    using Functional = typename Base::Functional;
    using InverseA   = typename Base::InverseA;
    using Base::gradient;  // keep gradient(dst), which computes the residual

    EnergyIteration(Functional &func,
                    std::shared_ptr<const Vector<double>> x_ptr,
                    SolverOptions options)
        : Base(func, x_ptr), m_options(options)
    {}


    GradInfo gradient(Vector<double>& output, double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};
        InverseA& A_inv = this->m_func.get_A_inv();

        if (residual > 0) {
            A_inv.set_tol(residual * m_options.tol_inner_res);
        }

        timer.start();
        kernels::grad_energy_adaptive(A_inv, this->m_func.get_M(), *(this->x_ptr), output);

        info.num_iter     = A_inv.control().last_step();
        info.tolerance    = A_inv.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        timer.stop();
        return info;
    }

private:
    SolverOptions m_options;
};


/** @brief Riemannian gradient in the F-metric, see FrobeniusOracle. */
template <typename System>
class FrobeniusIteration : public GrossPitaevskiiIteration<System>
{
public:
    using Base       = GrossPitaevskiiIteration<System>;
    using Functional = typename Base::Functional;

    FrobeniusIteration(Functional &func,
                       std::shared_ptr<const Vector<double>> x_ptr,
                       SolverOptions = {})
        : Base(func, x_ptr)
    {}

    // override from base, no matrix inversions (tolerance) needed
    GradInfo gradient(Vector<double>& output) const final
    {
        return gradient(output, -1.0);
    }

    GradInfo gradient(Vector<double>& output, double) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        timer.start();
        kernels::grad_frobenius(this->m_func.get_A(), this->m_func.get_M(), *(this->x_ptr), output);
        timer.stop();

        // F-gradient evaluation does not involve a linear solver.
        info.elapsed_time = timer.cpu_time();
        return info;
    }

};

} // namespace rmo::gpe

#endif //RMO_GPE_ITERATION_H
