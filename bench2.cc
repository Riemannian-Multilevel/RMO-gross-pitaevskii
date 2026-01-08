//
// Created by Ferdinand Vanmaele on 08.01.26.
//
#include "function.h"
#include "main.h"

#include <deal.II/base/timer.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <memory>
#include <vector>

using namespace dealii;
using namespace gpe;

template <int dim> void
prolongate_between_meshes(const DoFHandler<dim> &dof_coarse,
                          const Vector<double> &x_coarse,
                          const DoFHandler<dim> &dof_fine,
                          const AffineConstraints<double> &constraints_fine,
                          Vector<double> &y0_fine, double c = 1.0)
{
    y0_fine.reinit(dof_fine.n_dofs());
    y0_fine = 0.0;

    VectorTools::interpolate_to_finer_mesh(dof_coarse, x_coarse,
        dof_fine, constraints_fine, y0_fine);
    y0_fine *= c;
}

template <int dim> void
restrict_between_meshes(const DoFHandler<dim> &dof_fine,
                        const Vector<double> &x_fine,
                        const DoFHandler<dim> &dof_coarse,
                        const AffineConstraints<double> &constraints_coarse,
                        Vector<double> &y0_coarse, double c = 1.0)
{
    y0_coarse.reinit(dof_coarse.n_dofs());
    y0_coarse = 0.0;

    VectorTools::interpolate_to_coarser_mesh(dof_fine, x_fine, dof_coarse,
        constraints_coarse, y0_coarse);
    y0_coarse *= c;
}

template <int dim>
class Bench2
{
public:
    Bench2(const SparseMatrix<double>& M_fine_, const SparseMatrix<double>& M_coarse_,
           const SparseMatrix<double>& A0_fine_, const SparseMatrix<double>& A0_coarse_,
           const Vector<double>& x0_fine_, const Vector<double>& x0_coarse_, double beta_,
           const DoFHandler<dim> &dof_fine_, const DoFHandler<dim> &dof_coarse_,
           const AffineConstraints<double> &ac_fine_, const AffineConstraints<double> &ac_coarse_,
           const GdOptions& options_)
    :
        M_fine(M_fine_), M_coarse(M_coarse_), A0_fine(A0_fine_), A0_coarse(A0_coarse_),
        x0_fine(x0_fine_), x0_coarse(x0_coarse_), beta(beta_),
        dof_fine(dof_fine_), dof_coarse(dof_coarse_), ac_fine(ac_fine_), ac_coarse(ac_coarse_),
        options(options_)
    {
        // Assemble A_{x0_h}
        A_x0_fine.reinit(A0_fine.get_sparsity_pattern());
        A_x0_fine.copy_from(A0_fine);

        SparseMatrix<double> Mpp_x0_fine(M_fine.get_sparsity_pattern());
        assemble_mass_phiphi<dim>(execution::seq, Mpp_x0_fine,
            dof_fine, x0_fine, ac_fine);
        A_x0_fine.add(beta, Mpp_x0_fine);


        // Assemble A_{x0_H}
        A_x0_coarse.reinit(A0_coarse.get_sparsity_pattern());
        A_x0_coarse.copy_from(A0_coarse);

        SparseMatrix<double> Mpp_x0_coarse(M_coarse.get_sparsity_pattern());
        assemble_mass_phiphi(execution::seq, Mpp_x0_coarse,
            dof_coarse, x0_coarse, ac_coarse);
        A_x0_coarse.add(beta, Mpp_x0_coarse);


        // Compute grad E(x0_h)
        Vector<double> Mx0_fine(x0_fine.size());
        M_fine.vmult(Mx0_fine, x0_fine);          // M_h x_{h,0}

        auto [z_fine,it1] = solve_sparse(A_x0_fine, Mx0_fine, options.solver,
            PreconditionIdentity{}, options.max_inner, options.tol_inner);
        grad_E0_fine = energy::project_onto_tangent_space(z_fine, x0_fine, M_fine);


        // Compute grad E(x0_H)
        Vector<double> Mx0_coarse(x0_coarse.size());
        M_coarse.vmult(Mx0_coarse, x0_coarse);    // M_H x_{H,0}

        auto [z_coarse, it2] = solve_sparse(A_x0_coarse, Mx0_coarse, options.solver,
            PreconditionIdentity{}, options.max_inner, options.tol_inner);
        grad_E0_coarse = energy::project_onto_tangent_space(z_coarse, x0_coarse, M_coarse);


        // Compute kappa(x_{h,0}, x_{H,0})
        kappa = grad_E0_fine;

        Vector<double> P_grad_E0(x0_fine.size());
        prolongate_between_meshes(dof_coarse, grad_E0_coarse, dof_fine,
            ac_fine, P_grad_E0);
        kappa -= P_grad_E0;
    }

    Vector<double> update()
    {
        throw ExcNotImplemented("not implemented");
    }

    double value() const
    {
        throw ExcNotImplemented("not implemented");
    }

    Vector<double> gradient(const Vector<double>& x_coarse) const
    {
        AssertDimension(x_coarse.size(), x0_coarse.size());

        Vector<double> Ah0_kappa(kappa.size());
        A_x0_fine.vmult(Ah0_kappa, kappa);

        Vector<double> Px(x0_fine.size());                          // P(x_H)
        prolongate_between_meshes(dof_coarse, x_coarse, dof_fine, ac_fine, Px);

        SparseMatrix<double> A_Px(A0_fine.get_sparsity_pattern());    // A_P(x_H)
        A_Px.copy_from(A0_fine);

        SparseMatrix<double> Mpp_Px(A0_fine.get_sparsity_pattern());
        assemble_mass_phiphi(execution::seq, Mpp_Px, dof_fine, Px, ac_fine);
        A_Px.add(beta, Mpp_Px);

        Vector<double> A_PxPx(x0_fine.size());
        A_Px.vmult(A_PxPx, Px);


        // M_h^{-1} A_P(x) P(x) - A_x{h,0} (grad E(x{h,0} - P grad E(x{H,0})
        auto [y, it1] = solve_sparse(M_fine, A_PxPx, options.solver,
            PreconditionIdentity{}, options.max_inner, options.tol_inner);
        y -= Ah0_kappa;

        // TODO: correct factor for R = c P^T  (or access prolongation matrices explicitly -> multigrid code)
        Vector<double> z_coarse(x0_coarse.size());
        double c = 1.0;
        restrict_between_meshes(dof_fine, y, dof_coarse, ac_coarse, z_coarse, c);

        Vector<double> Mz(x0_coarse.size());
        M_coarse.vmult(Mz, z_coarse);

        // TODO: reuse solution vectors
        auto [Amz, it] = solve_sparse(A_x0_coarse, Mz, options.solver,
            PreconditionIdentity{}, options.max_inner, options.tol_inner);


        // Orthogonal projection onto tangent space
        Vector<double> Mh_Az(x0_coarse.size());
        M_coarse.vmult(Mh_Az, Amz);

        auto [Ainv_Mz, it2] = solve_sparse(A_x0_coarse, Mh_Az, options.solver,
            PreconditionIdentity{}, options.max_inner, options.tol_inner);

        return energy::project_onto_tangent_space(Ainv_Mz, Amz, M_coarse);
    }

private:
    // Arguments ----------------------------------------------------
    const SparseMatrix<double>& M_fine;
    const SparseMatrix<double>& M_coarse;
    const SparseMatrix<double>& A0_fine;
    const SparseMatrix<double>& A0_coarse;

    Vector<double> x0_fine;
    Vector<double> x0_coarse;
    double beta;

    const DoFHandler<dim> dof_fine;
    const DoFHandler<dim> dof_coarse;

    const AffineConstraints<double> ac_fine;
    const AffineConstraints<double> ac_coarse;
    GdOptions options;

    // Data fixed per iteration -------------------------------------
    SparseMatrix<double> A_x0_fine;
    SparseMatrix<double> A_x0_coarse;

    Vector<double> grad_E0_fine;
    Vector<double> grad_E0_coarse;
    Vector<double> kappa;
};

int main()
{
    TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

    // --- options as before ---
    GdOptions options_gd{};
    options_gd.tol_inner    = 1e-6;
    options_gd.tol_lambda   = 1e-8;
    options_gd.tol_residual = 1e-6;
    options_gd.step_size    = 1.0;
    options_gd.max_iter     = 20;
    options_gd.max_inner    = 500;
    options_gd.solver       = SolverMethod::MINRES;

    GPE_Options options{};
    options.dimension = 2;
    options.degree    = 1;
    options.radius    = 10;
    options.beta      = 100;
    options.bc        = BoundaryCondition::DIRICHLET;

    constexpr int dim = 2;
    Square<dim> V;

    using Exec = execution::seq_t;

}