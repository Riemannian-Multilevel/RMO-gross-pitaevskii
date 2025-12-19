#ifndef GPE_ASSEMBLE_H
#define GPE_ASSEMBLE_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

namespace gpe
{
using dealii::types::global_dof_index;
using dealii::numbers::invalid_unsigned_int;

//! Mass matrix assembly, implementation based on tutorial/step-3
//! @tparam dim Problem dimension
//! @tparam Assembly
//! @param system_matrix Matrix to be populated with entries
//! @param dof_handler DOF object, contains triangulation and finite element
//! @param flags Required update flags, typically set in Assembly object
//! @param assemble_cell Function object which iterates over local cells
//! @param level Multigrid level; if valid, uses multigrid iterators instead of active cell ones
//! @param constraints Affine constraints applied to matrix rows and columns
template <int dim, typename Assembly>
// BUG: DoFHandler::get_fe() - error: variable type 'FiniteElement<1, 1>' is an abstract class
void assemble_system(dealii::SparseMatrix<double>& system_matrix,
                     const dealii::DoFHandler<dim>& dof_handler,
                     dealii::UpdateFlags flags, Assembly&& assemble_cell,
                     const dealii::AffineConstraints<double>& constraints,
                     unsigned int level = invalid_unsigned_int)
{
    // Quadrature formula for the evaluation of the integrals on each cell
    const auto& element = dof_handler.get_fe();
    const dealii::QGauss<dim> quadrature_formula(element.degree + 1);

    // Class which handles finite element, quadrature, and mapping objects
    dealii::FEValues<dim> fe_values(element, quadrature_formula, flags);

    // Compute contributions of each cell in a local dense matrix, to avoid
    // updating a large sparse matrix in every step
    // Consistent between active and multigrid approaches
    const unsigned int dofs_per_cell = element.n_dofs_per_cell();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<global_dof_index> local_dof_indices(dofs_per_cell);

    // Clear existing matrix entries
    // XXX: optional parameter if (reinit) ... (default true)
    system_matrix = 0;  // system_matrix.reinit(system_matrix.get_sparsity_pattern())

    // Generic lambda: works for active-cell range and mg-level range
    auto assemble_over_cells = [&](const auto &cell_range)
    {
        for (const auto &cell : cell_range)  // cell is a DoFHandler<dim>::(active|level)_cell_iterator
        {
            fe_values.reinit(cell); // convertible to Triangulation::cell_iterator
            cell_matrix = 0;
            cell->get_active_or_mg_dof_indices(local_dof_indices);

            // Pass on populated DoF indices to assemble matrix
            assemble_cell(fe_values, cell_matrix, local_dof_indices);

            // Apply boundary conditions (Dirichlet and hanging nodes, if any)
            // when distributing local (cell) matrix entries
            constraints.distribute_local_to_global(cell_matrix, local_dof_indices, system_matrix);
        }
    };

    // Iterate over cells / degrees of freedom
    if (level == invalid_unsigned_int) {
        AssertDimension(system_matrix.m(), dof_handler.n_dofs());
        AssertDimension(system_matrix.n(), dof_handler.n_dofs());

        // Iterate over active cells
        // assemble_system_impl(dof_handler.active_cell_iterators(),
        //     fe_values, system_matrix, cell_matrix, local_dof_indices, assemble_cell, constraints);
        assemble_over_cells(dof_handler.active_cell_iterators());
    }
    else {
        AssertDimension(system_matrix.m(), dof_handler.n_dofs(level));
        AssertDimension(system_matrix.n(), dof_handler.n_dofs(level));

        // Iterate over multigrid cells on given level
        // assemble_system_impl(dof_handler.mg_cell_iterators_on_level(level),
        //     fe_values, system_matrix, cell_matrix, local_dof_indices, assemble_cell, constraints);
        assemble_over_cells(dof_handler.mg_cell_iterators_on_level(level));
    }
}

namespace parallel
{
template <int dim>
struct ScratchData
{
    dealii::FEValues<dim> fe_values;

    ScratchData(const dealii::FiniteElement<dim> &fe,
                const dealii::QGauss<dim> &quadrature,
                dealii::UpdateFlags flags)
        : fe_values(fe, quadrature, flags)
    {}

    ScratchData(const ScratchData<dim> &scratch)
        : fe_values(scratch.fe_values.get_fe(),
                    scratch.fe_values.get_quadrature(),
                    scratch.fe_values.get_update_flags())
    {}
};

template <int dim>
struct PerTaskData
{
    dealii::FullMatrix<double> cell_matrix;
    std::vector<global_dof_index> local_dof_indices;

    PerTaskData (const dealii::FiniteElement<dim> &fe)
        : cell_matrix (fe.dofs_per_cell, fe.dofs_per_cell),
          local_dof_indices (fe.dofs_per_cell)
    {}
};

template <int dim, class Assembly>
void assemble_system(dealii::SparseMatrix<double>& system_matrix,
                     const dealii::DoFHandler<dim>& dof_handler,
                     dealii::UpdateFlags flags,
                     Assembly&& assemble_cell,
                     const dealii::AffineConstraints<double>& constraints,
                     unsigned int level = dealii::numbers::invalid_unsigned_int)
{
#ifdef DEBUG
    std::cerr << "gpe: Parallel assembly enabled" << std::endl;
#endif
    const auto& fe = dof_handler.get_fe();
    const dealii::QGauss<dim> quadrature(fe.degree + 1);

    // Clear existing matrix entries
    system_matrix = 0;

    ScratchData<dim> scratch(fe, quadrature, flags);
    PerTaskData<dim> data(fe);

    // generic cell worker: works for active and level cells
    auto cell_worker = [&assemble_cell](const auto& cell, ScratchData<dim>& scratch, PerTaskData<dim>& data)
    {
        scratch.fe_values.reinit(cell); // cell is active_cell_iterator or level_cell_iterator
        data.cell_matrix = 0;

        //const unsigned int dofs_per_cell = fe.dofs_per_cell;
        //data.local_dof_indices.resize(dofs_per_cell);

        cell->get_active_or_mg_dof_indices(data.local_dof_indices);

        // user-provided local assembly
        assemble_cell(scratch.fe_values, data.cell_matrix, data.local_dof_indices);
    };

    auto copier = [&constraints, &system_matrix](const PerTaskData<dim>& data)
    {
        constraints.distribute_local_to_global(data.cell_matrix, data.local_dof_indices, system_matrix);
    };

    if (level == dealii::numbers::invalid_unsigned_int) {
        AssertDimension(system_matrix.m(), dof_handler.n_dofs());
        AssertDimension(system_matrix.n(), dof_handler.n_dofs());

        dealii::WorkStream::run(dof_handler.begin_active(), dof_handler.end(),
            cell_worker, copier, scratch, data);
    }
    else {
        AssertDimension(system_matrix.m(), dof_handler.n_dofs(level));
        AssertDimension(system_matrix.n(), dof_handler.n_dofs(level));

        dealii::WorkStream::run(dof_handler.begin_mg(level), dof_handler.end_mg(level),
            cell_worker, copier, scratch, data);
    }
}

} // namespace parallel

namespace execution
{
struct seq_t {};
struct par_t {};

inline constexpr seq_t seq{};
inline constexpr par_t par{};
}

// Tag-based approach for selecting parallel or sequential version
template <int dim, class Assembly>
void assemble_system(execution::seq_t,
                     dealii::SparseMatrix<double> &system_matrix,
                     const dealii::DoFHandler<dim> &dof_handler,
                     dealii::UpdateFlags flags,
                     Assembly&& assemble_cell,
                     const dealii::AffineConstraints<double> &constraints,
                     unsigned int level = invalid_unsigned_int)
{
    assemble_system(system_matrix, dof_handler, flags,
        std::forward<Assembly>(assemble_cell), constraints, level);
}

template <int dim, class Assembly>
void assemble_system(execution::par_t,
                     dealii::SparseMatrix<double> &system_matrix,
                     const dealii::DoFHandler<dim> &dof_handler,
                     dealii::UpdateFlags flags,
                     Assembly &&assemble_cell,
                     const dealii::AffineConstraints<double> &constraints,
                     unsigned int level = invalid_unsigned_int)
{
    parallel::assemble_system(system_matrix, dof_handler, flags,
        std::forward<Assembly>(assemble_cell), constraints, level);
}

} // namespace gpe

#endif //GPE_ASSEMBLE_H