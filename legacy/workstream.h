
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
    auto cell_worker = [&assemble_cell](const auto& cell, ScratchData<dim>& scratch_data, PerTaskData<dim>& task_data)
    {
        scratch_data.fe_values.reinit(cell); // cell is active_cell_iterator or level_cell_iterator
        task_data.cell_matrix = 0;

        //const unsigned int dofs_per_cell = fe.dofs_per_cell;
        //data.local_dof_indices.resize(dofs_per_cell);
        cell->get_active_or_mg_dof_indices(task_data.local_dof_indices);

        // user-provided local assembly
        assemble_cell(scratch_data.fe_values, task_data.cell_matrix, task_data.local_dof_indices);
    };

    auto copier = [&constraints, &system_matrix](const PerTaskData<dim>& task_data)
    {
        constraints.distribute_local_to_global(task_data.cell_matrix, task_data.local_dof_indices, system_matrix);
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


// Tag-based approach for selecting parallel or sequential version
template <int dim, class Assembly>
void assemble_system(dealii::SparseMatrix<double> &system_matrix,
                     const dealii::DoFHandler<dim> &dof_handler,
                     dealii::UpdateFlags flags,
                     Assembly&& assemble_cell,
                     const dealii::AffineConstraints<double> &constraints,
                     unsigned int level = invalid_unsigned_int)
{
    assemble_system(dof_handler, flags, std::forward<Assembly>(assemble_cell),
                    constraints, level);
}

template <int dim, class Assembly>
void assemble_system(dealii::SparseMatrix<double> &system_matrix,
                     const dealii::DoFHandler<dim> &dof_handler,
                     dealii::UpdateFlags flags,
                     Assembly &&assemble_cell,
                     const dealii::AffineConstraints<double> &constraints,
                     unsigned int level = invalid_unsigned_int)
{
    parallel::assemble_system(system_matrix, dof_handler, flags,
        std::forward<Assembly>(assemble_cell), constraints, level);
}

namespace execution
{
struct seq_t {};
struct par_t {};  // parallel
struct mul_t {};  // matrix-free

inline constexpr seq_t seq{};
inline constexpr par_t par{};
inline constexpr mul_t mul{};
}
