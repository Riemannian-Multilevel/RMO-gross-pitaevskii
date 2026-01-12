
// This class iterates iterating over active cells, and multigrid cells on a certain level,
// using the same interface (get_active_or_mg_dof_indices(), unsigned int marker)
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
    // TODO: note that additional steps need to be taken for geometric multigrid approaches (step-16)
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

// TODO: when using geometric multigrid (eg. as preconditioner, in contrast to multilevel methods),
//       additional constraints are needed -- see step-16 assemble_multigrid()
template <int dim>
class FeSpaceMG : public FeSpaceBase<dim>
{
public:
    FeSpaceMG(const dealii::Triangulation<dim>& triangulation, unsigned int degree)
        : FeSpaceBase<dim>(triangulation, degree)
    {}

    void setup_dofs(Ordering order) override
    {
        const unsigned int n_levels = this->dof_handler.get_triangulation().n_levels();

        // Distribute degrees of freedom according to (default or other) ordering,
        // such that a basis of V_h can be enumerated in a deterministic way
        this->dof_handler.distribute_dofs(this->element);

        // Distribute level degrees of freedom on each level for geometric multigrid
        this->dof_handler.distribute_mg_dofs();

        // Reorder degrees of freedom for improved conditioning of system matrix
        // (default: order vertices, faces, ... by refinement level)
        if (order != Ordering::DEFAULT) {
            for (unsigned i = 0; i < n_levels; i++) {
                renumber_dofs<dim>(this->dof_handler, order, i);
            }
        }
    }

    void setup_constraints(BoundaryCondition bounds) override
    {
        dealii::Functions::ZeroFunction<dim> boundary_function(this->dof_handler.get_fe().n_components());

        // Define hanging nodes (optional for global refinement)
        this->constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);

        // MG constraints
        this->mg_constrained_dofs.clear();
        this->mg_constrained_dofs.initialize(this->dof_handler);

        // Set boundary condition for linear system (after dof distribution)
        if (bounds == BoundaryCondition::DIRICHLET) {
            dealii::VectorTools::interpolate_boundary_values(this->dof_handler,
                0, boundary_function, this->constraints);

            this->mg_constrained_dofs.make_zero_boundary_constraints(this->dof_handler, {0});
        }
        this->constraints.close();
    }

    const dealii::AffineConstraints<double>&
        get_level_constraints(unsigned int level) const
    {
        return mg_constrained_dofs.get_level_constraints(level);
    }
    const dealii::MGConstrainedDoFs& get_mg_dofs() const
    {
        return mg_constrained_dofs;
    }

private:
    dealii::MGConstrainedDoFs mg_constrained_dofs;
};

template <int dim>
void renumber_dofs(dealii::DoFHandler<dim>& dof_handler,
                   const Ordering order     = Ordering::CUTHILL_MCKEE,
                   unsigned int level       = invalid_unsigned_int,
                   bool use_constraints     = false,
                   bool reversed_numbering  = false)
{
    if (level != invalid_unsigned_int) {
        AssertIndexRange(level, dof_handler.get_triangulation().n_levels());
    }

    switch (order) {
        case Ordering::DEFAULT:
            break;
        case Ordering::RANDOM:
            if (level == invalid_unsigned_int) {
                dealii::DoFRenumbering::random(dof_handler);
            } else {
                dealii::DoFRenumbering::random(dof_handler, level);
            }
            break;
        case Ordering::CUTHILL_MCKEE:
            if (level == invalid_unsigned_int) {
                dealii::DoFRenumbering::Cuthill_McKee(dof_handler, reversed_numbering, use_constraints);
            } else {
                dealii::DoFRenumbering::Cuthill_McKee(dof_handler, level, reversed_numbering);
            }
            break;
        default:
            throw std::invalid_argument("unknown ordering");
    }
}
